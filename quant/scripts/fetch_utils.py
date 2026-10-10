"""Shared helpers for bounded market-data workers."""
from __future__ import annotations

import os
import json
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from market_data import normalize_daily_bars

try:
    import requests
except ImportError:  # pragma: no cover - AkShare normally brings requests
    requests = None


UTC = timezone.utc

# 基准连续性守卫阈值（2026-09-02 毒数据事故的防线，见 quant/docs/研究管线不变量.md）
# 相邻交易日收盘比值超过该值视为复权基准断裂（跨源拼接的 60 倍台阶特征）。
BASIS_STEP_RATIO = 1.5
# 重叠日期收盘比值中位数偏离超过该值视为两段序列口径不一致。
BASIS_OVERLAP_MEDIAN_DEVIATION = 0.10
BASIS_OVERLAP_MIN_DATES = 3


class BaselineDiscontinuityError(RuntimeError):
    """增量拼接的前后收盘价基准断裂（跨源/复权口径切换特征）。

    实证：2026-09-02 部分标的在东财/腾讯失败后落到新浪后复权通道，
    ~200 元的新行被拼到 ~3 元历史上，60 倍台阶毒化全部下游指标。
    调用方（fetch_stock_data / fetch_fund_data）捕获后应把本地 CSV
    挪开并降级为同源全量重取，而不是把拼接结果写盘。
    """

    def __init__(self, source: str, detail: str) -> None:
        self.failure_class = "baseline_discontinuity"
        self.deterministic = True
        super().__init__(f"{source}: {detail}")


class FundNavValidationError(ValueError):
    """Deterministic OTC NAV schema or value validation failure."""

    def __init__(
        self,
        source: str,
        reason: str,
        *,
        total_rows: int,
        invalid_dates: int = 0,
        invalid_navs: int = 0,
        samples: list[dict[str, Any]] | None = None,
    ) -> None:
        self.source = source
        self.reason = reason
        self.total_rows = int(total_rows)
        self.invalid_dates = int(invalid_dates)
        self.invalid_navs = int(invalid_navs)
        self.samples = list(samples or [])
        super().__init__(
            f"{source}: {reason}; total_rows={self.total_rows}, "
            f"invalid_dates={self.invalid_dates}, invalid_navs={self.invalid_navs}, "
            f"samples={self.samples[:3]}"
        )


class RateLimiter:
    """Process-local spacing limiter shared by all worker threads."""

    def __init__(self, requests_per_second: float) -> None:
        self.interval = 0.0 if requests_per_second <= 0 else 1.0 / requests_per_second
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self) -> None:
        if self.interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next_allowed - now)
            self._next_allowed = max(now, self._next_allowed) + self.interval
        if delay:
            time.sleep(delay)


def install_request_timeout(seconds: float) -> None:
    """Set a default timeout for requests issued by AkShare in this process."""
    if requests is None or seconds <= 0:
        return
    session_class = requests.sessions.Session
    if getattr(session_class.request, "_dsh_timeout_wrapper", False):
        return
    original = session_class.request

    def request_with_timeout(self, method, url, **kwargs):
        kwargs.setdefault("timeout", float(seconds))
        return original(self, method, url, **kwargs)

    request_with_timeout._dsh_timeout_wrapper = True
    session_class.request = request_with_timeout


def target_date(value: str | None = None) -> str:
    if value:
        parsed = pd.Timestamp(value)
        if pd.isna(parsed):
            raise ValueError(f"invalid target date: {value}")
        return parsed.date().isoformat()
    return datetime.now(UTC).date().isoformat()


def next_start_date(data_end_date: str | None, default_start: str) -> str:
    if not data_end_date:
        return default_start
    parsed = pd.Timestamp(data_end_date)
    if pd.isna(parsed):
        return default_start
    return (parsed + timedelta(days=1)).strftime("%Y%m%d")


def _date_column(frame: pd.DataFrame) -> object | None:
    aliases = {"date", "日期", "净值日期", "交易日期"}
    for column in frame.columns:
        if str(column).strip().lower() in aliases:
            return column
    return None


def cached_end_date(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path, usecols=lambda column: str(column).strip() in {
            "date",
            "日期",
            "净值日期",
            "交易日期",
        })
        column = _date_column(frame)
        if column is None:
            return None
        values = pd.to_datetime(
            frame[column], errors="coerce", format="mixed"
        ).dropna()
        return values.max().date().isoformat() if not values.empty else None
    except (OSError, ValueError, pd.errors.ParserError):
        return None


def atomic_write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8-sig")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def atomic_write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def assert_basis_continuity(
    existing: pd.DataFrame,
    combined: pd.DataFrame,
    *,
    source: str,
    symbol: str,
) -> None:
    """校验拼接边界与重叠日期的收盘价基准连续性，断裂时抛异常拒写。

    ``existing`` 与 ``combined`` 都应是 ``normalize_daily_bars`` 的输出
    （已排序去重、date/close 列名规范）。两道检查：
    1. 边界台阶：existing 末行 close 与 combined 首个新增日期 close 的比值
       超出 ``[1/BASIS_STEP_RATIO, BASIS_STEP_RATIO]`` → 断裂；
    2. 重叠日期：两段序列同日收盘比值中位数偏离超过
       ``BASIS_OVERLAP_MEDIAN_DEVIATION`` → 口径不一致。
    """
    if existing is None or existing.empty or combined is None or combined.empty:
        return
    cut = existing["date"].max()
    new_rows = combined[combined["date"] > cut]
    prev_rows = combined[combined["date"] <= cut]
    if not new_rows.empty and not prev_rows.empty:
        prev_close = float(prev_rows["close"].iloc[-1])
        first_close = float(new_rows["close"].iloc[0])
        if prev_close > 0 and first_close > 0:
            ratio = first_close / prev_close
            if ratio > BASIS_STEP_RATIO or ratio < 1.0 / BASIS_STEP_RATIO:
                prev_date = pd.Timestamp(prev_rows["date"].iloc[-1]).date().isoformat()
                new_date = pd.Timestamp(new_rows["date"].iloc[0]).date().isoformat()
                raise BaselineDiscontinuityError(
                    source,
                    f"拼接边界基准断裂 close {prev_close:.4g}({prev_date}) -> "
                    f"{first_close:.4g}({new_date})，比值 {ratio:.2f} "
                    f"超出 [{1.0 / BASIS_STEP_RATIO:.2f}, {BASIS_STEP_RATIO:.2f}]",
                )
    shared = existing.merge(combined, on="date", suffixes=("_old", "_new"))
    shared = shared.dropna(subset=["close_old", "close_new"])
    shared = shared[(shared["close_old"] > 0) & (shared["close_new"] > 0)]
    if len(shared) < BASIS_OVERLAP_MIN_DATES:
        return
    deviation = (shared["close_new"] / shared["close_old"] - 1.0).abs()
    median_deviation = float(deviation.median())
    if median_deviation > BASIS_OVERLAP_MEDIAN_DEVIATION:
        raise BaselineDiscontinuityError(
            source,
            f"重叠日期收盘中位偏离 {median_deviation:.2%} 超过 "
            f"{BASIS_OVERLAP_MEDIAN_DEVIATION:.0%}（{len(shared)} 个重叠日，疑似口径切换）",
        )


def merge_daily_frames(
    existing_path: Path,
    incoming: pd.DataFrame,
    *,
    source: str,
) -> pd.DataFrame:
    if existing_path.exists():
        try:
            existing = pd.read_csv(existing_path)
        except pd.errors.EmptyDataError:
            existing = pd.DataFrame()
        combined = pd.concat([existing, incoming], ignore_index=True)
    else:
        existing = pd.DataFrame()
        combined = incoming.copy()
    result = normalize_daily_bars(combined, source=source)
    if not existing.empty:
        # 基准连续性守卫：在写盘前拦下跨源/跨口径拼接（毒数据事故根源）。
        # normalize 会重排日期，所以用整体归一后的 existing 对照 combined；
        # 守卫自身异常不阻断正常抓取，只有基准断裂才拒写。
        try:
            assert_basis_continuity(
                normalize_daily_bars(existing, source=source),
                result,
                source=source,
                symbol=existing_path.stem,
            )
        except BaselineDiscontinuityError:
            raise
        except Exception as _guard_exc:
            print("  [" + existing_path.stem + "] 基准连续性守卫跳过: " + type(_guard_exc).__name__)
    # ---- T1 基准 seam 探测（只判 incoming 行 vs 紧邻历史）----
    # 复权收盘出现台阶而单位修正后的原始 VWAP 连续 → 基准/口径切换，拒写
    # （走 worker 的 .discont.bak 全量重取路径）。只看 seam，历史内部的断裂
    # 一律不发射，避免每轮合并都重取同一段历史。
    if not existing.empty:
        try:
            from price_guards import detect_basis_seam as _bs_detect
            _existing_norm = normalize_daily_bars(existing, source=source)
            _existing_dates = set(_existing_norm["date"].astype(str))
            # 只把「新行」交给 seam 探测器（历史内部断裂不发射）。
            _new_mask = ~result["date"].astype(str).isin(_existing_dates)
            _new_rows = result.loc[_new_mask].reset_index(drop=True)
            _bs_raise, _bs_hold, _bs_reasons = _bs_detect(
                _existing_norm, _new_rows,
                source=source, symbol=existing_path.stem,
            )
            if bool(_bs_raise.any()):
                _r = _bs_reasons[0] if _bs_reasons else {}
                raise BaselineDiscontinuityError(
                    f"{source} {existing_path.stem}",
                    f"拼接 seam 基准台阶 close_step={_r.get('close_step')} "
                    f"vwap_step={_r.get('vwap_step')} ({_r.get('date')})，疑似口径切换",
                )
            if bool(_bs_hold.any()):
                # 未确认的尾部 seam 行：不写该行及之后任何行（不删、不制造缺口），
                # 下轮合并重新判定。只对「新行」生效，不动既有历史。
                _hold_pos = int(_bs_hold.values.argmax())
                _hold_dates = set(_new_rows["date"].astype(str).iloc[_hold_pos:])
                result = result.loc[~result["date"].astype(str).isin(_hold_dates)].reset_index(drop=True)
        except BaselineDiscontinuityError:
            raise
        except Exception as _bs_exc:
            print("  [" + existing_path.stem + "] 基准 seam 守卫跳过: " + type(_bs_exc).__name__)
    # ---- 价格守卫：分诊（删行仅在内部孤立尖刺时保留）----
    try:
        from price_guards import validate_daily_bars as _pg_validate
        from price_guards import write_quarantine as _pg_quarantine
        from price_guards import triage_price_flags as _pg_triage
        _pg_clean, _pg_bad, _pg_reasons = _pg_validate(result, source=source, symbol=existing_path.stem)
        if len(_pg_bad) > 0:
            _flag_mask = result["date"].isin(_pg_bad["date"])
            _raise_m, _delete_m, _hold_m = _pg_triage(result, _flag_mask)
            if bool(_raise_m.any()):
                _bad_dates = result.loc[_raise_m, "date"].astype(str).tolist()
                raise BaselineDiscontinuityError(
                    f"{source} {existing_path.stem}",
                    f"价格守卫确认段级断裂（{len(_bad_dates)} 行，{_bad_dates[:3]}），拒写并要求全量重取",
                )
            if bool(_hold_m.any()):
                _hpos = int(_hold_m.values.argmax())
                _hdates = set(result["date"].astype(str).iloc[_hpos:])
                result = result.loc[~result["date"].astype(str).isin(_hdates)].reset_index(drop=True)
            if bool(_delete_m.any()):
                _del = result.loc[_delete_m].reset_index(drop=True)
                _d_reasons = [r for r in _pg_reasons if str(r.get("date")) in set(_del["date"].astype(str))]
                _qpath = _pg_quarantine(existing_path.parent, existing_path.stem, _del, _d_reasons)
                print("  [" + existing_path.stem + "] 价格守卫隔离内部尖刺 " + str(len(_del)) + " 行 -> " + str(_qpath))
                result = result.loc[~_delete_m].reset_index(drop=True)
    except BaselineDiscontinuityError:
        raise
    except Exception as _pg_exc:
        print("  [" + existing_path.stem + "] 价格守卫跳过: " + type(_pg_exc).__name__)
    # ---- 单位守卫：修复字段而非删行 ----
    try:
        from price_guards import detect_unit_breaks as _ug_detect
        from price_guards import repair_unit_breaks as _ug_repair
        from price_guards import write_quarantine as _ug_quarantine
        _ug_flags, _ug_reasons = _ug_detect(result, source=source, symbol=existing_path.stem)
        if bool(_ug_flags.any()):
            _ug_bad = result.loc[_ug_flags].reset_index(drop=True)
            _ug_qpath = _ug_quarantine(existing_path.parent, existing_path.stem, _ug_bad, _ug_reasons)
            print("  [" + existing_path.stem + "] 单位守卫修正 " + str(len(_ug_bad)) + " 行（保留不删）-> " + str(_ug_qpath))
            result = _ug_repair(result, _ug_flags, source=source, symbol=existing_path.stem)
    except Exception as _ug_exc:
        print("  [" + existing_path.stem + "] 单位守卫跳过: " + type(_ug_exc).__name__)
    for key, value in incoming.attrs.items():
        result.attrs[key] = value
    return result


def _fund_columns(frame: pd.DataFrame) -> tuple[object | None, object | None]:
    date_column = next(
        (
            column
            for column in frame.columns
            if str(column).strip() in {"净值日期", "日期", "date"}
        ),
        None,
    )
    nav_column = next(
        (
            column
            for column in frame.columns
            if str(column).strip() in {"单位净值", "unit_nav", "nav", "close"}
        ),
        None,
    )
    return date_column, nav_column


def merge_otc_frames(
    existing_path: Path,
    incoming: pd.DataFrame,
    *,
    source: str,
) -> pd.DataFrame:
    frames = []
    if existing_path.exists():
        try:
            frames.append(pd.read_csv(existing_path))
        except pd.errors.EmptyDataError:
            pass
    frames.append(incoming)
    # 每帧先各自归一到 date/close 再拼接：已回填 CSV 的列名（date/close）与
    # akshare 原始列名（净值日期/单位净值）不同，直接 concat 会让交叉列
    # 互为 NaN，把有效的每日更新误判为数据异常并触发隔离。
    normalized = []
    for frame in frames:
        date_column, nav_column = _fund_columns(frame)
        if date_column is None or nav_column is None:
            raise FundNavValidationError(
                source,
                "fund NAV requires date and unit NAV columns",
                total_rows=len(frame),
                samples=[
                    {"columns": [str(column) for column in frame.columns[:10]]}
                ],
            )
        normalized.append(
            pd.DataFrame(
                {
                    "date": pd.to_datetime(
                        frame[date_column], errors="coerce", format="mixed"
                    ),
                    "close": pd.to_numeric(frame[nav_column], errors="coerce"),
                }
            )
        )
    combined = pd.concat(normalized, ignore_index=True)
    parsed_dates = combined["date"]
    parsed_navs = combined["close"]
    result = pd.DataFrame(
        {
            "date": parsed_dates,
            "close": parsed_navs,
        }
    )
    invalid_dates = parsed_dates.isna()
    invalid_navs = parsed_navs.isna()
    invalid = invalid_dates | invalid_navs
    if invalid.any():
        samples = []
        for index in combined.index[invalid][:3]:
            samples.append(
                {
                    "row": int(index) if isinstance(index, int) else str(index),
                    "date": str(combined.at[index, "date"]),
                    "nav": str(combined.at[index, "close"]),
                }
            )
        raise FundNavValidationError(
            source,
            "invalid fund NAV rows",
            total_rows=len(combined),
            invalid_dates=int(invalid_dates.sum()),
            invalid_navs=int(invalid_navs.sum()),
            samples=samples,
        )
    result = (
        result.sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )
    for key, value in incoming.attrs.items():
        result.attrs[key] = value
    return result
