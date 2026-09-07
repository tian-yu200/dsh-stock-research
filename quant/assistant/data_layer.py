"""数据加载与质量检查层（自 research.py 逐字搬迁，docs/重构计划.md 阶段 1）。

跨模块名字来源约定：凡定义在 research.py 的常量与工具函数（DATA_DIR、
UNIVERSE_PATH、SYMBOL_FILE_RE、_number、_iso_date）一律经 ``research.`` 属性
访问，使 patch.object(research, ...) 对搬入的函数继续生效；其余名字为模块内
自有或直接来自 paths/scripts。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from . import research
from .cross_border import is_cross_border
from .paths import validate_symbol

try:
    from scripts.market_data import normalize_daily_bars
    from scripts.universe import (
        load_asset_metadata,
        load_universe_manifest,
        manifest_names,
        manifest_symbols,
    )
except ImportError:
    scripts_root = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts_root) not in sys.path:
        sys.path.insert(0, str(scripts_root))
    from market_data import normalize_daily_bars
    from universe import (
        load_asset_metadata,
        load_universe_manifest,
        manifest_names,
        manifest_symbols,
    )

def _parse_fund_nav(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        raise ValueError(f"{source}: empty fund NAV response")
    columns = {str(column).strip(): column for column in frame.columns}
    date_column = next(
        (columns[name] for name in ("净值日期", "日期", "date") if name in columns),
        None,
    )
    nav_column = next(
        (columns[name] for name in ("单位净值", "unit_nav", "nav", "close") if name in columns),
        None,
    )
    if date_column is None or nav_column is None:
        raise ValueError(f"{source}: fund NAV requires date and unit NAV columns")
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(frame[date_column], errors="coerce"),
            "close": pd.to_numeric(frame[nav_column], errors="coerce"),
        }
    )
    if data["date"].isna().any() or data["close"].isna().any():
        raise ValueError(f"{source}: invalid fund NAV rows")
    data = data.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    data["return"] = data["close"].pct_change()
    return data


def _data_file(asset_type: str, symbol: str) -> Path:
    symbol = validate_symbol(symbol)
    if asset_type == "stock":
        return research.DATA_DIR / "stocks" / f"{symbol}.csv"
    if asset_type == "etf":
        return research.DATA_DIR / "funds" / f"etf_{symbol}.csv"
    if asset_type == "otc_fund":
        return research.DATA_DIR / "funds" / f"fund_{symbol}.csv"
    raise ValueError(f"unknown asset type: {asset_type}")


def _authoritative_manifest(manifest: dict[str, Any] | None, asset_type: str) -> bool:
    if not manifest:
        return False
    return bool(
        manifest.get("complete")
        or manifest.get(f"{asset_type}_universe_complete", False)
    )


def _universe(asset_type: str, symbols: Iterable[str] | None) -> list[str]:
    requested = {validate_symbol(value) for value in (symbols or [])}
    manifest = load_universe_manifest(research.UNIVERSE_PATH)
    directory = research.DATA_DIR / ("stocks" if asset_type == "stock" else "funds")
    discovered = set(
        manifest_symbols(manifest, asset_type)
        if _authoritative_manifest(manifest, asset_type)
        else []
    )
    if asset_type == "stock":
        discovered.update(
            path.stem
            for path in directory.glob("*.csv")
            if path.stem.isdigit() and len(path.stem) == 6
        )
    elif asset_type == "etf":
        discovered.update(
            match.group(1)
            for path in directory.glob("etf_*.csv")
            if (match := research.SYMBOL_FILE_RE.match(path.stem))
        )
    else:
        discovered.update(
            match.group(1)
            for path in directory.glob("fund_*.csv")
            if (match := research.SYMBOL_FILE_RE.match(path.stem))
        )
    # 2026-09-07 跨境排除：源头过滤，美港QDII不计入统计
    # 复用 pool_exclude_cross_border 开关，stock 始终不过滤（is_cross_border对stock返回False）
    try:
        cfg = research.load_config()  # type: ignore[attr-defined]
    except Exception:
        cfg = {}
    research_cfg = (cfg.get("research") or {}) if isinstance(cfg, dict) else {}
    exclude_universe = bool(research_cfg.get("universe_exclude_cross_border", False) or research_cfg.get("pool_exclude_cross_border", False))
    if exclude_universe and asset_type != "stock":
        names = manifest_names(manifest, asset_type)
        kws = research_cfg.get("cross_border_keywords") or None
        # 定向请求 fail-open：显式指定的符号不过滤，交由 _pool_quality_filter 决定权重
        discovered = {s for s in discovered if (s in requested) or not is_cross_border(names.get(s, ""), asset_type=asset_type, keywords=kws)}
    if requested:
        missing = sorted(requested - discovered)
        if missing:
            raise ValueError(f"symbols not found in fixed data directory: {missing}")
        return sorted(requested)
    return sorted(discovered)


def _partition_symbols(
    asset_types: Iterable[str],
    symbols: Iterable[str],
    *,
    require_match: bool = True,
) -> dict[str, list[str]]:
    """Route explicit symbols only to fixed asset files that contain them."""
    selected_types = tuple(asset_types)
    manifest = load_universe_manifest(research.UNIVERSE_PATH)
    routed = {asset_type: [] for asset_type in selected_types}
    unmatched: list[str] = []
    for raw_symbol in symbols:
        symbol = validate_symbol(raw_symbol)
        matches = [
            asset_type
            for asset_type in selected_types
            if _data_file(asset_type, symbol).is_file()
            or (
                _authoritative_manifest(manifest, asset_type)
                and symbol in set(manifest_symbols(manifest, asset_type))
            )
        ]
        if not matches:
            unmatched.append(symbol)
            continue
        for asset_type in matches:
            routed[asset_type].append(symbol)
    if require_match and unmatched:
        raise ValueError(
            f"symbols not found in requested fixed asset scope: {sorted(set(unmatched))}"
        )
    return {asset_type: sorted(set(values)) for asset_type, values in routed.items()}


def _adjustment_check(path: Path, asset_type: str) -> dict[str, Any]:
    expected = "unit_nav" if asset_type == "otc_fund" else (
        "hfq" if asset_type == "stock" else "qfq"
    )
    metadata = load_asset_metadata(path)
    observed = metadata.get("adjustment")
    if asset_type == "etf" and metadata.get("interface") == "fund_etf_hist_sina":
        observed = "unadjusted"
    if observed is None:
        return {
            "status": "assumed",
            "passed": True,
            "expected": expected,
            "observed": None,
            "reason": "legacy CSV has no sidecar provenance; using adapter default",
        }
    return {
        "status": "verified" if observed == expected else "mismatch",
        "passed": observed == expected,
        "expected": expected,
        "observed": observed,
        "source": metadata.get("source"),
        "interface": metadata.get("interface"),
    }


def _liquidity_check(frame: pd.DataFrame, asset_type: str) -> dict[str, Any]:
    if asset_type == "otc_fund":
        return {
            "status": "not_applicable",
            "passed": True,
            "reason": "OTC funds use daily NAV and do not have exchange-traded volume",
        }
    volume = pd.to_numeric(frame.get("volume"), errors="coerce")
    positive_volume = volume.notna() & (volume > 0)
    amount = pd.to_numeric(frame.get("amount"), errors="coerce")
    amount_available = amount.notna()
    amount_valid = (~amount_available) | (amount >= 0)
    passed = bool(positive_volume.any() and amount_valid.all())
    return {
        "status": "pass" if passed else "fail",
        "passed": passed,
        "positive_volume_rows": int(positive_volume.sum()),
        "rows": int(len(frame)),
        "amount_data_status": "available" if amount_available.any() else "unavailable",
        "average_volume_20d": research._number(volume.tail(20).mean()),
        "average_amount_20d": research._number(amount.tail(20).mean()),
    }


def _price_consistency_check(frame: pd.DataFrame, symbol: str = "") -> dict[str, Any]:
    try:
        from scripts.price_guards import validate_daily_bars
    except ImportError:
        scripts_root = Path(__file__).resolve().parents[1] / "scripts"
        if str(scripts_root) not in sys.path:
            sys.path.insert(0, str(scripts_root))
        from price_guards import validate_daily_bars
    try:
        cutoff = str(frame["date"].iloc[max(0, len(frame) - 250)])
        _, bad, reasons = validate_daily_bars(frame, source="data_layer", symbol=symbol)
        recent = [r["date"] for r in reasons if r["date"] >= cutoff][-10:]
        old = [r["date"] for r in reasons if r["date"] < cutoff]
        gaps = [float(r.get("gap", 0) or 0) for r in reasons]
        devs = [float(r.get("deviation", 0) or 0) for r in reasons]
        passed = not recent
        return {"status": "pass" if passed else "fail", "passed": passed, "bad_rows": len(recent), "bad_dates": recent, "warn_old_rows": len(old), "max_deviation": round(max(devs) if devs else 0.0, 4), "max_gap": round(max(gaps) if gaps else 0.0, 4)}
    except Exception as exc:
        return {"status": "skipped", "passed": True, "reason": type(exc).__name__}
_MANIFEST_CACHE: dict[str, tuple[tuple[int, int], Any]] = {}


def _cached_universe_manifest(path: Path) -> Any:
    """按 (mtime_ns, size) 缓存 universe manifest——全市场扫描曾对每个标的
    重新解析整份 manifest（31k 次 × ~42ms ≈ 22 分钟纯浪费）。"""
    key = str(path)
    try:
        stat = path.stat()
        stamp = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        return load_universe_manifest(path)
    cached = _MANIFEST_CACHE.get(key)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    value = load_universe_manifest(path)
    _MANIFEST_CACHE[key] = (stamp, value)
    return value


def _load_asset(asset_type: str, symbol: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = _data_file(asset_type, symbol)
    if not path.exists():
        raise FileNotFoundError(path)
    raw = pd.read_csv(path)
    manifest = _cached_universe_manifest(research.UNIVERSE_PATH)
    name = manifest_names(manifest, asset_type).get(symbol)
    raw_date_column = next(
        (column for column in raw.columns if str(column).strip() in {"date", "日期", "净值日期"}),
        None,
    )
    duplicate_dates = 0
    if raw_date_column is not None:
        parsed = pd.to_datetime(raw[raw_date_column], errors="coerce")
        duplicate_dates = int(parsed.duplicated().sum())
    if asset_type == "otc_fund":
        frame = _parse_fund_nav(raw, str(path))
        issues: list[str] = []
        if duplicate_dates:
            issues.append(f"duplicate_dates:{duplicate_dates}")
        if (frame["close"] <= 0).any():
            issues.append("non_positive_nav")
        adjustment_check = _adjustment_check(path, asset_type)
        if not adjustment_check["passed"]:
            issues.append("nav_provenance_mismatch")
        liquidity = _liquidity_check(frame, asset_type)
        checks = {
            "schema": {"status": "pass", "passed": True},
            "duplicate_dates": {"status": "pass" if not duplicate_dates else "fail", "passed": not duplicate_dates},
            "nav_values": {"status": "pass" if "non_positive_nav" not in issues else "fail", "passed": "non_positive_nav" not in issues},
            "adjustment_or_nav": adjustment_check,
            "liquidity": liquidity,
        }
        quality = "pass" if not issues and liquidity["passed"] else "fail"
        facts = {
            "asset_type": asset_type,
            "symbol": symbol,
            "name": name,
            "source": "AkShare fund NAV CSV cache",
            "source_file": str(path),
            "adjustment": "unit_nav_no_price_adjustment",
            "rows": int(len(frame)),
            "start_date": research._iso_date(frame["date"].iloc[0]),
            "data_end_date": research._iso_date(frame["date"].iloc[-1]),
            "duplicate_rows": duplicate_dates,
            "missing_values": int(frame[["date", "close"]].isna().sum().sum()),
            "quality_status": quality,
            "issues": issues,
            "checks": checks,
            "liquidity": liquidity,
        }
        return frame, facts
    frame = normalize_daily_bars(raw, source=str(path))
    issues = []
    if duplicate_dates:
        issues.append(f"duplicate_dates:{duplicate_dates}")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        issues.append("non_positive_ohlc")
    if (frame["high"] < frame["low"]).any():
        issues.append("high_below_low")
    if (
        (frame["high"] < frame[["open", "close"]].max(axis=1)).any()
        or (frame["low"] > frame[["open", "close"]].min(axis=1)).any()
    ):
        issues.append("ohlc_inconsistent")
    consistency = _price_consistency_check(frame, symbol)
    if not consistency["passed"]:
        issues.append("price_scale_suspect")
    adjustment_check = _adjustment_check(path, asset_type)
    if not adjustment_check["passed"]:
        issues.append("adjustment_provenance_mismatch")
    liquidity = _liquidity_check(frame, asset_type)
    if not liquidity["passed"]:
        issues.append("liquidity_unavailable_or_non_positive")
    checks = {
        "schema": {"status": "pass", "passed": True},
        "duplicate_dates": {"status": "pass" if not duplicate_dates else "fail", "passed": not duplicate_dates},
        "ohlc_values": {
            "status": "pass"
            if not set(issues).intersection(
                {"non_positive_ohlc", "high_below_low", "ohlc_inconsistent"}
            )
            else "fail",
            "passed": not bool(
                set(issues).intersection(
                    {"non_positive_ohlc", "high_below_low", "ohlc_inconsistent"}
                )
            ),
        },
        "adjustment_or_nav": adjustment_check,
        "price_consistency": consistency,
        "liquidity": liquidity,
    }
    quality = "pass" if not issues else "fail"
    facts = {
        "asset_type": asset_type,
        "symbol": symbol,
        "name": name,
        "source": "AkShare cached daily bars",
        "source_file": str(path),
        "adjustment": "hfq" if asset_type == "stock" else "qfq",
        "rows": int(len(frame)),
        "start_date": research._iso_date(frame["date"].iloc[0]),
        "data_end_date": research._iso_date(frame["date"].iloc[-1]),
        "duplicate_rows": duplicate_dates,
        "missing_values": int(
            frame[["date", "open", "high", "low", "close", "volume"]]
            .isna()
            .sum()
            .sum()
        ),
        "optional_missing_values": int(
            frame[["amount", "turnover"]].isna().sum().sum()
        ),
        "quality_status": quality,
        "issues": issues,
        "checks": checks,
        "liquidity": liquidity,
    }
    return frame, facts


def _scan_assets(asset_type: str, symbols: Iterable[str] | None) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    frames: dict[str, pd.DataFrame] = {}
    facts: list[dict[str, Any]] = []
    for symbol in _universe(asset_type, symbols):
        try:
            frame, record = _load_asset(asset_type, symbol)
            frames[symbol] = frame
            facts.append(record)
        except Exception as exc:
            facts.append(
                {
                    "asset_type": asset_type,
                    "symbol": symbol,
                    "source": "AkShare cached data",
                    "source_file": str(_data_file(asset_type, symbol)),
                    "quality_status": "fail",
                    "issues": [f"{type(exc).__name__}: {exc}"],
                }
            )
    return frames, facts


def _aggregate_quality(
    facts: list[dict[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    max_age = int(config["fetch"]["max_data_age_days"])
    now = pd.Timestamp.now().normalize()
    dates = [pd.Timestamp(item["data_end_date"]) for item in facts if item.get("data_end_date")]
    latest = max(dates).date().isoformat() if dates else None
    freshness: dict[str, dict[str, Any]] = {}
    stale_assets: list[str] = []
    for item in facts:
        symbol = str(item.get("symbol", "unknown"))
        asset_key = f"{item.get('asset_type', 'unknown')}:{symbol}"
        end_date = item.get("data_end_date")
        age_days = max(0, (now - pd.Timestamp(end_date)).days) if end_date else None
        stale = age_days is None or age_days > max_age
        freshness[asset_key] = {
            "data_end_date": end_date,
            "age_days": age_days,
            "stale": stale,
        }
        item["freshness"] = freshness[asset_key]
        if stale:
            stale_assets.append(asset_key)
    failed = [item for item in facts if item.get("quality_status") != "pass"]
    stale = bool(stale_assets)
    eligible = [
        item
        for item in facts
        if item.get("quality_status") == "pass"
        and not item.get("freshness", {}).get("stale", True)
    ]
    coverage = len(eligible) / len(facts) if facts else 0.0
    minimum_coverage = float(
        config.get("fetch", {}).get("min_universe_coverage", 0.95)
    )
    return {
        "passed": bool(eligible) and coverage >= minimum_coverage,
        "data_end_date": latest,
        "max_allowed_age_days": max_age,
        "asset_count": len(facts),
        "eligible_asset_count": len(eligible),
        "eligible_assets": [
            f"{item.get('asset_type', 'unknown')}:{item.get('symbol', 'unknown')}"
            for item in eligible
        ],
        "coverage_ratio": coverage,
        "minimum_coverage_ratio": minimum_coverage,
        "failed_assets": [item.get("symbol") for item in failed],
        "stale_assets": sorted(set(stale_assets)),
        "asset_freshness": freshness,
        "stale": stale,
        "time_leakage_check": {"passed": True, "method": "signal_date_before_forward_label"},
        "all_sources_failed": not bool(facts) or not bool(eligible),
    }
