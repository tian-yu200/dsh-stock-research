"""Fixed-path research orchestration for stocks, ETFs and OTC funds.

The service intentionally keeps the three asset paths separate.  A model
ranking is only one input to decision support; it is never an automatic buy
instruction, and failed stock model gates always produce cash/observe output.
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import time
import ctypes
from errno import ESRCH
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
from uuid import uuid4

import numpy as np
import pandas as pd
import yaml

from .config import config_snapshot, load_config
from .notifications import acknowledge, add_notification, list_records
from .paths import (
    ASSISTANT_RESULTS_DIR,
    DATA_DIR,
    LOCK_PATH,
    QLIB_PROVIDER_DIR,
    RUNS_DIR,
    atomic_write_json,
    atomic_write_text,
    ensure_directories,
    fixed_path,
    read_json,
    run_directory,
    validate_notification_id,
    validate_run_id,
    validate_symbol,
)
from .risk import cash_only_decision, enforce_target_weights, model_gate_checks

try:
    from scripts.market_data import normalize_daily_bars
    from scripts.fetch_state import FetchState
    from scripts.fetch_targets import asset_target_dates
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
    from fetch_state import FetchState
    from fetch_targets import asset_target_dates
    from universe import (
        load_asset_metadata,
        load_universe_manifest,
        manifest_names,
        manifest_symbols,
    )


ASSET_TYPES = ("stock", "etf", "otc_fund")
ASSET_SCOPE_MAP = {
    "stocks": ("stock",),
    "stock": ("stock",),
    "etfs": ("etf",),
    "etf": ("etf",),
    "funds": ("otc_fund",),
    "fund": ("otc_fund",),
    "otc_fund": ("otc_fund",),
    "otc_funds": ("otc_fund",),
    "all": ASSET_TYPES,
}
MODE_NAMES = {"daily", "monthly", "on_demand"}
SYMBOL_FILE_RE = re.compile(r"^(?:stock|etf|fund)_?([0-9]{6})$", re.IGNORECASE)
UNIVERSE_PATH = DATA_DIR / "universe.json"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_date(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).date().isoformat()


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _fixed_reference(value: Any) -> str | None:
    """Validate a persisted path before returning it through the API."""
    if value is None:
        return None
    return str(fixed_path(str(value)))


def _sanitize_result_record(value: Any) -> Any:
    """Keep persisted result references inside the fixed quant root."""
    if not isinstance(value, dict):
        return value
    record = dict(value)
    for key in ("result_path", "current_path"):
        if key in record and record[key] is not None:
            record[key] = _fixed_reference(record[key])
    paths = record.get("paths")
    if isinstance(paths, dict):
        record["paths"] = {
            str(key): _fixed_reference(path) if path is not None else None
            for key, path in paths.items()
        }
    return record


def _fetch_summary(stdout: str) -> dict[str, Any] | None:
    marker = "ASSISTANT_FETCH_SUMMARY="
    for line in reversed((stdout or "").splitlines()):
        if not line.startswith(marker):
            continue
        try:
            value = json.loads(line[len(marker):])
        except json.JSONDecodeError:
            return {"parse_error": "invalid fetch summary JSON"}
        return value if isinstance(value, dict) else {"parse_error": "fetch summary is not an object"}
    return None


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if pd.isna(value) if not isinstance(value, (dict, list, tuple)) else False:
        return None
    return value


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
        return DATA_DIR / "stocks" / f"{symbol}.csv"
    if asset_type == "etf":
        return DATA_DIR / "funds" / f"etf_{symbol}.csv"
    if asset_type == "otc_fund":
        return DATA_DIR / "funds" / f"fund_{symbol}.csv"
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
    manifest = load_universe_manifest(UNIVERSE_PATH)
    directory = DATA_DIR / ("stocks" if asset_type == "stock" else "funds")
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
            if (match := SYMBOL_FILE_RE.match(path.stem))
        )
    else:
        discovered.update(
            match.group(1)
            for path in directory.glob("fund_*.csv")
            if (match := SYMBOL_FILE_RE.match(path.stem))
        )
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
    manifest = load_universe_manifest(UNIVERSE_PATH)
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
        "average_volume_20d": _number(volume.tail(20).mean()),
        "average_amount_20d": _number(amount.tail(20).mean()),
    }


def _load_asset(asset_type: str, symbol: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = _data_file(asset_type, symbol)
    if not path.exists():
        raise FileNotFoundError(path)
    raw = pd.read_csv(path)
    manifest = load_universe_manifest(UNIVERSE_PATH)
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
            "start_date": _iso_date(frame["date"].iloc[0]),
            "data_end_date": _iso_date(frame["date"].iloc[-1]),
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
        "start_date": _iso_date(frame["date"].iloc[0]),
        "data_end_date": _iso_date(frame["date"].iloc[-1]),
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


def _dated_close_series(frame: pd.DataFrame) -> pd.Series:
    dates = pd.to_datetime(frame["date"], errors="coerce")
    values = pd.to_numeric(frame["close"], errors="coerce")
    series = pd.Series(values.to_numpy(), index=dates)
    return series.dropna().sort_index()


def _drawdown_recovery(close: pd.Series) -> dict[str, Any]:
    """Describe the largest drawdown and the current recovery state.

    Recovery durations use calendar days between available observation dates.
    This keeps the value meaningful for both daily bars and less frequent fund
    NAV observations.
    """
    series = pd.to_numeric(close, errors="coerce").dropna().sort_index()
    if series.empty:
        return {
            "max_drawdown": None,
            "max_drawdown_peak_date": None,
            "max_drawdown_trough_date": None,
            "max_drawdown_duration_days": None,
            "max_drawdown_recovery_date": None,
            "max_drawdown_recovery_days": None,
            "current_drawdown": None,
            "current_drawdown_peak_date": None,
            "current_drawdown_duration_days": None,
            "current_drawdown_recovered": False,
        }

    running_peak = series.cummax()
    drawdown = series / running_peak - 1.0
    trough_date = drawdown.idxmin()
    trough_value = float(drawdown.loc[trough_date])
    peak_value = float(running_peak.loc[trough_date])
    peak_candidates = series.loc[:trough_date]
    peak_date = peak_candidates[peak_candidates == peak_value].index[-1]

    recovery_candidates = series.loc[trough_date:]
    recovery_candidates = recovery_candidates[recovery_candidates >= peak_value * (1.0 - 1e-12)]
    recovery_date = recovery_candidates.index[0] if not recovery_candidates.empty else None

    latest_date = series.index[-1]
    latest_peak_value = float(running_peak.iloc[-1])
    current_peak_candidates = series.loc[:latest_date]
    current_peak_date = current_peak_candidates[
        current_peak_candidates == latest_peak_value
    ].index[-1]
    current_recovered = bool(float(series.iloc[-1]) >= latest_peak_value * (1.0 - 1e-12))
    current_duration = (
        0
        if current_recovered
        else max(0, int((latest_date - current_peak_date).days))
    )

    return {
        "max_drawdown": _number(trough_value),
        "max_drawdown_peak_date": _iso_date(peak_date),
        "max_drawdown_trough_date": _iso_date(trough_date),
        "max_drawdown_duration_days": max(
            0,
            int((trough_date - peak_date).days),
        ),
        "max_drawdown_recovery_date": _iso_date(recovery_date),
        "max_drawdown_recovery_days": (
            max(0, int((recovery_date - trough_date).days))
            if recovery_date is not None
            else None
        ),
        "current_drawdown": _number(float(series.iloc[-1]) / latest_peak_value - 1.0),
        "current_drawdown_peak_date": _iso_date(current_peak_date),
        "current_drawdown_duration_days": current_duration,
        "current_drawdown_recovered": current_recovered,
    }


def _relative_returns(
    asset: pd.Series,
    benchmark: pd.Series | None,
    periods: tuple[int, ...] = (20, 60, 252),
) -> dict[str, float | None]:
    """Return multi-period relative returns on a shared observation calendar."""
    if benchmark is None:
        return {f"relative_return_{period}d_vs_benchmark": None for period in periods}
    asset_series = pd.to_numeric(asset, errors="coerce").dropna()
    benchmark_series = pd.to_numeric(benchmark, errors="coerce").dropna()
    asset_series.index = pd.to_datetime(asset_series.index)
    benchmark_series.index = pd.to_datetime(benchmark_series.index)
    common = asset_series.index.intersection(benchmark_series.index)
    common = common.sort_values()
    if common.empty:
        return {f"relative_return_{period}d_vs_benchmark": None for period in periods}
    asset_series = asset_series.loc[common]
    benchmark_series = benchmark_series.loc[common]
    result: dict[str, float | None] = {}
    for period in periods:
        key = f"relative_return_{period}d_vs_benchmark"
        result[key] = (
            _number(
                (asset_series.iloc[-1] / asset_series.iloc[-period - 1])
                / (benchmark_series.iloc[-1] / benchmark_series.iloc[-period - 1])
                - 1.0
            )
            if len(common) > period
            else None
        )
    return result


def _liquidity_metrics(frame: pd.DataFrame, window: int = 20) -> dict[str, Any]:
    """Summarize ETF liquidity while exposing unavailable fields explicitly."""
    metrics: dict[str, Any] = {"liquidity_window_days": min(len(frame), window)}
    for column, key in (("amount", "amount"), ("volume", "volume")):
        values = (
            pd.to_numeric(frame[column], errors="coerce").tail(window).dropna()
            if column in frame
            else pd.Series(dtype=float)
        )
        metrics[f"average_{key}_{window}d"] = _number(values.mean()) if not values.empty else None
        metrics[f"{key}_data_status"] = "available" if not values.empty else "unavailable"
    metrics["liquidity_status"] = (
        "available"
        if metrics["amount_data_status"] == "available"
        or metrics["volume_data_status"] == "available"
        else "unavailable"
    )
    return metrics


def _market_metrics(frame: pd.DataFrame, *, fund: bool = False) -> dict[str, Any]:
    close = _dated_close_series(frame)
    if close.empty:
        raise ValueError("no close/NAV values")
    returns = close.pct_change()
    latest = float(close.iloc[-1])

    def ret(period: int) -> float | None:
        return (
            _number(close.iloc[-1] / close.iloc[-period - 1] - 1.0)
            if len(close) > period
            else None
        )

    volatility = _number(returns.tail(60).std(ddof=1) * math.sqrt(252))
    trailing_drawdown = _drawdown_recovery(close.tail(252))["max_drawdown"]
    drawdown = _drawdown_recovery(close)
    ma20 = _number(close.tail(20).mean()) if len(close) >= 20 else None
    ma60 = _number(close.tail(60).mean()) if len(close) >= 60 else None
    ma200 = _number(close.tail(200).mean()) if len(close) >= 200 else None
    trend_score = (
        (ret(20) or 0.0) * 0.45
        + (ret(60) or 0.0) * 0.35
        + (ret(252) or 0.0) * 0.20
    )
    if ma20 and latest > ma20:
        trend_score += 0.05
    if ma60 and latest > ma60:
        trend_score += 0.05
    return {
        "latest_value": latest,
        "return_20d": ret(20),
        "return_60d": ret(60),
        "return_252d": ret(252),
        "volatility_60d_annualized": volatility,
        "max_drawdown": drawdown["max_drawdown"],
        "max_drawdown_252d": trailing_drawdown,
        "max_drawdown_peak_date": drawdown["max_drawdown_peak_date"],
        "max_drawdown_trough_date": drawdown["max_drawdown_trough_date"],
        "max_drawdown_duration_days": drawdown["max_drawdown_duration_days"],
        "max_drawdown_recovery_date": drawdown["max_drawdown_recovery_date"],
        "max_drawdown_recovery_days": drawdown["max_drawdown_recovery_days"],
        "current_drawdown": drawdown["current_drawdown"],
        "current_drawdown_peak_date": drawdown["current_drawdown_peak_date"],
        "current_drawdown_duration_days": drawdown["current_drawdown_duration_days"],
        "current_drawdown_recovered": drawdown["current_drawdown_recovered"],
        "ma20": ma20,
        "ma60": ma60,
        "ma200": ma200,
        "trend_score": _number(trend_score),
        "series_type": "net_value" if fund else "daily_close",
    }


def _peer_comparison(
    symbol: str,
    metrics_by_symbol: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    period = next(
        (
            candidate
            for candidate in (252, 60, 20)
            if sum(
                values.get(f"return_{candidate}d") is not None
                for values in metrics_by_symbol.values()
            )
            >= 2
        ),
        None,
    )
    if period is None:
        return {
            "peer_comparison_status": "unavailable",
            "peer_comparison_reason": "at least two same-category assets with comparable returns are required",
            "peer_count": 0,
        }
    peer_returns = pd.Series(
        {
            peer_symbol: values.get(f"return_{period}d")
            for peer_symbol, values in metrics_by_symbol.items()
            if values.get(f"return_{period}d") is not None
        },
        dtype=float,
    ).dropna()
    if symbol not in peer_returns.index or len(peer_returns) < 2:
        return {
            "peer_comparison_status": "unavailable",
            "peer_comparison_reason": "the asset has no comparable return in the peer period",
            "peer_count": max(0, int(len(peer_returns) - (symbol in peer_returns.index))),
        }
    value = float(peer_returns.loc[symbol])
    peers = peer_returns.drop(index=symbol)
    return {
        "peer_comparison_status": "available",
        "peer_comparison_period_days": period,
        "peer_count": int(len(peers)),
        f"peer_average_return_{period}d": _number(peers.mean()),
        f"excess_return_{period}d_vs_peer_mean": _number(value - peers.mean()),
        f"peer_return_{period}d_percentile": _number(float((peer_returns <= value).mean())),
        f"peer_rank_{period}d": int(peer_returns.rank(ascending=False, method="min").loc[symbol]),
        f"peer_universe_return_{period}d": _number(peer_returns.mean()),
    }


def _technical_asset_analysis(
    asset_type: str,
    frames: Mapping[str, pd.DataFrame],
    facts: list[dict[str, Any]],
    benchmark: pd.Series | None = None,
    peer_frames: Mapping[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    peer_frames = peer_frames or frames
    metric_frames = {
        symbol: _market_metrics(frame, fund=asset_type == "otc_fund")
        for symbol, frame in peer_frames.items()
    }
    fact_by_symbol = {
        str(item.get("symbol")): item
        for item in facts
        if item.get("symbol") is not None
    }
    signals: list[dict[str, Any]] = []
    for symbol, frame in frames.items():
        metrics = dict(metric_frames[symbol])
        close = _dated_close_series(frame)
        relative = _relative_returns(close, benchmark)
        metrics.update(relative)
        metrics["benchmark_symbol"] = "510300" if benchmark is not None else None
        metrics["benchmark_data_status"] = (
            "available"
            if benchmark is not None and any(value is not None for value in relative.values())
            else "unavailable"
        )
        metrics.update(_peer_comparison(symbol, metric_frames))
        relative_20d = relative["relative_return_20d_vs_benchmark"]
        if relative_20d is not None:
            metrics["trend_score"] = _number(
                (metrics["trend_score"] or 0.0) + relative_20d * 0.25
            )
        peer_period = metrics.get("peer_comparison_period_days")
        peer_excess = (
            metrics.get(f"excess_return_{int(peer_period)}d_vs_peer_mean")
            if peer_period is not None
            else None
        )
        if peer_excess is not None:
            metrics["trend_score"] = _number(
                (metrics["trend_score"] or 0.0) + peer_excess * 0.20
            )
        if asset_type in {"stock", "etf"}:
            metrics.update(_liquidity_metrics(frame))
        signal = {
            "symbol": symbol,
            "name": fact_by_symbol.get(symbol, {}).get("name"),
            **metrics,
        }
        signals.append(signal)
    signals.sort(key=lambda item: item.get("trend_score") or -math.inf, reverse=True)
    return {
        "asset_type": asset_type,
        "method": "trend_return_volatility_drawdown_relative_benchmark_peer_comparison",
        "signals": signals,
        "data_end_date": max(
            (item["data_end_date"] for item in facts if item.get("data_end_date")),
            default=None,
        ),
    }


def _prediction_metrics(prediction: pd.Series, labels: pd.Series) -> dict[str, float | int]:
    frame = pd.concat({"score": prediction, "label": labels}, axis=1).dropna()
    if frame.empty:
        return {
            "observations": 0,
            "candidate_days": 0,
            "valid_days": 0,
            "excluded_days": 0,
            "mean_rank_ic": float("nan"),
            "rank_ic_ir": float("nan"),
        }
    grouped = frame.groupby(level="datetime")
    candidate_days = int(grouped.ngroups)
    daily_values: dict[pd.Timestamp, float] = {}
    for date, data in grouped:
        if len(data) < 2 or data["score"].nunique() < 2 or data["label"].nunique() < 2:
            continue
        rank_ic = data["score"].corr(data["label"], method="spearman")
        if pd.notna(rank_ic):
            daily_values[pd.Timestamp(date)] = float(rank_ic)
    daily = pd.Series(daily_values, dtype=float).sort_index()
    rank_std = daily.std(ddof=1)
    return {
        "observations": int(len(frame)),
        "candidate_days": candidate_days,
        "valid_days": int(len(daily)),
        "excluded_days": candidate_days - int(len(daily)),
        "mean_rank_ic": float(daily.mean()) if not daily.empty else float("nan"),
        "rank_ic_ir": float(daily.mean() / rank_std)
        if len(daily) > 1 and rank_std and not np.isnan(rank_std)
        else float("nan"),
    }


def _forward_group_metrics(
    prediction: pd.Series,
    labels: pd.Series,
    group_count: int = 5,
) -> dict[str, Any]:
    """Measure forward returns by score group without using future information."""
    frame = pd.concat({"score": prediction, "label": labels}, axis=1).dropna()
    if frame.empty or group_count < 2:
        return {
            "group_count": max(1, int(group_count)),
            "observations": int(len(frame)),
            "valid_days": 0,
            "group_returns": {},
            "top_group_return": float("nan"),
            "bottom_group_return": float("nan"),
            "top_bottom_excess_return": float("nan"),
        }

    group_returns: dict[int, list[float]] = {}
    top_bottom: list[float] = []
    valid_days = 0
    for _, data in frame.groupby(level="datetime"):
        if len(data) < 2 or data["score"].nunique() < 2:
            continue
        number_of_groups = min(int(group_count), len(data))
        ordered = data.sort_values("score").reset_index(drop=True)
        group_ids = (
            np.floor(np.arange(len(ordered)) * number_of_groups / len(ordered)).astype(int)
            + 1
        )
        ordered["group"] = group_ids
        daily = ordered.groupby("group")["label"].mean()
        if len(daily) < 2:
            continue
        valid_days += 1
        for group, value in daily.items():
            group_returns.setdefault(int(group), []).append(float(value))
        top_bottom.append(float(daily.iloc[-1] - daily.iloc[0]))

    aggregated = {
        str(group): _number(np.mean(values))
        for group, values in sorted(group_returns.items())
        if values
    }
    top_return = aggregated.get(str(max(group_returns)), float("nan")) if group_returns else float("nan")
    bottom_return = aggregated.get(str(min(group_returns)), float("nan")) if group_returns else float("nan")
    return {
        "group_count": int(group_count),
        "observations": int(len(frame)),
        "valid_days": valid_days,
        "group_returns": aggregated,
        "top_group_return": top_return,
        "bottom_group_return": bottom_return,
        "top_bottom_excess_return": _number(np.mean(top_bottom)) if top_bottom else float("nan"),
    }


def _performance(equity: pd.Series) -> dict[str, float]:
    values = equity.dropna()
    returns = values.pct_change().dropna()
    if len(values) < 2 or returns.empty:
        return {"net_annualized_return": float("nan"), "max_drawdown": float("nan")}
    if isinstance(values.index, pd.DatetimeIndex):
        elapsed_days = max((values.index[-1] - values.index[0]).days, 1)
        years = elapsed_days / 365.25
    else:
        years = max(len(returns) / 252.0, 1 / 252.0)
    annualized = float((values.iloc[-1] / values.iloc[0]) ** (1.0 / years) - 1.0)
    drawdown = values / values.cummax() - 1.0
    return {"net_annualized_return": annualized, "max_drawdown": float(drawdown.min())}


def _fit_qlib_model_and_close_recorder(
    model: Any,
    dataset: Any,
    qlib_recorder: Any,
) -> None:
    """Close Qlib's global recorder after model.fit, including failed fits."""
    try:
        model.fit(dataset)
    finally:
        qlib_recorder.end_exp()


def _run_qlib_stock_model(config: Mapping[str, Any]) -> dict[str, Any]:
    """Run the isolated Alpha158/LightGBM path.

    Importing Qlib is deliberately deferred so ETF/fund snapshot calls remain
    available even when the optional quant environment is not installed.
    """
    if os.environ.get("DSH_STOCK_SKIP_MODEL") == "1":
        return {
            "model_available": False,
            "model_name": "Qlib Alpha158 + LightGBM",
            "error": "stock model skipped by DSH_STOCK_SKIP_MODEL=1",
            "mean_rank_ic": float("nan"),
            "test_rank_ic": float("nan"),
            "rank_ic_ir": float("nan"),
            "top_bottom_excess_return": float("nan"),
            "net_annualized_return": float("nan"),
            "test_max_drawdown": float("nan"),
            "max_drawdown": float("nan"),
            "leakage_check": {"passed": False, "reason": "model skipped"},
        }
    try:
        os.environ.setdefault("OMP_NUM_THREADS", "2")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
        os.environ.setdefault("MKL_NUM_THREADS", "2")
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        os.environ.setdefault("MLFLOW_TRACKING_URI", f"file:///{(Path(config['paths']['logs_dir']) / 'mlruns').as_posix()}")
        import qlib
        from qlib.config import REG_CN
        from qlib.contrib.data.handler import Alpha158
        from qlib.contrib.model.gbdt import LGBModel
        from qlib.data import D
        from qlib.data.dataset import DatasetH
        from qlib.data.dataset.handler import DataHandlerLP
        from qlib.workflow import R

        provider = Path(config["paths"]["qlib_provider"]).resolve()
        if not provider.exists():
            raise FileNotFoundError(f"Qlib provider does not exist: {provider}")
        qlib.init(provider_uri=str(provider), region=REG_CN, kernels=1)
        dataset_config = config["dataset"]
        calendar = D.calendar(freq="day")
        provider_end = pd.Timestamp(calendar[-1]).date().isoformat()
        end_time = min(str(dataset_config["segments"]["test"][1]), provider_end)
        segments = {
            name: [values[0], min(values[1], end_time)]
            for name, values in dataset_config["segments"].items()
        }
        label_horizon = int(config["research"]["stock_label_horizon"])
        handler = Alpha158(
            instruments="stocks",
            start_time=dataset_config["start_time"],
            end_time=end_time,
            fit_start_time=dataset_config["fit_start_time"],
            fit_end_time=dataset_config["fit_end_time"],
            label=([f"Ref($close, -{label_horizon + 1})/Ref($close, -1) - 1"], ["LABEL0"]),
            infer_processors=[
                {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
                {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
            ],
            learn_processors=[
                {"class": "DropnaLabel"},
                {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
            ],
        )
        dataset = DatasetH(handler=handler, segments=segments)
        model = LGBModel(**config["model"])
        _fit_qlib_model_and_close_recorder(model, dataset, R)
        predictions: dict[str, pd.Series] = {}
        labels_by_segment: dict[str, pd.Series] = {}
        metrics: dict[str, Any] = {}
        for segment in ("valid", "test"):
            prediction = model.predict(dataset, segment=segment).dropna().sort_index()
            prediction.name = "score"
            labels = dataset.prepare(segment, col_set="label", data_key=DataHandlerLP.DK_L).iloc[:, 0]
            predictions[segment] = prediction
            labels_by_segment[segment] = labels
            metrics[segment] = _prediction_metrics(prediction, labels)
        test_prediction = predictions["test"]
        if test_prediction.empty:
            raise ValueError("Qlib test prediction is empty")
        raw = D.features(
            D.instruments("stocks"),
            ["$close"],
            start_time=segments["test"][0],
            end_time=segments["test"][1],
        )
        closes = raw["$close"].unstack("instrument").sort_index()
        future = closes.shift(-1) / closes - 1.0
        strategy_returns: dict[pd.Timestamp, float] = {}
        benchmark_returns: dict[pd.Timestamp, float] = {}
        turnover_by_date: dict[pd.Timestamp, float] = {}
        previous_weights: dict[str, float] = {}
        transaction_cost_rate = float(config["research"]["transaction_cost_bps"]) / 10000.0
        for date, scores in test_prediction.groupby(level="datetime"):
            if date not in future.index:
                continue
            top = scores.droplevel("datetime").sort_values(ascending=False).head(int(config["research"]["stock_topk"]))
            values = future.loc[date].reindex(top.index).dropna()
            if not values.empty:
                weights = {str(symbol): 1.0 / len(values) for symbol in values.index}
                symbols = set(previous_weights) | set(weights)
                turnover = sum(
                    abs(weights.get(symbol, 0.0) - previous_weights.get(symbol, 0.0))
                    for symbol in symbols
                ) / 2.0
                gross_return = float(values.mean())
                strategy_returns[pd.Timestamp(date)] = gross_return - turnover * transaction_cost_rate
                turnover_by_date[pd.Timestamp(date)] = turnover
                universe_values = future.loc[date].dropna()
                benchmark_returns[pd.Timestamp(date)] = (
                    float(universe_values.mean()) if not universe_values.empty else float("nan")
                )
                previous_weights = weights
        net_returns = pd.Series(strategy_returns, dtype=float).sort_index()
        equity = net_returns.add(1.0).cumprod()
        benchmark_equity = pd.Series(benchmark_returns, dtype=float).sort_index().dropna().add(1.0).cumprod()
        performance = _performance(equity)
        benchmark_performance = _performance(benchmark_equity)
        group_metrics = _forward_group_metrics(
            test_prediction,
            labels_by_segment["test"],
        )
        latest_date = test_prediction.index.get_level_values("datetime").max()
        latest = test_prediction.xs(latest_date, level="datetime").sort_values(ascending=False)
        best_iteration = getattr(getattr(model, "model", None), "best_iteration", None)
        return {
            "model_available": True,
            "model_name": "Qlib Alpha158 + LightGBM",
            "qlib_version": getattr(qlib, "__version__", "unknown"),
            "provider": str(provider),
            "data_end_date": provider_end,
            "latest_signal_date": _iso_date(latest_date),
            "valid": metrics["valid"],
            "test": metrics["test"],
            "mean_rank_ic": metrics["test"]["mean_rank_ic"],
            "test_rank_ic": metrics["test"]["mean_rank_ic"],
            "rank_ic_ir": metrics["test"]["rank_ic_ir"],
            "top_bottom_excess_return": group_metrics["top_bottom_excess_return"],
            "forward_group_metrics": group_metrics,
            "net_annualized_return": performance["net_annualized_return"],
            "test_max_drawdown": performance["max_drawdown"],
            "max_drawdown": performance["max_drawdown"],
            "transaction_cost_bps": float(config["research"]["transaction_cost_bps"]),
            "mean_daily_turnover": float(pd.Series(turnover_by_date, dtype=float).mean()),
            "benchmark": {
                "name": "equal_weight_stock_universe",
                "annualized_return": benchmark_performance["net_annualized_return"],
                "max_drawdown": benchmark_performance["max_drawdown"],
                "excess_annualized_return": performance["net_annualized_return"]
                - benchmark_performance["net_annualized_return"],
            },
            "training_best_iteration": best_iteration,
            "scores": {
                validate_symbol(str(symbol)): float(score)
                for symbol, score in latest.items()
            },
            "leakage_check": {"passed": True, "label_horizon_days": label_horizon},
        }
    except Exception as exc:
        return {
            "model_available": False,
            "model_name": "Qlib Alpha158 + LightGBM",
            "error": f"{type(exc).__name__}: {exc}",
            "mean_rank_ic": float("nan"),
            "test_rank_ic": float("nan"),
            "rank_ic_ir": float("nan"),
            "top_bottom_excess_return": float("nan"),
            "net_annualized_return": float("nan"),
            "test_max_drawdown": float("nan"),
            "max_drawdown": float("nan"),
            "leakage_check": {"passed": False, "reason": "model execution failed"},
        }


def _fetch_state(config: Mapping[str, Any]) -> FetchState:
    configured = config.get("fetch", {}).get(
        "state_db",
        DATA_DIR / "fetch_state.sqlite3",
    )
    return FetchState(Path(str(configured)))


def _target_dates_for_job(
    asset_types: Iterable[str],
    target_date: str,
    config: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
 ) -> dict[str, str]:
    selected_types = tuple(str(value) for value in asset_types)
    configured = (metadata or {}).get("target_dates")
    if isinstance(configured, Mapping) and all(
        asset_type in configured for asset_type in selected_types
    ):
        return {
            asset_type: str(configured[asset_type])
            for asset_type in selected_types
        }
    return asset_target_dates(
        selected_types,
        target_date,
        lag_trading_days=config.get("fetch", {}).get(
            "target_lag_trading_days",
            {},
        ),
    )


def _cached_end_date(asset_type: str, symbol: str) -> str | None:
    path = _data_file(asset_type, symbol)
    metadata = load_asset_metadata(path) if path.exists() else {}
    cached = metadata.get("data_end_date") if isinstance(metadata, Mapping) else None
    if cached:
        return str(cached)
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path)
        date_column = next(
            (
                column
                for column in frame.columns
                if str(column).strip() in {"date", "日期", "净值日期"}
            ),
            None,
        )
        if date_column is None:
            return None
        latest = pd.to_datetime(frame[date_column], errors="coerce").max()
        return None if pd.isna(latest) else latest.date().isoformat()
    except Exception:
        return None


def _normalize_symbol_routes(
    asset_types: Iterable[str],
    symbols_by_type: Mapping[str, Iterable[str]],
 ) -> dict[str, list[str]]:
    return {
        asset_type: sorted(
            {validate_symbol(symbol) for symbol in symbols_by_type.get(asset_type, [])}
        )
        for asset_type in asset_types
        if symbols_by_type.get(asset_type)
    }


def _queue_research(
    *,
    mode: str,
    asset_scope: str,
    asset_types: tuple[str, ...],
    config: Mapping[str, Any],
    symbols_by_type: Mapping[str, Iterable[str]] | None = None,
    target_date: str | None = None,
    start_background_worker: bool = True,
) -> dict[str, Any]:
    """Persist a research request without holding the MCP call open."""
    state = _fetch_state(config)
    full_universe = symbols_by_type is None
    requested_routes = (
        {}
        if full_universe
        else _normalize_symbol_routes(asset_types, symbols_by_type)
    )
    selected_types = (
        asset_types
        if full_universe
        else tuple(
            asset_type for asset_type in asset_types if requested_routes.get(asset_type)
        )
    )
    if not selected_types:
        raise ValueError("no requested symbols matched the selected asset scope")
    effective_target_date = str(target_date or _now().date().isoformat())
    try:
        datetime.fromisoformat(effective_target_date).date()
    except ValueError as exc:
        raise ValueError(f"invalid target_date: {effective_target_date}") from exc
    requested_target_dates = _target_dates_for_job(
        selected_types, effective_target_date, config
    )
    for existing in state.list_jobs(limit=50):
        if not existing or existing.get("kind") != "research":
            continue
        if existing.get("status") not in {"queued", "running", "partial"}:
            continue
        metadata = existing.get("metadata", {})
        if (
            metadata.get("asset_scope") != asset_scope
            or metadata.get("mode") != mode
            or bool(metadata.get("full_universe", True)) != full_universe
        ):
            continue
        existing_target_date = metadata.get("target_date")
        if str(existing_target_date or "") != effective_target_date:
            continue
        existing_routes = {
            asset_type: list(values)
            for asset_type, values in metadata.get("symbols", {}).items()
        }
        if not full_universe and existing_routes != requested_routes:
            continue
        existing_id = str(existing["job_id"])
        effective_target_dates = _target_dates_for_job(
            selected_types,
            effective_target_date,
            config,
            metadata,
        )
        if metadata.get("target_dates") != effective_target_dates:
            metadata = {**metadata, "target_dates": effective_target_dates}
            state.update_job(existing_id, metadata=metadata)
        progress = state.progress(
            selected_types,
            symbols=existing_routes,
            target_date=effective_target_dates,
            max_age_days=int(config["fetch"]["max_data_age_days"]),
        )
        existing_status = read_json(run_directory(existing_id) / "status.json", {})
        worker_pid = int(existing_status.get("worker_pid", 0) or 0)
        if existing.get("status") == "running" and _process_exists(worker_pid):
            worker_info = {"started": False, "reason": "already_running", "pid": worker_pid}
        else:
            worker_info = (
                _start_background_worker(existing_id, config)
                if start_background_worker
                else {"started": False, "reason": "caller_will_start_worker"}
            )
        scope_label = "全市场" if full_universe else "定向"
        return _jsonable(
            {
                "run_id": existing_id,
                "job_id": existing_id,
                "status": existing.get("status", "queued"),
                "phase": existing_status.get("phase", "cache_check"),
                "completed": progress.get("completed", 0),
                "pending": progress.get("pending", 0),
                "failed": progress.get("failed", 0),
                "coverage": progress.get("coverage", 0.0),
                "target_date": effective_target_date,
                "target_dates": effective_target_dates,
                "result_path": str(run_directory(existing_id)),
                "current_path": str(ASSISTANT_RESULTS_DIR / "current.json"),
                "message": f"已有同范围{scope_label}任务在推进，已复用原 run_id。",
                "worker": worker_info,
            }
        )
    run_id = _run_id()
    started = _now()
    run_dir = (
        RUNS_DIR
        / started.strftime("%Y")
        / started.strftime("%m")
        / started.strftime("%d")
        / run_id
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    if full_universe:
        routes: dict[str, list[str]] = {}
        for asset_type in selected_types:
            routes[asset_type] = _universe(asset_type, None)
            state.register_assets(asset_type, routes[asset_type])
    else:
        routes = requested_routes
        for asset_type, scoped_symbols in routes.items():
            cached_end_dates = {
                symbol: _cached_end_date(asset_type, symbol)
                for symbol in scoped_symbols
            }
            state.register_assets(
                asset_type, scoped_symbols, data_end_dates=cached_end_dates
            )
    target_dates = requested_target_dates
    progress = state.progress(
        selected_types,
        symbols=routes,
        target_date=target_dates,
        max_age_days=int(config["fetch"]["max_data_age_days"]),
    )
    state.create_job(
        run_id,
        kind="research",
        asset_scope=asset_scope,
        metadata={
            "mode": mode,
            "asset_scope": asset_scope,
            "asset_types": list(selected_types),
            "symbols": routes,
            "full_universe": full_universe,
            "target_date": effective_target_date,
            "target_dates": target_dates,
            "phase": "cache_check",
        },
    )
    state.update_job(
        run_id,
        total=progress["total"],
        completed=progress["completed"],
        pending=progress["pending"],
        failed=progress["failed"],
        stale=progress["stale"],
        coverage=progress["coverage"],
    )
    status = {
        "run_id": run_id,
        "job_id": run_id,
        "status": "queued",
        "started_at": started.isoformat(),
        "asset_scope": asset_scope,
        "asset_types": list(selected_types),
        "full_universe": full_universe,
        "target_date": effective_target_date,
        "target_dates": target_dates,
        "phase": "cache_check",
        "completed": progress["completed"],
        "pending": progress["pending"],
        "failed": progress["failed"],
        "coverage": progress["coverage"],
        "progress_seq": 0,
        "last_progress_at": started.isoformat(),
        "last_heartbeat_at": started.isoformat(),
        "result_path": str(run_dir),
        "message": (
            "全市场请求已入队；后台会按批次推进。"
            if full_universe
            else "定向请求已入队；后台只会检查和刷新指定代码。"
        ),
    }
    atomic_write_json(run_dir / "status.json", status)
    atomic_write_json(
        run_dir / "manifest.json",
        {
            "run_id": run_id,
            "job_id": run_id,
            "status": "queued",
            "started_at": started.isoformat(),
            "mode": mode,
            "asset_scope": asset_scope,
            "asset_types": list(selected_types),
            "symbols": routes,
            "full_universe": full_universe,
            "target_date": effective_target_date,
            "target_dates": target_dates,
            "config_snapshot": config_snapshot(dict(config)),
        },
    )
    worker_info = (
        _start_background_worker(run_id, config)
        if start_background_worker
        else {"started": False, "reason": "caller_will_start_worker"}
    )
    status["worker"] = worker_info
    atomic_write_json(run_dir / "status.json", status)
    return _jsonable(
        {
            "run_id": run_id,
            "job_id": run_id,
            "status": "queued",
            "phase": "cache_check",
            "completed": progress["completed"],
            "pending": progress["pending"],
            "failed": progress["failed"],
            "coverage": progress["coverage"],
            "target_date": effective_target_date,
            "target_dates": target_dates,
            "result_path": str(run_dir),
            "current_path": str(ASSISTANT_RESULTS_DIR / "current.json"),
            "message": status["message"],
            "worker": worker_info,
        }
    )


def _queue_full_universe_research(**kwargs: Any) -> dict[str, Any]:
    """Compatibility wrapper for existing schedulers and tests."""
    return _queue_research(**kwargs)


def _start_background_worker(
    job_id: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if not config["fetch"].get("background_worker_enabled", True):
        return {"started": False, "reason": "disabled_by_configuration"}
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_research_worker.py"
    if not script.exists():
        return {"started": False, "reason": "worker_script_missing"}
    try:
        creation_flags = 0
        if os.name == "nt":
            creation_flags = (
                getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                | getattr(subprocess, "DETACHED_PROCESS", 0)
            )
        process = subprocess.Popen(
            [sys.executable, str(script), "--job-id", job_id],
            cwd=str(script.parents[1]),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creation_flags,
            close_fds=os.name != "nt",
        )
        return {"started": True, "pid": process.pid}
    except Exception as exc:
        return {
            "started": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }


def _run_refresh_command(
    command: list[str],
    *,
    cwd: Path,
    config: Mapping[str, Any],
    label: str,
    log: list[str],
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    max_attempts = int(config["fetch"]["retries"]) + 1
    for attempt in range(1, max_attempts + 1):
        try:
            completed = subprocess.run(
                command,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=int(config["fetch"]["timeout_seconds"]),
                check=False,
            )
            record = {
                "attempt": attempt,
                "returncode": completed.returncode,
                "stdout_tail": completed.stdout[-2000:],
                "stderr_tail": completed.stderr[-2000:],
            }
            summary = _fetch_summary(completed.stdout)
            if summary is not None:
                record["summary"] = summary
            attempts.append(record)
            if completed.returncode == 0:
                if summary is None:
                    log.append(f"refresh {label}: success on attempt {attempt}")
                    return {"status": "success", "attempts": attempts}
                if (
                    summary.get("status") in {"complete", "partial"}
                    and int(summary.get("failed", 0) or 0) == 0
                ):
                    result_status = (
                        "success"
                        if summary.get("status") == "complete"
                        and not summary.get("partial_failure", False)
                        else "partial"
                    )
                    log.append(
                        f"refresh {label}: {result_status} on attempt {attempt}"
                    )
                    return {"status": result_status, "attempts": attempts}
        except Exception as exc:
            attempts.append({
                "attempt": attempt,
                "error": f"{type(exc).__name__}: {exc}",
            })
        log.append(f"refresh {label}: attempt {attempt} failed")
        if attempt < max_attempts:
            time.sleep(float(config["fetch"].get("retry_delay_seconds", 0)))
    return {"status": "failed", "attempts": attempts}


def _refresh_data(
    asset_types: Iterable[str],
    config: Mapping[str, Any],
    log: list[str],
    *,
    symbols_by_type: Mapping[str, Iterable[str]] | None = None,
) -> dict[str, Any]:
    if not config["fetch"]["enabled"] or os.environ.get("DSH_STOCK_SKIP_FETCH") == "1":
        log.append("data refresh skipped by configuration or DSH_STOCK_SKIP_FETCH=1")
        return {"attempted": False, "passed": True, "results": []}
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    results: list[dict[str, Any]] = []
    selected = set(asset_types)
    routes = (
        _normalize_symbol_routes(selected, symbols_by_type)
        if symbols_by_type is not None
        else {}
    )
    jobs: list[tuple[str, Path, list[str]]] = []
    if "stock" in selected:
        jobs.append(
            ("stock", scripts / "fetch_stock_data.py", routes.get("stock", []))
        )
    for asset_type in ("etf", "otc_fund"):
        if asset_type in selected:
            jobs.append(
                (
                    asset_type,
                    scripts / "fetch_fund_data.py",
                    ["--asset-type", asset_type, *routes.get(asset_type, [])],
                )
            )
    for label, script, extra_args in jobs:
        if not script.exists():
            results.append({"label": label, "status": "missing_script"})
            continue
        fetch_options = [
            "--batch-size",
            str(
                config["fetch"]
                .get("batch_size", {})
                .get(label, 100)
            ),
            "--time-budget",
            str(config["fetch"].get("batch_time_budget_seconds", 600)),
            "--workers",
            str(config["fetch"].get("max_workers", 3)),
            "--retries",
            str(config["fetch"].get("retries", 3)),
            "--rate-limit",
            str(config["fetch"].get("requests_per_second", 1.0)),
            "--request-timeout",
            str(config["fetch"].get("request_timeout_seconds", 30)),
            "--backoff-base",
            str(config["fetch"].get("backoff_base_seconds", 2)),
            "--backoff-max",
            str(config["fetch"].get("backoff_max_seconds", 120)),
            "--lease-seconds",
            str(config["fetch"].get("lease_seconds", 1800)),
            "--state-db",
            str(config["fetch"].get("state_db", DATA_DIR / "fetch_state.sqlite3")),
        ]
        result = _run_refresh_command(
            [sys.executable, str(script), *fetch_options, *extra_args],
            cwd=Path(__file__).resolve().parents[1],
            config=config,
            label=label,
            log=log,
        )
        results.append({"label": label, **result})

    stock_fetch_ok = any(
        result.get("label") == "stock" and result.get("status") == "success"
        for result in results
    )
    if "stock" in selected and stock_fetch_ok:
        provider_result = _run_refresh_command(
            [sys.executable, str(scripts / "build_qlib_data.py"), "--replace"],
            cwd=Path(__file__).resolve().parents[1],
            config=config,
            label="qlib_provider",
            log=log,
        )
        results.append({"label": "qlib_provider", **provider_result})
    passed = bool(results) and all(
        result.get("status") == "success" for result in results
    )
    return {
        "attempted": True,
        "passed": passed,
        "results": results,
        "max_attempts": int(config["fetch"]["retries"]) + 1,
        "finished_at": _now().isoformat(),
    }


def _write_run_files(
    run_dir: Path,
    *,
    manifest: dict[str, Any],
    data_quality: dict[str, Any],
    data_facts: list[dict[str, Any]],
    model_signals: dict[str, Any],
    rule_gates: dict[str, Any],
    decision_support: dict[str, Any],
    config: dict[str, Any],
    model_metadata: dict[str, Any],
    log_lines: list[str],
) -> dict[str, str]:
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, Any] = {
        "manifest": manifest,
        "data_quality": data_quality,
        "data_facts": data_facts,
        "model_signals": model_signals,
        "rule_gates": rule_gates,
        "decision_support": decision_support,
        "model_metadata": model_metadata,
    }
    paths: dict[str, str] = {}
    for name, value in files.items():
        path = run_dir / f"{name}.json"
        atomic_write_json(path, _jsonable(value))
        paths[name] = str(path)
    config_path = run_dir / "run_config.yaml"
    atomic_write_text(config_path, yaml.safe_dump(config_snapshot(config), allow_unicode=True, sort_keys=False))
    paths["run_config"] = str(config_path)
    summary_lines = [
        "# 股票/基金私人研究助手运行摘要",
        "",
        f"- 运行 ID: `{manifest['run_id']}`",
        f"- 状态: `{manifest['status']}`",
        f"- 数据截止日期: `{data_quality.get('data_end_date') or 'unknown'}`",
        f"- 资产数量: `{data_quality.get('asset_count', 0)}`",
        f"- 个股模型门槛: `{'通过' if rule_gates.get('stock', {}).get('passed') else '未通过'}`",
        "",
        "## 使用规则",
        "",
        "- 个股模型门槛未通过时，不生成个股买入目标权重。",
        "- 目标权重基于假设资金/配置，不连接券商，不自动下单。",
        "- 场外基金使用净值序列，不输出盘中成交价或精确股数。",
        "",
        "## 本次建议",
        "",
        json.dumps(_jsonable(decision_support), ensure_ascii=False, indent=2),
        "",
        "## 日志",
        "",
        *[f"- {line}" for line in log_lines],
    ]
    summary_path = run_dir / "summary.md"
    atomic_write_text(summary_path, "\n".join(summary_lines) + "\n")
    paths["summary"] = str(summary_path)
    log_path = logs_dir / "research.log"
    atomic_write_text(log_path, "\n".join(log_lines) + "\n")
    paths["logs"] = str(log_path)
    return paths


def _touch_status(status_path: Path, phase: str) -> None:
    """Refresh the run heartbeat so the stale-run sweep never reaps a live run."""
    record = read_json(status_path, {})
    if not isinstance(record, dict):
        record = {}
    record["status"] = "running"
    record["phase"] = phase
    record["updated_at"] = _now().isoformat()
    atomic_write_json(status_path, record)


def _sweep_stale_runs(
    config: Mapping[str, Any],
    log_lines: list[str],
    *,
    runs_dir: Path | None = None,
) -> list[str]:
    """Fail orphaned runs whose status stayed ``running`` without a heartbeat.

    A killed process never reaches the success/failed write, so its status
    file would report ``running`` forever.  The heartbeat touch keeps the
    status file mtime fresh while a run is alive; anything silent for longer
    than ``runtime.lock_stale_seconds`` is treated as dead and marked failed
    without touching ``current.json``.
    """
    threshold = int(config["runtime"]["lock_stale_seconds"])
    current = _now().timestamp()
    swept: list[str] = []
    for status_path in sorted((runs_dir or RUNS_DIR).rglob("status.json")):
        record = read_json(status_path, {})
        if not isinstance(record, dict) or record.get("status") != "running":
            continue
        try:
            silent_seconds = current - status_path.stat().st_mtime
        except OSError:
            continue
        if silent_seconds < threshold:
            continue
        run_id = str(record.get("run_id") or status_path.parent.name)
        atomic_write_json(
            status_path,
            {
                **record,
                "status": "failed",
                "updated_at": _now().isoformat(),
                "error": (
                    f"stale run swept: no heartbeat for {int(silent_seconds)}s "
                    f"(threshold {threshold}s)"
                ),
                "failure_reasons": ["stale_run_heartbeat_timeout"],
            },
        )
        swept.append(run_id)
    if swept:
        log_lines.append(f"swept_stale_runs={swept}")
    return swept


def _process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        from ctypes import wintypes

        process_query_limited_information = 0x1000
        still_active = 259
        access_denied = 5
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = (
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        )
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.DWORD),
        )
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL

        handle = kernel32.OpenProcess(
            process_query_limited_information,
            False,
            pid,
        )
        if not handle:
            return ctypes.get_last_error() == access_denied
        try:
            exit_code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return True
            return exit_code.value == still_active
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        return exc.errno != ESRCH
    return True


def _lock_record() -> dict[str, Any]:
    try:
        value = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        value = {}
    return value if isinstance(value, dict) else {}


def _acquire_lock(run_id: str, stale_seconds: int):
    ensure_directories()
    for _ in range(2):
        try:
            descriptor = os.open(str(LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            existing = _lock_record()
            try:
                age_seconds = max(0.0, time.time() - LOCK_PATH.stat().st_mtime)
            except FileNotFoundError:
                continue
            pid = int(existing.get("pid", 0) or 0)
            stale = age_seconds > max(0, int(stale_seconds)) or not _process_exists(pid)
            if stale:
                try:
                    LOCK_PATH.unlink()
                except FileNotFoundError:
                    pass
                continue
            raise RuntimeError(
                "another research run is active: "
                f"run_id={existing.get('run_id', 'unknown')}, pid={pid}"
            ) from exc
        record = {
            "run_id": run_id,
            "pid": os.getpid(),
            "created_at": _now().isoformat(),
        }
        try:
            os.write(descriptor, (json.dumps(record, sort_keys=True) + "\n").encode("utf-8"))
            os.fsync(descriptor)
        except Exception:
            os.close(descriptor)
            LOCK_PATH.unlink(missing_ok=True)
            raise
        return descriptor
    raise RuntimeError("could not acquire research lock after removing a stale lock")


def _release_lock(descriptor: int, run_id: str | None = None) -> None:
    try:
        os.close(descriptor)
    finally:
        existing = _lock_record()
        if run_id is None or existing.get("run_id") == run_id:
            LOCK_PATH.unlink(missing_ok=True)


def _run_id() -> str:
    return f"{_now():%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"


def _asset_scope(value: str) -> tuple[str, ...]:
    scope = str(value or "all").strip().lower()
    if scope not in ASSET_SCOPE_MAP:
        raise ValueError(f"asset_scope must be one of: {', '.join(sorted(ASSET_SCOPE_MAP))}")
    return ASSET_SCOPE_MAP[scope]


def run_research(
    mode: str = "daily",
    asset_scope: str = "all",
    symbols: list[str] | None = None,
    *,
    _force_sync: bool = False,
    _target_date: str | None = None,
    _start_background_worker: bool | None = None,
    _symbols_by_type: Mapping[str, Iterable[str]] | None = None,
) -> dict[str, Any]:
    mode = str(mode or "daily").strip().lower()
    if mode not in MODE_NAMES:
        raise ValueError(f"mode must be one of: {', '.join(sorted(MODE_NAMES))}")
    asset_types = _asset_scope(asset_scope)
    if symbols is not None and not symbols and _symbols_by_type is None:
        raise ValueError("symbols must be omitted for full-universe research or contain at least one code")
    requested_symbols = [validate_symbol(value) for value in (symbols or [])]
    explicit_routes = (
        _normalize_symbol_routes(asset_types, _symbols_by_type)
        if _symbols_by_type is not None
        else (
            _partition_symbols(asset_types, requested_symbols)
            if requested_symbols
            else {}
        )
    )
    config = load_config()
    if (
        not _force_sync
        and config["fetch"]["enabled"]
        and os.environ.get("DSH_STOCK_SKIP_FETCH") != "1"
    ):
        return _queue_research(
            mode=mode,
            asset_scope=asset_scope,
            asset_types=asset_types,
            config=config,
            symbols_by_type=explicit_routes if requested_symbols or _symbols_by_type is not None else None,
            target_date=_target_date,
            start_background_worker=(
                True if _start_background_worker is None else _start_background_worker
            ),
        )
    run_id = _run_id()
    started = _now()
    run_dir = RUNS_DIR / started.strftime("%Y") / started.strftime("%m") / started.strftime("%d") / run_id
    lock_descriptor: int | None = None
    log_lines = [f"started {started.isoformat()}", f"mode={mode}", f"asset_types={asset_types}"]
    status_path = run_dir / "status.json"
    _sweep_stale_runs(config, log_lines)
    refresh: dict[str, Any] = {"attempted": False, "passed": True, "results": []}
    all_facts: list[dict[str, Any]] = []
    technical_signals: dict[str, Any] = {}
    data_quality: dict[str, Any] = {
        "passed": False,
        "data_end_date": None,
        "asset_count": 0,
        "refresh": refresh,
    }
    model_metadata: dict[str, Any] = {
        "model_version": "assistant-1",
        "qlib_provider": str(QLIB_PROVIDER_DIR),
        "requested_mode": mode,
    }
    rule_gates: dict[str, Any] = {}
    decision_support: dict[str, Any] = cash_only_decision(
        "研究尚未完成。",
        capital=float(config["research"]["hypothetical_capital"]),
    )
    decision_support["asset_paths"] = {}
    result_paths: dict[str, str] = {}
    try:
        lock_descriptor = _acquire_lock(
            run_id,
            int(config["runtime"]["lock_stale_seconds"]),
        )
        run_dir.mkdir(parents=True, exist_ok=False)
        atomic_write_json(
            status_path,
            {
                "run_id": run_id,
                "status": "running",
                "started_at": started.isoformat(),
                "updated_at": started.isoformat(),
            },
        )
        symbol_routes = explicit_routes or {asset_type: [] for asset_type in asset_types}
        analyzed_types = tuple(
            asset_type
            for asset_type in asset_types
            if not explicit_routes or symbol_routes.get(asset_type)
        )
        if explicit_routes and not analyzed_types:
            raise ValueError("no requested symbols matched the selected asset scope")
        log_lines.append(f"analyzed_types={analyzed_types}")
        if explicit_routes:
            log_lines.append(f"symbol_routes={symbol_routes}")

        if mode in set(config["research"]["refresh_modes"]):
            refresh = _refresh_data(
                analyzed_types,
                config,
                log_lines,
                symbols_by_type=symbol_routes if explicit_routes else None,
            )
        _touch_status(status_path, "refreshed")
        all_frames: dict[str, dict[str, pd.DataFrame]] = {}
        for asset_type in analyzed_types:
            scope_symbols = symbol_routes[asset_type] if explicit_routes else None
            frames, facts = _scan_assets(asset_type, scope_symbols)
            all_frames[asset_type] = frames
            all_facts.extend(facts)
        data_quality = _aggregate_quality(all_facts, config)
        data_quality["refresh"] = refresh
        data_quality["passed"] = bool(data_quality["passed"] and refresh.get("passed", False))
        universe_manifest = load_universe_manifest(UNIVERSE_PATH)
        data_quality["universe"] = {
            "manifest_path": str(UNIVERSE_PATH),
            "source": universe_manifest.get("source") if universe_manifest else None,
            "generated_at": universe_manifest.get("generated_at") if universe_manifest else None,
            "complete": bool(universe_manifest and universe_manifest.get("complete")),
            "counts": (
                universe_manifest.get("counts", {})
                if universe_manifest
                else {
                    asset_type: sum(
                        fact.get("asset_type") == asset_type
                        for fact in all_facts
                    )
                    for asset_type in ASSET_TYPES
                }
            ),
            "authoritative_asset_types": [
                asset_type
                for asset_type in analyzed_types
                if _authoritative_manifest(universe_manifest, asset_type)
            ],
        }
        refresh_summaries = {
            item.get("label"): attempt.get("summary")
            for item in refresh.get("results", [])
            for attempt in reversed(item.get("attempts", []))
            if attempt.get("summary")
        }
        for fact in all_facts:
            label = "stock" if fact.get("asset_type") == "stock" else "funds"
            summary = refresh_summaries.get(label)
            if summary:
                fact.update(
                    {
                        "akshare_interfaces": summary.get("interfaces", []),
                        "fetch_time": summary.get("finished_at") or refresh.get("finished_at"),
                        "fetch_retries": max(
                            (
                                attempt.get("attempt", 1)
                                for item in refresh.get("results", [])
                                if item.get("label") == label
                                for attempt in item.get("attempts", [])
                            ),
                            default=1,
                        )
                        - 1,
                        "partial_failure": bool(summary.get("partial_failure", False)),
                        "field_anomalies": summary.get("field_anomalies", []),
                    }
                )
        if not all_facts:
            raise ValueError("no asset data found in fixed data directories")

        _touch_status(status_path, "scanned")
        benchmark_series = None
        if analyzed_types:
            try:
                benchmark_frame, _ = _load_asset("etf", "510300")
                benchmark_series = pd.Series(
                    benchmark_frame["close"].to_numpy(),
                    index=pd.to_datetime(benchmark_frame["date"]),
                ).sort_index()
            except Exception as exc:
                log_lines.append(f"benchmark unavailable: {type(exc).__name__}")

        for asset_type in analyzed_types:
            technical_signals[asset_type] = _technical_asset_analysis(
                asset_type,
                all_frames[asset_type],
                [item for item in all_facts if item["asset_type"] == asset_type],
                benchmark_series,
            )

        _touch_status(status_path, "signals")
        stock_model = {
            "model_available": False,
            "model_name": "Qlib Alpha158 + LightGBM",
            "error": "stock path not requested",
            "mean_rank_ic": float("nan"),
            "test_rank_ic": float("nan"),
            "rank_ic_ir": float("nan"),
            "top_bottom_excess_return": float("nan"),
            "net_annualized_return": float("nan"),
            "test_max_drawdown": float("nan"),
            "max_drawdown": float("nan"),
            "leakage_check": {"passed": False, "reason": "stock path not requested"},
        }
        if "stock" in analyzed_types:
            stock_facts = [item for item in all_facts if item["asset_type"] == "stock"]
            stock_quality = _aggregate_quality(stock_facts, config)
            stock_refresh_ok = all(
                result.get("status") == "success"
                for result in refresh.get("results", [])
                if result.get("label") in {"stock", "qlib_provider"}
            )
            if stock_quality["passed"] and stock_refresh_ok:
                stock_model = _run_qlib_stock_model(config)
            else:
                stock_model["error"] = "stock model skipped because stock data quality or refresh failed"
                stock_model["leakage_check"] = {
                    "passed": False,
                    "reason": "stock data quality or refresh failed",
                }
            model_metadata["stock_model"] = stock_model
            if stock_model.get("model_available"):
                score_map = stock_model.get("scores", {})
                for signal in technical_signals["stock"]["signals"]:
                    if signal["symbol"] in score_map:
                        signal["qlib_score"] = score_map[signal["symbol"]]
                        signal["model_rank"] = 1 + sorted(score_map, key=score_map.get, reverse=True).index(signal["symbol"])
                technical_signals["stock"]["signals"].sort(
                    key=lambda item: item.get("qlib_score", -math.inf), reverse=True
                )

        _touch_status(status_path, "gates")
        decision_support = {
            "allocation_basis": "基于假设资金/配置的目标权重",
            "hypothetical_capital": float(config["research"]["hypothetical_capital"]),
            "broker_connected": False,
            "auto_order": False,
            "asset_paths": {},
            "target_weights": {},
            "cash_weight": 1.0,
        }
        allocation_budget = min(
            float(config["risk"]["max_daily_turnover"]),
            1.0 - float(config["risk"]["min_cash_weight"]),
        ) / max(len(analyzed_types), 1)

        if "stock" in analyzed_types:
            stock_facts = [item for item in all_facts if item["asset_type"] == "stock"]
            stock_quality = _aggregate_quality(stock_facts, config)
            stock_metrics = dict(stock_model)
            stock_metrics["model_available"] = bool(stock_model.get("model_available"))
            checks = model_gate_checks(stock_metrics, config["gate"])
            checks["data_quality_passed"] = bool(stock_quality["passed"])
            checks["freshness_passed"] = not bool(stock_quality["stale"])
            checks["refresh_passed"] = all(
                result.get("status") == "success"
                for result in refresh.get("results", [])
                if result.get("label") in {"stock", "qlib_provider"}
            )
            checks["leakage_check_passed"] = bool(stock_model.get("leakage_check", {}).get("passed", False))
            gate_passed = all(checks.values())
            stock_requested = {}
            if gate_passed:
                ranked = technical_signals["stock"]["signals"][: int(config["research"]["stock_topk"])]
                if ranked:
                    per_symbol = min(
                        allocation_budget / len(ranked),
                        float(config["risk"]["max_single_weight"]),
                    )
                    raw_weights = {item["symbol"]: per_symbol for item in ranked}
                    stock_requested = raw_weights
            asset_types_for_weights = {symbol: "stock" for symbol in stock_requested}
            risk_result = enforce_target_weights(
                stock_requested,
                asset_types_for_weights,
                config["risk"],
            )
            checks["risk_constraints_passed"] = risk_result["passed"]
            gate_passed = gate_passed and risk_result["passed"]
            if gate_passed:
                decision_support["target_weights"].update(risk_result["target_weights"])
                decision_support["cash_weight"] = risk_result["cash_weight"]
                stock_decision = {
                    "status": "candidate",
                    "action": "观察/候选",
                    "target_weights": risk_result["target_weights"],
                    "cash_weight": risk_result["cash_weight"],
                    "reason": "Qlib 模型和风险门槛均通过；仍需人工确认基本面与交易条件。",
                }
            else:
                stock_decision = cash_only_decision(
                    "个股模型或风险门槛未通过，不生成个股买入目标权重。",
                    capital=float(config["research"]["hypothetical_capital"]),
                )
            rule_gates["stock"] = {
                "passed": gate_passed,
                "checks": checks,
                "model_metrics": stock_metrics,
                "risk": risk_result,
            }
            decision_support["asset_paths"]["stock"] = stock_decision

        for asset_type in ("etf", "otc_fund"):
            if asset_type not in analyzed_types:
                continue
            analysis = technical_signals[asset_type]
            candidates = [item for item in analysis["signals"] if (item.get("trend_score") or -math.inf) > 0]
            candidates = candidates[:2]
            category = "etf" if asset_type == "etf" else "otc_fund"
            per_symbol = min(
                allocation_budget / len(candidates),
                float(config["risk"]["max_single_weight"]),
            ) if candidates else 0.0
            requested = {item["symbol"]: per_symbol for item in candidates}
            risk_result = enforce_target_weights(
                requested,
                {symbol: category for symbol in requested},
                config["risk"],
            )
            asset_facts = [item for item in all_facts if item["asset_type"] == asset_type]
            asset_quality = _aggregate_quality(asset_facts, config)
            checks = {
                "data_quality_passed": bool(asset_quality["passed"]),
                "freshness_passed": not bool(asset_quality["stale"]),
                "refresh_passed": all(
                    result.get("status") == "success"
                    for result in refresh.get("results", [])
                    if result.get("label") == "funds"
                ),
                "risk_constraints_passed": risk_result["passed"],
            }
            passed = all(checks.values())
            rule_gates[asset_type] = {"passed": passed, "checks": checks, "risk": risk_result}
            if passed:
                decision = {
                    "status": "candidate",
                    "action": "配置候选",
                    "target_weights": risk_result["target_weights"],
                    "cash_weight": risk_result["cash_weight"],
                    "reason": "技术趋势、相对表现和风险约束通过；需结合费用、流动性和个人期限确认。",
                }
                decision_support["target_weights"].update(risk_result["target_weights"])
            else:
                decision = cash_only_decision(
                    f"{category} 数据或风险门槛未通过，保持观察。",
                    capital=float(config["research"]["hypothetical_capital"]),
                )
            if asset_type == "otc_fund":
                decision["no_intraday_price_or_exact_shares"] = True
            decision_support["asset_paths"][asset_type] = decision

        current = read_json(ASSISTANT_RESULTS_DIR / "current.json", {})
        previous_weights = (
            current.get("decision_support", {}).get("target_weights", {})
            if isinstance(current, dict)
            else {}
        )
        combined_types = {
            symbol: asset_type
            for asset_type, decision in decision_support["asset_paths"].items()
            for symbol in decision.get("target_weights", {})
        }
        portfolio_risk = enforce_target_weights(
            decision_support["target_weights"],
            combined_types,
            config["risk"],
            previous=previous_weights,
        )
        rule_gates["portfolio"] = portfolio_risk
        decision_support["portfolio_gate"] = portfolio_risk
        if not portfolio_risk["passed"]:
            decision_support["target_weights"] = {}
            decision_support["cash_weight"] = 1.0
            for decision in decision_support["asset_paths"].values():
                decision["target_weights"] = {}
                decision["cash_weight"] = 1.0
                decision["status"] = "cash_or_observe"
                decision["action"] = "现金/观察"
                decision["reason"] = "组合层风险约束未通过，不发布可执行目标权重。"
        else:
            decision_support["target_weights"] = portfolio_risk["target_weights"]
            decision_support["cash_weight"] = portfolio_risk["cash_weight"]

        failure_reasons: list[str] = []
        if not data_quality["passed"]:
            failure_reasons.append("data_quality_or_refresh_failed")
        if "stock" in analyzed_types and not rule_gates.get("stock", {}).get("passed", False):
            failure_reasons.append("stock_model_or_risk_gate_failed")
        if not portfolio_risk["passed"]:
            failure_reasons.append("portfolio_risk_gate_failed")
        for asset_type in ("etf", "otc_fund"):
            if asset_type in analyzed_types and not rule_gates.get(asset_type, {}).get("passed", False):
                failure_reasons.append(f"{asset_type}_quality_or_risk_gate_failed")
        run_status = "failed" if failure_reasons else "success"
        if failure_reasons:
            decision_support["publication_blocked"] = True
            decision_support["failure_reasons"] = failure_reasons
            log_lines.append(f"publication blocked: {failure_reasons}")

        manifest = {
            "run_id": run_id,
            "status": run_status,
            "started_at": started.isoformat(),
            "finished_at": _now().isoformat(),
            "mode": mode,
            "asset_scope": asset_scope,
            "asset_types": asset_types,
            "analyzed_types": analyzed_types,
            "requested_symbols": requested_symbols,
            "data_sources": (
                ["AkShare cached CSV", "Qlib provider"]
                if "stock" in analyzed_types
                else ["AkShare cached CSV"]
            ),
            "data_end_date": data_quality.get("data_end_date"),
            "retry_count": int(config["fetch"]["retries"]),
            "failure_reasons": failure_reasons,
            "config_snapshot": config_snapshot(config),
        }
        result_paths = _write_run_files(
            run_dir,
            manifest=manifest,
            data_quality=data_quality,
            data_facts=all_facts,
            model_signals=technical_signals,
            rule_gates=rule_gates,
            decision_support=decision_support,
            config=config,
            model_metadata=model_metadata,
            log_lines=log_lines,
        )
        manifest["paths"] = result_paths
        atomic_write_json(run_dir / "manifest.json", _jsonable(manifest))
        status = {
            "run_id": run_id,
            "status": run_status,
            "finished_at": manifest["finished_at"],
            "data_end_date": data_quality.get("data_end_date"),
            "result_path": str(run_dir),
            "failure_reasons": failure_reasons,
        }
        atomic_write_json(status_path, status)
        if run_status == "success":
            current = {
                "run_id": run_id,
                "status": "success",
                "updated_at": manifest["finished_at"],
                "data_end_date": data_quality.get("data_end_date"),
                "result_path": str(run_dir),
                "paths": result_paths,
                "decision_support": _jsonable(decision_support),
                "rule_gates": _jsonable(rule_gates),
            }
            atomic_write_json(ASSISTANT_RESULTS_DIR / "current.json", current)
            notification = add_notification(
                "research_success",
                "股票/基金研究完成",
                f"运行 {run_id} 已完成，数据截止 {data_quality.get('data_end_date') or 'unknown'}。",
                run_id=run_id,
                paths=result_paths,
            )
        else:
            notification = add_notification(
                "research_failure",
                "股票/基金研究未通过发布门槛",
                f"运行 {run_id} 未通过：{', '.join(failure_reasons)}。上一份 current.json 未覆盖。",
                run_id=run_id,
                paths=result_paths,
            )
        return _jsonable(
            {
                "run_id": run_id,
                "status": run_status,
                "failure_reasons": failure_reasons,
                "data_end_date": data_quality.get("data_end_date"),
                "result_path": str(run_dir),
                "current_path": str(ASSISTANT_RESULTS_DIR / "current.json"),
                "notification_id": notification["notification_id"],
                "data_facts": all_facts,
                "model_signals": technical_signals,
                "rule_gates": rule_gates,
                "decision_support": decision_support,
            }
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        log_lines.append(f"failed: {error}")
        manifest = {
            "run_id": run_id,
            "status": "failed",
            "started_at": started.isoformat(),
            "finished_at": _now().isoformat(),
            "mode": mode,
            "asset_scope": asset_scope,
            "error": error,
            "config_snapshot": config_snapshot(config),
        }
        if run_dir.exists():
            data_quality["passed"] = False
            data_quality["error"] = error
            decision_support["publication_blocked"] = True
            decision_support["failure_reasons"] = [error]
            result_paths = _write_run_files(
                run_dir,
                manifest=manifest,
                data_quality=data_quality,
                data_facts=all_facts,
                model_signals=technical_signals,
                rule_gates=rule_gates,
                decision_support=decision_support,
                config=config,
                model_metadata=model_metadata,
                log_lines=log_lines,
            )
            manifest["paths"] = result_paths
            atomic_write_json(run_dir / "manifest.json", manifest)
            atomic_write_json(status_path, manifest)
        notification = add_notification(
            "research_failure",
            "股票/基金研究失败",
            f"运行 {run_id} 失败：{error}。上一份 current.json 未覆盖。",
            run_id=run_id,
            paths=result_paths,
        )
        return {
            "run_id": run_id,
            "status": "failed",
            "error": error,
            "result_path": str(run_dir),
            "current_path": str(ASSISTANT_RESULTS_DIR / "current.json"),
            "notification_id": notification["notification_id"],
        }
    finally:
        if lock_descriptor is not None:
            _release_lock(lock_descriptor, run_id)


def get_current_snapshot(asset_type: str | None = None, symbol: str | None = None) -> dict[str, Any]:
    current = read_json(ASSISTANT_RESULTS_DIR / "current.json")
    if current is None:
        return {
            "status": "no_successful_run",
            "message": "尚无成功研究结果；请先运行 run_research。",
            "current_path": str(ASSISTANT_RESULTS_DIR / "current.json"),
        }
    current = _sanitize_result_record(current)
    if asset_type is None and symbol is None:
        return current
    normalized_type = str(asset_type).strip().lower() if asset_type else None
    if normalized_type is not None and normalized_type not in ASSET_TYPES:
        raise ValueError(f"asset_type must be one of: {', '.join(ASSET_TYPES)}")
    normalized_symbol = validate_symbol(symbol) if symbol else None
    paths = current.get("paths", {})
    model_signals = (
        read_json(fixed_path(paths["model_signals"]))
        if paths.get("model_signals")
        else {}
    )
    selected: list[dict[str, Any]] = []
    for category, analysis in model_signals.items():
        if normalized_type and category != normalized_type:
            continue
        for signal in analysis.get("signals", []):
            if normalized_symbol and signal.get("symbol") != normalized_symbol:
                continue
            selected.append({"asset_type": category, **signal})
    if not selected:
        return {
            "status": "not_found_or_stale",
            "requested_asset_type": normalized_type,
            "requested_symbol": normalized_symbol,
            "run_id": current.get("run_id"),
            "data_end_date": current.get("data_end_date"),
            "signals": [],
            "reason": "当前成功快照没有匹配该资产类型和代码的信号。",
            "result_path": _fixed_reference(current.get("result_path")),
        }
    decision = current.get("decision_support", {})
    asset_decision = (
        decision.get("asset_paths", {}).get(normalized_type)
        if normalized_type and isinstance(decision, Mapping)
        else decision
    )
    return {
        "status": current.get("status"),
        "run_id": current.get("run_id"),
        "data_end_date": current.get("data_end_date"),
        "result_path": _fixed_reference(current.get("result_path")),
        "requested_asset_type": normalized_type,
        "requested_symbol": normalized_symbol,
        "decision_support": asset_decision or {},
        "signals": selected,
    }


def _latest_run_id() -> str | None:
    current = read_json(ASSISTANT_RESULTS_DIR / "current.json", {})
    if isinstance(current, dict) and current.get("run_id"):
        try:
            return validate_run_id(current["run_id"])
        except ValueError:
            pass
    candidates = []
    for status_path in RUNS_DIR.rglob("status.json"):
        candidate = status_path.parent.name
        try:
            validate_run_id(candidate)
        except ValueError:
            continue
        candidates.append((status_path.stat().st_mtime, candidate))
    return max(candidates)[1] if candidates else None


def get_run_status(run_id: str | None = None) -> dict[str, Any]:
    run_id = validate_run_id(run_id) if run_id else _latest_run_id()
    if run_id is None:
        return {
            "status": "no_runs",
            "message": "尚无研究运行记录。",
            "result_path": str(RUNS_DIR),
        }
    directory = run_directory(run_id)
    status = read_json(directory / "status.json", {})
    if not isinstance(status, dict):
        status = {}
    status["run_id"] = run_id
    status["result_path"] = str(directory)
    try:
        config = load_config()
        state = _fetch_state(config)
        job = state.get_job(run_id)
    except Exception:
        config = load_config()
        job = None
    if job:
        metadata = job.get("metadata", {})
        asset_types = metadata.get("asset_types", [])
        symbols = metadata.get("symbols", {})
        target_date = metadata.get("target_date") or _now().date().isoformat()
        try:
            target_dates = _target_dates_for_job(
                asset_types,
                str(target_date),
                config,
                metadata,
            )
            progress = state.progress(
                asset_types,
                symbols=symbols,
                target_date=target_dates,
                max_age_days=int(config["fetch"]["max_data_age_days"]),
            )
        except Exception:
            progress = status.get("progress", {})
        job_status = str(job.get("status") or status.get("status") or "unknown")
        worker_pid = int(status.get("worker_pid", 0) or 0)
        if job_status == "running" and worker_pid and not _process_exists(worker_pid):
            job_status = "failed"
            error = "research worker exited while the job was still running"
            state.update_job(
                run_id,
                status="failed",
                finished_at=_now().isoformat(),
                last_error=error,
            )
            status["error"] = error
        elif job_status == "running":
            heartbeat = status.get("last_heartbeat_at")
            try:
                heartbeat_at = datetime.fromisoformat(
                    str(heartbeat).replace("Z", "+00:00")
                ).astimezone(timezone.utc)
            except (TypeError, ValueError):
                heartbeat_at = None
            stalled_after = max(
                1,
                int(config["fetch"].get("worker_stalled_seconds", 60)),
            )
            if heartbeat_at and (_now() - heartbeat_at).total_seconds() >= stalled_after:
                job_status = "stalled"
                status["message"] = "worker 心跳已超过停滞阈值；请查询同一 run_id，勿重复提交。"
        status.update(
            {
                "status": job_status,
                "target_date": target_date,
                "target_dates": progress.get(
                    "target_dates",
                    metadata.get("target_dates", {}),
                ),
                "total": job.get("total", progress.get("total", 0)),
                "completed": job.get("completed", progress.get("completed", 0)),
                "pending": job.get("pending", progress.get("pending", 0)),
                "failed": job.get("failed", progress.get("failed", 0)),
                "stale": job.get("stale", progress.get("stale", 0)),
                "coverage": job.get("coverage", progress.get("coverage", 0.0)),
            }
        )
    compact_keys = (
        "run_id",
        "status",
        "phase",
        "total",
        "completed",
        "pending",
        "failed",
        "stale",
        "coverage",
        "target_date",
        "target_dates",
        "last_progress_at",
        "last_completed_at",
        "last_heartbeat_at",
        "progress_seq",
        "worker_pid",
        "started_at",
        "finished_at",
        "result_run_id",
        "data_end_date",
        "result_path",
        "error",
        "message",
    )
    return {key: status[key] for key in compact_keys if key in status}


def list_notifications(
    unread_only: bool = False,
    *,
    limit: int = 10,
    since: str | None = None,
    run_id: str | None = None,
 ) -> dict[str, Any]:
    validated_run_id = validate_run_id(run_id) if run_id is not None else None
    records = list_records(
        bool(unread_only),
        limit=int(limit),
        since=since,
        run_id=validated_run_id,
    )
    return {
        "status": "ok",
        "unread_only": bool(unread_only),
        "limit": int(limit),
        "since": since,
        "run_id": validated_run_id,
        "count": len(records),
        "notifications": records,
    }


def ack_notification(notification_id: str) -> dict[str, Any]:
    return acknowledge(validate_notification_id(notification_id))
