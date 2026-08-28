"""Resolve per-asset fetch targets from the local trading calendar."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALENDAR_PATH = ROOT / "data" / "benchmarks" / "trade_calendar.csv"
DEFAULT_LAG_TRADING_DAYS = {
    "stock": 0,
    "etf": 1,
    "otc_fund": 1,
}


def asset_target_dates(
    asset_types: Iterable[str],
    run_date: str,
    *,
    lag_trading_days: Mapping[str, int] | None = None,
    calendar_path: Path = DEFAULT_CALENDAR_PATH,
 ) -> dict[str, str]:
    """Return the latest complete trading date required for each asset type.

    A lag of one means the previous trading day while the market date itself is
    in progress. On a weekend or holiday, the most recent trading day is already
    complete and therefore satisfies the same one-day disclosure lag.
    """
    try:
        effective_run_date = date.fromisoformat(str(run_date))
    except ValueError as exc:
        raise ValueError(f"invalid run date: {run_date}") from exc

    path = Path(calendar_path)
    if not path.is_file():
        raise FileNotFoundError(f"trading calendar cache does not exist: {path}")
    frame = pd.read_csv(path)
    date_column = next(
        (
            column
            for column in frame.columns
            if str(column).strip() in {"trade_date", "日期", "date"}
        ),
        None,
    )
    if date_column is None:
        raise ValueError(f"{path}: trade date column is missing")
    parsed = pd.to_datetime(frame[date_column], errors="coerce").dropna()
    trading_dates = sorted(set(parsed.dt.date.tolist()))
    if not trading_dates:
        raise ValueError(f"{path}: no valid trade dates")

    eligible = [value for value in trading_dates if value <= effective_run_date]
    if not eligible:
        raise ValueError(
            f"run date {effective_run_date.isoformat()} precedes calendar coverage"
        )
    is_trading_day = eligible[-1] == effective_run_date
    configured_lags = dict(DEFAULT_LAG_TRADING_DAYS)
    configured_lags.update(lag_trading_days or {})

    targets: dict[str, str] = {}
    for raw_asset_type in asset_types:
        asset_type = str(raw_asset_type).strip()
        if not asset_type or asset_type in targets:
            continue
        lag = int(configured_lags.get(asset_type, 0))
        if lag < 0:
            raise ValueError(f"negative trading-day lag for {asset_type}: {lag}")
        offset = lag if is_trading_day else max(0, lag - 1)
        index = len(eligible) - 1 - offset
        if index < 0:
            raise ValueError(
                f"calendar does not cover lag {lag} for {asset_type} on {run_date}"
            )
        targets[asset_type] = eligible[index].isoformat()
    return targets
