"""Resolve the China market trading calendar without relying on Qlib data."""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import akshare as ak
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = ROOT / "data" / "benchmarks" / "trade_calendar.csv"
TIMEZONE = ZoneInfo("Asia/Shanghai")



def _normalize_calendar(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        raise ValueError(f"{source}: empty trading calendar")
    date_column = next(
        (column for column in frame.columns if str(column).strip() in {"trade_date", "日期", "date"}),
        None,
    )
    if date_column is None:
        raise ValueError(f"{source}: trade date column is missing")
    values = pd.to_datetime(frame[date_column], errors="coerce").dropna().dt.normalize()
    values = values.drop_duplicates().sort_values()
    if values.empty:
        raise ValueError(f"{source}: no valid trade dates")
    return pd.DataFrame({"trade_date": values.dt.date.astype(str)})



def _atomic_write_calendar(frame: pd.DataFrame, path: Path = CACHE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)



def _online_calendar(retries: int = 3, delay_seconds: float = 2.0) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return _normalize_calendar(
                ak.tool_trade_date_hist_sina(),
                "AkShare tool_trade_date_hist_sina",
            )
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(delay_seconds)
    raise RuntimeError(f"AkShare trading calendar failed after {retries} attempts: {last_error}")



def _cached_calendar(path: Path = CACHE_PATH) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"trading calendar cache does not exist: {path}")
    return _normalize_calendar(pd.read_csv(path), str(path))



def resolve_trading_day(target: date, cache_path: Path = CACHE_PATH) -> dict[str, object]:
    online_error: str | None = None
    try:
        calendar = _online_calendar()
        _atomic_write_calendar(calendar, cache_path)
        source = "akshare:tool_trade_date_hist_sina"
    except Exception as exc:
        online_error = f"{type(exc).__name__}: {exc}"
        calendar = _cached_calendar(cache_path)
        source = "local_cache"

    dates = pd.to_datetime(calendar["trade_date"], errors="raise").dt.date.tolist()
    first_date, last_date = dates[0], dates[-1]
    if target < first_date or target > last_date:
        raise ValueError(
            f"target date {target.isoformat()} is outside calendar coverage "
            f"{first_date.isoformat()}..{last_date.isoformat()}"
        )
    month_dates = [
        value for value in dates
        if value.year == target.year and value.month == target.month
    ]
    is_trading_day = target in dates
    return {
        "status": "ok",
        "date": target.isoformat(),
        "is_trading_day": is_trading_day,
        "is_month_end_trading_day": bool(
            is_trading_day and month_dates and target == month_dates[-1]
        ),
        "source": source,
        "cache_path": str(cache_path.resolve()),
        "coverage_start": first_date.isoformat(),
        "coverage_end": last_date.isoformat(),
        "online_error": online_error,
    }



def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--date",
        default=datetime.now(TIMEZONE).date().isoformat(),
    )
    args = parser.parse_args()
    try:
        target = date.fromisoformat(args.date)
        result = resolve_trading_day(target)
    except Exception as exc:
        result = {
            "status": "error",
            "date": args.date,
            "error": f"{type(exc).__name__}: {exc}",
            "cache_path": str(CACHE_PATH.resolve()),
        }
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
