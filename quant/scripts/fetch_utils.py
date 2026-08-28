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
        combined = incoming.copy()
    result = normalize_daily_bars(combined, source=source)
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
    combined = pd.concat(frames, ignore_index=True)
    date_column, nav_column = _fund_columns(combined)
    if date_column is None or nav_column is None:
        raise FundNavValidationError(
            source,
            "fund NAV requires date and unit NAV columns",
            total_rows=len(combined),
            samples=[
                {"columns": [str(column) for column in combined.columns[:10]]}
            ],
        )
    parsed_dates = pd.to_datetime(
        combined[date_column], errors="coerce", format="mixed"
    )
    parsed_navs = pd.to_numeric(combined[nav_column], errors="coerce")
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
                    "date": str(combined.at[index, date_column]),
                    "nav": str(combined.at[index, nav_column]),
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
