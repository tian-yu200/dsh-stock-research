"""Canonical daily-bar schema shared by data-source adapters and Qlib export."""
from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

DAILY_COLUMNS = ("date", "open", "high", "low", "close", "volume", "amount", "turnover")
REQUIRED_COLUMNS = ("date", "open", "high", "low", "close", "volume")

class EmptyDailyBarResponse(ValueError):
    """An upstream daily-bar endpoint returned no rows."""


_COLUMN_ALIASES = {
    "日期": "date",
    "净值日期": "date",
    "交易日期": "date",
    "date": "date",
    "开盘": "open",
    "open": "open",
    "最高": "high",
    "high": "high",
    "最低": "low",
    "low": "low",
    "收盘": "close",
    "close": "close",
    "成交量": "volume",
    "volume": "volume",
    "成交额": "amount",
    "amount": "amount",
    "换手率": "turnover",
    "turnover": "turnover",
}

def _canonical_name(value: object) -> str:
    name = str(value).strip()
    return _COLUMN_ALIASES.get(name, _COLUMN_ALIASES.get(name.lower(), name.lower()))


def normalize_daily_bars(frame: pd.DataFrame, *, source: str = "unknown") -> pd.DataFrame:
    """Return sorted, duplicate-free bars with one stable cross-source schema.

    Price and volume fields are required. Amount and turnover are optional and are
    represented by ``NaN`` when an upstream source does not provide them.
    """
    if frame is None or frame.empty:
        raise EmptyDailyBarResponse(f"{source}: empty daily-bar response")

    data = frame.copy()
    canonical_columns = {_canonical_name(column) for column in data.columns}
    if "date" not in canonical_columns:
        index_is_date = isinstance(data.index, pd.DatetimeIndex) or _canonical_name(data.index.name) == "date"
        if not index_is_date:
            raise ValueError(f"{source}: no date column or date index")
        data = data.reset_index()
        data = data.rename(columns={data.columns[0]: "date"})

    data = data.rename(columns={column: _canonical_name(column) for column in data.columns})
    if data.columns.duplicated().any():
        duplicate_names = sorted(set(data.columns[data.columns.duplicated()].tolist()))
        raise ValueError(f"{source}: duplicate canonical columns: {duplicate_names}")

    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"{source}: missing required columns: {missing}")

    for column in DAILY_COLUMNS:
        if column not in data.columns:
            data[column] = pd.NA

    data = data.loc[:, DAILY_COLUMNS].copy()
    parsed_dates = pd.to_datetime(data["date"], errors="coerce")
    if parsed_dates.isna().any():
        raise ValueError(f"{source}: {int(parsed_dates.isna().sum())} invalid dates")
    data["date"] = parsed_dates.dt.normalize()

    numeric_columns: Iterable[str] = DAILY_COLUMNS[1:]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    invalid_required = data.loc[:, REQUIRED_COLUMNS[1:]].isna().any(axis=1)
    if invalid_required.any():
        raise ValueError(f"{source}: {int(invalid_required.sum())} rows have missing OHLCV values")
    if (data["volume"] < 0).any():
        raise ValueError(f"{source}: volume cannot be negative")

    data = data.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    return data
