"""Canonical daily-bar schema shared by data-source adapters and Qlib export."""
from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

DAILY_COLUMNS = ("date", "open", "high", "low", "close", "volume", "amount", "turnover")
REQUIRED_COLUMNS = ("date", "open", "high", "low", "close", "volume")

class DailyBarSchemaError(ValueError):
    """A deterministic incompatibility in an upstream daily-bar payload."""

    def __init__(
        self,
        message: str,
        *,
        failure_class: str,
        field_anomalies: dict[str, dict[str, int]] | None = None,
    ) -> None:
        self.failure_class = failure_class
        self.field_anomalies = field_anomalies or {}
        super().__init__(message)

    @classmethod
    def for_source(
        cls,
        source: str,
        detail: str,
        *,
        failure_class: str,
        field_anomalies: dict[str, dict[str, int]] | None = None,
    ) -> "DailyBarSchemaError":
        return cls(
            f"{source}: {detail}",
            failure_class=failure_class,
            field_anomalies=field_anomalies,
        )

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
    groups: dict[str, list[pd.Series]] = {}
    for position, column in enumerate(data.columns):
        canonical = _canonical_name(column)
        series = data.iloc[:, position].reset_index(drop=True)
        groups.setdefault(canonical, []).append(series)

    index_is_date = isinstance(data.index, pd.DatetimeIndex) or (
        data.index.name is not None and _canonical_name(data.index.name) == "date"
    )
    if index_is_date:
        index_values = pd.Series(data.index.to_numpy(), index=range(len(data)))
        groups.setdefault("date", []).append(index_values)

    if "date" not in groups:
        raise DailyBarSchemaError.for_source(
            source,
            "no date column or date index",
            failure_class="missing_required_columns",
            field_anomalies={"date": {"missing": 1}},
        )

    field_anomalies: dict[str, dict[str, int]] = {}
    merged: dict[str, pd.Series] = {}
    for column, candidates in groups.items():
        if column not in DAILY_COLUMNS:
            continue
        normalized_candidates = [_coerce_field(candidate, column) for candidate in candidates]
        selected = normalized_candidates[0]
        conflicts = 0
        for candidate in normalized_candidates[1:]:
            both_present = selected.notna() & candidate.notna()
            conflicts += int((both_present & selected.ne(candidate)).sum())
        if conflicts:
            anomaly = {
                "candidate_count": len(candidates),
                "conflicting_rows": conflicts,
            }
            field_anomalies[column] = anomaly
            if column in REQUIRED_COLUMNS:
                raise DailyBarSchemaError.for_source(
                    source,
                    f"conflicting duplicate values for {column} ({conflicts} rows)",
                    failure_class="schema_conflict",
                    field_anomalies={column: anomaly},
                )
        if len(normalized_candidates) > 1:
            selected = max(
                normalized_candidates,
                key=lambda candidate: int(candidate.notna().sum()),
            )
            for candidate in normalized_candidates:
                selected = selected.combine_first(candidate)
        merged[column] = selected

    data = pd.DataFrame(merged)
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise DailyBarSchemaError.for_source(
            source,
            f"missing required columns: {missing}",
            failure_class="missing_required_columns",
            field_anomalies={column: {"missing": 1} for column in missing},
        )

    for column in DAILY_COLUMNS:
        if column not in data.columns:
            data[column] = pd.NA

    data = data.loc[:, DAILY_COLUMNS].copy()
    parsed_dates = pd.to_datetime(data["date"], errors="coerce")
    if parsed_dates.isna().any():
        invalid_count = int(parsed_dates.isna().sum())
        raise DailyBarSchemaError.for_source(
            source,
            f"{invalid_count} invalid dates",
            failure_class="invalid_required_values",
            field_anomalies={"date": {"invalid_rows": invalid_count}},
        )
    data["date"] = parsed_dates.dt.normalize()

    numeric_columns: Iterable[str] = DAILY_COLUMNS[1:]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    invalid_required = data.loc[:, REQUIRED_COLUMNS[1:]].isna().any(axis=1)
    if invalid_required.any():
        invalid_count = int(invalid_required.sum())
        raise DailyBarSchemaError.for_source(
            source,
            f"{invalid_count} rows have missing OHLCV values",
            failure_class="invalid_required_values",
            field_anomalies={
                column: {"invalid_rows": int(data[column].isna().sum())}
                for column in REQUIRED_COLUMNS[1:]
                if data[column].isna().any()
            },
        )
    if (data["volume"] < 0).any():
        invalid_count = int((data["volume"] < 0).sum())
        raise DailyBarSchemaError.for_source(
            source,
            "volume cannot be negative",
            failure_class="invalid_required_values",
            field_anomalies={"volume": {"negative_rows": invalid_count}},
        )

    data = data.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    if field_anomalies:
        data.attrs["field_anomalies"] = field_anomalies
    return data


def _coerce_field(series: pd.Series, column: str) -> pd.Series:
    if column == "date":
        return pd.to_datetime(series, errors="coerce").dt.normalize()
    return pd.to_numeric(series, errors="coerce")
