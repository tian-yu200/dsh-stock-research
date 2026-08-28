"""Build a self-contained Qlib provider from canonical AkShare CSV files."""
from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from market_data import normalize_daily_bars

FEATURE_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "vwap",
    "volume",
    "amount",
    "turnover",
    "change",
    "factor",
)


@dataclass(frozen=True)
class SourceGroup:
    name: str
    directory: Path
    pattern: str


@dataclass(frozen=True)
class BuildSummary:
    target_dir: str
    calendar_start: str
    calendar_end: str
    calendar_rows: int
    instruments: int
    markets: dict[str, int]


def to_qlib_symbol(value: str) -> str:
    """Convert an A-share or ETF filename stem to Qlib's exchange-prefixed symbol."""
    symbol = value.strip().upper()
    symbol = re.sub(r"^(ETF|FUND)_", "", symbol)
    if re.fullmatch(r"(?:SH|SZ|BJ)\d{6}", symbol):
        return symbol
    if not re.fullmatch(r"\d{6}", symbol):
        raise ValueError(f"cannot infer exchange for symbol: {value}")
    if symbol.startswith(("4", "8", "92")):
        exchange = "BJ"
    else:
        exchange = "SH" if symbol.startswith(("5", "6", "9")) else "SZ"
    return f"{exchange}{symbol}"


def default_source_groups(root: Path) -> tuple[SourceGroup, ...]:
    return (
        SourceGroup("stocks", root / "data" / "stocks", "*.csv"),
        SourceGroup("etfs", root / "data" / "funds", "etf_*.csv"),
        SourceGroup("benchmarks", root / "data" / "benchmarks", "*.csv"),
    )


def _load_sources(groups: Iterable[SourceGroup]) -> dict[str, tuple[pd.DataFrame, set[str], Path]]:
    records: dict[str, tuple[pd.DataFrame, set[str], Path]] = {}
    for group in groups:
        if not group.directory.exists():
            continue
        for path in sorted(group.directory.glob(group.pattern)):
            try:
                symbol = to_qlib_symbol(path.stem)
            except ValueError:
                # The assistant stores its AkShare trading-calendar cache in
                # the benchmarks directory, but it is metadata rather than an
                # instrument source.
                if group.name == "benchmarks":
                    continue
                raise
            if symbol in records:
                other = records[symbol][2]
                raise ValueError(f"duplicate instrument {symbol}: {other} and {path}")
            frame = normalize_daily_bars(pd.read_csv(path), source=str(path))
            records[symbol] = (frame, {group.name}, path)
    if not records:
        raise ValueError("no AkShare CSV files were found")
    return records


def _write_feature(path: Path, start_index: int, values: pd.Series) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = np.hstack(([start_index], values.to_numpy(dtype=np.float32))).astype("<f4")
    payload.tofile(path)


def _build_staging(
    staging_dir: Path,
    records: dict[str, tuple[pd.DataFrame, set[str], Path]],
    groups: Iterable[SourceGroup],
) -> BuildSummary:
    calendars = sorted({date for frame, _, _ in records.values() for date in frame["date"]})
    calendar_index = pd.DatetimeIndex(calendars)
    calendar_dir = staging_dir / "calendars"
    instrument_dir = staging_dir / "instruments"
    calendar_dir.mkdir(parents=True)
    instrument_dir.mkdir(parents=True)
    (calendar_dir / "day.txt").write_text(
        "".join(f"{date:%Y-%m-%d}\n" for date in calendar_index),
        encoding="utf-8",
    )

    market_lines: dict[str, list[str]] = {"all": []}
    for group in groups:
        market_lines.setdefault(group.name, [])

    manifest_instruments = []
    for symbol, (frame, markets, source_path) in sorted(records.items()):
        start = frame["date"].iloc[0]
        end = frame["date"].iloc[-1]
        line = f"{symbol}\t{start:%Y-%m-%d}\t{end:%Y-%m-%d}\n"
        market_lines["all"].append(line)
        for market in markets:
            market_lines[market].append(line)

        aligned = frame.set_index("date").reindex(calendar_index[
            (calendar_index >= start) & (calendar_index <= end)
        ])
        # Upstream sources do not expose a consistent volume unit, so amount /
        # volume is not a reliable VWAP across adapters. Close is the stable proxy.
        aligned["vwap"] = aligned["close"]
        aligned["change"] = aligned["close"].pct_change(fill_method=None)
        aligned["factor"] = 1.0
        start_index = int(calendar_index.get_loc(start))
        for field in FEATURE_COLUMNS:
            _write_feature(
                staging_dir / "features" / symbol.lower() / f"{field}.day.bin",
                start_index,
                aligned[field],
            )
        manifest_instruments.append(
            {
                "instrument": symbol,
                "source": str(source_path.resolve()),
                "start": start.strftime("%Y-%m-%d"),
                "end": end.strftime("%Y-%m-%d"),
                "rows": len(frame),
                "markets": sorted(markets),
            }
        )

    for market, lines in market_lines.items():
        (instrument_dir / f"{market}.txt").write_text("".join(lines), encoding="utf-8")

    summary = BuildSummary(
        target_dir="",
        calendar_start=calendar_index[0].strftime("%Y-%m-%d"),
        calendar_end=calendar_index[-1].strftime("%Y-%m-%d"),
        calendar_rows=len(calendar_index),
        instruments=len(records),
        markets={market: len(lines) for market, lines in market_lines.items()},
    )
    manifest = {
        "format": "qlib-file-provider",
        "frequency": "day",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": asdict(summary),
        "instruments": manifest_instruments,
    }
    (staging_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def build_qlib_provider(
    target_dir: Path,
    groups: Iterable[SourceGroup],
    *,
    replace: bool = False,
) -> BuildSummary:
    """Build and atomically publish a Qlib provider from source groups."""
    target_dir = target_dir.resolve()
    groups = tuple(groups)
    records = _load_sources(groups)
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = target_dir.parent / f".{target_dir.name}.staging-{uuid.uuid4().hex}"
    backup_dir = target_dir.parent / f".{target_dir.name}.backup-{uuid.uuid4().hex}"
    staging_dir.mkdir()
    try:
        summary = _build_staging(staging_dir, records, groups)
        if target_dir.exists():
            if not replace:
                raise FileExistsError(f"target exists; pass --replace to rebuild: {target_dir}")
            os.replace(target_dir, backup_dir)
        try:
            os.replace(staging_dir, target_dir)
        except Exception:
            if backup_dir.exists() and not target_dir.exists():
                os.replace(backup_dir, target_dir)
            raise
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        return BuildSummary(**{**asdict(summary), "target_dir": str(target_dir)})
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)


def validate_provider(target_dir: Path) -> BuildSummary:
    """Validate provider structure and binary lengths without initializing Qlib."""
    target_dir = target_dir.resolve()
    calendar = (target_dir / "calendars" / "day.txt").read_text(encoding="utf-8").splitlines()
    if not calendar or calendar != sorted(set(calendar)):
        raise ValueError("calendar must be non-empty, unique, and sorted")
    calendar_lookup = {date: index for index, date in enumerate(calendar)}
    instruments = []
    for line in (target_dir / "instruments" / "all.txt").read_text(encoding="utf-8").splitlines():
        symbol, start, end = line.split("\t")
        if start not in calendar_lookup or end not in calendar_lookup:
            raise ValueError(f"{symbol}: instrument range is outside calendar")
        expected = calendar_lookup[end] - calendar_lookup[start] + 2
        for field in FEATURE_COLUMNS:
            path = target_dir / "features" / symbol.lower() / f"{field}.day.bin"
            values = np.fromfile(path, dtype="<f4")
            if len(values) != expected or int(values[0]) != calendar_lookup[start]:
                raise ValueError(f"{symbol}/{field}: invalid binary extent")
        instruments.append(symbol)
    market_counts = {}
    for path in (target_dir / "instruments").glob("*.txt"):
        market_counts[path.stem] = len(path.read_text(encoding="utf-8").splitlines())
    return BuildSummary(
        target_dir=str(target_dir),
        calendar_start=calendar[0],
        calendar_end=calendar[-1],
        calendar_rows=len(calendar),
        instruments=len(instruments),
        markets=market_counts,
    )
