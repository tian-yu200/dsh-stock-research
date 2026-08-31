"""Build a self-contained Qlib provider from canonical AkShare CSV files."""
from __future__ import annotations

import json
import hashlib
import os
import re
import shutil
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from market_data import normalize_daily_bars
from provider_lock import ProviderLock

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


ProgressCallback = Callable[[dict[str, object]], None]


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
    cache_hit: bool = False
    source_fingerprint: str = ""


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


def source_fingerprint(groups: Iterable[SourceGroup]) -> dict[str, int | str]:
    """Hash bounded file metadata before loading thousands of source CSV files."""
    digest = hashlib.sha256()
    file_count = 0
    total_size = 0
    for group in groups:
        directory = group.directory.resolve()
        digest.update(f"group\0{group.name}\0{directory}\0{group.pattern}\n".encode("utf-8"))
        if not directory.exists():
            continue
        for path in sorted(directory.glob(group.pattern)):
            if not path.is_file():
                continue
            stat = path.stat()
            relative = path.relative_to(directory).as_posix()
            digest.update(
                f"file\0{group.name}\0{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode(
                    "utf-8"
                )
            )
            file_count += 1
            total_size += stat.st_size
    return {
        "version": 1,
        "algorithm": "sha256:path-size-mtime_ns",
        "digest": digest.hexdigest(),
        "file_count": file_count,
        "total_size": total_size,
    }


def provider_fingerprint(target_dir: Path) -> dict[str, int | str] | None:
    manifest_path = Path(target_dir).resolve() / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    value = manifest.get("source_fingerprint") if isinstance(manifest, dict) else None
    return value if isinstance(value, dict) else None


def _load_sources(
    groups: Iterable[SourceGroup],
    *,
    progress_callback: Callable[[int, int], None] | None = None,
    progress_total: int = 0,
) -> dict[str, tuple[pd.DataFrame, set[str], Path]]:
    records: dict[str, tuple[pd.DataFrame, set[str], Path]] = {}
    processed = 0
    progress_step = max(1, progress_total // 100) if progress_total else 1
    for group in groups:
        if not group.directory.exists():
            continue
        for path in sorted(group.directory.glob(group.pattern)):
            processed += 1
            try:
                symbol = to_qlib_symbol(path.stem)
            except ValueError:
                # The assistant stores its AkShare trading-calendar cache in
                # the benchmarks directory, but it is metadata rather than an
                # instrument source.
                if group.name == "benchmarks":
                    if progress_callback and processed % progress_step == 0:
                        progress_callback(processed, progress_total)
                    continue
                raise
            if symbol in records:
                other = records[symbol][2]
                raise ValueError(f"duplicate instrument {symbol}: {other} and {path}")
            frame = normalize_daily_bars(pd.read_csv(path), source=str(path))
            records[symbol] = (frame, {group.name}, path)
            if progress_callback and processed % progress_step == 0:
                progress_callback(processed, progress_total)
    if not records:
        raise ValueError("no AkShare CSV files were found")
    if progress_callback:
        progress_callback(processed, progress_total)
    return records


def _write_feature(path: Path, start_index: int, values: pd.Series) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = np.hstack(([start_index], values.to_numpy(dtype=np.float32))).astype("<f4")
    payload.tofile(path)


def _build_staging(
    staging_dir: Path,
    records: dict[str, tuple[pd.DataFrame, set[str], Path]],
    groups: Iterable[SourceGroup],
    fingerprint: dict[str, int | str],
    *,
    progress_callback: Callable[[int, int], None] | None = None,
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
    total = len(records)
    progress_step = max(1, total // 100) if total else 1
    for index, (symbol, (frame, markets, source_path)) in enumerate(
        sorted(records.items()),
        start=1,
    ):
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
        if progress_callback and (index % progress_step == 0 or index == total):
            progress_callback(index, total)

    for market, lines in market_lines.items():
        (instrument_dir / f"{market}.txt").write_text("".join(lines), encoding="utf-8")

    summary = BuildSummary(
        target_dir="",
        calendar_start=calendar_index[0].strftime("%Y-%m-%d"),
        calendar_end=calendar_index[-1].strftime("%Y-%m-%d"),
        calendar_rows=len(calendar_index),
        instruments=len(records),
        markets={market: len(lines) for market, lines in market_lines.items()},
        source_fingerprint=str(fingerprint["digest"]),
    )
    manifest = {
        "format": "qlib-file-provider",
        "frequency": "day",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": asdict(summary),
        "source_fingerprint": fingerprint,
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
    skip_if_unchanged: bool = False,
    lock_timeout_seconds: float = 30.0,
    progress_callback: ProgressCallback | None = None,
) -> BuildSummary:
    """Build and atomically publish a Qlib provider from source groups."""
    target_dir = target_dir.resolve()
    groups = tuple(groups)
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    progress_seq = 0

    def emit(
        phase: str,
        *,
        completed: int = 0,
        total: int = 0,
        **fields: object,
    ) -> None:
        nonlocal progress_seq
        progress_seq += 1
        payload: dict[str, object] = {
            "phase": str(phase),
            "phase_seq": progress_seq,
            "completed": max(0, int(completed)),
            "total": max(0, int(total)),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        payload.update(fields)
        if progress_callback is not None:
            try:
                progress_callback(payload)
            except Exception:
                # Progress reporting must never make a provider build fail.
                pass

    def child_progress(phase: str) -> Callable[[int, int], None]:
        return lambda completed, total: emit(
            phase, completed=completed, total=total
        )

    emit("fingerprinting", total=1)
    fingerprint = source_fingerprint(groups)
    emit(
        "fingerprinting",
        completed=1,
        total=1,
        source_fingerprint_digest=str(fingerprint["digest"]),
    )
    lock = ProviderLock(target_dir, purpose="provider-build", exclusive=True)
    lock.acquire(timeout_seconds=lock_timeout_seconds)
    try:
        if (
            skip_if_unchanged
            and target_dir.exists()
            and provider_fingerprint(target_dir) == fingerprint
        ):
            emit("validating", total=1)
            summary = validate_provider(
                target_dir,
                progress_callback=child_progress("validating"),
            )
            result = BuildSummary(
                **{
                    **asdict(summary),
                    "cache_hit": True,
                    "source_fingerprint": str(fingerprint["digest"]),
                }
            )
            emit(
                "cache_hit",
                completed=1,
                total=1,
                cache_hit=True,
                source_fingerprint_digest=str(fingerprint["digest"]),
            )
            return result
        progress_total = int(fingerprint.get("file_count", 0) or 0)
        emit("loading_sources", total=progress_total)
        records = _load_sources(
            groups,
            progress_callback=child_progress("loading_sources"),
            progress_total=progress_total,
        )
        emit(
            "loading_sources",
            completed=progress_total or len(records),
            total=progress_total or len(records),
        )
        staging_dir = target_dir.parent / f".{target_dir.name}.staging-{uuid.uuid4().hex}"
        backup_dir = target_dir.parent / f".{target_dir.name}.backup-{uuid.uuid4().hex}"
        staging_dir.mkdir()
        try:
            emit("writing_features", total=len(records))
            summary = _build_staging(
                staging_dir,
                records,
                groups,
                fingerprint,
                progress_callback=child_progress("writing_features"),
            )
            emit(
                "writing_features",
                completed=len(records),
                total=len(records),
            )
            emit("publishing", total=1)
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
            emit("publishing", completed=1, total=1)
            emit("validating", total=summary.instruments)
            validated = validate_provider(
                target_dir,
                progress_callback=child_progress("validating"),
            )
            result = BuildSummary(
                **{
                    **asdict(validated),
                    "cache_hit": False,
                    "source_fingerprint": str(fingerprint["digest"]),
                }
            )
            emit(
                "complete",
                completed=1,
                total=1,
                cache_hit=False,
                source_fingerprint_digest=str(fingerprint["digest"]),
            )
            return result
        finally:
            if staging_dir.exists():
                shutil.rmtree(staging_dir)
    finally:
        lock.release()


def validate_provider(
    target_dir: Path,
    *,
    progress_callback: Callable[[int, int], None] | None = None,
) -> BuildSummary:
    """Validate provider structure and binary lengths without initializing Qlib."""
    target_dir = target_dir.resolve()
    calendar = (target_dir / "calendars" / "day.txt").read_text(encoding="utf-8").splitlines()
    if not calendar or calendar != sorted(set(calendar)):
        raise ValueError("calendar must be non-empty, unique, and sorted")
    calendar_lookup = {date: index for index, date in enumerate(calendar)}
    instruments = []
    instrument_lines = (
        target_dir / "instruments" / "all.txt"
    ).read_text(encoding="utf-8").splitlines()
    total = len(instrument_lines)
    progress_step = max(1, total // 100) if total else 1
    if progress_callback:
        progress_callback(0, total)
    for index, line in enumerate(instrument_lines, start=1):
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
        if progress_callback and (index % progress_step == 0 or index == total):
            progress_callback(index, total)
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
        source_fingerprint=str((provider_fingerprint(target_dir) or {}).get("digest", "")),
    )
