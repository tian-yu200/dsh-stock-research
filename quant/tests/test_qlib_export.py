from __future__ import annotations

import pathlib
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from qlib_export import (
    FEATURE_COLUMNS,
    SourceGroup,
    build_qlib_provider,
    to_qlib_symbol,
    validate_provider,
)
from provider_lock import ProviderLock
from build_qlib_data import _ProgressWriter

class QlibExportTests(unittest.TestCase):
    def test_symbol_mapping(self) -> None:
        self.assertEqual(to_qlib_symbol("600519"), "SH600519")
        self.assertEqual(to_qlib_symbol("000001"), "SZ000001")
        self.assertEqual(to_qlib_symbol("830001"), "BJ830001")
        self.assertEqual(to_qlib_symbol("BJ830001"), "BJ830001")
        self.assertEqual(to_qlib_symbol("etf_510300"), "SH510300")
        self.assertEqual(to_qlib_symbol("sz300750"), "SZ300750")
        with self.assertRaises(ValueError):
            to_qlib_symbol("not-a-symbol")

    def test_builds_binary_provider_with_qlib_readable_features(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = pathlib.Path(temporary_dir)
            source_dir = root / "stocks"
            source_dir.mkdir()
            pd.DataFrame(
                {
                    "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
                    "open": [10, 11, 12],
                    "high": [11, 12, 13],
                    "low": [9, 10, 11],
                    "close": [10.5, 11.5, 12.5],
                    "volume": [100, 110, 120],
                    "amount": [1000, 1100, 1200],
                    "turnover": [1.0, 1.1, 1.2],
                }
            ).to_csv(source_dir / "600519.csv", index=False)
            target = root / "provider"

            summary = build_qlib_provider(
                target,
                [SourceGroup("stocks", source_dir, "*.csv")],
            )
            validated = validate_provider(target)

            self.assertEqual(summary.instruments, 1)
            self.assertEqual(validated.calendar_rows, 3)
            self.assertEqual(validated.markets, {"all": 1, "stocks": 1})
            for field in FEATURE_COLUMNS:
                values = np.fromfile(
                    target / "features" / "sh600519" / f"{field}.day.bin",
                    dtype="<f4",
                )
                self.assertEqual(len(values), 4)
                self.assertEqual(values[0], 0)
            vwap = np.fromfile(
                target / "features" / "sh600519" / "vwap.day.bin",
                dtype="<f4",
            )
            np.testing.assert_allclose(vwap[1:], [10.5, 11.5, 12.5])

            import qlib
            from qlib.config import REG_CN
            from qlib.data import D

            qlib.init(provider_uri=str(target), region=REG_CN, kernels=1)
            features = D.features(
                ["SH600519"],
                ["$close", "$vwap", "$factor"],
                start_time="2024-01-02",
                end_time="2024-01-04",
            )
            self.assertEqual(features.shape, (3, 3))
            np.testing.assert_allclose(features["$close"], features["$vwap"])
            np.testing.assert_allclose(features["$factor"], 1.0)

    def test_benchmark_calendar_cache_is_not_treated_as_an_instrument(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = pathlib.Path(temporary_dir)
            benchmark_dir = root / "benchmarks"
            benchmark_dir.mkdir()
            pd.DataFrame({"trade_date": ["2024-01-02", "2024-01-03"]}).to_csv(
                benchmark_dir / "trade_calendar.csv",
                index=False,
            )
            pd.DataFrame(
                {
                    "date": ["2024-01-02", "2024-01-03"],
                    "open": [10, 11],
                    "high": [11, 12],
                    "low": [9, 10],
                    "close": [10.5, 11.5],
                    "volume": [100, 110],
                    "amount": [1000, 1100],
                    "turnover": [1.0, 1.1],
                }
            ).to_csv(benchmark_dir / "000300.csv", index=False)

            summary = build_qlib_provider(
                root / "provider",
                [SourceGroup("benchmarks", benchmark_dir, "*.csv")],
            )

            self.assertEqual(summary.instruments, 1)
            self.assertEqual(summary.markets, {"all": 1, "benchmarks": 1})

    def test_provider_build_does_not_replace_provider_while_reader_holds_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = pathlib.Path(temporary_dir)
            source_dir = root / "stocks"
            source_dir.mkdir()
            pd.DataFrame(
                {
                    "date": ["2024-01-02", "2024-01-03"],
                    "open": [10, 11],
                    "high": [11, 12],
                    "low": [9, 10],
                    "close": [10.5, 11.5],
                    "volume": [100, 110],
                    "amount": [1000, 1100],
                    "turnover": [1.0, 1.1],
                }
            ).to_csv(source_dir / "600519.csv", index=False)
            target = root / "provider"
            build_qlib_provider(
                target,
                [SourceGroup("stocks", source_dir, "*.csv")],
            )
            marker = target / "reader-visible-marker"
            marker.write_text("old-provider", encoding="utf-8")


            reader_lock = ProviderLock(target, purpose="research")
            reader_lock.acquire(timeout_seconds=0.1)
            try:
                with self.assertRaises(TimeoutError):
                    build_qlib_provider(
                        target,
                        [SourceGroup("stocks", source_dir, "*.csv")],
                        replace=True,
                        lock_timeout_seconds=0.05,
                    )
                self.assertEqual(marker.read_text(encoding="utf-8"), "old-provider")
                self.assertTrue(target.exists())
            finally:
                reader_lock.release()


            build_qlib_provider(
                target,
                [SourceGroup("stocks", source_dir, "*.csv")],
                replace=True,
                lock_timeout_seconds=1.0,
            )
            self.assertFalse(marker.exists())

    def test_unchanged_sources_reuse_existing_provider_without_loading_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = pathlib.Path(temporary_dir)
            source_dir = root / "stocks"
            source_dir.mkdir()
            pd.DataFrame(
                {
                    "date": ["2024-01-02", "2024-01-03"],
                    "open": [10, 11],
                    "high": [11, 12],
                    "low": [9, 10],
                    "close": [10.5, 11.5],
                    "volume": [100, 110],
                    "amount": [1000, 1100],
                    "turnover": [1.0, 1.1],
                }
            ).to_csv(source_dir / "600519.csv", index=False)
            target = root / "provider"
            groups = [SourceGroup("stocks", source_dir, "*.csv")]
            build_qlib_provider(target, groups)
            marker = target / "cache-marker"
            marker.write_text("preserved", encoding="utf-8")

            with patch("qlib_export._load_sources") as load_sources:
                progress_events: list[dict[str, object]] = []
                summary = build_qlib_provider(
                    target,
                    groups,
                    replace=True,
                    skip_if_unchanged=True,
                    progress_callback=progress_events.append,
                )

            load_sources.assert_not_called()
            self.assertTrue(summary.cache_hit)
            self.assertEqual(marker.read_text(encoding="utf-8"), "preserved")
            self.assertEqual(progress_events[-1]["phase"], "cache_hit")
            self.assertTrue(progress_events[-1]["cache_hit"])

    def test_provider_build_reports_bounded_monotonic_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = pathlib.Path(temporary_dir)
            source_dir = root / "stocks"
            source_dir.mkdir()
            pd.DataFrame(
                {
                    "date": ["2024-01-02", "2024-01-03"],
                    "open": [10, 11],
                    "high": [11, 12],
                    "low": [9, 10],
                    "close": [10.5, 11.5],
                    "volume": [100, 110],
                    "amount": [1000, 1100],
                    "turnover": [1.0, 1.1],
                }
            ).to_csv(source_dir / "600519.csv", index=False)
            events: list[dict[str, object]] = []

            summary = build_qlib_provider(
                root / "provider",
                [SourceGroup("stocks", source_dir, "*.csv")],
                progress_callback=events.append,
            )

        self.assertFalse(summary.cache_hit)
        self.assertTrue(events)
        sequences = [int(event["phase_seq"]) for event in events]
        self.assertEqual(sequences, sorted(set(sequences)))
        self.assertTrue(all(
            right > left for left, right in zip(sequences, sequences[1:])
        ))
        required_phases = {
            "fingerprinting",
            "loading_sources",
            "writing_features",
            "publishing",
            "validating",
            "complete",
        }
        self.assertTrue(
            required_phases.issubset(
                {str(event["phase"]) for event in events}
            )
        )
        for event in events:
            completed = int(event.get("completed", 0) or 0)
            total = int(event.get("total", 0) or 0)
            self.assertGreaterEqual(completed, 0)
            self.assertGreaterEqual(total, 0)
            self.assertLessEqual(completed, total)
        self.assertEqual(events[-1]["phase"], "complete")

    def test_progress_writer_discards_unbounded_payload_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = pathlib.Path(temporary_dir) / "progress.json"
            _ProgressWriter(path)(
                {
                    "phase": "writing_features",
                    "phase_seq": 4,
                    "completed": 1,
                    "total": 2,
                    "updated_at": "2026-08-31T00:00:00+00:00",
                    "cache_hit": False,
                    "source_fingerprint_digest": "digest",
                    "stdout": "x" * 100000,
                    "records": ["large"] * 1000,
                }
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            file_size = path.stat().st_size

        self.assertEqual(
            set(payload),
            {
                "phase", "phase_seq", "completed", "total", "updated_at",
                "cache_hit", "source_fingerprint_digest"
            },
        )
        self.assertLess(file_size, 2048)

if __name__ == "__main__":
    unittest.main()
