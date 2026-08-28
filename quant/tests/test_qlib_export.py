from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

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

if __name__ == "__main__":
    unittest.main()
