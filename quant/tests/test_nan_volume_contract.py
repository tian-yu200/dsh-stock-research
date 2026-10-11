# -*- coding: utf-8 -*-
"""T11: volume may be NaN in normalize_daily_bars (OHLC still required).

T1 unit repair keeps OHLC and sets an unprovable volume/turnover to NaN instead
of inventing a number. Before T11, ``normalize_daily_bars`` required non-null
OHLCV, so a full provider build (qlib_export.py:149) crashed on
``stocks/300212.csv`` and every incremental merge re-normalising that CSV failed.
"""
from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from market_data import DailyBarSchemaError, normalize_daily_bars  # noqa: E402


def _frame(**overrides):
    base = {
        "date": ["2024-09-26", "2024-09-27", "2024-09-30"],
        "open": [140.0, 150.43, 184.52],
        "high": [150.0, 176.22, 209.07],
        "low": [139.0, 148.94, 179.63],
        "close": [147.78, 175.89, 208.66],
        "volume": [31373200.0, 70974400.0, 80104400.0],
        "amount": [530672000.0, 1375630800.0, 1859997000.0],
        "turnover": [0.0449, 3.2692, 0.1147],
    }
    base.update(overrides)
    return pd.DataFrame(base)


class NormalizeNanVolumeTests(unittest.TestCase):
    def test_nan_volume_row_is_accepted(self) -> None:
        frame = _frame(volume=[31373200.0, None, None],
                       turnover=[0.0449, None, None])
        out = normalize_daily_bars(frame, source="t")
        self.assertEqual(len(out), 3)
        self.assertEqual(out["close"].tolist(), [147.78, 175.89, 208.66])
        self.assertTrue(pd.isna(out["volume"].iloc[1]))
        self.assertTrue(pd.isna(out["volume"].iloc[2]))

    def test_nan_close_is_rejected(self) -> None:
        with self.assertRaises(DailyBarSchemaError):
            normalize_daily_bars(_frame(close=[147.78, None, 208.66]), source="t")

    def test_nan_open_high_low_are_rejected(self) -> None:
        for column in ("open", "high", "low"):
            with self.subTest(column=column):
                values = _frame()[column].tolist()
                values[1] = None
                with self.assertRaises(DailyBarSchemaError):
                    normalize_daily_bars(_frame(**{column: values}), source="t")

    def test_negative_volume_is_rejected(self) -> None:
        with self.assertRaises(DailyBarSchemaError):
            normalize_daily_bars(_frame(volume=[100.0, -5.0, 100.0]), source="t")


class MergeNanVolumeTests(unittest.TestCase):
    def test_incremental_merge_keeps_nan_volume_rows(self) -> None:
        from fetch_utils import merge_daily_frames

        existing = _frame(volume=[31373200.0, None, None],
                          turnover=[0.0449, None, None])
        incoming = _frame(date=["2024-10-08"], open=[184.0], high=[185.0],
                          low=[180.0], close=[183.0], volume=[1000.0],
                          amount=[183000.0], turnover=[0.001])
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "300212.csv"
            existing.to_csv(path, index=False)
            merged = merge_daily_frames(path, incoming, source="t")
        dates = merged["date"].astype(str).tolist()
        self.assertIn("2024-09-27", dates)
        self.assertIn("2024-09-30", dates)
        self.assertEqual(len(merged), 4)


class ProviderBuildNanVolumeTests(unittest.TestCase):
    def test_provider_build_accepts_nan_volume_row(self) -> None:
        from qlib_export import FEATURE_COLUMNS, SourceGroup, build_qlib_provider, validate_provider
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            src = root / "stocks"
            src.mkdir()
            _frame(volume=[31373200.0, None, None],
                   turnover=[0.0449, None, None]).to_csv(src / "300212.csv", index=False)
            _frame(date=["2024-09-26", "2024-09-27", "2024-09-30"]).to_csv(src / "600519.csv", index=False)
            target = root / "provider"
            summary = build_qlib_provider(target, [SourceGroup("stocks", src, "*.csv")])
            validate_provider(target)
            self.assertGreaterEqual(summary.instruments, 2)
            # The NaN-volume row's close must be present in the provider binary.
            import numpy as np
            close_bin = target / "features" / "sz300212" / "close.day.bin"
            if not close_bin.exists():
                close_bin = target / "features" / "300212" / "close.day.bin"
            values = np.fromfile(close_bin, dtype="<f4")[1:]
            self.assertTrue(np.isclose(values, 175.89).any())


class DataLayerNanVolumeTests(unittest.TestCase):
    def test_data_layer_loads_nan_volume_csv(self) -> None:
        from unittest import mock
        from assistant import data_layer, research

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            stocks = root / "stocks"
            stocks.mkdir()
            _frame(volume=[31373200.0, None, None],
                   turnover=[0.0449, None, None]).to_csv(stocks / "300212.csv", index=False)
            with mock.patch.object(research, "DATA_DIR", root), mock.patch.object(
                data_layer, "_data_file", return_value=stocks / "300212.csv"
            ):
                frame, info = data_layer._load_asset("stock", "300212")
            self.assertEqual(len(frame), 3)
            self.assertTrue(pd.isna(frame["volume"].iloc[1]))


if __name__ == "__main__":
    unittest.main()
