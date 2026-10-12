# -*- coding: utf-8 -*-
"""T15: the fetch path must not persist an in-progress intraday daily bar.

2026-10-12 incident: the East Money hist endpoint returned the still-forming
bar for the current day, which was written as a daily bar (89 stock CSVs,
10:07-10:08 CST). Once written it is permanent (postclose only forces
data_end < today; the next incremental fetch starts at data_end + 1).

Rule (boss): drop rows dated after today (China date, UTC+8) always; drop
rows dated today when local China time is before the 16:00 close cutoff.
"""
from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

CHINA_TZ = timezone(timedelta(hours=8))
TODAY = "2026-10-12"
PREV = "2026-10-09"


def _frame(dates, closes=None):
    closes = closes or [10.0 + i for i in range(len(dates))]
    return pd.DataFrame(
        {
            "date": dates,
            "open": closes,
            "high": [c + 0.1 for c in closes],
            "low": [c - 0.1 for c in closes],
            "close": closes,
            "volume": [1000.0] * len(dates),
            "amount": [10000.0] * len(dates),
            "turnover": [0.01] * len(dates),
        }
    )


def _cst(y, m, d, hh, mm):
    return datetime(y, m, d, hh, mm, tzinfo=CHINA_TZ)


class DropInProgressBarTests(unittest.TestCase):
    """The shared helper drops future/today-before-cutoff rows."""

    def test_today_row_dropped_before_cutoff(self):
        from fetch_utils import drop_unfinished_bars
        frame = _frame([PREV, TODAY])
        out, dropped = drop_unfinished_bars(frame, now=_cst(2026, 10, 12, 10, 8))
        self.assertEqual(out["date"].astype(str).tolist(), [PREV])
        self.assertEqual(dropped, 1)

    def test_today_row_kept_after_cutoff(self):
        from fetch_utils import drop_unfinished_bars
        frame = _frame([PREV, TODAY])
        out, dropped = drop_unfinished_bars(frame, now=_cst(2026, 10, 12, 16, 30))
        self.assertEqual(out["date"].astype(str).tolist(), [PREV, TODAY])
        self.assertEqual(dropped, 0)

    def test_future_row_always_dropped(self):
        from fetch_utils import drop_unfinished_bars
        frame = _frame([PREV, TODAY, "2026-10-13"])
        out, dropped = drop_unfinished_bars(frame, now=_cst(2026, 10, 12, 16, 30))
        self.assertEqual(out["date"].astype(str).tolist(), [PREV, TODAY])
        self.assertEqual(dropped, 1)


class StockWriterTests(unittest.TestCase):
    def test_stock_write_drops_today_before_cutoff_and_keeps_data_end(self):
        from unittest import mock
        from fetch_stock_data import _write_success
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "600519.csv"
            existing = _frame([PREV])
            existing.to_csv(path, index=False)
            incoming = _frame([TODAY])
            with mock.patch("fetch_utils._china_now", return_value=_cst(2026, 10, 12, 10, 8)):
                metadata = _write_success("600519", incoming, path)
            written = pd.read_csv(path)
            self.assertEqual(written["date"].astype(str).tolist(), [PREV])
            self.assertEqual(metadata["data_end_date"], PREV)

    def test_stock_write_keeps_today_after_cutoff(self):
        from unittest import mock
        from fetch_stock_data import _write_success
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "600519.csv"
            _frame([PREV]).to_csv(path, index=False)
            with mock.patch("fetch_utils._china_now", return_value=_cst(2026, 10, 12, 16, 30)):
                metadata = _write_success("600519", _frame([TODAY]), path)
            self.assertEqual(metadata["data_end_date"], TODAY)


class FundWriterTests(unittest.TestCase):
    def test_etf_write_drops_today_before_cutoff(self):
        from unittest import mock
        from fetch_fund_data import _write_success
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "etf_510300.csv"
            _frame([PREV]).to_csv(path, index=False)
            with mock.patch("fetch_utils._china_now", return_value=_cst(2026, 10, 12, 10, 8)):
                metadata = _write_success("etf", "510300", _frame([TODAY]), path)
            written = pd.read_csv(path)
            self.assertEqual(written["date"].astype(str).tolist(), [PREV])
            self.assertEqual(metadata["data_end_date"], PREV)


class ThsPatchChokePointTests(unittest.TestCase):
    def test_ths_patched_today_row_dropped_before_cutoff(self):
        # ths_patch_recent (data_source) can only rewrite rows already present in
        # the frame; that frame still flows through merge_daily_frames. Simulate a
        # THS-patched frame that carries today's row and assert the choke point
        # drops it before the cutoff (so it cannot be re-introduced).
        from unittest import mock
        from fetch_utils import merge_daily_frames
        from market_data import normalize_daily_bars
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "600519.csv"
            _frame([PREV]).to_csv(path, index=False)
            patched = _frame([PREV, TODAY])
            patched.loc[patched.index[-1], "close"] = 12.34
            with mock.patch("fetch_utils._china_now", return_value=_cst(2026, 10, 12, 10, 8)):
                merged = merge_daily_frames(
                    path, normalize_daily_bars(patched, source="ths"), source="ths"
                )
            self.assertEqual(merged["date"].astype(str).tolist(), [PREV])
            self.assertNotIn(12.34, merged["close"].tolist())
