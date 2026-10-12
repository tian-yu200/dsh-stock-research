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


def _otc_frame(dates, navs=None):
    navs = navs or [1.0 + 0.01 * i for i in range(len(dates))]
    return pd.DataFrame({"净值日期": dates, "单位净值": navs})


class OtcFundTests(unittest.TestCase):
    """A1: the guard must actually apply to OTC raw frames (净值日期/单位净值)."""

    def test_otc_today_nav_dropped_before_cutoff(self):
        from unittest import mock
        from fetch_utils import merge_otc_frames
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "otc_110020.csv"
            pd.DataFrame({"date": [PREV], "close": [1.0]}).to_csv(path, index=False)
            with mock.patch("fetch_utils._china_now", return_value=_cst(2026, 10, 12, 10, 8)):
                merged = merge_otc_frames(path, _otc_frame([TODAY], [1.5]), source="otc")
            self.assertEqual(merged["date"].astype(str).tolist(), [PREV])

    def test_otc_today_nav_kept_after_cutoff(self):
        from unittest import mock
        from fetch_utils import merge_otc_frames
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "otc_110020.csv"
            pd.DataFrame({"date": [PREV], "close": [1.0]}).to_csv(path, index=False)
            with mock.patch("fetch_utils._china_now", return_value=_cst(2026, 10, 12, 16, 30)):
                merged = merge_otc_frames(path, _otc_frame([TODAY], [1.5]), source="otc")
            self.assertEqual(merged["date"].astype(str).tolist(), [PREV, TODAY])

    def test_otc_future_nav_always_dropped(self):
        from unittest import mock
        from fetch_utils import merge_otc_frames
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "otc_110020.csv"
            pd.DataFrame({"date": [PREV], "close": [1.0]}).to_csv(path, index=False)
            with mock.patch("fetch_utils._china_now", return_value=_cst(2026, 10, 12, 16, 30)):
                merged = merge_otc_frames(path, _otc_frame(["2026-10-13"], [1.5]), source="otc")
            self.assertEqual(merged["date"].astype(str).tolist(), [PREV])


class FundMetadataTests(unittest.TestCase):
    """A3: the fund writer must record dropped bars in .meta.json."""

    def test_etf_metadata_records_dropped_anomaly(self):
        from unittest import mock
        from fetch_fund_data import _write_success
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "etf_510300.csv"
            _frame([PREV]).to_csv(path, index=False)
            with mock.patch("fetch_utils._china_now", return_value=_cst(2026, 10, 12, 10, 8)):
                metadata = _write_success("etf", "510300", _frame([TODAY]), path)
            self.assertIn("field_anomalies", metadata)
            self.assertEqual(
                metadata["field_anomalies"]["intraday_bar_dropped"]["dropped_rows"], 1
            )

    def test_otc_metadata_records_dropped_anomaly(self):
        from unittest import mock
        from fetch_fund_data import _write_success
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "otc_110020.csv"
            pd.DataFrame({"date": [PREV], "close": [1.0]}).to_csv(path, index=False)
            with mock.patch("fetch_utils._china_now", return_value=_cst(2026, 10, 12, 10, 8)):
                metadata = _write_success("otc_fund", "110020", _otc_frame([TODAY], [1.5]), path)
            self.assertIn("field_anomalies", metadata)
            self.assertEqual(
                metadata["field_anomalies"]["intraday_bar_dropped"]["dropped_rows"], 1
            )


class GuardNeverRaisesTests(unittest.TestCase):
    """B1: the guard degrades, never raises; unparseable dates pass through."""

    def test_all_garbage_dates_do_not_raise(self):
        from fetch_utils import drop_unfinished_bars
        frame = pd.DataFrame({"date": ["garbage", "junk"], "close": [1.0, 2.0]})
        out, dropped = drop_unfinished_bars(frame, now=_cst(2026, 10, 12, 10, 8))
        self.assertEqual(dropped, 0)
        self.assertEqual(out["date"].tolist(), ["garbage", "junk"])

    def test_mixed_frame_keeps_unparseable_row(self):
        from fetch_utils import drop_unfinished_bars
        frame = pd.DataFrame(
            {"date": [PREV, "garbage", TODAY], "close": [1.0, 2.0, 3.0]}
        )
        out, dropped = drop_unfinished_bars(frame, now=_cst(2026, 10, 12, 10, 8))
        self.assertEqual(dropped, 1)
        self.assertEqual(out["date"].tolist(), [PREV, "garbage"])

    def test_batch_otc_frame_dropped_before_cutoff(self):
        # batch path builds DataFrame(..., columns=["净值日期","单位净值"]) and
        # calls the same _write_success -> merge_otc_frames choke point.
        from unittest import mock
        from fetch_fund_data import _write_success
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "otc_110020.csv"
            pd.DataFrame({"date": [PREV], "close": [1.0]}).to_csv(path, index=False)
            frame = pd.DataFrame(
                [(TODAY, 1.5)], columns=["净值日期", "单位净值"]
            )
            frame.attrs["adjustment"] = "unit_nav"
            with mock.patch("fetch_utils._china_now", return_value=_cst(2026, 10, 12, 10, 8)):
                metadata = _write_success("otc_fund", "110020", frame, path)
            self.assertEqual(metadata["data_end_date"], PREV)
