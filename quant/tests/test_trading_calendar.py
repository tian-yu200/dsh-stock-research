from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest
from datetime import date
from unittest.mock import patch

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import trading_calendar


class TradingCalendarTests(unittest.TestCase):
    def test_online_calendar_is_cached_and_marks_month_end(self) -> None:
        calendar = pd.DataFrame(
            {"trade_date": ["2026-08-25", "2026-08-26", "2026-08-31"]}
        )
        with tempfile.TemporaryDirectory(prefix="calendar-", dir=ROOT / "tests") as directory:
            cache = pathlib.Path(directory) / "trade_calendar.csv"
            with patch.object(
                trading_calendar,
                "_online_calendar",
                return_value=calendar,
            ):
                result = trading_calendar.resolve_trading_day(
                    date(2026, 8, 31),
                    cache,
                )

            self.assertEqual(result["source"], "akshare:tool_trade_date_hist_sina")
            self.assertTrue(result["is_trading_day"])
            self.assertTrue(result["is_month_end_trading_day"])
            self.assertTrue(cache.is_file())

    def test_online_failure_uses_only_local_cache(self) -> None:
        with tempfile.TemporaryDirectory(prefix="calendar-", dir=ROOT / "tests") as directory:
            cache = pathlib.Path(directory) / "trade_calendar.csv"
            pd.DataFrame(
                {"trade_date": ["2026-08-25", "2026-08-26", "2026-08-31"]}
            ).to_csv(cache, index=False)

            with patch.object(
                trading_calendar,
                "_online_calendar",
                side_effect=RuntimeError("offline"),
            ):
                result = trading_calendar.resolve_trading_day(
                    date(2026, 8, 26),
                    cache,
                )

            self.assertEqual(result["source"], "local_cache")
            self.assertTrue(result["is_trading_day"])
            self.assertFalse(result["is_month_end_trading_day"])
            self.assertIn("RuntimeError: offline", result["online_error"])


if __name__ == "__main__":
    unittest.main()
