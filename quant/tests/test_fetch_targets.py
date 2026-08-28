from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from fetch_targets import asset_target_dates


class FetchTargetTests(unittest.TestCase):
    def test_funds_use_previous_complete_trading_day(self) -> None:
        with tempfile.TemporaryDirectory(prefix="fetch-targets-", dir=ROOT / "tests") as directory:
            calendar_path = pathlib.Path(directory) / "trade_calendar.csv"
            pd.DataFrame(
                {
                    "trade_date": [
                        "2026-08-25",
                        "2026-08-26",
                        "2026-08-27",
                        "2026-08-28",
                    ]
                }
            ).to_csv(calendar_path, index=False)

            targets = asset_target_dates(
                ["stock", "etf", "otc_fund"],
                "2026-08-27",
                lag_trading_days={"stock": 0, "etf": 1, "otc_fund": 1},
                calendar_path=calendar_path,
            )

        self.assertEqual(
            targets,
            {
                "stock": "2026-08-27",
                "etf": "2026-08-26",
                "otc_fund": "2026-08-26",
            },
        )

    def test_weekend_uses_latest_prior_trading_day_for_one_day_lag(self) -> None:
        with tempfile.TemporaryDirectory(prefix="fetch-targets-", dir=ROOT / "tests") as directory:
            calendar_path = pathlib.Path(directory) / "trade_calendar.csv"
            pd.DataFrame(
                {"trade_date": ["2026-08-27", "2026-08-28", "2026-08-31"]}
            ).to_csv(calendar_path, index=False)

            targets = asset_target_dates(
                ["etf"],
                "2026-08-29",
                lag_trading_days={"etf": 1},
                calendar_path=calendar_path,
            )

        self.assertEqual(targets, {"etf": "2026-08-28"})


if __name__ == "__main__":
    unittest.main()
