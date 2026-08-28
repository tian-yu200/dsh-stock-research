from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from assistant import research
from assistant.config import DEFAULTS
from assistant.risk import model_gate_checks


class ResearchMetricTests(unittest.TestCase):
    def test_drawdown_reports_true_maximum_and_recovery(self) -> None:
        dates = pd.to_datetime(
            ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-08"]
        )
        close = pd.Series([100.0, 120.0, 90.0, 121.0], index=dates)

        result = research._drawdown_recovery(close)

        self.assertAlmostEqual(result["max_drawdown"], -0.25)
        self.assertEqual(result["max_drawdown_peak_date"], "2024-01-03")
        self.assertEqual(result["max_drawdown_trough_date"], "2024-01-04")
        self.assertEqual(result["max_drawdown_recovery_date"], "2024-01-08")
        self.assertEqual(result["max_drawdown_recovery_days"], 4)
        self.assertTrue(result["current_drawdown_recovered"])
        self.assertEqual(result["current_drawdown_duration_days"], 0)

    def test_market_metrics_use_window_max_drawdown_not_current_drawdown(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=5)
        frame = pd.DataFrame(
            {
                "date": dates,
                "close": [100.0, 120.0, 90.0, 110.0, 115.0],
            }
        )

        result = research._market_metrics(frame, fund=True)

        self.assertAlmostEqual(result["max_drawdown_252d"], -0.25)
        self.assertAlmostEqual(result["current_drawdown"], 115.0 / 120.0 - 1.0)
        self.assertFalse(result["current_drawdown_recovered"])
        self.assertEqual(result["series_type"], "net_value")

    def test_relative_returns_and_etf_liquidity_are_explicit(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=30)
        asset = pd.Series(np.linspace(100.0, 130.0, len(dates)), index=dates)
        benchmark = pd.Series(np.linspace(100.0, 115.0, len(dates)), index=dates)
        frame = pd.DataFrame(
            {
                "date": dates,
                "close": asset.to_numpy(),
                "volume": np.arange(30, dtype=float) + 100.0,
                "amount": np.arange(30, dtype=float) + 1000.0,
            }
        )

        relative = research._relative_returns(asset, benchmark)
        liquidity = research._liquidity_metrics(frame)

        self.assertGreater(relative["relative_return_20d_vs_benchmark"], 0.0)
        self.assertIsNone(relative["relative_return_60d_vs_benchmark"])
        self.assertEqual(liquidity["liquidity_status"], "available")
        self.assertEqual(liquidity["amount_data_status"], "available")
        self.assertAlmostEqual(
            liquidity["average_volume_20d"],
            frame["volume"].tail(20).mean(),
        )

    def test_forward_groups_measure_top_bottom_spread(self) -> None:
        index = pd.MultiIndex.from_product(
            [
                pd.to_datetime(["2024-01-02", "2024-01-03"]),
                ["A", "B", "C", "D"],
            ],
            names=["datetime", "instrument"],
        )
        prediction = pd.Series([1.0, 2.0, 3.0, 4.0] * 2, index=index)
        labels = pd.Series([0.01, 0.02, 0.03, 0.04] * 2, index=index)

        result = research._forward_group_metrics(
            prediction,
            labels,
            group_count=2,
        )

        self.assertEqual(result["valid_days"], 2)
        self.assertAlmostEqual(result["bottom_group_return"], 0.015)
        self.assertAlmostEqual(result["top_group_return"], 0.035)
        self.assertAlmostEqual(result["top_bottom_excess_return"], 0.02)

    def test_model_gate_uses_new_metric_contract(self) -> None:
        metrics = {
            "model_available": True,
            "mean_rank_ic": 0.03,
            "rank_ic_ir": 0.11,
            "top_bottom_excess_return": 0.001,
            "test_max_drawdown": -0.20,
        }

        self.assertTrue(all(model_gate_checks(metrics, DEFAULTS["gate"]).values()))
        old_contract = {
            "model_available": True,
            "test_rank_ic": 0.03,
            "rank_ic_ir": 0.11,
            "max_drawdown": -0.20,
        }
        self.assertFalse(all(model_gate_checks(old_contract, DEFAULTS["gate"]).values()))

    def test_otc_fund_analysis_compares_same_category_when_peers_exist(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=70)
        frame = pd.DataFrame(
            {
                "date": dates,
                "close": np.linspace(1.0, 1.2, len(dates)),
            }
        )
        peer = pd.DataFrame(
            {
                "date": dates,
                "close": np.linspace(1.0, 1.1, len(dates)),
            }
        )
        facts = [
            {"symbol": "110020", "data_end_date": dates[-1].date().isoformat()},
            {"symbol": "001594", "data_end_date": dates[-1].date().isoformat()},
        ]
        benchmark = pd.Series(
            np.linspace(100.0, 110.0, len(dates)),
            index=dates,
        )

        result = research._technical_asset_analysis(
            "otc_fund",
            {"110020": frame, "001594": peer},
            facts,
            benchmark,
        )
        signal = result["signals"][0]

        self.assertEqual(signal["peer_comparison_status"], "available")
        self.assertEqual(signal["peer_comparison_period_days"], 60)
        self.assertEqual(signal["peer_count"], 1)
        self.assertGreater(signal["excess_return_60d_vs_peer_mean"], 0.0)
        self.assertEqual(signal["benchmark_data_status"], "available")
        self.assertIn("max_drawdown_recovery_days", signal)


if __name__ == "__main__":
    unittest.main()
