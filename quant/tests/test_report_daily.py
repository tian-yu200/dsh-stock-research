"""Focused tests for the thin daily dashboard renderer (no network/Qlib)."""
from __future__ import annotations

import pathlib
import sys
import unittest
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from assistant import report_daily


def _current() -> dict:
    return {
        "run_id": "20260903T150136Z-fa84c8e5",
        "status": "success",
        "data_end_date": "2026-09-03",
        "decision_support": {
            "allocation_basis": "t",
            "cash_weight": 0.8,
            "hypothetical_capital": 100000.0,
            "auto_order": False,
            "broker_connected": False,
            "target_weights": {"603826": 0.2},
            "asset_paths": {"stock": {"action": "a", "status": "s", "cash_weight": 0.8, "reason": "r"}},
        },
        "rule_gates": {
            "portfolio": {"passed": True, "cash_weight": 0.8, "turnover": 0.2, "checks": {"a": True}},
            "stock": {"passed": True, "checks": {"b": True}, "model_metrics": {"mean_rank_ic": 0.09}},
        },
        "result_path": "runs/x",
        "paths": {"manifest": "m"},
    }


def _rolling(n: int = 10) -> dict:
    return {
        "coverage": {"coverage": 0.97, "success": 97, "total": 100},
        "pending_remaining": 5,
        "analyzed_this_pass": n,
        "eligible_count": n,
        "disclaimer": "prelim",
        "pool": [
            {"rank": i + 1, "symbol": f"{600000 + i}", "name": f"N{i}", "trend_score": 100.0 - i, "extra": "drop"}
            for i in range(n)
        ],
    }


class ReportDailyTests(unittest.TestCase):
    def test_summarize_checks_counts_failures(self) -> None:
        ok, total, failed = report_daily._summarize_checks({"a": True, "b": False, "c": False})
        self.assertEqual((ok, total, failed), (1, 3, ["b", "c"]))

    def test_pick_pool_top_truncates_and_slims(self) -> None:
        rolling = {
            "pool": [
                {"rank": 2, "symbol": "000001", "name": "A", "trend_score": 2.0, "extra": "drop"},
                {"rank": 1, "symbol": "600519", "name": "B", "trend_score": 9.0, "extra": "drop"},
            ]
        }
        top = report_daily._pick_pool_top(rolling, 1)
        self.assertEqual([r["symbol"] for r in top], ["600519"])
        self.assertNotIn("extra", top[0])
        self.assertLessEqual(set(top[0]), set(report_daily.POOL_FIELDS))

    def test_render_no_success_is_bounded(self) -> None:
        md, payload = report_daily.render_daily(None, None, 5, "2026-09-04T00:00:00+00:00")
        self.assertIn("no_successful_run", payload["status"])
        self.assertIn("current.json", md)
        self.assertEqual(payload["pool_top"], [])

    def test_render_targets_and_bounded_pool(self) -> None:
        md, payload = report_daily.render_daily(_current(), _rolling(10), 3, "2026-09-04T00:00:00+00:00")
        self.assertEqual(payload["target_count"], 1)
        self.assertEqual(len(payload["pool_top"]), 3)
        self.assertIn("603826", md)
        self.assertIn("Top-3", md)

    def test_build_daily_writes_fixed_paths_only(self) -> None:
        writes: dict[str, object] = {}
        with patch.object(report_daily, "read_json") as mock_read, patch.object(
            report_daily, "atomic_write_text"
        ) as mock_text, patch.object(report_daily, "atomic_write_json") as mock_json, patch.object(
            report_daily, "ensure_directories"
        ):
            mock_read.side_effect = [_current(), _rolling(4)]
            payload = report_daily.build_daily(3)
            writes["text_path"] = mock_text.call_args[0][0]
            writes["json_path"] = mock_json.call_args[0][0]
        self.assertEqual(len(payload["pool_top"]), 3)
        self.assertTrue(str(writes["text_path"]).endswith("daily.md"))
        self.assertTrue(str(writes["json_path"]).endswith("daily.json"))

    def test_main_rejects_top_n_out_of_range(self) -> None:
        self.assertEqual(report_daily.main(["--top-n", "0", "--no-notify"]), 2)
        self.assertEqual(report_daily.main(["--top-n", "31", "--no-notify"]), 2)


if __name__ == "__main__":
    unittest.main()
