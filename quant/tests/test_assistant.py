from __future__ import annotations

import copy
import json
import os
import pathlib
import sys
import tempfile
import unittest
from contextlib import ExitStack
from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from assistant import notifications, paths, research
from assistant.config import DEFAULTS
from assistant.risk import enforce_target_weights


class RiskConstraintTests(unittest.TestCase):
    def setUp(self) -> None:
        self.risk = copy.deepcopy(DEFAULTS["risk"])

    def test_accepts_weights_at_all_hard_limits(self) -> None:
        result = enforce_target_weights(
            requested := {
                "600519": 0.20,
                "000001": 0.20,
                "510300": 0.20,
                "110020": 0.20,
            },
            {
                "600519": "stock",
                "000001": "stock",
                "510300": "etf",
                "110020": "otc_fund",
            },
            self.risk,
            previous=requested,
        )

        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["cash_weight"], 0.20)
        self.assertAlmostEqual(result["turnover"], 0.0)

    def test_rejects_single_weight_category_cash_and_turnover_breaches(self) -> None:
        cases = [
            (
                {"600519": 0.21},
                {"600519": "stock"},
                None,
                "max_single_weight",
            ),
            (
                {"600519": 0.20, "000001": 0.20, "300750": 0.01},
                {"600519": "stock", "000001": "stock", "300750": "stock"},
                None,
                "stock_exposure",
            ),
            (
                {
                    "600519": 0.20,
                    "000001": 0.20,
                    "510300": 0.20,
                    "159915": 0.20,
                    "110020": 0.01,
                },
                {
                    "600519": "stock",
                    "000001": "stock",
                    "510300": "etf",
                    "159915": "etf",
                    "110020": "otc_fund",
                },
                None,
                "min_cash_weight",
            ),
            (
                {"600519": 0.20, "600036": 0.20},
                {"600519": "stock", "600036": "stock"},
                {"000001": 0.20, "300750": 0.20},
                "max_daily_turnover",
            ),
        ]

        for requested, asset_types, previous, expected_reason in cases:
            with self.subTest(expected_reason=expected_reason):
                result = enforce_target_weights(
                    requested,
                    asset_types,
                    self.risk,
                    previous=previous,
                )
                self.assertFalse(result["passed"])
                self.assertIn(expected_reason, result["failure_reasons"])
                self.assertEqual(result["target_weights"], {})
                self.assertEqual(result["cash_weight"], 1.0)

    def test_turnover_counts_cash_weight_change(self) -> None:
        risk = {**self.risk, "max_daily_turnover": 1.0}

        result = enforce_target_weights(
            {"600519": 0.20},
            {"600519": "stock"},
            risk,
            previous={},
        )

        self.assertAlmostEqual(result["turnover"], 0.20)


class QlibLifecycleTests(unittest.TestCase):
    def test_model_fit_closes_recorder_even_when_fit_fails(self) -> None:
        model = unittest.mock.Mock()
        dataset = object()
        recorder = unittest.mock.Mock()

        research._fit_qlib_model_and_close_recorder(model, dataset, recorder)
        model.fit.assert_called_once_with(dataset)
        recorder.end_exp.assert_called_once_with()

        model.reset_mock()
        recorder.reset_mock()
        model.fit.side_effect = RuntimeError("fit failed")

        with self.assertRaisesRegex(RuntimeError, "fit failed"):
            research._fit_qlib_model_and_close_recorder(model, dataset, recorder)

        model.fit.assert_called_once_with(dataset)
        recorder.end_exp.assert_called_once_with()


class AssistantStateMachineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            prefix="assistant-",
            dir=ROOT / "tests",
        )
        self.temp_root = pathlib.Path(self.temporary.name)
        self.data_dir = self.temp_root / "data"
        self.results_dir = self.temp_root / "results" / "assistant"
        self.runs_dir = self.results_dir / "runs"
        self.logs_dir = self.temp_root / "logs"
        self.provider_dir = self.temp_root / "qlib_data" / "akshare_data"
        self.lock_path = self.logs_dir / "research.lock"
        self.notifications_path = self.logs_dir / "notifications.json"
        for directory in (
            self.data_dir / "stocks",
            self.data_dir / "funds",
            self.results_dir,
            self.logs_dir,
            self.provider_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        self.config = copy.deepcopy(DEFAULTS)
        self.config["paths"] = {
            "data_dir": str(self.data_dir),
            "qlib_provider": str(self.provider_dir),
            "results_dir": str(self.results_dir),
            "logs_dir": str(self.logs_dir),
        }
        self.config["fetch"]["enabled"] = False
        self.config["fetch"]["max_data_age_days"] = 100000
        self.config["fetch"]["state_db"] = str(self.temp_root / "fetch_state.sqlite3")
        self.config["fetch"]["worker_stalled_seconds"] = 60

        self.patches = ExitStack()
        for module, name, value in (
            (research, "DATA_DIR", self.data_dir),
            (research, "ASSISTANT_RESULTS_DIR", self.results_dir),
            (research, "RUNS_DIR", self.runs_dir),
            (research, "LOCK_PATH", self.lock_path),
            (research, "QLIB_PROVIDER_DIR", self.provider_dir),
            (paths, "RUNS_DIR", self.runs_dir),
            (notifications, "NOTIFICATIONS_PATH", self.notifications_path),
        ):
            self.patches.enter_context(patch.object(module, name, value))
        self.patches.enter_context(
            patch.object(research, "load_config", return_value=self.config)
        )
        self.patches.enter_context(
            patch.object(
                research,
                "_now",
                return_value=datetime(2026, 8, 25, 10, 0, tzinfo=timezone.utc),
            )
        )

    def tearDown(self) -> None:
        self.patches.close()
        self.temporary.cleanup()

    def _write_market_data(
        self,
        asset_type: str,
        symbol: str,
        *,
        end: str = "2026-08-25",
        rising: bool = True,
    ) -> pathlib.Path:
        dates = pd.bdate_range(end=end, periods=300)
        close = np.linspace(10.0, 20.0, len(dates))
        if not rising:
            close = close[::-1]
        if asset_type == "stock":
            path = self.data_dir / "stocks" / f"{symbol}.csv"
        elif asset_type == "etf":
            path = self.data_dir / "funds" / f"etf_{symbol}.csv"
        elif asset_type == "otc_fund":
            path = self.data_dir / "funds" / f"fund_{symbol}.csv"
        else:
            raise AssertionError(asset_type)
        if asset_type == "otc_fund":
            frame = pd.DataFrame({"date": dates, "close": close})
        else:
            frame = pd.DataFrame(
                {
                    "date": dates,
                    "open": close * 0.99,
                    "high": close * 1.01,
                    "low": close * 0.98,
                    "close": close,
                    "volume": np.arange(len(dates)) + 1000,
                    "amount": close * (np.arange(len(dates)) + 1000),
                    "turnover": 1.0,
                }
            )
        frame.to_csv(path, index=False)
        return path

    def _write_sentinel_current(self) -> dict[str, object]:
        sentinel = {
            "run_id": "sentinel-success",
            "status": "success",
            "decision_support": {"target_weights": {"OLD": 0.10}},
        }
        paths.atomic_write_json(self.results_dir / "current.json", sentinel)
        return sentinel

    def _assert_current_unchanged(self, sentinel: dict[str, object]) -> None:
        with (self.results_dir / "current.json").open("r", encoding="utf-8") as handle:
            self.assertEqual(json.load(handle), sentinel)

    def test_partitions_explicit_symbols_across_fixed_asset_files(self) -> None:
        self._write_market_data("stock", "600519")
        self._write_market_data("etf", "510300")
        self._write_market_data("otc_fund", "110020")

        routed = research._partition_symbols(
            research.ASSET_TYPES,
            ["SH600519", "510300", "110020"],
        )

        self.assertEqual(routed["stock"], ["600519"])
        self.assertEqual(routed["etf"], ["510300"])
        self.assertEqual(routed["otc_fund"], ["110020"])
        with self.assertRaisesRegex(ValueError, "symbols not found"):
            research._partition_symbols(research.ASSET_TYPES, ["999999"])

    def test_limited_universe_manifest_is_not_authoritative(self) -> None:
        limited = {
            "complete": False,
            "stock_universe_complete": False,
            "etf_universe_complete": True,
        }
        self.assertFalse(research._authoritative_manifest(limited, "stock"))
        self.assertTrue(research._authoritative_manifest(limited, "etf"))

    def test_stock_data_fact_records_hfq_adjustment(self) -> None:
        self._write_market_data("stock", "600519")

        _, fact = research._load_asset("stock", "600519")

        self.assertEqual(fact["adjustment"], "hfq")
        self.assertEqual(fact["quality_status"], "pass")

    def test_stale_stock_data_skips_model_and_preserves_current(self) -> None:
        self._write_market_data("stock", "600519", end="2000-01-03")
        self.config["fetch"]["max_data_age_days"] = 0
        sentinel = self._write_sentinel_current()

        with patch.object(
            research,
            "_run_id",
            return_value="20260825T100000Z-00000001",
        ), patch.object(research, "_run_qlib_stock_model") as model:
            result = research.run_research("on_demand", "stock", ["600519"])

        self.assertEqual(result["status"], "failed")
        self.assertIn("data_quality_or_refresh_failed", result["failure_reasons"])
        self.assertEqual(result["decision_support"]["target_weights"], {})
        self.assertEqual(result["decision_support"]["cash_weight"], 1.0)
        model.assert_not_called()
        self._assert_current_unchanged(sentinel)

    def test_refresh_failure_skips_model_and_preserves_current(self) -> None:
        self._write_market_data("stock", "600519")
        self.config["fetch"]["enabled"] = True
        sentinel = self._write_sentinel_current()
        refresh = {
            "attempted": True,
            "passed": False,
            "results": [{"label": "stock", "status": "failed"}],
        }

        with patch.object(
            research,
            "_run_id",
            return_value="20260825T100000Z-00000002",
        ), patch.object(
            research,
            "_refresh_data",
            return_value=refresh,
        ), patch.object(
            research,
            "_run_qlib_stock_model",
        ) as model:
            result = research.run_research(
                "daily",
                "stock",
                ["600519"],
                _force_sync=True,
            )

        self.assertEqual(result["status"], "failed")
        self.assertIn("data_quality_or_refresh_failed", result["failure_reasons"])
        self.assertIn("stock_model_or_risk_gate_failed", result["failure_reasons"])
        model.assert_not_called()
        self._assert_current_unchanged(sentinel)

    def test_unavailable_stock_model_is_cash_only_and_preserves_current(self) -> None:
        self._write_market_data("stock", "600519")
        sentinel = self._write_sentinel_current()
        unavailable = {
            "model_available": False,
            "model_name": "Qlib Alpha158 + LightGBM",
            "error": "test model unavailable",
            "test_rank_ic": float("nan"),
            "rank_ic_ir": float("nan"),
            "net_annualized_return": float("nan"),
            "max_drawdown": float("nan"),
            "leakage_check": {"passed": False, "reason": "test"},
        }

        with patch.object(
            research,
            "_run_id",
            return_value="20260825T100000Z-00000003",
        ), patch.object(
            research,
            "_run_qlib_stock_model",
            return_value=unavailable,
        ):
            result = research.run_research("on_demand", "stock", ["600519"])

        stock = result["decision_support"]["asset_paths"]["stock"]
        self.assertEqual(result["status"], "failed")
        self.assertEqual(stock["status"], "cash_or_observe")
        self.assertEqual(stock["target_weights"], {})
        self.assertEqual(stock["cash_weight"], 1.0)
        self.assertFalse(result["rule_gates"]["stock"]["passed"])
        self._assert_current_unchanged(sentinel)
        status = research.get_run_status(result["run_id"])
        self.assertEqual(status["status"], "failed")

    def test_etf_path_can_publish_without_stock_model(self) -> None:
        self._write_market_data("etf", "510300")

        with patch.object(
            research,
            "_run_id",
            return_value="20260825T100000Z-00000004",
        ), patch.object(research, "_run_qlib_stock_model") as model:
            result = research.run_research("on_demand", "etf", ["510300"])

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["decision_support"]["target_weights"], {"510300": 0.2})
        self.assertAlmostEqual(result["decision_support"]["cash_weight"], 0.8)
        model.assert_not_called()
        snapshot = research.get_current_snapshot("etf", "SH510300")
        self.assertEqual(snapshot["run_id"], result["run_id"])
        self.assertEqual(len(snapshot["signals"]), 1)
        self.assertEqual(snapshot["signals"][0]["symbol"], "510300")

        missing = research.get_current_snapshot("stock", "600999")
        self.assertEqual(missing["status"], "not_found_or_stale")
        self.assertEqual(missing["requested_asset_type"], "stock")
        self.assertEqual(missing["requested_symbol"], "600999")
        self.assertEqual(missing["signals"], [])
        self.assertNotIn("decision_support", missing)

    def test_run_status_is_compact_and_detects_stalled_worker(self) -> None:
        run_id = "20260825T095900Z-55555555"
        state = research._fetch_state(self.config)
        state.register_assets("stock", ["600519"])
        state.create_job(
            run_id,
            kind="research",
            asset_scope="stock",
            metadata={
                "asset_types": ["stock"],
                "symbols": {"stock": ["600519"]},
                "target_date": "2026-08-25",
                "target_dates": {"stock": "2026-08-25"},
            },
        )
        state.update_job(run_id, status="running")
        run_dir = (
            self.runs_dir
            / run_id[:4]
            / run_id[4:6]
            / run_id[6:8]
            / run_id
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        paths.atomic_write_json(
            run_dir / "status.json",
            {
                "run_id": run_id,
                "status": "running",
                "phase": "fetch",
                "worker_pid": os.getpid(),
                "last_progress_at": "2026-08-25T09:58:00+00:00",
                "last_heartbeat_at": "2026-08-25T09:58:00+00:00",
            },
        )
        with patch.object(research, "_process_exists", return_value=True):
            status = research.get_run_status(run_id)
        self.assertEqual(status["status"], "stalled")
        self.assertEqual(status["phase"], "fetch")
        self.assertNotIn("job", status)
        self.assertNotIn("progress", status)
        self.assertNotIn("symbols", json.dumps(status))

    def test_run_status_marks_exited_worker_failed(self) -> None:
        run_id = "20260825T095900Z-66666666"
        state = research._fetch_state(self.config)
        state.register_assets("stock", ["600519"])
        state.create_job(
            run_id,
            kind="research",
            asset_scope="stock",
            metadata={
                "asset_types": ["stock"],
                "symbols": {"stock": ["600519"]},
                "target_date": "2026-08-25",
                "target_dates": {"stock": "2026-08-25"},
            },
        )
        state.update_job(run_id, status="running")
        run_dir = (
            self.runs_dir
            / run_id[:4]
            / run_id[4:6]
            / run_id[6:8]
            / run_id
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        paths.atomic_write_json(
            run_dir / "status.json",
            {
                "run_id": run_id,
                "status": "running",
                "phase": "fetch",
                "worker_pid": 999999999,
                "last_heartbeat_at": "2026-08-25T10:00:00+00:00",
            },
        )
        with patch.object(research, "_process_exists", return_value=False):
            status = research.get_run_status(run_id)
        self.assertEqual(status["status"], "failed")
        self.assertIn("worker exited", status["error"])
        self.assertEqual(state.get_job(run_id)["status"], "failed")

    def test_active_lock_is_rejected_and_dead_lock_is_recovered(self) -> None:
        active = {
            "run_id": "20260825T100000Z-11111111",
            "pid": os.getpid(),
            "created_at": "2026-08-25T10:00:00+00:00",
        }
        self.lock_path.write_text(json.dumps(active), encoding="utf-8")

        with self.assertRaisesRegex(RuntimeError, "another research run is active"):
            research._acquire_lock("20260825T100000Z-22222222", 3600)

        dead = {**active, "pid": 999999999}
        self.lock_path.write_text(json.dumps(dead), encoding="utf-8")
        with patch.object(research, "_process_exists", return_value=False):
            descriptor = research._acquire_lock(
                "20260825T100000Z-33333333",
                3600,
            )
        research._release_lock(descriptor, "20260825T100000Z-33333333")
        self.assertFalse(self.lock_path.exists())

    def test_notification_query_and_acknowledgement(self) -> None:
        record = notifications.add_notification(
            "research_success",
            "完成",
            "测试通知",
            run_id="20260825T100000Z-44444444",
        )

        unread = research.list_notifications(unread_only=True)
        self.assertEqual(len(unread["notifications"]), 1)
        acknowledged = research.ack_notification(record["notification_id"])
        self.assertTrue(acknowledged["ok"])
        self.assertEqual(
            research.list_notifications(unread_only=True)["notifications"],
            [],
        )
        with self.assertRaisesRegex(ValueError, "invalid notification_id"):
            research.ack_notification("../notifications.json")

    def test_notification_queries_are_bounded_and_filterable(self) -> None:
        first_run = "20260825T100000Z-77777777"
        second_run = "20260825T100000Z-88888888"
        records = [
            notifications.add_notification(
                "research_success",
                f"通知 {index}",
                "测试分页",
                run_id=first_run if index % 2 == 0 else second_run,
            )
            for index in range(12)
        ]
        notifications.acknowledge(records[-1]["notification_id"])

        default_page = research.list_notifications()
        self.assertEqual(default_page["count"], 10)
        self.assertEqual(default_page["notifications"][0]["title"], "通知 11")
        limited = research.list_notifications(limit=3)
        self.assertEqual(limited["count"], 3)
        filtered = research.list_notifications(run_id=first_run, limit=100)
        self.assertEqual(filtered["count"], 6)
        self.assertTrue(
            all(item["run_id"] == first_run for item in filtered["notifications"])
        )
        since = research.list_notifications(
            since=records[-2]["created_at"],
            limit=100,
        )
        self.assertEqual(since["count"], 2)
        unread = research.list_notifications(unread_only=True, limit=100)
        self.assertEqual(unread["count"], 11)
        with self.assertRaisesRegex(ValueError, "ISO-8601"):
            research.list_notifications(since="not-a-date")

    def test_targeted_queue_is_idempotent_and_distinct_from_full_universe(self) -> None:
        symbols = ["600999", "601696", "600030", "000776", "601688", "601211"]
        with patch.object(research, "_universe", return_value=["600000", "600001"]):
            full = research._queue_research(
                mode="daily",
                asset_scope="stock",
                asset_types=("stock",),
                config=self.config,
                target_date="2026-08-25",
                start_background_worker=False,
            )
        targeted = research._queue_research(
            mode="daily",
            asset_scope="stock",
            asset_types=("stock",),
            config=self.config,
            symbols_by_type={"stock": symbols},
            target_date="2026-08-25",
            start_background_worker=False,
        )
        repeated = research._queue_research(
            mode="daily",
            asset_scope="stock",
            asset_types=("stock",),
            config=self.config,
            symbols_by_type={"stock": symbols},
            target_date="2026-08-25",
            start_background_worker=False,
        )

        self.assertNotEqual(full["run_id"], targeted["run_id"])
        self.assertEqual(targeted["run_id"], repeated["run_id"])
        jobs = research._fetch_state(self.config).list_jobs(limit=10)
        self.assertEqual(len(jobs), 2)
        targeted_job = next(job for job in jobs if job["job_id"] == targeted["run_id"])
        self.assertFalse(targeted_job["metadata"]["full_universe"])
        self.assertEqual(
            targeted_job["metadata"]["symbols"],
            {"stock": sorted(symbols)},
        )


if __name__ == "__main__":
    unittest.main()
