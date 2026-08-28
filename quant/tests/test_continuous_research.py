from __future__ import annotations

import copy
import hashlib
import io
import json
import pathlib
import sys
import tempfile
import unittest
from contextlib import ExitStack
from datetime import datetime, timezone
from unittest.mock import Mock, patch


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from assistant import paths, research
from assistant.config import DEFAULTS
import continuous_research as continuous
from fetch_state import FetchState


TARGET_DATE = "2026-08-26"
JOB_ID = "20260826T100000Z-00000001"


def _digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ContinuousResearchTests(unittest.TestCase):
    def _environment(
        self,
        stack: ExitStack,
        root: pathlib.Path,
        state: FetchState,
    ) -> dict:
        continuous_dir = root / "logs" / "continuous_research"
        run_root = root / "runs"
        run_root.mkdir(parents=True, exist_ok=True)
        config = copy.deepcopy(DEFAULTS)
        config["fetch"]["state_db"] = str(state.path)
        config["fetch"]["max_data_age_days"] = 5
        config["runtime"]["lock_stale_seconds"] = 60

        stack.enter_context(patch.object(continuous, "QUANT_ROOT", root))
        stack.enter_context(patch.object(continuous, "CONTINUOUS_DIR", continuous_dir))
        stack.enter_context(
            patch.object(
                continuous,
                "LATEST_STATUS_PATH",
                continuous_dir / "latest_status.json",
            )
        )
        stack.enter_context(
            patch.object(
                continuous,
                "LOCK_PATH",
                continuous_dir / "continuous_research.lock",
            )
        )
        stack.enter_context(
            patch.object(continuous, "PAUSE_PATH", continuous_dir / "paused.json")
        )
        stack.enter_context(
            patch.object(continuous, "CONTROL_PATH", continuous_dir / "control.json")
        )
        stack.enter_context(
            patch.object(
                continuous,
                "LOG_PATH",
                continuous_dir / "continuous_research.log",
            )
        )
        stack.enter_context(
            patch.object(continuous, "ALERT_PATH", continuous_dir / "alerts.log")
        )
        stack.enter_context(
            patch.object(
                continuous,
                "WORKER_LOG_DIR",
                continuous_dir / "workers",
            )
        )
        stack.enter_context(patch.object(continuous, "load_config", return_value=config))
        stack.enter_context(
            patch.object(
                continuous,
                "_calendar_today",
                return_value={
                    "status": "ok",
                    "date": TARGET_DATE,
                    "is_trading_day": True,
                    "is_month_end_trading_day": False,
                    "source": "test",
                },
            )
        )
        stack.enter_context(
            patch.object(
                continuous,
                "run_directory",
                side_effect=lambda job_id: run_root / job_id,
            )
        )
        return {
            "config": config,
            "continuous_dir": continuous_dir,
            "run_root": run_root,
        }

    @staticmethod
    def _create_job(
        state: FetchState,
        run_root: pathlib.Path,
        *,
        status: str = "failed",
        complete: bool = False,
        job_id: str = JOB_ID,
        target_date: str = TARGET_DATE,
        asset_scope: str = "all",
    ) -> None:
        all_symbols = {
            "stock": ["000001"],
            "etf": ["510300"],
            "otc_fund": ["000001"],
        }
        symbols = (
            all_symbols
            if asset_scope == "all"
            else {asset_scope: all_symbols[asset_scope]}
        )
        for asset_type, values in symbols.items():
            state.register_assets(
                asset_type,
                values,
                data_end_dates={
                    symbol: target_date if complete else "2026-08-24"
                    for symbol in values
                },
            )
        state.create_job(
            job_id,
            kind="research",
            asset_scope=asset_scope,
            metadata={
                "mode": "daily",
                "asset_scope": asset_scope,
                "asset_types": list(symbols),
                "symbols": symbols,
                "target_date": target_date,
                "phase": "fetch",
            },
        )
        state.update_job(job_id, status=status)
        directory = run_root / job_id
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "status.json").write_text(
            json.dumps({"job_id": job_id, "status": status}),
            encoding="utf-8",
        )

    def test_timeout_retains_checkpoint_and_next_round_reuses_job(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="continuous-timeout-",
            dir=ROOT / "tests",
        ) as directory:
            root = pathlib.Path(directory)
            state = FetchState(root / "state.sqlite3")
            with ExitStack() as stack:
                environment = self._environment(stack, root, state)
                self._create_job(state, environment["run_root"])
                current_path = root / "results" / "assistant" / "current.json"
                current_path.parent.mkdir(parents=True, exist_ok=True)
                current_path.write_text(
                    json.dumps({"run_id": "previous-success", "status": "success"}),
                    encoding="utf-8",
                )
                before = _digest(current_path)

                timed_out = continuous.WorkerOutcome(
                    pid=321,
                    returncode=None,
                    timed_out=True,
                    duration_seconds=1.0,
                    error="worker wall timeout",
                )
                first = continuous.run_once(
                    worker_runner=lambda _job_id, _logger: timed_out
                )

                def advance_same_job(job_id: str, _logger: object):
                    self.assertEqual(job_id, JOB_ID)
                    state.mark_success(
                        "stock",
                        "000001",
                        data_end_date=TARGET_DATE,
                        source="test",
                        rows=10,
                        refresh_interval_seconds=86400,
                    )
                    state.update_job(JOB_ID, status="queued")
                    return continuous.WorkerOutcome(
                        pid=322,
                        returncode=0,
                        timed_out=False,
                        duration_seconds=1.0,
                    )

                second = continuous.run_once(worker_runner=advance_same_job)

            self.assertEqual(first["status"], "failed")
            self.assertTrue(first["worker_timeout"])
            self.assertTrue(first["checkpoint_retained"])
            self.assertEqual(second["run_id"], JOB_ID)
            self.assertEqual(second["status"], "partial")
            self.assertEqual(second["completed"], 1)
            self.assertEqual(_digest(current_path), before)
            self.assertTrue(
                (environment["continuous_dir"] / "latest_status.json").is_file()
            )
            self.assertTrue(
                (environment["continuous_dir"] / "continuous_research.log").is_file()
            )

    def test_old_stock_job_resumes_without_today_calendar_gate(self) -> None:
        old_target = "2026-08-25"
        with tempfile.TemporaryDirectory(
            prefix="continuous-old-stock-",
            dir=ROOT / "tests",
        ) as directory:
            root = pathlib.Path(directory)
            state = FetchState(root / "state.sqlite3")
            with ExitStack() as stack:
                environment = self._environment(stack, root, state)
                self._create_job(
                    state,
                    environment["run_root"],
                    status="queued",
                    target_date=old_target,
                    asset_scope="stock",
                )
                stack.enter_context(
                    patch.object(
                        continuous,
                        "_calendar_today",
                        side_effect=AssertionError(
                            "an existing job must not be blocked by today's calendar"
                        ),
                    )
                )
                worker = Mock(
                    return_value=continuous.WorkerOutcome(
                        pid=501,
                        returncode=0,
                        timed_out=False,
                        duration_seconds=1.0,
                    )
                )
                result = continuous.run_once(worker_runner=worker)

            worker.assert_called_once()
            self.assertEqual(worker.call_args.args[0], JOB_ID)
            self.assertEqual(result["run_id"], JOB_ID)
            self.assertEqual(result["target_date"], old_target)
            self.assertEqual(result["asset_scope"], "stock")
            self.assertEqual(result["asset_types"], ["stock"])

    def test_explicit_job_id_takes_priority_over_other_queued_jobs(self) -> None:
        selected_id = "20260825T100000Z-selected"
        other_id = "20260826T100000Z-other"
        with tempfile.TemporaryDirectory(
            prefix="continuous-explicit-",
            dir=ROOT / "tests",
        ) as directory:
            root = pathlib.Path(directory)
            state = FetchState(root / "state.sqlite3")
            with ExitStack() as stack:
                environment = self._environment(stack, root, state)
                self._create_job(
                    state,
                    environment["run_root"],
                    status="queued",
                    job_id=selected_id,
                    target_date="2026-08-25",
                    asset_scope="stock",
                )
                self._create_job(
                    state,
                    environment["run_root"],
                    status="queued",
                    job_id=other_id,
                    asset_scope="stock",
                )
                worker = Mock(
                    return_value=continuous.WorkerOutcome(
                        pid=502,
                        returncode=0,
                        timed_out=False,
                        duration_seconds=1.0,
                    )
                )
                result = continuous.run_once(
                    job_id=selected_id,
                    worker_runner=worker,
                )

            worker.assert_called_once()
            self.assertEqual(worker.call_args.args[0], selected_id)
            self.assertEqual(result["run_id"], selected_id)
            self.assertEqual(result["target_date"], "2026-08-25")

    def test_no_backlog_creates_today_full_market_job(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="continuous-new-today-",
            dir=ROOT / "tests",
        ) as directory:
            root = pathlib.Path(directory)
            state = FetchState(root / "state.sqlite3")
            with ExitStack() as stack:
                environment = self._environment(stack, root, state)

                def create_job(**kwargs: object) -> dict[str, str]:
                    self._create_job(
                        state,
                        environment["run_root"],
                        status="queued",
                    )
                    return {"status": "queued", "job_id": JOB_ID}

                create = stack.enter_context(
                    patch.object(continuous, "run_research", side_effect=create_job)
                )
                worker = Mock(
                    return_value=continuous.WorkerOutcome(
                        pid=503,
                        returncode=0,
                        timed_out=False,
                        duration_seconds=1.0,
                    )
                )
                result = continuous.run_once(worker_runner=worker)

            create.assert_called_once_with(
                mode="daily",
                asset_scope="all",
                symbols=None,
                _target_date=TARGET_DATE,
                _start_background_worker=False,
            )
            worker.assert_called_once()
            self.assertEqual(result["run_id"], JOB_ID)
            self.assertEqual(result["asset_scope"], "all")

    def test_overlapping_scheduler_tick_is_skipped_by_lock(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="continuous-lock-",
            dir=ROOT / "tests",
        ) as directory:
            root = pathlib.Path(directory)
            state = FetchState(root / "state.sqlite3")
            with ExitStack() as stack:
                self._environment(stack, root, state)
                lock = continuous.SchedulerLock(stale_seconds=60)
                self.assertTrue(lock.acquire("outer-test-run"))
                try:
                    result = continuous.run_once(
                        worker_runner=Mock(
                            side_effect=AssertionError("worker must not start")
                        )
                    )
                finally:
                    lock.release()
            self.assertEqual(result["status"], "running")
            self.assertEqual(result["skipped"], "scheduler_lock_held")

    def test_completed_target_does_not_start_another_worker(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="continuous-complete-",
            dir=ROOT / "tests",
        ) as directory:
            root = pathlib.Path(directory)
            state = FetchState(root / "state.sqlite3")
            with ExitStack() as stack:
                environment = self._environment(stack, root, state)
                self._create_job(
                    state,
                    environment["run_root"],
                    status="complete",
                    complete=True,
                )
                worker = Mock(
                    side_effect=AssertionError("completed date must remain idle")
                )
                result = continuous.run_once(worker_runner=worker)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["action"], "target_completed")
            worker.assert_not_called()

    def test_failed_asset_keeps_round_partial_instead_of_stopping_market(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="continuous-partial-",
            dir=ROOT / "tests",
        ) as directory:
            root = pathlib.Path(directory)
            state = FetchState(root / "state.sqlite3")
            with ExitStack() as stack:
                environment = self._environment(stack, root, state)
                self._create_job(state, environment["run_root"], status="queued")
                claimed = state.claim_due(
                    "stock",
                    symbols=["000001"],
                    limit=1,
                    target_date=TARGET_DATE,
                    owner="test-owner",
                )
                self.assertEqual(len(claimed), 1)
                state.mark_failure(
                    "stock",
                    "000001",
                    error="AkShare rate limited",
                    backoff_base_seconds=1,
                    backoff_max_seconds=10,
                    owner="test-owner",
                )

                def finish_normally(job_id: str, _logger: object):
                    state.update_job(job_id, status="queued")
                    return continuous.WorkerOutcome(
                        pid=400,
                        returncode=0,
                        timed_out=False,
                        duration_seconds=1.0,
                    )

                result = continuous.run_once(worker_runner=finish_normally)
            self.assertEqual(result["status"], "partial")
            self.assertEqual(result["failed"], 1)
            self.assertEqual(result["failure_streak"], 0)

    def test_three_consecutive_failures_write_high_priority_alert(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="continuous-alert-",
            dir=ROOT / "tests",
        ) as directory:
            root = pathlib.Path(directory)
            state = FetchState(root / "state.sqlite3")
            with ExitStack() as stack:
                environment = self._environment(stack, root, state)
                logger = continuous._logger()
                control = continuous._load_control()
                for _ in range(3):
                    continuous._record_failure(
                        control,
                        failed=True,
                        job_id=JOB_ID,
                        target_date=TARGET_DATE,
                        coverage=0.1,
                        logger=logger,
                    )
                for handler in logger.handlers:
                    handler.flush()
                alert_logger = continuous._rotating_logger(
                    "dsh.continuous_research.alert",
                    continuous.ALERT_PATH,
                    max_bytes=continuous.ALERT_MAX_BYTES,
                    backup_count=continuous.ALERT_BACKUP_COUNT,
                    formatter="%(asctime)s HIGH %(message)s",
                )
                for handler in alert_logger.handlers:
                    handler.flush()
            alert_text = (
                environment["continuous_dir"] / "alerts.log"
            ).read_text(encoding="utf-8")
            self.assertIn("3 consecutive rounds", alert_text)
            self.assertIn(JOB_ID, alert_text)

    def test_asset_type_filter_preserves_lowercase_schema_values(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="continuous-assets-",
            dir=ROOT / "tests",
        ) as directory:
            state = FetchState(pathlib.Path(directory) / "state.sqlite3")
            state.register_assets("stock", ["000001"])
            state.register_assets("etf", ["510300"])
            rows = state.list_assets(["stock"], limit=10)
            self.assertEqual(
                [(row["asset_type"], row["symbol"]) for row in rows],
                [("stock", "000001")],
            )

    def test_main_emits_json_when_stdout_cannot_encode_payload(self) -> None:
        buffer = io.BytesIO()
        stream = io.TextIOWrapper(buffer, encoding="cp936", newline="\n")
        with (
            patch.object(continuous, "status", return_value={"name": "\u05e5"}),
            patch.object(sys, "stdout", stream),
        ):
            exit_code = continuous.main(["--action", "status"])
            stream.flush()

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            json.loads(buffer.getvalue().decode("cp936")),
            {"name": "\u05e5"},
        )


class QueuedTargetDateTests(unittest.TestCase):
    def test_new_target_date_does_not_reuse_old_queued_job(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="queued-target-date-",
            dir=ROOT / "tests",
        ) as directory:
            root = pathlib.Path(directory)
            data_dir = root / "data"
            results_dir = root / "results"
            runs_dir = results_dir / "runs"
            logs_dir = root / "logs"
            for path in (data_dir, results_dir, runs_dir, logs_dir):
                path.mkdir(parents=True, exist_ok=True)
            manifest_path = data_dir / "universe.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "complete": False,
                        "stock_universe_complete": True,
                        "stock": [
                            {
                                "symbol": "600519",
                                "name": "test",
                                "asset_type": "stock",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            config = copy.deepcopy(DEFAULTS)
            config["fetch"]["state_db"] = str(root / "state.sqlite3")
            config["fetch"]["background_worker_enabled"] = False
            state = FetchState(pathlib.Path(config["fetch"]["state_db"]))
            old_id = "20260825T100000Z-00000001"
            state.create_job(
                old_id,
                kind="research",
                asset_scope="stock",
                metadata={
                    "mode": "daily",
                    "asset_scope": "stock",
                    "asset_types": ["stock"],
                    "symbols": {"stock": ["600519"]},
                    "target_date": "2026-08-25",
                },
            )
            with ExitStack() as stack:
                stack.enter_context(patch.object(research, "DATA_DIR", data_dir))
                stack.enter_context(patch.object(research, "UNIVERSE_PATH", manifest_path))
                stack.enter_context(
                    patch.object(research, "ASSISTANT_RESULTS_DIR", results_dir)
                )
                stack.enter_context(patch.object(research, "RUNS_DIR", runs_dir))
                stack.enter_context(patch.object(paths, "RUNS_DIR", runs_dir))
                stack.enter_context(patch.object(research, "load_config", return_value=config))
                stack.enter_context(
                    patch.object(
                        research,
                        "_now",
                        return_value=datetime(
                            2026,
                            8,
                            26,
                            10,
                            0,
                            tzinfo=timezone.utc,
                        ),
                    )
                )
                stack.enter_context(
                    patch.object(research, "_run_id", return_value=JOB_ID)
                )
                result = research.run_research(
                    "daily",
                    "stock",
                    _target_date=TARGET_DATE,
                    _start_background_worker=False,
                )
            self.assertEqual(result["job_id"], JOB_ID)
            self.assertEqual(result["target_date"], TARGET_DATE)
            self.assertNotEqual(result["job_id"], old_id)


if __name__ == "__main__":
    unittest.main()
