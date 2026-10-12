from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import pathlib
import sys
import tempfile
import unittest
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from unittest.mock import ANY, Mock, patch


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from assistant import paths, research
from assistant.config import DEFAULTS
import continuous_research as continuous
from fetch_state import FetchState, iso_now


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

                def timeout_and_claim(job_id: str, _logger: object):
                    self.assertTrue(state.claim_job(job_id))
                    started_at = str(state.get_job(job_id)["started_at"])
                    return continuous.WorkerOutcome(
                        pid=321,
                        returncode=None,
                        timed_out=True,
                        duration_seconds=1.0,
                        error="worker wall timeout",
                        started_at=started_at,
                    )
                first = continuous.run_once(worker_runner=timeout_and_claim)

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

            self.assertEqual(first["status"], "queued")
            self.assertTrue(first["worker_timeout"])
            self.assertTrue(first["checkpoint_retained"])
            self.assertTrue(first["job_requeued"])
            self.assertEqual(state.get_job(JOB_ID)["status"], "queued")
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

    def test_timeout_cannot_requeue_a_new_worker_generation(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="continuous-timeout-generation-",
            dir=ROOT / "tests",
        ) as directory:
            root = pathlib.Path(directory)
            state = FetchState(root / "state.sqlite3")
            with ExitStack() as stack:
                self._environment(stack, root, state)
                self._create_job(state, root / "runs", status="queued")
                old_generation = "2026-08-26T10:00:00+00:00"
                new_generation = "2026-08-26T10:01:00+00:00"

                def timed_out_old_worker(job_id: str, _logger: object):
                    state.update_job(
                        job_id,
                        status="running",
                        started_at=old_generation,
                    )
                    state.update_job(
                        job_id,
                        status="running",
                        started_at=new_generation,
                    )
                    return continuous.WorkerOutcome(
                        pid=322,
                        returncode=None,
                        timed_out=True,
                        duration_seconds=1.0,
                        error="old worker wall timeout",
                        started_at=old_generation,
                    )

                result = continuous.run_once(
                    worker_runner=timed_out_old_worker
                )

            current = state.get_job(JOB_ID)
            self.assertIsNotNone(current)
            self.assertEqual(current["status"], "running")
            self.assertEqual(current["started_at"], new_generation)
            self.assertFalse(result["job_requeued"])
            self.assertEqual(result["status"], "failed")

    def test_update_job_if_requires_matching_generation(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="continuous-update-job-if-",
            dir=ROOT / "tests",
        ) as directory:
            state = FetchState(pathlib.Path(directory) / "state.sqlite3")
            run_root = pathlib.Path(directory) / "runs"
            self._create_job(state, run_root, status="queued")
            self.assertTrue(state.claim_job(JOB_ID))
            current = state.get_job(JOB_ID)
            self.assertIsNotNone(current)
            started_at = str(current["started_at"])

            self.assertFalse(
                state.update_job_if(
                    JOB_ID,
                    expected_status="running",
                    expected_started_at="wrong-generation",
                    status="queued",
                )
            )
            self.assertEqual(state.get_job(JOB_ID)["status"], "running")
            self.assertTrue(
                state.update_job_if(
                    JOB_ID,
                    expected_status="running",
                    expected_started_at=started_at,
                    status="queued",
                )
            )
            self.assertEqual(state.get_job(JOB_ID)["status"], "queued")


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

    def test_lock_held_by_live_process_is_not_stolen(self) -> None:
        """锁文件存在且记录的是**存活** PID 时，不得判过期、不得删除。

        修复前 _process_exists 在 Windows 恒返回 False（os.kill(pid,0) 对非本进程
        创建的存活进程抛 winerror=87），stale 恒为真 → 去 unlink 仍被持有者打开的
        锁文件 → PermissionError 抛穿到 main()，整个 CLI 以退出码 2 结束
        （2026-09-16 17:30 postclose 计划任务实测）。
        """
        with tempfile.TemporaryDirectory(
            prefix="continuous-lock-live-",
            dir=ROOT / "tests",
        ) as directory:
            root = pathlib.Path(directory)
            state = FetchState(root / "state.sqlite3")
            with ExitStack() as stack:
                self._environment(stack, root, state)
                lock = continuous.SchedulerLock(stale_seconds=60)
                self.assertTrue(lock.acquire("first-run"))
                try:
                    # 锁记录的是本进程这个存活 PID：必须让出而不是删锁。
                    second = continuous.SchedulerLock(stale_seconds=60)
                    self.assertFalse(second.acquire("second-run"))
                    self.assertTrue(continuous.LOCK_PATH.is_file())
                finally:
                    lock.release()

    def test_lock_with_dead_pid_is_recovered_without_raising(self) -> None:
        """锁记录的 PID 已死时应正常回收，且不得抛异常。"""
        with tempfile.TemporaryDirectory(
            prefix="continuous-lock-stale-",
            dir=ROOT / "tests",
        ) as directory:
            root = pathlib.Path(directory)
            state = FetchState(root / "state.sqlite3")
            with ExitStack() as stack:
                self._environment(stack, root, state)
                continuous.LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
                continuous.LOCK_PATH.write_text(
                    json.dumps(
                        {
                            "run_id": "dead-holder",
                            "pid": 999_999_999,
                            "created_at": "2026-09-16T00:00:00+08:00",
                        }
                    ),
                    encoding="utf-8",
                )
                lock = continuous.SchedulerLock(stale_seconds=60)
                self.assertTrue(lock.acquire("recovered-run"))
                lock.release()

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

    def test_coverage_floor_closes_target_and_writes_terminal_job_state(self) -> None:
        """2026-09-15 收尾门：判据看 progress 的 coverage（不再看 job.status），
        且**必须同时写 DB 终态**——否则下一 tick 的 _find_resumable_job 会捡回
        仍是 running 的旧作业，target_date 一步不动（死循环从忙等变闲等）。
        """
        with tempfile.TemporaryDirectory(
            prefix="continuous-covfloor-",
            dir=ROOT / "tests",
        ) as directory:
            root = pathlib.Path(directory)
            state = FetchState(root / "state.sqlite3")
            with ExitStack() as stack:
                environment = self._environment(stack, root, state)
                # 把收尾门降到 0.5，让"3 个资产里 2 个达标"(≈0.667) 即可收尾，
                # 使本用例聚焦判据逻辑而不绑死具体阈值。
                environment["config"]["fetch"]["completion_coverage_floor"] = 0.5
                environment["config"]["fetch"]["completion_allow_failed"] = True
                self._create_job(
                    state,
                    environment["run_root"],
                    status="running",
                    complete=False,
                )
                # stock / etf 达标，otc_fund 仍滞后 → coverage = 2/3、pending = 1
                # 用 mark_success 而非再调 register_assets：后者是 INSERT-or-keep
                # 语义（docstring: "without overwriting progress for existing
                # assets"），对已存在资产不会更新 data_end_date，等于白写。
                for asset_type, symbol in (("stock", "000001"), ("etf", "510300")):
                    state.mark_success(
                        asset_type,
                        symbol,
                        data_end_date=TARGET_DATE,
                        source="test",
                        rows=1,
                        refresh_interval_seconds=3600,
                    )
                worker = Mock(
                    side_effect=AssertionError(
                        "coverage floor met; must not start a worker"
                    )
                )
                result = continuous.run_once(worker_runner=worker)
                job_after = state.get_job(JOB_ID)
            worker.assert_not_called()
            self.assertEqual(result["action"], "target_completed")
            self.assertEqual(result["status"], "completed")
            self.assertIsNotNone(job_after)
            assert job_after is not None
            # 终态：completed 不在 RESUMABLE_JOB_STATUSES 里
            self.assertEqual(job_after["status"], "completed")
            # 豁免留痕进 metadata；且整包替换不得擦掉原有 target_date
            metadata = job_after.get("metadata") or {}
            self.assertEqual(metadata.get("target_date"), TARGET_DATE)
            self.assertEqual(metadata.get("asset_scope"), "all")
            completion = metadata.get("completion") or {}
            self.assertEqual(
                completion.get("reason"), "closure by completion_coverage_floor"
            )
            self.assertEqual(completion.get("exempted_pending"), 1)

    def test_coverage_floor_is_strict_by_default(self) -> None:
        """DEFAULTS 下 floor=1.0 / allow_failed=False：97.9% 覆盖率不得收尾
        （等价于原来的 pending == 0，合成夹具与 under50 路径行为不变）。"""
        with tempfile.TemporaryDirectory(
            prefix="continuous-covstrict-",
            dir=ROOT / "tests",
        ) as directory:
            root = pathlib.Path(directory)
            state = FetchState(root / "state.sqlite3")
            with ExitStack() as stack:
                environment = self._environment(stack, root, state)
                fetch_cfg = environment["config"]["fetch"]
                self.assertEqual(fetch_cfg["completion_coverage_floor"], 1.0)
                self.assertFalse(fetch_cfg["completion_allow_failed"])
                self._create_job(
                    state,
                    environment["run_root"],
                    status="running",
                    complete=False,
                )
                for asset_type, symbol in (("stock", "000001"), ("etf", "510300")):
                    state.mark_success(
                        asset_type,
                        symbol,
                        data_end_date=TARGET_DATE,
                        source="test",
                        rows=1,
                        refresh_interval_seconds=3600,
                    )

                def finish(job_id: str, _logger: object):
                    state.update_job(job_id, status="queued")
                    return continuous.WorkerOutcome(
                        pid=401,
                        returncode=0,
                        timed_out=False,
                        duration_seconds=1.0,
                    )

                result = continuous.run_once(worker_runner=finish)
                job_after = state.get_job(JOB_ID)
            # coverage = 2/3 未达严格门 → 不得收尾，必须继续跑 worker
            # （用 .get：继续跑 worker 的路径不带 action 键，只有 target_completed 分支带）
            self.assertNotEqual(result.get("action"), "target_completed")
            self.assertIsNotNone(job_after)
            assert job_after is not None
            self.assertNotEqual(job_after["status"], "completed")

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

    def test_gate_failure_counts_retry_even_with_pending(self) -> None:
        # 2026-09-10 防空转A：failed + pending≠0 + 结构化gate 仍计次。
        with tempfile.TemporaryDirectory(
            prefix="continuous-gate-",
            dir=ROOT / "tests",
        ) as directory:
            root = pathlib.Path(directory)
            state = FetchState(root / "state.sqlite3")
            with ExitStack() as stack:
                environment = self._environment(stack, root, state)
                self._create_job(state, environment["run_root"], status="failed")
                run_dir = environment["run_root"] / JOB_ID
                (run_dir / "status.json").write_text(
                    json.dumps(
                        {
                            "run_id": JOB_ID,
                            "status": "failed",
                            "failure_reasons": ["stock_model_or_risk_gate_failed"],
                        }
                    ),
                    encoding="utf-8",
                )
                state.register_assets("stock", ["000001"])
                control: dict = {"analysis_attempts": {}}
                job = state.get_job(JOB_ID)
                assert job is not None
                progress = {"pending": 5, "total": 10}
                _, _, action = continuous._ensure_job(
                    state,
                    TARGET_DATE,
                    "daily",
                    environment["config"],
                    control,
                    continuous._logger(),
                    job,
                )
                self.assertIsNone(action)
                self.assertEqual(control["analysis_attempts"][JOB_ID], 1)

    def test_gate_failure_pauses_after_three_retries_with_pending(self) -> None:
        # 2026-09-10 防空转A：满3次后 paused，不再起 worker。
        with tempfile.TemporaryDirectory(
            prefix="continuous-gate-pause-",
            dir=ROOT / "tests",
        ) as directory:
            root = pathlib.Path(directory)
            state = FetchState(root / "state.sqlite3")
            with ExitStack() as stack:
                environment = self._environment(stack, root, state)
                self._create_job(state, environment["run_root"], status="failed")
                run_dir = environment["run_root"] / JOB_ID
                (run_dir / "status.json").write_text(
                    json.dumps(
                        {
                            "run_id": JOB_ID,
                            "status": "failed",
                            "failure_reasons": ["test_max_drawdown_threshold"],
                        }
                    ),
                    encoding="utf-8",
                )
                state.register_assets("stock", ["000001"])
                control: dict = {"analysis_attempts": {JOB_ID: 3}}
                job = state.get_job(JOB_ID)
                assert job is not None
                with patch.object(
                    continuous.state if hasattr(continuous, "state") else continuous,
                    "_job_progress",
                    return_value={"pending": 5, "total": 10},
                    create=True,
                ):
                    pass
                # 直接走 _ensure_job 的计数分支：pending≠0 但 gate 命中。
                job2, _, action = continuous._ensure_job(
                    state,
                    TARGET_DATE,
                    "daily",
                    environment["config"],
                    control,
                    continuous._logger(),
                    job,
                )
                self.assertEqual(action, "analysis_retry_exhausted")
                self.assertEqual((job2 or {}).get("status"), "paused")

    def test_pure_fetch_failure_still_requeues_without_counting(self) -> None:
        # 2026-09-10 防空转A：纯抓取失败（无结构化gate）不计数，仍 requeue。
        with tempfile.TemporaryDirectory(
            prefix="continuous-fetch-",
            dir=ROOT / "tests",
        ) as directory:
            root = pathlib.Path(directory)
            state = FetchState(root / "state.sqlite3")
            with ExitStack() as stack:
                environment = self._environment(stack, root, state)
                self._create_job(state, environment["run_root"], status="failed")
                state.register_assets("stock", ["000001"])
                control: dict = {"analysis_attempts": {}}
                job = state.get_job(JOB_ID)
                assert job is not None
                _, _, action = continuous._ensure_job(
                    state,
                    TARGET_DATE,
                    "daily",
                    environment["config"],
                    control,
                    continuous._logger(),
                    job,
                )
                self.assertIsNone(action)
                self.assertNotIn(JOB_ID, control["analysis_attempts"])

    def test_streak_auto_pause_writes_paused_file_once(self) -> None:
        # 2026-09-10 防空转B：streak 2→3 边沿写 paused.json 且只一次。
        with tempfile.TemporaryDirectory(
            prefix="continuous-autopause-",
            dir=ROOT / "tests",
        ) as directory:
            root = pathlib.Path(directory)
            state = FetchState(root / "state.sqlite3")
            with ExitStack() as stack:
                self._environment(stack, root, state)
                logger = continuous._logger()
                control = continuous._load_control()
                control["failure_streak"] = 2
                control["last_alert_failure_streak"] = 0
                streak = continuous._record_failure(
                    control,
                    failed=True,
                    job_id=JOB_ID,
                    target_date=TARGET_DATE,
                    coverage=0.9,
                    logger=logger,
                )
                self.assertEqual(streak, 3)
                paused = continuous._read_json(continuous.PAUSE_PATH, {})
                self.assertEqual(paused.get("job_id"), JOB_ID)
                self.assertIn("automatic pause", paused.get("reason", ""))
                before = (continuous.PAUSE_PATH).stat().st_mtime_ns
                streak2 = continuous._record_failure(
                    control,
                    failed=True,
                    job_id=JOB_ID,
                    target_date=TARGET_DATE,
                    coverage=0.9,
                    logger=logger,
                )
                self.assertEqual(streak2, 4)
                after = (continuous.PAUSE_PATH).stat().st_mtime_ns
                self.assertEqual(before, after)

    def test_resume_clears_streak_for_fresh_window(self) -> None:
        # 2026-09-10 防空转B：resume 清 streak，给 3 轮验证窗口。
        with tempfile.TemporaryDirectory(
            prefix="continuous-resume-",
            dir=ROOT / "tests",
        ) as directory:
            root = pathlib.Path(directory)
            state = FetchState(root / "state.sqlite3")
            with ExitStack() as stack:
                self._environment(stack, root, state)
                control = continuous._load_control()
                control["failure_streak"] = 3
                control["last_alert_failure_streak"] = 3
                continuous._save_control(control)
                continuous.resume()
                fresh = continuous._load_control()
                self.assertEqual(fresh.get("failure_streak"), 0)
                self.assertEqual(fresh.get("last_alert_failure_streak"), 0)
                self.assertFalse(continuous.PAUSE_PATH.is_file())

    def test_structured_gate_failure_fail_open_on_bad_job_id(self) -> None:
        # 2026-09-10 轻量验证补回归：非法 job_id 不抛异常，按非门槛处理。
        self.assertFalse(
            continuous._structured_gate_failure({"job_id": "x", "metadata_json": "{}"})
        )
        self.assertTrue(
            continuous._structured_gate_failure(
                {
                    "job_id": "x",
                    "metadata_json": '{"failure_reasons":["stock_model_or_risk_gate_failed"]}',
                }
            )
        )
        self.assertFalse(continuous._structured_gate_failure(None))

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

    def test_published_round_with_abnormal_worker_exit_is_not_a_failure(self) -> None:
        # 2026-09-11 P0-4：worker 退出码非 0，但本轮产物已发布（manifest success +
        # publication.published）→ 不计入 failure_streak，不许写 paused.json。
        # 实测事故：job 20260911T065625Z-d6d4eede 已成功发布，却被记为失败，
        # failure_streak 累到 3 触发 _auto_pause_on_streak，22:05 起调度器停摆。
        # 2026-09-13：补 partial 一档——P0-3 分类别发布后"已发布"包含
        # status=partial / publication.state=partial（就绪类别已落盘、被拦类别归零），
        # _round_published 只看 publication.published，两档都不得计入连败。
        for manifest_status, publication_state in (
            ("success", "full"),
            ("partial", "partial"),
        ):
            with self.subTest(manifest_status=manifest_status):
                with tempfile.TemporaryDirectory(
                    prefix="continuous-p04-published-",
                    dir=ROOT / "tests",
                ) as directory:
                    root = pathlib.Path(directory)
                    state = FetchState(root / "state.sqlite3")
                    with ExitStack() as stack:
                        environment = self._environment(stack, root, state)
                        self._create_job(state, environment["run_root"])
                        run_dir = environment["run_root"] / JOB_ID

                        def publish_then_exit_one(
                            job_id: str,
                            _logger: object,
                            *,
                            _status: str = manifest_status,
                            _state: str = publication_state,
                        ):
                            self.assertTrue(state.claim_job(job_id))
                            run_dir.mkdir(parents=True, exist_ok=True)
                            (run_dir / "manifest.json").write_text(
                                json.dumps(
                                    {
                                        "run_id": job_id,
                                        "status": _status,
                                        "failure_reasons": [],
                                        "finished_at": continuous._now().isoformat(),
                                        "publication": {
                                            "published": True,
                                            "state": _state,
                                        },
                                    }
                                ),
                                encoding="utf-8",
                            )
                            return continuous.WorkerOutcome(
                                pid=321,
                                returncode=1,
                                timed_out=False,
                                duration_seconds=1.0,
                                error="worker exited with code 1",
                                started_at=str(state.get_job(job_id)["started_at"]),
                            )

                        continuous.run_once(worker_runner=publish_then_exit_one)
                        control = continuous._load_control()
                        self.assertEqual(control.get("failure_streak"), 0)
                        self.assertFalse(continuous.PAUSE_PATH.is_file())

    def test_round_published_rejects_stale_or_unpublished_artifacts(self) -> None:
        # 2026-09-11 P0-4：只有"本轮产出且已发布"的产物才算成功；
        # 复用 run 目录时上一次 pass 留下的旧产物、以及未发布的产物都必须被拒。
        with tempfile.TemporaryDirectory(
            prefix="continuous-p04-stale-",
            dir=ROOT / "tests",
        ) as directory:
            root = pathlib.Path(directory)
            state = FetchState(root / "state.sqlite3")
            with ExitStack() as stack:
                environment = self._environment(stack, root, state)
                run_dir = environment["run_root"] / JOB_ID
                run_dir.mkdir(parents=True, exist_ok=True)
                manifest_path = run_dir / "manifest.json"
                since = continuous._now()

                def write(**overrides: object) -> None:
                    payload = {
                        "run_id": JOB_ID,
                        "status": "success",
                        "finished_at": continuous._now().isoformat(),
                        "publication": {"published": True, "state": "full"},
                    }
                    payload.update(overrides)
                    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

                write()
                self.assertTrue(continuous._round_published(JOB_ID, since=since))

                # 旧产物（finished_at 早于本轮起始）→ 拒。
                write(finished_at="2026-09-11T10:00:00+08:00")
                self.assertFalse(continuous._round_published(JOB_ID, since=since))

                # publication.published 为假 → 拒。
                write(publication={"published": False, "state": "blocked"})
                self.assertFalse(continuous._round_published(JOB_ID, since=since))

                # run_id 不匹配 → 拒。
                write(run_id="other-round")
                self.assertFalse(continuous._round_published(JOB_ID, since=since))

                # 无 manifest → 拒。
                manifest_path.unlink()
                self.assertFalse(continuous._round_published(JOB_ID, since=since))

    def test_abnormal_exit_without_artifacts_still_counts_failure(self) -> None:
        # 2026-09-11 P0-4 反向保护：真的没产出时，熔断照常累加——
        # 修的是误判，不是削弱 _auto_pause_on_streak（2026-09-10 防空转B）。
        with tempfile.TemporaryDirectory(
            prefix="continuous-p04-noartifact-",
            dir=ROOT / "tests",
        ) as directory:
            root = pathlib.Path(directory)
            state = FetchState(root / "state.sqlite3")
            with ExitStack() as stack:
                environment = self._environment(stack, root, state)
                self._create_job(state, environment["run_root"])
                run_dir = environment["run_root"] / JOB_ID

                def exit_one_without_artifacts(job_id: str, _logger: object):
                    self.assertTrue(state.claim_job(job_id))
                    self.assertFalse((run_dir / "manifest.json").is_file())
                    return continuous.WorkerOutcome(
                        pid=321,
                        returncode=1,
                        timed_out=False,
                        duration_seconds=1.0,
                        error="worker exited with code 1",
                        started_at=str(state.get_job(job_id)["started_at"]),
                    )

                continuous.run_once(worker_runner=exit_one_without_artifacts)
                control = continuous._load_control()
                self.assertEqual(control.get("failure_streak"), 1)
                self.assertFalse(continuous._round_published(JOB_ID, since=continuous._now()))

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

    # --- T14: partial 作业被更新交易日取代（sched-stuck-rca 2026-10-12）---

    NEWER_DATE = "2026-08-27"
    NEW_JOB_ID = "20260827T100000Z-00000002"

    def _patch_calendar(self, stack: ExitStack, **overrides: object) -> Mock:
        payload = {
            "status": "ok",
            "date": self.NEWER_DATE,
            "is_trading_day": True,
            "is_month_end_trading_day": False,
            "source": "test",
        }
        payload.update(overrides)
        return stack.enter_context(
            patch.object(continuous, "_calendar_today", return_value=payload)
        )

    def test_partial_job_superseded_by_newer_trading_day(self) -> None:
        """partial + 更新交易日：旧作业按 superseded 收尾，新作业为该交易日创建。"""
        with tempfile.TemporaryDirectory(
            prefix="continuous-t14-supersede-",
            dir=ROOT / "tests",
        ) as directory:
            root = pathlib.Path(directory)
            state = FetchState(root / "state.sqlite3")
            with ExitStack() as stack:
                environment = self._environment(stack, root, state)
                self._create_job(
                    state,
                    environment["run_root"],
                    status="partial",
                )
                calendar = self._patch_calendar(stack)

                def create_job(**kwargs: object) -> dict[str, str]:
                    self._create_job(
                        state,
                        environment["run_root"],
                        status="queued",
                        job_id=self.NEW_JOB_ID,
                        target_date=self.NEWER_DATE,
                    )
                    return {"status": "queued", "job_id": self.NEW_JOB_ID}

                create = stack.enter_context(
                    patch.object(continuous, "run_research", side_effect=create_job)
                )
                worker = Mock(
                    return_value=continuous.WorkerOutcome(
                        pid=601,
                        returncode=0,
                        timed_out=False,
                        duration_seconds=1.0,
                    )
                )
                result = continuous.run_once(worker_runner=worker)

            # 日历只取一次（取代分支与新作业分支复用同一份）
            calendar.assert_called_once()
            # 新作业按更新交易日创建
            create.assert_called_once_with(
                mode="daily",
                asset_scope="all",
                symbols=None,
                _target_date=self.NEWER_DATE,
                _start_background_worker=False,
            )
            self.assertEqual(result["run_id"], self.NEW_JOB_ID)
            worker.assert_called_once_with(self.NEW_JOB_ID, ANY)
            # 旧作业：completed 终态 + completion 块 + metadata 保留
            old_job = state.get_job(JOB_ID)
            assert old_job is not None
            self.assertEqual(old_job["status"], "completed")
            self.assertIsNotNone(old_job.get("finished_at"))
            metadata = json.loads(old_job["metadata"]) if isinstance(
                old_job["metadata"], str
            ) else old_job["metadata"]
            self.assertEqual(metadata["mode"], "daily")
            self.assertEqual(metadata["asset_scope"], "all")
            self.assertEqual(metadata["target_date"], TARGET_DATE)
            self.assertEqual(
                metadata["asset_types"], ["stock", "etf", "otc_fund"]
            )
            self.assertIn("symbols", metadata)
            completion = metadata["completion"]
            self.assertEqual(
                completion["reason"], "superseded_by_newer_trading_day"
            )
            self.assertEqual(completion["target_date"], TARGET_DATE)
            self.assertIn("completed_at", completion)
            self.assertIn("coverage", completion)
            self.assertIn("exempted_failed", completion)
            self.assertIn("exempted_pending", completion)
            new_job = state.get_job(self.NEW_JOB_ID)
            assert new_job is not None
            self.assertIn(new_job["status"], {"queued", "running"})

    def test_partial_job_not_superseded_while_live_worker_exists(self) -> None:
        """T14-N1：partial 作业仍有存活 worker 时不得取代，本轮照旧 resume。"""
        with tempfile.TemporaryDirectory(
            prefix="continuous-t14-live-",
            dir=ROOT / "tests",
        ) as directory:
            root = pathlib.Path(directory)
            state = FetchState(root / "state.sqlite3")
            with ExitStack() as stack:
                environment = self._environment(stack, root, state)
                self._create_job(
                    state, environment["run_root"], status="partial"
                )
                self._patch_calendar(stack)
                stack.enter_context(
                    patch.object(
                        continuous,
                        "_find_live_worker_for_job",
                        return_value=4242,
                    )
                )
                create = stack.enter_context(
                    patch.object(
                        continuous,
                        "run_research",
                        side_effect=AssertionError(
                            "run_research must not be called while resuming"
                        ),
                    )
                )
                worker = Mock(
                    return_value=continuous.WorkerOutcome(
                        pid=607, returncode=0, timed_out=False,
                        duration_seconds=1.0,
                    )
                )
                result = continuous.run_once(worker_runner=worker)
            create.assert_not_called()
            self.assertEqual(result["run_id"], JOB_ID)
            old_job = state.get_job(JOB_ID)
            assert old_job is not None
            self.assertNotEqual(old_job["status"], "completed")

    def test_partial_job_resumed_on_non_trading_day(self) -> None:
        """非交易日：partial 作业照旧 resume，不得取代。"""
        with tempfile.TemporaryDirectory(
            prefix="continuous-t14-nontt-",
            dir=ROOT / "tests",
        ) as directory:
            root = pathlib.Path(directory)
            state = FetchState(root / "state.sqlite3")
            with ExitStack() as stack:
                environment = self._environment(stack, root, state)
                self._create_job(
                    state, environment["run_root"], status="partial"
                )
                self._patch_calendar(stack, is_trading_day=False)
                create = stack.enter_context(
                    patch.object(continuous, "run_research", Mock())
                )
                worker = Mock(
                    return_value=continuous.WorkerOutcome(
                        pid=602, returncode=0, timed_out=False,
                        duration_seconds=1.0,
                    )
                )
                result = continuous.run_once(worker_runner=worker)
            create.assert_not_called()
            self.assertEqual(result["run_id"], JOB_ID)
            old_job = state.get_job(JOB_ID)
            assert old_job is not None
            self.assertNotEqual(old_job["status"], "completed")

    def test_partial_job_resumed_on_same_date(self) -> None:
        """日历日期与作业目标同日：照旧 resume。"""
        with tempfile.TemporaryDirectory(
            prefix="continuous-t14-sameday-",
            dir=ROOT / "tests",
        ) as directory:
            root = pathlib.Path(directory)
            state = FetchState(root / "state.sqlite3")
            with ExitStack() as stack:
                environment = self._environment(stack, root, state)
                self._create_job(
                    state, environment["run_root"], status="partial"
                )
                self._patch_calendar(stack, date=TARGET_DATE)
                create = stack.enter_context(
                    patch.object(continuous, "run_research", Mock())
                )
                worker = Mock(
                    return_value=continuous.WorkerOutcome(
                        pid=603, returncode=0, timed_out=False,
                        duration_seconds=1.0,
                    )
                )
                result = continuous.run_once(worker_runner=worker)
            create.assert_not_called()
            self.assertEqual(result["run_id"], JOB_ID)
            old_job = state.get_job(JOB_ID)
            assert old_job is not None
            self.assertNotEqual(old_job["status"], "completed")

    def test_queued_and_running_jobs_not_superseded(self) -> None:
        """queued/running 作业即使目标日更旧也不得被取代。"""
        for status_value in ("queued", "running"):
            with self.subTest(status=status_value):
                with tempfile.TemporaryDirectory(
                    prefix=f"continuous-t14-{status_value}-",
                    dir=ROOT / "tests",
                ) as directory:
                    root = pathlib.Path(directory)
                    state = FetchState(root / "state.sqlite3")
                    with ExitStack() as stack:
                        environment = self._environment(stack, root, state)
                        self._create_job(
                            state,
                            environment["run_root"],
                            status=status_value,
                        )
                        self._patch_calendar(stack)
                        create = stack.enter_context(
                            patch.object(continuous, "run_research", Mock())
                        )
                        worker = Mock(
                            return_value=continuous.WorkerOutcome(
                                pid=604, returncode=0, timed_out=False,
                                duration_seconds=1.0,
                            )
                        )
                        result = continuous.run_once(worker_runner=worker)
                    create.assert_not_called()
                    self.assertEqual(result["run_id"], JOB_ID)
                    old_job = state.get_job(JOB_ID)
                    assert old_job is not None
                    self.assertNotEqual(old_job["status"], "completed")

    def test_explicit_job_id_never_superseded(self) -> None:
        """显式 --job-id：即使 partial + 更新交易日也只 resume。"""
        with tempfile.TemporaryDirectory(
            prefix="continuous-t14-explicit-",
            dir=ROOT / "tests",
        ) as directory:
            root = pathlib.Path(directory)
            state = FetchState(root / "state.sqlite3")
            with ExitStack() as stack:
                environment = self._environment(stack, root, state)
                self._create_job(
                    state, environment["run_root"], status="partial"
                )
                self._patch_calendar(stack)
                create = stack.enter_context(
                    patch.object(continuous, "run_research", Mock())
                )
                worker = Mock(
                    return_value=continuous.WorkerOutcome(
                        pid=605, returncode=0, timed_out=False,
                        duration_seconds=1.0,
                    )
                )
                result = continuous.run_once(
                    job_id=JOB_ID, worker_runner=worker
                )
            create.assert_not_called()
            self.assertEqual(result["run_id"], JOB_ID)
            old_job = state.get_job(JOB_ID)
            assert old_job is not None
            self.assertNotEqual(old_job["status"], "completed")

    def test_partial_job_resumed_on_calendar_error(self) -> None:
        """日历错误：partial 作业照旧 resume，不得取代。"""
        with tempfile.TemporaryDirectory(
            prefix="continuous-t14-calerr-",
            dir=ROOT / "tests",
        ) as directory:
            root = pathlib.Path(directory)
            state = FetchState(root / "state.sqlite3")
            with ExitStack() as stack:
                environment = self._environment(stack, root, state)
                self._create_job(
                    state, environment["run_root"], status="partial"
                )
                self._patch_calendar(
                    stack,
                    status="error",
                    is_trading_day=False,
                    error="calendar unavailable",
                )
                create = stack.enter_context(
                    patch.object(continuous, "run_research", Mock())
                )
                worker = Mock(
                    return_value=continuous.WorkerOutcome(
                        pid=606, returncode=0, timed_out=False,
                        duration_seconds=1.0,
                    )
                )
                result = continuous.run_once(worker_runner=worker)
            create.assert_not_called()
            self.assertEqual(result["run_id"], JOB_ID)
            old_job = state.get_job(JOB_ID)
            assert old_job is not None
            self.assertNotEqual(old_job["status"], "completed")


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


class PostCloseTests(unittest.TestCase):
    def test_latest_expected_trading_date_rolls_back_to_previous_trading_day(self) -> None:
        from datetime import date as date_type

        calendar = ["2026-09-04", "2026-09-07", "2026-09-08"]
        # 周五当天 → 周五
        self.assertEqual(
            continuous.latest_expected_trading_date(calendar, date_type(2026, 9, 4)),
            "2026-09-04",
        )
        # 周六 → 退回周五
        self.assertEqual(
            continuous.latest_expected_trading_date(calendar, date_type(2026, 9, 5)),
            "2026-09-04",
        )
        # 周日 → 仍退回周五
        self.assertEqual(
            continuous.latest_expected_trading_date(calendar, date_type(2026, 9, 6)),
            "2026-09-04",
        )
        # 日历为空 → None
        self.assertIsNone(continuous.latest_expected_trading_date([], date_type(2026, 9, 6)))

    def test_postclose_skips_research_when_all_data_current(self) -> None:
        with tempfile.TemporaryDirectory(prefix="postclose-fresh-", dir=ROOT / "tests") as directory:
            root = pathlib.Path(directory)
            state = FetchState(root / "fetch_state.sqlite3")
            state.register_assets("stock", ["600000"])
            state.mark_success("stock", "600000", data_end_date="2026-09-05",
                               source="test", rows=100, refresh_interval_seconds=86400)
            config = copy.deepcopy(DEFAULTS)
            config["fetch"]["state_db"] = str(root / "fetch_state.sqlite3")
            with patch.object(continuous, "load_config", return_value=config),                     patch.object(continuous, "_refresh_calendar_dates",
                                 return_value=["2026-09-04", "2026-09-05"]),                     patch.object(continuous, "run_once") as run_once_mock:
                value = continuous.run_postclose()
            self.assertEqual(value["status"], "ok")
            self.assertEqual(value["trading_date"], "2026-09-05")
            self.assertEqual(value["total_forced"], 0)
            self.assertEqual(value["research"]["status"], "skipped")
            run_once_mock.assert_not_called()

    def test_postclose_forces_stale_symbols_then_runs_research(self) -> None:
        with tempfile.TemporaryDirectory(prefix="postclose-stale-", dir=ROOT / "tests") as directory:
            root = pathlib.Path(directory)
            state = FetchState(root / "fetch_state.sqlite3")
            state.register_assets("stock", ["600000", "600001"])
            state.mark_success("stock", "600000", data_end_date="2026-09-03",
                               source="test", rows=100, refresh_interval_seconds=86400)
            state.mark_success("stock", "600001", data_end_date="2026-09-05",
                               source="test", rows=100, refresh_interval_seconds=86400)
            config = copy.deepcopy(DEFAULTS)
            config["fetch"]["state_db"] = str(root / "fetch_state.sqlite3")
            run_once_mock = Mock(return_value={"status": "ok", "job_id": "job-1"})
            with patch.object(continuous, "load_config", return_value=config),                     patch.object(continuous, "_refresh_calendar_dates",
                                 return_value=["2026-09-04", "2026-09-05"]),                     patch.object(continuous, "run_once", run_once_mock):
                value = continuous.run_postclose(job_id="postclose-1")
            self.assertEqual(value["status"], "ok")
            self.assertEqual(value["forced_due"], {"stock": 1, "etf": 0, "otc_fund": 0})
            self.assertEqual(value["total_forced"], 1)
            run_once_mock.assert_called_once_with(job_id="postclose-1")
            # 被强制的标的已可被 wave 立即认领
            self.assertEqual(
                len(state.claim_due("stock", symbols=["600000"], limit=5,
                                    target_date="2026-09-05", owner="w")),
                1,
            )

    def test_postclose_can_defer_research_to_scheduler_tick(self) -> None:
        with tempfile.TemporaryDirectory(prefix="postclose-defer-", dir=ROOT / "tests") as directory:
            root = pathlib.Path(directory)
            state = FetchState(root / "fetch_state.sqlite3")
            state.register_assets("stock", ["600000"])
            state.mark_success("stock", "600000", data_end_date="2026-09-03",
                               source="test", rows=100, refresh_interval_seconds=86400)
            config = copy.deepcopy(DEFAULTS)
            config["fetch"]["state_db"] = str(root / "fetch_state.sqlite3")
            with patch.object(continuous, "load_config", return_value=config),                     patch.object(continuous, "_refresh_calendar_dates",
                                 return_value=["2026-09-05"]),                     patch.object(continuous, "run_once") as run_once_mock:
                value = continuous.run_postclose(run_research_now=False)
            self.assertEqual(value["total_forced"], 1)
            self.assertEqual(value["research"]["status"], "deferred_to_scheduler")
            run_once_mock.assert_not_called()

    def test_postclose_falls_back_to_observed_max_date_without_calendar(self) -> None:
        with tempfile.TemporaryDirectory(prefix="postclose-nocal-", dir=ROOT / "tests") as directory:
            root = pathlib.Path(directory)
            state = FetchState(root / "fetch_state.sqlite3")
            state.register_assets("stock", ["600000", "600001"])
            state.mark_success("stock", "600000", data_end_date="2026-09-03",
                               source="test", rows=100, refresh_interval_seconds=86400)
            state.mark_success("stock", "600001", data_end_date="2026-09-04",
                               source="test", rows=100, refresh_interval_seconds=86400)
            config = copy.deepcopy(DEFAULTS)
            config["fetch"]["state_db"] = str(root / "fetch_state.sqlite3")
            with patch.object(continuous, "load_config", return_value=config),                     patch.object(continuous, "_refresh_calendar_dates", return_value=[]),                     patch.object(continuous, "run_once") as run_once_mock:
                value = continuous.run_postclose(run_research_now=False)
            # 退化基准 = 全市场观测最大日期 2026-09-04，只唤醒 9-03 的标的
            self.assertEqual(value["trading_date"], "2026-09-04")
            self.assertEqual(value["total_forced"], 1)
            run_once_mock.assert_not_called()

    def test_postclose_reclaims_verified_through_row_same_day(self) -> None:
        """T16 (hermes A2 端到端): 盘中 bar 被守卫丢弃后 data_end=09-04 而
        last_success=09-05(verified_through)。postclose 强制后，同一目标日
        09-05 必须能被 claim_due 重新认领——否则当日完整 bar 只能等次日。"""
        with tempfile.TemporaryDirectory(prefix="postclose-t16-a2-", dir=ROOT / "tests") as directory:
            root = pathlib.Path(directory)
            state = FetchState(root / "fetch_state.sqlite3")
            state.register_assets("stock", ["600000"])
            state.mark_success("stock", "600000", data_end_date="2026-09-04",
                               verified_through="2026-09-05",
                               source="test", rows=100, refresh_interval_seconds=86400)
            config = copy.deepcopy(DEFAULTS)
            config["fetch"]["state_db"] = str(root / "fetch_state.sqlite3")
            with patch.object(continuous, "load_config", return_value=config),                     patch.object(continuous, "_refresh_calendar_dates",
                                 return_value=["2026-09-04", "2026-09-05"]),                     patch.object(continuous, "run_once") as run_once_mock:
                value = continuous.run_postclose(run_research_now=False)
            self.assertEqual(value["trading_date"], "2026-09-05")
            self.assertEqual(value["total_forced"], 1)
            run_once_mock.assert_not_called()
            # 关键断言：同一目标日立即可认领（旧实现 last_success 未回拨 → 0）
            claimed = state.claim_due("stock", symbols=["600000"], limit=5,
                                      target_date="2026-09-05", owner="w")
            self.assertEqual([c["symbol"] for c in claimed], ["600000"])
            with state._connect() as connection:
                row = connection.execute(
                    "SELECT last_success_date FROM assets WHERE symbol='600000'"
                ).fetchone()
            self.assertEqual(row[0], "2026-09-04")

    def test_postclose_runs_research_when_only_last_success_blocks(self) -> None:
        """T16: 唯一未完成行的计时器已到期、只被 last_success>=target 挡住时，
        旧 advance_due_for_stale 计 0 → run_postclose 误判"全市场已最新"跳过研究。
        新语义必须强制该行并继续触发 run_once。"""
        with tempfile.TemporaryDirectory(prefix="postclose-t16-due-", dir=ROOT / "tests") as directory:
            root = pathlib.Path(directory)
            state = FetchState(root / "fetch_state.sqlite3")
            state.register_assets("stock", ["600000"])
            state.mark_success("stock", "600000", data_end_date="2026-09-04",
                               verified_through="2026-09-05",
                               source="test", rows=100, refresh_interval_seconds=86400)
            past = iso_now(datetime.now(timezone.utc) - timedelta(hours=1))
            with state._connect() as connection:
                connection.execute(
                    "UPDATE assets SET next_retry_at = ? WHERE symbol='600000'",
                    (past,),
                )
            config = copy.deepcopy(DEFAULTS)
            config["fetch"]["state_db"] = str(root / "fetch_state.sqlite3")
            run_once_mock = Mock(return_value={"status": "ok", "job_id": "job-1"})
            with patch.object(continuous, "load_config", return_value=config),                     patch.object(continuous, "_refresh_calendar_dates",
                                 return_value=["2026-09-04", "2026-09-05"]),                     patch.object(continuous, "run_once", run_once_mock):
                value = continuous.run_postclose()
            self.assertEqual(value["total_forced"], 1)
            run_once_mock.assert_called_once_with(job_id=None)

    def test_live_pid_is_reported_alive_by_process_probe(self) -> None:
        """2026-09-16 修复：os.kill(pid, 0) 在 Windows 上**不是**判活探针。

        它映射到 TerminateProcess，对非本进程创建的存活进程抛
        OSError(winerror=87)，于是"抛异常即判死"。实测在跑的调度器与 worker
        都被判 dead，导致锁被误判过期（postclose 退出码 2）与在跑作业被误 requeue。
        回归判据：对当前进程自身的 PID 必须返回 True；对不存在的 PID 返回 False。
        """
        self.assertTrue(continuous._process_exists(os.getpid()))
        self.assertFalse(continuous._process_exists(999_999_999))
        self.assertFalse(continuous._process_exists(0))
        self.assertFalse(continuous._process_exists(None))



if __name__ == "__main__":
    unittest.main()
