from __future__ import annotations

import copy
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from assistant.config import DEFAULTS
from fetch_state import FetchState
import run_research_worker as worker


class ResearchWorkerTests(unittest.TestCase):
    def test_compact_status_omits_large_nested_payloads(self) -> None:
        compact = worker._compact_status(
            {
                "job_id": "job",
                "status": "complete",
                "completed": 10,
                "research_result": {"rows": ["large"] * 1000},
                "fetch_results": [
                    {
                        "asset_type": "otc_fund",
                        "status": "partial",
                        "summary": {
                            "processed": 3,
                            "ok": 3,
                            "source_by_symbol": {"000001": {"rows": 1000}},
                        },
                        "stdout_tail": "large",
                    }
                ],
            }
        )

        self.assertNotIn("research_result", compact)
        self.assertNotIn("stdout_tail", compact["fetch_results"][0])
        self.assertNotIn(
            "source_by_symbol",
            compact["fetch_results"][0]["summary"],
        )
        self.assertLess(len(json.dumps(compact)), 1000)

    def test_timeout_refreshes_progress_and_requeues_partial_job(self) -> None:
        with tempfile.TemporaryDirectory(prefix="research-worker-", dir=ROOT / "tests") as directory:
            root = pathlib.Path(directory)
            run_dir = root / "run"
            state = FetchState(root / "state.sqlite3")
            job_id = "20260827T000000Z-timeout1"
            symbols = ["510300", "510500"]
            state.register_assets("etf", symbols)
            state.create_job(
                job_id,
                kind="research",
                asset_scope="etf",
                metadata={
                    "mode": "daily",
                    "asset_scope": "etf",
                    "asset_types": ["etf"],
                    "symbols": {"etf": symbols},
                    "target_date": "2026-08-27",
                },
            )
            state.update_job(job_id, total=2, completed=0, pending=2)

            config = copy.deepcopy(DEFAULTS)
            config["fetch"]["state_db"] = str(state.path)
            config["fetch"]["max_data_age_days"] = 0

            def timed_out_fetch(*args, **kwargs):
                self.assertEqual(kwargs["target_date"], "2026-08-26")
                state.mark_success(
                    "etf",
                    "510300",
                    data_end_date="2026-08-27",
                    source="test",
                    rows=1,
                    refresh_interval_seconds=86400,
                )
                return {
                    "asset_type": "etf",
                    "status": "timeout",
                    "returncode": None,
                    "summary": None,
                    "stdout_tail": "[完成] etf:510300",
                    "stderr_tail": "",
                    "timeout_seconds": 60,
                    "time_budget_seconds": 15,
                }

            with (
                patch.object(worker, "load_config", return_value=config),
                patch.object(worker, "run_directory", return_value=run_dir),
                patch.object(worker, "_run_fetch_script", side_effect=timed_out_fetch),
            ):
                exit_code = worker.main(["--job-id", job_id, "--time-budget", "60"])

            job = state.get_job(job_id)
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 0)
            self.assertEqual(job["status"], "queued")
            self.assertEqual(job["completed"], 1)
            self.assertEqual(job["pending"], 1)
            self.assertEqual(status["status"], "queued")
            self.assertEqual(status["progress"]["completed"], 1)
            self.assertEqual(status["progress"]["pending"], 1)
            self.assertEqual(status["fetch_results"][0]["status"], "timeout")
            self.assertEqual(status["target_dates"], {"etf": "2026-08-26"})

    def test_timeout_without_new_completions_still_requeues_when_work_remains(self) -> None:
        """预算耗尽且本轮零新完成时，作业仍须回到队列而不是判死。

        契约来自 2026-08-29 全市场作业事故：抓取子进程在限速等待中到达
        时间预算、只把在途标的标记失败、没有新增完成数，旧逻辑把整个
        作业置为 failed 并让连续调度器停止推进。
        """
        with tempfile.TemporaryDirectory(prefix="research-worker-", dir=ROOT / "tests") as directory:
            root = pathlib.Path(directory)
            run_dir = root / "run"
            state = FetchState(root / "state.sqlite3")
            job_id = "20260827T000000Z-timeout2"
            symbols = ["510300", "510500"]
            state.register_assets("etf", symbols)
            state.create_job(
                job_id,
                kind="research",
                asset_scope="etf",
                metadata={
                    "mode": "daily",
                    "asset_scope": "etf",
                    "asset_types": ["etf"],
                    "symbols": {"etf": symbols},
                    "target_date": "2026-08-27",
                },
            )
            state.update_job(job_id, total=2, completed=0, pending=2)

            config = copy.deepcopy(DEFAULTS)
            config["fetch"]["state_db"] = str(state.path)
            config["fetch"]["max_data_age_days"] = 0

            def timed_out_without_progress(*args, **kwargs):
                return {
                    "asset_type": "etf",
                    "status": "timeout",
                    "returncode": None,
                    "summary": None,
                    "stdout_tail": "",
                    "stderr_tail": "",
                    "timeout_seconds": 60,
                    "time_budget_seconds": 15,
                }

            with (
                patch.object(worker, "load_config", return_value=config),
                patch.object(worker, "run_directory", return_value=run_dir),
                patch.object(
                    worker, "_run_fetch_script", side_effect=timed_out_without_progress
                ),
            ):
                exit_code = worker.main(["--job-id", job_id, "--time-budget", "60"])

            job = state.get_job(job_id)
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 0)
            self.assertEqual(job["status"], "queued")
            self.assertEqual(job["completed"], 0)
            self.assertEqual(job["pending"], 2)
            self.assertEqual(status["status"], "queued")
            self.assertIn("回到队列", str(status.get("message", "")))

    def test_failed_fetch_child_blocks_research_even_when_coverage_passes(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="research-worker-failed-",
            dir=ROOT / "tests",
        ) as directory:
            root = pathlib.Path(directory)
            run_dir = root / "run"
            state = FetchState(root / "state.sqlite3")
            job_id = "20260827T000000Z-failed01"
            symbols = [f"{index:06d}" for index in range(21)]
            state.register_assets("stock", symbols)
            for symbol in symbols[:-1]:
                state.mark_success(
                    "stock",
                    symbol,
                    data_end_date="2026-08-27",
                    source="test",
                    rows=1,
                    refresh_interval_seconds=86400,
                )
            state.create_job(
                job_id,
                kind="research",
                asset_scope="stock",
                metadata={
                    "mode": "daily",
                    "asset_scope": "stock",
                    "asset_types": ["stock"],
                    "symbols": {"stock": symbols},
                    "full_universe": False,
                    "target_date": "2026-08-27",
                },
            )

            config = copy.deepcopy(DEFAULTS)
            config["fetch"]["state_db"] = str(state.path)
            config["fetch"]["max_data_age_days"] = 5
            failed_fetch = {
                "asset_type": "stock",
                "status": "failed",
                "returncode": 1,
                "summary": None,
                "stdout_tail": "",
                "stderr_tail": "child crashed",
                "timeout_seconds": 60,
                "time_budget_seconds": 15,
            }
            with (
                patch.object(worker, "load_config", return_value=config),
                patch.object(worker, "run_directory", return_value=run_dir),
                patch.object(worker, "_run_fetch_script", return_value=failed_fetch),
                patch.object(worker, "_run_rolling_scan"),
                patch.object(worker, "_run_research_process") as research,
            ):
                exit_code = worker.main(["--job-id", job_id, "--time-budget", "60"])

            job = state.get_job(job_id)
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 1)
        self.assertEqual(job["status"], "failed")
        self.assertEqual(status["status"], "failed")
        self.assertFalse(status["fetch_gate"]["passed"])
        self.assertTrue(status["fetch_gate"]["child_failed"])
        research.assert_not_called()

    def test_minimum_coverage_allows_research_with_pending_assets(self) -> None:
        with tempfile.TemporaryDirectory(prefix="research-coverage-", dir=ROOT / "tests") as directory:
            root = pathlib.Path(directory)
            run_dir = root / "run"
            state = FetchState(root / "state.sqlite3")
            job_id = "20260827T000000Z-cover01"
            symbols = ["510300", "510500"]
            state.register_assets("etf", symbols)
            state.mark_success(
                "etf",
                "510300",
                data_end_date="2026-08-26",
                source="test",
                rows=1,
                refresh_interval_seconds=86400,
            )
            state.create_job(
                job_id,
                kind="research",
                asset_scope="etf",
                metadata={
                    "mode": "daily",
                    "asset_scope": "etf",
                    "asset_types": ["etf"],
                    "symbols": {"etf": symbols},
                    "target_date": "2026-08-27",
                    "target_dates": {"etf": "2026-08-26"},
                },
            )

            config = copy.deepcopy(DEFAULTS)
            config["fetch"]["state_db"] = str(state.path)
            config["fetch"]["min_universe_coverage"] = 0.5
            config["fetch"]["max_data_age_days"] = 5

            fetch_result = {
                "asset_type": "etf",
                "status": "partial",
                "returncode": 0,
                "summary": {"status": "partial"},
                "stdout_tail": "",
                "stderr_tail": "",
                "timeout_seconds": 30,
                "time_budget_seconds": 10,
            }
            research_result = {
                "run_id": "result-run",
                "status": "success",
                "result_path": str(root / "result"),
            }
            with (
                patch.object(worker, "load_config", return_value=config),
                patch.object(worker, "run_directory", return_value=run_dir),
                patch.object(worker, "_run_fetch_script", return_value=fetch_result),
                patch.object(
                    worker,
                    "_run_research_process",
                    return_value=research_result,
                ) as research,
            ):
                exit_code = worker.main(["--job-id", job_id, "--time-budget", "60"])

            job = state.get_job(job_id)
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 0)
            self.assertEqual(job["status"], "partial")
            self.assertEqual(status["status"], "partial")
            self.assertEqual(status["progress"]["coverage"], 0.5)
            self.assertEqual(status["progress"]["pending"], 1)
            research.assert_called_once()
            self.assertEqual(research.call_args.args, (job_id,))
            self.assertIn("timeout_seconds", research.call_args.kwargs)
            self.assertIn("heartbeat", research.call_args.kwargs)

    def test_fresh_targeted_cache_skips_fetch_and_preserves_routes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="research-cache-", dir=ROOT / "tests") as directory:
            root = pathlib.Path(directory)
            run_dir = root / "run"
            state = FetchState(root / "state.sqlite3")
            job_id = "20260827T000000Z-cache001"
            symbols = ["600999", "601696", "600030", "000776", "601688", "601211"]
            state.register_assets("stock", symbols)
            for symbol in symbols:
                state.mark_success(
                    "stock",
                    symbol,
                    data_end_date="2026-08-27",
                    source="test",
                    rows=1,
                    refresh_interval_seconds=86400,
                )
            state.create_job(
                job_id,
                kind="research",
                asset_scope="stock",
                metadata={
                    "mode": "daily",
                    "asset_scope": "stock",
                    "asset_types": ["stock"],
                    "symbols": {"stock": symbols},
                    "full_universe": False,
                    "target_date": "2026-08-27",
                    "target_dates": {"stock": "2026-08-27"},
                },
            )

            config = copy.deepcopy(DEFAULTS)
            config["fetch"]["state_db"] = str(state.path)
            config["fetch"]["max_data_age_days"] = 5
            research_result = {
                "run_id": "result-run",
                "status": "success",
                "result_path": str(root / "result"),
            }
            with (
                patch.object(worker, "load_config", return_value=config),
                patch.object(worker, "run_directory", return_value=run_dir),
                patch.object(worker, "_run_fetch_script") as fetch,
                patch.object(
                    worker,
                    "_run_research_process",
                    return_value=research_result,
                ) as research,
            ):
                exit_code = worker.main(["--job-id", job_id, "--time-budget", "60"])

            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 0)
            fetch.assert_not_called()
            self.assertEqual(status["fetch_results"], [
                {"asset_type": "stock", "status": "cache_hit", "requested": 6}
            ])
            self.assertEqual(research.call_args.args, (job_id,))
            self.assertIn("timeout_seconds", research.call_args.kwargs)
            self.assertIn("heartbeat", research.call_args.kwargs)

    def test_fetch_command_reserves_shutdown_grace_and_forwards_symbols(self) -> None:
        config = copy.deepcopy(DEFAULTS)
        config["fetch"]["request_timeout_seconds"] = 30
        config["fetch"]["shutdown_grace_seconds"] = 45
        symbols = ["600999", "601696", "600030", "000776", "601688", "601211"]

        with patch.object(worker.time, "monotonic", return_value=400.0):
            command, child_budget, timeout_seconds = worker._build_fetch_command(
                ROOT / "scripts" / "fetch_stock_data.py",
                asset_type="stock",
                config=config,
                time_budget=595,
                job_id="job",
                target_date="2026-08-27",
                symbols=symbols,
                deadline=1000.0,
            )

        self.assertLessEqual(child_budget, 555)
        self.assertGreaterEqual(timeout_seconds - child_budget, 45)
        self.assertEqual(command[-len(symbols):], symbols)
        self.assertNotIn("--asset-type", command)

    def test_fetch_command_caps_request_timeout_for_short_budget(self) -> None:
        config = copy.deepcopy(DEFAULTS)
        config["fetch"]["request_timeout_seconds"] = 30
        config["fetch"]["shutdown_grace_seconds"] = 45

        with patch.object(worker.time, "monotonic", return_value=400.0):
            command, child_budget, _ = worker._build_fetch_command(
                ROOT / "scripts" / "fetch_fund_data.py",
                asset_type="otc_fund",
                config=config,
                time_budget=15,
                job_id="job",
                target_date="2026-08-26",
                symbols=["000001", "000002"],
                deadline=460.0,
            )

        request_timeout = float(command[command.index("--request-timeout") + 1])
        self.assertLessEqual(request_timeout, child_budget / 2.0)
        self.assertEqual(
            command[command.index("--asset-type") + 1 :],
            ["otc_fund", "000001", "000002"],
        )

    def test_bounded_command_heartbeats_and_terminates_process_tree(self) -> None:
        config = copy.deepcopy(DEFAULTS)
        config["fetch"]["heartbeat_interval_seconds"] = 1
        config["fetch"]["shutdown_grace_seconds"] = 1
        process = Mock(pid=12345, returncode=None)
        process.wait.side_effect = [
            subprocess.TimeoutExpired(["test"], 1),
            0,
        ]
        heartbeat = Mock()

        with (
            patch.object(worker.subprocess, "Popen", return_value=process),
            patch.object(worker.time, "monotonic", side_effect=[0.0, 0.0, 2.0]),
            patch.object(worker, "_terminate_process_tree") as terminate,
        ):
            result = worker._run_bounded_command(
                ["test"],
                timeout_seconds=1.5,
                config=config,
                heartbeat=heartbeat,
            )

        self.assertTrue(result["timed_out"])
        heartbeat.assert_called_once_with()
        terminate.assert_called_once_with(process, 1.0)

    def test_windows_process_tree_cleanup_does_not_decode_taskkill_output(self) -> None:
        process = Mock(pid=12345)
        process.poll.return_value = None

        with (
            patch.object(worker.os, "name", "nt"),
            patch.object(worker.subprocess, "run") as run,
        ):
            worker._terminate_process_tree(process, 1.0)

        run.assert_called_once_with(
            ["taskkill", "/PID", "12345", "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        process.wait.assert_called_once_with(timeout=1.0)

    def test_bounded_command_replaces_invalid_utf8_output(self) -> None:
        config = copy.deepcopy(DEFAULTS)
        result = worker._run_bounded_command(
            [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "sys.stdout.buffer.write(bytes([111, 117, 116, 255])); "
                    "sys.stderr.buffer.write(bytes([101, 114, 114, 254]))"
                ),
            ],
            timeout_seconds=5,
            config=config,
        )

        self.assertFalse(result["timed_out"])
        self.assertEqual(result["returncode"], 0)
        self.assertEqual(result["stdout"], "out\ufffd")
        self.assertEqual(result["stderr"], "err\ufffd")


if __name__ == "__main__":
    unittest.main()
