from __future__ import annotations

import contextlib
import copy
import io
import json
import pathlib
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from typing import Any
from unittest.mock import Mock, patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from assistant.config import DEFAULTS
from fetch_state import FetchState
import run_research_worker as worker
import run_assistant_research as queued_cli


class ResearchWorkerTests(unittest.TestCase):
    def test_parses_bounded_qlib_build_summary(self) -> None:
        summary = worker._qlib_build_summary(
            "noise\n"
            'QLIB_BUILD_SUMMARY={"cache_hit":true,'
            '"source_fingerprint_digest":"provider-v1"}\n'
        )
        self.assertEqual(
            summary,
            {
                "cache_hit": True,
                "source_fingerprint_digest": "provider-v1",
            },
        )

    def test_merges_qlib_build_progress_only_when_phase_sequence_advances(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            run_dir = pathlib.Path(temporary_dir)
            progress_path = run_dir / "qlib-build-progress.json"
            status = {"phase_seq": 3, "qlib_build_phase_seq": 0}
            progress_path.write_text(
                json.dumps(
                    {
                        "phase": "loading_sources",
                        "phase_seq": 2,
                        "completed": 4,
                        "total": 10,
                        "updated_at": "2026-08-31T00:00:00+00:00",
                        "cache_hit": False,
                        "source_fingerprint_digest": "digest",
                    }
                ),
                encoding="utf-8",
            )

            self.assertTrue(worker._merge_qlib_build_progress(run_dir, status))
            self.assertEqual(status["qlib_build_phase"], "loading_sources")
            self.assertEqual(status["qlib_build_phase_seq"], 2)
            self.assertEqual(status["qlib_build_completed"], 4)
            self.assertEqual(status["qlib_build_total"], 10)
            self.assertEqual(status["qlib_build_updated_at"], "2026-08-31T00:00:00+00:00")
            self.assertFalse(worker._merge_qlib_build_progress(run_dir, status))

            progress_path.write_text(
                json.dumps({"phase": "complete", "phase_seq": 1}),
                encoding="utf-8",
            )
            self.assertFalse(worker._merge_qlib_build_progress(run_dir, status))

    def test_reset_qlib_build_progress_discards_previous_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            run_dir = pathlib.Path(temporary_dir)
            progress_path = run_dir / "qlib-build-progress.json"
            progress_path.write_text("{}", encoding="utf-8")
            status = {
                "phase": "qlib_build",
                "qlib_build_phase": "complete",
                "qlib_build_phase_seq": 8,
                "qlib_build_completed": 1,
                "qlib_build_total": 1,
                "qlib_build_updated_at": "now",
                "qlib_build": {"cache_hit": True},
                "cache_hit": True,
                "source_fingerprint_digest": "old",
            }

            worker._reset_qlib_build_progress(run_dir, status)
            self.assertFalse(progress_path.exists())
            self.assertEqual(status, {"phase": "qlib_build"})

    def test_reset_research_progress_discards_an_earlier_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            run_dir = pathlib.Path(temporary_dir)
            progress_path = run_dir / "research-progress.json"
            progress_path.write_text(
                json.dumps({"phase": "backtest", "phase_seq": 12}),
                encoding="utf-8",
            )
            status = {
                "phase": "research",
                "research_phase": "backtest",
                "research_phase_seq": 12,
                "cache_hit": False,
                "cache_miss_reason": "cache_file_missing_or_invalid",
                "provider_end": "2026-08-28",
            }

            worker._reset_research_progress(run_dir, status)

            self.assertFalse(progress_path.exists())
            self.assertEqual(status, {"phase": "research"})
            progress_path.write_text(
                json.dumps({"phase": "provider_cache_check", "phase_seq": 1}),
                encoding="utf-8",
            )
            self.assertTrue(worker._merge_research_progress(run_dir, status))
            self.assertEqual(status["research_phase_seq"], 1)
            self.assertEqual(status["research_phase"], "provider_cache_check")

    def test_queued_cli_reuses_original_run_id_and_directory(self) -> None:
        run_id = "20260827T000000Z-acde0001"
        run_dir = ROOT / "tests" / "queued-run-placeholder"
        with patch.object(
            queued_cli,
            "_queued_request",
            return_value=("on_demand", "stock", {"stock": ["002942"]}, "quick"),
        ), patch.object(
            queued_cli, "run_directory", return_value=run_dir
        ), patch.object(
            queued_cli,
            "_run_fixed_research",
            return_value={"run_id": run_id, "status": "success"},
        ) as run_fixed:
            exit_code = queued_cli.main(["--queued-run-id", run_id])

        self.assertEqual(exit_code, 0)
        run_fixed.assert_called_once_with(
            mode="on_demand",
            asset_scope="stock",
            symbols_by_type={"stock": ["002942"]},
            depth="quick",
            force_sync=True,
            run_id_override=run_id,
            run_dir_override=run_dir,
        )

    def test_queued_cli_defaults_depth_for_legacy_manifests(self) -> None:
        run_id = "20260827T000000Z-acde0002"
        run_dir = ROOT / "tests" / "queued-run-placeholder"
        with patch.object(
            queued_cli,
            "_queued_request",
            return_value=("daily", "stock", None, "standard"),
        ), patch.object(
            queued_cli, "run_directory", return_value=run_dir
        ), patch.object(
            queued_cli,
            "_run_fixed_research",
            return_value={"run_id": run_id, "status": "success"},
        ) as run_fixed:
            exit_code = queued_cli.main(["--queued-run-id", run_id])

        self.assertEqual(exit_code, 0)
        self.assertEqual(run_fixed.call_args.kwargs["depth"], "standard")

    def test_compact_status_keeps_bounded_provider_build_summary(self) -> None:
        compact = worker._compact_status(
            {
                "job_id": "job",
                "status": "running",
                "worker_pid": 1234,
                "qlib_build_phase": "writing_features",
                "qlib_build_phase_seq": 5,
                "qlib_build_completed": 12,
                "qlib_build_total": 20,
                "qlib_build_updated_at": "now",
                "qlib_build": {
                    "returncode": 0,
                    "timeout_seconds": 900,
                    "child_pid": 1234,
                    "cache_hit": True,
                    "stdout_tail": "x" * 100000,
                    "stderr_tail": "y" * 100000,
                },
            }
        )

        self.assertEqual(
            compact["qlib_build"],
            {
                "returncode": 0,
                "timeout_seconds": 900,
                "child_pid": 1234,
                "cache_hit": True,
            },
        )
        self.assertEqual(compact["worker_pid"], 1234)
        self.assertEqual(compact["qlib_build_phase"], "writing_features")
        self.assertEqual(compact["qlib_build_phase_seq"], 5)
        self.assertEqual(compact["qlib_build_completed"], 12)
        self.assertEqual(compact["qlib_build_total"], 20)
        self.assertEqual(compact["qlib_build_updated_at"], "now")
        self.assertLess(len(json.dumps(compact)), 1000)

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
                        "requested": 3,
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
        self.assertEqual(compact["fetch_results"][0]["requested"], 3)
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
                "run_id": job_id,
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
            # 研究阶段不再被波次预算截断（模型缓存冷启动需要更长预算），
            # 但仍受 fetch.timeout_seconds 上限约束。
            self.assertGreater(research.call_args.kwargs["timeout_seconds"], 15.0)
            self.assertLessEqual(
                research.call_args.kwargs["timeout_seconds"],
                float(config["fetch"]["timeout_seconds"]),
            )
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
                "run_id": job_id,
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

    def test_quick_depth_skips_fetch_scan_and_gate(self) -> None:
        with tempfile.TemporaryDirectory(prefix="research-quick-", dir=ROOT / "tests") as directory:
            root = pathlib.Path(directory)
            run_dir = root / "run"
            state = FetchState(root / "state.sqlite3")
            job_id = "20260827T000000Z-quick001"
            symbols = ["600999", "601696", "600030"]
            state.register_assets("stock", symbols)
            # 全部标的都落后于 target_date：standard 会触发抓取，quick 必须跳过。
            for symbol in symbols:
                state.mark_success(
                    "stock",
                    symbol,
                    data_end_date="2026-08-20",
                    source="test",
                    rows=1,
                    refresh_interval_seconds=86400,
                )
            state.create_job(
                job_id,
                kind="research",
                asset_scope="stock",
                metadata={
                    "mode": "on_demand",
                    "depth": "quick",
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
                "run_id": job_id,
                "status": "success",
                "result_path": str(run_dir),
            }
            with (
                patch.object(worker, "load_config", return_value=config),
                patch.object(worker, "run_directory", return_value=run_dir),
                patch.object(worker, "_run_fetch_script") as fetch,
                patch.object(worker, "_run_rolling_scan") as scan,
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
            scan.assert_not_called()
            self.assertEqual(status["depth"], "quick")
            self.assertEqual(
                status["fetch_results"],
                [
                    {
                        "asset_type": "stock",
                        "status": "skipped_by_depth",
                        "requested": 3,
                    }
                ],
            )
            self.assertTrue(status["fetch_gate"]["passed"])
            self.assertEqual(status["fetch_gate"]["bypassed_by_depth"], "quick")
            self.assertTrue(status["stage_timings"])
            self.assertIn("research", status["stage_timings"])
            research.assert_called_once()

    def test_set_phase_accumulates_stage_timings(self) -> None:
        status: dict[str, Any] = {}
        worker._set_phase(status, "fetch")
        worker._set_phase(status, "qlib_build")
        worker._set_phase(status, "qlib_build")
        worker._set_phase(status, "research")

        self.assertEqual(status["phase"], "research")
        self.assertEqual(status["phase_seq"], 3)
        self.assertEqual(
            sorted(status["stage_timings"]),
            ["fetch", "qlib_build"],
        )
        for entry in status["stage_timings"].values():
            self.assertIn("started_at", entry)
            self.assertIn("finished_at", entry)
            self.assertGreaterEqual(entry["elapsed_seconds"], 0.0)

    def test_close_stage_entry_never_double_counts(self) -> None:
        # 终态收口后异常分支可能再次收口同一阶段；elapsed 不允许翻倍。
        status: dict[str, Any] = {}
        worker._set_phase(status, "research")
        worker._close_current_stage(status)
        elapsed_once = status["stage_timings"]["research"]["elapsed_seconds"]
        worker._close_current_stage(status)
        self.assertEqual(
            status["stage_timings"]["research"]["elapsed_seconds"],
            elapsed_once,
        )
        # 新阶段照常开始计时（entry 在其收口时才落盘）。
        worker._set_phase(status, "publish")
        self.assertEqual(status["phase"], "publish")
        self.assertIsNotNone(status["phase_started_at"])
        worker._close_current_stage(status)
        self.assertIn("started_at", status["stage_timings"]["publish"])

    def test_print_status_is_ascii_safe(self) -> None:
        # worker 的 stdout 可能是 GBK 管道；状态里带着抓取子进程输出
        # 残留的 U+FFFD 时打印绝不能像旧版那样抛 UnicodeEncodeError。
        status = {
            "run_id": "20260902T164252Z-e7a3bfbc",
            "message": "定向请求已入队；后台只会检查和刷新指定代码。",
            "error": "bad\ufffdchar",
        }
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            worker._print_status(status)
        payload = captured.getvalue()
        payload.encode("ascii")  # 不可编码字符必须已被转义
        self.assertEqual(json.loads(payload)["error"], "bad\ufffdchar")

    def _make_two_type_job(
        self,
        root: pathlib.Path,
        job_id: str,
        *,
        parallel: bool,
    ) -> tuple[pathlib.Path, FetchState, dict[str, Any]]:
        run_dir = root / "run"
        state = FetchState(root / "state.sqlite3")
        state.register_assets("stock", ["600999"])
        state.register_assets("etf", ["510300"])
        state.create_job(
            job_id,
            kind="research",
            asset_scope="all",
            metadata={
                "mode": "daily",
                "asset_scope": "all",
                "asset_types": ["stock", "etf"],
                "symbols": {"stock": [], "etf": []},
                "full_universe": True,
                "target_date": "2026-08-27",
                "target_dates": {"stock": "2026-08-27", "etf": "2026-08-26"},
            },
        )
        config = copy.deepcopy(DEFAULTS)
        config["fetch"]["state_db"] = str(state.path)
        config["fetch"]["max_data_age_days"] = 5
        config["fetch"]["parallel_asset_fetch"] = parallel
        return run_dir, state, config

    def test_parallel_fetch_lanes_run_stock_and_fund_concurrently(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="research-parallel-",
            dir=ROOT / "tests",
        ) as directory:
            root = pathlib.Path(directory)
            job_id = "20260827T000000Z-paral001"
            run_dir, state, config = self._make_two_type_job(
                root, job_id, parallel=True
            )
            # 两个通道必须同时在 _run_fetch_script 内会合，否则 Barrier 超时。
            barrier = threading.Barrier(2, timeout=10)

            def fetch(script: object, **kwargs: Any) -> dict[str, Any]:
                barrier.wait()
                return {
                    "asset_type": str(kwargs["asset_type"]),
                    "status": "success",
                    "time_budget_seconds": 60,
                }

            with (
                patch.object(worker, "load_config", return_value=config),
                patch.object(worker, "run_directory", return_value=run_dir),
                patch.object(worker, "_run_fetch_script", side_effect=fetch) as fetch_mock,
                patch.object(worker, "_run_rolling_scan"),
                patch.object(
                    worker,
                    "_run_research_process",
                    return_value={"run_id": job_id, "status": "success"},
                ),
            ):
                exit_code = worker.main(["--job-id", job_id, "--time-budget", "60"])

            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            [call.kwargs["asset_type"] for call in fetch_mock.call_args_list],
            ["stock", "etf"],
        )
        self.assertEqual(
            [item["asset_type"] for item in status["fetch_results"]],
            ["stock", "etf"],
        )
        self.assertFalse(status["fetch_gate"]["child_failed"])

    def test_parallel_fetch_disabled_keeps_lanes_serial(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="research-serial-",
            dir=ROOT / "tests",
        ) as directory:
            root = pathlib.Path(directory)
            job_id = "20260827T000000Z-serial01"
            run_dir, state, config = self._make_two_type_job(
                root, job_id, parallel=False
            )
            in_flight = {"value": False}

            def fetch(script: object, **kwargs: Any) -> dict[str, Any]:
                self.assertFalse(
                    in_flight["value"],
                    "关闭并行后不允许出现并发抓取",
                )
                in_flight["value"] = True
                time.sleep(0.05)
                in_flight["value"] = False
                return {
                    "asset_type": str(kwargs["asset_type"]),
                    "status": "success",
                }

            with (
                patch.object(worker, "load_config", return_value=config),
                patch.object(worker, "run_directory", return_value=run_dir),
                patch.object(worker, "_run_fetch_script", side_effect=fetch),
                patch.object(worker, "_run_rolling_scan"),
                patch.object(
                    worker,
                    "_run_research_process",
                    return_value={"run_id": job_id, "status": "success"},
                ),
            ):
                exit_code = worker.main(["--job-id", job_id, "--time-budget", "60"])

            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            [item["asset_type"] for item in status["fetch_results"]],
            ["stock", "etf"],
        )

    def test_targeted_stock_refresh_reuses_provider_without_build(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="research-targeted-refresh-",
            dir=ROOT / "tests",
        ) as directory:
            root = pathlib.Path(directory)
            run_dir = root / "run"
            state = FetchState(root / "state.sqlite3")
            job_id = "20260827T000000Z-acde0002"
            symbol = "002942"
            state.register_assets("stock", [symbol])
            state.create_job(
                job_id,
                kind="research",
                asset_scope="stock",
                metadata={
                    "mode": "on_demand",
                    "asset_scope": "stock",
                    "asset_types": ["stock"],
                    "symbols": {"stock": [symbol]},
                    "full_universe": False,
                    "target_date": "2026-08-27",
                    "target_dates": {"stock": "2026-08-27"},
                },
            )
            config = copy.deepcopy(DEFAULTS)
            config["fetch"]["state_db"] = str(state.path)
            config["fetch"]["max_data_age_days"] = 5
            research_result = {
                "run_id": job_id,
                "status": "success",
                "result_path": str(run_dir),
            }

            def refresh_target(*args, **kwargs):
                self.assertEqual(kwargs["asset_type"], "stock")
                self.assertEqual(kwargs["symbols"], [symbol])
                state.mark_success(
                    "stock",
                    symbol,
                    data_end_date="2026-08-27",
                    source="test",
                    rows=300,
                    refresh_interval_seconds=86400,
                )
                return {
                    "asset_type": "stock",
                    "status": "success",
                    "returncode": 0,
                    "summary": {
                        "status": "success",
                        "requested": 1,
                        "processed": 1,
                        "ok": 1,
                        "failed": 0,
                        "pending": 0,
                    },
                    "stdout_tail": "",
                    "stderr_tail": "",
                    "timeout_seconds": 60,
                    "time_budget_seconds": 15,
                }

            with (
                patch.object(worker, "load_config", return_value=config),
                patch.object(worker, "run_directory", return_value=run_dir),
                patch.object(
                    worker,
                    "_run_fetch_script",
                    side_effect=refresh_target,
                ) as fetch,
                patch.object(worker, "_run_rolling_scan"),
                patch.object(worker, "_run_bounded_command") as build,
                patch.object(
                    worker,
                    "_run_research_process",
                    return_value=research_result,
                ) as research,
            ):
                exit_code = worker.main(
                    ["--job-id", job_id, "--time-budget", "60"]
                )

            status_bytes = (run_dir / "status.json").read_bytes()
            status = json.loads(status_bytes.decode("utf-8"))

        self.assertEqual(exit_code, 0)
        fetch.assert_called_once()
        build.assert_not_called()
        research.assert_called_once()
        self.assertEqual(
            status["provider_strategy"],
            "reuse_existing_provider_for_targeted_run",
        )
        self.assertNotIn("qlib_build", status)
        self.assertNotIn("research_result", status)
        self.assertLess(len(status_bytes), 16 * 1024)

    def test_full_universe_stock_refresh_builds_provider_with_run_progress_file(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="research-full-universe-",
            dir=ROOT / "tests",
        ) as directory:
            root = pathlib.Path(directory)
            run_dir = root / "run"
            state = FetchState(root / "state.sqlite3")
            job_id = "20260827T000000Z-acde0003"
            symbol = "002942"
            state.register_assets("stock", [symbol])
            state.create_job(
                job_id,
                kind="research",
                asset_scope="stock",
                metadata={
                    "mode": "daily",
                    "asset_scope": "stock",
                    "asset_types": ["stock"],
                    "symbols": {"stock": []},
                    "full_universe": True,
                    "target_date": "2026-08-27",
                    "target_dates": {"stock": "2026-08-27"},
                },
            )
            config = copy.deepcopy(DEFAULTS)
            config["fetch"]["state_db"] = str(state.path)
            config["fetch"]["max_data_age_days"] = 0
            research_result = {
                "run_id": job_id,
                "status": "success",
                "result_path": str(run_dir),
            }

            def refresh_universe(*args, **kwargs):
                self.assertIsNone(kwargs["symbols"])
                state.mark_success(
                    "stock",
                    symbol,
                    data_end_date="2026-08-27",
                    source="test",
                    rows=300,
                    refresh_interval_seconds=86400,
                )
                return {
                    "asset_type": "stock",
                    "status": "success",
                    "returncode": 0,
                    "summary": {
                        "status": "success",
                        "requested": 1,
                        "processed": 1,
                        "ok": 1,
                        "failed": 0,
                        "pending": 0,
                    },
                    "stdout_tail": "",
                    "stderr_tail": "",
                    "timeout_seconds": 60,
                    "time_budget_seconds": 15,
                }

            def build_provider(command, **kwargs):
                self.assertIn("--progress-file", command)
                progress_path = pathlib.Path(
                    command[command.index("--progress-file") + 1]
                )
                self.assertEqual(progress_path, run_dir / "qlib-build-progress.json")
                progress_path.parent.mkdir(parents=True, exist_ok=True)
                progress_path.write_text(
                    json.dumps(
                        {
                            "phase": "complete",
                            "phase_seq": 3,
                            "completed": 1,
                            "total": 1,
                            "updated_at": "now",
                        }
                    ),
                    encoding="utf-8",
                )
                return {
                    "returncode": 0,
                    "stdout": "QLIB_BUILD_SUMMARY={\"cache_hit\":false}\n",
                    "stderr": "",
                    "timeout_seconds": 60,
                    "child_pid": 123,
                    "timed_out": False,
                }

            with (
                patch.object(worker, "load_config", return_value=config),
                patch.object(worker, "run_directory", return_value=run_dir),
                patch.object(
                    worker,
                    "_run_fetch_script",
                    side_effect=refresh_universe,
                ) as fetch,
                patch.object(worker, "_run_rolling_scan"),
                patch.object(
                    worker,
                    "_run_bounded_command",
                    side_effect=build_provider,
                ) as build,
                patch.object(
                    worker,
                    "_run_research_process",
                    return_value=research_result,
                ) as research,
            ):
                exit_code = worker.main(["--job-id", job_id, "--time-budget", "60"])

            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        fetch.assert_called_once()
        build.assert_called_once()
        research.assert_called_once()
        self.assertEqual(status["provider_strategy"], "rebuild_if_source_changed")
        self.assertEqual(status["qlib_build_phase"], "complete")
        self.assertEqual(status["qlib_build_completed"], 1)
        self.assertEqual(status["qlib_build_total"], 1)

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

    def test_stage_timeout_is_capped_by_global_deadline(self) -> None:
        with patch.object(worker.time, "monotonic", return_value=400.0):
            timeout_seconds = worker._remaining_stage_timeout(
                deadline=460.0,
                configured_timeout=900.0,
                shutdown_grace=45.0,
                stage="research",
            )

        self.assertEqual(timeout_seconds, 15.0)

    def test_stage_timeout_rejects_launch_when_only_shutdown_grace_remains(
        self,
    ) -> None:
        with (
            patch.object(worker.time, "monotonic", return_value=415.0),
            self.assertRaisesRegex(
                TimeoutError,
                "worker time budget exhausted before research",
            ),
        ):
            worker._remaining_stage_timeout(
                deadline=460.0,
                configured_timeout=900.0,
                shutdown_grace=45.0,
                stage="research",
            )

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
