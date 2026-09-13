"""P0-4：内层研究子进程的成败必须以产物为准，而不是退出码/stdout 解析。

事故（2026-09-11）：job `20260911T065625Z-d6d4eede` 已成功发布
（`manifest.json` status=success、`current.json` 已更新、`publication.state=full`），
内层子进程却返回了 {0,1} 之外的退出码，`_run_research_process` 只按退出码判定，
于是整轮被记成失败 → `failure_streak` 累到 3 → `_auto_pause_on_streak`
在 22:05:49 把整个调度器停摆。

《研究管线不变量》§7：判断运行成败以 manifest.json / current.json 为准。
"""
from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_research_worker as worker


JOB_ID = "20260911T065625Z-d6d4eede"


class PublishedResearchResultTests(unittest.TestCase):
    def _write_manifest(self, run_dir: pathlib.Path, **overrides: object) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        payload: dict[str, object] = {
            "run_id": JOB_ID,
            "status": "success",
            "failure_reasons": [],
            "data_end_date": "2026-09-11",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "publication": {"published": True, "state": "full"},
        }
        payload.update(overrides)
        (run_dir / "manifest.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    def _invoke(self, run_root: pathlib.Path, *, invoked_at: datetime):
        with patch.object(worker, "run_directory", return_value=run_root / JOB_ID):
            return worker._published_research_result(JOB_ID, invoked_at=invoked_at)

    def test_success_manifest_is_recovered(self) -> None:
        with tempfile.TemporaryDirectory(prefix="p04-worker-", dir=ROOT / "tests") as directory:
            run_root = pathlib.Path(directory)
            invoked_at = datetime.now(timezone.utc)
            self._write_manifest(run_root / JOB_ID)
            result = self._invoke(run_root, invoked_at=invoked_at)
            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(result["run_id"], JOB_ID)
            self.assertEqual(result["status"], "success")
            self.assertTrue(result["recovered_from_artifacts"])
            self.assertEqual(result["result_path"], str(run_root / JOB_ID))

    def test_partial_manifest_is_recovered(self) -> None:
        # 2026-09-13 P0-3 × P0-4 交汇路径：分类别发布后 manifest.status 可能是
        # "partial"（股票门槛失败但 ETF/基金已发布），对 worker 而言与 success
        # 同为"已发布"，不得因内层退出码异常而记整轮失败。
        with tempfile.TemporaryDirectory(prefix="p04-worker-", dir=ROOT / "tests") as directory:
            run_root = pathlib.Path(directory)
            invoked_at = datetime.now(timezone.utc)
            self._write_manifest(
                run_root / JOB_ID,
                status="partial",
                failure_reasons=["stock_model_or_risk_gate_failed"],
                publication={
                    "state": "partial",
                    "published": True,
                    "ready_asset_types": ["etf", "otc_fund"],
                    "blocked_asset_types": ["stock"],
                },
            )
            result = self._invoke(run_root, invoked_at=invoked_at)
            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["publication_state"], "partial")
            self.assertTrue(result["recovered_from_artifacts"])

    def test_stale_manifest_from_previous_pass_is_rejected(self) -> None:
        # run 目录会被同一 job_id 的后续 pass 复用：上一次 pass 的成功产物
        # 不能当作本轮成功。
        with tempfile.TemporaryDirectory(prefix="p04-worker-", dir=ROOT / "tests") as directory:
            run_root = pathlib.Path(directory)
            invoked_at = datetime.now(timezone.utc)
            self._write_manifest(
                run_root / JOB_ID,
                finished_at=(invoked_at - timedelta(minutes=30)).isoformat(),
            )
            self.assertIsNone(self._invoke(run_root, invoked_at=invoked_at))

    def test_failed_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="p04-worker-", dir=ROOT / "tests") as directory:
            run_root = pathlib.Path(directory)
            invoked_at = datetime.now(timezone.utc)
            self._write_manifest(run_root / JOB_ID, status="failed")
            self.assertIsNone(self._invoke(run_root, invoked_at=invoked_at))

    def test_mismatched_run_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="p04-worker-", dir=ROOT / "tests") as directory:
            run_root = pathlib.Path(directory)
            invoked_at = datetime.now(timezone.utc)
            self._write_manifest(run_root / JOB_ID, run_id="20260911T000000Z-deadbeef")
            self.assertIsNone(self._invoke(run_root, invoked_at=invoked_at))

    def test_missing_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="p04-worker-", dir=ROOT / "tests") as directory:
            run_root = pathlib.Path(directory)
            invoked_at = datetime.now(timezone.utc)
            self.assertIsNone(self._invoke(run_root, invoked_at=invoked_at))

    def test_unparseable_finished_at_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="p04-worker-", dir=ROOT / "tests") as directory:
            run_root = pathlib.Path(directory)
            invoked_at = datetime.now(timezone.utc)
            self._write_manifest(run_root / JOB_ID, finished_at="not-a-timestamp")
            self.assertIsNone(self._invoke(run_root, invoked_at=invoked_at))


class InnerOutputPersistenceTests(unittest.TestCase):
    def test_inner_output_is_persisted_for_diagnosis(self) -> None:
        # P0-4 定位时缺的正是这份证据：子进程输出原先落在临时文件里，退出即删。
        with tempfile.TemporaryDirectory(prefix="p04-log-", dir=ROOT / "tests") as directory:
            log_dir = pathlib.Path(directory) / "workers"
            with patch.object(worker, "RESEARCH_WORKER_LOG_DIR", log_dir):
                path = worker._persist_inner_research_output(
                    JOB_ID,
                    {
                        "child_pid": 13824,
                        "returncode": 2,
                        "timed_out": False,
                        "timeout_seconds": 3600.0,
                        "stdout": "{\"status\": \"error\"}\n",
                        "stderr": "qlib successfully initialized\n",
                    },
                )
            self.assertIsNotNone(path)
            assert path is not None
            text = pathlib.Path(path).read_text(encoding="utf-8")
            self.assertIn("returncode=2", text)
            self.assertIn("pid=13824", text)
            self.assertIn("status", text)
            stderr_path = log_dir / f"{JOB_ID}.research.stderr.log"
            self.assertTrue(stderr_path.is_file())
            self.assertIn(
                "qlib successfully initialized",
                stderr_path.read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
