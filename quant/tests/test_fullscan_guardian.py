"""全市场扫描监护 tick 语义的回归测试。

事故（2026-09-13 评审发现）：`DONE_FLAG`（`logs/codex-runs/fullscan-guardian/
guardian-done.json`）是**全局单文件、跨轮次持久**，不按 job_id 隔离。
`tick()` 原先只判断“文件是否存在”就短路返回 completed，于是：

- 上一轮残留的 done flag 会让**重新注册**的监护每次 tick 都短路返回 completed，
  既不巡检 paused.json、也不 resume 瞬时暂停 → 表现为“任务在、日志看着正常、
  但监护已静默失效”，比没有监护更危险。
- 反过来，无在跑作业时（非交易日 `latest.status` 同样是 `"paused"`）会落进
  `paused-transient` 分支，每 5 分钟调用一次 `resume()` 并写一个 diagnosis
  产物（整夜 12 次/小时 × 10 小时 ≈ 120 个文件）。

验收：done flag 只在 `job_id` 与当前作业一致时才可采信；无作业时监护保持被动
（idle），绝不自动修复。
"""
from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import fullscan_guardian as guardian


CURRENT_JOB_ID = "20260914T070000Z-newround"
STALE_JOB_ID = "20260911T065625Z-d6d4eede"


class GuardianTickTests(unittest.TestCase):
    """把所有路径常量重定向到临时目录，绝不碰真实 logs/。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="guardian-", dir=ROOT / "tests")
        base = pathlib.Path(self._tmp.name)
        self.base = base
        guardian_dir = base / "fullscan-guardian"
        redirects = (
            ("GUARDIAN_DIR", guardian_dir),
            ("GUARDIAN_LOG", guardian_dir / "guardian.log"),
            ("LATEST_GUARDIAN_PATH", guardian_dir / "latest-guardian.json"),
            ("DONE_FLAG", guardian_dir / "guardian-done.json"),
            ("LATEST_STATUS_PATH", base / "latest_status.json"),
            ("CONTROL_PATH", base / "control.json"),
            ("PAUSE_PATH", base / "paused.json"),
            ("ALERT_PATH", base / "alerts.log"),
            ("SCHED_LOG_PATH", base / "continuous_research.log"),
        )
        for name, value in redirects:
            patcher = patch.object(guardian, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        # 作业目录不参与本组断言，统一屏蔽，避免读到真实 run 产物。
        run_dir_patcher = patch.object(guardian, "_run_dir", return_value=None)
        run_dir_patcher.start()
        self.addCleanup(run_dir_patcher.stop)
        self.addCleanup(self._tmp.cleanup)

    def _write_json(self, path: pathlib.Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def _write_done_flag(self, job_id: str) -> None:
        self._write_json(
            guardian.DONE_FLAG,
            {
                "guardian_status": "completed",
                "job_id": job_id,
                "manifest_finished_at": "2026-09-11T11:51:16+00:00",
                "completed_at": "2026-09-11T19:52:30+08:00",
            },
        )

    def _write_status(self, **overrides: object) -> None:
        payload: dict[str, object] = {
            "status": "running",
            "phase": "fetch",
            "job_id": CURRENT_JOB_ID,
        }
        payload.update(overrides)
        self._write_json(guardian.LATEST_STATUS_PATH, payload)

    def test_stale_done_flag_from_previous_round_is_ignored(self) -> None:
        """上一轮残留的 done flag 不得让新作业的监护空转（本次修复的核心）。"""
        self._write_done_flag(STALE_JOB_ID)
        self._write_status()
        summary, code = guardian.tick()
        self.assertNotEqual(summary.get("guardian_status"), "completed")
        self.assertEqual(summary.get("guardian_status"), "healthy-running")
        self.assertEqual(summary.get("job_id"), CURRENT_JOB_ID)
        self.assertEqual(code, 0)

    def test_done_flag_matching_current_job_still_completes(self) -> None:
        """同一作业的 done flag 仍须短路为 completed（保护既有语义）。"""
        self._write_done_flag(CURRENT_JOB_ID)
        self._write_status()
        summary, code = guardian.tick()
        self.assertEqual(summary.get("guardian_status"), "completed")
        self.assertTrue(summary.get("done"))
        self.assertEqual(code, guardian.EXIT_COMPLETED)

    def test_no_active_job_stays_idle_and_never_resumes(self) -> None:
        """非交易日 latest.status 也是 paused，不得判 paused-transient 去 resume。"""
        self._write_status(
            status="paused",
            phase="calendar",
            job_id=None,
            run_id=None,
            last_error="2026-09-13 is not a trading day",
        )
        with patch("continuous_research.resume") as resume_mock:
            summary, code = guardian.tick()
        self.assertEqual(summary.get("guardian_status"), "idle")
        self.assertEqual(code, 0)
        resume_mock.assert_not_called()

    def test_no_active_job_with_stale_flag_stays_idle(self) -> None:
        """stale flag + 无作业：应判 idle，而不是把上一轮的完成态当成本轮完成。"""
        self._write_done_flag(STALE_JOB_ID)
        self._write_status(status="paused", phase="calendar", job_id=None, run_id=None)
        summary, code = guardian.tick()
        self.assertEqual(summary.get("guardian_status"), "idle")
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
