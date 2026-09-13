"""Full-market scan guardian: supervise, diagnose, safe-fix, self-terminate.

Design notes (must-respect invariants, see quant/docs/研究管线不变量.md):
- Read-only tick: only bounded JSON aggregates + log tails (<10s, <100 lines).
- NEVER start a second worker wave; NEVER modify gates/thresholds, data_source
  order, breaker, price guards, training window, or DELETE asset rows.
- Structured failure classification only (manifest/status.json failure_reasons
  + rule_gates.json stock checks). job.last_error may carry stderr false
  positives (invariant section 7) and must NOT drive decisions.
- The only allowed auto-fix is resuming a TRANSIENT pause via the scheduler's
  own resume() (clears streak, gives a 3-round window). Deterministic
  gate/model/drawdown failures stay paused for a human decision.
- Completion = run manifest status PUBLISHED (success OR partial, per P0-3
  category-wise publication) AND current.json status published with the same
  run_id. Then write the done flag (carrying job_id), best-effort unregister the
  scheduled task, and exit with code 10. The done flag is a single GLOBAL file, so
  it is honored only when its job_id matches the current job; a stale flag left by
  a previous round must never short-circuit a freshly registered guardian.
- No active job -> stay passive (idle) and never auto-fix. The scheduler reports
  status "paused" on non-trading days too, so treating that as a transient pause
  would call resume() every 5 minutes all night and pile up diagnosis artifacts.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

QUANT_ROOT = Path(__file__).resolve().parents[1]
if str(QUANT_ROOT) not in sys.path:
    sys.path.insert(0, str(QUANT_ROOT))

TIMEZONE = ZoneInfo("Asia/Shanghai")
CONTINUOUS_DIR = QUANT_ROOT / "logs" / "continuous_research"
GUARDIAN_DIR = QUANT_ROOT / "logs" / "codex-runs" / "fullscan-guardian"
GUARDIAN_LOG = GUARDIAN_DIR / "guardian.log"
LATEST_GUARDIAN_PATH = GUARDIAN_DIR / "latest-guardian.json"
DONE_FLAG = GUARDIAN_DIR / "guardian-done.json"
LATEST_STATUS_PATH = CONTINUOUS_DIR / "latest_status.json"
CONTROL_PATH = CONTINUOUS_DIR / "control.json"
PAUSE_PATH = CONTINUOUS_DIR / "paused.json"
ALERT_PATH = CONTINUOUS_DIR / "alerts.log"
SCHED_LOG_PATH = CONTINUOUS_DIR / "continuous_research.log"
TASK_PATH = "\\DSH-股票助手\\"
TASK_NAME = "全市场扫描监护"

# Exit code 10 = scan completed, results published, guardian may self-remove.
EXIT_COMPLETED = 10

DETERMINISTIC_TOKENS = ("gate", "model", "drawdown", "回撤", "门槛", "risk")

# 2026-09-13：P0-3 分类别发布后，manifest/current 的 status 是三态
# success / partial / failed，其中 partial = 至少一个资产类别已发布
# （被拦类别权重归零并标注）。监护判"本轮是否已发布"必须认 partial，
# 否则 partial 轮永远不算 completed → 不写 DONE_FLAG、不自我注销计划任务。
PUBLISHED_RUN_STATUSES = ("success", "partial")


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value
    except (OSError, ValueError):
        return default


def _tail_text(path: Path, max_bytes: int = 8192, max_lines: int = 10) -> list[str]:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            text = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    return text.splitlines()[-max_lines:]


def _run_dir(job_id: str) -> Path | None:
    try:
        from assistant.paths import run_directory

        return run_directory(job_id)
    except (ValueError, FileNotFoundError, ImportError):
        return None


def _compact_manifest(manifest: Any) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        return {}
    error = manifest.get("error")
    return {
        "status": manifest.get("status"),
        "finished_at": manifest.get("finished_at"),
        "data_end_date": manifest.get("data_end_date"),
        "failure_reasons": manifest.get("failure_reasons"),
        # 2026-09-13 P0-3：带上分类别发布状态，诊断产物才能看出
        # "已发布但被拦类别"与"整体失败"的区别。
        "publication": manifest.get("publication"),
        "error_tail": str(error)[-500:] if error else None,
    }


def _stock_gate_summary(gates: Any) -> dict[str, Any]:
    if not isinstance(gates, dict):
        return {}
    stock = gates.get("stock")
    if not isinstance(stock, dict):
        return {}
    checks = stock.get("checks")
    failed_checks: list[str] = []
    if isinstance(checks, dict):
        failed_checks = sorted(
            str(name) for name, passed in checks.items() if not passed
        )
    return {
        "passed": stock.get("passed"),
        "failed_checks": failed_checks[:12],
    }


def _has_deterministic_signal(*reasons_sources: Any) -> bool:
    joined: list[str] = []
    for source in reasons_sources:
        if isinstance(source, list):
            joined.extend(str(item) for item in source)
        elif source:
            joined.append(str(source))
    text = " ".join(joined).lower()
    return any(token in text for token in DETERMINISTIC_TOKENS)


def _try_self_unregister() -> bool:
    commands = [
        ["schtasks.exe", "/Delete", "/TN", f"{TASK_PATH}{TASK_NAME}", "/F"],
    ]
    for command in commands:
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, timeout=30, check=False
            )
            if completed.returncode == 0:
                return True
        except (OSError, subprocess.SubprocessError):
            continue
    return False


def _append_log(record: dict[str, Any]) -> None:
    try:
        GUARDIAN_DIR.mkdir(parents=True, exist_ok=True)
        with GUARDIAN_LOG.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass


def tick() -> tuple[dict[str, Any], int]:
    GUARDIAN_DIR.mkdir(parents=True, exist_ok=True)

    latest = _read_json(LATEST_STATUS_PATH, {})
    control = _read_json(CONTROL_PATH, {})
    if not isinstance(latest, dict):
        latest = {}
    if not isinstance(control, dict):
        control = {}
    paused = PAUSE_PATH.is_file()
    pause_record = _read_json(PAUSE_PATH, {})

    job_id = str(latest.get("job_id") or latest.get("run_id") or "")

    # 2026-09-13：DONE_FLAG 是**全局单文件**、跨轮次持久（不按 job_id 隔离），
    # 必须比对 job_id 才可采信。否则上一轮残留的 done flag 会让重新注册的监护
    # 每次 tick 都短路返回 completed：既不巡检 paused.json、也不 resume 瞬时暂停，
    # 表现为“任务在、日志正常、但监护静默失效”——比没有监护更危险。
    if DONE_FLAG.is_file():
        done_record = _read_json(DONE_FLAG, {})
        done_job = (
            str(done_record.get("job_id") or "") if isinstance(done_record, dict) else ""
        )
        if done_job and job_id and done_job == job_id:
            summary = {
                "guardian_status": "completed",
                "done": True,
                "job_id": job_id,
                "detail": "done flag matches the current job; already published",
            }
            return summary, EXIT_COMPLETED

    # 2026-09-13：无在跑作业时监护必须保持被动。非交易日 latest.status 同样是
    # "paused"，若无脑走 anomaly 分支会判定 paused-transient → 每 5 分钟调用一次
    # resume() 并写一个 diagnosis 产物（整夜 12 次/小时 × 10 小时 = 120 个文件）。
    if not job_id:
        summary = {
            "guardian_status": "idle",
            "done": False,
            "job_id": None,
            "scheduler_status": str(latest.get("status") or "unknown"),
            "phase": str(latest.get("phase") or "unknown"),
            "detail": "no active job; guardian stays passive",
        }
        try:
            LATEST_GUARDIAN_PATH.write_text(
                json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8"
            )
        except OSError:
            pass
        _append_log(summary)
        return summary, 0

    run_dir = _run_dir(job_id) if job_id else None
    manifest = _read_json(run_dir / "manifest.json", {}) if run_dir else {}
    worker_status = _read_json(run_dir / "status.json", {}) if run_dir else {}
    gates = _read_json(run_dir / "rule_gates.json", {}) if run_dir else {}
    current = _read_json(
        QUANT_ROOT / "results" / "assistant" / "current.json", {}
    )

    manifest_compact = _compact_manifest(manifest)
    gate_summary = _stock_gate_summary(gates)
    current_compact = (
        {
            "run_id": current.get("run_id"),
            "status": current.get("status"),
            "updated_at": current.get("updated_at"),
            "data_end_date": current.get("data_end_date"),
        }
        if isinstance(current, dict)
        else {}
    )

    # ---- completion: manifest published + current.json published for same run ----
    # 2026-09-13 P0-3：认 partial（= 至少一类资产已发布）。只认 success 会让
    # partial 轮永远落进 anomaly 分支，监护语义与发布契约脱节。
    publication_block = manifest_compact.get("publication")
    completed = (
        bool(job_id)
        and manifest_compact.get("status") in PUBLISHED_RUN_STATUSES
        and current_compact.get("status") in PUBLISHED_RUN_STATUSES
        and current_compact.get("run_id") == job_id
    )
    if completed:
        record = {
            "guardian_status": "completed",
            "job_id": job_id,
            "manifest_finished_at": manifest_compact.get("finished_at"),
            "current_updated_at": current_compact.get("updated_at"),
            "data_end_date": current_compact.get("data_end_date"),
            "publication_state": (
                publication_block.get("state")
                if isinstance(publication_block, dict)
                else None
            ),
            "completed_at": datetime.now(TIMEZONE).isoformat(),
        }
        try:
            DONE_FLAG.write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError:
            pass
        record["task_unregistered"] = _try_self_unregister()
        _append_log(record)
        try:
            LATEST_GUARDIAN_PATH.write_text(
                json.dumps(record, ensure_ascii=True, indent=2), encoding="utf-8"
            )
        except OSError:
            pass
        return record, EXIT_COMPLETED

    # ---- anomaly classification (structured signals only) ----
    manifest_reasons = manifest_compact.get("failure_reasons")
    worker_reasons = (
        worker_status.get("failure_reasons") if isinstance(worker_status, dict) else None
    )
    deterministic = _has_deterministic_signal(
        manifest_reasons, worker_reasons, gate_summary.get("failed_checks")
    )
    scheduler_status = str(latest.get("status") or "unknown")
    phase = str(latest.get("phase") or "unknown")
    action = str(latest.get("action") or "")
    last_error = str(latest.get("last_error") or "")
    failure_streak = int(control.get("failure_streak", 0) or 0)
    retry_exhausted = (
        action == "analysis_retry_exhausted"
        or "analysis retry limit" in last_error.lower()
    )

    if paused and retry_exhausted:
        guardian_status = "paused-retry-exhausted"
    elif paused and deterministic:
        guardian_status = "paused-deterministic"
    elif paused:
        guardian_status = "paused-transient"
    elif scheduler_status in {"failed", "partial"} and deterministic:
        guardian_status = "watching-deterministic-failure"
    elif scheduler_status in {"failed", "partial"}:
        guardian_status = "watching-transient-failure"
    elif scheduler_status == "running":
        guardian_status = "healthy-running"
    elif scheduler_status == "paused":
        guardian_status = "paused-transient" if not deterministic else "paused-deterministic"
    else:
        guardian_status = "watching-unknown"

    # ---- diagnosis artifact (拆解 + 方案讨论, bounded) ----
    diagnosis: dict[str, Any] = {
        "guardian_status": guardian_status,
        "job_id": job_id,
        "scheduler_status": scheduler_status,
        "phase": phase,
        "action": action or None,
        "paused": paused,
        "failure_streak": failure_streak,
        "coverage": latest.get("coverage"),
        "completed": latest.get("completed"),
        "pending": latest.get("pending"),
        "failed": latest.get("failed"),
        "manifest": manifest_compact,
        "stock_gate": gate_summary,
        "pause_reason": (
            str(pause_record.get("reason"))[-300:]
            if isinstance(pause_record, dict) and pause_record.get("reason")
            else None
        ),
        "alerts_tail": _tail_text(ALERT_PATH, max_bytes=4096, max_lines=6),
        "scheduler_log_tail": _tail_text(SCHED_LOG_PATH, max_bytes=4096, max_lines=6),
        "checked_at": datetime.now(TIMEZONE).isoformat(),
    }

    if guardian_status in {"paused-deterministic", "watching-deterministic-failure"}:
        diagnosis["root_cause"] = (
            "结构化门槛失败(非基础设施异常): manifest/rule_gates 含 gate/model/drawdown "
            f"类 token, 股票门槛未通过项={gate_summary.get('failed_checks')}, "
            "属确定性发布门槛拦截, 按不变量§6/§7 不得视为基础设施中断。"
        )
        diagnosis["options_discussed"] = [
            {
                "option": "放宽回撤/IC门槛让本次发布",
                "verdict": "否决",
                "reason": "改变交易语义(不变量§6), 需用户显式决策, 监护无权执行。",
            },
            {
                "option": "缩短训练窗口或强制重训",
                "verdict": "否决",
                "reason": "16GB内存约束与模型缓存键(不变量§5), 盲目重训可致 OOM。",
            },
            {
                "option": "保持暂停/等待调度器按断点续跑与下个交易日数据重评",
                "verdict": "采用",
                "reason": "断点有效(不变量§7), 确定性失败由 analysis_retry 计数兜底, 不烧预算。",
            },
        ]
        diagnosis["chosen_fix"] = "不执行修复, 仅记录并继续观察; 达到重试上限后保持暂停等人审。"
        fix_applied: str | None = None
    elif guardian_status == "paused-retry-exhausted":
        diagnosis["root_cause"] = (
            "调度器已达 MAX_ANALYSIS_RETRIES, job 置 paused 防空转, 属预期内的熔断状态。"
        )
        diagnosis["options_discussed"] = [
            {
                "option": "强制 requeue 超过上限",
                "verdict": "否决",
                "reason": "违反防空转设计, 会恢复 10 分钟无限闭环。",
            },
            {
                "option": "保持暂停等人审/等新数据",
                "verdict": "采用",
                "reason": "不断点有效, 新交易日由调度器自然认领。",
            },
        ]
        diagnosis["chosen_fix"] = "不执行修复, 保持暂停。"
        fix_applied = None
    elif guardian_status == "paused-transient":
        diagnosis["root_cause"] = (
            "全局暂停但无确定性门槛信号, 判为瞬时性中断(timeout/队列/日历/锁)。"
        )
        diagnosis["options_discussed"] = [
            {
                "option": "调用调度器自带 resume() 恢复(清 streak, 给3轮验证窗口)",
                "verdict": "采用",
                "reason": "官方恢复语义, 不破坏断点, 不并发第二波。",
            },
            {
                "option": "重建全市场 job",
                "verdict": "否决",
                "reason": "破坏断点与 coverage 基线(不变量§7/§10.1)。",
            },
            {
                "option": "并发启动 worker 补跑",
                "verdict": "否决",
                "reason": "AGENTS 稳定性规则禁止第二波 worker。",
            },
        ]
        fix_applied = None
        try:
            from continuous_research import resume as scheduler_resume

            result = scheduler_resume()
            still_paused = PAUSE_PATH.is_file()
            fix_applied = (
                f"resume executed: {result.get('status')}, "
                f"still_paused={still_paused}"
            )
            diagnosis["chosen_fix"] = fix_applied
        except (OSError, ValueError, ImportError) as exc:
            diagnosis["chosen_fix"] = f"resume failed: {type(exc).__name__}: {exc}"
            diagnosis["traceback_tail"] = traceback.format_exc()[-1500:]
    else:
        diagnosis["root_cause"] = (
            "调度器运行中或瞬时失败轮次, 断点续跑由 10 分钟 tick 认领, 监护不介入。"
        )
        diagnosis["options_discussed"] = [
            {
                "option": "等待调度器下一轮自动续跑",
                "verdict": "采用",
                "reason": "不断点有效, 避免重叠重活(AGENTS 规则)。",
            }
        ]
        diagnosis["chosen_fix"] = "未执行修复, 继续观察。"
        fix_applied = None

    summary = {
        "guardian_status": guardian_status,
        "job_id": job_id,
        "scheduler_status": scheduler_status,
        "phase": phase,
        "paused": paused,
        "failure_streak": failure_streak,
        "coverage": latest.get("coverage"),
        "pending": latest.get("pending"),
        "failed": latest.get("failed"),
        "stock_gate_passed": gate_summary.get("passed"),
        "stock_gate_failed_checks": gate_summary.get("failed_checks"),
        "fix_applied": fix_applied,
        "done": False,
    }

    # Persist: every tick updates the bounded latest file; anomalies also get a
    # timestamped diagnosis artifact. Healthy ticks only append a one-liner.
    try:
        LATEST_GUARDIAN_PATH.write_text(
            json.dumps({**summary, "checked_at": diagnosis["checked_at"]},
                       ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
    except OSError:
        pass
    if guardian_status != "healthy-running":
        try:
            artifact = GUARDIAN_DIR / f"diagnosis-{_utc_stamp()}-{guardian_status}.json"
            artifact.write_text(
                json.dumps(diagnosis, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            summary["diagnosis_artifact"] = artifact.name
        except OSError:
            pass
    _append_log(summary)
    return summary, 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("tick", "status"), default="tick")
    args = parser.parse_args(argv)
    try:
        if args.action == "status":
            value = _read_json(LATEST_GUARDIAN_PATH, {})
            print(json.dumps(value, ensure_ascii=True, sort_keys=True, default=str))
            return 0
        summary, code = tick()
        print(json.dumps(summary, ensure_ascii=True, sort_keys=True, default=str))
        return code
    except Exception as exc:
        print(
            json.dumps(
                {"guardian_status": "error",
                 "error": f"{type(exc).__name__}: {exc}"},
                ensure_ascii=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
