"""Advance one queued full-universe research job by one bounded worker run."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable


QUANT_ROOT = Path(__file__).resolve().parents[1]
if str(QUANT_ROOT) not in sys.path:
    sys.path.insert(0, str(QUANT_ROOT))

from assistant.config import load_config
from assistant.paths import atomic_write_json, read_json, run_directory
from assistant.research import _jsonable
from assistant.rolling_scan import run_rolling_pass
from fetch_state import FetchState
from fetch_targets import asset_target_dates

# 2026-09-11 整改(P0-4)：内层研究子进程的 stdout/stderr 原先只落在
# `_run_bounded_command` 的临时文件里（with 块退出即删）。误判发生时无法定死真因，
# 只能靠猜，所以这里给它们一个持久化落点。
RESEARCH_WORKER_LOG_DIR = QUANT_ROOT / "logs" / "continuous_research" / "workers"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", default=None, help="queued research run ID; omitted uses the newest queued job")
    parser.add_argument("--time-budget", type=float, default=None, help="override one worker run budget in seconds")
    return parser


def _summary(stdout: str) -> dict[str, Any] | None:
    marker = "ASSISTANT_FETCH_SUMMARY="
    for line in reversed((stdout or "").splitlines()):
        if line.startswith(marker):
            try:
                value = json.loads(line[len(marker):])
            except json.JSONDecodeError:
                return None
            return value if isinstance(value, dict) else None
    return None


def _qlib_build_summary(stdout: str) -> dict[str, Any] | None:
    marker = "QLIB_BUILD_SUMMARY="
    for line in reversed((stdout or "").splitlines()):
        if line.startswith(marker):
            try:
                value = json.loads(line[len(marker):])
            except json.JSONDecodeError:
                return None
            return value if isinstance(value, dict) else None
    return None


def _captured_text(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value or ""


def _read_tail(handle: Any, limit: int = 512 * 1024) -> bytes:
    """Read the last `limit` bytes of a seekable temp file without loading it all."""
    handle.seek(0, os.SEEK_END)
    size = handle.tell()
    handle.seek(max(0, size - limit))
    return handle.read()


def _compact_status(value: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "run_id",
        "job_id",
        "status",
        "worker_pid",
        "phase",
        "phase_seq",
        "research_phase",
        "research_phase_seq",
        "qlib_build_phase",
        "qlib_build_phase_seq",
        "qlib_build_completed",
        "qlib_build_total",
        "qlib_build_updated_at",
        "target_date",
        "target_dates",
        "asset_scope",
        "asset_types",
        "full_universe",
        "completed",
        "pending",
        "failed",
        "stale",
        "coverage",
        "progress_seq",
        "last_progress_at",
        "last_heartbeat_at",
        "started_at",
        "result_run_id",
        "result_path",
        "finished_at",
        "depth",
        "queued_at",
        "queue_wait_seconds",
        "stage_timings",
        "message",
        "error",
        "cache_hit",
        "cache_miss_reason",
        "provider_end",
        "provider_strategy",
    )
    result = {key: value[key] for key in keys if key in value}
    progress = value.get("progress")
    if isinstance(progress, dict):
        result["progress"] = {
            key: progress[key]
            for key in ("total", "completed", "pending", "failed", "stale", "coverage")
            if key in progress
        }
    fetch_gate = value.get("fetch_gate")
    if isinstance(fetch_gate, dict):
        result["fetch_gate"] = {
            key: fetch_gate[key]
            for key in (
                "passed",
                "child_failed",
                "coverage",
                "minimum_coverage",
                "pending",
                "failed",
                "stale",
                "bypassed_by_depth",
            )
            if key in fetch_gate
        }
    compact_fetches = []
    for fetch in value.get("fetch_results", []):
        compact = {
            key: fetch[key]
            for key in (
                "asset_type",
                "status",
                "requested",
                "returncode",
                "time_budget_seconds",
                "timeout_seconds",
            )
            if key in fetch
        }
        summary = fetch.get("summary")
        if isinstance(summary, dict):
            compact["summary"] = {
                key: summary[key]
                for key in (
                    "status",
                    "requested",
                    "processed",
                    "ok",
                    "failed",
                    "pending",
                    "time_budget_exhausted",
                )
                if key in summary
            }
        compact_fetches.append(compact)
    if compact_fetches:
        result["fetch_results"] = compact_fetches
    qlib_build = value.get("qlib_build")
    if isinstance(qlib_build, dict):
        result["qlib_build"] = {
            key: qlib_build[key]
            for key in (
                "returncode",
                "timeout_seconds",
                "child_pid",
                "cache_hit",
                "skipped",
                "source_fingerprint_digest",
            )
            if key in qlib_build
        }
    rolling = value.get("rolling_scan")
    if isinstance(rolling, dict):
        result["rolling_scan"] = {
            key: rolling[key]
            for key in (
                "status",
                "pool_status",
                "analyzed_this_pass",
                "scored_total",
                "eligible_count",
                "pool_size",
                "events_added",
                "elapsed_seconds",
                "error",
            )
            if key in rolling
        }
    return result


def _print_status(value: dict[str, Any]) -> None:
    # ensure_ascii=True：worker 的 stdout 可能是 GBK 编码的 DEVNULL/重定向
    # 文件，而状态里带着抓取子进程输出中的 U+FFFD 等不可编码字符；
    # ASCII 转义保证打印永不失败（AGENTS.md 编码安全规则）。
    print(json.dumps(_jsonable(_compact_status(value)), ensure_ascii=True, sort_keys=True))


def _write_status(job_id: str, value: dict[str, Any]) -> None:
    atomic_write_json(
        run_directory(job_id) / "status.json",
        _jsonable(_compact_status(value)),
    )


def _set_phase(status_value: dict[str, Any], phase: str) -> None:
    phase = str(phase)
    if status_value.get("phase") == phase:
        return
    now = datetime.now(timezone.utc).isoformat()
    previous = str(status_value.get("phase") or "")
    if previous:
        _close_stage_entry(status_value, previous, now)
    status_value["phase"] = phase
    status_value["phase_started_at"] = now
    status_value["phase_seq"] = int(status_value.get("phase_seq", 0)) + 1


def _close_stage_entry(
    status_value: dict[str, Any],
    stage: str,
    finished_at: str,
) -> None:
    """Accumulate one phase transition into the per-run stage timings."""
    started_at = status_value.get("phase_started_at")
    if not started_at:
        # 该阶段已收口过（如异常分支在终态之后再次收口），防止重复累计。
        return
    timings = status_value.setdefault("stage_timings", {})
    entry = timings.setdefault(stage, {"started_at": str(started_at)})
    try:
        elapsed = (
            datetime.fromisoformat(finished_at) - datetime.fromisoformat(str(started_at))
        ).total_seconds()
    except ValueError:
        elapsed = 0.0
    entry["elapsed_seconds"] = round(
        float(entry.get("elapsed_seconds") or 0.0) + max(0.0, elapsed),
        3,
    )
    entry["finished_at"] = finished_at
    status_value["phase_started_at"] = None


def _close_current_stage(status_value: dict[str, Any]) -> None:
    phase = str(status_value.get("phase") or "")
    if phase:
        _close_stage_entry(
            status_value,
            phase,
            datetime.now(timezone.utc).isoformat(),
        )


def _merge_research_progress(
    run_dir: Path,
    status_value: dict[str, Any],
) -> bool:
    progress = read_json(run_dir / "research-progress.json", {})
    if not isinstance(progress, dict):
        return False
    child_seq = int(progress.get("phase_seq", 0) or 0)
    if child_seq <= int(status_value.get("research_phase_seq", 0) or 0):
        return False
    status_value["research_phase_seq"] = child_seq
    status_value["research_phase"] = str(progress.get("phase") or "unknown")
    status_value["phase_seq"] = int(status_value.get("phase_seq", 0)) + 1
    for key in ("cache_hit", "cache_miss_reason", "provider_end"):
        if key in progress:
            status_value[key] = progress[key]
    return True


def _merge_qlib_build_progress(
    run_dir: Path,
    status_value: dict[str, Any],
) -> bool:
    progress = read_json(run_dir / "qlib-build-progress.json", {})
    if not isinstance(progress, dict):
        return False
    child_seq = int(progress.get("phase_seq", 0) or 0)
    if child_seq <= int(status_value.get("qlib_build_phase_seq", 0) or 0):
        return False
    status_value["qlib_build_phase_seq"] = child_seq
    status_value["qlib_build_phase"] = str(progress.get("phase") or "unknown")
    for source_key, status_key in (
        ("completed", "qlib_build_completed"),
        ("total", "qlib_build_total"),
        ("updated_at", "qlib_build_updated_at"),
    ):
        if source_key in progress:
            status_value[status_key] = progress[source_key]
    for key in ("cache_hit", "source_fingerprint_digest"):
        if key in progress:
            status_value[key] = progress[key]
    status_value["phase_seq"] = int(status_value.get("phase_seq", 0)) + 1
    return True


def _reset_qlib_build_progress(
    run_dir: Path,
    status_value: dict[str, Any],
) -> None:
    """Discard progress and summary from an earlier provider-build attempt."""
    (run_dir / "qlib-build-progress.json").unlink(missing_ok=True)
    for key in (
        "qlib_build_phase",
        "qlib_build_phase_seq",
        "qlib_build_completed",
        "qlib_build_total",
        "qlib_build_updated_at",
    ):
        status_value.pop(key, None)
    status_value.pop("qlib_build", None)
    for key in ("cache_hit", "source_fingerprint_digest"):
        status_value.pop(key, None)


def _reset_research_progress(
    run_dir: Path,
    status_value: dict[str, Any],
) -> None:
    """Discard progress from an earlier attempt before starting a new child."""
    (run_dir / "research-progress.json").unlink(missing_ok=True)
    for key in (
        "research_phase",
        "research_phase_seq",
        "cache_hit",
        "cache_miss_reason",
        "provider_end",
    ):
        status_value.pop(key, None)


def _latest_queued(state: FetchState) -> dict[str, Any] | None:
    for job in state.list_jobs(limit=50):
        if job and job.get("kind") == "research" and job.get("status") in {
            "queued",
            "running",
            "partial",
        }:
            return job
    return None


def _publish_fetch_results(
    status_value: dict[str, Any],
    asset_types: tuple[str, ...],
    results_by_type: dict[str, dict[str, Any]],
) -> None:
    status_value["fetch_results"] = [
        results_by_type[asset_type]
        for asset_type in asset_types
        if asset_type in results_by_type
    ]


def _run_fetch_wave(
    asset_types: tuple[str, ...],
    *,
    scripts: Path,
    config: dict[str, Any],
    job_id: str,
    symbols: dict[str, list[str]],
    full_universe: bool,
    target_dates: dict[str, str],
    deadline: float,
    shutdown_grace: float,
    progress_by_type: dict[str, Any],
    status_value: dict[str, Any],
    heartbeat: Callable[[], None],
    heartbeat_lock: threading.RLock,
) -> list[dict[str, Any]]:
    """推进一个波次内全部资产类型的抓取，结果按 asset_types 原顺序返回。

    股票脚本与基金脚本访问不同数据源，两通道并行不会提高同一端点的请求
    速率；etf 与 otc_fund 共用基金脚本和数据源，保持同通道串行。所有通道
    共享同一个 worker deadline；单通道或配置关闭时退化为纯串行。
    """
    stock_lane = tuple(asset_type for asset_type in asset_types if asset_type == "stock")
    fund_lane = tuple(asset_type for asset_type in asset_types if asset_type != "stock")
    lanes = [lane for lane in (stock_lane, fund_lane) if lane]
    parallel = (
        bool(config["fetch"].get("parallel_asset_fetch", True))
        and len(lanes) > 1
    )
    results_by_type: dict[str, dict[str, Any]] = {}

    def run_lane(lane: tuple[str, ...], count_basis: tuple[str, ...]) -> None:
        for index, asset_type in enumerate(lane):
            asset_progress = progress_by_type.get(asset_type) or {}
            if int(asset_progress.get("pending", 0)) == 0:
                with heartbeat_lock:
                    results_by_type[asset_type] = {
                        "asset_type": asset_type,
                        "status": "cache_hit",
                        "requested": len(symbols.get(asset_type, [])),
                    }
                    _publish_fetch_results(status_value, asset_types, results_by_type)
                    heartbeat()
                continue
            with heartbeat_lock:
                _set_phase(status_value, "fetch")
                remaining = deadline - time.monotonic()
            if remaining <= shutdown_grace:
                break
            allocation = max(
                1.0,
                (remaining - shutdown_grace) / max(1, len(count_basis) - index),
            )
            script = (
                scripts / "fetch_stock_data.py"
                if asset_type == "stock"
                else scripts / "fetch_fund_data.py"
            )
            result = _run_fetch_script(
                script,
                asset_type=asset_type,
                config=config,
                time_budget=allocation,
                job_id=job_id,
                target_date=target_dates[asset_type],
                deadline=deadline,
                symbols=None if full_universe else symbols.get(asset_type, []),
                heartbeat=heartbeat,
            )
            with heartbeat_lock:
                results_by_type[asset_type] = result
                _publish_fetch_results(status_value, asset_types, results_by_type)
                heartbeat()
                _run_rolling_scan(job_id, config, status_value)
                _write_status(job_id, status_value)
            if result["status"] in {"failed", "timeout"}:
                # 超时是有检查点的正常波次收尾；真正的 child failed
                # 仍允许完成其余类别，但本轮绝不进入研究发布阶段。
                status_value["last_error"] = f"{asset_type} fetch worker {result['status']}"
                print(
                    f"[wave] {asset_type} fetch worker {result['status']}; "
                    "continuing with remaining asset types",
                    flush=True,
                )
            if time.monotonic() >= deadline:
                break

    if parallel:
        threads = [
            threading.Thread(
                target=run_lane,
                args=(lane, lane),
                name=f"fetch-{lane[0]}",
            )
            for lane in lanes
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        run_lane(asset_types, asset_types)
    return [
        results_by_type[asset_type]
        for asset_type in asset_types
        if asset_type in results_by_type
    ]


def _terminate_process_tree(process: subprocess.Popen[str], grace_seconds: float) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=max(0.1, grace_seconds))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    try:
        process.wait(timeout=max(0.1, grace_seconds))
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _run_bounded_command(
    command: list[str],
    *,
    timeout_seconds: float,
    config: dict[str, Any],
    heartbeat: Callable[[], None] | None = None,
    env: dict[str, str] | None = None,
 ) -> dict[str, Any]:
    # 子进程统一 UTF-8（AGENTS.md 编码安全规则）：抓取/构建/研究子进程的
    # 输出会被本进程按 UTF-8 解码，若子进程沿用 GBK 会把中文打成 U+FFFD，
    # 这些字符再进入状态文件与 JSON 打印链路就会引爆编码错误。
    child_env = {
        **os.environ,
        **(env or {}),
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
    }
    creation_flags = 0
    popen_options: dict[str, Any] = {}
    if os.name == "nt":
        creation_flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        popen_options["start_new_session"] = True
    # 输出重定向到临时文件而非 PIPE：全市场规模的子进程输出（逐标的日志、
    # tqdm 进度条）远超 64KB 管道缓冲，而带超时的 communicate 重试在
    # Windows 上无法可靠排空管道，父子进程互相等待形成死锁。
    # 临时文件随 with 块退出自动删除，读取末尾片段即可覆盖调用方的用途。
    with tempfile.TemporaryFile() as out_file, tempfile.TemporaryFile() as err_file:
        process = subprocess.Popen(
            command,
            cwd=str(QUANT_ROOT),
            stdout=out_file,
            stderr=err_file,
            text=False,
            creationflags=creation_flags,
            env=child_env,
            **popen_options,
        )
        started = time.monotonic()
        heartbeat_interval = max(
            1.0,
            float(config["fetch"].get("heartbeat_interval_seconds", 5)),
        )
        timed_out = False
        while True:
            remaining = timeout_seconds - (time.monotonic() - started)
            if remaining <= 0:
                timed_out = True
                break
            try:
                process.wait(timeout=min(heartbeat_interval, remaining))
                break
            except subprocess.TimeoutExpired:
                if heartbeat is not None:
                    heartbeat()
        if timed_out:
            _terminate_process_tree(
                process,
                float(config["fetch"].get("shutdown_grace_seconds", 45)),
            )
            process.wait()
        stdout = _read_tail(out_file)
        stderr = _read_tail(err_file)
    return {
        "timed_out": timed_out,
        "returncode": None if timed_out else process.returncode,
        "stdout": _captured_text(stdout),
        "stderr": _captured_text(stderr),
        "child_pid": process.pid,
        "timeout_seconds": timeout_seconds,
    }


def _last_json_object(stdout: str) -> dict[str, Any] | None:
    for line in reversed((stdout or "").splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _persist_inner_research_output(job_id: str, completed: dict[str, Any]) -> str | None:
    """把内层研究子进程的输出落盘，返回 stdout 路径（失败返回 None）。

    P0-4 定位时缺的正是这份证据：`_run_bounded_command` 把子进程输出写进临时文件，
    退出即删，于是"内层为何返回 {0,1} 之外的退出码"无法判定，只能猜。
    """
    try:
        RESEARCH_WORKER_LOG_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESEARCH_WORKER_LOG_DIR / f"{job_id}.research.stdout.log"
        err_path = RESEARCH_WORKER_LOG_DIR / f"{job_id}.research.stderr.log"
        header = (
            "# inner research child"
            f" pid={completed.get('child_pid')}"
            f" returncode={completed.get('returncode')!r}"
            f" timed_out={completed.get('timed_out')}"
            f" timeout_seconds={completed.get('timeout_seconds')}\n"
        )
        out_path.write_text(
            header + str(completed.get("stdout") or ""), encoding="utf-8"
        )
        err_path.write_text(
            header + str(completed.get("stderr") or ""), encoding="utf-8"
        )
        return str(out_path)
    except OSError:
        return None


def _published_research_result(
    job_id: str,
    *,
    invoked_at: datetime,
) -> dict[str, Any] | None:
    """以产物为准判断内层研究是否其实已经成功；否则返回 None。

    背景（2026-09-11 P0-4）：`_run_research_process` 原先只看内层子进程的原始退出码
    和 stdout 是否能解析出 JSON。实测 job `20260911T065625Z-d6d4eede` 已成功发布
    （`manifest.json` status=success、`current.json` 已更新、`publication.state=full`），
    内层却返回了 {0,1} 之外的退出码，于是整轮被记成失败，`failure_streak` 累到 3，
    `_auto_pause_on_streak` 在 22:05:49 把调度器整体停摆。

    《研究管线不变量》§7 明确：判断运行成败以 manifest.json / current.json 为准，
    不要以退出码/台账文本为准。`invoked_at` 用于排除"同一个 run 目录里上一次 pass
    留下的旧成功产物"被误当成本轮成功。
    """
    try:
        run_dir = run_directory(job_id)
    except Exception:
        return None
    manifest = read_json(run_dir / "manifest.json", {})
    if not isinstance(manifest, dict):
        return None
    if str(manifest.get("run_id") or "") != str(job_id):
        return None
    if str(manifest.get("status") or "") not in {"success", "partial"}:
        return None
    finished_raw = manifest.get("finished_at")
    try:
        finished_at = datetime.fromisoformat(str(finished_raw))
    except (TypeError, ValueError):
        return None
    if finished_at.tzinfo is None:
        finished_at = finished_at.replace(tzinfo=timezone.utc)
    # 5 秒容差：manifest 的 finished_at 由子进程写入，与本进程取时存在微小偏差。
    if finished_at < invoked_at - timedelta(seconds=5):
        return None
    return {
        "run_id": str(job_id),
        # 2026-09-11 P0-3 之后 manifest 可能是 "partial"（分类别发布）；
        # 对 worker 而言与 "success" 同为"已发布"，不构成失败。
        "status": "success",
        "publication_state": manifest.get("status"),
        "failure_reasons": manifest.get("failure_reasons") or [],
        "data_end_date": manifest.get("data_end_date"),
        "result_path": str(run_dir),
        # 供排查：本轮是从产物"救回"的成功，内层退出码与此字段一并留在状态里。
        "recovered_from_artifacts": True,
    }


def _run_research_process(
    job_id: str,
    *,
    config: dict[str, Any],
    timeout_seconds: float,
    heartbeat: Callable[[], None] | None = None,
 ) -> dict[str, Any]:
    # P0-4：记录发起时刻，用于把"本轮产物"与同一 run 目录里上一次 pass 的旧产物区分开。
    invoked_at = datetime.now(timezone.utc)
    completed = _run_bounded_command(
        [
            sys.executable,
            str(QUANT_ROOT / "scripts" / "run_assistant_research.py"),
            "--queued-run-id",
            job_id,
        ],
        timeout_seconds=timeout_seconds,
        config=config,
        heartbeat=heartbeat,
    )
    if completed["timed_out"]:
        raise TimeoutError(
            "research process exceeded its deadline; "
            "if the model cache was cold (provider rebuilt), the one-time "
            "full training needs a larger budget "
            "(fetch.targeted_time_budget_seconds for targeted jobs)"
        )
    result = _last_json_object(completed["stdout"])
    if completed["returncode"] in {0, 1} and result is not None:
        return result
    # 2026-09-11 整改(P0-4)：退出码/输出异常**不等于**研究失败。
    # 判失败前先核对产物（不变量 §7）：内层已成功发布就不该让整轮背上失败。
    inner_log = _persist_inner_research_output(job_id, completed)
    published = _published_research_result(job_id, invoked_at=invoked_at)
    if published is not None:
        published["inner_returncode"] = completed["returncode"]
        published["inner_log"] = inner_log
        return published
    detail = completed["stderr"][-1000:] or completed["stdout"][-1000:]
    raise RuntimeError(
        "research process failed: "
        f"inner_returncode={completed['returncode']!r} "
        f"inner_log={inner_log or 'unavailable'}; {detail}"
    )


def _remaining_stage_timeout(
    *,
    deadline: float,
    configured_timeout: float,
    shutdown_grace: float,
    stage: str,
 ) -> float:
    remaining = deadline - time.monotonic()
    available = remaining - max(0.0, shutdown_grace)
    if available <= 0:
        raise TimeoutError(f"worker time budget exhausted before {stage}")
    return max(0.1, min(float(configured_timeout), available))


def _build_fetch_command(
    script: Path,
    *,
    asset_type: str,
    config: dict[str, Any],
    time_budget: float,
    job_id: str,
    target_date: str,
    symbols: list[str] | None,
    deadline: float,
 ) -> tuple[list[str], int, int]:
    if symbols is not None and not symbols:
        raise ValueError(f"no symbols configured for targeted {asset_type} research job")
    batch_sizes = config["fetch"].get("batch_size", {})
    batch_size = int(batch_sizes.get(asset_type, 100))
    remaining = max(1.0, deadline - time.monotonic())
    shutdown_grace = max(
        1,
        int(config["fetch"].get("shutdown_grace_seconds", 45)),
    )
    child_budget = max(
        1,
        min(int(time_budget), max(1, int(remaining) - shutdown_grace)),
    )
    request_timeout = min(
        float(config["fetch"].get("request_timeout_seconds", 20)),
        max(1.0, child_budget / 2.0),
    )
    timeout_seconds = max(
        child_budget,
        int(min(remaining, child_budget + shutdown_grace)),
    )
    command = [
        sys.executable,
        str(script),
        "--batch-size",
        str(batch_size),
        "--time-budget",
        str(child_budget),
        "--workers",
        str(int(config["fetch"].get("max_workers", 3))),
        "--retries",
        str(int(config["fetch"].get("retries", 2))),
        "--rate-limit",
        str(float(config["fetch"].get("requests_per_second", 1.0))),
        "--request-timeout",
        str(request_timeout),
        "--backoff-base",
        str(float(config["fetch"].get("backoff_base_seconds", 2))),
        "--backoff-max",
        str(float(config["fetch"].get("backoff_max_seconds", 120))),
        "--lease-seconds",
        str(int(config["fetch"].get("lease_seconds", 1800))),
        "--state-db",
        str(config["fetch"]["state_db"]),
        "--job-id",
        job_id,
        "--target-date",
        target_date,
    ]
    if asset_type in {"etf", "otc_fund"}:
        command.extend(["--asset-type", asset_type])
    if symbols is not None:
        command.extend(symbols)
    return command, child_budget, timeout_seconds


def _record_progress(
    status_value: dict[str, Any],
    progress: dict[str, Any],
    *,
    previous_completed: int,
 ) -> int:
    now = datetime.now(timezone.utc).isoformat()
    completed = int(progress.get("completed", 0))
    status_value.update(
        {
            "completed": completed,
            "pending": int(progress.get("pending", 0)),
            "failed": int(progress.get("failed", 0)),
            "stale": int(progress.get("stale", 0)),
            "coverage": float(progress.get("coverage", 0.0)),
            "last_heartbeat_at": now,
        }
    )
    if completed > previous_completed:
        status_value["progress_seq"] = int(status_value.get("progress_seq", 0)) + 1
        status_value["last_progress_at"] = now
        status_value["last_completed_at"] = now
    return completed


def _run_rolling_scan(
    job_id: str,
    config: dict[str, Any],
    status_value: dict[str, Any],
) -> None:
    """Advance the rolling stock pool by one incremental pass (never fatal)."""
    settings = config.get("rolling_scan") or {}
    if not settings.get("enabled", True):
        return
    try:
        status_value["rolling_scan"] = run_rolling_pass(job_id=job_id, config=config)
    except Exception as exc:
        status_value["rolling_scan"] = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _run_fetch_script(
    script: Path,
    *,
    asset_type: str,
    config: dict[str, Any],
    time_budget: float,
    job_id: str,
    target_date: str,
    deadline: float,
    symbols: list[str] | None,
    heartbeat: Callable[[], None] | None = None,
) -> dict[str, Any]:
    command, child_budget, timeout_seconds = _build_fetch_command(
        script,
        asset_type=asset_type,
        config=config,
        time_budget=time_budget,
        job_id=job_id,
        target_date=target_date,
        symbols=symbols,
        deadline=deadline,
    )
    completed = _run_bounded_command(
        command,
        timeout_seconds=timeout_seconds,
        config=config,
        heartbeat=heartbeat,
    )
    timed_out = bool(completed["timed_out"])
    stdout = str(completed["stdout"])
    stderr = str(completed["stderr"])
    summary = _summary(stdout)
    return {
        "asset_type": asset_type,
        "status": "timeout" if timed_out else (summary.get("status") if summary else "failed"),
        "returncode": completed["returncode"],
        "summary": summary,
        "stdout_tail": _captured_text(stdout)[-3000:],
        "stderr_tail": _captured_text(stderr)[-3000:],
        "timeout_seconds": timeout_seconds,
        "time_budget_seconds": child_budget,
        "child_pid": completed["child_pid"],
    }


def _hold_awake() -> None:
    """本进程存活期间向系统声明"勿待机"。

    Modern Standby（S0）机器在屏幕关闭后会进入待机冻结后台进程；
    Windows 只认线程级电源请求，不认电源计划里的"从不睡眠"。
    请求随线程退出自动失效，无需清理。
    """
    if os.name != "nt":
        return
    try:
        import ctypes

        es_continuous = 0x80000000
        es_system_required = 0x00000040
        ctypes.windll.kernel32.SetThreadExecutionState(
            ctypes.c_uint(es_continuous | es_system_required)
        )
    except Exception:
        return


def main(argv: list[str] | None = None) -> int:
    # Windows 上 stdout 可能是 GBK 管道；遇到不可编码字符降级而不是崩溃。
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(errors="replace")
            except (ValueError, OSError):
                pass
    _hold_awake()
    args = _parser().parse_args(argv)
    config = load_config()
    state = FetchState(Path(config["fetch"]["state_db"]))
    job = state.get_job(args.job_id) if args.job_id else _latest_queued(state)
    if not job:
        print(json.dumps({"status": "no_jobs"}, ensure_ascii=True))
        return 0
    job_id = str(job["job_id"])
    if not state.claim_job(
        job_id,
        stale_after_seconds=int(config["runtime"].get("lock_stale_seconds", 21600)),
    ):
        print(json.dumps(
            {"job_id": job_id, "status": "already_running_or_finished"},
            ensure_ascii=True,
            sort_keys=True,
        ))
        return 0
    metadata = job.get("metadata", {})
    asset_types = tuple(metadata.get("asset_types", []))
    asset_scope = str(metadata.get("asset_scope", "all"))
    mode = str(metadata.get("mode", "daily"))
    target = str(metadata.get("target_date") or datetime.now(timezone.utc).date().isoformat())
    configured_target_dates = metadata.get("target_dates")
    if isinstance(configured_target_dates, dict) and all(
        asset_type in configured_target_dates for asset_type in asset_types
    ):
        target_dates = {
            asset_type: str(configured_target_dates[asset_type])
            for asset_type in asset_types
        }
    else:
        target_dates = asset_target_dates(
            asset_types,
            target,
            lag_trading_days=config["fetch"].get("target_lag_trading_days", {}),
        )
        metadata = {**metadata, "target_dates": target_dates}
        state.update_job(job_id, metadata=metadata)
    symbols = {
        asset_type: [str(symbol) for symbol in metadata.get("symbols", {}).get(asset_type, [])]
        for asset_type in asset_types
    }
    full_universe = bool(
        metadata.get("full_universe", not any(symbols.values()))
    )
    depth = str(metadata.get("depth") or "standard")
    run_dir = run_directory(job_id)
    started = datetime.now(timezone.utc)
    queued_at = str(job.get("created_at") or "") or None
    queue_wait_seconds = None
    if queued_at:
        try:
            queue_wait_seconds = round(
                max(0.0, (started - datetime.fromisoformat(queued_at)).total_seconds()),
                3,
            )
        except ValueError:
            queue_wait_seconds = None
    state.update_job(job_id, status="running", started_at=started.isoformat())
    status_value = {
        "run_id": job_id,
        "job_id": job_id,
        "status": "running",
        "worker_pid": os.getpid(),
        "started_at": started.isoformat(),
        "queued_at": queued_at,
        "queue_wait_seconds": queue_wait_seconds,
        "asset_scope": asset_scope,
        "asset_types": list(asset_types),
        "full_universe": full_universe,
        "depth": depth,
        "target_date": target,
        "target_dates": target_dates,
        "result_path": str(run_dir),
        "phase": "cache_check",
        "phase_started_at": started.isoformat(),
        "stage_timings": {},
        "phase_seq": 1,
        "fetch_results": [],
        "progress_seq": 0,
        "last_progress_at": started.isoformat(),
        "last_heartbeat_at": started.isoformat(),
    }

    total_budget = (
        float(args.time_budget)
        if args.time_budget is not None
        else float(config["fetch"].get("batch_time_budget_seconds", 600))
    )
    deadline = time.monotonic() + max(1.0, total_budget)
    fetch_results: list[dict[str, Any]] = []
    initial_progress = state.progress(
        asset_types,
        symbols=symbols,
        target_date=target_dates,
        max_age_days=int(config["fetch"]["max_data_age_days"]),
    )
    last_completed = _record_progress(
        status_value,
        initial_progress,
        previous_completed=int(initial_progress.get("completed", 0)),
    )
    _write_status(job_id, status_value)
    heartbeat_lock = threading.RLock()

    def heartbeat() -> None:
        # 抓取波次可能以两条线程并行调用；RLock 允许波次函数在持有锁的
        # 情况下再嵌套调用 heartbeat。
        with heartbeat_lock:
            nonlocal last_completed
            progress = state.progress(
                asset_types,
                symbols=symbols,
                target_date=target_dates,
                max_age_days=int(config["fetch"]["max_data_age_days"]),
            )
            last_completed = _record_progress(
                status_value,
                progress,
                previous_completed=last_completed,
            )
            state.update_job(
                job_id,
                total=progress["total"],
                completed=progress["completed"],
                pending=progress["pending"],
                failed=progress["failed"],
                stale=progress["stale"],
                coverage=progress["coverage"],
            )
            _merge_research_progress(run_dir, status_value)
            _merge_qlib_build_progress(run_dir, status_value)
            _write_status(job_id, status_value)

    try:
        scripts = QUANT_ROOT / "scripts"
        shutdown_grace = max(
            1.0,
            float(config["fetch"].get("shutdown_grace_seconds", 45)),
        )
        # 快速深度只读缓存：跳过全部抓取子进程，也不因 pending 回队。
        fetch_scope_types = () if depth == "quick" else tuple(asset_types)
        if fetch_scope_types:
            fetch_results.extend(
                _run_fetch_wave(
                    fetch_scope_types,
                    scripts=scripts,
                    config=config,
                    job_id=job_id,
                    symbols=symbols,
                    full_universe=full_universe,
                    target_dates=target_dates,
                    deadline=deadline,
                    shutdown_grace=shutdown_grace,
                    progress_by_type=initial_progress.get("by_asset_type", {}),
                    status_value=status_value,
                    heartbeat=heartbeat,
                    heartbeat_lock=heartbeat_lock,
                )
            )
            status_value["fetch_results"] = fetch_results

        if depth == "quick":
            fetch_results.extend(
                {
                    "asset_type": asset_type,
                    "status": "skipped_by_depth",
                    "requested": len(symbols.get(asset_type, [])),
                }
                for asset_type in asset_types
            )
            status_value["fetch_results"] = fetch_results
        progress = state.progress(
            asset_types,
            symbols=symbols,
            target_date=target_dates,
            max_age_days=int(config["fetch"]["max_data_age_days"]),
        )
        state.update_job(
            job_id,
            total=progress["total"],
            completed=progress["completed"],
            pending=progress["pending"],
            failed=progress["failed"],
            stale=progress["stale"],
            coverage=progress["coverage"],
        )
        status_value.update(
            {
                "progress": progress,
            }
        )
        if depth != "quick":
            _run_rolling_scan(job_id, config, status_value)
        last_completed = _record_progress(
            status_value,
            progress,
            previous_completed=last_completed,
        )
        minimum_coverage = float(
            config["fetch"].get("min_universe_coverage", 0.95)
        )
        child_failed = any(
            result.get("status") == "failed" for result in fetch_results
        )
        fetch_gate_passed = bool(progress["total"]) and (
            not child_failed
            and
            progress["coverage"] >= minimum_coverage
        )
        status_value["fetch_gate"] = {
            "passed": fetch_gate_passed,
            "child_failed": child_failed,
            "coverage": progress["coverage"],
            "minimum_coverage": minimum_coverage,
            "pending": progress["pending"],
            "failed": progress["failed"],
            "stale": progress["stale"],
        }
        if depth == "quick":
            fetch_gate_passed = True
            status_value["fetch_gate"]["passed"] = True
            status_value["fetch_gate"]["bypassed_by_depth"] = "quick"
        if child_failed:
            failed_types = [
                str(result.get("asset_type") or "unknown")
                for result in fetch_results
                if result.get("status") == "failed"
            ]
            message = (
                "抓取子进程失败，禁止进入研究发布阶段："
                + ", ".join(failed_types)
            )
            finished_at = datetime.now(timezone.utc).isoformat()
            _close_current_stage(status_value)
            status_value.update(
                {
                    "status": "failed",
                    "phase": "fetch",
                    "message": message,
                    "error": message,
                    "finished_at": finished_at,
                }
            )
            state.update_job(
                job_id,
                status="failed",
                finished_at=finished_at,
                last_error=message,
            )
            _write_status(job_id, status_value)
            _print_status(status_value)
            return 1
        if not fetch_gate_passed:
            state.update_job(job_id, status="queued")
            fetch_type_interrupted = any(
                result.get("status") in {"failed", "timeout"}
                for result in fetch_results
            )
            _close_current_stage(status_value)
            status_value.update(
                {
                    "status": "queued",
                    "phase": "fetch",
                    "message": (
                        "部分类别抓取超时或失败，作业回到队列等待下一轮继续。"
                        if fetch_type_interrupted
                        else "本轮批次完成，等待下一轮继续推进。"
                    ),
                }
            )
            _write_status(job_id, status_value)
            _print_status(status_value)
            return 0

        stock_refresh_performed = any(
            result.get("asset_type") == "stock"
            and result.get("status") not in {"cache_hit", "skipped_by_depth"}
            for result in fetch_results
        )
        if stock_refresh_performed and full_universe:
            status_value["provider_strategy"] = "rebuild_if_source_changed"
            _reset_qlib_build_progress(run_dir, status_value)
            _set_phase(status_value, "qlib_build")
            heartbeat()
            build_timeout = _remaining_stage_timeout(
                deadline=deadline,
                configured_timeout=max(
                    60.0,
                    float(config["fetch"].get("timeout_seconds", 900)),
                ),
                shutdown_grace=shutdown_grace,
                stage="Qlib provider build",
            )
            build = _run_bounded_command(
                [
                    sys.executable,
                    str(scripts / "build_qlib_data.py"),
                    "--replace",
                    "--progress-file",
                    str(run_dir / "qlib-build-progress.json"),
                ],
                timeout_seconds=build_timeout,
                config=config,
                heartbeat=heartbeat,
            )
            _merge_qlib_build_progress(run_dir, status_value)
            build_summary = _qlib_build_summary(build["stdout"]) or {}
            status_value["qlib_build"] = {
                "returncode": build["returncode"],
                "stdout_tail": build["stdout"][-3000:],
                "stderr_tail": build["stderr"][-3000:],
                "timeout_seconds": build["timeout_seconds"],
                "child_pid": build["child_pid"],
                "cache_hit": bool(build_summary.get("cache_hit", False)),
                "source_fingerprint_digest": build_summary.get(
                    "source_fingerprint_digest"
                ),
            }
            if build["timed_out"]:
                raise TimeoutError("Qlib provider build exceeded its deadline")
            if build["returncode"] != 0:
                raise RuntimeError("Qlib provider build failed")
        elif "stock" in asset_types and not full_universe:
            status_value["provider_strategy"] = (
                "reuse_existing_provider_for_targeted_run"
            )

        _reset_research_progress(run_dir, status_value)
        _set_phase(status_value, "research")
        heartbeat()
        # 模型缓存冷启动（provider 重建后的首次完整训练）没有中间检查点，
        # 不能被波次预算截断：抓取已经完成，研究阶段允许延长到
        # targeted_time_budget_seconds 的底线，训练成功后缓存随即可用。
        research_deadline = max(
            deadline,
            time.monotonic()
            + max(
                60.0,
                float(config["fetch"].get("targeted_time_budget_seconds", 3600)),
            ),
        )
        research_timeout = _remaining_stage_timeout(
            deadline=research_deadline,
            configured_timeout=max(
                60.0,
                float(config["fetch"].get("timeout_seconds", 900)),
            ),
            shutdown_grace=shutdown_grace,
            stage="research",
        )
        research_result = _run_research_process(
            job_id,
            config=config,
            timeout_seconds=research_timeout,
            heartbeat=heartbeat,
        )
        _merge_research_progress(run_dir, status_value)
        result_run_id = research_result.get("run_id")
        if result_run_id != job_id:
            raise ValueError(
                "research result run_id does not match queued job_id: "
                f"expected {job_id}, got {result_run_id!r}"
            )
        status_value["result_run_id"] = result_run_id
        status_value["result_path"] = research_result.get("result_path", str(run_dir))
        # 2026-09-11 P0-3：分类别发布后 research 的 status 可能是 "partial"
        # （部分类别未过门槛但已发布就绪类别），对 worker 而言同样算成功。
        research_succeeded = research_result.get("status") in {"success", "partial"}
        has_incomplete_data = any(
            int(progress.get(key, 0)) > 0 for key in ("pending", "failed", "stale")
        )
        status_value["status"] = (
            "partial"
            if research_succeeded and has_incomplete_data
            else "complete"
            if research_succeeded
            else "failed"
        )
        _close_current_stage(status_value)
        status_value["finished_at"] = datetime.now(timezone.utc).isoformat()
        state.update_job(
            job_id,
            status=status_value["status"],
            finished_at=status_value["finished_at"],
            last_error=(
                None
                if status_value["status"] == "complete"
                else str(research_result.get("error") or research_result.get("failure_reasons"))
            ),
        )
        _write_status(job_id, status_value)
        _print_status(status_value)
        return 0 if status_value["status"] in {"complete", "partial"} else 1
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        state.reclaim_orphaned_leases()
        progress = state.progress(
            asset_types,
            symbols=symbols,
            target_date=target_dates,
            max_age_days=int(config["fetch"]["max_data_age_days"]),
        )
        progress_fields = {
            "total": progress["total"],
            "completed": progress["completed"],
            "pending": progress["pending"],
            "failed": progress["failed"],
            "stale": progress["stale"],
            "coverage": progress["coverage"],
        }
        # 抓取到达时间预算是每一轮的正常收尾方式（运行说明：下一轮从上次
        # 成功日期之后继续）。逐标的进度已即时提交，因此只要仍有剩余工作，
        # 预算耗尽就把作业送回队列，而不是把整次研究判死。
        fetch_timed_out = any(
            result.get("status") == "timeout" for result in fetch_results
        )
        work_remains = (
            int(progress["pending"]) + int(progress["failed"]) > 0
        )
        requeue_after_timeout = fetch_timed_out and work_remains
        final_status = (
            "queued"
            if requeue_after_timeout
            else "timeout"
            if isinstance(exc, TimeoutError)
            else "failed"
        )
        _close_current_stage(status_value)
        status_value.update(
            {
                "status": final_status,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "error": error,
                "fetch_results": fetch_results,
                "progress": progress,
                **progress_fields,
            }
        )
        if requeue_after_timeout:
            status_value["message"] = (
                "抓取进程到达本轮时间预算，仍有剩余工作，作业回到队列等待下一轮继续。"
            )
        state.update_job(
            job_id,
            status=final_status,
            finished_at=status_value["finished_at"],
            last_error=error,
            **progress_fields,
        )
        _write_status(job_id, status_value)
        _print_status(status_value)
        return 0 if requeue_after_timeout else 1


if __name__ == "__main__":
    raise SystemExit(main())
