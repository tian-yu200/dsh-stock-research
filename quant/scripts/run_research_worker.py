"""Advance one queued full-universe research job by one bounded worker run."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


QUANT_ROOT = Path(__file__).resolve().parents[1]
if str(QUANT_ROOT) not in sys.path:
    sys.path.insert(0, str(QUANT_ROOT))

from assistant.config import load_config
from assistant.paths import atomic_write_json, run_directory
from assistant.research import _jsonable
from assistant.rolling_scan import run_rolling_pass
from fetch_state import FetchState
from fetch_targets import asset_target_dates


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


def _captured_text(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value or ""


def _compact_status(value: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "run_id",
        "job_id",
        "status",
        "phase",
        "target_date",
        "target_dates",
        "completed",
        "pending",
        "failed",
        "stale",
        "coverage",
        "result_run_id",
        "finished_at",
        "message",
        "error",
    )
    result = {key: value[key] for key in keys if key in value}
    compact_fetches = []
    for fetch in value.get("fetch_results", []):
        compact = {
            key: fetch[key]
            for key in (
                "asset_type",
                "status",
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
    print(json.dumps(_jsonable(_compact_status(value)), ensure_ascii=False, sort_keys=True))


def _write_status(job_id: str, value: dict[str, Any]) -> None:
    atomic_write_json(run_directory(job_id) / "status.json", _jsonable(value))


def _latest_queued(state: FetchState) -> dict[str, Any] | None:
    for job in state.list_jobs(limit=50):
        if job and job.get("kind") == "research" and job.get("status") in {
            "queued",
            "running",
            "partial",
        }:
            return job
    return None


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
    creation_flags = 0
    popen_options: dict[str, Any] = {}
    if os.name == "nt":
        creation_flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        popen_options["start_new_session"] = True
    process = subprocess.Popen(
        command,
        cwd=str(QUANT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
        creationflags=creation_flags,
        env=env,
        **popen_options,
    )
    started = time.monotonic()
    heartbeat_interval = max(
        1.0,
        float(config["fetch"].get("heartbeat_interval_seconds", 5)),
    )
    timed_out = False
    stdout = ""
    stderr = ""
    while True:
        remaining = timeout_seconds - (time.monotonic() - started)
        if remaining <= 0:
            timed_out = True
            break
        try:
            stdout, stderr = process.communicate(
                timeout=min(heartbeat_interval, remaining)
            )
            break
        except subprocess.TimeoutExpired:
            if heartbeat is not None:
                heartbeat()
    if timed_out:
        _terminate_process_tree(
            process,
            float(config["fetch"].get("shutdown_grace_seconds", 45)),
        )
        stdout, stderr = process.communicate()
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


def _run_research_process(
    job_id: str,
    *,
    config: dict[str, Any],
    timeout_seconds: float,
    heartbeat: Callable[[], None] | None = None,
 ) -> dict[str, Any]:
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
        raise TimeoutError("research process exceeded its deadline")
    result = _last_json_object(completed["stdout"])
    if completed["returncode"] not in {0, 1} or result is None:
        detail = completed["stderr"][-1000:] or completed["stdout"][-1000:]
        raise RuntimeError(f"research process failed: {detail}")
    return result


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


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config()
    state = FetchState(Path(config["fetch"]["state_db"]))
    job = state.get_job(args.job_id) if args.job_id else _latest_queued(state)
    if not job:
        print(json.dumps({"status": "no_jobs"}, ensure_ascii=False))
        return 0
    job_id = str(job["job_id"])
    if not state.claim_job(
        job_id,
        stale_after_seconds=int(config["runtime"].get("lock_stale_seconds", 21600)),
    ):
        print(json.dumps(
            {"job_id": job_id, "status": "already_running_or_finished"},
            ensure_ascii=False,
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
    run_dir = run_directory(job_id)
    started = datetime.now(timezone.utc)
    state.update_job(job_id, status="running", started_at=started.isoformat())
    status_value = {
        "run_id": job_id,
        "job_id": job_id,
        "status": "running",
        "worker_pid": os.getpid(),
        "started_at": started.isoformat(),
        "asset_scope": asset_scope,
        "asset_types": list(asset_types),
        "full_universe": full_universe,
        "target_date": target,
        "target_dates": target_dates,
        "result_path": str(run_dir),
        "phase": "cache_check",
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
    atomic_write_json(run_dir / "status.json", _jsonable(status_value))

    def heartbeat() -> None:
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
        atomic_write_json(run_dir / "status.json", _jsonable(status_value))

    try:
        scripts = QUANT_ROOT / "scripts"
        shutdown_grace = max(
            1.0,
            float(config["fetch"].get("shutdown_grace_seconds", 45)),
        )
        for index, asset_type in enumerate(asset_types):
            asset_progress = initial_progress.get("by_asset_type", {}).get(
                asset_type, {}
            )
            if int(asset_progress.get("pending", 0)) == 0:
                fetch_results.append(
                    {
                        "asset_type": asset_type,
                        "status": "cache_hit",
                        "requested": len(symbols.get(asset_type, [])),
                    }
                )
                status_value["fetch_results"] = fetch_results
                heartbeat()
                continue
            status_value["phase"] = "fetch"
            remaining = deadline - time.monotonic()
            if remaining <= shutdown_grace:
                break
            remaining_types = max(1, len(asset_types) - index)
            allocation = max(
                1.0,
                (remaining - shutdown_grace) / remaining_types,
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
            fetch_results.append(result)
            status_value["fetch_results"] = fetch_results
            heartbeat()
            _run_rolling_scan(job_id, config, status_value)
            atomic_write_json(run_dir / "status.json", _jsonable(status_value))
            if result["status"] in {"failed", "timeout"}:
                raise RuntimeError(
                    f"{asset_type} fetch worker {result['status']}"
                )
            if time.monotonic() >= deadline:
                break

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
        _run_rolling_scan(job_id, config, status_value)
        last_completed = _record_progress(
            status_value,
            progress,
            previous_completed=last_completed,
        )
        minimum_coverage = float(
            config["fetch"].get("min_universe_coverage", 0.95)
        )
        fetch_gate_passed = bool(progress["total"]) and (
            progress["coverage"] >= minimum_coverage
        )
        status_value["fetch_gate"] = {
            "passed": fetch_gate_passed,
            "coverage": progress["coverage"],
            "minimum_coverage": minimum_coverage,
            "pending": progress["pending"],
            "failed": progress["failed"],
            "stale": progress["stale"],
        }
        if not fetch_gate_passed:
            state.update_job(job_id, status="queued")
            status_value.update(
                {
                    "status": "queued",
                    "phase": "fetch",
                    "message": "本轮批次完成，等待下一轮继续推进。",
                }
            )
            atomic_write_json(run_dir / "status.json", _jsonable(status_value))
            _print_status(status_value)
            return 0

        stock_refresh_performed = any(
            result.get("asset_type") == "stock"
            and result.get("status") != "cache_hit"
            for result in fetch_results
        )
        if stock_refresh_performed:
            status_value["phase"] = "qlib_build"
            heartbeat()
            build = _run_bounded_command(
                [
                    sys.executable,
                    str(scripts / "build_qlib_data.py"),
                    "--replace",
                ],
                timeout_seconds=max(
                    60.0,
                    float(config["fetch"].get("timeout_seconds", 900)),
                ),
                config=config,
                heartbeat=heartbeat,
            )
            status_value["qlib_build"] = {
                "returncode": build["returncode"],
                "stdout_tail": build["stdout"][-3000:],
                "stderr_tail": build["stderr"][-3000:],
                "timeout_seconds": build["timeout_seconds"],
                "child_pid": build["child_pid"],
            }
            if build["timed_out"]:
                raise TimeoutError("Qlib provider build exceeded its deadline")
            if build["returncode"] != 0:
                raise RuntimeError("Qlib provider build failed")

        status_value["phase"] = "research"
        heartbeat()
        research_result = _run_research_process(
            job_id,
            config=config,
            timeout_seconds=max(
                60.0,
                float(config["fetch"].get("timeout_seconds", 900)),
            ),
            heartbeat=heartbeat,
        )
        status_value["research_result"] = research_result
        status_value["result_run_id"] = research_result.get("run_id")
        status_value["result_path"] = research_result.get("result_path", str(run_dir))
        research_succeeded = research_result.get("status") == "success"
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
        atomic_write_json(run_dir / "status.json", _jsonable(status_value))
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
        atomic_write_json(run_dir / "status.json", _jsonable(status_value))
        _print_status(status_value)
        return 0 if requeue_after_timeout else 1


if __name__ == "__main__":
    raise SystemExit(main())
