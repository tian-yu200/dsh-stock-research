"""Resumable, single-worker full-market research scheduler.

The scheduler is deliberately a local CLI.  It coordinates the existing
research queue and fetch state database, but it does not replace the fetchers
or publish result snapshots itself.
"""
from __future__ import annotations

import argparse
import json
import logging
from logging.handlers import RotatingFileHandler
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterator
from uuid import uuid4
from zoneinfo import ZoneInfo


QUANT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = QUANT_ROOT.parent
if str(QUANT_ROOT) not in sys.path:
    sys.path.insert(0, str(QUANT_ROOT))

from assistant.config import load_config
from assistant.paths import atomic_write_json, read_json, run_directory
from assistant.research import run_research
from fetch_state import FetchState, parse_iso, iso_now
from fetch_targets import asset_target_dates
from trading_calendar import resolve_trading_day


ASSET_TYPES = ("stock", "etf", "otc_fund")
TARGET_SCOPE = "all"
RESUMABLE_JOB_STATUSES = frozenset({"queued", "partial", "running"})
TIMEZONE = ZoneInfo("Asia/Shanghai")
CONTINUOUS_DIR = QUANT_ROOT / "logs" / "continuous_research"
LATEST_STATUS_PATH = CONTINUOUS_DIR / "latest_status.json"
LOCK_PATH = CONTINUOUS_DIR / "continuous_research.lock"
PAUSE_PATH = CONTINUOUS_DIR / "paused.json"
CONTROL_PATH = CONTINUOUS_DIR / "control.json"
LOG_PATH = CONTINUOUS_DIR / "continuous_research.log"
ALERT_PATH = CONTINUOUS_DIR / "alerts.log"
WORKER_LOG_DIR = CONTINUOUS_DIR / "workers"

# 每个 wave 的抓取预算。外部触发间隔约 10 分钟；预算必须接近触发间隔，
# 否则大部分时间空转等待下一次触发（240s 预算 + 10min 间隔时吞吐仅约 2 标的/分钟）。
# 壁钟超时必须大于预算 + 研究阶段的耗时，否则 worker 会在抓取中途被杀。
WORKER_TIME_BUDGET_SECONDS = 3300
WORKER_WALL_TIMEOUT_SECONDS = 4200
STALE_AFTER_SECONDS = 21600
LONG_STALE_EXTRA_DAYS = 20
LONG_STALE_RETRY_INTERVAL_SECONDS = 6 * 60 * 60
LONG_STALE_MIN_ATTEMPTS = 3
MAX_ANALYSIS_RETRIES = 3
LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_BACKUP_COUNT = 5
ALERT_MAX_BYTES = 2 * 1024 * 1024
ALERT_BACKUP_COUNT = 3


class ClosingRotatingFileHandler(RotatingFileHandler):
    """Release the Windows file handle after every log record."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            super().emit(record)
        finally:
            self.close()


def _now() -> datetime:
    return datetime.now(TIMEZONE)


def _utc_now() -> datetime:
    return datetime.now().astimezone(ZoneInfo("UTC"))


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return read_json(path, default)
    except (OSError, json.JSONDecodeError, ValueError):
        return default


def _write_json(path: Path, value: Any) -> None:
    CONTINUOUS_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, value)


def _rotating_logger(
    name: str,
    path: Path,
    *,
    max_bytes: int,
    backup_count: int,
    formatter: str,
) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    resolved = str(path.resolve())
    for handler in list(logger.handlers):
        if not isinstance(handler, RotatingFileHandler) or handler.baseFilename != resolved:
            logger.removeHandler(handler)
            handler.close()
    if not logger.handlers:
        handler = ClosingRotatingFileHandler(
            path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
            delay=True,
        )
        handler.setFormatter(logging.Formatter(formatter))
        logger.addHandler(handler)
    return logger


def _logger() -> logging.Logger:
    return _rotating_logger(
        "dsh.continuous_research",
        LOG_PATH,
        max_bytes=LOG_MAX_BYTES,
        backup_count=LOG_BACKUP_COUNT,
        formatter="%(asctime)s %(levelname)s %(message)s",
    )


def _high_alert(logger: logging.Logger, message: str) -> None:
    alert_logger = _rotating_logger(
        "dsh.continuous_research.alert",
        ALERT_PATH,
        max_bytes=ALERT_MAX_BYTES,
        backup_count=ALERT_BACKUP_COUNT,
        formatter="%(asctime)s HIGH %(message)s",
    )
    alert_logger.error(message)
    logger.error("HIGH PRIORITY: %s", message)


def _process_exists(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, ProcessLookupError, PermissionError):
        return False
    return True


class SchedulerLock:
    """Small exclusive file lock with stale PID recovery."""

    def __init__(self, *, stale_seconds: int = STALE_AFTER_SECONDS) -> None:
        self.stale_seconds = max(60, int(stale_seconds))
        self._descriptor: int | None = None
        self.run_id: str | None = None

    def acquire(self, run_id: str) -> bool:
        CONTINUOUS_DIR.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        for _ in range(2):
            try:
                descriptor = os.open(
                    str(LOCK_PATH),
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
            except FileExistsError:
                record = _read_json(LOCK_PATH, {})
                try:
                    age_seconds = max(0.0, time.time() - LOCK_PATH.stat().st_mtime)
                except FileNotFoundError:
                    continue
                pid = int(record.get("pid", 0) or 0) if isinstance(record, dict) else 0
                stale = (
                    age_seconds > self.stale_seconds
                    or (pid > 0 and not _process_exists(pid))
                )
                if stale:
                    LOCK_PATH.unlink(missing_ok=True)
                    continue
                return False
            record = {
                "run_id": run_id,
                "pid": os.getpid(),
                "created_at": _now().isoformat(),
            }
            try:
                os.write(
                    descriptor,
                    (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"),
                )
                os.fsync(descriptor)
            except Exception:
                os.close(descriptor)
                LOCK_PATH.unlink(missing_ok=True)
                raise
            self._descriptor = descriptor
            return True
        raise RuntimeError("could not recover stale continuous scheduler lock")

    def release(self) -> None:
        if self._descriptor is not None:
            os.close(self._descriptor)
            self._descriptor = None
        record = _read_json(LOCK_PATH, {})
        if isinstance(record, dict) and record.get("run_id") == self.run_id:
            LOCK_PATH.unlink(missing_ok=True)

    def __enter__(self) -> "SchedulerLock":
        if self.run_id is None:
            raise RuntimeError("SchedulerLock.acquire must be called first")
        return self

    def __exit__(self, *_: Any) -> None:
        self.release()


@contextmanager
def _held_lock(stale_seconds: int) -> Iterator[SchedulerLock]:
    lock = SchedulerLock(stale_seconds=stale_seconds)
    run_id = f"continuous-{_now():%Y%m%dT%H%M%S}-{uuid4().hex[:8]}"
    if not lock.acquire(run_id):
        yield None  # type: ignore[misc]
        return
    try:
        yield lock
    finally:
        lock.release()


def _load_control() -> dict[str, Any]:
    value = _read_json(CONTROL_PATH, {})
    if not isinstance(value, dict):
        value = {}
    value.setdefault("failure_streak", 0)
    value.setdefault("last_alert_failure_streak", 0)
    value.setdefault("analysis_attempts", {})
    value.setdefault("stale_deferred", {})
    value.setdefault("coverage_by_target", {})
    return value


def _save_control(value: dict[str, Any]) -> None:
    _write_json(CONTROL_PATH, value)


def _is_paused() -> bool:
    return PAUSE_PATH.is_file()


def _paused_record() -> dict[str, Any]:
    value = _read_json(PAUSE_PATH, {})
    return value if isinstance(value, dict) else {}


def _job_metadata(job: dict[str, Any]) -> dict[str, Any]:
    value = job.get("metadata", {})
    return value if isinstance(value, dict) else {}


def _job_target(job: dict[str, Any]) -> str | None:
    value = _job_metadata(job).get("target_date")
    return str(value) if value else None


def _job_asset_types(job: dict[str, Any]) -> tuple[str, ...]:
    metadata = _job_metadata(job)
    configured = metadata.get("asset_types")
    if isinstance(configured, (list, tuple)):
        values = tuple(str(value) for value in configured if str(value) in ASSET_TYPES)
        if values:
            return values
    scope = str(metadata.get("asset_scope") or job.get("asset_scope") or TARGET_SCOPE)
    return (scope,) if scope in ASSET_TYPES else ASSET_TYPES


def _job_progress(
    state: FetchState,
    job: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    metadata = _job_metadata(job)
    asset_types = tuple(metadata.get("asset_types", ASSET_TYPES))
    target_date = str(metadata.get("target_date") or _now().date().isoformat())
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
            target_date,
            lag_trading_days=config["fetch"].get(
                "target_lag_trading_days",
                {},
            ),
        )
    return state.progress(
        asset_types,
        symbols=metadata.get("symbols", {}),
        target_date=target_dates,
        max_age_days=int(config["fetch"]["max_data_age_days"]),
    )


def _long_stale_assets(
    state: FetchState,
    target_date: str,
    config: dict[str, Any],
    asset_types: tuple[str, ...] = ASSET_TYPES,
) -> list[dict[str, Any]]:
    cutoff = date.fromisoformat(target_date) - timedelta(
        days=int(config["fetch"]["max_data_age_days"]) + LONG_STALE_EXTRA_DAYS
    )
    rows = []
    for row in state.list_assets(asset_types, statuses=["failed"], limit=100000):
        end_date = row.get("data_end_date")
        stale = not end_date
        if end_date:
            try:
                stale = date.fromisoformat(str(end_date)) < cutoff
            except ValueError:
                stale = True
        if stale and int(row.get("attempt_count", 0) or 0) >= LONG_STALE_MIN_ATTEMPTS:
            rows.append(row)
    return rows


def _apply_long_stale_policy(
    state: FetchState,
    target_date: str,
    config: dict[str, Any],
    control: dict[str, Any],
    logger: logging.Logger,
    asset_types: tuple[str, ...] = ASSET_TYPES,
) -> int:
    deferred = control.setdefault("stale_deferred", {})
    now = _utc_now()
    changed = 0
    for row in _long_stale_assets(state, target_date, config, asset_types):
        key = f"{row['asset_type']}:{row['symbol']}"
        last_attempt = str(row.get("last_attempt_at") or "")
        if deferred.get(key) == last_attempt:
            continue
        next_retry = parse_iso(row.get("next_retry_at"))
        if next_retry is not None and next_retry > now:
            continue
        retry_at = now + timedelta(seconds=LONG_STALE_RETRY_INTERVAL_SECONDS)
        if state.defer_retry(
            str(row["asset_type"]),
            str(row["symbol"]),
            next_retry_at=retry_at.isoformat(),
            reason=(
                f"long_stale_policy: retry deferred until {retry_at.isoformat()}"
            ),
        ):
            deferred[key] = last_attempt
            changed += 1
    if changed:
        logger.warning(
            "long-stale policy deferred %s failed assets for %s hours",
            changed,
            LONG_STALE_RETRY_INTERVAL_SECONDS / 3600,
        )
    return changed


def _find_target_job(
    state: FetchState,
    target_date: str,
) -> dict[str, Any] | None:
    for job in state.list_jobs(limit=200):
        if not job or job.get("kind") != "research":
            continue
        metadata = _job_metadata(job)
        if (
            str(job.get("asset_scope")) == TARGET_SCOPE
            and _job_target(job) == target_date
        ):
            return job
    return None


def _find_resumable_job(
    state: FetchState,
    job_id: str | None = None,
    *,
    limit: int = 500,
 ) -> dict[str, Any] | None:
    if job_id:
        job = state.get_job(job_id)
        if not job:
            raise RuntimeError(f"research job does not exist: {job_id}")
        if job.get("kind") != "research":
            raise RuntimeError(f"job is not a research job: {job_id}")
        if not _job_target(job):
            raise RuntimeError(f"research job has no target_date: {job_id}")
        return job
    for job in state.list_jobs(limit=limit):
        if (
            job
            and job.get("kind") == "research"
            and str(job.get("status")) in RESUMABLE_JOB_STATUSES
            and _job_target(job)
        ):
            return job
    return None


def _latest_worker_pid(job_id: str) -> int | None:
    try:
        directory = run_directory(job_id)
    except (FileNotFoundError, ValueError):
        return None
    status = _read_json(directory / "status.json", {})
    if not isinstance(status, dict):
        return None
    for key in ("worker_pid", "pid"):
        try:
            value = int(status.get(key, 0) or 0)
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            return value
    return None


def _ensure_job(
    state: FetchState,
    target_date: str,
    mode: str,
    config: dict[str, Any],
    control: dict[str, Any],
    logger: logging.Logger,
    existing_job: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str | None]:
    state.reclaim_orphaned_leases()
    job = existing_job or _find_target_job(state, target_date)
    if job:
        progress = _job_progress(state, job, config)
        current_status = str(job.get("status", ""))
        if current_status == "running":
            pid = _latest_worker_pid(str(job["job_id"]))
            if pid and _process_exists(pid):
                return job, progress, "worker_already_running"
            started_at = parse_iso(job.get("started_at"))
            age = (
                (_utc_now() - started_at).total_seconds()
                if started_at is not None
                else STALE_AFTER_SECONDS + 1
            )
            if age <= STALE_AFTER_SECONDS:
                _high_alert(
                    logger,
                    f"状态库 job {job['job_id']} 仍为 running，但找不到存活 worker PID；"
                    "暂不重复启动。",
                )
                return job, progress, "job_lock_suspect"
            state.update_job(
                str(job["job_id"]),
                status="queued",
                last_error="orphaned running job recovered by continuous scheduler",
            )
            _high_alert(logger, f"恢复 stale running job {job['job_id']}")
            job = state.get_job(str(job["job_id"])) or job
        if current_status == "paused":
            return job, progress, "analysis_retry_exhausted"
        if current_status == "failed":
            if progress["pending"] == 0:
                attempts = int(control["analysis_attempts"].get(job["job_id"], 0))
                if attempts >= MAX_ANALYSIS_RETRIES:
                    state.update_job(
                        str(job["job_id"]),
                        status="paused",
                        last_error="analysis retry limit reached for this target date",
                    )
                    return (
                        state.get_job(str(job["job_id"])) or job,
                        progress,
                        "analysis_retry_exhausted",
                    )
                control["analysis_attempts"][job["job_id"]] = attempts + 1
            state.update_job(
                str(job["job_id"]),
                status="queued",
                last_error=job.get("last_error"),
            )
            job = state.get_job(str(job["job_id"])) or job
        if (
            str(job.get("status")) in {"complete", "completed", "success"}
            and progress["total"] > 0
            and progress["pending"] == 0
        ):
            return job, progress, "target_completed"
        return job, _job_progress(state, job, config), None

    result = run_research(
        mode=mode,
        asset_scope=TARGET_SCOPE,
        symbols=None,
        _target_date=target_date,
        _start_background_worker=False,
    )
    if result.get("status") not in {"queued", "partial"}:
        raise RuntimeError(
            f"full-market job creation failed: {json.dumps(result, ensure_ascii=False, default=_json_default)}"
        )
    job_id = str(result.get("job_id") or result.get("run_id"))
    job = state.get_job(job_id)
    if not job:
        raise RuntimeError(f"created job is missing from state database: {job_id}")
    return job, _job_progress(state, job, config), None


@dataclass
class WorkerOutcome:
    pid: int | None
    returncode: int | None
    timed_out: bool
    duration_seconds: float
    stdout: str = ""
    stderr: str = ""
    error: str | None = None


def _kill_process_tree(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill.exe", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            text=True,
            check=False,
        )
    else:
        try:
            os.kill(pid, 9)
        except OSError:
            pass


def _run_worker(job_id: str, logger: logging.Logger) -> WorkerOutcome:
    WORKER_LOG_DIR.mkdir(parents=True, exist_ok=True)
    stdout_path = WORKER_LOG_DIR / f"{job_id}.stdout.log"
    stderr_path = WORKER_LOG_DIR / f"{job_id}.stderr.log"
    python = QUANT_ROOT / ".venv" / "Scripts" / "python.exe"
    script = QUANT_ROOT / "scripts" / "run_research_worker.py"
    command = [
        str(python),
        str(script),
        "--job-id",
        job_id,
        "--time-budget",
        str(WORKER_TIME_BUDGET_SECONDS),
    ]
    started = time.monotonic()
    process: subprocess.Popen[str] | None = None
    try:
        with (
            stdout_path.open("w", encoding="utf-8") as stdout_handle,
            stderr_path.open("w", encoding="utf-8") as stderr_handle,
        ):
            process = subprocess.Popen(
                command,
                cwd=str(QUANT_ROOT),
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
                creationflags=(
                    getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                    if os.name == "nt"
                    else 0
                ),
            )
            logger.info(
                "phase=worker_started run_id=%s worker_pid=%s",
                job_id,
                process.pid,
            )
            live_status = _read_json(LATEST_STATUS_PATH, {})
            if isinstance(live_status, dict):
                live_status.update(
                    {
                        "status": "running",
                        "phase": "fetch",
                        "worker_pid": process.pid,
                        "updated_at": _now().isoformat(),
                    }
                )
                _write_json(LATEST_STATUS_PATH, live_status)
            try:
                process.wait(timeout=WORKER_WALL_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                logger.error(
                    "phase=worker_timeout run_id=%s worker_pid=%s wall_timeout=%s",
                    job_id,
                    process.pid,
                    WORKER_WALL_TIMEOUT_SECONDS,
                )
                _kill_process_tree(process.pid)
                process.wait(timeout=30)
                return WorkerOutcome(
                    pid=process.pid,
                    returncode=None,
                    timed_out=True,
                    duration_seconds=time.monotonic() - started,
                    error=(
                        f"worker wall timeout after {WORKER_WALL_TIMEOUT_SECONDS} seconds"
                    ),
                )
            returncode = process.returncode
    except Exception as exc:
        return WorkerOutcome(
            pid=process.pid if process is not None else None,
            returncode=None,
            timed_out=False,
            duration_seconds=time.monotonic() - started,
            error=f"{type(exc).__name__}: {exc}",
        )
    stdout = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.exists() else ""
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else ""
    return WorkerOutcome(
        pid=process.pid if process is not None else None,
        returncode=returncode,
        timed_out=False,
        duration_seconds=time.monotonic() - started,
        stdout=stdout,
        stderr=stderr,
        error=None if returncode == 0 else f"worker exited with code {returncode}",
    )


def _latest_json(text: str) -> dict[str, Any]:
    for line in reversed((text or "").splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {}


def _public_status(
    *,
    status: str,
    phase: str,
    job: dict[str, Any] | None,
    progress: dict[str, Any] | None,
    target_date: str | None,
    start_time: datetime | None,
    finished_at: datetime | None = None,
    worker_pid: int | None = None,
    worker_returncode: int | None = None,
    worker_timeout: bool = False,
    last_error: str | None = None,
    failure_streak: int = 0,
    analysis_attempts: int = 0,
    round_duration_seconds: float | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    progress = progress or {}
    metadata = _job_metadata(job) if job else {}
    asset_scope = str(
        metadata.get("asset_scope")
        or (job.get("asset_scope") if job else None)
        or TARGET_SCOPE
    )
    asset_types = _job_asset_types(job) if job else ASSET_TYPES
    value: dict[str, Any] = {
        "status": status,
        "phase": phase,
        "start_time": start_time.isoformat() if start_time else None,
        "finished_at": finished_at.isoformat() if finished_at else None,
        "updated_at": _now().isoformat(),
        "run_id": job.get("job_id") if job else None,
        "job_id": job.get("job_id") if job else None,
        "worker_pid": worker_pid,
        "worker_returncode": worker_returncode,
        "worker_timeout": bool(worker_timeout),
        "target_date": target_date,
        "asset_scope": asset_scope,
        "asset_types": list(asset_types),
        "total": int(progress.get("total", 0) or 0),
        "completed": int(progress.get("completed", 0) or 0),
        "coverage": float(progress.get("coverage", 0.0) or 0.0),
        "failed": int(progress.get("failed", 0) or 0),
        "pending": int(progress.get("pending", 0) or 0),
        "stale": int(progress.get("stale", 0) or 0),
        "by_asset_type": progress.get("by_asset_type", {}),
        "last_error": last_error,
        "failure_streak": int(failure_streak),
        "analysis_attempts": int(analysis_attempts),
        "round_duration_seconds": round_duration_seconds,
        "result_path": (
            str(run_directory(str(job["job_id"]))) if job else None
        ),
        "current_path": str(
            QUANT_ROOT / "results" / "assistant" / "current.json"
        ),
    }
    if extra:
        value.update(extra)
    return value


def _log_progress(
    logger: logging.Logger,
    status: dict[str, Any],
    *,
    phase: str | None = None,
) -> None:
    logger.info(
        "phase=%s run_id=%s worker_pid=%s completed=%s total=%s coverage=%.4f "
        "failed=%s pending=%s stale=%s last_error=%s duration=%s",
        phase or status.get("phase"),
        status.get("run_id"),
        status.get("worker_pid"),
        status.get("completed"),
        status.get("total"),
        float(status.get("coverage", 0.0) or 0.0),
        status.get("failed"),
        status.get("pending"),
        status.get("stale"),
        status.get("last_error"),
        status.get("round_duration_seconds"),
    )


def _record_failure(
    control: dict[str, Any],
    *,
    failed: bool,
    job_id: str | None,
    target_date: str | None,
    coverage: float,
    logger: logging.Logger,
) -> int:
    streak = int(control.get("failure_streak", 0) or 0)
    if failed:
        streak += 1
    else:
        streak = 0
    control["failure_streak"] = streak
    if target_date:
        previous = control.setdefault("coverage_by_target", {}).get(target_date)
        if previous is not None and coverage < float(previous) - 1e-9:
            _high_alert(
                logger,
                f"target_date={target_date} coverage dropped from {previous:.4f} to {coverage:.4f}",
            )
        control["coverage_by_target"][target_date] = coverage
    if streak >= 3 and int(control.get("last_alert_failure_streak", 0) or 0) < 3:
        _high_alert(
            logger,
            f"continuous research failed for {streak} consecutive rounds; "
            f"job_id={job_id or 'unknown'}",
        )
        control["last_alert_failure_streak"] = streak
    elif streak < 3:
        control["last_alert_failure_streak"] = 0
    return streak


def _calendar_today(logger: logging.Logger) -> dict[str, Any]:
    today = _now().date()
    try:
        value = resolve_trading_day(today)
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        _high_alert(logger, f"trading calendar unavailable; no fetch started: {message}")
        return {
            "status": "error",
            "date": today.isoformat(),
            "is_trading_day": False,
            "error": message,
        }
    logger.info(
        "phase=calendar date=%s source=%s coverage=%s..%s online_error=%s",
        value.get("date"),
        value.get("source"),
        value.get("coverage_start"),
        value.get("coverage_end"),
        value.get("online_error"),
    )
    return value


def run_once(
    *,
    job_id: str | None = None,
    worker_runner: Callable[[str, logging.Logger], WorkerOutcome] | None = None,
) -> dict[str, Any]:
    config = load_config()
    logger = _logger()
    stale_seconds = int(config.get("runtime", {}).get("lock_stale_seconds", STALE_AFTER_SECONDS))
    with _held_lock(stale_seconds) as lock:
        if lock is None:
            current = _read_json(LATEST_STATUS_PATH, {})
            logger.info("phase=lock status=running; skipped overlapping scheduler tick")
            return {
                "status": "running",
                "skipped": "scheduler_lock_held",
                "latest_status": current,
            }
        control = _load_control()
        started = _now()
        if _is_paused():
            record = _paused_record()
            status = _public_status(
                status="paused",
                phase="paused",
                job=None,
                progress={},
                target_date=started.date().isoformat(),
                start_time=started,
                finished_at=_now(),
                last_error=str(record.get("reason") or "automatic continuation paused"),
                failure_streak=int(control.get("failure_streak", 0) or 0),
            )
            _write_json(LATEST_STATUS_PATH, status)
            _log_progress(logger, status)
            return status

        state = FetchState(Path(config["fetch"]["state_db"]))
        try:
            existing_job = _find_resumable_job(state, job_id)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            status = _public_status(
                status="failed",
                phase="queue",
                job=None,
                progress={},
                target_date=started.date().isoformat(),
                start_time=started,
                finished_at=_now(),
                last_error=error,
                failure_streak=int(control.get("failure_streak", 0) or 0),
            )
            _write_json(LATEST_STATUS_PATH, status)
            _log_progress(logger, status)
            return status

        if existing_job:
            target_date = str(_job_target(existing_job))
            mode = str(_job_metadata(existing_job).get("mode") or "daily")
            calendar = {
                "status": "not_required",
                "date": target_date,
                "reason": "resuming_existing_job",
            }
            logger.info(
                "phase=queue action=resume_existing_job job_id=%s "
                "target_date=%s asset_scope=%s",
                existing_job["job_id"],
                target_date,
                existing_job.get("asset_scope"),
            )
        else:
            calendar = _calendar_today(logger)
        if not existing_job and (
            calendar.get("status") != "ok"
            or not calendar.get("is_trading_day")
        ):
            reason = (
                str(calendar.get("error"))
                if calendar.get("status") != "ok"
                else f"{calendar.get('date')} is not a trading day"
            )
            status = _public_status(
                status="paused",
                phase="calendar",
                job=None,
                progress={},
                target_date=str(calendar.get("date") or started.date().isoformat()),
                start_time=started,
                finished_at=_now(),
                last_error=reason,
                failure_streak=int(control.get("failure_streak", 0) or 0),
                extra={"calendar": calendar},
            )
            _write_json(LATEST_STATUS_PATH, status)
            _log_progress(logger, status)
            return status

        if not existing_job:
            target_date = str(calendar["date"])
            mode = "monthly" if calendar.get("is_month_end_trading_day") else "daily"
        asset_types = _job_asset_types(existing_job) if existing_job else ASSET_TYPES
        _apply_long_stale_policy(
            state,
            target_date,
            config,
            control,
            logger,
            asset_types,
        )
        _write_json(
            LATEST_STATUS_PATH,
            _public_status(
                status="running",
                phase="queue",
                job=None,
                progress={},
                target_date=target_date,
                start_time=started,
                failure_streak=int(control.get("failure_streak", 0) or 0),
            ),
        )
        try:
            job, progress, action = _ensure_job(
                state,
                target_date,
                mode,
                config,
                control,
                logger,
                existing_job,
            )
            _save_control(control)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            _high_alert(logger, f"queue/state failure; no worker started: {error}")
            status = _public_status(
                status="failed",
                phase="queue",
                job=None,
                progress={},
                target_date=target_date,
                start_time=started,
                finished_at=_now(),
                last_error=error,
                failure_streak=_record_failure(
                    control,
                    failed=True,
                    job_id=None,
                    target_date=target_date,
                    coverage=0.0,
                    logger=logger,
                ),
            )
            _save_control(control)
            _write_json(LATEST_STATUS_PATH, status)
            _log_progress(logger, status)
            return status

        if job is None:
            raise RuntimeError("scheduler could not resolve a research job")
        target_date = str(_job_target(job) or target_date)
        job_id = str(job["job_id"])
        analysis_attempts = int(control["analysis_attempts"].get(job_id, 0) or 0)
        if action in {"worker_already_running", "job_lock_suspect"}:
            status = _public_status(
                status="running",
                phase="worker",
                job=job,
                progress=progress,
                target_date=target_date,
                start_time=started,
                worker_pid=_latest_worker_pid(job_id),
                last_error=(
                    "已有 worker 运行"
                    if action == "worker_already_running"
                    else "状态库 job 可能被遗留锁定"
                ),
                failure_streak=int(control.get("failure_streak", 0) or 0),
                analysis_attempts=analysis_attempts,
                extra={"action": action},
            )
            _write_json(LATEST_STATUS_PATH, status)
            _log_progress(logger, status)
            return status
        if action == "analysis_retry_exhausted":
            status = _public_status(
                status="paused",
                phase="research",
                job=job,
                progress=progress,
                target_date=target_date,
                start_time=started,
                finished_at=_now(),
                last_error=job.get("last_error"),
                failure_streak=int(control.get("failure_streak", 0) or 0),
                analysis_attempts=analysis_attempts,
                extra={"action": action},
            )
            _write_json(LATEST_STATUS_PATH, status)
            _log_progress(logger, status)
            return status
        if action == "target_completed":
            status = _public_status(
                status="completed",
                phase="idle",
                job=job,
                progress=progress,
                target_date=target_date,
                start_time=started,
                finished_at=_now(),
                last_error=None,
                failure_streak=int(control.get("failure_streak", 0) or 0),
                analysis_attempts=analysis_attempts,
                extra={
                    "action": action,
                    "message": "目标交易日已完成，等待下一交易日刷新。",
                    "checkpoint_retained": True,
                },
            )
            _save_control(control)
            _write_json(LATEST_STATUS_PATH, status)
            _log_progress(logger, status)
            return status

        status = _public_status(
            status="running",
            phase="fetch",
            job=job,
            progress=progress,
            target_date=target_date,
            start_time=started,
            failure_streak=int(control.get("failure_streak", 0) or 0),
            analysis_attempts=analysis_attempts,
        )
        _write_json(LATEST_STATUS_PATH, status)
        _log_progress(logger, status)
        outcome = (worker_runner or _run_worker)(job_id, logger)
        worker_json = _latest_json(outcome.stdout)
        state.reclaim_orphaned_leases()
        job = state.get_job(job_id) or job
        progress = _job_progress(state, job, config)
        last_error = (
            outcome.error
            or worker_json.get("error")
            or job.get("last_error")
            or (
                str(worker_json.get("failure_reasons"))
                if worker_json.get("failure_reasons")
                else None
            )
        )
        data_quality_failure = any(
            token in str(last_error or "").lower()
            for token in ("data_quality", "quality", "gate", "qlib", "model")
        )
        round_failed = bool(
            outcome.timed_out
            or outcome.returncode not in (0, None)
            or str(job.get("status")) == "failed"
            or data_quality_failure
        )
        failure_streak = _record_failure(
            control,
            failed=round_failed,
            job_id=job_id,
            target_date=target_date,
            coverage=float(progress.get("coverage", 0.0) or 0.0),
            logger=logger,
        )
        if data_quality_failure:
            _high_alert(logger, f"data quality/model gate failure in job {job_id}: {last_error}")
        if outcome.timed_out:
            _high_alert(logger, f"worker timeout in job {job_id}; fetch checkpoint retained")
        if outcome.returncode not in (0, None) and not outcome.timed_out:
            _high_alert(
                logger,
                "worker abnormal exit "
                f"run_id={job_id} returncode={outcome.returncode} "
                f"stderr_tail={outcome.stderr[-1000:]}",
            )
        if int(progress.get("failed", 0) or 0) > 0:
            logger.warning(
                "phase=fetch_partial run_id=%s failed_assets=%s; "
                "individual checkpoints retained for retry",
                job_id,
                progress.get("failed"),
            )
        if progress["completed"] == progress["total"] and progress["total"] > 0:
            final_status = (
                "completed"
                if str(job.get("status")) in {"complete", "completed", "success"}
                and not round_failed
                else "failed"
            )
        else:
            final_status = "failed" if round_failed else "partial"
        finished = _now()
        status = _public_status(
            status=final_status,
            phase="research" if progress["pending"] == 0 else "fetch",
            job=job,
            progress=progress,
            target_date=target_date,
            start_time=started,
            finished_at=finished,
            worker_pid=outcome.pid,
            worker_returncode=outcome.returncode,
            worker_timeout=outcome.timed_out,
            last_error=last_error,
            failure_streak=failure_streak,
            analysis_attempts=analysis_attempts,
            round_duration_seconds=(finished - started).total_seconds(),
            extra={
                "worker_result": worker_json,
                "calendar": calendar,
                "checkpoint_retained": True,
            },
        )
        _save_control(control)
        _write_json(LATEST_STATUS_PATH, status)
        _log_progress(logger, status)
        return status


def status() -> dict[str, Any]:
    value = _read_json(LATEST_STATUS_PATH, {})
    if not isinstance(value, dict):
        value = {}
    value["scheduler_paused"] = _is_paused()
    value["pause_record"] = _paused_record() if _is_paused() else None
    config = load_config()
    try:
        state = FetchState(Path(config["fetch"]["state_db"]))
        target = str(value.get("target_date") or _now().date().isoformat())
        job_id = value.get("job_id")
        job = state.get_job(str(job_id)) if job_id else _find_target_job(state, target)
        if job:
            progress = _job_progress(state, job, config)
            value.update(
                {
                    "job_id": job["job_id"],
                    "run_id": job["job_id"],
                    "job_status": job.get("status"),
                    "total": progress["total"],
                    "completed": progress["completed"],
                    "coverage": progress["coverage"],
                    "failed": progress["failed"],
                    "pending": progress["pending"],
                    "stale": progress["stale"],
                    "by_asset_type": progress["by_asset_type"],
                }
            )
    except Exception as exc:
        value["status_error"] = f"{type(exc).__name__}: {exc}"
    return value


def coverage() -> dict[str, Any]:
    value = status()
    return {
        "status": value.get("status", "unknown"),
        "run_id": value.get("run_id"),
        "target_date": value.get("target_date"),
        "total": value.get("total", 0),
        "completed": value.get("completed", 0),
        "coverage": value.get("coverage", 0.0),
        "failed": value.get("failed", 0),
        "pending": value.get("pending", 0),
        "stale": value.get("stale", 0),
        "by_asset_type": value.get("by_asset_type", {}),
    }


def errors(limit: int = 50) -> dict[str, Any]:
    config = load_config()
    state = FetchState(Path(config["fetch"]["state_db"]))
    rows = state.list_assets(ASSET_TYPES, statuses=["failed"], limit=max(1, int(limit)))
    return {
        "status": "ok",
        "count": len(rows),
        "errors": [
            {
                "asset_type": row.get("asset_type"),
                "symbol": row.get("symbol"),
                "last_error": row.get("last_error"),
                "last_attempt_at": row.get("last_attempt_at"),
                "last_retry_time": row.get("last_attempt_at"),
                "next_retry_at": row.get("next_retry_at"),
                "updated_at": row.get("updated_at"),
                "attempt_count": row.get("attempt_count"),
                "data_end_date": row.get("data_end_date"),
            }
            for row in rows
        ],
    }


def pause(reason: str = "paused by operator") -> dict[str, Any]:
    record = {
        "status": "paused",
        "reason": reason,
        "paused_at": _now().isoformat(),
        "pid": os.getpid(),
    }
    _write_json(PAUSE_PATH, record)
    _write_json(
        LATEST_STATUS_PATH,
        {
            **status(),
            "status": "paused",
            "phase": "paused",
            "last_error": reason,
            "updated_at": _now().isoformat(),
        },
    )
    _logger().warning("automatic continuation paused: %s", reason)
    return record


def resume() -> dict[str, Any]:
    PAUSE_PATH.unlink(missing_ok=True)
    value = {
        "status": "resumed",
        "resumed_at": _now().isoformat(),
    }
    _logger().info("automatic continuation resumed")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action",
        choices=("once", "start", "status", "pause", "resume", "errors", "coverage"),
        default="status",
    )
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--reason", default="paused by operator")
    parser.add_argument("--job-id")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.action in {"once", "start"}:
            value = run_once(job_id=args.job_id)
        elif args.action == "status":
            value = status()
        elif args.action == "pause":
            value = pause(args.reason)
        elif args.action == "resume":
            value = resume()
        elif args.action == "errors":
            value = errors(args.limit)
        else:
            value = coverage()
        print(json.dumps(value, ensure_ascii=True, sort_keys=True, default=_json_default))
        return 0
    except Exception as exc:
        value = {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }
        print(json.dumps(value, ensure_ascii=True, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
