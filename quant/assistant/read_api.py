"""对外只读 API（自 research.py 逐字搬迁，docs/重构计划.md 阶段 1）。

get_current_snapshot / get_run_status / 通知查询与确认。跨模块名字来源约定：
定义在 research.py 的常量与工具（ASSISTANT_RESULTS_DIR、RUNS_DIR、
ASSET_TYPES、load_config、_now、
_process_exists）经 ``research.`` 属性访问，使 patch.object(research, ...)
对搬入的函数继续生效；_fetch_state/_target_dates_for_job 来自 queueing，
其余名字直接来自 paths/notifications。
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from . import research
from .notifications import acknowledge, list_records
from .paths import (
    fixed_path,
    read_json,
    run_directory,
    validate_notification_id,
    validate_run_id,
    validate_symbol,
)
from .queueing import _fetch_state, _target_dates_for_job

def _fetch_progress_snapshot(
    asset_types: tuple[str, ...],
    symbols: dict[str, list[str]] | None,
) -> dict[str, Any] | None:
    """只读抓取状态库，给出一个切片相对交易日目标的时效。

    与抓取队列共用同一套 target_lag_trading_days 判定，因此这里的
    pending/stale 就是"现在刷新会做什么"的直接答案。任何失败都返回
    None，绝不影响快照主流程。
    """
    try:
        config = research.load_config()
        state = _fetch_state(config)
        target_date = research._now().date().isoformat()
        target_dates = _target_dates_for_job(asset_types, target_date, config)
        progress = state.progress(
            asset_types,
            symbols=symbols,
            target_date=target_dates,
            max_age_days=int(config["fetch"]["max_data_age_days"]),
        )
    except Exception:
        return None
    return {
        "target_dates": target_dates,
        "total": int(progress.get("total", 0) or 0),
        "completed": int(progress.get("completed", 0) or 0),
        "pending": int(progress.get("pending", 0) or 0),
        "failed": int(progress.get("failed", 0) or 0),
        "stale": int(progress.get("stale", 0) or 0),
        "coverage": float(progress.get("coverage", 0.0) or 0.0),
    }


def get_current_snapshot(asset_type: str | None = None, symbol: str | None = None) -> dict[str, Any]:
    current = read_json(research.ASSISTANT_RESULTS_DIR / "current.json")
    normalized_type = str(asset_type).strip().lower() if asset_type else None
    if normalized_type is not None and normalized_type not in research.ASSET_TYPES:
        raise ValueError(f"asset_type must be one of: {', '.join(research.ASSET_TYPES)}")
    normalized_symbol = validate_symbol(symbol) if symbol else None
    if normalized_type:
        freshness_types = (normalized_type,)
        freshness_symbols = (
            {normalized_type: [normalized_symbol]} if normalized_symbol else None
        )
    else:
        freshness_types = tuple(research.ASSET_TYPES)
        freshness_symbols = None
    freshness = _fetch_progress_snapshot(freshness_types, freshness_symbols)

    def attach(payload: dict[str, Any]) -> dict[str, Any]:
        if freshness is not None:
            payload["freshness"] = freshness
        return payload

    if current is None:
        return attach(
            {
                "status": "no_successful_run",
                "message": "尚无成功研究结果；请先运行 run_research。",
                "current_path": str(research.ASSISTANT_RESULTS_DIR / "current.json"),
            }
        )
    current = research._sanitize_result_record(current)
    if asset_type is None and symbol is None:
        return attach(current)
    paths = current.get("paths", {})
    model_signals = (
        read_json(fixed_path(paths["model_signals"]))
        if paths.get("model_signals")
        else {}
    )
    selected: list[dict[str, Any]] = []
    for category, analysis in model_signals.items():
        if normalized_type and category != normalized_type:
            continue
        for signal in analysis.get("signals", []):
            if normalized_symbol and signal.get("symbol") != normalized_symbol:
                continue
            selected.append({"asset_type": category, **signal})
    if not selected:
        return attach(
            {
                "status": "not_found_or_stale",
                "requested_asset_type": normalized_type,
                "requested_symbol": normalized_symbol,
                "run_id": current.get("run_id"),
                "data_end_date": current.get("data_end_date"),
                "signals": [],
                "reason": "当前成功快照没有匹配该资产类型和代码的信号。",
                "result_path": research._fixed_reference(current.get("result_path")),
            }
        )
    decision = current.get("decision_support", {})
    asset_decision = (
        decision.get("asset_paths", {}).get(normalized_type)
        if normalized_type and isinstance(decision, Mapping)
        else decision
    )
    return attach(
        {
            "status": current.get("status"),
            "run_id": current.get("run_id"),
            "data_end_date": current.get("data_end_date"),
            "result_path": research._fixed_reference(current.get("result_path")),
            "requested_asset_type": normalized_type,
            "requested_symbol": normalized_symbol,
            "decision_support": asset_decision or {},
            "signals": selected,
        }
    )


def _latest_run_id() -> str | None:
    current = read_json(research.ASSISTANT_RESULTS_DIR / "current.json", {})
    if isinstance(current, dict) and current.get("run_id"):
        try:
            return validate_run_id(current["run_id"])
        except ValueError:
            pass
    candidates = []
    for status_path in research.RUNS_DIR.rglob("status.json"):
        candidate = status_path.parent.name
        try:
            validate_run_id(candidate)
        except ValueError:
            continue
        candidates.append((status_path.stat().st_mtime, candidate))
    return max(candidates)[1] if candidates else None


def get_run_status(run_id: str | None = None) -> dict[str, Any]:
    run_id = validate_run_id(run_id) if run_id else _latest_run_id()
    if run_id is None:
        return {
            "status": "no_runs",
            "message": "尚无研究运行记录。",
            "result_path": str(research.RUNS_DIR),
        }
    directory = run_directory(run_id)
    status = read_json(directory / "status.json", {})
    if not isinstance(status, dict):
        status = {}
    status["run_id"] = run_id
    status["result_path"] = str(directory)
    try:
        config = research.load_config()
        state = _fetch_state(config)
        job = state.get_job(run_id)
    except Exception:
        config = research.load_config()
        job = None
    if job:
        metadata = job.get("metadata", {})
        asset_types = metadata.get("asset_types", [])
        symbols = metadata.get("symbols", {})
        target_date = metadata.get("target_date") or research._now().date().isoformat()
        try:
            target_dates = _target_dates_for_job(
                asset_types,
                str(target_date),
                config,
                metadata,
            )
            progress = state.progress(
                asset_types,
                symbols=symbols,
                target_date=target_dates,
                max_age_days=int(config["fetch"]["max_data_age_days"]),
            )
        except Exception:
            progress = status.get("progress", {})
        job_status = str(job.get("status") or status.get("status") or "unknown")
        worker_pid = int(status.get("worker_pid", 0) or 0)
        if job_status == "running" and worker_pid and not research._process_exists(worker_pid):
            job_status = "failed"
            error = "research worker exited while the job was still running"
            state.update_job(
                run_id,
                status="failed",
                finished_at=research._now().isoformat(),
                last_error=error,
            )
            status["error"] = error
        elif job_status == "running":
            heartbeat = status.get("last_heartbeat_at")
            try:
                heartbeat_at = datetime.fromisoformat(
                    str(heartbeat).replace("Z", "+00:00")
                ).astimezone(timezone.utc)
            except (TypeError, ValueError):
                heartbeat_at = None
            stalled_after = max(
                1,
                int(config["fetch"].get("worker_stalled_seconds", 60)),
            )
            if heartbeat_at and (research._now() - heartbeat_at).total_seconds() >= stalled_after:
                job_status = "stalled"
                status["message"] = "worker 心跳已超过停滞阈值；请查询同一 run_id，勿重复提交。"
        status.update(
            {
                "status": job_status,
                "target_date": target_date,
                "target_dates": progress.get(
                    "target_dates",
                    metadata.get("target_dates", {}),
                ),
                "total": job.get("total", progress.get("total", 0)),
                "completed": job.get("completed", progress.get("completed", 0)),
                "pending": job.get("pending", progress.get("pending", 0)),
                "failed": job.get("failed", progress.get("failed", 0)),
                "stale": job.get("stale", progress.get("stale", 0)),
                "coverage": job.get("coverage", progress.get("coverage", 0.0)),
            }
        )
    compact_keys = (
        "run_id",
        "status",
        "phase",
        "depth",
        "queue_wait_seconds",
        "stage_timings",
        "research_phase",
        "total",
        "completed",
        "pending",
        "failed",
        "stale",
        "coverage",
        "target_date",
        "target_dates",
        "last_progress_at",
        "last_completed_at",
        "last_heartbeat_at",
        "progress_seq",
        "worker_pid",
        "started_at",
        "finished_at",
        "result_run_id",
        "data_end_date",
        "result_path",
        "error",
        "message",
    )
    return {key: status[key] for key in compact_keys if key in status}


def get_rolling_scan(
    asset_type: str | None = None,
    symbol: str | None = None,
    *,
    limit: int = 30,
) -> dict[str, Any]:
    """Read the rolling-scan snapshot (preliminary pool built during a scan)."""
    config = research.load_config()
    snapshot_path = fixed_path(
        (config.get("rolling_scan") or {}).get(
            "snapshot_path",
            research.ASSISTANT_RESULTS_DIR / "rolling_scan.json",
        )
    )
    snapshot = read_json(snapshot_path)
    if snapshot is None:
        return {
            "status": "no_rolling_scan",
            "message": "尚无滚动扫描快照；全市场抓取进行中会按波次自动生成。",
            "snapshot_path": str(snapshot_path),
        }
    normalized_type = str(asset_type).strip().lower() if asset_type else None
    if normalized_type is not None and normalized_type not in research.ASSET_TYPES:
        raise ValueError(f"asset_type must be one of: {', '.join(research.ASSET_TYPES)}")
    normalized_symbol = validate_symbol(symbol) if symbol else None
    bound = max(1, min(int(limit), 100))

    def select(entries: Any) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        if not isinstance(entries, list):
            return selected
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if normalized_type and entry.get("asset_type") != normalized_type:
                continue
            if normalized_symbol and entry.get("symbol") != normalized_symbol:
                continue
            selected.append(entry)
            if len(selected) >= bound:
                break
        return selected

    result: dict[str, Any] = {
        key: snapshot[key]
        for key in (
            "status",
            "pool_status",
            "preliminary",
            "disclaimer",
            "updated_at",
            "job_id",
            "coverage",
            "scored_total",
            "eligible_count",
            "analyzed_this_pass",
        )
        if key in snapshot
    }
    result.update(
        {
            "limit": bound,
            "pool": select(snapshot.get("pool")),
            "score_table": select(snapshot.get("score_table")),
            "recent_events": select(snapshot.get("recent_events")),
        }
    )
    return result


def list_notifications(
    unread_only: bool = False,
    *,
    limit: int = 10,
    since: str | None = None,
    run_id: str | None = None,
 ) -> dict[str, Any]:
    validated_run_id = validate_run_id(run_id) if run_id is not None else None
    records = list_records(
        bool(unread_only),
        limit=int(limit),
        since=since,
        run_id=validated_run_id,
    )
    return {
        "status": "ok",
        "unread_only": bool(unread_only),
        "limit": int(limit),
        "since": since,
        "run_id": validated_run_id,
        "count": len(records),
        "notifications": records,
    }


def ack_notification(notification_id: str) -> dict[str, Any]:
    return acknowledge(validate_notification_id(notification_id))


def _fixed_reference(value: Any) -> str | None:
    """Validate a persisted path before returning it through the API."""
    if value is None:
        return None
    return str(fixed_path(str(value)))


def _sanitize_result_record(value: Any) -> Any:
    """Keep persisted result references inside the fixed quant root."""
    if not isinstance(value, dict):
        return value
    record = dict(value)
    for key in ("result_path", "current_path"):
        if key in record and record[key] is not None:
            record[key] = _fixed_reference(record[key])
    paths = record.get("paths")
    if isinstance(paths, dict):
        record["paths"] = {
            str(key): _fixed_reference(path) if path is not None else None
            for key, path in paths.items()
        }
    return record
