"""排队 / 后台 worker / 数据刷新 / run 状态落盘与进程探活（自 research.py 逐字搬迁，docs/重构计划.md 阶段 1）。

跨模块名字来源约定：定义在 research.py 的常量与工具（DATA_DIR、RUNS_DIR、
ASSISTANT_RESULTS_DIR、_now、_run_id、_process_exists、_universe、_jsonable、
_fetch_summary）经 ``research.`` 属性访问，使 patch.object(research, ...)
对搬入的函数继续生效；_data_file 来自 data_layer，其余名字为模块内自有或
直接来自 paths/config/scripts。
"""
from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from errno import ESRCH
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from . import research
from .config import config_snapshot
from .data_layer import _data_file
from .paths import (
    atomic_write_json,
    atomic_write_text,
    read_json,
    run_directory,
    validate_symbol,
)

try:
    from scripts.fetch_state import FetchState
    from scripts.fetch_targets import asset_target_dates
    from scripts.universe import load_asset_metadata
except ImportError:
    scripts_root = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts_root) not in sys.path:
        sys.path.insert(0, str(scripts_root))
    from fetch_state import FetchState
    from fetch_targets import asset_target_dates
    from universe import load_asset_metadata

def _fetch_state(config: Mapping[str, Any]) -> FetchState:
    configured = config.get("fetch", {}).get(
        "state_db",
        research.DATA_DIR / "fetch_state.sqlite3",
    )
    return FetchState(Path(str(configured)))


def _target_dates_for_job(
    asset_types: Iterable[str],
    target_date: str,
    config: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
 ) -> dict[str, str]:
    selected_types = tuple(str(value) for value in asset_types)
    configured = (metadata or {}).get("target_dates")
    if isinstance(configured, Mapping) and all(
        asset_type in configured for asset_type in selected_types
    ):
        return {
            asset_type: str(configured[asset_type])
            for asset_type in selected_types
        }
    return asset_target_dates(
        selected_types,
        target_date,
        lag_trading_days=config.get("fetch", {}).get(
            "target_lag_trading_days",
            {},
        ),
    )


def _cached_end_date(asset_type: str, symbol: str) -> str | None:
    path = _data_file(asset_type, symbol)
    metadata = load_asset_metadata(path) if path.exists() else {}
    cached = metadata.get("data_end_date") if isinstance(metadata, Mapping) else None
    if cached:
        return str(cached)
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path)
        date_column = next(
            (
                column
                for column in frame.columns
                if str(column).strip() in {"date", "日期", "净值日期"}
            ),
            None,
        )
        if date_column is None:
            return None
        latest = pd.to_datetime(frame[date_column], errors="coerce").max()
        return None if pd.isna(latest) else latest.date().isoformat()
    except Exception:
        return None


def _normalize_symbol_routes(
    asset_types: Iterable[str],
    symbols_by_type: Mapping[str, Iterable[str]],
 ) -> dict[str, list[str]]:
    return {
        asset_type: sorted(
            {validate_symbol(symbol) for symbol in symbols_by_type.get(asset_type, [])}
        )
        for asset_type in asset_types
        if symbols_by_type.get(asset_type)
    }


def _queue_research(
    *,
    mode: str,
    asset_scope: str,
    asset_types: tuple[str, ...],
    config: Mapping[str, Any],
    symbols_by_type: Mapping[str, Iterable[str]] | None = None,
    target_date: str | None = None,
    start_background_worker: bool = True,
    depth: str = "standard",
) -> dict[str, Any]:
    """Persist a research request without holding the MCP call open."""
    depth = str(depth or "standard").strip().lower()
    if depth not in research.DEPTH_NAMES:
        raise ValueError(
            f"depth must be one of: {', '.join(sorted(research.DEPTH_NAMES))}"
        )
    state = _fetch_state(config)
    full_universe = symbols_by_type is None
    requested_routes = (
        {}
        if full_universe
        else _normalize_symbol_routes(asset_types, symbols_by_type)
    )
    selected_types = (
        asset_types
        if full_universe
        else tuple(
            asset_type for asset_type in asset_types if requested_routes.get(asset_type)
        )
    )
    if not selected_types:
        raise ValueError("no requested symbols matched the selected asset scope")
    effective_target_date = str(target_date or research._now().date().isoformat())
    try:
        datetime.fromisoformat(effective_target_date).date()
    except ValueError as exc:
        raise ValueError(f"invalid target_date: {effective_target_date}") from exc
    requested_target_dates = _target_dates_for_job(
        selected_types, effective_target_date, config
    )
    for existing in state.list_jobs(limit=50):
        if not existing or existing.get("kind") != "research":
            continue
        if existing.get("status") not in {"queued", "running", "partial"}:
            continue
        metadata = existing.get("metadata", {})
        if (
            metadata.get("asset_scope") != asset_scope
            or metadata.get("mode") != mode
            or bool(metadata.get("full_universe", True)) != full_universe
            or str(metadata.get("depth", "standard")) != depth
        ):
            continue
        existing_target_date = metadata.get("target_date")
        if str(existing_target_date or "") != effective_target_date:
            continue
        existing_routes = {
            asset_type: list(values)
            for asset_type, values in metadata.get("symbols", {}).items()
        }
        if not full_universe and existing_routes != requested_routes:
            continue
        existing_id = str(existing["job_id"])
        effective_target_dates = _target_dates_for_job(
            selected_types,
            effective_target_date,
            config,
            metadata,
        )
        if metadata.get("target_dates") != effective_target_dates:
            metadata = {**metadata, "target_dates": effective_target_dates}
            state.update_job(existing_id, metadata=metadata)
        progress = state.progress(
            selected_types,
            symbols=existing_routes,
            target_date=effective_target_dates,
            max_age_days=int(config["fetch"]["max_data_age_days"]),
        )
        existing_status = read_json(run_directory(existing_id) / "status.json", {})
        worker_pid = int(existing_status.get("worker_pid", 0) or 0)
        if existing.get("status") == "running" and research._process_exists(worker_pid):
            worker_info = {"started": False, "reason": "already_running", "pid": worker_pid}
        else:
            worker_info = (
                _start_background_worker(existing_id, config)
                if start_background_worker
                else {"started": False, "reason": "caller_will_start_worker"}
            )
        scope_label = "全市场" if full_universe else "定向"
        return research._jsonable(
            {
                "run_id": existing_id,
                "job_id": existing_id,
                "status": existing.get("status", "queued"),
                "phase": existing_status.get("phase", "cache_check"),
                "completed": progress.get("completed", 0),
                "pending": progress.get("pending", 0),
                "failed": progress.get("failed", 0),
                "coverage": progress.get("coverage", 0.0),
                "target_date": effective_target_date,
                "target_dates": effective_target_dates,
                "result_path": str(run_directory(existing_id)),
                "current_path": str(research.ASSISTANT_RESULTS_DIR / "current.json"),
                "message": f"已有同范围{scope_label}任务在推进，已复用原 run_id。",
                "worker": worker_info,
            }
        )
    run_id = research._run_id()
    started = research._now()
    run_dir = (
        research.RUNS_DIR
        / started.strftime("%Y")
        / started.strftime("%m")
        / started.strftime("%d")
        / run_id
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    if full_universe:
        routes: dict[str, list[str]] = {}
        for asset_type in selected_types:
            routes[asset_type] = research._universe(asset_type, None)
            state.register_assets(asset_type, routes[asset_type])
    else:
        routes = requested_routes
        for asset_type, scoped_symbols in routes.items():
            cached_end_dates = {
                symbol: _cached_end_date(asset_type, symbol)
                for symbol in scoped_symbols
            }
            state.register_assets(
                asset_type, scoped_symbols, data_end_dates=cached_end_dates
            )
    target_dates = requested_target_dates
    progress = state.progress(
        selected_types,
        symbols=routes,
        target_date=target_dates,
        max_age_days=int(config["fetch"]["max_data_age_days"]),
    )
    state.create_job(
        run_id,
        kind="research",
        asset_scope=asset_scope,
        metadata={
            "mode": mode,
            "depth": depth,
            "asset_scope": asset_scope,
            "asset_types": list(selected_types),
            "symbols": routes,
            "full_universe": full_universe,
            "target_date": effective_target_date,
            "target_dates": target_dates,
            "phase": "cache_check",
        },
    )
    state.update_job(
        run_id,
        total=progress["total"],
        completed=progress["completed"],
        pending=progress["pending"],
        failed=progress["failed"],
        stale=progress["stale"],
        coverage=progress["coverage"],
    )
    status = {
        "run_id": run_id,
        "job_id": run_id,
        "status": "queued",
        "started_at": started.isoformat(),
        "asset_scope": asset_scope,
        "asset_types": list(selected_types),
        "full_universe": full_universe,
        "depth": depth,
        "target_date": effective_target_date,
        "target_dates": target_dates,
        "phase": "cache_check",
        "completed": progress["completed"],
        "pending": progress["pending"],
        "failed": progress["failed"],
        "coverage": progress["coverage"],
        "progress_seq": 0,
        "last_progress_at": started.isoformat(),
        "last_heartbeat_at": started.isoformat(),
        "result_path": str(run_dir),
        "message": (
            "快速请求已入队：只读缓存数据，不刷新行情。"
            if depth == "quick"
            else (
                "全市场请求已入队；后台会按批次推进。"
                if full_universe
                else "定向请求已入队；后台只会检查和刷新指定代码。"
            )
        ),
    }
    atomic_write_json(run_dir / "status.json", status)
    atomic_write_json(
        run_dir / "manifest.json",
        {
            "run_id": run_id,
            "job_id": run_id,
            "status": "queued",
            "started_at": started.isoformat(),
            "mode": mode,
            "depth": depth,
            "asset_scope": asset_scope,
            "asset_types": list(selected_types),
            "symbols": routes,
            "full_universe": full_universe,
            "target_date": effective_target_date,
            "target_dates": target_dates,
            "config_snapshot": config_snapshot(dict(config)),
        },
    )
    worker_info = (
        _start_background_worker(run_id, config)
        if start_background_worker
        else {"started": False, "reason": "caller_will_start_worker"}
    )
    status["worker"] = worker_info
    atomic_write_json(run_dir / "status.json", status)
    return research._jsonable(
        {
            "run_id": run_id,
            "job_id": run_id,
            "status": "queued",
            "phase": "cache_check",
            "completed": progress["completed"],
            "pending": progress["pending"],
            "failed": progress["failed"],
            "coverage": progress["coverage"],
            "target_date": effective_target_date,
            "target_dates": target_dates,
            "result_path": str(run_dir),
            "current_path": str(research.ASSISTANT_RESULTS_DIR / "current.json"),
            "message": status["message"],
            "worker": worker_info,
        }
    )


def _queue_full_universe_research(**kwargs: Any) -> dict[str, Any]:
    """Compatibility wrapper for existing schedulers and tests."""
    return _queue_research(**kwargs)


def _start_background_worker(
    job_id: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if not config["fetch"].get("background_worker_enabled", True):
        return {"started": False, "reason": "disabled_by_configuration"}
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_research_worker.py"
    if not script.exists():
        return {"started": False, "reason": "worker_script_missing"}
    command = [sys.executable, str(script), "--job-id", job_id]
    # 全市场任务按波次预算分批推进并回队；定向任务是一次性请求，必须给
    # 模型缓存冷启动（provider 重建后的首次完整训练）留足时间，否则研究
    # 阶段会在波次预算处被杀，且每次重试都从零训练，永远无法发布缓存。
    wave_budget = float(config["fetch"].get("batch_time_budget_seconds", 600))
    targeted_budget = float(
        config["fetch"].get("targeted_time_budget_seconds", 3600)
    )
    try:
        job = _fetch_state(config).get_job(job_id)
        metadata = job.get("metadata", {}) if isinstance(job, dict) else {}
        if job and not bool(metadata.get("full_universe", True)):
            command += ["--time-budget", str(max(wave_budget, targeted_budget))]
    except Exception:
        pass
    try:
        creation_flags = 0
        if os.name == "nt":
            creation_flags = (
                getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                | getattr(subprocess, "DETACHED_PROCESS", 0)
            )
        child_env = {
            **os.environ,
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
        }
        process = subprocess.Popen(
            command,
            cwd=str(script.parents[1]),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creation_flags,
            env=child_env,
            close_fds=os.name != "nt",
        )
        return {"started": True, "pid": process.pid}
    except Exception as exc:
        return {
            "started": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }


def _run_refresh_command(
    command: list[str],
    *,
    cwd: Path,
    config: Mapping[str, Any],
    label: str,
    log: list[str],
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    max_attempts = int(config["fetch"]["retries"]) + 1
    for attempt in range(1, max_attempts + 1):
        try:
            completed = subprocess.run(
                command,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=int(config["fetch"]["timeout_seconds"]),
                check=False,
            )
            record = {
                "attempt": attempt,
                "returncode": completed.returncode,
                "stdout_tail": completed.stdout[-2000:],
                "stderr_tail": completed.stderr[-2000:],
            }
            summary = research._fetch_summary(completed.stdout)
            if summary is not None:
                record["summary"] = summary
            attempts.append(record)
            if completed.returncode == 0:
                if summary is None:
                    log.append(f"refresh {label}: success on attempt {attempt}")
                    return {"status": "success", "attempts": attempts}
                if (
                    summary.get("status") in {"complete", "partial"}
                    and int(summary.get("failed", 0) or 0) == 0
                ):
                    result_status = (
                        "success"
                        if summary.get("status") == "complete"
                        and not summary.get("partial_failure", False)
                        else "partial"
                    )
                    log.append(
                        f"refresh {label}: {result_status} on attempt {attempt}"
                    )
                    return {"status": result_status, "attempts": attempts}
        except Exception as exc:
            attempts.append({
                "attempt": attempt,
                "error": f"{type(exc).__name__}: {exc}",
            })
        log.append(f"refresh {label}: attempt {attempt} failed")
        if attempt < max_attempts:
            time.sleep(float(config["fetch"].get("retry_delay_seconds", 0)))
    return {"status": "failed", "attempts": attempts}


def _refresh_data(
    asset_types: Iterable[str],
    config: Mapping[str, Any],
    log: list[str],
    *,
    symbols_by_type: Mapping[str, Iterable[str]] | None = None,
) -> dict[str, Any]:
    if not config["fetch"]["enabled"] or os.environ.get("DSH_STOCK_SKIP_FETCH") == "1":
        log.append("data refresh skipped by configuration or DSH_STOCK_SKIP_FETCH=1")
        return {"attempted": False, "passed": True, "results": []}
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    results: list[dict[str, Any]] = []
    selected = set(asset_types)
    routes = (
        _normalize_symbol_routes(selected, symbols_by_type)
        if symbols_by_type is not None
        else {}
    )
    jobs: list[tuple[str, Path, list[str]]] = []
    if "stock" in selected:
        jobs.append(
            ("stock", scripts / "fetch_stock_data.py", routes.get("stock", []))
        )
    for asset_type in ("etf", "otc_fund"):
        if asset_type in selected:
            jobs.append(
                (
                    asset_type,
                    scripts / "fetch_fund_data.py",
                    ["--asset-type", asset_type, *routes.get(asset_type, [])],
                )
            )
    for label, script, extra_args in jobs:
        if not script.exists():
            results.append({"label": label, "status": "missing_script"})
            continue
        fetch_options = [
            "--batch-size",
            str(
                config["fetch"]
                .get("batch_size", {})
                .get(label, 100)
            ),
            "--time-budget",
            str(config["fetch"].get("batch_time_budget_seconds", 600)),
            "--workers",
            str(config["fetch"].get("max_workers", 3)),
            "--retries",
            str(config["fetch"].get("retries", 3)),
            "--rate-limit",
            str(config["fetch"].get("requests_per_second", 1.0)),
            "--request-timeout",
            str(config["fetch"].get("request_timeout_seconds", 30)),
            "--backoff-base",
            str(config["fetch"].get("backoff_base_seconds", 2)),
            "--backoff-max",
            str(config["fetch"].get("backoff_max_seconds", 120)),
            "--lease-seconds",
            str(config["fetch"].get("lease_seconds", 1800)),
            "--state-db",
            str(config["fetch"].get("state_db", research.DATA_DIR / "fetch_state.sqlite3")),
        ]
        result = _run_refresh_command(
            [sys.executable, str(script), *fetch_options, *extra_args],
            cwd=Path(__file__).resolve().parents[1],
            config=config,
            label=label,
            log=log,
        )
        results.append({"label": label, **result})

    stock_fetch_ok = any(
        result.get("label") == "stock" and result.get("status") == "success"
        for result in results
    )
    targeted_stock_refresh = symbols_by_type is not None and bool(routes.get("stock"))
    if "stock" in selected and stock_fetch_ok and not targeted_stock_refresh:
        provider_result = _run_refresh_command(
            [sys.executable, str(scripts / "build_qlib_data.py"), "--replace"],
            cwd=Path(__file__).resolve().parents[1],
            config=config,
            label="qlib_provider",
            log=log,
        )
        results.append({"label": "qlib_provider", **provider_result})
    elif "stock" in selected and stock_fetch_ok and targeted_stock_refresh:
        log.append("qlib provider rebuild skipped for targeted stock refresh; existing provider reused")
        results.append(
            {
                "label": "qlib_provider",
                "status": "reused",
                "reason": "targeted_stock_refresh",
            }
        )
    passed = bool(results) and all(
        result.get("status") in {"success", "reused"} for result in results
    )
    return {
        "attempted": True,
        "passed": passed,
        "results": results,
        "max_attempts": int(config["fetch"]["retries"]) + 1,
        "finished_at": research._now().isoformat(),
    }


def _write_run_files(
    run_dir: Path,
    *,
    manifest: dict[str, Any],
    data_quality: dict[str, Any],
    data_facts: list[dict[str, Any]],
    model_signals: dict[str, Any],
    rule_gates: dict[str, Any],
    decision_support: dict[str, Any],
    config: dict[str, Any],
    model_metadata: dict[str, Any],
    log_lines: list[str],
) -> dict[str, str]:
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, Any] = {
        "manifest": manifest,
        "data_quality": data_quality,
        "data_facts": data_facts,
        "model_signals": model_signals,
        "rule_gates": rule_gates,
        "decision_support": decision_support,
        "model_metadata": model_metadata,
    }
    paths: dict[str, str] = {}
    for name, value in files.items():
        path = run_dir / f"{name}.json"
        atomic_write_json(path, research._jsonable(value))
        paths[name] = str(path)
    config_path = run_dir / "run_config.yaml"
    atomic_write_text(config_path, yaml.safe_dump(config_snapshot(config), allow_unicode=True, sort_keys=False))
    paths["run_config"] = str(config_path)
    summary_lines = [
        "# 股票/基金私人研究助手运行摘要",
        "",
        f"- 运行 ID: `{manifest['run_id']}`",
        f"- 状态: `{manifest['status']}`",
        f"- 数据截止日期: `{data_quality.get('data_end_date') or 'unknown'}`",
        f"- 资产数量: `{data_quality.get('asset_count', 0)}`",
        f"- 个股模型门槛: `{'通过' if rule_gates.get('stock', {}).get('passed') else '未通过'}`",
        "",
        "## 使用规则",
        "",
        "- 个股模型门槛未通过时，不生成个股买入目标权重。",
        "- 目标权重基于假设资金/配置，不连接券商，不自动下单。",
        "- 场外基金使用净值序列，不输出盘中成交价或精确股数。",
        "",
        "## 本次建议",
        "",
        json.dumps(research._jsonable(decision_support), ensure_ascii=False, indent=2),
        "",
        "## 日志",
        "",
        *[f"- {line}" for line in log_lines],
    ]
    summary_path = run_dir / "summary.md"
    atomic_write_text(summary_path, "\n".join(summary_lines) + "\n")
    paths["summary"] = str(summary_path)
    log_path = logs_dir / "research.log"
    atomic_write_text(log_path, "\n".join(log_lines) + "\n")
    paths["logs"] = str(log_path)
    return paths


def _touch_status(status_path: Path, phase: str) -> None:
    """Refresh the run heartbeat so the stale-run sweep never reaps a live run."""
    record = read_json(status_path, {})
    if not isinstance(record, dict):
        record = {}
    record["status"] = "running"
    record["phase"] = phase
    record["updated_at"] = research._now().isoformat()
    atomic_write_json(status_path, record)


def _sweep_stale_runs(
    config: Mapping[str, Any],
    log_lines: list[str],
    *,
    runs_dir: Path | None = None,
) -> list[str]:
    """Fail orphaned runs whose status stayed ``running`` without a heartbeat.

    A killed process never reaches the success/failed write, so its status
    file would report ``running`` forever.  The heartbeat touch keeps the
    status file mtime fresh while a run is alive; anything silent for longer
    than ``runtime.lock_stale_seconds`` is treated as dead and marked failed
    without touching ``current.json``.
    """
    threshold = int(config["runtime"]["lock_stale_seconds"])
    current = research._now().timestamp()
    swept: list[str] = []
    for status_path in sorted((runs_dir or research.RUNS_DIR).rglob("status.json")):
        record = read_json(status_path, {})
        if not isinstance(record, dict) or record.get("status") != "running":
            continue
        try:
            silent_seconds = current - status_path.stat().st_mtime
        except OSError:
            continue
        if silent_seconds < threshold:
            continue
        run_id = str(record.get("run_id") or status_path.parent.name)
        atomic_write_json(
            status_path,
            {
                **record,
                "status": "failed",
                "updated_at": research._now().isoformat(),
                "error": (
                    f"stale run swept: no heartbeat for {int(silent_seconds)}s "
                    f"(threshold {threshold}s)"
                ),
                "failure_reasons": ["stale_run_heartbeat_timeout"],
            },
        )
        swept.append(run_id)
    if swept:
        log_lines.append(f"swept_stale_runs={swept}")
    return swept


def _process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        from ctypes import wintypes

        process_query_limited_information = 0x1000
        still_active = 259
        access_denied = 5
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = (
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        )
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.DWORD),
        )
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL

        handle = kernel32.OpenProcess(
            process_query_limited_information,
            False,
            pid,
        )
        if not handle:
            return ctypes.get_last_error() == access_denied
        try:
            exit_code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return True
            return exit_code.value == still_active
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        return exc.errno != ESRCH
    return True


def _fetch_summary(stdout: str) -> dict[str, Any] | None:
    marker = "ASSISTANT_FETCH_SUMMARY="
    for line in reversed((stdout or "").splitlines()):
        if not line.startswith(marker):
            continue
        try:
            value = json.loads(line[len(marker):])
        except json.JSONDecodeError:
            return {"parse_error": "invalid fetch summary JSON"}
        return value if isinstance(value, dict) else {"parse_error": "fetch summary is not an object"}
    return None
