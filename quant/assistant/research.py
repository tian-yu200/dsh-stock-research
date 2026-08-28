"""Fixed-path research orchestration for stocks, ETFs and OTC funds.

The service intentionally keeps the three asset paths separate.  A model
ranking is only one input to decision support; it is never an automatic buy
instruction, and failed stock model gates always produce cash/observe output.
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import time
import ctypes
from errno import ESRCH
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
from uuid import uuid4

import numpy as np
import pandas as pd
import yaml

from .config import config_snapshot, load_config
from .notifications import acknowledge, add_notification, list_records
from .paths import (
    ASSISTANT_RESULTS_DIR,
    DATA_DIR,
    LOCK_PATH,
    QLIB_PROVIDER_DIR,
    RUNS_DIR,
    atomic_write_json,
    atomic_write_text,
    ensure_directories,
    fixed_path,
    read_json,
    run_directory,
    validate_notification_id,
    validate_run_id,
    validate_symbol,
)
from .risk import cash_only_decision, enforce_target_weights, model_gate_checks

try:
    from scripts.market_data import normalize_daily_bars
    from scripts.fetch_state import FetchState
    from scripts.fetch_targets import asset_target_dates
    from scripts.universe import (
        load_asset_metadata,
        load_universe_manifest,
        manifest_names,
        manifest_symbols,
    )
except ImportError:
    scripts_root = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts_root) not in sys.path:
        sys.path.insert(0, str(scripts_root))
    from market_data import normalize_daily_bars
    from fetch_state import FetchState
    from fetch_targets import asset_target_dates
    from universe import (
        load_asset_metadata,
        load_universe_manifest,
        manifest_names,
        manifest_symbols,
    )


ASSET_TYPES = ("stock", "etf", "otc_fund")
ASSET_SCOPE_MAP = {
    "stocks": ("stock",),
    "stock": ("stock",),
    "etfs": ("etf",),
    "etf": ("etf",),
    "funds": ("otc_fund",),
    "fund": ("otc_fund",),
    "otc_fund": ("otc_fund",),
    "otc_funds": ("otc_fund",),
    "all": ASSET_TYPES,
}
MODE_NAMES = {"daily", "monthly", "on_demand"}
SYMBOL_FILE_RE = re.compile(r"^(?:stock|etf|fund)_?([0-9]{6})$", re.IGNORECASE)
UNIVERSE_PATH = DATA_DIR / "universe.json"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_date(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).date().isoformat()


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


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


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if pd.isna(value) if not isinstance(value, (dict, list, tuple)) else False:
        return None
    return value


# --- 阶段 1 兼容层（docs/重构计划.md §4 纪律 2）：数据加载与质量函数组已逐字
# 搬迁至 data_layer.py，此处把原名重新挂回 research 命名空间，旧调用方与测试
# （含 patch.object(research, ...)）零改动；阶段 1 结束时评估是否移除。
from .data_layer import (
    _aggregate_quality,
    _adjustment_check,
    _authoritative_manifest,
    _data_file,
    _liquidity_check,
    _load_asset,
    _parse_fund_nav,
    _partition_symbols,
    _scan_assets,
    _universe,
)


# --- 阶段 1 兼容层：技术面信号函数组已逐字搬迁至 signals.py（同名重新挂回）。
from .signals import (
    _dated_close_series,
    _drawdown_recovery,
    _liquidity_metrics,
    _market_metrics,
    _peer_comparison,
    _relative_returns,
    _technical_asset_analysis,
)


# --- 阶段 1 兼容层：Qlib 训练与指标函数组已逐字搬迁至 qlib_model.py（同名重新挂回）。
from .qlib_model import (
    _fit_qlib_model_and_close_recorder,
    _forward_group_metrics,
    _performance,
    _prediction_metrics,
    _run_qlib_stock_model,
)


# --- 阶段 1 兼容层：排队/worker/刷新函数组已逐字搬迁至 queueing.py（同名重新挂回）。
from .queueing import (
    _cached_end_date,
    _fetch_state,
    _normalize_symbol_routes,
    _queue_full_universe_research,
    _queue_research,
    _refresh_data,
    _run_refresh_command,
    _start_background_worker,
    _target_dates_for_job,
)


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
        atomic_write_json(path, _jsonable(value))
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
        json.dumps(_jsonable(decision_support), ensure_ascii=False, indent=2),
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
    record["updated_at"] = _now().isoformat()
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
    current = _now().timestamp()
    swept: list[str] = []
    for status_path in sorted((runs_dir or RUNS_DIR).rglob("status.json")):
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
                "updated_at": _now().isoformat(),
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


def _lock_record() -> dict[str, Any]:
    try:
        value = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        value = {}
    return value if isinstance(value, dict) else {}


def _acquire_lock(run_id: str, stale_seconds: int):
    ensure_directories()
    for _ in range(2):
        try:
            descriptor = os.open(str(LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            existing = _lock_record()
            try:
                age_seconds = max(0.0, time.time() - LOCK_PATH.stat().st_mtime)
            except FileNotFoundError:
                continue
            pid = int(existing.get("pid", 0) or 0)
            stale = age_seconds > max(0, int(stale_seconds)) or not _process_exists(pid)
            if stale:
                try:
                    LOCK_PATH.unlink()
                except FileNotFoundError:
                    pass
                continue
            raise RuntimeError(
                "another research run is active: "
                f"run_id={existing.get('run_id', 'unknown')}, pid={pid}"
            ) from exc
        record = {
            "run_id": run_id,
            "pid": os.getpid(),
            "created_at": _now().isoformat(),
        }
        try:
            os.write(descriptor, (json.dumps(record, sort_keys=True) + "\n").encode("utf-8"))
            os.fsync(descriptor)
        except Exception:
            os.close(descriptor)
            LOCK_PATH.unlink(missing_ok=True)
            raise
        return descriptor
    raise RuntimeError("could not acquire research lock after removing a stale lock")


def _release_lock(descriptor: int, run_id: str | None = None) -> None:
    try:
        os.close(descriptor)
    finally:
        existing = _lock_record()
        if run_id is None or existing.get("run_id") == run_id:
            LOCK_PATH.unlink(missing_ok=True)


def _run_id() -> str:
    return f"{_now():%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"


def _asset_scope(value: str) -> tuple[str, ...]:
    scope = str(value or "all").strip().lower()
    if scope not in ASSET_SCOPE_MAP:
        raise ValueError(f"asset_scope must be one of: {', '.join(sorted(ASSET_SCOPE_MAP))}")
    return ASSET_SCOPE_MAP[scope]


def run_research(
    mode: str = "daily",
    asset_scope: str = "all",
    symbols: list[str] | None = None,
    *,
    _force_sync: bool = False,
    _target_date: str | None = None,
    _start_background_worker: bool | None = None,
    _symbols_by_type: Mapping[str, Iterable[str]] | None = None,
) -> dict[str, Any]:
    mode = str(mode or "daily").strip().lower()
    if mode not in MODE_NAMES:
        raise ValueError(f"mode must be one of: {', '.join(sorted(MODE_NAMES))}")
    asset_types = _asset_scope(asset_scope)
    if symbols is not None and not symbols and _symbols_by_type is None:
        raise ValueError("symbols must be omitted for full-universe research or contain at least one code")
    requested_symbols = [validate_symbol(value) for value in (symbols or [])]
    explicit_routes = (
        _normalize_symbol_routes(asset_types, _symbols_by_type)
        if _symbols_by_type is not None
        else (
            _partition_symbols(asset_types, requested_symbols)
            if requested_symbols
            else {}
        )
    )
    config = load_config()
    if (
        not _force_sync
        and config["fetch"]["enabled"]
        and os.environ.get("DSH_STOCK_SKIP_FETCH") != "1"
    ):
        return _queue_research(
            mode=mode,
            asset_scope=asset_scope,
            asset_types=asset_types,
            config=config,
            symbols_by_type=explicit_routes if requested_symbols or _symbols_by_type is not None else None,
            target_date=_target_date,
            start_background_worker=(
                True if _start_background_worker is None else _start_background_worker
            ),
        )
    run_id = _run_id()
    started = _now()
    run_dir = RUNS_DIR / started.strftime("%Y") / started.strftime("%m") / started.strftime("%d") / run_id
    lock_descriptor: int | None = None
    log_lines = [f"started {started.isoformat()}", f"mode={mode}", f"asset_types={asset_types}"]
    status_path = run_dir / "status.json"
    _sweep_stale_runs(config, log_lines)
    refresh: dict[str, Any] = {"attempted": False, "passed": True, "results": []}
    all_facts: list[dict[str, Any]] = []
    technical_signals: dict[str, Any] = {}
    data_quality: dict[str, Any] = {
        "passed": False,
        "data_end_date": None,
        "asset_count": 0,
        "refresh": refresh,
    }
    model_metadata: dict[str, Any] = {
        "model_version": "assistant-1",
        "qlib_provider": str(QLIB_PROVIDER_DIR),
        "requested_mode": mode,
    }
    rule_gates: dict[str, Any] = {}
    decision_support: dict[str, Any] = cash_only_decision(
        "研究尚未完成。",
        capital=float(config["research"]["hypothetical_capital"]),
    )
    decision_support["asset_paths"] = {}
    result_paths: dict[str, str] = {}
    try:
        lock_descriptor = _acquire_lock(
            run_id,
            int(config["runtime"]["lock_stale_seconds"]),
        )
        run_dir.mkdir(parents=True, exist_ok=False)
        atomic_write_json(
            status_path,
            {
                "run_id": run_id,
                "status": "running",
                "started_at": started.isoformat(),
                "updated_at": started.isoformat(),
            },
        )
        symbol_routes = explicit_routes or {asset_type: [] for asset_type in asset_types}
        analyzed_types = tuple(
            asset_type
            for asset_type in asset_types
            if not explicit_routes or symbol_routes.get(asset_type)
        )
        if explicit_routes and not analyzed_types:
            raise ValueError("no requested symbols matched the selected asset scope")
        log_lines.append(f"analyzed_types={analyzed_types}")
        if explicit_routes:
            log_lines.append(f"symbol_routes={symbol_routes}")

        if mode in set(config["research"]["refresh_modes"]):
            refresh = _refresh_data(
                analyzed_types,
                config,
                log_lines,
                symbols_by_type=symbol_routes if explicit_routes else None,
            )
        _touch_status(status_path, "refreshed")
        all_frames: dict[str, dict[str, pd.DataFrame]] = {}
        for asset_type in analyzed_types:
            scope_symbols = symbol_routes[asset_type] if explicit_routes else None
            frames, facts = _scan_assets(asset_type, scope_symbols)
            all_frames[asset_type] = frames
            all_facts.extend(facts)
        data_quality = _aggregate_quality(all_facts, config)
        data_quality["refresh"] = refresh
        data_quality["passed"] = bool(data_quality["passed"] and refresh.get("passed", False))
        universe_manifest = load_universe_manifest(UNIVERSE_PATH)
        data_quality["universe"] = {
            "manifest_path": str(UNIVERSE_PATH),
            "source": universe_manifest.get("source") if universe_manifest else None,
            "generated_at": universe_manifest.get("generated_at") if universe_manifest else None,
            "complete": bool(universe_manifest and universe_manifest.get("complete")),
            "counts": (
                universe_manifest.get("counts", {})
                if universe_manifest
                else {
                    asset_type: sum(
                        fact.get("asset_type") == asset_type
                        for fact in all_facts
                    )
                    for asset_type in ASSET_TYPES
                }
            ),
            "authoritative_asset_types": [
                asset_type
                for asset_type in analyzed_types
                if _authoritative_manifest(universe_manifest, asset_type)
            ],
        }
        refresh_summaries = {
            item.get("label"): attempt.get("summary")
            for item in refresh.get("results", [])
            for attempt in reversed(item.get("attempts", []))
            if attempt.get("summary")
        }
        for fact in all_facts:
            label = "stock" if fact.get("asset_type") == "stock" else "funds"
            summary = refresh_summaries.get(label)
            if summary:
                fact.update(
                    {
                        "akshare_interfaces": summary.get("interfaces", []),
                        "fetch_time": summary.get("finished_at") or refresh.get("finished_at"),
                        "fetch_retries": max(
                            (
                                attempt.get("attempt", 1)
                                for item in refresh.get("results", [])
                                if item.get("label") == label
                                for attempt in item.get("attempts", [])
                            ),
                            default=1,
                        )
                        - 1,
                        "partial_failure": bool(summary.get("partial_failure", False)),
                        "field_anomalies": summary.get("field_anomalies", []),
                    }
                )
        if not all_facts:
            raise ValueError("no asset data found in fixed data directories")

        _touch_status(status_path, "scanned")
        benchmark_series = None
        if analyzed_types:
            try:
                benchmark_frame, _ = _load_asset("etf", "510300")
                benchmark_series = pd.Series(
                    benchmark_frame["close"].to_numpy(),
                    index=pd.to_datetime(benchmark_frame["date"]),
                ).sort_index()
            except Exception as exc:
                log_lines.append(f"benchmark unavailable: {type(exc).__name__}")

        for asset_type in analyzed_types:
            technical_signals[asset_type] = _technical_asset_analysis(
                asset_type,
                all_frames[asset_type],
                [item for item in all_facts if item["asset_type"] == asset_type],
                benchmark_series,
            )

        _touch_status(status_path, "signals")
        stock_model = {
            "model_available": False,
            "model_name": "Qlib Alpha158 + LightGBM",
            "error": "stock path not requested",
            "mean_rank_ic": float("nan"),
            "test_rank_ic": float("nan"),
            "rank_ic_ir": float("nan"),
            "top_bottom_excess_return": float("nan"),
            "net_annualized_return": float("nan"),
            "test_max_drawdown": float("nan"),
            "max_drawdown": float("nan"),
            "leakage_check": {"passed": False, "reason": "stock path not requested"},
        }
        if "stock" in analyzed_types:
            stock_facts = [item for item in all_facts if item["asset_type"] == "stock"]
            stock_quality = _aggregate_quality(stock_facts, config)
            stock_refresh_ok = all(
                result.get("status") == "success"
                for result in refresh.get("results", [])
                if result.get("label") in {"stock", "qlib_provider"}
            )
            if stock_quality["passed"] and stock_refresh_ok:
                stock_model = _run_qlib_stock_model(config)
            else:
                stock_model["error"] = "stock model skipped because stock data quality or refresh failed"
                stock_model["leakage_check"] = {
                    "passed": False,
                    "reason": "stock data quality or refresh failed",
                }
            model_metadata["stock_model"] = stock_model
            if stock_model.get("model_available"):
                score_map = stock_model.get("scores", {})
                for signal in technical_signals["stock"]["signals"]:
                    if signal["symbol"] in score_map:
                        signal["qlib_score"] = score_map[signal["symbol"]]
                        signal["model_rank"] = 1 + sorted(score_map, key=score_map.get, reverse=True).index(signal["symbol"])
                technical_signals["stock"]["signals"].sort(
                    key=lambda item: item.get("qlib_score", -math.inf), reverse=True
                )

        _touch_status(status_path, "gates")
        decision_support = {
            "allocation_basis": "基于假设资金/配置的目标权重",
            "hypothetical_capital": float(config["research"]["hypothetical_capital"]),
            "broker_connected": False,
            "auto_order": False,
            "asset_paths": {},
            "target_weights": {},
            "cash_weight": 1.0,
        }
        allocation_budget = min(
            float(config["risk"]["max_daily_turnover"]),
            1.0 - float(config["risk"]["min_cash_weight"]),
        ) / max(len(analyzed_types), 1)

        if "stock" in analyzed_types:
            stock_facts = [item for item in all_facts if item["asset_type"] == "stock"]
            stock_quality = _aggregate_quality(stock_facts, config)
            stock_metrics = dict(stock_model)
            stock_metrics["model_available"] = bool(stock_model.get("model_available"))
            checks = model_gate_checks(stock_metrics, config["gate"])
            checks["data_quality_passed"] = bool(stock_quality["passed"])
            checks["freshness_passed"] = not bool(stock_quality["stale"])
            checks["refresh_passed"] = all(
                result.get("status") == "success"
                for result in refresh.get("results", [])
                if result.get("label") in {"stock", "qlib_provider"}
            )
            checks["leakage_check_passed"] = bool(stock_model.get("leakage_check", {}).get("passed", False))
            gate_passed = all(checks.values())
            stock_requested = {}
            if gate_passed:
                ranked = technical_signals["stock"]["signals"][: int(config["research"]["stock_topk"])]
                if ranked:
                    per_symbol = min(
                        allocation_budget / len(ranked),
                        float(config["risk"]["max_single_weight"]),
                    )
                    raw_weights = {item["symbol"]: per_symbol for item in ranked}
                    stock_requested = raw_weights
            asset_types_for_weights = {symbol: "stock" for symbol in stock_requested}
            risk_result = enforce_target_weights(
                stock_requested,
                asset_types_for_weights,
                config["risk"],
            )
            checks["risk_constraints_passed"] = risk_result["passed"]
            gate_passed = gate_passed and risk_result["passed"]
            if gate_passed:
                decision_support["target_weights"].update(risk_result["target_weights"])
                decision_support["cash_weight"] = risk_result["cash_weight"]
                stock_decision = {
                    "status": "candidate",
                    "action": "观察/候选",
                    "target_weights": risk_result["target_weights"],
                    "cash_weight": risk_result["cash_weight"],
                    "reason": "Qlib 模型和风险门槛均通过；仍需人工确认基本面与交易条件。",
                }
            else:
                stock_decision = cash_only_decision(
                    "个股模型或风险门槛未通过，不生成个股买入目标权重。",
                    capital=float(config["research"]["hypothetical_capital"]),
                )
            rule_gates["stock"] = {
                "passed": gate_passed,
                "checks": checks,
                "model_metrics": stock_metrics,
                "risk": risk_result,
            }
            decision_support["asset_paths"]["stock"] = stock_decision

        for asset_type in ("etf", "otc_fund"):
            if asset_type not in analyzed_types:
                continue
            analysis = technical_signals[asset_type]
            candidates = [item for item in analysis["signals"] if (item.get("trend_score") or -math.inf) > 0]
            candidates = candidates[:2]
            category = "etf" if asset_type == "etf" else "otc_fund"
            per_symbol = min(
                allocation_budget / len(candidates),
                float(config["risk"]["max_single_weight"]),
            ) if candidates else 0.0
            requested = {item["symbol"]: per_symbol for item in candidates}
            risk_result = enforce_target_weights(
                requested,
                {symbol: category for symbol in requested},
                config["risk"],
            )
            asset_facts = [item for item in all_facts if item["asset_type"] == asset_type]
            asset_quality = _aggregate_quality(asset_facts, config)
            checks = {
                "data_quality_passed": bool(asset_quality["passed"]),
                "freshness_passed": not bool(asset_quality["stale"]),
                "refresh_passed": all(
                    result.get("status") == "success"
                    for result in refresh.get("results", [])
                    if result.get("label") == "funds"
                ),
                "risk_constraints_passed": risk_result["passed"],
            }
            passed = all(checks.values())
            rule_gates[asset_type] = {"passed": passed, "checks": checks, "risk": risk_result}
            if passed:
                decision = {
                    "status": "candidate",
                    "action": "配置候选",
                    "target_weights": risk_result["target_weights"],
                    "cash_weight": risk_result["cash_weight"],
                    "reason": "技术趋势、相对表现和风险约束通过；需结合费用、流动性和个人期限确认。",
                }
                decision_support["target_weights"].update(risk_result["target_weights"])
            else:
                decision = cash_only_decision(
                    f"{category} 数据或风险门槛未通过，保持观察。",
                    capital=float(config["research"]["hypothetical_capital"]),
                )
            if asset_type == "otc_fund":
                decision["no_intraday_price_or_exact_shares"] = True
            decision_support["asset_paths"][asset_type] = decision

        current = read_json(ASSISTANT_RESULTS_DIR / "current.json", {})
        previous_weights = (
            current.get("decision_support", {}).get("target_weights", {})
            if isinstance(current, dict)
            else {}
        )
        combined_types = {
            symbol: asset_type
            for asset_type, decision in decision_support["asset_paths"].items()
            for symbol in decision.get("target_weights", {})
        }
        portfolio_risk = enforce_target_weights(
            decision_support["target_weights"],
            combined_types,
            config["risk"],
            previous=previous_weights,
        )
        rule_gates["portfolio"] = portfolio_risk
        decision_support["portfolio_gate"] = portfolio_risk
        if not portfolio_risk["passed"]:
            decision_support["target_weights"] = {}
            decision_support["cash_weight"] = 1.0
            for decision in decision_support["asset_paths"].values():
                decision["target_weights"] = {}
                decision["cash_weight"] = 1.0
                decision["status"] = "cash_or_observe"
                decision["action"] = "现金/观察"
                decision["reason"] = "组合层风险约束未通过，不发布可执行目标权重。"
        else:
            decision_support["target_weights"] = portfolio_risk["target_weights"]
            decision_support["cash_weight"] = portfolio_risk["cash_weight"]

        failure_reasons: list[str] = []
        if not data_quality["passed"]:
            failure_reasons.append("data_quality_or_refresh_failed")
        if "stock" in analyzed_types and not rule_gates.get("stock", {}).get("passed", False):
            failure_reasons.append("stock_model_or_risk_gate_failed")
        if not portfolio_risk["passed"]:
            failure_reasons.append("portfolio_risk_gate_failed")
        for asset_type in ("etf", "otc_fund"):
            if asset_type in analyzed_types and not rule_gates.get(asset_type, {}).get("passed", False):
                failure_reasons.append(f"{asset_type}_quality_or_risk_gate_failed")
        run_status = "failed" if failure_reasons else "success"
        if failure_reasons:
            decision_support["publication_blocked"] = True
            decision_support["failure_reasons"] = failure_reasons
            log_lines.append(f"publication blocked: {failure_reasons}")

        manifest = {
            "run_id": run_id,
            "status": run_status,
            "started_at": started.isoformat(),
            "finished_at": _now().isoformat(),
            "mode": mode,
            "asset_scope": asset_scope,
            "asset_types": asset_types,
            "analyzed_types": analyzed_types,
            "requested_symbols": requested_symbols,
            "data_sources": (
                ["AkShare cached CSV", "Qlib provider"]
                if "stock" in analyzed_types
                else ["AkShare cached CSV"]
            ),
            "data_end_date": data_quality.get("data_end_date"),
            "retry_count": int(config["fetch"]["retries"]),
            "failure_reasons": failure_reasons,
            "config_snapshot": config_snapshot(config),
        }
        result_paths = _write_run_files(
            run_dir,
            manifest=manifest,
            data_quality=data_quality,
            data_facts=all_facts,
            model_signals=technical_signals,
            rule_gates=rule_gates,
            decision_support=decision_support,
            config=config,
            model_metadata=model_metadata,
            log_lines=log_lines,
        )
        manifest["paths"] = result_paths
        atomic_write_json(run_dir / "manifest.json", _jsonable(manifest))
        status = {
            "run_id": run_id,
            "status": run_status,
            "finished_at": manifest["finished_at"],
            "data_end_date": data_quality.get("data_end_date"),
            "result_path": str(run_dir),
            "failure_reasons": failure_reasons,
        }
        atomic_write_json(status_path, status)
        if run_status == "success":
            current = {
                "run_id": run_id,
                "status": "success",
                "updated_at": manifest["finished_at"],
                "data_end_date": data_quality.get("data_end_date"),
                "result_path": str(run_dir),
                "paths": result_paths,
                "decision_support": _jsonable(decision_support),
                "rule_gates": _jsonable(rule_gates),
            }
            atomic_write_json(ASSISTANT_RESULTS_DIR / "current.json", current)
            notification = add_notification(
                "research_success",
                "股票/基金研究完成",
                f"运行 {run_id} 已完成，数据截止 {data_quality.get('data_end_date') or 'unknown'}。",
                run_id=run_id,
                paths=result_paths,
            )
        else:
            notification = add_notification(
                "research_failure",
                "股票/基金研究未通过发布门槛",
                f"运行 {run_id} 未通过：{', '.join(failure_reasons)}。上一份 current.json 未覆盖。",
                run_id=run_id,
                paths=result_paths,
            )
        return _jsonable(
            {
                "run_id": run_id,
                "status": run_status,
                "failure_reasons": failure_reasons,
                "data_end_date": data_quality.get("data_end_date"),
                "result_path": str(run_dir),
                "current_path": str(ASSISTANT_RESULTS_DIR / "current.json"),
                "notification_id": notification["notification_id"],
                "data_facts": all_facts,
                "model_signals": technical_signals,
                "rule_gates": rule_gates,
                "decision_support": decision_support,
            }
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        log_lines.append(f"failed: {error}")
        manifest = {
            "run_id": run_id,
            "status": "failed",
            "started_at": started.isoformat(),
            "finished_at": _now().isoformat(),
            "mode": mode,
            "asset_scope": asset_scope,
            "error": error,
            "config_snapshot": config_snapshot(config),
        }
        if run_dir.exists():
            data_quality["passed"] = False
            data_quality["error"] = error
            decision_support["publication_blocked"] = True
            decision_support["failure_reasons"] = [error]
            result_paths = _write_run_files(
                run_dir,
                manifest=manifest,
                data_quality=data_quality,
                data_facts=all_facts,
                model_signals=technical_signals,
                rule_gates=rule_gates,
                decision_support=decision_support,
                config=config,
                model_metadata=model_metadata,
                log_lines=log_lines,
            )
            manifest["paths"] = result_paths
            atomic_write_json(run_dir / "manifest.json", manifest)
            atomic_write_json(status_path, manifest)
        notification = add_notification(
            "research_failure",
            "股票/基金研究失败",
            f"运行 {run_id} 失败：{error}。上一份 current.json 未覆盖。",
            run_id=run_id,
            paths=result_paths,
        )
        return {
            "run_id": run_id,
            "status": "failed",
            "error": error,
            "result_path": str(run_dir),
            "current_path": str(ASSISTANT_RESULTS_DIR / "current.json"),
            "notification_id": notification["notification_id"],
        }
    finally:
        if lock_descriptor is not None:
            _release_lock(lock_descriptor, run_id)


# --- 阶段 1 兼容层：对外只读 API 函数组已逐字搬迁至 read_api.py（同名重新挂回）。
from .read_api import (
    _latest_run_id,
    ack_notification,
    get_current_snapshot,
    get_run_status,
    list_notifications,
)
