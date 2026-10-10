"""Fixed-path research orchestration for stocks, ETFs and OTC funds.

The service intentionally keeps the three asset paths separate.  A model
ranking is only one input to decision support; it is never an automatic buy
instruction, and failed stock model gates always produce cash/observe output.
"""
from __future__ import annotations

import gc
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
from uuid import uuid4

import numpy as np
import pandas as pd

from .config import config_snapshot, load_config
from .cross_border import CROSS_BORDER_KEYWORDS, is_cross_border
from .notifications import add_notification
from .regime_detector import RegimeAwareModelManager, get_benchmark_returns
from .paths import (
    ASSISTANT_RESULTS_DIR,
    DATA_DIR,
    LOCK_PATH,
    QLIB_PROVIDER_DIR,
    RUNS_DIR,
    atomic_write_json,
    ensure_directories,
    read_json,
    validate_run_id,
    validate_symbol,
)
from .portfolio import allocation_budget, build_target_weights
from .risk import cash_only_decision, enforce_target_weights, model_gate_checks

try:
    from scripts.universe import load_universe_manifest
except ImportError:
    scripts_root = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts_root) not in sys.path:
        sys.path.insert(0, str(scripts_root))
    from universe import load_universe_manifest


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
DEPTH_NAMES = {"quick", "standard", "full"}
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


def _offline_stability_summary(report_path: Path | str | None = None) -> dict[str, Any]:
    """只读关联离线稳定性报告到发布快照（诊断用途，永不参与门槛）。

    读取 ``results/assistant/stability_report.json``（独立脚本
    ``quant/scripts/stability_check.py`` 的产物，不进日常管线），只摘录
    有界摘要字段。任何异常（缺失/损坏/形态不符）都 fail-open 返回
    ``{"available": False, ...}``，绝不抛异常、绝不触发额外训练。

    禁忌：返回值不得喂给 ``risk.model_gate_checks`` 或任何发布判据；
    稳定性检查接入阻断需另行决策（见门槛语义不变量 §6）。
    """
    path = Path(report_path) if report_path is not None else (
        ASSISTANT_RESULTS_DIR / "stability_report.json"
    )
    try:
        # 哨兵 None 区分"缺失/损坏"与"存在但形态不符"（read_json 缺失时回 default）。
        report = read_json(path, None)
    except Exception:
        return {"available": False, "reason": "unreadable report file"}
    if report is None:
        return {"available": False, "reason": "unreadable report file"}
    if not isinstance(report, dict):
        return {"available": False, "reason": "report is not a mapping"}
    stability = report.get("stability")
    if not isinstance(stability, dict):
        return {"available": False, "reason": "missing stability section"}
    stable = stability.get("stable")
    dataset = report.get("dataset")
    fit_window = None
    if isinstance(dataset, dict):
        fit_window = [dataset.get("fit_start_time"), dataset.get("fit_end_time")]
    return {
        "available": True,
        "status": report.get("status"),
        "segment": report.get("segment"),
        "stable": stable if isinstance(stable, bool) or stable is None else None,
        "ic_mean": _number(stability.get("ic_mean")),
        "ic_std": _number(stability.get("ic_std")),
        "threshold": _number(stability.get("threshold")),
        "windows_used": stability.get("windows_used"),
        "windows_requested": stability.get("windows_requested"),
        "fit_window": fit_window,
        "report_path": str(path),
    }


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
    _fetch_summary,
    _fetch_state,
    _normalize_symbol_routes,
    _queue_full_universe_research,
    _queue_research,
    _refresh_data,
    _run_refresh_command,
    _start_background_worker,
    _process_exists,
    _sweep_stale_runs,
    _target_dates_for_job,
    _touch_status,
    _write_run_files,
)


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


def _freshness_passed(
    quality: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    full_universe: bool = True,
) -> bool:
    """Apply stale-data tolerance only to full-universe research."""
    if not quality.get("stale"):
        return True
    if not full_universe:
        return False
    asset_count = int(quality.get("asset_count") or 0)
    stale_count = len(quality.get("stale_assets") or [])
    if asset_count <= 0:
        return False
    tolerance = max(
        0.0,
        1.0 - float(config.get("fetch", {}).get("min_universe_coverage", 0.95)),
    )
    return (stale_count / asset_count) <= tolerance


def _quality_for_depth(
    quality: Mapping[str, Any],
    facts: list[dict[str, Any]],
    config: Mapping[str, Any],
    *,
    depth: str,
) -> dict[str, Any]:
    """按研究深度调整质量判定；quick 只读缓存，允许陈旧但仍拦截质量失败。

    quick 容忍陈旧是把 freshness 判定整体放行，因此这里直接把 stale 置为
    False；stale_assets 保留仅供结果展示。质量失败的标的仍会压低 coverage。
    """
    if depth != "quick":
        return dict(quality)
    tolerated = dict(quality)
    passing = [item for item in facts if item.get("quality_status") == "pass"]
    minimum = float(config.get("fetch", {}).get("min_universe_coverage", 0.95))
    coverage = len(passing) / len(facts) if facts else 0.0
    tolerated.update(
        {
            "stale": False,
            "passed": bool(passing) and coverage >= minimum,
            "coverage_ratio": coverage,
            "eligible_asset_count": len(passing),
            "eligible_assets": [
                f"{item.get('asset_type', 'unknown')}:{item.get('symbol', 'unknown')}"
                for item in passing
            ],
            "all_sources_failed": not bool(facts) or not bool(passing),
            "stale_allowed_by_depth": True,
        }
    )
    return tolerated


_CROSS_BORDER_NAME_KEYWORDS = CROSS_BORDER_KEYWORDS  # centralized in assistant.cross_border


def _pool_quality_filter(
    signals: list[dict[str, Any]],
    facts_by_symbol: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
    as_of_date: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """宁缺毋滥：在生成买入权重前剔除不合格候选（打分后、取 top-k 前）。

    剔除四类：ST/*ST/退市整理（名称标记）、上市不满 min_listing_age_calendar_days
    （约 60 个交易日）的次新、20 日均成交额低于 min_avg_amount_20d 的低流动性标的、
    以及名称含境外市场关键词的跨境标的（QDII/港股通等，用户规则：不配美股港股）。
    缺数据的候选按合格处理（fail-open）：数据层对所有股票始终产出
    name/start_date/liquidity，缺项只出现在测试夹具与极端场景。
    返回 (合格信号, 剔除明细)，剔除明细保持原始排名顺序（前列即最可惜的落榜者）。
    """
    research_cfg = config.get("research") or {}
    exclude_st = bool(research_cfg.get("pool_exclude_st", False))
    exclude_bj = bool(research_cfg.get("pool_exclude_bj", False))
    min_age_days = int(research_cfg.get("min_listing_age_calendar_days", 90) or 0)
    min_avg_amount = float(research_cfg.get("min_avg_amount_20d", 0.0) or 0.0)
    exclude_cross_border = bool(research_cfg.get("pool_exclude_cross_border", False))
    as_of = pd.Timestamp(as_of_date) if as_of_date else None
    eligible: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for signal in signals:
        symbol = str(signal.get("symbol") or "")
        fact = facts_by_symbol.get(symbol) or {}
        name = str(fact.get("name") or "")
        reason = None
        # 2026-09-10 回撤根治：北交所（92/43/83/87/4 开头）±30% 板
        # 在打分后、取 top-k 前剔除；仅对股票生效（ETF/场外不动），
        # 缺号/缺类型 fail-open。
        if exclude_bj and reason is None and symbol:
            fact_type = (fact.get("asset_type") or "") if isinstance(fact, dict) else ""
            if fact_type in ("", "stock"):
                prefix = str(symbol).strip().upper()
                if prefix.startswith(("92", "43", "83", "87", "4")):
                    reason = "bj_board"
        if exclude_st and name:
            upper = name.upper()
            if "ST" in upper or "退" in name:
                reason = "st_or_delisting"
        if reason is None and exclude_cross_border and name:
            keywords = research_cfg.get("cross_border_keywords") or None
            fact_type = fact.get("asset_type")
            if is_cross_border(name, asset_type=fact_type, keywords=keywords):
                reason = "cross_border"
        if reason is None and min_age_days > 0 and as_of is not None:
            start_date = fact.get("start_date")
            if start_date:
                try:
                    age_days = (as_of - pd.Timestamp(start_date)).days
                except (TypeError, ValueError):
                    age_days = None
                if age_days is not None and age_days < min_age_days:
                    reason = "new_listing"
        if reason is None and min_avg_amount > 0:
            liquidity = (fact.get("checks") or {}).get("liquidity") or {}
            avg_amount = liquidity.get("average_amount_20d")
            if avg_amount is not None:
                try:
                    if float(avg_amount) < min_avg_amount:
                        reason = "low_liquidity"
                except (TypeError, ValueError):
                    pass
        if reason:
            exclusions.append({"symbol": symbol, "name": name, "reason": reason})
        else:
            eligible.append(signal)
    return eligible, exclusions


def run_research(
    mode: str = "daily",
    asset_scope: str = "all",
    symbols: list[str] | None = None,
    *,
    depth: str | None = None,
    _force_sync: bool = False,
    _target_date: str | None = None,
    _start_background_worker: bool | None = None,
    _symbols_by_type: Mapping[str, Iterable[str]] | None = None,
    _run_id_override: str | None = None,
    _run_dir_override: Path | None = None,
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
    if _symbols_by_type is not None:
        requested_symbols = sorted({
            symbol
            for routed_symbols in explicit_routes.values()
            for symbol in routed_symbols
        })
    directed_research = bool(requested_symbols) or _symbols_by_type is not None
    config = load_config()
    if depth is None:
        # daily/monthly 是定时维护链路，缺省必须保持全量刷新语义。
        depth = (
            "standard"
            if mode in {"daily", "monthly"}
            else str(config["research"].get("default_depth", "standard"))
        )
    depth = str(depth).strip().lower()
    if depth not in DEPTH_NAMES:
        raise ValueError(f"depth must be one of: {', '.join(sorted(DEPTH_NAMES))}")
    if depth == "full" and explicit_routes:
        raise ValueError(
            "depth=full refreshes the whole selected universe; "
            "omit symbols or use depth=standard/quick"
        )
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
            depth=depth,
        )
    started = _now()
    if _run_id_override is None:
        if _run_dir_override is not None:
            raise ValueError("_run_dir_override requires _run_id_override")
        run_id = _run_id()
        run_dir = (
            RUNS_DIR
            / started.strftime("%Y")
            / started.strftime("%m")
            / started.strftime("%d")
            / run_id
        )
        reuse_run_dir = False
    else:
        run_id = validate_run_id(_run_id_override)
        expected_run_dir = (
            RUNS_DIR / run_id[:4] / run_id[4:6] / run_id[6:8] / run_id
        ).resolve()
        run_dir = Path(_run_dir_override or expected_run_dir).resolve()
        if run_dir != expected_run_dir or run_dir.name != run_id:
            raise ValueError("queued run directory does not match run_id")
        if not run_dir.is_dir():
            raise FileNotFoundError(f"queued run directory does not exist: {run_id}")
        reuse_run_dir = True
    lock_descriptor: int | None = None
    log_lines = [f"started {started.isoformat()}", f"mode={mode}", f"asset_types={asset_types}"]
    status_path = run_dir / "status.json"
    _sweep_stale_runs(config, log_lines)
    
    # === Regime检测：在熊市时自动调整风险参数 ===
    current_regime = 'unknown'  # 默认值
    regime_manager = None       # 2026-09-11 整改(P2-9)：供发布阶段回写训练状态
    try:
        benchmark_returns = get_benchmark_returns(str(DATA_DIR))
        regime_manager = RegimeAwareModelManager(state_file=str(DATA_DIR / "model_regime_state.json"))
        regime_check = regime_manager.check_should_retrain(
            benchmark_returns,
            current_date=started.strftime('%Y-%m-%d')
        )
        
        current_regime = regime_check['current_regime']
        log_lines.append(f"market_regime={current_regime}")
        
        # 熊市自动降低风险
        if 'bear' in current_regime:
            log_lines.append("bear_market_detected: adjusting_risk_params")
            # 动态调整config（不修改文件，只影响本次运行）
            config = dict(config)  # 浅拷贝
            config['risk'] = dict(config.get('risk', {}))
            config['risk']['min_cash_weight'] = max(config['risk'].get('min_cash_weight', 0.2), 0.4)
            config['risk']['max_stock_exposure'] = min(config['risk'].get('max_stock_exposure', 0.4), 0.2)
            config['risk']['max_daily_turnover'] = min(config['risk'].get('max_daily_turnover', 0.2), 0.15)
            log_lines.append(f"risk_adjusted: cash≥40%, stock≤20%, turnover≤15%")
        
        # 2026-09-11 整改(P2-9)：这里只作为"建议"记录，不做额外动作。
        # 模型缓存键已包含 provider_end，provider 每次重建都会自然触发一次重训；
        # 原先该分支恒为 True（save_state 从未被调用，last_trained_regime 永远为 None），
        # 且只写日志、无任何行为，容易误导为"已有 regime 感知重训"。
        if regime_check['should_retrain']:
            log_lines.append(f"retrain_recommended(advisory): {regime_check['reason']}")
            
    except Exception as e:
        log_lines.append(f"regime_detection_failed: {e}")
        # 失败不影响正常流程，继续使用原配置
    
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
    research_progress_path = run_dir / "research-progress.json"
    research_phase_seq = 0
    stage_timings: dict[str, dict[str, Any]] = {}
    current_stage: dict[str, Any] = {"name": None, "started": None}

    def _record_stage(phase: str) -> None:
        now = _now()
        if current_stage["name"] is not None and current_stage["name"] != str(phase):
            name = current_stage["name"]
            entry = stage_timings.setdefault(
                name,
                {"started_at": current_stage["started"].isoformat()},
            )
            entry["finished_at"] = now.isoformat()
            entry["elapsed_seconds"] = round(
                float(entry.get("elapsed_seconds") or 0.0)
                + (now - current_stage["started"]).total_seconds(),
                3,
            )
        if current_stage["name"] != str(phase):
            stage_timings.setdefault(str(phase), {"started_at": now.isoformat()})
            current_stage["name"] = str(phase)
            current_stage["started"] = now

    def _close_stage() -> None:
        if current_stage["name"] is None:
            return
        now = _now()
        name = current_stage["name"]
        entry = stage_timings.setdefault(
            name,
            {"started_at": current_stage["started"].isoformat()},
        )
        entry["finished_at"] = now.isoformat()
        entry["elapsed_seconds"] = round(
            float(entry.get("elapsed_seconds") or 0.0)
            + (now - current_stage["started"]).total_seconds(),
            3,
        )
        current_stage["name"] = None
        current_stage["started"] = None

    def report_research_phase(phase: str, **details: Any) -> None:
        nonlocal research_phase_seq
        _record_stage(phase)
        research_phase_seq += 1
        record = {
            "run_id": run_id,
            "status": "running",
            "phase": str(phase),
            "phase_seq": research_phase_seq,
            "updated_at": _now().isoformat(),
        }
        for key in ("cache_hit", "cache_miss_reason", "provider_end", "message"):
            if key in details and details[key] is not None:
                record[key] = details[key]
        atomic_write_json(research_progress_path, record)
    def touch_phase(phase: str) -> None:
        _record_stage(phase)
        if reuse_run_dir:
            report_research_phase(phase)
        else:
            _touch_status(status_path, phase)

    try:
        lock_descriptor = _acquire_lock(
            run_id,
            int(config["runtime"]["lock_stale_seconds"]),
        )
        run_dir.mkdir(parents=True, exist_ok=reuse_run_dir)
        if reuse_run_dir:
            report_research_phase("research_start")
        else:
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

        if depth == "quick":
            refresh["skipped_by_depth"] = "quick"
            log_lines.append("depth=quick：跳过数据刷新，只读缓存数据与既有 provider。")
        elif mode in set(config["research"]["refresh_modes"]):
            refresh = _refresh_data(
                analyzed_types,
                config,
                log_lines,
                symbols_by_type=symbol_routes if explicit_routes else None,
            )
        touch_phase("refreshed")
        all_frames: dict[str, dict[str, pd.DataFrame]] = {}
        for asset_type in analyzed_types:
            scope_symbols = symbol_routes[asset_type] if explicit_routes else None
            frames, facts = _scan_assets(asset_type, scope_symbols)
            all_frames[asset_type] = frames
            all_facts.extend(facts)
        data_quality = _quality_for_depth(
            _aggregate_quality(all_facts, config),
            all_facts,
            config,
            depth=depth,
        )
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
            if depth == "quick":
                raise ValueError(
                    "depth=quick 未找到任何缓存数据；请先用 depth=standard 或 full 刷新后重试。"
                )
            raise ValueError("no asset data found in fixed data directories")

        touch_phase("scanned")
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

        touch_phase("signals")
        # 模型工作流要一次性分配数 GiB 的 Alpha158 特征矩阵；技术信号已算完，
        # 行情帧此后不再被使用（模型走 Qlib provider 自有通道），显式释放腾内存。
        all_frames.clear()
        gc.collect()
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
            stock_quality = _quality_for_depth(
                _aggregate_quality(stock_facts, config),
                stock_facts,
                config,
                depth=depth,
            )
            stock_refresh_ok = all(
                result.get("status") in {"success", "reused"}
                for result in refresh.get("results", [])
                if result.get("label") in {"stock", "qlib_provider"}
            )
            if stock_quality["passed"] and stock_refresh_ok:
                model_options: dict[str, Any] = {}
                if explicit_routes:
                    model_options["requested_symbols"] = symbol_routes.get("stock", [])
                if reuse_run_dir:
                    model_options["progress_callback"] = report_research_phase
                stock_model = _run_qlib_stock_model(config, **model_options)
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

        touch_phase("gates")
        decision_support = {
            "allocation_basis": "基于假设资金/配置的目标权重",
            "hypothetical_capital": float(config["research"]["hypothetical_capital"]),
            "broker_connected": False,
            "auto_order": False,
            "asset_paths": {},
            "target_weights": {},
            "cash_weight": 1.0,
        }
        run_budget = allocation_budget(config["risk"], len(analyzed_types))
        weight_strategy = str(config["research"]["weight_strategy"])

        if "stock" in analyzed_types:
            stock_facts = [item for item in all_facts if item["asset_type"] == "stock"]
            stock_quality = _quality_for_depth(
                _aggregate_quality(stock_facts, config),
                stock_facts,
                config,
                depth=depth,
            )
            stock_metrics = dict(stock_model)
            stock_metrics["model_available"] = bool(stock_model.get("model_available"))
            checks = model_gate_checks(stock_metrics, config["gate"])
            checks["data_quality_passed"] = bool(stock_quality["passed"])
            checks["freshness_passed"] = _freshness_passed(
                stock_quality,
                config,
                full_universe=not directed_research,
            )
            checks["refresh_passed"] = all(
                result.get("status") in {"success", "reused"}
                for result in refresh.get("results", [])
                if result.get("label") in {"stock", "qlib_provider"}
            )
            checks["leakage_check_passed"] = bool(stock_model.get("leakage_check", {}).get("passed", False))
            # 海选原则（有能者居之）：定向研究只输出信号与门槛，不参与权重排名。
            require_screen = bool(config["research"].get("require_full_universe_for_weights", True))
            checks["full_universe_screen_passed"] = (not require_screen) or (not directed_research)
            checks["positive_signal_passed"] = True
            stock_requested = {}
            screen_block_reason = ""
            if not checks["full_universe_screen_passed"]:
                screen_block_reason = (
                    "定向研究仅输出信号与门槛；个股目标权重只由全市场海选排名产生，不生成个股买入目标权重。"
                )
            gate_passed = all(checks.values())
            if gate_passed:
                facts_by_symbol = {item.get("symbol"): item for item in stock_facts}
                eligible_signals, pool_exclusions = _pool_quality_filter(
                    technical_signals["stock"]["signals"],
                    facts_by_symbol,
                    config,
                    str(data_quality.get("data_end_date") or ""),
                )
                for exclusion in pool_exclusions[:10]:
                    log_lines.append(
                        f"[宁缺毋滥] {exclusion['symbol']} {exclusion.get('name', '')} 剔除: {exclusion['reason']}"
                    )
                if pool_exclusions:
                    log_lines.append(
                        f"[宁缺毋滥] 共剔除 {len(pool_exclusions)} 个候选（ST/退市整理/次新/低流动性）"
                    )
                # 2026-09-11 整改(P1-8)：移除 LowTurnoverTopkStrategy 的"假接入"。
                # 原接入每次运行都新建实例，current_position/entry_prices 恒为空：
                # 持仓惯性分支永不执行；止损分支未传 current_prices 且只是 continue 空操作；
                # 返回的权重被丢弃（只取 keys() 重排），实质只施加了一个 regime 相关的
                # |score| 阈值过滤。而回测路径（qlib_model.py）并不经过该策略，
                # 造成实盘选股口径与回测不一致。此处恢复"按得分取 top-k"的文档化契约。
                # 低换手/持仓惯性若要真正生效，必须持久化持仓与成本价状态，并同时
                # 接入回测（见 docs/reports/量化系统评估报告-20260911.md 的 P1-8）。
                ranked = eligible_signals[: int(config["research"]["stock_topk"])]

                min_score = float(config["research"].get("min_qlib_score_for_weights", 0.0))
                screened = []
                for signal in ranked:
                    try:
                        score = float(signal.get("qlib_score"))
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(score) and score > min_score:
                        screened.append(signal)
                checks["positive_signal_passed"] = bool(screened)
                if not screened:
                    gate_passed = False
                    screen_block_reason = "海选前列无个体得分为正的候选，不生成个股买入目标权重。"
                else:
                    stock_requested = build_target_weights(
                        screened,
                        "stock",
                        config["risk"],
                        budget=run_budget,
                        strategy=weight_strategy,
                    )
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
                    screen_block_reason or "个股模型或风险门槛未通过，不生成个股买入目标权重。",
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
            facts_by_type = {item.get("symbol"): item for item in all_facts if item.get("asset_type") == asset_type}
            candidates, cross_exclusions = _pool_quality_filter(
                candidates,
                facts_by_type,
                config,
                str(data_quality.get("data_end_date") or ""),
            )
            for exclusion in cross_exclusions[:10]:
                log_lines.append(
                    f"[宁缺毋滥] {exclusion['symbol']} {exclusion.get('name', '')} 剔除: {exclusion['reason']}"
                )
            candidates = candidates[:2]
            category = "etf" if asset_type == "etf" else "otc_fund"
            requested = build_target_weights(
                candidates,
                category,
                config["risk"],
                budget=run_budget,
                strategy=weight_strategy,
            )
            risk_result = enforce_target_weights(
                requested,
                {symbol: category for symbol in requested},
                config["risk"],
            )
            asset_facts = [item for item in all_facts if item["asset_type"] == asset_type]
            asset_quality = _quality_for_depth(
                _aggregate_quality(asset_facts, config),
                asset_facts,
                config,
                depth=depth,
            )
            checks = {
                "data_quality_passed": bool(asset_quality["passed"]),
                "freshness_passed": _freshness_passed(
                    asset_quality,
                    config,
                    full_universe=not directed_research,
                ),
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
            # 2026-09-11 整改(P0-2)：横截面退化（数据不足）与"模型不合格"必须区分，
            # 否则日历/数据对齐问题会被记成模型能力问题，污染调度器 failure_streak。
            _stock_metrics_out = rule_gates.get("stock", {}).get("model_metrics") or {}
            if _stock_metrics_out.get("degenerate"):
                failure_reasons.append("stock_signal_cross_section_degenerate")
                log_lines.append(
                    "degenerate_signals: "
                    + str(_stock_metrics_out.get("degenerate_reason"))
                )
            else:
                failure_reasons.append("stock_model_or_risk_gate_failed")
        if not portfolio_risk["passed"]:
            failure_reasons.append("portfolio_risk_gate_failed")
        for asset_type in ("etf", "otc_fund"):
            if asset_type in analyzed_types and not rule_gates.get(asset_type, {}).get("passed", False):
                failure_reasons.append(f"{asset_type}_quality_or_risk_gate_failed")
        # 2026-09-11 整改(P0-3)：**分类别发布**——类别内一票否决，类别间独立发布。
        # 原先 `published` 只看"有没有任何 failure_reasons"，于是股票门槛一失败，
        # 已通过的 ETF/基金结论会被一并丢弃，界面只能看到陈旧快照，且无从察觉。
        # 现在：只要还有类别就绪就落盘 current.json，被拦类别显式归零并标注原因。
        ready_asset_types = [
            asset_type
            for asset_type in analyzed_types
            if rule_gates.get(asset_type, {}).get("passed", False)
        ]
        blocked_asset_types = [
            asset_type for asset_type in analyzed_types
            if asset_type not in ready_asset_types
        ]
        publication = {
            "state": (
                "blocked"
                if not ready_asset_types
                else ("full" if not blocked_asset_types else "partial")
            ),
            "published": bool(ready_asset_types),
            "ready_asset_types": ready_asset_types,
            "blocked_asset_types": blocked_asset_types,
            "blocked_reasons": {
                asset_type: [
                    name
                    for name, ok in (rule_gates.get(asset_type, {}).get("checks") or {}).items()
                    if not ok
                ]
                for asset_type in blocked_asset_types
            },
        }
        decision_support["publication"] = publication
        log_lines.append(
            "publication_readiness: ready=%s blocked=%s" % (
                ready_asset_types or "[]",
                blocked_asset_types or "[]",
            )
        )
        # 被拦类别：权重必须归零，并从总权重表里剔除其标的。
        # 安全性质：绝不能把"未发布类别"的权重当成新结论（宁可空仓）。
        # 若该类别已在自身门槛路径里归零并给了更具体的原因（如个股的
        # `cash_or_observe` + "个股模型或风险门槛未通过…"），保留它，不要用泛化文案覆盖。
        if blocked_asset_types:
            for asset_type in blocked_asset_types:
                decision = decision_support["asset_paths"].get(asset_type)
                if not isinstance(decision, dict):
                    continue
                leaked = decision.get("target_weights") or {}
                for symbol in leaked:
                    decision_support["target_weights"].pop(symbol, None)
                decision["target_weights"] = {}
                decision["cash_weight"] = 1.0
                if leaked:
                    decision["status"] = "blocked"
                    decision["action"] = "不发布"
                    decision["reason"] = (
                        "该资产类别未通过质量/风险门槛，本轮不发布其目标权重。"
                    )
            log_lines.append(
                "publication partial: blocked types withheld: %s" % (blocked_asset_types,)
            )

        if not ready_asset_types:
            run_status = "failed"
        elif blocked_asset_types:
            run_status = "partial"
        else:
            run_status = "success"
        decision_support["publication_blocked"] = not ready_asset_types
        if failure_reasons:
            decision_support["failure_reasons"] = failure_reasons
        if run_status == "failed":
            log_lines.append(f"publication blocked: {failure_reasons}")
        if run_status != "failed" and regime_manager is not None:
            # 2026-09-11 整改(P2-9)：把 regime 训练状态真正落盘。原先 save_state
            # 在生产代码中从未被调用，导致 should_retrain 永远读不到上次状态
            # （last_trained_regime 恒为 None），regime 感知重训形同虚设。
            # 必须在 _write_run_files 之前执行，日志才能随本轮一起落盘。
            try:
                _ic_raw = (model_metadata.get("stock_model") or {}).get("mean_rank_ic")
                _ic = float(_ic_raw) if isinstance(_ic_raw, (int, float)) else None
                regime_manager.save_state(
                    regime=current_regime,
                    date=_now().isoformat(),
                    ic=_ic,
                )
                log_lines.append(f"regime_state_saved: {current_regime}")
            except Exception as exc:
                log_lines.append(
                    f"regime_state_save_failed: {type(exc).__name__}: {exc}"
                )

        _close_stage()
        manifest = {
            "run_id": run_id,
            "status": run_status,
            "started_at": started.isoformat(),
            "finished_at": _now().isoformat(),
            "mode": mode,
            "depth": depth,
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
            "publication": publication,
            # 只读可观测性：离线 stability 报告摘要（诊断用，不进门槛）。
            "offline_stability": _offline_stability_summary(),
            "stage_timings": _jsonable(stage_timings),
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
        if reuse_run_dir:
            report_research_phase("publish", message=run_status)
        status = {
            "run_id": run_id,
            "status": run_status,
            "depth": depth,
            "finished_at": manifest["finished_at"],
            "data_end_date": data_quality.get("data_end_date"),
            "result_path": str(run_dir),
            "failure_reasons": failure_reasons,
        }
        if not reuse_run_dir:
            atomic_write_json(status_path, status)
        # 2026-09-11 整改(P0-3)：只要还有类别就绪就落盘（含 partial）。
        # 被拦类别的权重已在上面归零标注，因此 current.json 里不会混入未发布的旧结论。
        if ready_asset_types:
            current = {
                "run_id": run_id,
                "status": run_status,
                "updated_at": manifest["finished_at"],
                "data_end_date": data_quality.get("data_end_date"),
                "result_path": str(run_dir),
                "paths": result_paths,
                "decision_support": _jsonable(decision_support),
                "rule_gates": _jsonable(rule_gates),
            }
            atomic_write_json(ASSISTANT_RESULTS_DIR / "current.json", current)
            if blocked_asset_types:
                notification = add_notification(
                    "research_partial",
                    "股票/基金研究部分发布",
                    f"运行 {run_id} 部分发布：已就绪 {ready_asset_types}；"
                    f"未通过门槛、本轮不发布 {blocked_asset_types}"
                    f"（数据截止 {data_quality.get('data_end_date') or 'unknown'}）。",
                    run_id=run_id,
                    paths=result_paths,
                )
            else:
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
        _close_stage()
        manifest = {
            "run_id": run_id,
            "status": "failed",
            "started_at": started.isoformat(),
            "finished_at": _now().isoformat(),
            "mode": mode,
            "depth": depth,
            "asset_scope": asset_scope,
            "error": error,
            # 只读可观测性：离线 stability 报告摘要（诊断用，不进门槛）。
            "offline_stability": _offline_stability_summary(),
            "stage_timings": _jsonable(stage_timings),
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
            if not reuse_run_dir:
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
    _fixed_reference,
    _latest_run_id,
    ack_notification,
    get_current_snapshot,
    get_run_status,
    list_notifications,
    _sanitize_result_record,
)
