"""Risk gates and percentage-first target-weight constraints."""
from __future__ import annotations

import math
from typing import Any, Mapping


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def model_gate_checks(metrics: Mapping[str, Any], gate: Mapping[str, Any]) -> dict[str, bool]:
    rank_ic = metrics.get("mean_rank_ic")
    rank_ic_ir = metrics.get("rank_ic_ir")
    top_bottom_excess = metrics.get("top_bottom_excess_return")
    max_drawdown = metrics.get("test_max_drawdown")
    benchmark_drawdown = (metrics.get("benchmark") or {}).get("max_drawdown")
    # The backtest portfolio is always fully invested, so its drawdown scales with the
    # market regime; constrain the excess over the benchmark instead of an absolute cap.
    if _finite(benchmark_drawdown):
        drawdown_limit = abs(float(benchmark_drawdown)) + float(gate["max_excess_test_drawdown"])
    else:
        drawdown_limit = float(gate["max_test_drawdown"])
    # 拟合质量门槛（**默认关闭，且不建议启用绝对值口径**）。
    #
    # 2026-09-11 初次引入时把 train_l2≈0.995 误判为"模型没学到"，据此设了 0.02。
    # 当日复核证明该判断错误：截面 z-score 标签下方差为 1，故 R² ≈ IC²；
    # 实测 train_l2=0.995423 → 隐含 |ρ|=0.0677，与同期 test IC=0.0563 自洽，
    # 且 5 分组标签均值严格单调。要求 R² ≥ 2% 等于要求 IC ≥ 0.141，A 股不可达。
    # 因此生产值已回退为 0.0（关闭）；模型质量由 RankIC / IR / 分组超额 / 回撤负责。
    #
    # 保留该分支的用途：0 表示关闭；一旦显式启用，指标缺失时判失败（fail-closed），
    # 避免"配置写了门槛却因取不到指标而静默放行"。
    min_fit_improvement = float(gate.get("min_train_l2_improvement", 0.0) or 0.0)
    train_l2_improvement = metrics.get("train_l2_improvement")
    fit_quality_ok = (
        True
        if min_fit_improvement <= 0.0
        else bool(_finite(train_l2_improvement))
        and float(train_l2_improvement) >= min_fit_improvement
    )
    return {
        "data_quality_and_model_available": bool(metrics.get("model_available", False)),
        "mean_rank_ic_threshold": _finite(rank_ic)
        and float(rank_ic) >= float(gate["min_mean_rank_ic"]),
        "rank_ic_ir_threshold": _finite(rank_ic_ir) and float(rank_ic_ir) >= float(gate["min_rank_ic_ir"]),
        "top_bottom_excess_return_threshold": _finite(top_bottom_excess)
        and float(top_bottom_excess) >= float(gate["min_top_bottom_excess_return"]),
        "test_max_drawdown_threshold": _finite(max_drawdown)
        and abs(float(max_drawdown)) <= drawdown_limit,
        "fit_quality_threshold": fit_quality_ok,
        "stability_ic_std_threshold": _stability_ic_std_check(metrics, gate),
    }


def _stability_ic_std_check(metrics: Mapping[str, Any], gate: Mapping[str, Any]) -> bool:
    """多窗口滚动一致性门槛：跨窗口 IC 标准差超限即判不稳定。

    - ``stability_stable`` 为 ``None`` 表示稳定性检查未启用（未配置
      ``research.stability_windows``），此时**不阻断**，与既有可选门槛
      （如默认关闭的 fit_quality）保持一致的语义。
    - 已启用但取不到有效 ``stability_ic_std`` 时判失败（fail-closed），
      避免"配置了门槛却因缺指标而静默放行"。
    """
    stable = metrics.get("stability_stable")
    if stable is None:
        return True
    ic_std = metrics.get("stability_ic_std")
    max_ic_std = float(gate.get("max_ic_std", 0.02))
    if not _finite(ic_std):
        return False
    if not stable:
        return False
    return float(ic_std) <= max_ic_std


def enforce_target_weights(
    requested: Mapping[str, Any],
    asset_types: Mapping[str, str],
    risk: Mapping[str, Any],
    previous: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate target weights without silently turning invalid requests executable."""
    weights = {
        str(symbol): max(0.0, float(value))
        for symbol, value in requested.items()
        if _finite(value) and float(value) > 0
    }
    single_ok = all(value <= float(risk["max_single_weight"]) + 1e-12 for value in weights.values())
    categories = {"stock": 0.0, "etf": 0.0, "otc_fund": 0.0}
    unknown_symbols = sorted(
        symbol
        for symbol in weights
        if asset_types.get(symbol) not in categories
    )
    for symbol, value in weights.items():
        category = asset_types.get(symbol)
        if category in categories:
            categories[category] += value
    exposure_limits = {
        "stock": float(risk["max_stock_exposure"]),
        "etf": float(risk["max_etf_exposure"]),
        "otc_fund": float(risk["max_otc_fund_exposure"]),
    }
    category_checks = {
        f"{category}_exposure": total <= exposure_limits[category] + 1e-12
        for category, total in categories.items()
    }
    invested = sum(weights.values())
    cash_weight = 1.0 - invested
    cash_ok = cash_weight + 1e-12 >= float(risk["min_cash_weight"])
    previous_weights = {
        str(symbol): max(0.0, float(value))
        for symbol, value in (previous or {}).items()
        if _finite(value)
    }
    symbols = set(weights) | set(previous_weights)
    target_cash = 1.0 - sum(weights.values())
    previous_cash = 1.0 - sum(previous_weights.values())
    turnover = (
        sum(
            abs(weights.get(symbol, 0.0) - previous_weights.get(symbol, 0.0))
            for symbol in symbols
        )
        + abs(target_cash - previous_cash)
    ) / 2.0
    turnover_ok = turnover <= float(risk["max_daily_turnover"]) + 1e-12
    checks = {
        "max_single_weight": single_ok,
        "known_asset_types": not unknown_symbols,
        **category_checks,
        "min_cash_weight": cash_ok,
        "max_daily_turnover": turnover_ok,
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "checks": checks,
        "requested_weights": weights,
        "target_weights": weights if passed else {},
        "category_exposure": categories,
        "cash_weight": cash_weight if passed else 1.0,
        "turnover": turnover,
        "unknown_symbols": unknown_symbols,
        "failure_reasons": [name for name, ok in checks.items() if not ok],
    }


def cash_only_decision(reason: str, *, capital: float = 100000.0) -> dict[str, Any]:
    return {
        "status": "cash_or_observe",
        "action": "现金",
        "target_weights": {},
        "cash_weight": 1.0,
        "allocation_basis": "基于假设资金/配置的目标权重",
        "hypothetical_capital": capital,
        "reason": reason,
        "broker_connected": False,
        "auto_order": False,
    }
