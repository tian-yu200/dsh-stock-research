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
    return {
        "data_quality_and_model_available": bool(metrics.get("model_available", False)),
        "mean_rank_ic_threshold": _finite(rank_ic)
        and float(rank_ic) >= float(gate["min_mean_rank_ic"]),
        "rank_ic_ir_threshold": _finite(rank_ic_ir) and float(rank_ic_ir) >= float(gate["min_rank_ic_ir"]),
        "top_bottom_excess_return_threshold": _finite(top_bottom_excess)
        and float(top_bottom_excess) >= float(gate["min_top_bottom_excess_return"]),
        "test_max_drawdown_threshold": _finite(max_drawdown)
        and abs(float(max_drawdown)) <= float(gate["max_test_drawdown"]),
    }


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
