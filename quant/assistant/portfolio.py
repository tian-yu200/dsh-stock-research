"""目标权重生成（docs/重构计划.md 阶段 2）。

把带分数的候选列表转换成 ``risk.enforce_target_weights`` 可直接消费的
requested 字典。策略可插拔：

- ``equal_score``（默认）：逐字复刻拆分前 run_research 内嵌的等权切分——
  预算按候选数平分，再对单票上限取 ``min``。
- ``score_weighted``：按分数比例分摊预算后裁剪到单票上限（不回流溢出
  份额，差额自然落入现金）；分数全部缺失或非正时回退为等权切分。

策略仅在配置 ``research.weight_strategy`` 中显式切换，默认永远是
``equal_score``。本模块是纯函数层，不依赖 research 编排层。

候选契约：每个候选是含 ``symbol`` 键的映射；分数读取优先级为显式
``score`` 键 > ``qlib_score`` > ``trend_score`` > 0.0（首个存在的键生效，
非有限数值按 0 处理，负分数不参与分摊且不产生权重）。``previous`` 参数
为未来"换手感知分配"预留，当前两个策略都不使用。
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

DEFAULT_STRATEGY = "equal_score"
STRATEGIES = ("equal_score", "score_weighted")


def allocation_budget(risk_config: Mapping[str, Any], analyzed_type_count: int) -> float:
    """每个已分析资产类型分得的预算份额（拆分前 run_research 的原公式）。"""
    return min(
        float(risk_config["max_daily_turnover"]),
        1.0 - float(risk_config["min_cash_weight"]),
    ) / max(int(analyzed_type_count), 1)


def _candidate_score(candidate: Mapping[str, Any]) -> float:
    for key in ("score", "qlib_score", "trend_score"):
        if key in candidate:
            try:
                value = float(candidate[key])
            except (TypeError, ValueError):
                return 0.0
            return value if math.isfinite(value) else 0.0
    return 0.0


def _score_weighted(
    ranked: list[tuple[str, float]],
    budget: float,
    risk_config: Mapping[str, Any],
) -> dict[str, float]:
    max_single = float(risk_config["max_single_weight"])
    scores = {symbol: max(0.0, score) for symbol, score in ranked}
    total = sum(scores.values())
    if total <= 0.0:
        per_symbol = min(budget / len(ranked), max_single)
        return {symbol: per_symbol for symbol, _ in ranked}
    return {
        symbol: min(budget * score / total, max_single)
        for symbol, score in scores.items()
        if score > 0.0
    }


def build_target_weights(
    candidates: Iterable[Mapping[str, Any]],
    asset_type: str,
    risk_config: Mapping[str, Any],
    previous: Mapping[str, Any] | None = None,
    *,
    budget: float,
    strategy: str = DEFAULT_STRATEGY,
) -> dict[str, float]:
    """生成 requested 权重字典；空候选永远返回空字典。

    ``previous`` 当前不参与计算，仅作为未来换手感知策略的入参占位。
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"weight strategy must be one of: {', '.join(STRATEGIES)}")
    ranked = [
        (str(item["symbol"]), _candidate_score(item))
        for item in candidates
        if item.get("symbol") is not None
    ]
    if not ranked:
        return {}
    if strategy == "score_weighted":
        return _score_weighted(ranked, float(budget), risk_config)
    per_symbol = min(
        float(budget) / len(ranked),
        float(risk_config["max_single_weight"]),
    )
    return {symbol: per_symbol for symbol, _ in ranked}
