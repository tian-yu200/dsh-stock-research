"""阶段 2 单元测试：portfolio.build_target_weights 的策略与边界。

覆盖重构计划 §5 要求的四类边界（零候选、单候选超单票上限、类别敞口
顶格、换手超限回退现金）以及 score_weighted 的比例分摊、裁剪与回退。
enforce_target_weights（risk.py）本身不做任何语义修改，仅作为链条
下游参与断言。
"""
from __future__ import annotations

import copy
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from assistant.config import DEFAULTS
from assistant.portfolio import (
    DEFAULT_STRATEGY,
    STRATEGIES,
    allocation_budget,
    build_target_weights,
)
from assistant.risk import enforce_target_weights


class AllocationBudgetTests(unittest.TestCase):
    def test_matches_original_inline_formula(self) -> None:
        risk = DEFAULTS["risk"]
        self.assertAlmostEqual(allocation_budget(risk, 3), 0.2 / 3, places=15)
        self.assertAlmostEqual(allocation_budget(risk, 1), 0.2, places=15)
        self.assertAlmostEqual(allocation_budget(risk, 0), 0.2, places=15)

    def test_turnover_binds_when_stricter_than_cash_floor(self) -> None:
        risk = {**DEFAULTS["risk"], "max_daily_turnover": 0.10}
        self.assertAlmostEqual(allocation_budget(risk, 2), 0.05, places=15)

    def test_cash_floor_binds_when_stricter_than_turnover(self) -> None:
        risk = {**DEFAULTS["risk"], "min_cash_weight": 0.90}
        self.assertAlmostEqual(allocation_budget(risk, 2), 0.05, places=15)


class BuildTargetWeightsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.risk = copy.deepcopy(DEFAULTS["risk"])
        self.budget = allocation_budget(self.risk, 1)

    def test_zero_candidates_request_nothing(self) -> None:
        self.assertEqual(
            build_target_weights([], "stock", self.risk, budget=self.budget), {}
        )

    def test_equal_score_single_candidate_is_capped_at_single_weight(self) -> None:
        result = build_target_weights(
            [{"symbol": "600519", "score": 9.9}], "stock", self.risk, budget=self.budget
        )
        self.assertAlmostEqual(result["600519"], 0.20, places=15)

    def test_equal_score_splits_budget_by_candidate_count(self) -> None:
        candidates = [{"symbol": symbol} for symbol in ("600519", "000001", "300750")]
        result = build_target_weights(
            candidates, "stock", self.risk, budget=0.06666666666666667
        )
        self.assertEqual(set(result), {"600519", "000001", "300750"})
        for symbol in result:
            self.assertAlmostEqual(result[symbol], 0.022222222222222223, places=15)

    def test_equal_score_ignores_scores(self) -> None:
        candidates = [
            {"symbol": "600519", "score": 100.0},
            {"symbol": "000001", "score": 0.01},
        ]
        result = build_target_weights(candidates, "stock", self.risk, budget=0.2)
        self.assertAlmostEqual(result["600519"], 0.1, places=15)
        self.assertAlmostEqual(result["000001"], 0.1, places=15)

    def test_equal_score_accepts_previous_placeholder(self) -> None:
        with_previous = build_target_weights(
            [{"symbol": "600519"}],
            "stock",
            self.risk,
            previous={"OLD": 0.10},
            budget=self.budget,
        )
        self.assertAlmostEqual(with_previous["600519"], 0.20, places=15)

    def test_score_weighted_proportional_to_scores(self) -> None:
        candidates = [
            {"symbol": "510300", "score": 3.0},
            {"symbol": "159915", "score": 1.0},
        ]
        result = build_target_weights(
            candidates, "etf", self.risk, budget=0.2, strategy="score_weighted"
        )
        self.assertAlmostEqual(result["510300"], 0.15, places=15)
        self.assertAlmostEqual(result["159915"], 0.05, places=15)

    def test_score_weighted_clips_to_single_cap(self) -> None:
        candidates = [
            {"symbol": "510300", "score": 9.0},
            {"symbol": "159915", "score": 1.0},
        ]
        result = build_target_weights(
            candidates, "etf", self.risk, budget=0.5, strategy="score_weighted"
        )
        self.assertAlmostEqual(result["510300"], 0.20, places=15)
        self.assertAlmostEqual(result["159915"], 0.05, places=15)

    def test_score_weighted_all_zero_scores_falls_back_to_equal(self) -> None:
        candidates = [{"symbol": "110020"}, {"symbol": "161725", "score": 0.0}]
        result = build_target_weights(
            candidates, "otc_fund", self.risk, budget=0.2, strategy="score_weighted"
        )
        self.assertAlmostEqual(result["110020"], 0.1, places=15)
        self.assertAlmostEqual(result["161725"], 0.1, places=15)

    def test_score_weighted_drops_non_positive_scores(self) -> None:
        candidates = [
            {"symbol": "600519", "score": 2.0},
            {"symbol": "300750", "qlib_score": -1.0, "trend_score": 5.0},
        ]
        result = build_target_weights(
            candidates, "stock", self.risk, budget=0.2, strategy="score_weighted"
        )
        self.assertAlmostEqual(result["600519"], 0.2, places=15)
        self.assertNotIn("300750", result)

    def test_score_precedence_explicit_score_wins(self) -> None:
        candidates = [
            {"symbol": "510300", "score": 1.0, "qlib_score": 7.0, "trend_score": 9.0},
            {"symbol": "159915", "score": 1.0},
        ]
        result = build_target_weights(
            candidates, "etf", self.risk, budget=0.2, strategy="score_weighted"
        )
        self.assertAlmostEqual(result["510300"], 0.1, places=15)

    def test_unknown_strategy_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "weight strategy"):
            build_target_weights(
                [{"symbol": "600519"}], "stock", self.risk, budget=0.2, strategy="all_in"
            )


class BuildEnforceChainTests(unittest.TestCase):
    """build 的输出必须能被 enforce_target_weights 直接消费。"""

    def setUp(self) -> None:
        self.risk = copy.deepcopy(DEFAULTS["risk"])

    def test_category_exposure_at_cap_passes_enforcement(self) -> None:
        risk = {**self.risk, "max_daily_turnover": 0.50, "min_cash_weight": 0.0}
        budget = allocation_budget(risk, 1)
        candidates = [{"symbol": "510300"}, {"symbol": "159915"}, {"symbol": "560010"}]
        requested = build_target_weights(candidates, "etf", risk, budget=budget)
        self.assertAlmostEqual(sum(requested.values()), 0.50, places=12)
        result = enforce_target_weights(
            requested, {symbol: "etf" for symbol in requested}, risk
        )
        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["category_exposure"]["etf"], 0.50, places=12)

    def test_turnover_breach_falls_back_to_cash(self) -> None:
        requested = build_target_weights(
            [{"symbol": "600519"}, {"symbol": "000001"}],
            "stock",
            self.risk,
            budget=allocation_budget(self.risk, 1),
        )
        previous = {"600519": 0.0, "000001": 0.0, "300750": 0.20, "510300": 0.20}
        result = enforce_target_weights(
            requested,
            {symbol: "stock" for symbol in requested},
            self.risk,
            previous=previous,
        )
        self.assertFalse(result["passed"])
        self.assertIn("max_daily_turnover", result["failure_reasons"])
        self.assertEqual(result["target_weights"], {})
        self.assertEqual(result["cash_weight"], 1.0)


class StrategyConfigTests(unittest.TestCase):
    def test_default_strategy_is_equal_score_and_opt_in_only(self) -> None:
        self.assertEqual(DEFAULT_STRATEGY, "equal_score")
        self.assertEqual(STRATEGIES, ("equal_score", "score_weighted"))
        self.assertEqual(DEFAULTS["research"]["weight_strategy"], "equal_score")


if __name__ == "__main__":
    unittest.main()
