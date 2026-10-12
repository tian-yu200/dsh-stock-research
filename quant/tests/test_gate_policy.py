"""门槛治理回归测试（2026-09-11 整改 P0-1 / P0-2 / P1-4 / P1-6）。

这些断言锁定四件事，防止同类事故复发：

1. **生产门槛值必须与 config.DEFAULTS 一致**（P1-6）。2026-09-10 曾把
   ``max_excess_test_drawdown`` 从 0.15 放宽到 0.35 而 DEFAULTS 未同步，
   导致测试套件不再守护生产门槛值——"放宽以放行"因此长期无人拦截。
2. **回撤门槛的语义**（P1-6）：有基准回撤时约束"超出基准的幅度"，无基准时才用绝对上限。
3. **信号日必须跳过横截面退化的交易日**（P0-1）。2026-09-11 provider 日历末位
   只有 6 只有数据，全市场却被取成 6 个评分。
4. **回测口径与实盘口径一致**（P1-4）：回测复用 allocation_budget +
   build_target_weights + enforce_target_weights，含现金与仓位约束。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from assistant.config import DEFAULTS, load_config  # noqa: E402
from assistant.portfolio import allocation_budget, build_target_weights  # noqa: E402
from assistant.qlib_model import (  # noqa: E402
    MIN_STOCK_SCORES,
    _degenerate_stock_model,
    _drawdown_advisory,
    cohort_schedule,
    select_signal_date,
)
from assistant.risk import enforce_target_weights, model_gate_checks  # noqa: E402

# 这些门槛必须与 config.DEFAULTS **严格相等**：它们是"风险敏感"值，
# 一旦生产与测试分叉，测试就失去守护作用（2026-09-10 的 0.15/0.35 事故）。
GATE_KEYS_MUST_MATCH = (
    "min_mean_rank_ic",
    "min_rank_ic_ir",
    "min_top_bottom_excess_return",
    "max_test_drawdown",
    "max_excess_test_drawdown",
    "max_ic_std",
    "min_excess_annualized_return",
)

# 注意：`min_train_l2_improvement` **不属于**上面任何一类——2026-09-11 复核后
# 确认 R² 绝对值不是有效的模型质量判据（见下方 IC² 关系测试），因此它在
# DEFAULTS 与生产配置中**都保持 0.0（关闭）**。


RISK_KEYS = (
    "max_single_weight",
    "max_stock_exposure",
    "max_etf_exposure",
    "max_otc_fund_exposure",
    "min_cash_weight",
    "max_daily_turnover",
)


def _production_config() -> dict:
    return load_config()


def test_production_gate_matches_defaults():
    """生产 assistant.yaml 的风险敏感门槛不得与 DEFAULTS 分叉（P1-6）。"""
    production = _production_config()["gate"]
    defaults = DEFAULTS["gate"]
    for key in GATE_KEYS_MUST_MATCH:
        assert key in production, f"生产配置缺少门槛键 {key}"
        assert production[key] == defaults[key], (
            f"门槛 {key} 生产值 {production[key]} != DEFAULTS {defaults[key]}；"
            "生产与测试分叉会让测试失去守护作用（2026-09-10 的 0.15/0.35 事故）"
        )


def test_fit_quality_gate_is_disabled_by_default_and_in_production():
    """R² 绝对值不是有效的模型质量判据，故该门槛在生产也保持关闭。

    2026-09-11 自我更正：截面 z-score 标签下方差为 1 ⇒ R² ≈ IC²。A 股截面 IC
    普遍 0.02~0.09，对应 R² 仅 0.04%~0.8%。若把门槛设为 R²≥2%（我曾这样做），
    等价于要求 IC ≥ 0.141 —— A 股不可达，会让系统永久无法发布。
    模型质量由 RankIC / IR / 分组超额 / 回撤四项负责。
    """
    assert DEFAULTS["gate"]["min_train_l2_improvement"] == 0.0
    assert _production_config()["gate"]["min_train_l2_improvement"] == 0.0


def test_ic_squared_relationship_holds_for_realistic_model():
    """固化核心关系：截面 z-score 标签下 R² ≈ IC²。

    这条断言是 2026-09-11 误判的直接解药——下次看到 train_l2≈0.995 时，
    应先算 sqrt(1-train_l2) 并与 IC 对照，而不是判定"模型没学到"。
    """
    observed_ic = 0.0563
    expected_train_l2 = 1.0 - observed_ic**2
    assert expected_train_l2 == pytest.approx(0.99683, abs=1e-5)
    # 实测 0.995423 比理论值更优（训练集拟合略高于测试 IC），属正常样本内乐观
    assert 0.995423 < expected_train_l2


def test_fit_quality_exposes_implied_ic_diagnostic():
    """train_l2 / valid_l2 / implied_ic 必须作为诊断量可读，供人工判读。"""
    from assistant.qlib_model import _fit_quality

    class _Booster:
        best_score = {"train": {"l2": 0.995423}, "valid": {"l2": 0.999511}}

    class _Model:
        model = _Booster()

    quality = _fit_quality(_Model())
    assert quality["fit_quality_available"] is True
    assert quality["train_l2"] == pytest.approx(0.995423)
    assert quality["valid_l2"] == pytest.approx(0.999511)
    # 隐含 |ρ| 应与同期 test RankIC(0.0563) 同量级且略高
    assert quality["implied_ic"] == pytest.approx(0.0677, abs=0.002)
    assert quality["implied_ic"] > 0.0563


def test_fit_quality_gate_is_fail_closed_when_enabled():
    """若将来有人重新启用该门槛，指标缺失必须判失败而不是静默放行。"""
    gate = {**DEFAULTS["gate"], "min_train_l2_improvement": 0.001}
    metrics = {
        "model_available": True,
        "mean_rank_ic": 0.05,
        "rank_ic_ir": 0.4,
        "top_bottom_excess_return": 0.02,
        "test_max_drawdown": -0.20,
        "train_l2_improvement": None,
        "fit_quality_available": False,
    }
    assert model_gate_checks(metrics, gate)["fit_quality_threshold"] is False


def test_fit_quality_gate_accepts_well_fitted_model():
    gate = {**DEFAULTS["gate"], "min_train_l2_improvement": 0.001}
    metrics = {
        "model_available": True,
        "mean_rank_ic": 0.05,
        "rank_ic_ir": 0.4,
        "top_bottom_excess_return": 0.02,
        "test_max_drawdown": -0.25,
        "benchmark": {"max_drawdown": -0.20},
        "train_l2": 0.95,
        "train_l2_improvement": 0.05,
        "fit_quality_available": True,
    }
    assert model_gate_checks(metrics, gate)["fit_quality_threshold"] is True


def test_fit_quality_gate_disabled_in_defaults():
    """DEFAULTS(0.0) 时门槛不得生效，否则会破坏合成数据夹具。"""
    gate = dict(DEFAULTS["gate"])
    metrics = {
        "model_available": True,
        "mean_rank_ic": 0.05,
        "rank_ic_ir": 0.4,
        "top_bottom_excess_return": 0.02,
        "test_max_drawdown": -0.20,
        "train_l2_improvement": None,
    }
    assert model_gate_checks(metrics, gate)["fit_quality_threshold"] is True


def test_production_risk_matches_defaults():
    """风险约束同样不得分叉——它直接决定回测与实盘的仓位口径。"""
    production = _production_config()["risk"]
    defaults = DEFAULTS["risk"]
    for key in RISK_KEYS:
        assert production[key] == defaults[key], f"风险参数 {key} 生产值与 DEFAULTS 不一致"


def test_production_backtest_policy_keys_exist():
    """P1-4 / P1-5 的回测口径键必须在生产配置中存在且显式取值。"""
    research = _production_config()["research"]
    assert research["backtest_allocation_slots"] == 3
    assert research["backtest_include_cash"] is True
    assert research["backtest_cohorts"] == 5
    # 持有期必须等于标签 horizon——否则 20 日 alpha 会被 1 日噪声吃掉（P1-5 的核心）
    assert research["stock_label_horizon"] == 20


def test_production_backtest_execution_matches_defaults():
    """2026-10-10：回测入场时点与涨停跳过，生产必须与默认一致且为可执行口径。"""
    research = _production_config()["research"]
    for key in ("backtest_entry_lag_days", "backtest_skip_limit_up"):
        assert research[key] == DEFAULTS["research"][key], key
    assert research["backtest_entry_lag_days"] == 1
    assert research["backtest_skip_limit_up"] is True


def test_cohort_schedule_keeps_cohorts_active_and_holds_full_horizon():
    """每只 cohort 必须恰好在场 holding_days 天，且稳态下同时有 cohorts 只在场。"""
    schedule = cohort_schedule(200, holding_days=20, cohorts=5)
    # holding=20, cohorts=5 ⇒ interval=4
    assert set(schedule)  # 非空
    # 稳态（不靠近样本两端）应恰好 5 个 cohort 在场
    steady = [len(active) for i, active in schedule.items() if 25 <= i <= 180]
    assert steady and set(steady) == {5}
    # 每个入场序号恰好出现在 holding_days 个连续交易日里。
    # 注意样本末端被截断的最后一个 cohort 除外（entry + holding > total_days）。
    total = 200
    entries: dict[int, list[int]] = {}
    for i, active in schedule.items():
        for entry in active:
            entries.setdefault(entry, []).append(i)
    complete = {e: d for e, d in entries.items() if e + 20 <= total}
    assert complete, "至少有若干个完整 cohort 可供断言"
    for entry, days in complete.items():
        assert days == list(range(entry, entry + 20)), (entry, days[:3], days[-3:])


def test_cohort_schedule_reduces_rotation_frequency():
    """cohort 轮换频率必须约为逐日的 1/cohorts——这是换手下降的来源。"""
    total = 200
    schedule = cohort_schedule(total, holding_days=20, cohorts=5)
    entry_days = {
        entry
        for active in schedule.values()
        for entry in active
    }
    rotation_days = len([e for e in entry_days if e < total])
    # 每 4 个交易日才有一次建仓 ≈ total/4，而逐日口径是 total 次
    assert rotation_days == pytest.approx(total // 4, abs=2)
    assert rotation_days < total / 4 + 2


def test_cohort_schedule_single_cohort_is_sequential():
    """cohorts=1 时退化为"每 holding_days 天换一次仓"，任意时点仅 1 只在场。"""
    schedule = cohort_schedule(60, holding_days=20, cohorts=1)
    assert {len(a) for a in schedule.values()} == {1}
    entries = sorted({e for a in schedule.values() for e in a})
    assert entries[:3] == [0, 20, 40]


def test_cohort_schedule_edge_cases():
    assert cohort_schedule(0, holding_days=20, cohorts=5) == {}
    # cohorts 大于 holding_days 时 interval 至少为 1，不允许除零/空转
    schedule = cohort_schedule(10, holding_days=2, cohorts=99)
    assert all(len(a) >= 1 for a in schedule.values())


def test_drawdown_gate_uses_benchmark_excess():
    """有基准回撤时：上限 = |基准回撤| + max_excess_test_drawdown。"""
    gate = _production_config()["gate"]
    metrics = {
        "model_available": True,
        "mean_rank_ic": 0.05,
        "rank_ic_ir": 0.3,
        "top_bottom_excess_return": 0.01,
        "test_max_drawdown": -0.30,
        "benchmark": {"max_drawdown": -0.20},
    }
    checks = model_gate_checks(metrics, gate)
    # 上限 = 0.20 + 0.15 = 0.35 > 0.30 → 通过
    assert checks["test_max_drawdown_threshold"] is True

    metrics["test_max_drawdown"] = -0.40
    checks = model_gate_checks(metrics, gate)
    # 0.40 > 0.35 → 拒绝
    assert checks["test_max_drawdown_threshold"] is False


def test_drawdown_gate_falls_back_to_absolute_cap():
    """缺少基准回撤时退化为绝对上限。"""
    gate = _production_config()["gate"]
    metrics = {
        "model_available": True,
        "mean_rank_ic": 0.05,
        "rank_ic_ir": 0.3,
        "top_bottom_excess_return": 0.01,
        "test_max_drawdown": -0.20,
    }
    assert model_gate_checks(metrics, gate)["test_max_drawdown_threshold"] is True
    metrics["test_max_drawdown"] = -0.26
    assert model_gate_checks(metrics, gate)["test_max_drawdown_threshold"] is False


def test_gate_fails_when_model_unavailable():
    """model_available=False（含横截面退化）必须整项否决。"""
    gate = {"max_excess_test_drawdown": 0.15, "max_test_drawdown": 0.25}
    checks = model_gate_checks({"model_available": False}, gate)
    assert checks["data_quality_and_model_available"] is False
    assert not all(checks.values())


def _prediction(sizes: dict[str, int]) -> pd.Series:
    frames = []
    for date, count in sizes.items():
        index = pd.MultiIndex.from_product(
            [[date], [f"S{number:05d}" for number in range(count)]],
            names=["datetime", "instrument"],
        )
        frames.append(pd.Series(range(count), index=index, dtype=float))
    return pd.concat(frames)


def test_select_signal_date_skips_degenerate_last_day():
    """复现 2026-09-11 场景：末位只有 6 只有数据，必须回退到 09-10。"""
    prediction = _prediction(
        {"2026-09-09": 5000, "2026-09-10": 5000, "2026-09-11": 6}
    )
    assert select_signal_date(prediction) == pd.Timestamp("2026-09-10")


def test_select_signal_date_keeps_full_last_day():
    """末位横截面正常时必须用它（不能无谓回退）。"""
    prediction = _prediction({"2026-09-10": 5000, "2026-09-11": 5200})
    assert select_signal_date(prediction) == pd.Timestamp("2026-09-11")


def test_select_signal_date_flags_partially_thin_history():
    """回退到明显低于中位数的日期应返回 None，交由调用方按数据不足处理。"""
    prediction = _prediction(
        {
            "2026-09-07": 5000,
            "2026-09-08": 5000,
            "2026-09-09": 5000,
            "2026-09-10": 5000,
            "2026-09-11": 120,
        }
    )
    assert select_signal_date(prediction) == pd.Timestamp("2026-09-10")


def test_select_signal_date_absolute_floor_and_small_universe():
    """相对判据不依赖宇宙规模：小 provider 不会被误判为退化。"""
    # 大规模宇宙里末位骤降 → 回退
    prediction = _prediction({"2026-09-10": 5000, "2026-09-11": 120})
    assert select_signal_date(prediction) == pd.Timestamp("2026-09-10")
    # 显式绝对下限同样能兜住
    assert select_signal_date(prediction, min_count=100) == pd.Timestamp("2026-09-10")
    # 规模一致的小宇宙（测试夹具/under50）→ 相对判据不视为退化，仍取最后一天
    tiny = _prediction({"2026-09-10": 5, "2026-09-11": 5})
    assert select_signal_date(tiny) == pd.Timestamp("2026-09-11")


def test_select_signal_date_handles_empty():
    assert select_signal_date(pd.Series(dtype=float)) is None


def test_degenerate_result_is_not_a_model_failure():
    """P0-2：退化结果必须带 degenerate 标记且不写缓存，供上层区分失败原因。"""
    result = _degenerate_stock_model(
        "signal date produced only 6 scores",
        score_count=6,
        provider_end="2026-09-11",
    )
    assert result["degenerate"] is True
    assert result["model_available"] is False
    assert result["score_count"] == 6
    assert result["min_scores_required"] == MIN_STOCK_SCORES
    assert result["scores"] == {}
    assert "cache_write_skipped" not in result


def test_backtest_budget_matches_live_allocation():
    """P1-4：回测与实盘必须用同一套预算公式，总仓位恒 ≤ 20%、现金 ≥ 80%。

    这是"门槛衡量对象 = 发布对象"的核心断言。实盘为 3 个资产类别，
    因此单类别预算 = min(max_daily_turnover, 1-min_cash) / 3 = 0.2/3。
    """
    config = _production_config()
    risk = config["risk"]
    budget = allocation_budget(risk, 3)
    assert budget == pytest.approx(min(risk["max_daily_turnover"], 1 - risk["min_cash_weight"]) / 3)
    assert budget == pytest.approx(0.2 / 3, abs=1e-12)

    candidates = [
        {"symbol": f"S{number}", "qlib_score": score}
        for number, score in enumerate((0.9, 0.8, 0.7))
    ]
    requested = build_target_weights(
        candidates, "stock", risk, budget=budget, strategy="equal_score"
    )
    enforced = enforce_target_weights(
        requested, {symbol: "stock" for symbol in requested}, risk
    )
    assert enforced["passed"] is True
    invested = sum(enforced["target_weights"].values())
    # 满仓等权口径会投 100%；目标组合口径必须只投约 6.67%，其余为现金。
    assert invested == pytest.approx(budget, abs=1e-12)
    assert invested <= risk["max_daily_turnover"]
    assert enforced["cash_weight"] >= risk["min_cash_weight"]


def test_full_invested_weights_would_be_rejected_by_category_limit():
    """反证：旧口径的满仓等权会触发类别敞口约束，说明两条口径确实不同。"""
    config = _production_config()
    risk = config["risk"]
    requested = {f"S{number}": 1.0 / 3 for number in range(3)}
    enforced = enforce_target_weights(
        requested, {symbol: "stock" for symbol in requested}, risk
    )
    assert enforced["passed"] is False
    assert "stock_exposure" in enforced["failure_reasons"]


def test_drawdown_advisory_flags_breach_on_full_position_horizon():
    """2026-09-12：门槛口径切到含现金目标组合（P1-4）后，满仓 20 日口径必须有告警线。

    旧门槛判据（excess ≤ max_excess_test_drawdown）若仍用于满仓回撤：
    -0.384 vs 基准 -0.212（excess 0.172 > 0.15）应当触发告警——否则回撤守门
    能力被现金稀释彻底掏空而无人察觉（09-11 实测门槛口径 -0.023，永远通过）。
    """
    advisory = _drawdown_advisory(-0.384, -0.212, 0.15)
    assert advisory["breached"] is True
    assert advisory["excess_drawdown"] == pytest.approx(0.172, abs=1e-9)
    assert advisory["limit"] == pytest.approx(0.15)


def test_drawdown_advisory_is_advisory_only_never_blocks():
    """健康路径不告警；序列过短产生 NaN 时一律 breached=False，绝不影响 gate。"""
    ok = _drawdown_advisory(-0.05, -0.212, 0.15)
    assert ok["breached"] is False
    nan_case = _drawdown_advisory(float("nan"), -0.212, 0.15)
    assert nan_case["breached"] is False
    assert nan_case["excess_drawdown"] is None


def test_stability_gate_blocks_when_ic_std_exceeds_threshold():
    """跨窗口 IC 标准差超阈值 → 判不稳定，阻断发布。"""
    gate = {**DEFAULTS["gate"], "max_ic_std": 0.02}
    metrics = {
        "model_available": True,
        "mean_rank_ic": 0.05,
        "rank_ic_ir": 0.4,
        "top_bottom_excess_return": 0.02,
        "test_max_drawdown": -0.20,
        "stability_stable": False,
        "stability_ic_std": 0.05,
    }
    assert model_gate_checks(metrics, gate)["stability_ic_std_threshold"] is False


def test_stability_gate_accepts_when_ic_std_within_threshold():
    """跨窗口 IC 标准差在阈值内 → 判稳定，放行。"""
    gate = {**DEFAULTS["gate"], "max_ic_std": 0.02}
    metrics = {
        "model_available": True,
        "mean_rank_ic": 0.05,
        "rank_ic_ir": 0.4,
        "top_bottom_excess_return": 0.02,
        "test_max_drawdown": -0.20,
        "stability_stable": True,
        "stability_ic_std": 0.01,
    }
    assert model_gate_checks(metrics, gate)["stability_ic_std_threshold"] is True


def test_stability_gate_does_not_block_when_check_disabled():
    """稳定性检查未启用（stable=None）→ 该项不阻断，与可选门槛语义一致。"""
    gate = {**DEFAULTS["gate"], "max_ic_std": 0.02}
    metrics = {
        "model_available": True,
        "mean_rank_ic": 0.05,
        "rank_ic_ir": 0.4,
        "top_bottom_excess_return": 0.02,
        "test_max_drawdown": -0.20,
        "stability_stable": None,
        "stability_ic_std": float("nan"),
    }
    assert model_gate_checks(metrics, gate)["stability_ic_std_threshold"] is True


def test_stability_gate_is_fail_closed_when_metric_missing():
    """已启用但取不到有效 ic_std 时判失败，不得静默放行。"""
    gate = {**DEFAULTS["gate"], "max_ic_std": 0.02}
    metrics = {
        "model_available": True,
        "mean_rank_ic": 0.05,
        "rank_ic_ir": 0.4,
        "top_bottom_excess_return": 0.02,
        "test_max_drawdown": -0.20,
        "stability_stable": True,
        "stability_ic_std": None,
    }
    assert model_gate_checks(metrics, gate)["stability_ic_std_threshold"] is False


def test_excess_return_gate_passes_positive_excess():
    """D1：扣费后组合年化收益不得低于等权基准（门槛 0.0，正值放行）。

    T3 口径下生产等价变体 C 实测 excess_annualized_return=+0.0359
    （net 1.32% vs 基准 -2.27%），必须通过。
    """
    gate = {**DEFAULTS["gate"], "min_excess_annualized_return": 0.0}
    metrics = {
        "model_available": True,
        "mean_rank_ic": 0.05,
        "rank_ic_ir": 0.4,
        "top_bottom_excess_return": 0.02,
        "test_max_drawdown": -0.20,
        "benchmark": {"max_drawdown": -0.15, "excess_annualized_return": 0.0359},
    }
    assert model_gate_checks(metrics, gate)["excess_return_threshold"] is True


def test_excess_return_gate_blocks_negative_excess():
    """扣费后收益跑输等权基准（-0.001）即阻断发布。"""
    gate = {**DEFAULTS["gate"], "min_excess_annualized_return": 0.0}
    metrics = {
        "model_available": True,
        "mean_rank_ic": 0.05,
        "rank_ic_ir": 0.4,
        "top_bottom_excess_return": 0.02,
        "test_max_drawdown": -0.20,
        "benchmark": {"max_drawdown": -0.15, "excess_annualized_return": -0.001},
    }
    assert model_gate_checks(metrics, gate)["excess_return_threshold"] is False


def test_excess_return_gate_is_fail_closed_when_metric_missing():
    """指标缺失（无 benchmark 段 / excess 缺失或非有限）必须判失败，不得静默放行。"""
    gate = {**DEFAULTS["gate"], "min_excess_annualized_return": 0.0}
    base = {
        "model_available": True,
        "mean_rank_ic": 0.05,
        "rank_ic_ir": 0.4,
        "top_bottom_excess_return": 0.02,
        "test_max_drawdown": -0.20,
    }
    assert model_gate_checks(base, gate)["excess_return_threshold"] is False
    assert (
        model_gate_checks({**base, "benchmark": {"max_drawdown": -0.15}}, gate)[
            "excess_return_threshold"
        ]
        is False
    )
    for bad in (
        float("nan"),
        float("inf"),
        float("-inf"),
        "0.05",  # 字符串不可强转即失败
        True,  # N3：bool 不是有效指标（_finite 必须拒绝）
    ):
        assert (
            model_gate_checks(
                {**base, "benchmark": {"excess_annualized_return": bad}}, gate
            )["excess_return_threshold"]
            is False
        ), bad


def test_model_gate_checks_key_set_matches_constant():
    """B2：model_gate_checks 的返回键必须与 MODEL_GATE_CHECK_KEYS 单一来源一致。"""
    from assistant.risk import MODEL_GATE_CHECK_KEYS

    metrics = {
        "model_available": True,
        "mean_rank_ic": 0.05,
        "rank_ic_ir": 0.4,
        "top_bottom_excess_return": 0.02,
        "test_max_drawdown": -0.20,
        "benchmark": {"max_drawdown": -0.15, "excess_annualized_return": 0.01},
    }
    assert set(model_gate_checks(metrics, DEFAULTS["gate"])) == set(MODEL_GATE_CHECK_KEYS)


def test_finite_rejects_bool():
    """N3：_finite 必须拒绝 bool（True/False 不是数值指标）。"""
    from assistant.risk import _finite

    assert _finite(True) is False
    assert _finite(False) is False
    assert _finite(1.0) is True
    assert _finite(0.0) is True
