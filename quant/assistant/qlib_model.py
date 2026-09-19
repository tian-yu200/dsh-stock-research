"""Qlib 训练与模型指标层 + 共享训练入口（docs/重构计划.md 阶段 1/3）。

阶段 3 起，provider 初始化（init_qlib）、数据集构建（build_dataset）、
LGBModel 训练（train_lgbm）与分段预测（segment_prediction）是主路径与
run_under50_research.py 共享的训练层；Qlib 的导入仍延迟到各函数内部，
保证未安装可选依赖时 ETF/基金快照链路可用。定义在 research.py 的工具
（_iso_date）经 ``research.`` 属性访问；其余名字为模块内自有或来自 paths。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import research
from .paths import atomic_write_json, read_json, validate_symbol
from .portfolio import allocation_budget, build_target_weights
from .provider_lock import ProviderLock
from .risk import enforce_target_weights


# 2026-09-11 整改：v2 = 回测口径改为目标组合（含现金）+ 信号日改为横截面覆盖达标日；
# v3 = 持有期与标签 horizon 对齐（重叠 cohort，P1-5）。
# 每次改变模型缓存承载的指标语义都必须 bump，否则旧缓存会被复用并掩盖修复效果
# （不变量 §3b 的教训）。
MODEL_CACHE_SCHEMA = 3

# 全市场股票模型的最小有效评分数。低于该值说明当日横截面退化
# （典型成因：provider 日历末位只被极少数标的填充），此时既不能写缓存，
# 也不能把结果当作"模型不合格"来判定门槛。
MIN_STOCK_SCORES = 100


def _report_progress(
    callback: Callable[..., None] | None,
    phase: str,
    **details: Any,
 ) -> None:
    if callback is not None:
        callback(phase, **details)


def _provider_source_fingerprint(provider: Path) -> dict[str, Any] | None:
    manifest = read_json(provider / "manifest.json", {})
    if not isinstance(manifest, dict):
        return None
    fingerprint = manifest.get("source_fingerprint")
    return dict(fingerprint) if isinstance(fingerprint, dict) else None


def _model_cache_identity(
    config: Mapping[str, Any],
    *,
    provider_fingerprint: Mapping[str, Any],
    provider_end: str,
 ) -> tuple[str, dict[str, Any]]:
    identity = {
        "schema": MODEL_CACHE_SCHEMA,
        "provider_source_fingerprint": dict(provider_fingerprint),
        "provider_end": str(provider_end),
        "dataset": dict(config["dataset"]),
        "model": dict(config["model"]),
        "research": {
            key: config["research"][key]
            for key in (
                "stock_label_horizon",
                "stock_topk",
                "transaction_cost_bps",
                # 2026-09-10 回撤根治：BJ 过滤改变回测组合构成，
                # 必须进缓存键，否则旧缓存复用掩盖修复效果。
                "pool_exclude_bj",
                # 2026-09-11 整改(P1-4)：回测目标组合的预算口径。
                "backtest_allocation_slots",
                "backtest_include_cash",
                # 2026-09-11 整改(P1-5)：持有期/cohort 数改变回测语义。
                "backtest_cohorts",
                # 2026-09-11 整改(P0-2)：信号日退化的绝对下限会影响 scores 集合，
                # 必须进缓存键，否则改它可能复用旧缓存而看不出差异。
                "min_signal_scores",
            )
            if key in config["research"]
        },
    }
    canonical = json.dumps(identity, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest(), identity


def _model_cache_path(config: Mapping[str, Any], cache_key: str) -> Path:
    return (
        Path(config["paths"]["results_dir"])
        / "model-cache"
        / "qlib-alpha158-lightgbm"
        / f"{cache_key}.json"
    )


def _cached_model_result(path: Path, cache_key: str) -> dict[str, Any] | None:
    cached = read_json(path, {})
    if (
        not isinstance(cached, dict)
        or cached.get("schema") != MODEL_CACHE_SCHEMA
        or cached.get("cache_key") != cache_key
        or not isinstance(cached.get("result"), dict)
        or not cached["result"].get("model_available")
    ):
        return None
    return dict(cached["result"])


def cohort_schedule(
    total_days: int,
    *,
    holding_days: int,
    cohorts: int,
) -> dict[int, tuple[int, ...]]:
    """重叠组合（overlapping cohorts）排程：``{日期序号: 当日活跃的入场序号}``。

    背景（P1-5）：模型预测的是**未来 ``holding_days`` 个交易日**的相对收益，
    而原回测按 1 日持有、每日重选 top-k：

    - 20 日 alpha 被 1 日噪声抵消，且 A 股短期反转会让高动量高分股次日系统回吐；
    - 日换手 51.2% × 10bp × 250 ≈ **12.8%/年** 成本。

    改为把资金等分为 ``cohorts`` 个 cohort，每 ``holding_days // cohorts`` 个交易日
    有一个 cohort 建仓并持有 ``holding_days`` 个交易日；任意时点同时有 ``cohorts``
    个 cohort 在场，因此换手降到约 ``1/cohorts``，持有期与标签 horizon 一致。

    纯函数，便于单测：不依赖行情，只依赖天数。
    """
    holding = max(1, int(holding_days))
    count = max(1, int(cohorts))
    interval = max(1, holding // count)
    total = max(0, int(total_days))
    entry_indices = list(range(0, total, interval)) if total else []
    schedule: dict[int, tuple[int, ...]] = {}
    for index in range(total):
        active = tuple(
            entry for entry in entry_indices if entry <= index < entry + holding
        )
        if active:
            schedule[index] = active
    return schedule


def _check_label_leakage(
    calendar: pd.DatetimeIndex,
    fit_end_time: str,
    label_horizon: int,
    valid_start: str,
) -> dict[str, Any]:
    """
    检查标签窗口是否跨越 fit_end_time 进入 valid 段，防止数据泄漏。
    
    训练集最后一天是 fit_end_time，其标签需要未来 label_horizon 个交易日的价格。
    若 fit_end_time + label_horizon 个交易日 > valid_start，则训练集末尾样本的标签
    使用了 valid 期的价格，构成泄漏。
    
    Args:
        calendar: 交易日历（qlib provider 的 D.calendar(freq="day")）
        fit_end_time: 训练集结束日期（YYYY-MM-DD 字符串）
        label_horizon: 标签前瞻天数（交易日）
        valid_start: valid 段起始日期（YYYY-MM-DD 字符串）
    
    Returns:
        {"passed": bool, "label_horizon_days": int, "reason": str | None}
    """
    try:
        # D.calendar(freq="day") 返回 numpy.ndarray，没有 get_loc；
        # 统一转成 DatetimeIndex 后再做精确查找（保持 KeyError 时 fail-closed 语义）。
        if not isinstance(calendar, pd.DatetimeIndex):
            calendar = pd.DatetimeIndex(calendar)
        fit_end_ts = pd.Timestamp(fit_end_time)
        valid_start_ts = pd.Timestamp(valid_start)
        
        # 找到 fit_end_time 在日历中的位置
        fit_end_loc = calendar.get_loc(fit_end_ts)
        
        # 往后推 label_horizon 个交易日
        label_window_end_loc = fit_end_loc + label_horizon
        if label_window_end_loc >= len(calendar):
            # 标签窗口超出日历末尾，无法判定（理论上不应发生）
            return {
                "passed": False,
                "label_horizon_days": label_horizon,
                "reason": f"label window (fit_end + {label_horizon} days) exceeds calendar length",
            }
        
        label_window_end_ts = calendar[label_window_end_loc]
        
        # 检查标签窗口是否进入 valid 段
        if label_window_end_ts >= valid_start_ts:
            return {
                "passed": False,
                "label_horizon_days": label_horizon,
                "reason": (
                    f"label window crosses into valid segment: "
                    f"fit_end={fit_end_time}, label_window_end={label_window_end_ts.date().isoformat()}, "
                    f"valid_start={valid_start}"
                ),
            }
        
        return {
            "passed": True,
            "label_horizon_days": label_horizon,
            "reason": None,
        }
    except Exception as exc:
        # 日历查找失败等异常情况，fail-closed
        return {
            "passed": False,
            "label_horizon_days": label_horizon,
            "reason": f"leakage check failed: {type(exc).__name__}: {exc}",
        }


def _filter_scores(
    result: Mapping[str, Any],
    requested_symbols: list[str] | None,
) -> dict[str, Any]:
    filtered = dict(result)
    scores = result.get("scores", {})
    if not isinstance(scores, Mapping):
        scores = {}
    normalized_scores = {validate_symbol(str(symbol)): float(score) for symbol, score in scores.items()}
    if requested_symbols is not None:
        requested = {validate_symbol(symbol) for symbol in requested_symbols}
        normalized_scores = {
            symbol: score
            for symbol, score in normalized_scores.items()
            if symbol in requested
        }
    filtered["scores"] = normalized_scores
    return filtered

def _prediction_metrics(prediction: pd.Series, labels: pd.Series) -> dict[str, float | int]:
    frame = pd.concat({"score": prediction, "label": labels}, axis=1).dropna()
    if frame.empty:
        return {
            "observations": 0,
            "candidate_days": 0,
            "valid_days": 0,
            "excluded_days": 0,
            "mean_rank_ic": float("nan"),
            "rank_ic_ir": float("nan"),
        }
    grouped = frame.groupby(level="datetime")
    candidate_days = int(grouped.ngroups)
    daily_values: dict[pd.Timestamp, float] = {}
    for date, data in grouped:
        if len(data) < 2 or data["score"].nunique() < 2 or data["label"].nunique() < 2:
            continue
        rank_ic = data["score"].corr(data["label"], method="spearman")
        if pd.notna(rank_ic):
            daily_values[pd.Timestamp(date)] = float(rank_ic)
    daily = pd.Series(daily_values, dtype=float).sort_index()
    rank_std = daily.std(ddof=1)
    return {
        "observations": int(len(frame)),
        "candidate_days": candidate_days,
        "valid_days": int(len(daily)),
        "excluded_days": candidate_days - int(len(daily)),
        "mean_rank_ic": float(daily.mean()) if not daily.empty else float("nan"),
        "rank_ic_ir": float(daily.mean() / rank_std)
        if len(daily) > 1 and rank_std and not np.isnan(rank_std)
        else float("nan"),
    }


def _forward_group_metrics(
    prediction: pd.Series,
    labels: pd.Series,
    group_count: int = 5,
) -> dict[str, Any]:
    """Measure forward returns by score group without using future information."""
    frame = pd.concat({"score": prediction, "label": labels}, axis=1).dropna()
    if frame.empty or group_count < 2:
        return {
            "group_count": max(1, int(group_count)),
            "observations": int(len(frame)),
            "valid_days": 0,
            "group_returns": {},
            "top_group_return": float("nan"),
            "bottom_group_return": float("nan"),
            "top_bottom_excess_return": float("nan"),
        }

    group_returns: dict[int, list[float]] = {}
    top_bottom: list[float] = []
    valid_days = 0
    for _, data in frame.groupby(level="datetime"):
        if len(data) < 2 or data["score"].nunique() < 2:
            continue
        number_of_groups = min(int(group_count), len(data))
        ordered = data.sort_values("score").reset_index(drop=True)
        group_ids = (
            np.floor(np.arange(len(ordered)) * number_of_groups / len(ordered)).astype(int)
            + 1
        )
        ordered["group"] = group_ids
        daily = ordered.groupby("group")["label"].mean()
        if len(daily) < 2:
            continue
        valid_days += 1
        for group, value in daily.items():
            group_returns.setdefault(int(group), []).append(float(value))
        top_bottom.append(float(daily.iloc[-1] - daily.iloc[0]))

    aggregated = {
        str(group): research._number(np.mean(values))
        for group, values in sorted(group_returns.items())
        if values
    }
    top_return = aggregated.get(str(max(group_returns)), float("nan")) if group_returns else float("nan")
    bottom_return = aggregated.get(str(min(group_returns)), float("nan")) if group_returns else float("nan")
    return {
        "group_count": int(group_count),
        "observations": int(len(frame)),
        "valid_days": valid_days,
        "group_returns": aggregated,
        "top_group_return": top_return,
        "bottom_group_return": bottom_return,
        "top_bottom_excess_return": research._number(np.mean(top_bottom)) if top_bottom else float("nan"),
    }


def _performance(equity: pd.Series) -> dict[str, float]:
    values = equity.dropna()
    returns = values.pct_change().dropna()
    if len(values) < 2 or returns.empty:
        return {"net_annualized_return": float("nan"), "max_drawdown": float("nan")}
    if isinstance(values.index, pd.DatetimeIndex):
        elapsed_days = max((values.index[-1] - values.index[0]).days, 1)
        years = elapsed_days / 365.25
    else:
        years = max(len(returns) / 252.0, 1 / 252.0)
    annualized = float((values.iloc[-1] / values.iloc[0]) ** (1.0 / years) - 1.0)
    drawdown = values / values.cummax() - 1.0
    return {"net_annualized_return": annualized, "max_drawdown": float(drawdown.min())}


def select_signal_date(
    prediction: pd.Series,
    *,
    min_count: int = 0,
    min_ratio: float = 0.5,
) -> pd.Timestamp | None:
    """返回横截面被充分填充的最后一个交易日。

    Qlib provider 的日历是所有标的日期的**并集**，因此日历末位可能只被极少数标的
    填充（典型成因：只有部分数据源在当日收盘前发布了当日 K 线）。若直接取该日作为
    信号日，排名会退化到个位数标的——实证：2026-09-11 全市场 5562 只产出 6 个评分。

    判定标准是**相对**的：样本数 ≥ max(min_count, 中位数 × min_ratio)。相对判据
    不依赖宇宙规模，因此对全市场（5000+）与小规模 provider（测试夹具、under50）
    同样成立；``min_count`` 只是可选绝对下限，默认 0（不启用）。

    没有任何交易日达标时返回 None，由调用方按"数据不足"处理（而非"模型不合格"）。
    """
    if prediction is None or len(prediction) == 0:
        return None
    sizes = prediction.groupby(level="datetime").size().sort_index()
    if sizes.empty:
        return None
    threshold = max(
        int(min_count),
        int(math.ceil(float(sizes.median()) * float(min_ratio))),
    )
    eligible = sizes[sizes >= threshold]
    if eligible.empty:
        return None
    return pd.Timestamp(eligible.index[-1])


def _drawdown_advisory(
    horizon_drawdown: float,
    benchmark_drawdown: float,
    excess_limit: float,
) -> dict[str, Any]:
    """满仓 20 日口径的 excess-drawdown 告警（**非阻断**，只记录、不判 gate）。

    2026-09-12 整改：``test_max_drawdown`` 门槛（P1-4 之后）衡量的是含现金目标
    组合，回撤被 ~95% 现金稀释（09-11 实测 -0.023 vs 上限 0.15），几乎不可能
    触发——守门能力被口径切换掏空。这里用与旧门槛**同一条判据**
    （excess ≤ gate.max_excess_test_drawdown）监控满仓 20 日
    ``horizon_matched_backtest`` 口径：超限时记入 model_metadata 供台账判读，
    但不改变任何 gate 的 passed 结果。
    """

    def _finite(value: Any) -> bool:
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    limit = float(excess_limit) if _finite(excess_limit) else 0.15
    if not (_finite(horizon_drawdown) and _finite(benchmark_drawdown)):
        return {"breached": False, "excess_drawdown": None, "limit": limit}
    excess = abs(float(horizon_drawdown)) - abs(float(benchmark_drawdown))
    return {
        "breached": bool(excess > limit),
        "excess_drawdown": float(excess),
        "limit": limit,
    }


def _degenerate_stock_model(
    reason: str,
    *,
    score_count: int,
    provider_end: str,
) -> dict[str, Any]:
    """横截面退化（数据不足）时的模型结果。

    与"模型执行失败"刻意区分：这是**数据/日历对齐**问题，不是模型能力问题。
    因此标记 ``degenerate=True``，不写缓存，并让上层记入独立的失败原因
    （而不是 ``stock_model_or_risk_gate_failed``），以免污染调度器的 failure_streak。
    """
    return {
        "model_available": False,
        "model_name": "Qlib Alpha158 + LightGBM",
        "data_end_date": provider_end,
        "degenerate": True,
        "degenerate_reason": reason,
        "score_count": int(score_count),
        "min_scores_required": MIN_STOCK_SCORES,
        "error": f"degenerate_signal_cross_section: {reason}",
        "mean_rank_ic": float("nan"),
        "test_rank_ic": float("nan"),
        "rank_ic_ir": float("nan"),
        "top_bottom_excess_return": float("nan"),
        "net_annualized_return": float("nan"),
        "test_max_drawdown": float("nan"),
        "max_drawdown": float("nan"),
        "scores": {},
        "leakage_check": {"passed": False, "reason": reason},
    }


def _fit_quality(model: Any) -> dict[str, Any]:
    """提取训练/验证 L2 与隐含 IC，作为模型诊断量（**不作为门槛默认值**）。

    标签是截面 z-score（方差≈1），因此 ``R² = 1 - train_l2 ≈ IC²``，
    故 ``implied_ic = sqrt(1 - train_l2)`` 可与 test RankIC 直接对照。

    2026-09-11 的原始判断（"train_l2≈0.995 说明模型什么都没学到"）是**错误的**，
    此处保留更正记录，避免后人重犯：

    - 实测 ``train_l2=0.995423`` → 隐含 |ρ| = 0.0677；同期 ``test mean_rank_ic`` = 0.0563。
      训练集拟合略高于测试 IC，是正常的轻度样本内乐观，**完全自洽**。
    - 理论对照：IC=0.0563 → R²=0.00317 → train_l2 应约 0.9968（实测 0.9954，更优）。
    - 5 分组标签均值严格单调（−0.0355 → −0.0193 → −0.0099 → +0.0166 → +0.0488），
      破损/常数模型不可能产生等距单调的分组排序。
    - 因此 ``train_l2 ≈ 0.995`` 是股票截面 alpha 模型的**正常值**：A 股 IC 通常
      0.02~0.09，对应 R² 仅 0.04%~0.8%。

    结论：**不要用 R² 的绝对值做门槛**（要求 R²≥2% 等于要求 IC≥0.141，A 股不可达）。
    模型质量由既有的 RankIC / RankIC IR / 分组超额 / 回撤四项门槛负责。
    """
    booster = getattr(model, "model", None)
    best_score = getattr(booster, "best_score", None)
    if not isinstance(best_score, Mapping):
        return {
            "fit_quality_available": False,
            "train_l2": None,
            "valid_l2": None,
            "train_l2_improvement": None,
        }

    def _l2(key: str) -> float | None:
        value = best_score.get(key)
        if isinstance(value, Mapping):
            value = value.get("l2")
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    train_l2 = _l2("train")
    valid_l2 = _l2("valid")
    improvement = None if train_l2 is None else max(0.0, 1.0 - train_l2)
    return {
        "fit_quality_available": train_l2 is not None,
        "train_l2": train_l2,
        "valid_l2": valid_l2,
        "train_l2_improvement": improvement,
        # 2026-09-11 自我更正：截面 z-score 标签（方差=1）下 R² ≈ IC²，
        # 因此隐含 |ρ| = sqrt(1 - train_l2)，可与 test IC 直接对照。
        # 实证：train_l2=0.9954 → 隐含 |ρ|=0.0677，同期 test IC=0.0563 —— 自洽。
        # 结论：train_l2≈0.995 对股票 alpha 模型是**正常值**（A 股截面 IC 通常
        # 0.02~0.09，对应 R² 0.04%~0.8%），绝不能当成"模型没学到"。
        "implied_ic": (
            None if improvement is None else float(math.sqrt(improvement))
        ),
    }


def _turnover(
    current: Mapping[str, float],
    previous: Mapping[str, float],
) -> float:
    """单边换手率：Σ|Δw| / 2（含现金由调用方按 1-Σw 处理）。"""
    symbols = set(current) | set(previous)
    return sum(
        abs(float(current.get(symbol, 0.0)) - float(previous.get(symbol, 0.0)))
        for symbol in symbols
    ) / 2.0


def _fit_qlib_model_and_close_recorder(
    model: Any,
    dataset: Any,
    qlib_recorder: Any,
    **fit_kwargs: Any,
) -> None:
    """Close Qlib's global recorder after model.fit, including failed fits."""
    try:
        model.fit(dataset, **fit_kwargs)
    finally:
        qlib_recorder.end_exp()


def init_qlib(provider_uri: str | Path, *, kernels: int = 1) -> Path:
    """qlib.init on a fixed provider directory; returns the resolved path.

    共享训练层入口（docs/重构计划.md 阶段 3）：主路径与 under50 脚本都以
    固定 provider 目录初始化，kernels 按各自原值传入。
    """
    import qlib
    from qlib.config import REG_CN

    provider = Path(provider_uri).resolve()
    if not provider.exists():
        raise FileNotFoundError(f"Qlib provider does not exist: {provider}")
    qlib.init(provider_uri=str(provider), region=REG_CN, kernels=kernels)
    return provider


class _SingleQueryQlibDataLoader:
    """Load Alpha158 fields and auxiliary backtest fields in one provider query.

    Qlib's stock loader normally calls ``D.features`` once per field group.
    Keeping the auxiliary close in a separate group preserves the model's
    feature/label contract while this loader shares the underlying query.
    """

    def __init__(
        self,
        field_groups: Mapping[str, tuple[list[str], list[str]]],
        *,
        filter_pipe: Any = None,
        freq: str = "day",
        inst_processors: Any = None,
        data_api: Any = None,
    ) -> None:
        self.fields = {
            group: (list(exprs), list(names))
            for group, (exprs, names) in field_groups.items()
        }
        for group, (exprs, names) in self.fields.items():
            if len(exprs) != len(names):
                raise ValueError(f"field/name count mismatch for Qlib group {group!r}")
        self.filter_pipe = filter_pipe
        self.freq = freq
        self.inst_processors = [] if inst_processors is None else inst_processors
        self._data_api = data_api

    def _api(self) -> Any:
        if self._data_api is not None:
            return self._data_api
        from qlib.data import D

        return D

    def load(
        self,
        instruments: Any = None,
        start_time: str | pd.Timestamp | None = None,
        end_time: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        data_api = self._api()
        query_instruments = instruments
        if isinstance(query_instruments, str):
            query_instruments = data_api.instruments(
                query_instruments,
                filter_pipe=self.filter_pipe,
            )

        fields: list[str] = []
        for exprs, _ in self.fields.values():
            for expression in exprs:
                if expression not in fields:
                    fields.append(expression)

        raw = data_api.features(
            query_instruments,
            fields,
            start_time,
            end_time,
            freq=self.freq,
            inst_processors=self.inst_processors,
        )
        if len(raw.columns) != len(fields):
            raise ValueError(
                "Qlib returned an unexpected field count: "
                f"expected {len(fields)}, got {len(raw.columns)}"
            )

        # D.features returns (instrument, datetime), matching QlibDataLoader's
        # per-group normalization before it concatenates the groups.
        raw = raw.copy()
        raw.columns = fields
        raw = raw.swaplevel().sort_index()
        groups = {}
        for group, (exprs, names) in self.fields.items():
            selected = raw.loc[:, exprs].copy()
            selected.columns = names
            groups[group] = selected
        return pd.concat(groups, axis=1)


def build_dataset(
    *,
    instruments: Any,
    start_time: str,
    end_time: str,
    fit_start_time: str,
    fit_end_time: str,
    label_horizon: int,
    segments: Mapping[str, Any],
    backtest_fields: Mapping[str, str] | None = None,
) -> Any:
    """Alpha158 handler + DatasetH：两条训练路径共享的同一数据集构建。"""
    from qlib.contrib.data.handler import Alpha158
    from qlib.data.dataset import DatasetH

    # Build the normal handler without loading first so the feature, label,
    # and backtest groups can share one Qlib provider query.
    handler = Alpha158(
        instruments=instruments,
        start_time=start_time,
        end_time=end_time,
        fit_start_time=fit_start_time,
        fit_end_time=fit_end_time,
        label=([f"Ref($close, -{int(label_horizon) + 1})/Ref($close, -1) - 1"], ["LABEL0"]),
        infer_processors=[
            {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
            {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
        ],
        learn_processors=[
            {"class": "DropnaLabel"},
            {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
        ],
        init_data=False,
    )
    field_groups = dict(handler.data_loader.fields)
    selected_backtest_fields = (
        {"$close": "BACKTEST_CLOSE"}
        if backtest_fields is None
        else dict(backtest_fields)
    )
    field_groups["backtest"] = (
        list(selected_backtest_fields),
        list(selected_backtest_fields.values()),
    )
    original_inst_processors = getattr(handler.data_loader, "inst_processors", None)
    if isinstance(original_inst_processors, dict):
        original_inst_processors = []
    handler.data_loader = _SingleQueryQlibDataLoader(
        field_groups,
        filter_pipe=getattr(handler.data_loader, "filter_pipe", None),
        freq=getattr(handler.data_loader, "freq", "day"),
        inst_processors=original_inst_processors,
    )
    handler.setup_data()
    return DatasetH(handler=handler, segments=segments)


def train_lgbm(
    dataset: Any,
    model_config: Mapping[str, Any],
    *,
    verbose_eval: int | None = None,
) -> Any:
    """LGBModel 训练并关闭全局 recorder；under50 传入 verbose_eval 保持原输出。"""
    from qlib.contrib.model.gbdt import LGBModel
    from qlib.workflow import R

    seed = model_config.get("seed", 42)
    # 统一全局随机源；LightGBM 自身由 seed/model_config 中的 seed 控制。
    random.seed(seed)
    np.random.seed(seed)
    model_config = {**model_config, "seed": seed}
    model = LGBModel(**model_config)
    if verbose_eval is None:
        _fit_qlib_model_and_close_recorder(model, dataset, R)
    else:
        evals_result: dict = {}
        _fit_qlib_model_and_close_recorder(
            model,
            dataset,
            R,
            evals_result=evals_result,
            verbose_eval=verbose_eval,
        )
    return model


def segment_prediction(dataset: Any, model: Any, segment: str) -> tuple[pd.Series, pd.Series]:
    """单个段的预测分数与学习标签（两条路径的原逐字实现）。"""
    from qlib.data.dataset.handler import DataHandlerLP

    prediction = model.predict(dataset, segment=segment).dropna().sort_index()
    prediction.name = "score"
    labels = dataset.prepare(segment, col_set="label", data_key=DataHandlerLP.DK_L).iloc[:, 0]
    return prediction, labels


def multi_window_stability_check(
    dataset: Any,
    model_config: Mapping[str, Any],
    windows: Sequence[Any],
    *,
    segment: str = "test",
    threshold: float = 0.02,
    min_windows: int = 2,
) -> dict[str, Any]:
    """多窗口滚动一致性检查：用多个训练窗口分别训练，收集各窗口 IC 并判别稳定性。

    背景：单窗口训练得到的 IC 只是一个点估计，无法区分"稳定的弱 alpha"与
    "某个窗口的偶然拟合"。此处用 ``windows`` 指定的多个训练窗口分别训练同构模型，
    在 ``segment`` 上评估 rank IC，再以 ``IC 的标准差``衡量跨窗口一致性。

    ``windows`` 中每个元素是 ``(start_time, end_time)`` 形式的时间范围（由
    :func:`_window_dataset` 负责切分为对应数据集）。判定规则：

    - ``std <= threshold`` → 稳定；否则不稳定；
    - ``len(windows) < min_windows`` → 样本不足以判定，``stable`` 为 ``False``
      且 ``reason`` 标明原因（避免用 1 个窗口"算出"标准差为 0 而误判稳定）。

    纯函数式设计：只依赖传入的 ``dataset``/``model_config``，不做任何缓存或写盘，
    便于单测直接注入替身。
    """
    window_list = [tuple(window) for window in windows]
    ics: list[float] = []
    details: list[dict[str, Any]] = []
    for index, (start_time, end_time) in enumerate(window_list):
        try:
            window_dataset = _window_dataset(
                dataset, start_time=start_time, end_time=end_time
            )
            model = train_lgbm(window_dataset, model_config)
            prediction, labels = segment_prediction(window_dataset, model, segment)
            metrics = _prediction_metrics(prediction, labels)
        except Exception as exc:  # 单个窗口失败不应吞掉整轮稳定性结论。
            details.append(
                {
                    "index": index,
                    "start_time": str(start_time),
                    "end_time": str(end_time),
                    "ic": None,
                    "observations": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        ic = metrics.get("mean_rank_ic")
        ic_value = float(ic) if ic is not None and not pd.isna(ic) else None
        if ic_value is not None:
            ics.append(ic_value)
        details.append(
            {
                "index": index,
                "start_time": str(start_time),
                "end_time": str(end_time),
                "ic": ic_value,
                "observations": int(metrics.get("observations", 0)),
            }
        )

    window_ic_mean = float(np.mean(ics)) if ics else float("nan")
    # ddof=1：跨窗口 IC 视为抽样，用样本标准差；窗口不足 2 个时为 NaN。
    window_ic_std = float(np.std(ics, ddof=1)) if len(ics) > 1 else float("nan")
    usable_windows = len(ics)
    if usable_windows < max(int(min_windows), 2):
        stable = False
        reason = (
            f"insufficient usable windows: {usable_windows} "
            f"(< min_windows={max(int(min_windows), 2)})"
        )
    elif not np.isnan(window_ic_std) and window_ic_std > float(threshold):
        stable = False
        reason = (
            f"window IC std {window_ic_std:.6f} exceeds threshold {float(threshold):.6f}"
        )
    else:
        stable = True
        reason = "window IC std within threshold"
    return {
        "stable": stable,
        "reason": reason,
        "threshold": float(threshold),
        "windows_requested": len(window_list),
        "windows_used": usable_windows,
        "ic_mean": window_ic_mean,
        "ic_std": window_ic_std,
        "ics": ics,
        "details": details,
    }


def _window_dataset(dataset: Any, *, start_time: Any, end_time: Any) -> Any:
    """把已有数据集裁剪到给定训练窗口。

    Qlib 的 ``DatasetH`` 通过 ``segments`` 描述时间范围，因此这里重写 ``train``
    段；非 Qlib 替身（单测）只需提供可变的 ``segments`` 属性即可。
    """
    import copy as _copy

    windowed = _copy.copy(dataset)
    segments = dict(getattr(dataset, "segments", {}) or {})
    segments["train"] = [start_time, end_time]
    windowed.segments = segments
    return windowed


def _run_qlib_stock_model(
    config: Mapping[str, Any],
    *,
    requested_symbols: list[str] | None = None,
    progress_callback: Callable[..., None] | None = None,
 ) -> dict[str, Any]:
    """Run the isolated Alpha158/LightGBM path.

    Importing Qlib is deliberately deferred so ETF/fund snapshot calls remain
    available even when the optional quant environment is not installed.
    """
    if os.environ.get("DSH_STOCK_SKIP_MODEL") == "1":
        return {
            "model_available": False,
            "model_name": "Qlib Alpha158 + LightGBM",
            "error": "stock model skipped by DSH_STOCK_SKIP_MODEL=1",
            "mean_rank_ic": float("nan"),
            "test_rank_ic": float("nan"),
            "rank_ic_ir": float("nan"),
            "top_bottom_excess_return": float("nan"),
            "net_annualized_return": float("nan"),
            "test_max_drawdown": float("nan"),
            "max_drawdown": float("nan"),
            "leakage_check": {"passed": False, "reason": "model skipped"},
        }
    try:
        os.environ.setdefault("OMP_NUM_THREADS", "2")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
        os.environ.setdefault("MKL_NUM_THREADS", "2")
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        os.environ.setdefault("MLFLOW_TRACKING_URI", f"file:///{(Path(config['paths']['logs_dir']) / 'mlruns').as_posix()}")
        import qlib
        from qlib.data import D

        provider = Path(config["paths"]["qlib_provider"]).resolve()
        _report_progress(progress_callback, "provider_cache_check")
        provider_lock = ProviderLock(
            provider,
            purpose="Alpha158 training, metrics, and backtest",
        ).acquire(
            timeout_seconds=float(config.get("runtime", {}).get("provider_lock_timeout_seconds", 30.0))
        )
        try:
            init_qlib(provider, kernels=1)
            dataset_config = config["dataset"]
            calendar = D.calendar(freq="day")
            if not len(calendar):
                raise ValueError("Qlib provider calendar is empty")
            provider_end = pd.Timestamp(calendar[-1]).date().isoformat()
            fingerprint = _provider_source_fingerprint(provider)
            cache_key = None
            cache_identity = None
            cache_path = None
            cache_miss_reason = "provider_source_fingerprint_missing"
            if fingerprint is not None and fingerprint.get("digest"):
                cache_key, cache_identity = _model_cache_identity(
                    config,
                    provider_fingerprint=fingerprint,
                    provider_end=provider_end,
                )
                cache_path = _model_cache_path(config, cache_key)
                cached = _cached_model_result(cache_path, cache_key)
                if cached is not None:
                    _report_progress(
                        progress_callback,
                        "model_cache_check",
                        cache_hit=True,
                        provider_end=provider_end,
                    )
                    result = _filter_scores(cached, requested_symbols)
                    result.pop("cache_miss_reason", None)
                    result.update(
                        {
                            "cache_hit": True,
                            "model_cache_key": cache_key,
                        }
                    )
                    _report_progress(progress_callback, "publish", cache_hit=True)
                    return result
                cache_miss_reason = "cache_file_missing_or_invalid"
            _report_progress(
                progress_callback,
                "model_cache_check",
                cache_hit=False,
                cache_miss_reason=cache_miss_reason,
                provider_end=provider_end,
            )
            end_time = min(str(dataset_config["segments"]["test"][1]), provider_end)
            segments = {
                name: [values[0], min(values[1], end_time)]
                for name, values in dataset_config["segments"].items()
            }
            label_horizon = int(config["research"]["stock_label_horizon"])
            _report_progress(progress_callback, "alpha158_prepare", provider_end=provider_end)
            dataset = build_dataset(
                instruments="stocks",
                start_time=dataset_config["start_time"],
                end_time=end_time,
                fit_start_time=dataset_config["fit_start_time"],
                fit_end_time=dataset_config["fit_end_time"],
                label_horizon=label_horizon,
                segments=segments,
            )
            _report_progress(progress_callback, "lightgbm_train")
            model = train_lgbm(dataset, config["model"])
            predictions: dict[str, pd.Series] = {}
            labels_by_segment: dict[str, pd.Series] = {}
            metrics: dict[str, Any] = {}
            for segment in ("valid", "test"):
                _report_progress(progress_callback, f"{segment}_metrics")
                prediction, labels = segment_prediction(dataset, model, segment)
                predictions[segment] = prediction
                labels_by_segment[segment] = labels
                metrics[segment] = _prediction_metrics(prediction, labels)
            test_prediction = predictions["test"]
            if test_prediction.empty:
                raise ValueError("Qlib test prediction is empty")
            # 多窗口滚动一致性检查（可选）：默认关闭，避免在既有训练成本上
            # 再叠加 N 次全量训练。开启方式：research.stability_windows 给出
            # [(start, end), ...] 窗口列表。
            stability_cfg = config.get("research") or {}
            stability_windows = stability_cfg.get("stability_windows") or []
            if stability_windows:
                _report_progress(progress_callback, "stability_check")
                stability = multi_window_stability_check(
                    dataset,
                    config["model"],
                    stability_windows,
                    threshold=float(
                        stability_cfg.get("stability_ic_std_threshold", 0.02)
                    ),
                )
            else:
                stability = {
                    "stable": None,
                    "reason": "stability check disabled (research.stability_windows empty)",
                    "threshold": float(
                        stability_cfg.get("stability_ic_std_threshold", 0.02)
                    ),
                    "windows_requested": 0,
                    "windows_used": 0,
                    "ic_mean": float("nan"),
                    "ic_std": float("nan"),
                    "ics": [],
                    "details": [],
                }
            # 2026-09-11 整改(P0-1 + P0-2)：先判定横截面是否退化，再决定要不要跑回测。
            # 实证：2026-09-11 provider 日历末位 2026-09-11 只有 6 只有数据，
            # scores 从 5562 退化为 6，且因 score_count<100 拒绝写缓存 →
            # 每轮冷启动重训 29 分钟。此处按"数据不足"提前返回，不再继续算门槛。
            min_signal_scores = int(
                (config.get("research") or {}).get("min_signal_scores", 0) or 0
            )
            signal_date = select_signal_date(test_prediction)
            if signal_date is None:
                return _degenerate_stock_model(
                    "no trading day reached the relative cross-section threshold "
                    "(samples >= 50% of the median daily universe)",
                    score_count=0,
                    provider_end=provider_end,
                )
            latest = test_prediction.xs(signal_date, level="datetime").sort_values(
                ascending=False
            )
            if min_signal_scores and len(latest) < min_signal_scores:
                return _degenerate_stock_model(
                    "signal date "
                    f"{research._iso_date(signal_date)} produced only {len(latest)} "
                    f"scores (< research.min_signal_scores={min_signal_scores})",
                    score_count=int(len(latest)),
                    provider_end=provider_end,
                )
            _report_progress(progress_callback, "backtest")
            raw = dataset.handler.fetch(
                selector=slice(segments["test"][0], segments["test"][1]),
                level="datetime",
                col_set="backtest",
                data_key="raw",
            )
            closes = raw["BACKTEST_CLOSE"].unstack("instrument").sort_index()
            future = closes.shift(-1) / closes - 1.0
            transaction_cost_rate = (
                float(config["research"]["transaction_cost_bps"]) / 10000.0
            )
            research_cfg = config.get("research") or {}
            exclude_bj = bool(research_cfg.get("pool_exclude_bj", False))
            topk = int(research_cfg["stock_topk"])
            include_cash = bool(research_cfg.get("backtest_include_cash", True))
            allocation_slots = max(
                int(research_cfg.get("backtest_allocation_slots", 3)), 1
            )
            # P1-5：持有期与标签 horizon 对齐（重叠 cohort）。
            holding_days = max(int(research_cfg.get("stock_label_horizon", 20)), 1)
            cohort_count = max(int(research_cfg.get("backtest_cohorts", 5)), 1)
            cohort_count = min(cohort_count, holding_days)
            # 2026-09-11 整改(P1-4)：门槛必须衡量"实盘真正会持有的组合"。
            traded_budget = (
                allocation_budget(config["risk"], allocation_slots)
                if include_cash
                else 1.0
            )
            weight_strategy = str(research_cfg.get("weight_strategy", "equal_score"))
            benchmark_returns: dict[pd.Timestamp, float] = {}
            legacy_returns: dict[pd.Timestamp, float] = {}
            legacy_turnovers: dict[pd.Timestamp, float] = {}
            horizon_returns: dict[pd.Timestamp, float] = {}
            horizon_turnovers: dict[pd.Timestamp, float] = {}
            traded_returns: dict[pd.Timestamp, float] = {}
            traded_turnovers: dict[pd.Timestamp, float] = {}
            legacy_previous: dict[str, float] = {}
            horizon_previous: dict[str, float] = {}
            traded_previous: dict[str, float] = {}

            def _top_symbols(scores: pd.Series) -> list[str]:
                # scores 已由 .xs(date, level="datetime") 取成"仅含 instrument"的单日截面。
                # 2026-09-10 回撤根治：回测组合与实盘选股同口径剔除北交所
                # （92/43/83/87/4 开头，±30% 板），pool_exclude_bj 开启时生效。
                # 注意回测 instrument 带 Qlib 前缀（BJ920855/SH600519），
                # 先判 BJ 前缀，再判裸码号段，二者任一命中即剔除。
                ranked = scores.sort_values(ascending=False)
                if exclude_bj:
                    names = ranked.index.to_series().astype(str).str.upper()
                    bare = names.str.replace(r"^(SH|SZ|BJ)", "", regex=True)
                    is_bj = names.str.startswith("BJ") | bare.str.startswith(
                        ("92", "43", "83", "87", "4")
                    )
                    ranked = ranked[~is_bj]
                return [str(symbol) for symbol in ranked.head(topk).index]

            ordered_dates = [
                date
                for date in sorted(
                    test_prediction.index.get_level_values("datetime").unique()
                )
                if date in future.index
            ]
            daily_picks: list[list[str]] = []
            daily_scores: list[dict[str, float]] = []
            daily_values: list[pd.Series] = []
            for date in ordered_dates:
                scores = test_prediction.xs(date, level="datetime")
                values = future.loc[date]
                picks = [s for s in _top_symbols(scores) if s in values.index]
                daily_picks.append(picks)
                daily_scores.append(
                    {s: float(scores.get(s, 0.0)) for s in picks}
                )
                daily_values.append(values)
                universe_values = values.dropna()
                benchmark_returns[pd.Timestamp(date)] = (
                    float(universe_values.mean())
                    if not universe_values.empty
                    else float("nan")
                )

            schedule = cohort_schedule(
                len(ordered_dates), holding_days=holding_days, cohorts=cohort_count
            )

            for index, date in enumerate(ordered_dates):
                stamp = pd.Timestamp(date)
                values = daily_values[index]
                # --- 口径 A：满仓等权、逐日重选（旧口径；仅作模型质量监控）---
                legacy_symbols = daily_picks[index]
                if legacy_symbols:
                    legacy_weights = {
                        symbol: 1.0 / len(legacy_symbols) for symbol in legacy_symbols
                    }
                    legacy_turnover = _turnover(legacy_weights, legacy_previous)
                    legacy_returns[stamp] = (
                        float(values.reindex(legacy_symbols).mean())
                        - legacy_turnover * transaction_cost_rate
                    )
                    legacy_turnovers[stamp] = legacy_turnover
                    legacy_previous = legacy_weights
                # --- 口径 C：满仓 + cohort（P1-5 的"纯净"对照，不受现金稀释影响）---
                active = schedule.get(index) or ()
                horizon_picks: list[str] = []
                horizon_scores: dict[str, float] = {}
                for entry in active:
                    for symbol in daily_picks[entry]:
                        if symbol not in horizon_scores:
                            horizon_scores[symbol] = daily_scores[entry][symbol]
                            horizon_picks.append(symbol)
                if horizon_picks:
                    horizon_weights = {
                        symbol: 1.0 / len(horizon_picks) for symbol in horizon_picks
                    }
                    horizon_turnover = _turnover(horizon_weights, horizon_previous)
                    horizon_returns[stamp] = (
                        float(values.reindex(horizon_picks).mean())
                        - horizon_turnover * transaction_cost_rate
                    )
                    horizon_turnovers[stamp] = horizon_turnover
                    horizon_previous = horizon_weights
                # --- 口径 B：目标组合 + cohort（与实盘同口径，参与门槛）---
                if not horizon_picks:
                    continue
                candidates = [
                    {"symbol": symbol, "qlib_score": horizon_scores[symbol]}
                    for symbol in horizon_picks
                ]
                requested = build_target_weights(
                    candidates,
                    "stock",
                    config["risk"],
                    budget=traded_budget,
                    strategy=weight_strategy,
                )
                enforced = enforce_target_weights(
                    requested,
                    {symbol: "stock" for symbol in requested},
                    config["risk"],
                )
                traded_weights = {
                    str(symbol): float(weight)
                    for symbol, weight in (enforced.get("target_weights") or {}).items()
                }
                traded_turnover = _turnover(traded_weights, traded_previous)
                gross_return = sum(
                    float(weight) * float(values.get(symbol, 0.0))
                    for symbol, weight in traded_weights.items()
                    if symbol in values.index and pd.notna(values.get(symbol))
                )
                # 未投入部分按现金处理（收益 0，保守）。
                traded_returns[stamp] = (
                    gross_return - traded_turnover * transaction_cost_rate
                )
                traded_turnovers[stamp] = traded_turnover
                traded_previous = traded_weights
            net_returns = pd.Series(traded_returns, dtype=float).sort_index()
            equity = net_returns.add(1.0).cumprod()
            legacy_equity = (
                pd.Series(legacy_returns, dtype=float).sort_index().add(1.0).cumprod()
            )
            horizon_equity = (
                pd.Series(horizon_returns, dtype=float).sort_index().add(1.0).cumprod()
            )
            benchmark_equity = (
                pd.Series(benchmark_returns, dtype=float)
                .sort_index()
                .dropna()
                .add(1.0)
                .cumprod()
            )
            performance = _performance(equity)
            legacy_performance = _performance(legacy_equity)
            horizon_performance = _performance(horizon_equity)
            benchmark_performance = _performance(benchmark_equity)
            group_metrics = _forward_group_metrics(
                test_prediction,
                labels_by_segment["test"],
            )
            # latest / signal_date 已在回测前算出（见上方 P0-1 守卫）。
            latest_date = signal_date
            best_iteration = getattr(
                getattr(model, "model", None), "best_iteration", None
            )
            full_result = {
                "model_available": True,
                "model_name": "Qlib Alpha158 + LightGBM",
                "qlib_version": getattr(qlib, "__version__", "unknown"),
                "provider": str(provider),
                "data_end_date": provider_end,
                "latest_signal_date": research._iso_date(latest_date),
                # 2026-09-11 整改(P0-1)：显式记录实际使用的信号日，便于判断
                # 它是否回退到了 provider 日历末位之前（回退即为日历错位的信号）。
                "signal_date_used": research._iso_date(signal_date),
                "signal_date_is_provider_end": bool(
                    research._iso_date(signal_date) == provider_end
                ),
                "score_count": int(len(latest)),
                "valid": metrics["valid"],
                "test": metrics["test"],
                "mean_rank_ic": metrics["test"]["mean_rank_ic"],
                "test_rank_ic": metrics["test"]["mean_rank_ic"],
                "rank_ic_ir": metrics["test"]["rank_ic_ir"],
                # 多窗口滚动一致性：跨窗口 IC 均值/标准差与稳定性判定。
                "stability": stability,
                "stability_stable": stability["stable"],
                "stability_ic_mean": stability["ic_mean"],
                "stability_ic_std": stability["ic_std"],
                "top_bottom_excess_return": group_metrics[
                    "top_bottom_excess_return"
                ],
                "forward_group_metrics": group_metrics,
                "net_annualized_return": performance["net_annualized_return"],
                "test_max_drawdown": performance["max_drawdown"],
                "max_drawdown": performance["max_drawdown"],
                "transaction_cost_bps": float(
                    config["research"]["transaction_cost_bps"]
                ),
                "mean_daily_turnover": float(
                    pd.Series(traded_turnovers, dtype=float).mean()
                ),
                # 2026-09-11 整改(P1-4)：显式声明门槛所衡量的口径。
                "backtest_mode": (
                    "target_portfolio_with_cash" if include_cash else "sleeve_only"
                ),
                "backtest_allocation_slots": allocation_slots,
                "backtest_include_cash": include_cash,
                "backtest_traded_budget": float(traded_budget),
                "backtest_holding_days": holding_days,
                "backtest_cohorts": cohort_count,
                # 旧口径基线：满仓等权 + **1 日**持有、逐日重选。仅作模型质量监控。
                "model_quality_backtest": {
                    "net_annualized_return": legacy_performance[
                        "net_annualized_return"
                    ],
                    "max_drawdown": legacy_performance["max_drawdown"],
                    "mean_daily_turnover": float(
                        pd.Series(legacy_turnovers, dtype=float).mean()
                    ),
                },
                # P1-5 纯净对照：满仓 + 同样的 cohort 构建。它与 model_quality_backtest
                # 构成**唯一变量对照**（只差持有期），用于回答"持有期对齐本身是否
                # 改善回撤/收益"；刻意不吃现金稀释、也不受仓位上限约束，因此**不参与门槛**。
                "horizon_matched_backtest": {
                    "net_annualized_return": horizon_performance[
                        "net_annualized_return"
                    ],
                    "max_drawdown": horizon_performance["max_drawdown"],
                    "mean_daily_turnover": float(
                        pd.Series(horizon_turnovers, dtype=float).mean()
                    ),
                },
                "benchmark": {
                    "name": "equal_weight_stock_universe",
                    "annualized_return": benchmark_performance[
                        "net_annualized_return"
                    ],
                    "max_drawdown": benchmark_performance["max_drawdown"],
                    "excess_annualized_return": performance[
                        "net_annualized_return"
                    ]
                    - benchmark_performance["net_annualized_return"],
                },
                # 2026-09-12 整改：满仓 20 日口径回撤告警（非阻断，见 _drawdown_advisory）。
                "drawdown_advisories": {
                    "horizon_matched": _drawdown_advisory(
                        horizon_performance["max_drawdown"],
                        benchmark_performance["max_drawdown"],
                        (config.get("gate") or {}).get(
                            "max_excess_test_drawdown", 0.15
                        ),
                    ),
                },
                "training_best_iteration": best_iteration,
                # 2026-09-11 整改：记录拟合质量，供 fit_quality_threshold 门槛判定。
                **_fit_quality(model),
                "scores": {
                    validate_symbol(str(symbol)): float(score)
                    for symbol, score in latest.items()
                },
                "leakage_check": _check_label_leakage(
                    calendar=calendar,
                    fit_end_time=dataset_config["fit_end_time"],
                    label_horizon=label_horizon,
                    valid_start=dataset_config["segments"]["valid"][0],
                ),
                "cache_hit": False,
                "cache_miss_reason": cache_miss_reason,
                "model_cache_key": cache_key,
            }
            _report_progress(progress_callback, "publish", cache_hit=False)
            # 健康检查：防止写入残缺缓存（2026-09-10 坏缓存事故后增设）
            # 正常全市场训练应产生数千个分数；少于 MIN_STOCK_SCORES 视为异常，不写缓存。
            # 注意：退化到这一步已在回测前被 _degenerate_stock_model 拦截；
            # 这里保留二次防线，防止未来改动绕过前置守卫。
            score_count = len(full_result["scores"])
            min_scores_for_cache = MIN_STOCK_SCORES
            if cache_path is not None and cache_key is not None:
                if score_count < min_scores_for_cache:
                    full_result["cache_write_skipped"] = True
                    full_result["cache_skip_reason"] = (
                        f"score_count={score_count} below min_scores_for_cache={min_scores_for_cache}"
                    )
                else:
                    atomic_write_json(
                        cache_path,
                        research._jsonable(
                            {
                                "schema": MODEL_CACHE_SCHEMA,
                                "cache_key": cache_key,
                                "identity": cache_identity,
                                "result": full_result,
                            }
                        ),
                    )
            return _filter_scores(full_result, requested_symbols)
        finally:
            provider_lock.release()
    except Exception as exc:
        return {
            "model_available": False,
            "model_name": "Qlib Alpha158 + LightGBM",
            "error": f"{type(exc).__name__}: {exc}",
            "mean_rank_ic": float("nan"),
            "test_rank_ic": float("nan"),
            "rank_ic_ir": float("nan"),
            "top_bottom_excess_return": float("nan"),
            "net_annualized_return": float("nan"),
            "test_max_drawdown": float("nan"),
            "max_drawdown": float("nan"),
            "leakage_check": {"passed": False, "reason": "model execution failed"},
        }
