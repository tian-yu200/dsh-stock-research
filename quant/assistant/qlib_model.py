"""Qlib 训练与模型指标层（自 research.py 逐字搬迁，docs/重构计划.md 阶段 1）。

Qlib 的导入刻意延迟到 _run_qlib_stock_model 内部，保证未安装可选依赖时
ETF/基金快照链路仍可用。定义在 research.py 的工具（_iso_date）经
``research.`` 属性访问；其余名字为模块内自有或直接来自 paths。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from . import research
from .paths import validate_symbol

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


def _fit_qlib_model_and_close_recorder(
    model: Any,
    dataset: Any,
    qlib_recorder: Any,
) -> None:
    """Close Qlib's global recorder after model.fit, including failed fits."""
    try:
        model.fit(dataset)
    finally:
        qlib_recorder.end_exp()


def _run_qlib_stock_model(config: Mapping[str, Any]) -> dict[str, Any]:
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
        from qlib.config import REG_CN
        from qlib.contrib.data.handler import Alpha158
        from qlib.contrib.model.gbdt import LGBModel
        from qlib.data import D
        from qlib.data.dataset import DatasetH
        from qlib.data.dataset.handler import DataHandlerLP
        from qlib.workflow import R

        provider = Path(config["paths"]["qlib_provider"]).resolve()
        if not provider.exists():
            raise FileNotFoundError(f"Qlib provider does not exist: {provider}")
        qlib.init(provider_uri=str(provider), region=REG_CN, kernels=1)
        dataset_config = config["dataset"]
        calendar = D.calendar(freq="day")
        provider_end = pd.Timestamp(calendar[-1]).date().isoformat()
        end_time = min(str(dataset_config["segments"]["test"][1]), provider_end)
        segments = {
            name: [values[0], min(values[1], end_time)]
            for name, values in dataset_config["segments"].items()
        }
        label_horizon = int(config["research"]["stock_label_horizon"])
        handler = Alpha158(
            instruments="stocks",
            start_time=dataset_config["start_time"],
            end_time=end_time,
            fit_start_time=dataset_config["fit_start_time"],
            fit_end_time=dataset_config["fit_end_time"],
            label=([f"Ref($close, -{label_horizon + 1})/Ref($close, -1) - 1"], ["LABEL0"]),
            infer_processors=[
                {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
                {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
            ],
            learn_processors=[
                {"class": "DropnaLabel"},
                {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
            ],
        )
        dataset = DatasetH(handler=handler, segments=segments)
        model = LGBModel(**config["model"])
        _fit_qlib_model_and_close_recorder(model, dataset, R)
        predictions: dict[str, pd.Series] = {}
        labels_by_segment: dict[str, pd.Series] = {}
        metrics: dict[str, Any] = {}
        for segment in ("valid", "test"):
            prediction = model.predict(dataset, segment=segment).dropna().sort_index()
            prediction.name = "score"
            labels = dataset.prepare(segment, col_set="label", data_key=DataHandlerLP.DK_L).iloc[:, 0]
            predictions[segment] = prediction
            labels_by_segment[segment] = labels
            metrics[segment] = _prediction_metrics(prediction, labels)
        test_prediction = predictions["test"]
        if test_prediction.empty:
            raise ValueError("Qlib test prediction is empty")
        raw = D.features(
            D.instruments("stocks"),
            ["$close"],
            start_time=segments["test"][0],
            end_time=segments["test"][1],
        )
        closes = raw["$close"].unstack("instrument").sort_index()
        future = closes.shift(-1) / closes - 1.0
        strategy_returns: dict[pd.Timestamp, float] = {}
        benchmark_returns: dict[pd.Timestamp, float] = {}
        turnover_by_date: dict[pd.Timestamp, float] = {}
        previous_weights: dict[str, float] = {}
        transaction_cost_rate = float(config["research"]["transaction_cost_bps"]) / 10000.0
        for date, scores in test_prediction.groupby(level="datetime"):
            if date not in future.index:
                continue
            top = scores.droplevel("datetime").sort_values(ascending=False).head(int(config["research"]["stock_topk"]))
            values = future.loc[date].reindex(top.index).dropna()
            if not values.empty:
                weights = {str(symbol): 1.0 / len(values) for symbol in values.index}
                symbols = set(previous_weights) | set(weights)
                turnover = sum(
                    abs(weights.get(symbol, 0.0) - previous_weights.get(symbol, 0.0))
                    for symbol in symbols
                ) / 2.0
                gross_return = float(values.mean())
                strategy_returns[pd.Timestamp(date)] = gross_return - turnover * transaction_cost_rate
                turnover_by_date[pd.Timestamp(date)] = turnover
                universe_values = future.loc[date].dropna()
                benchmark_returns[pd.Timestamp(date)] = (
                    float(universe_values.mean()) if not universe_values.empty else float("nan")
                )
                previous_weights = weights
        net_returns = pd.Series(strategy_returns, dtype=float).sort_index()
        equity = net_returns.add(1.0).cumprod()
        benchmark_equity = pd.Series(benchmark_returns, dtype=float).sort_index().dropna().add(1.0).cumprod()
        performance = _performance(equity)
        benchmark_performance = _performance(benchmark_equity)
        group_metrics = _forward_group_metrics(
            test_prediction,
            labels_by_segment["test"],
        )
        latest_date = test_prediction.index.get_level_values("datetime").max()
        latest = test_prediction.xs(latest_date, level="datetime").sort_values(ascending=False)
        best_iteration = getattr(getattr(model, "model", None), "best_iteration", None)
        return {
            "model_available": True,
            "model_name": "Qlib Alpha158 + LightGBM",
            "qlib_version": getattr(qlib, "__version__", "unknown"),
            "provider": str(provider),
            "data_end_date": provider_end,
            "latest_signal_date": research._iso_date(latest_date),
            "valid": metrics["valid"],
            "test": metrics["test"],
            "mean_rank_ic": metrics["test"]["mean_rank_ic"],
            "test_rank_ic": metrics["test"]["mean_rank_ic"],
            "rank_ic_ir": metrics["test"]["rank_ic_ir"],
            "top_bottom_excess_return": group_metrics["top_bottom_excess_return"],
            "forward_group_metrics": group_metrics,
            "net_annualized_return": performance["net_annualized_return"],
            "test_max_drawdown": performance["max_drawdown"],
            "max_drawdown": performance["max_drawdown"],
            "transaction_cost_bps": float(config["research"]["transaction_cost_bps"]),
            "mean_daily_turnover": float(pd.Series(turnover_by_date, dtype=float).mean()),
            "benchmark": {
                "name": "equal_weight_stock_universe",
                "annualized_return": benchmark_performance["net_annualized_return"],
                "max_drawdown": benchmark_performance["max_drawdown"],
                "excess_annualized_return": performance["net_annualized_return"]
                - benchmark_performance["net_annualized_return"],
            },
            "training_best_iteration": best_iteration,
            "scores": {
                validate_symbol(str(symbol)): float(score)
                for symbol, score in latest.items()
            },
            "leakage_check": {"passed": True, "label_horizon_days": label_horizon},
        }
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
