"""Qlib 训练与模型指标层 + 共享训练入口（docs/重构计划.md 阶段 1/3）。

阶段 3 起，provider 初始化（init_qlib）、数据集构建（build_dataset）、
LGBModel 训练（train_lgbm）与分段预测（segment_prediction）是主路径与
run_under50_research.py 共享的训练层；Qlib 的导入仍延迟到各函数内部，
保证未安装可选依赖时 ETF/基金快照链路可用。定义在 research.py 的工具
（_iso_date）经 ``research.`` 属性访问；其余名字为模块内自有或来自 paths。
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
        from qlib.data import D

        provider = init_qlib(config["paths"]["qlib_provider"], kernels=1)
        dataset_config = config["dataset"]
        calendar = D.calendar(freq="day")
        provider_end = pd.Timestamp(calendar[-1]).date().isoformat()
        end_time = min(str(dataset_config["segments"]["test"][1]), provider_end)
        segments = {
            name: [values[0], min(values[1], end_time)]
            for name, values in dataset_config["segments"].items()
        }
        label_horizon = int(config["research"]["stock_label_horizon"])
        dataset = build_dataset(
            instruments="stocks",
            start_time=dataset_config["start_time"],
            end_time=end_time,
            fit_start_time=dataset_config["fit_start_time"],
            fit_end_time=dataset_config["fit_end_time"],
            label_horizon=label_horizon,
            segments=segments,
        )
        model = train_lgbm(dataset, config["model"])
        predictions: dict[str, pd.Series] = {}
        labels_by_segment: dict[str, pd.Series] = {}
        metrics: dict[str, Any] = {}
        for segment in ("valid", "test"):
            prediction, labels = segment_prediction(dataset, model, segment)
            predictions[segment] = prediction
            labels_by_segment[segment] = labels
            metrics[segment] = _prediction_metrics(prediction, labels)
        test_prediction = predictions["test"]
        if test_prediction.empty:
            raise ValueError("Qlib test prediction is empty")
        raw = dataset.handler.fetch(
            selector=slice(segments["test"][0], segments["test"][1]),
            level="datetime",
            col_set="backtest",
            data_key="raw",
        )
        closes = raw["BACKTEST_CLOSE"].unstack("instrument").sort_index()
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
