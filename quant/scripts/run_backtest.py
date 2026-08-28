# -*- coding: utf-8 -*-
"""Run an Alpha158 + LightGBM backtest from a YAML configuration."""
import argparse
import multiprocessing
import os
import pathlib

# 16 GB 内存机器采用保守并发，避免特征生成挤占桌面资源。
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import pandas as pd
import yaml

import qlib
from qlib.config import REG_CN
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.evaluate import backtest_daily, risk_analysis
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.data.dataset import DatasetH

ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "qlib_config.yaml"
DEFAULT_MODEL_CONFIG = {
    "loss": "mse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 8,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "num_threads": 2,
}


def _project_path(value: str) -> pathlib.Path:
    path = pathlib.Path(value)
    return path if path.is_absolute() else ROOT / path


def run(config_path: pathlib.Path) -> None:
    config_path = config_path.resolve()
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    qlib_config = config["qlib_init"]
    dataset_config = config["dataset"]
    backtest_config = config["backtest"]
    result_dir = _project_path(config.get("results", {}).get("output_dir", "results"))
    result_dir.mkdir(parents=True, exist_ok=True)

    provider_uri = _project_path(qlib_config["provider_uri"]).resolve()
    if not provider_uri.exists():
        raise FileNotFoundError(f"Qlib provider does not exist: {provider_uri}")
    print(f"配置: {config_path}")
    print(f"Qlib 数据: {provider_uri}")
    qlib.init(provider_uri=str(provider_uri), region=REG_CN, kernels=2)

    handler = Alpha158(
        instruments=dataset_config["instruments"],
        start_time=dataset_config["start_time"],
        end_time=dataset_config["end_time"],
        fit_start_time=dataset_config["fit_start_time"],
        fit_end_time=dataset_config["fit_end_time"],
        infer_processors=[
            {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
            {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
        ],
        learn_processors=[
            {"class": "DropnaLabel"},
            {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
        ],
    )
    dataset = DatasetH(handler=handler, segments=dataset_config["segments"])

    model_config = {**DEFAULT_MODEL_CONFIG, **config.get("model", {})}
    model = LGBModel(**model_config)
    print("训练 LightGBM...")
    model.fit(dataset)

    print("生成测试集预测...")
    prediction = model.predict(dataset, segment="test").dropna().sort_index()
    if prediction.empty:
        raise ValueError("测试集没有生成任何预测，请检查数据区间和股票池")
    prediction.name = "score"
    prediction.to_csv(result_dir / "predictions.csv", encoding="utf-8-sig")

    strategy = TopkDropoutStrategy(
        signal=prediction,
        topk=backtest_config["topk"],
        n_drop=backtest_config["n_drop"],
    )
    print("执行策略回测...")
    report, _ = backtest_daily(
        start_time=backtest_config["start_time"],
        end_time=backtest_config["end_time"],
        strategy=strategy,
        account=backtest_config["account"],
        benchmark=backtest_config["benchmark"],
        exchange_kwargs=backtest_config["exchange_kwargs"],
    )
    report.to_csv(result_dir / "portfolio_report.csv", encoding="utf-8-sig")

    excess_without_cost = risk_analysis(report["return"] - report["bench"], freq="day")
    excess_with_cost = risk_analysis(report["return"] - report["bench"] - report["cost"], freq="day")
    summary = pd.concat(
        {"excess_without_cost": excess_without_cost, "excess_with_cost": excess_with_cost},
        axis=1,
    )
    summary.to_csv(result_dir / "risk_analysis.csv", encoding="utf-8-sig")
    with (result_dir / "run_config.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)

    print("\n回测完成，风险指标：")
    print(summary)
    print(f"\n结果目录: {result_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=pathlib.Path, default=CONFIG_PATH, help="回测 YAML 配置")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
