"""Configuration loading for the fixed-path stock assistant."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .paths import (
    ASSISTANT_RESULTS_DIR,
    CONFIG_DIR,
    DATA_DIR,
    LOGS_DIR,
    QLIB_PROVIDER_DIR,
    fixed_path,
)

DEFAULT_CONFIG_PATH = CONFIG_DIR / "assistant.yaml"


def _deep_merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = dict(left)
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


DEFAULTS: dict[str, Any] = {
    "project": {
        "name": "dsh-stock-advisor",
        "timezone": "Asia/Shanghai",
        "currency": "CNY",
    },
    "paths": {
        "data_dir": str(DATA_DIR),
        "qlib_provider": str(QLIB_PROVIDER_DIR),
        "results_dir": str(ASSISTANT_RESULTS_DIR),
        "logs_dir": str(LOGS_DIR),
    },
    "fetch": {
        "enabled": True,
        "timeout_seconds": 900,
        "retries": 2,
        "retry_delay_seconds": 3,
        "max_data_age_days": 5,
        "min_universe_coverage": 0.95,
        "target_lag_trading_days": {
            "stock": 0,
            "etf": 1,
            "otc_fund": 1,
        },
        "batch_size": {
            "stock": 100,
            "etf": 50,
            "otc_fund": 200,
        },
        "batch_time_budget_seconds": 600,
        "shutdown_grace_seconds": 45,
        "max_workers": 3,
        "requests_per_second": 1.0,
        "request_timeout_seconds": 20,
        "heartbeat_interval_seconds": 5,
        "worker_stalled_seconds": 60,
        "backoff_base_seconds": 2,
        "backoff_max_seconds": 120,
        "lease_seconds": 1800,
        "catalog_refresh_hours": 24,
        "state_db": str(DATA_DIR / "fetch_state.sqlite3"),
        "background_worker_enabled": True,
    },
    "runtime": {
        "lock_stale_seconds": 21600,
    },
    "rolling_scan": {
        "enabled": True,
        "state_db": str(DATA_DIR / "rolling_scan.sqlite3"),
        "snapshot_path": str(ASSISTANT_RESULTS_DIR / "rolling_scan.json"),
        "pool_asset_types": ["stock"],
        "pool_size": 30,
        "min_sample": 200,
        "min_rows": 120,
        "min_average_amount_20d": 30000000.0,
        "stale_after_days": 15,
        "max_symbols_per_pass": 400,
        "score_table_limit": 500,
    },
    "research": {
        "stock_label_horizon": 20,
        "stock_topk": 3,
        "weight_strategy": "equal_score",
        "technical_lookback_days": 260,
        "hypothetical_capital": 100000.0,
        "refresh_modes": ["daily", "monthly", "on_demand"],
        "transaction_cost_bps": 10.0,
    },
    "risk": {
        "max_single_weight": 0.20,
        "max_stock_exposure": 0.40,
        "max_etf_exposure": 0.50,
        "max_otc_fund_exposure": 0.30,
        "min_cash_weight": 0.20,
        "max_daily_turnover": 0.20,
    },
    "gate": {
        "min_mean_rank_ic": 0.02,
        "min_rank_ic_ir": 0.10,
        "min_top_bottom_excess_return": 0.00,
        "max_test_drawdown": 0.25,
        "max_excess_test_drawdown": 0.15,
    },
    "model": {
        "loss": "mse",
        "learning_rate": 0.03,
        "num_leaves": 15,
        "max_depth": 5,
        "colsample_bytree": 0.8,
        "subsample": 0.8,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "num_threads": 2,
        "num_boost_round": 300,
        "early_stopping_rounds": 30,
    },
    "dataset": {
        "start_time": "2020-01-02",
        "fit_start_time": "2020-04-01",
        "fit_end_time": "2022-12-28",
        "segments": {
            "train": ["2020-04-01", "2022-12-28"],
            "valid": ["2023-01-03", "2024-06-26"],
            "test": ["2024-07-01", "2099-12-31"],
        },
    },
}


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    path = fixed_path(path)
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
    else:
        loaded = {}
    if not isinstance(loaded, dict):
        raise ValueError("assistant.yaml must contain a mapping")
    config = _deep_merge(DEFAULTS, loaded)
    for key in ("data_dir", "qlib_provider", "results_dir", "logs_dir"):
        config["paths"][key] = str(fixed_path(config["paths"][key]))
    config["fetch"]["state_db"] = str(fixed_path(config["fetch"]["state_db"]))
    return config


def config_snapshot(config: dict[str, Any]) -> dict[str, Any]:
    """Return a serializable copy used in each result manifest."""
    return yaml.safe_load(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
