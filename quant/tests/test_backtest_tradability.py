"""回测可执行性（2026-10-10）：入场时点与涨停/停牌可成交性。

两件事：
1. **特征化基线**：``backtest_entry_lag_days=0`` + ``backtest_skip_limit_up=False``
   必须与改动前的回测逐位一致（数值取自改动前代码的实跑结果）。
2. **新口径**：信号日 t 收盘后才产生信号，入场在 t+1 收盘（与标签
   ``close(t+H+1)/close(t+1)`` 同口径）；入场日收盘封涨停或停牌的候选跳过、不补位。

场景刻意让高分股次日大概率涨停（分数 = 次日收益 + 噪声），以便新旧口径拉开差距。
"""
from __future__ import annotations

import copy
import math
import pathlib
import sys
import tempfile
import unittest
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import qlib.data

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from assistant import qlib_model
from assistant.config import DEFAULTS

INSTRUMENTS = [
    "SH600001",
    "SH600002",
    "SZ000003",
    "SZ300004",  # 创业板 ±20%
    "SH688005",  # 科创板 ±20%
    "SZ000006",
    "SZ002007",
    "SH600008",
]


def _scenario() -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DatetimeIndex]:
    rng = np.random.default_rng(20261010)
    dates = pd.bdate_range("2024-01-02", periods=45)
    daily = rng.normal(0.0, 0.02, size=(len(dates), len(INSTRUMENTS)))
    # 注入封板日：主板 +10%，创业/科创 +20%。
    for row, column in [(5, 0), (6, 0), (11, 3), (12, 4), (18, 1), (23, 5), (30, 6), (31, 3)]:
        limit = 0.20 if INSTRUMENTS[column][2:5] in {"300", "688"} else 0.10
        daily[row, column] = limit
    daily[0, :] = 0.0
    closes = pd.DataFrame(
        10.0 * np.cumprod(1.0 + daily, axis=0), index=dates, columns=INSTRUMENTS
    )
    # 停牌：SH600008 在第 14-16 日无 bar，第 17 日复牌。
    closes.iloc[14:17, 7] = np.nan

    next_return = (closes.shift(-1) / closes - 1.0).fillna(0.0)
    noise = rng.normal(0.0, 0.01, size=closes.shape)
    scores = next_return + noise
    prediction = scores.stack()
    prediction.index = prediction.index.set_names(["datetime", "instrument"])
    labels = prediction * 0.5
    return closes, prediction.sort_index(), labels.sort_index(), dates


def _run(overrides: dict[str, Any]) -> dict[str, Any]:
    closes, prediction, labels, dates = _scenario()
    with tempfile.TemporaryDirectory(prefix="bt-tradable-", dir=ROOT / "tests") as directory:
        root = pathlib.Path(directory)
        config = copy.deepcopy(DEFAULTS)
        config["paths"].update(
            {
                "qlib_provider": str(root / "provider"),
                "results_dir": str(root / "results"),
                "logs_dir": str(root / "logs"),
            }
        )
        for sub in ("provider", "results", "logs"):
            (root / sub).mkdir()
        config["dataset"]["segments"]["test"] = [
            dates[0].date().isoformat(),
            dates[-1].date().isoformat(),
        ]
        config["research"].update(overrides)

        raw = closes.stack(future_stack=True).rename("BACKTEST_CLOSE").to_frame()
        raw.index = raw.index.set_names(["datetime", "instrument"])
        handler = Mock()
        handler.fetch.return_value = raw
        dataset = Mock(handler=handler)
        model = Mock()
        model.model.best_iteration = 7
        data_provider = Mock()
        data_provider.calendar.return_value = list(dates)
        with patch.object(qlib_model, "init_qlib"), patch.object(
            qlib_model, "build_dataset", return_value=dataset
        ), patch.object(qlib_model, "train_lgbm", return_value=model), patch.object(
            qlib_model,
            "segment_prediction",
            side_effect=[(prediction.iloc[:0], labels.iloc[:0]), (prediction, labels)],
        ), patch.object(qlib.data.D, "_provider", data_provider):
            return qlib_model._run_qlib_stock_model(config)


def _metrics(result: dict[str, Any]) -> dict[str, float]:
    return {
        "net_annualized_return": result["net_annualized_return"],
        "test_max_drawdown": result["test_max_drawdown"],
        "mean_daily_turnover": result["mean_daily_turnover"],
        "legacy_return": result["model_quality_backtest"]["net_annualized_return"],
        "legacy_drawdown": result["model_quality_backtest"]["max_drawdown"],
        "legacy_turnover": result["model_quality_backtest"]["mean_daily_turnover"],
        "horizon_return": result["horizon_matched_backtest"]["net_annualized_return"],
        "horizon_drawdown": result["horizon_matched_backtest"]["max_drawdown"],
        "horizon_turnover": result["horizon_matched_backtest"]["mean_daily_turnover"],
        "benchmark_return": result["benchmark"]["annualized_return"],
        "benchmark_drawdown": result["benchmark"]["max_drawdown"],
    }


LEGACY = {"backtest_entry_lag_days": 0, "backtest_skip_limit_up": False}

# 改动前代码（commit 5b653dc）在同一场景上的实跑输出；改动后旧口径必须逐位复现。
LEGACY_EXPECTED: dict[str, float] = {
    "net_annualized_return": 0.07921579967097658,
    "test_max_drawdown": -0.0034384321668431417,
    "mean_daily_turnover": 0.002312169312169312,
    "legacy_return": 543.1857197527834,
    "legacy_drawdown": 0.0,
    "legacy_turnover": 0.6111111111111112,
    "horizon_return": 2.2037510212879536,
    "horizon_drawdown": -0.05061758475343148,
    "horizon_turnover": 0.03468253968253968,
    "benchmark_return": 1.8061337242206807,
    "benchmark_drawdown": -0.050617584753431144,
}


class LegacyCharacterizationTests(unittest.TestCase):
    def test_legacy_settings_reproduce_pre_change_backtest(self) -> None:
        result = _run(LEGACY)
        self.assertTrue(result["model_available"], result.get("error"))
        actual = _metrics(result)
        self.assertEqual(set(actual), set(LEGACY_EXPECTED))
        for key, expected in LEGACY_EXPECTED.items():
            self.assertTrue(
                math.isclose(actual[key], expected, rel_tol=0.0, abs_tol=1e-12),
                f"{key}: {actual[key]!r} != {expected!r}",
            )


class EntryReturnsAndTradabilityTests(unittest.TestCase):
    def _closes(self, columns: dict[str, list[float]]) -> pd.DataFrame:
        dates = pd.bdate_range("2024-03-01", periods=len(next(iter(columns.values()))))
        return pd.DataFrame(columns, index=dates)

    def test_legacy_settings_are_identity_with_old_formula(self) -> None:
        closes = self._closes({"SH600001": [10.0, 11.0, 10.5, 10.8]})
        future, entry_ok = qlib_model.entry_returns_and_tradability(
            closes, entry_lag_days=0, skip_limit_up=False
        )
        self.assertIsNone(entry_ok)
        pd.testing.assert_frame_equal(future, closes.shift(-1) / closes - 1.0)

    def test_lag_one_aligns_holding_window_with_label(self) -> None:
        closes = self._closes({"SH600001": [10.0, 11.0, 12.1, 12.1, 13.31]})
        future, _ = qlib_model.entry_returns_and_tradability(
            closes, entry_lag_days=1, skip_limit_up=False
        )
        # 信号日 t 承担 close(t+2)/close(t+1)-1：t=0 → 12.1/11 - 1。
        self.assertAlmostEqual(future.iloc[0, 0], 0.1)
        self.assertAlmostEqual(future.iloc[1, 0], 0.0)
        # H 日 cohort 复利 = close(t+1) → close(t+1+H)，与标签区间一致。
        horizon = 3
        compounded = float((1.0 + future.iloc[0:horizon, 0]).prod() - 1.0)
        self.assertAlmostEqual(compounded, closes.iloc[1 + horizon, 0] / closes.iloc[1, 0] - 1.0)

    def test_limit_up_threshold_by_board(self) -> None:
        cases = {
            "SH600519": 0.095,
            "SZ000001": 0.095,
            "SZ002594": 0.095,
            "600519": 0.095,
            "SZ300750": 0.195,
            "SZ301001": 0.195,
            "SH688981": 0.195,
            "SH689009": 0.195,
            "BJ920855": 0.295,
            "430047": 0.295,
            "835185": 0.295,
        }
        for instrument, expected in cases.items():
            self.assertEqual(qlib_model._limit_up_threshold(instrument), expected, instrument)

    def test_limit_up_on_entry_day_blocks_entry_by_board(self) -> None:
        # 第 1 日（lag=1 的入场日）各标的涨幅：主板 +10% 封板、主板 +9% 可买、
        # 创业板 +10% 可买（阈值 19.5%）、创业板 +20% 封板、北交所 +20% 可买。
        closes = self._closes(
            {
                "SH600001": [10.0, 11.0, 11.0],
                "SH600002": [10.0, 10.9, 11.0],
                "SZ300003": [10.0, 11.0, 11.0],
                "SZ300004": [10.0, 12.0, 12.0],
                "BJ920005": [10.0, 12.0, 12.0],
            }
        )
        _, entry_ok = qlib_model.entry_returns_and_tradability(
            closes, entry_lag_days=1, skip_limit_up=True
        )
        self.assertEqual(
            entry_ok.iloc[0].to_dict(),
            {
                "SH600001": False,
                "SH600002": True,
                "SZ300003": True,
                "SZ300004": False,
                "BJ920005": True,
            },
        )

    def test_suspension_and_resumption_day(self) -> None:
        # 第 1-2 日停牌（无 bar），第 3 日复牌 +10%（相对停牌前最后收盘）。
        closes = self._closes({"SH600001": [10.0, np.nan, np.nan, 11.0, 11.2]})
        _, entry_ok = qlib_model.entry_returns_and_tradability(
            closes, entry_lag_days=1, skip_limit_up=True
        )
        column = entry_ok["SH600001"].tolist()
        # 信号日 0、1 的入场日停牌；信号日 2 的入场日是复牌涨停；信号日 3 入场日正常。
        self.assertEqual(column[:4], [False, False, False, True])

    def test_suspension_blocks_entry_even_without_limit_check(self) -> None:
        closes = self._closes({"SH600001": [10.0, np.nan, 10.2]})
        _, entry_ok = qlib_model.entry_returns_and_tradability(
            closes, entry_lag_days=1, skip_limit_up=False
        )
        self.assertFalse(bool(entry_ok.iloc[0, 0]))


class ScenarioTests(unittest.TestCase):
    def test_defaults_use_executable_backtest_and_report_skips(self) -> None:
        result = _run({})
        self.assertTrue(result["model_available"], result.get("error"))
        self.assertEqual(result["backtest_entry_lag_days"], 1)
        self.assertTrue(result["backtest_skip_limit_up"])
        self.assertGreater(result["backtest_entry_candidates"], 0)
        self.assertGreater(result["backtest_entry_skipped"], 0)
        self.assertLess(result["backtest_entry_skipped"], result["backtest_entry_candidates"])
        # 场景里分数 = 次日收益 + 噪声：旧口径在信号日收盘成交，吃到了"被预测的那一天"；
        # 新口径从 t+1 收盘起算，这份虚高必须消失。
        legacy_one_day = result["model_quality_backtest"]["net_annualized_return"]
        self.assertLess(legacy_one_day, LEGACY_EXPECTED["legacy_return"] / 100.0)

    def test_legacy_mode_reports_no_entry_filtering(self) -> None:
        result = _run(LEGACY)
        self.assertEqual(result["backtest_entry_candidates"], 0)
        self.assertEqual(result["backtest_entry_skipped"], 0)

    def test_limit_up_skip_changes_result_beyond_timing_shift(self) -> None:
        lag_only = _metrics(_run({"backtest_entry_lag_days": 1, "backtest_skip_limit_up": False}))
        lag_and_skip = _metrics(_run({"backtest_entry_lag_days": 1, "backtest_skip_limit_up": True}))
        self.assertNotEqual(lag_only["horizon_return"], lag_and_skip["horizon_return"])


class CacheIdentityTests(unittest.TestCase):
    def test_execution_keys_change_model_cache_key(self) -> None:
        base = copy.deepcopy(DEFAULTS)
        identity = {"provider_fingerprint": {"digest": "x"}, "provider_end": "2026-10-09"}
        key, _ = qlib_model._model_cache_identity(base, **identity)
        for name, value in (("backtest_entry_lag_days", 0), ("backtest_skip_limit_up", False)):
            changed = copy.deepcopy(DEFAULTS)
            changed["research"][name] = value
            other, _ = qlib_model._model_cache_identity(changed, **identity)
            self.assertNotEqual(key, other, name)


if __name__ == "__main__":
    unittest.main()
