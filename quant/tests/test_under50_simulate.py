"""under50 模拟账户的“不追价”规则（任务书 T4，2026-10-10）。

报告文本（write_report L350）声称“任一拟买股票开盘较信号收盘上涨超过 5%、
停牌或无法成交，当次不追价”，但 simulate_account 此前只要开盘价是有限数就
按开盘价买入。本测试锁定三种跳过情形与一种正常买入：

1. 开盘价缺失（NaN / 无 bar）；
2. 开盘价较信号日收盘涨幅超过 5%（追价）；
3. 开盘即触及涨停（复用 ``qlib_model.limit_up_threshold``，主�� 0.095）；
4. 正常小幅高开 → 照常买入。

跳过时在 trades 里记一条 ``side="skip_buy"`` 记录（带 reason），不动现金；
``metrics["trade_orders"]`` 只统计真实买卖订单，不含跳过记录。
"""
from __future__ import annotations

import pathlib
import sys
import unittest

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_under50_research import simulate_account  # noqa: E402

SYMBOL = "SH600001"
SIGNAL_DATE = pd.Timestamp("2026-01-30")
EXEC_DATE = pd.Timestamp("2026-02-02")
END_DATE = pd.Timestamp("2026-02-03")
SIGNAL_CLOSE = 10.0
INITIAL_CASH = 6000.0

CONFIG = {
    "initial_cash": INITIAL_CASH,
    "open_cost": 0.0003,
    "close_cost": 0.0013,
    "min_cost": 5.0,
    "lot_size": 100,
    "max_positions": 1,
    "max_position_weight": 1.0,
    "reserve_cash_ratio": 0.0,
    "candidate_depth": 1,
}


def _prediction() -> pd.Series:
    index = pd.MultiIndex.from_tuples(
        [(SYMBOL, SIGNAL_DATE)], names=["symbol", "datetime"]
    )
    return pd.Series([1.0], index=index, dtype=float)


def _closes() -> pd.DataFrame:
    return pd.DataFrame(
        {SYMBOL: [SIGNAL_CLOSE, SIGNAL_CLOSE, SIGNAL_CLOSE * 1.02]},
        index=pd.DatetimeIndex([SIGNAL_DATE, EXEC_DATE, END_DATE], name="date"),
    )


def _opens(exec_open: float) -> pd.DataFrame:
    return pd.DataFrame(
        {SYMBOL: [SIGNAL_CLOSE, exec_open, SIGNAL_CLOSE * 1.02]},
        index=pd.DatetimeIndex([SIGNAL_DATE, EXEC_DATE, END_DATE], name="date"),
    )


class Under50NoChaseTests(unittest.TestCase):
    def _run(self, exec_open: float):
        return simulate_account(_prediction(), _opens(exec_open), _closes(), CONFIG)

    def _assert_skipped(self, trades: pd.DataFrame, metrics: dict, reason: str) -> None:
        self.assertFalse((trades["side"] == "buy").any(), "跳过时不得产生买单")
        skips = trades[trades["side"] == "skip_buy"]
        self.assertEqual(len(skips), 1)
        row = skips.iloc[0]
        self.assertEqual(row["symbol"], SYMBOL)
        self.assertEqual(row["trade_date"], EXEC_DATE)
        self.assertEqual(row["quantity"], 0)
        self.assertEqual(row["reason"], reason)
        self.assertEqual(metrics["trade_orders"], 0)
        self.assertAlmostEqual(metrics["ending_equity"], INITIAL_CASH)

    def test_missing_open_is_skipped(self) -> None:
        _, trades, metrics = self._run(float("nan"))
        self._assert_skipped(trades, metrics, "missing_open")

    def test_open_gap_over_5pct_is_skipped(self) -> None:
        # +6%：超过 5% 追价线但未达涨停阈值（主板 0.095）
        _, trades, metrics = self._run(SIGNAL_CLOSE * 1.06)
        self._assert_skipped(trades, metrics, "open_gap_over_5pct")

    def test_open_at_limit_up_is_skipped(self) -> None:
        # +10%：达到主板涨停阈值 0.095（Qlib limit_threshold 惯例）
        _, trades, metrics = self._run(SIGNAL_CLOSE * 1.10)
        self._assert_skipped(trades, metrics, "limit_up")

    def test_normal_small_gap_buys(self) -> None:
        _, trades, metrics = self._run(SIGNAL_CLOSE * 1.02)
        self.assertFalse((trades["side"] == "skip_buy").any(), "正常买入不得记跳过")
        buys = trades[trades["side"] == "buy"]
        self.assertEqual(len(buys), 1)
        self.assertGreaterEqual(int(buys.iloc[0]["quantity"]), 100)
        self.assertAlmostEqual(float(buys.iloc[0]["price"]), SIGNAL_CLOSE * 1.02)
        self.assertEqual(metrics["trade_orders"], 1)
        self.assertLess(metrics["ending_equity"], INITIAL_CASH)


if __name__ == "__main__":
    unittest.main()
