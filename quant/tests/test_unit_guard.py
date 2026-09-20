# -*- coding: utf-8 -*-
"""单位守卫测试（2026-08-28 volume/turnover 口径切换事故后增设）。

覆盖：纯单位跳变隔离 / 正常波动放行 / 000783 式全历史一致手口径放行 /
除权跳空不误判 / merge_daily_frames 钩子隔离。
"""
from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from fetch_utils import merge_daily_frames
from market_data import UNIT_CONTRACT, normalize_daily_bars
from price_guards import UNIT_BIG_HI, UNIT_JUMP_HI, UNIT_JUMP_LO, detect_unit_breaks

COLS = ["date", "open", "high", "low", "close", "volume", "amount", "turnover"]


def bars(start, n, price, vol, to, vol_walk=None):
    rows = []
    dates = pd.bdate_range(start, periods=n).strftime("%Y-%m-%d").tolist()
    for i, d in enumerate(dates):
        v = vol * (vol_walk[i] if vol_walk else 1.0)
        t = to * (vol_walk[i] if vol_walk else 1.0)
        rows.append([d, price, price, price, price, v, price * v, t])
    return pd.DataFrame(rows, columns=COLS)


class UnitGuardTests(unittest.TestCase):
    def test_contract_constants(self):
        self.assertEqual(UNIT_CONTRACT, {"volume": "shares", "turnover": "decimal-fraction"})
        self.assertEqual((UNIT_JUMP_HI, UNIT_JUMP_LO), (50.0, 0.02))
        self.assertEqual(UNIT_BIG_HI, 2000.0)

    def test_pure_unit_jump_flagged(self):
        df = bars("2026-08-01", 10, 10.0, 100000.0, 2.0)
        bad = df.iloc[[-1]].copy()
        bad["volume"] = bad["volume"] * 100.0
        bad["turnover"] = bad["turnover"] / 100.0
        df = pd.concat([df.iloc[:-1], bad], ignore_index=True)
        flags, reasons = detect_unit_breaks(df, source="t", symbol="X")
        self.assertEqual(int(flags.sum()), 1)
        self.assertTrue(flags.iloc[-1])
        self.assertEqual(reasons[0]["reasons"], ["unit_up_x10000"])

    def test_single_field_jump_flagged(self):
        df = bars("2026-08-01", 10, 10.0, 100000.0, 2.0)
        df.loc[df.index[-1], "turnover"] = df["turnover"].iloc[-1] / 100.0
        flags, reasons = detect_unit_breaks(df, source="t", symbol="X")
        self.assertEqual(int(flags.sum()), 1)
        self.assertEqual(reasons[0]["reasons"], ["unit_up_x100"])

    def test_normal_volatility_clean(self):
        walk = [1.0, 1.2, 0.85, 1.1, 0.9, 1.25, 0.8, 1.05, 0.95, 1.15,
                0.88, 1.02, 1.18, 0.92, 1.07, 0.97, 1.12, 0.86, 1.09, 0.99]
        df = bars("2026-08-01", 20, 10.0, 100000.0, 2.0, vol_walk=walk)
        flags, reasons = detect_unit_breaks(df, source="t", symbol="X")
        self.assertEqual(int(flags.sum()), 0)
        self.assertEqual(reasons, [])

    def test_consistent_hand_unit_clean(self):
        # 000783 式：全历史手口径（v/t 恒定 5.53e7），日环比 ~1，从不发射。
        walk = [1.0 + 0.05 * ((i * 37) % 11 - 5) for i in range(60)]
        df = bars("2026-01-05", 60, 50.0, 553000.0, 0.01, vol_walk=walk)
        flags, _ = detect_unit_breaks(df, source="t", symbol="000783")
        self.assertEqual(int(flags.sum()), 0)

    def test_exdiv_gap_not_flagged(self):
        df = bars("2026-08-01", 10, 20.0, 100000.0, 0.01)
        last = df.iloc[[-1]].copy()
        last[["open", "high", "low", "close"]] = 10.0
        last["volume"] = last["volume"] * 2.0
        df = pd.concat([df.iloc[:-1], last], ignore_index=True)
        flags, _ = detect_unit_breaks(df, source="t", symbol="X")
        self.assertEqual(int(flags.sum()), 0)

    def test_band_fires_when_amount_flat(self):
        # 300212 型 32x：volume 小动、turnover 大动、amount 跟 volume → 发射。
        df = bars("2026-08-01", 10, 10.0, 100000.0, 2.0)
        last = df.iloc[[-1]].copy()
        last["volume"] = last["volume"] * 2.26
        last["amount"] = last["amount"] * 2.59
        last["turnover"] = last["turnover"] * 72.8
        df = pd.concat([df.iloc[:-1], last], ignore_index=True)
        flags, reasons = detect_unit_breaks(df, source="t", symbol="X")
        self.assertEqual(int(flags.sum()), 1)
        self.assertEqual(reasons[0]["reasons"], ["unit_down_30_50_amtflat"])

    def test_band_quiet_when_amount_synced(self):
        # 同 40x 跳变但 amount 同步放大 → 真实交易，不发射。
        df = bars("2026-08-01", 10, 10.0, 100000.0, 2.0)
        last = df.iloc[[-1]].copy()
        last["volume"] = last["volume"] * 40.0
        last["amount"] = last["amount"] * 40.0
        df = pd.concat([df.iloc[:-1], last], ignore_index=True)
        flags, _ = detect_unit_breaks(df, source="t", symbol="X")
        self.assertEqual(int(flags.sum()), 0)

    def test_band_missing_amount_never_fires(self):
        df = bars("2026-08-01", 10, 10.0, 100000.0, 2.0)
        df["amount"] = float("nan")
        last = df.iloc[[-1]].copy()
        last["volume"] = last["volume"] * 2.26
        last["turnover"] = last["turnover"] * 72.8
        df = pd.concat([df.iloc[:-1], last], ignore_index=True)
        flags, _ = detect_unit_breaks(df, source="t", symbol="X")
        self.assertEqual(int(flags.sum()), 0)

    def test_halt_gap_exempt(self):
        # 真停牌复牌：15 天正常历史（隐含均价 ~10）+ 6 个月 gap + 量价同动复牌
        # （隐含 ~11，自洽）→ 走真实自洽路径豁免，不发射。
        hist = bars("2026-01-05", 15, 10.0, 100000.0, 0.01)
        resume = pd.DataFrame(
            [["2026-07-06", 11.0, 11.0, 11.0, 11.0, 10000000.0, 110000000.0, 0.01]],
            columns=COLS)
        df = pd.concat([hist, resume], ignore_index=True)
        flags, _ = detect_unit_breaks(df, source="t", symbol="HALT")
        self.assertEqual(int(flags.sum()), 0)

    def test_halt_gap_short_history_fallback(self):
        # 历史不足（trail 中位不可算）→ 沿用旧行为豁免，不发射。
        df = pd.DataFrame({
            "date": ["2026-01-05", "2026-01-06", "2026-07-06"],
            "open": [10.0, 10.0, 11.0], "high": [10.0, 10.0, 11.0],
            "low": [10.0, 10.0, 11.0], "close": [10.0, 10.0, 11.0],
            "volume": [100000.0, 100000.0, 10000000.0],
            "amount": [1000000.0, 1000000.0, 110000000.0],
            "turnover": [0.01, 0.01, 0.01]})
        flags, _ = detect_unit_breaks(df, source="t", symbol="HALT")
        self.assertEqual(int(flags.sum()), 0)

    def test_halt_gap_fake_suspension_not_exempt(self):
        # R1 回归：000875 型。11 天 gap（数据缺口）首行 volume 低估 100x，
        # 隐含均价偏离 100x（>X=5）→ 不得豁免，必须发射 unit_down_x100。
        hist = bars("2026-08-01", 15, 10.0, 100000.0, 0.01)
        bad = pd.DataFrame(
            [["2026-08-28", 10.0, 10.0, 10.0, 10.0, 1000.0, 1000000.0, 0.01]],
            columns=COLS)
        df = pd.concat([hist, bad], ignore_index=True)
        flags, reasons = detect_unit_breaks(df, source="t", symbol="FAKEGAP")
        self.assertEqual(int(flags.sum()), 1)
        self.assertEqual(reasons[0]["date"], "2026-08-28")
        self.assertEqual(reasons[0]["reasons"], ["unit_down_x100"])

    def test_real_000875_gap_row_flagged(self):
        path = ROOT / "data" / "stocks" / "000875.csv"
        if not path.exists():
            self.skipTest("000875.csv not present")
        df = pd.read_csv(path)
        _, reasons = detect_unit_breaks(df, source="t", symbol="000875")
        self.assertIn("2026-09-14", [r["date"] for r in reasons])

    def test_real_920427_exempt(self):
        path = ROOT / "data" / "stocks" / "920427.csv"
        if not path.exists():
            self.skipTest("920427.csv not present")
        df = pd.read_csv(path)
        _, reasons = detect_unit_breaks(df, source="t", symbol="920427")
        self.assertNotIn("2021-02-05", [r["date"] for r in reasons])

    def test_real_920726_exempt(self):
        path = ROOT / "data" / "stocks" / "920726.csv"
        if not path.exists():
            self.skipTest("920726.csv not present")
        df = pd.read_csv(path)
        _, reasons = detect_unit_breaks(df, source="t", symbol="920726")
        self.assertNotIn("2021-05-27", [r["date"] for r in reasons])

    def test_real_300212_caught(self):
        path = ROOT / "data" / "stocks" / "300212.csv"
        if not path.exists():
            self.skipTest("300212.csv not present")
        df = pd.read_csv(path)
        flags, reasons = detect_unit_breaks(df, source="t", symbol="300212")
        dates = [r["date"] for r in reasons]
        self.assertIn("2024-09-27", dates)

    def test_real_000783_not_flagged(self):
        path = ROOT / "data" / "stocks" / "000783.csv"
        if not path.exists():
            self.skipTest("000783.csv not present")
        df = pd.read_csv(path)
        flags, _ = detect_unit_breaks(df, source="t", symbol="000783")
        self.assertEqual(int(flags.sum()), 0)

    def test_merge_hook_quarantines_unit_break(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            hist = bars("2026-08-01", 20, 10.0, 100000.0, 2.0)
            csv_path = tmp_path / "T00001.csv"
            normalize_daily_bars(hist, source="t").to_csv(csv_path, index=False)
            incoming = pd.DataFrame(
                [["2026-08-29", 10.0, 10.0, 10.0, 10.0, 100000.0, 1000000.0, 2.0],
                 ["2026-08-30", 10.0, 10.0, 10.0, 10.0, 100000.0 * 100.0, 1000000.0, 2.0 / 100.0]],
                columns=COLS)
            result = merge_daily_frames(csv_path, normalize_daily_bars(incoming, source="t"),
                                        source="t")
            self.assertNotIn("2026-08-30", set(result["date"].astype(str)))
            self.assertIn("2026-08-29", set(result["date"].astype(str)))
            qpath = tmp_path / "quarantine" / "T00001.quarantine.csv"
            self.assertTrue(qpath.exists())
            quar = pd.read_csv(qpath)
            self.assertTrue(quar["quarantine_reasons"].str.contains("unit_up_x10000").any())


if __name__ == "__main__":
    unittest.main()
