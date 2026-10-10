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

from fetch_utils import BaselineDiscontinuityError, merge_daily_frames
from market_data import UNIT_CONTRACT, normalize_daily_bars
from price_guards import UNIT_BIG_HI, UNIT_JUMP_HI, UNIT_JUMP_LO, detect_unit_breaks

COLS = ["date", "open", "high", "low", "close", "volume", "amount", "turnover"]
FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures" / "unit_guard"


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
        # T1：不再读生产目录，改读夹具（真实行切片，隔离区是唯一原始存档）。
        df = pd.read_csv(FIXTURES / "000875_shape.csv")
        _, reasons = detect_unit_breaks(df, source="t", symbol="000875")
        self.assertIn("2026-09-04", [r["date"] for r in reasons])

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
        # T1：改读夹具（单行字段异常，真实行切片）。
        df = pd.read_csv(FIXTURES / "300212_shape.csv")
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
        # T1：单位守卫不再删行。标记行保留在 CSV 中，字段被修正（volume x100
        # 落回 [low, high]）或置 NaN，且写入隔离区留档。
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
            dates = set(result["date"].astype(str))
            self.assertIn("2026-08-30", dates)  # 不再删行
            self.assertIn("2026-08-29", dates)
            row = result.loc[result["date"].astype(str) == "2026-08-30"].iloc[0]
            # 精确 /100 修正后 VWAP=amount/volume=10 落在 [10,10] 内 → volume=1e5
            self.assertEqual(float(row["volume"]), 100000.0)
            # turnover 与尾随 v/t 比值不一致 → 置 NaN（不发明）
            self.assertTrue(pd.isna(row["turnover"]))
            self.assertEqual(float(row["close"]), 10.0)
            qpath = tmp_path / "quarantine" / "T00001.quarantine.csv"
            self.assertTrue(qpath.exists())
            quar = pd.read_csv(qpath)
            self.assertTrue(quar["quarantine_reasons"].str.contains("unit_up_x10000").any())

    def test_real_000875_shape_raises_no_rows_deleted(self):
        # 000875 形态：历史段（股，系数约 1.0）+ 12 交易日缺口 + 尾段（手，系数约 1.448）。
        # 断言合并时抛 BaselineDiscontinuityError，且没有任何行被删。
        df = pd.read_csv(FIXTURES / "000875_shape.csv")
        df["date"] = pd.to_datetime(df["date"])
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = pathlib.Path(tmp) / "000875.csv"
            hist = normalize_daily_bars(df[df["date"] < "2026-09-04"], source="t")
            hist.to_csv(csv_path, index=False)
            incoming = normalize_daily_bars(df[df["date"] >= "2026-09-04"], source="t")
            with self.assertRaises(BaselineDiscontinuityError):
                merge_daily_frames(csv_path, incoming, source="t")
            # 原 CSV 未被改写（拒绝写入而非删行）。
            back = pd.read_csv(csv_path)
            self.assertEqual(len(back), len(hist))
            self.assertEqual(set(back["date"].astype(str)),
                             set(hist["date"].dt.strftime("%Y-%m-%d")))

    def test_real_300212_shape_row_kept_field_repaired(self):
        # 300212 形态：单行 volume 单位异常（~9x，非精确 x100）。
        # 断言该行保留，字段被修正或置为 NaN（不发明价格）。
        df = pd.read_csv(FIXTURES / "300212_shape.csv")
        df["date"] = pd.to_datetime(df["date"])
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = pathlib.Path(tmp) / "300212.csv"
            hist = normalize_daily_bars(df[df["date"] < "2024-09-27"], source="t")
            hist.to_csv(csv_path, index=False)
            incoming = normalize_daily_bars(df[df["date"] >= "2024-09-27"], source="t")
            result = merge_daily_frames(csv_path, incoming, source="t")
            dates = set(result["date"].astype(str))
            self.assertIn("2024-09-27", dates)  # 行被保留
            self.assertIn("2024-09-30", dates)
            bad = result.loc[result["date"].astype(str) == "2024-09-27"].iloc[0]
            # 非精确 x100（~9x）→ 不发明，volume 置 NaN，turnover 同置 NaN
            self.assertTrue(pd.isna(bad["volume"]))
            self.assertTrue(pd.isna(bad["turnover"]))
            # OHLC 必须原样保留
            self.assertEqual(float(bad["close"]), 175.89)

    def test_erosion_regression_merge_three_times_no_new_gaps(self):
        # 侵蚀回归：同一份数据连续合并 3 次（模拟 3 个交易日），
        # 行数单调不减，且不出现新缺口。
        df = pd.read_csv(FIXTURES / "erosion_regression.csv")
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = pathlib.Path(tmp) / "EROSION.csv"
            normalize_daily_bars(df, source="t").to_csv(csv_path, index=False)
            prev_len = len(df)
            for _ in range(3):
                incoming = normalize_daily_bars(df.tail(3), source="t")
                result = merge_daily_frames(csv_path, incoming, source="t")
                self.assertGreaterEqual(len(result), prev_len)
                # 交易日序列无新缺口（相邻自然日 gap > 10 视为缺口）
                dates = pd.to_datetime(result["date"])
                gaps = dates.diff().dt.days.fillna(1)
                self.assertFalse(bool((gaps > 10).any()))
                prev_len = len(result)

    def test_seam_limit_is_log2_scale(self):
        # 修正：close_step 是 |log2|，板块限幅必须换算到同一尺度。
        # 真 +10% 涨板 log2(1.10)=0.1375 不得被当成「超过板限」。
        from price_guards import basis_seam_limit, basis_seam_limit_step
        self.assertAlmostEqual(basis_seam_limit("000001"), 0.095, places=4)
        self.assertAlmostEqual(basis_seam_limit_step("000001"),
                               __import__("math").log2(1.095), places=4)
        self.assertGreater(basis_seam_limit_step("300212"),
                           basis_seam_limit_step("000001"))
        self.assertGreater(basis_seam_limit_step("920779"),
                           basis_seam_limit_step("300212"))

    def test_seam_limit_move_written_not_held(self):
        # 真涨停（close +10%，VWAP 同步 +10%）不是候选，不得 hold/raise。
        df = pd.read_csv(FIXTURES / "erosion_regression.csv")
        df["date"] = pd.to_datetime(df["date"])
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = pathlib.Path(tmp) / "LIMITUP.csv"
            hist = normalize_daily_bars(df, source="t")
            hist.to_csv(csv_path, index=False)
            last = hist.iloc[-1].copy()
            up = last.copy()
            up["date"] = pd.Timestamp("2026-09-04")
            up["open"] = up["high"] = up["low"] = up["close"] = float(last["close"]) * 1.10
            up["volume"] = float(last["volume"])
            up["amount"] = float(last["amount"]) * 1.10  # VWAP 同步
            up["turnover"] = float(last["turnover"])
            incoming = normalize_daily_bars(pd.DataFrame([up]), source="t")
            result = merge_daily_frames(csv_path, incoming, source="t")
            self.assertIn("2026-09-04", set(result["date"].astype(str)))


if __name__ == "__main__":
    unittest.main()
