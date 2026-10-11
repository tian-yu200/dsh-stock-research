# -*- coding: utf-8 -*-
"""T10: ETF gap threshold (``price_guards.gap_threshold_for``).

Root cause (``quant/logs/codex-runs/etf-discont-rca.md``, accepted): the
classifier had **no ETF branch**, so every SH/SZ fund code fell through to the
main-board 0.13 threshold. 2024-10-08/09 were real +/-20% limit days for
ChiNext/STAR-tracking ETFs (125 flagged rows binned: 71 in (0.13, 0.20],
54 > 0.20), so the price guard called genuine moves "scale breaks" -> raise ->
full refetch -> the *same* upstream series -> raise again: 83 ETFs never
converged. Fix: flat 0.23 for the SH/SZ fund families 15/16/18/50/51/52/56/58.

Every test here is bounded and offline (synthetic frames only, no network, no
filesystem beyond a tmp_path in the merge tests).
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
from market_data import normalize_daily_bars
from price_guards import (
    OVERNIGHT_GAP_THRESHOLD,
    detect_unit_breaks,
    gap_threshold_for,
    repair_unit_breaks,
    triage_price_flags,
    validate_daily_bars,
)

COLS = ["date", "open", "high", "low", "close", "volume", "amount", "turnover"]

# SH/SZ fund families (production: data/funds/etf_<code>.csv -> stem is bare).
ETF_CODES = [
    "588000",  # 科创50
    "159915",  # 创业板
    "510300",  # 沪深300
    "513100",  # 纳指 (QDII; documented residual, still 0.23 flat)
    "512690",  # 酒
    "515180",  # 红利
    "520880",  # 港股通/跨境
    "560000",  # 沪市新段
    "161725",  # 深市 LOF
    "184801",  # 深市封基
]


def flat_bars(start: str, n: int, price: float = 1.5,
              volume: float = 1_000_000.0, turnover: float = 2.0) -> pd.DataFrame:
    """Production-shaped frame: consecutive business days, self-consistent
    amount = close * volume, constant volume/turnover (unit guard stays quiet)."""
    dates = pd.bdate_range(start, periods=n).strftime("%Y-%m-%d").tolist()
    rows = []
    for d in dates:
        rows.append([d, price, price, price, price, volume,
                     price * volume, turnover])
    return pd.DataFrame(rows, columns=COLS)


class GapThresholdEtfTests(unittest.TestCase):
    # ---- 1. RED: the new ETF branch -------------------------------------
    def test_etf_branch_thresholds(self):
        for code in ETF_CODES:
            self.assertEqual(gap_threshold_for(code), 0.23,
                             "%s must use the fund threshold" % code)
        # defensive forms: fetch_state asset key + exchange-prefixed
        for form in ("etf_588000", "sh510300", "sz159915"):
            self.assertEqual(gap_threshold_for(form), 0.23,
                             "%s must normalise to the fund threshold" % form)

    # ---- 2. regression lock: stocks byte-for-byte unchanged ------------
    def test_stock_thresholds_unchanged(self):
        for code in ("000001", "600519", "601225", "000002"):
            self.assertEqual(gap_threshold_for(code), 0.13)
        for code in ("300212", "301336", "688496"):
            self.assertEqual(gap_threshold_for(code), 0.23)
        for code in ("430047", "830799", "920001"):
            self.assertEqual(gap_threshold_for(code), 0.33)
        self.assertEqual(gap_threshold_for(""), OVERNIGHT_GAP_THRESHOLD)
        self.assertEqual(gap_threshold_for(None), OVERNIGHT_GAP_THRESHOLD)

    # ---- 3. RED: a real 20% ETF limit day is no longer "a scale break" --
    def test_validate_etf_real_20pct_day_passes(self):
        df = flat_bars("2024-09-02", 60)          # index 26 == 2024-10-08
        idx = 26
        prev_close = float(df.loc[idx - 1, "close"])
        for col in ("open", "high", "low", "close"):
            df.loc[idx, col] = round(prev_close * 1.20, 4)   # genuine +20% day
        df.loc[idx, "amount"] = float(df.loc[idx, "close"]) * float(df.loc[idx, "volume"])
        # trailing frontier row (the merge target of the real incident)
        tail = df.iloc[-1].copy()
        tail["date"] = "2026-10-09"
        df = pd.concat([df, pd.DataFrame([tail])], ignore_index=True)

        _clean, quar, reasons = validate_daily_bars(df, source="t", symbol="588000")
        self.assertEqual(len(quar), 0,
                         "a real +20%% ETF day must not be quarantined: %s" % reasons)
        # and the ETF-specific threshold is what makes it pass
        self.assertGreater(0.20, 0.13)
        self.assertLess(0.20, gap_threshold_for("588000"))

    # ---- 4. GREEN-locked: corruption is still caught -------------------
    def test_corrupt_etf_still_caught(self):
        # (a) persistent >23% scale zone (588xxx-like) -> still raise, never delete
        # NOTE: normalise first, exactly like the production path
        # (fetch_utils.py:320 passes the *normalised* result into the guard), so
        # the date dtypes match between the frame and the returned quarantine rows.
        df = normalize_daily_bars(flat_bars("2024-09-02", 24), source="t")
        base = float(df.loc[19, "close"])
        for idx in (20, 21, 22, 23):          # a new level that persists
            for col in ("open", "high", "low", "close"):
                df.loc[idx, col] = round(base * 3.0, 4)
            df.loc[idx, "amount"] = float(df.loc[idx, "close"]) * float(df.loc[idx, "volume"])
        _clean, bad, _r = validate_daily_bars(df, source="t", symbol="588000")
        self.assertGreaterEqual(len(bad), 2, "a persistent scale break must be flagged")
        mask = df["date"].isin(bad["date"])
        raise_m, delete_m, _hold_m = triage_price_flags(df, mask)
        self.assertEqual(int(raise_m.sum()), len(bad),
                         "every flagged row of a scale break must be raised")
        self.assertEqual(int(delete_m.sum()), 0,
                         "delete is reserved for isolated single spikes")

        # (b) 159xxx-like unit defect (amount x10^4) -> unit guard repairs, no row loss
        df2 = normalize_daily_bars(flat_bars("2024-09-02", 24), source="t")
        idx = 20
        df2.loc[idx, "turnover"] = 2.0 * 1e4
        df2.loc[idx, "amount"] = float(df2.loc[idx, "close"]) * float(df2.loc[idx, "volume"]) * 1e4
        flags, reasons = detect_unit_breaks(df2, source="t", symbol="159541")
        # The defect fires in BOTH directions: the broken row itself (unit_down)
        # and the row that returns to the normal level (unit_up). Assert the
        # content, not an exact count.
        self.assertGreaterEqual(int(flags.sum()), 1)
        self.assertIn("unit_down_x10000", [r for rec in reasons for r in rec["reasons"]])
        self.assertTrue(bool(flags.loc[idx]))
        out = repair_unit_breaks(df2, flags, source="t", symbol="159541")
        self.assertEqual(len(out), len(df2), "repair must never drop a row")
        for col in ("open", "high", "low", "close"):
            self.assertEqual(list(out[col]), list(df2[col]), "OHLC must be preserved")
        for pos in df2.index[flags]:
            vol = out.loc[pos, "volume"]
            if pd.notna(vol) and float(vol) > 0:
                vwap = float(out.loc[pos, "amount"]) / float(vol)
                self.assertTrue(float(out.loc[pos, "low"]) <= vwap <= float(out.loc[pos, "high"]))
            else:
                self.assertTrue(pd.isna(vol))

    # ---- 5. integration: cold start converges / QDII residual documented
    def test_merge_cold_start_etf_converges(self):
        df = flat_bars("2024-09-02", 60)
        idx = 26
        prev_close = float(df.loc[idx - 1, "close"])
        for col in ("open", "high", "low", "close"):
            df.loc[idx, col] = round(prev_close * 1.20, 4)
        df.loc[idx, "amount"] = float(df.loc[idx, "close"]) * float(df.loc[idx, "volume"])
        tail = df.iloc[-1].copy()
        tail["date"] = "2026-10-09"
        df = pd.concat([df, pd.DataFrame([tail])], ignore_index=True)

        with tempfile.TemporaryDirectory(prefix="t10-cold-") as directory:
            path = pathlib.Path(directory) / "etf_588000.csv"
            merged = merge_daily_frames(path, df, source="ETF 588000")
        self.assertEqual(str(merged["date"].iloc[-1])[:10], "2026-10-09")

    def test_merge_cold_start_qdii_26pct_still_raises(self):
        """Honest residual: a genuine >23% gap (513xxx-style, no price limit)
        is still refused -- documented in the plan, not a regression."""
        df = flat_bars("2024-09-02", 24)
        for idx in (20, 21):
            prev_close = float(df.loc[idx - 1, "close"])
            for col in ("open", "high", "low", "close"):
                df.loc[idx, col] = round(prev_close * 1.26, 4)
            df.loc[idx, "amount"] = float(df.loc[idx, "close"]) * float(df.loc[idx, "volume"])
        with tempfile.TemporaryDirectory(prefix="t10-qdii-") as directory:
            path = pathlib.Path(directory) / "etf_513100.csv"
            with self.assertRaises(BaselineDiscontinuityError):
                merge_daily_frames(path, df, source="ETF 513100")


if __name__ == "__main__":
    unittest.main()
