# -*- coding: utf-8 -*-
"""T1b: ths_patch_recent must fail closed -- no blind 7-column overwrite.

rootcause-0928: THS /01 (unadjusted, lot units) rows were written over local
hfq rows without conversion, then deleted by the unit guard, leaving 0.84x
close seams (~183 symbols). These tests pin the required behaviour:
1. unconverted/unprovable THS rows are REFUSED (local row kept) and a JSON
   line lands in the conflict log (durable, not stdout/stderr);
2. a genuine price-scale fix still applies when the conversion is provable.
"""
from __future__ import annotations

import json
import pathlib
import sys

import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import price_guards

F_REF = 2.0  # implied factor close*volume/amount of the local series
TOTAL_SHARES = 1.0e9


def _local_frame(n=30, *, corrupted=()):
    dates = pd.bdate_range("2026-09-01", periods=n).strftime("%Y-%m-%d").tolist()
    rows = []
    for i, d in enumerate(dates):
        close = 2000.0 + 10.0 * i
        volume = 1_000_000.0
        amount = close * volume / F_REF
        rows.append({
            "date": d,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": volume,
            "amount": amount,
            "turnover": volume / TOTAL_SHARES,
        })
    df = pd.DataFrame(rows)
    for i in corrupted:
        df.loc[i, "close"] *= 0.1  # local price-scale corruption on close only
    return df


def _ths_unadjusted_lot_frame(local):
    """THS /01 style: unadjusted basis (0.5x) + volume in lots + turnover x100."""
    ths = local.copy()
    for col in ("open", "high", "low", "close"):
        ths[col] = local[col] / 2.0
    ths["volume"] = local["volume"] / 100.0  # lots, not shares
    ths["amount"] = local["amount"]  # yuan, unchanged
    ths["turnover"] = local["turnover"] * 100.0  # c[7]/100 style parser scale
    return ths


def test_ths_patch_refuses_unproven_basis_and_units(monkeypatch, tmp_path):
    local = _local_frame()
    ths = _ths_unadjusted_lot_frame(local)
    log = tmp_path / "ths-conflicts.log"
    monkeypatch.setattr(price_guards, "THS_CONFLICT_LOG", log)
    monkeypatch.setattr(price_guards, "fetch_ths_daily", lambda symbol: ths.copy())

    out, report = price_guards.ths_patch_recent("000001", local.copy())

    # No blind overwrite: every local value is kept as-is.
    pd.testing.assert_frame_equal(out, local)
    assert report["patched"] == 0
    assert report["refused"] >= 1

    # Durable conflict trace: one JSON line, silent-error helper.
    lines = log.read_text(encoding="utf-8").splitlines()
    assert lines, "refusal must append a JSON line to the conflict log"
    rec = json.loads(lines[0])
    assert rec["action"] == "refused"
    assert rec["reason"] == "basis_or_units_unproven"
    assert rec["symbol"] == "000001"


def test_ths_patch_applies_proven_price_scale_fix(monkeypatch, tmp_path):
    corrupted_idx = (27, 28, 29)
    local = _local_frame(n=30, corrupted=corrupted_idx)
    ths = _local_frame(n=30)  # THS holds the true scale on every date
    ths["turnover"] = ths["turnover"] * 100.0  # still at parser c[7]/100 scale
    log = tmp_path / "ths-conflicts.log"
    monkeypatch.setattr(price_guards, "THS_CONFLICT_LOG", log)
    monkeypatch.setattr(price_guards, "fetch_ths_daily", lambda symbol: ths.copy())

    out, report = price_guards.ths_patch_recent("000001", local.copy())

    assert report["patched"] == len(corrupted_idx)
    # Corrupted closes are repaired to the true scale (basis conversion = 1.0,
    # volume/amount proven unchanged, turnover proven at 0.01 parser scale).
    for i in corrupted_idx:
        assert out.loc[i, "close"] == pytest.approx(2000.0 + 10.0 * i)
        assert out.loc[i, "turnover"] == pytest.approx(local.loc[i, "turnover"])
    # Good rows (close ratio <= threshold) are untouched.
    good = [i for i in range(len(local)) if i not in corrupted_idx]
    pd.testing.assert_frame_equal(out.loc[good].reset_index(drop=True),
                                  local.loc[good].reset_index(drop=True))

    lines = log.read_text(encoding="utf-8").splitlines()
    recs = [json.loads(x) for x in lines]
    assert [r for r in recs if r["action"] == "patched"]


def test_ths_patch_refuses_real_000001_20260928_row(monkeypatch, tmp_path):
    """N3 regression: the real quarantine row must be refused, with a log line.

    THS side is the exact 000001 2026-09-28 quarantine values
    (rootcause-0928 §1.5 / t1b-review §1.3: close 1878.37, volume 715341,
    amount 810575796.4, turnover 0.37, OHLC from the quarantine CSV).
    Local side is the real 09-17..09-24 neighbours (rootcause-0928 §1.2),
    whose segment implied factor is ~1.96 and volume/turnover ~1.93e8
    (hermes: ref_factor=1.9598, ref_shares=193,299,815). The 09-28 local
    pre-patch close sits >20% away from the THS close so the patch trigger
    fires and the proof runs: iths/ref = 1.658/1.96 = 0.846 > tolerance
    on every grid cell -> proof None -> refused.
    """
    neighbours = [
        ("2026-09-17", 2275.64, 691925.0, 806153100.0, 0.0036),
        ("2026-09-18", 2289.19, 853038.0, 999918100.0, 0.0044),
        ("2026-09-21", 2293.71, 817555.0, 954184700.0, 0.0042),
        ("2026-09-22", 2290.70, 759457.0, 887703100.0, 0.0039),
        ("2026-09-23", 2274.14, 906726.0, 1052015100.0, 0.0047),
        ("2026-09-24", 2266.46, 1043819.0, 1186736900.0, 0.0054),
    ]
    rows = [{"date": d, "open": c, "high": c, "low": c, "close": c,
             "volume": v, "amount": a, "turnover": t}
            for d, c, v, a, t in neighbours]
    # Target row: local pre-patch hfq scale (close differs >20% from the THS
    # value so the patch trigger fires); excluded from segment evidence.
    rows.append({"date": "2026-09-28", "open": 2400.0, "high": 2410.0,
                 "low": 2390.0, "close": 2400.0, "volume": 1000000.0,
                 "amount": 2400.0 * 1000000.0 / 1.96, "turnover": 0.0052})
    rows.append({"date": "2026-09-29", "open": 2410.0, "high": 2420.0,
                 "low": 2400.0, "close": 2410.0, "volume": 1000000.0,
                 "amount": 2410.0 * 1000000.0 / 1.96, "turnover": 0.0052})
    local = pd.DataFrame(rows)
    ths = pd.DataFrame([{
        "date": "2026-09-28", "open": 1875.87, "high": 1892.13,
        "low": 1874.62, "close": 1878.37, "volume": 715341.0,
        "amount": 810575796.4, "turnover": 0.37,
    }])
    log = tmp_path / "ths-conflicts.log"
    monkeypatch.setattr(price_guards, "THS_CONFLICT_LOG", log)
    monkeypatch.setattr(price_guards, "fetch_ths_daily", lambda symbol: ths.copy())

    # Segment evidence matches hermes's real-data numbers (~1.96 / ~1.93e8).
    ref_factor, ref_shares = price_guards._ths_segment_evidence(local, 6)
    assert ref_factor == pytest.approx(1.96, abs=0.01)
    assert ref_shares == pytest.approx(193_299_815, rel=0.02)

    out, report = price_guards.ths_patch_recent("000001", local.copy())

    # Refusal: local row untouched, real THS values never written.
    pd.testing.assert_frame_equal(out, local)
    assert report["patched"] == 0
    assert report["refused"] == 1

    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["action"] == "refused"
    assert rec["reason"] == "basis_or_units_unproven"
    assert rec["date"] == "2026-09-28"
    assert rec["close_ths"] == 1878.37
    assert rec["close_local"] == 2400.0
