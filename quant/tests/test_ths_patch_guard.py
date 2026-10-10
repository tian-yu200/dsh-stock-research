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
