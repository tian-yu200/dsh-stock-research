# -*- coding: utf-8 -*-
"""Offline scale-basis detector (2026-09-10, read-only by default).

Finds the exact poison pattern that beat validate_daily_bars:
  open-gap small (inside ex-div confirmation band) BUT close-to-close large
  AND implied (amount/volume/close) normal.
Such rows are NOT ex-div (THS referee disagrees) — they are wrong-basis
splices. This script only REPORTS; it never modifies CSVs, so no golden
baseline A/B is required (validate_daily_bars semantics untouched).

Usage:
  python detect_scale_basis.py                     # scan all stocks, report
  python detect_scale_basis.py --symbols 000002 600094
  python detect_scale_basis.py --symbols-file suspects.txt
  python detect_scale_basis.py --json-out suspects.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
STOCK_DIR = ROOT / "data" / "stocks"


def _gap_threshold_for(symbol: str) -> float:
    s = str(symbol or "")
    if s[:1] in ("4", "8") or s[:2] == "92":
        return 0.33
    if s[:1] == "3" or s.startswith("688"):
        return 0.23
    if s:
        return 0.13
    return 0.30


def detect_frame(symbol: str, frame: pd.DataFrame) -> list[dict]:
    """Return suspect rows: gap small, close-jump big, implied normal."""
    df = frame.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)
    thr = _gap_threshold_for(symbol)
    hits: list[dict] = []
    prev = None
    for _, row in df.iterrows():
        try:
            o = float(row.get("open") or 0)
            c = float(row.get("close") or 0)
            v = float(row.get("volume") or 0)
            a = float(row.get("amount") or 0)
        except (TypeError, ValueError):
            prev = None
            continue
        ds = str(row.get("date"))[:10]
        if prev is not None and prev > 0 and c > 0 and o > 0:
            gap = abs(o / prev - 1.0)
            jump = abs(c / prev - 1.0)
            implied = (a / v / c) if (v and c) else None
            # implied normal = within 25% of 1.0 (same band as validate legs)
            implied_ok = implied is not None and implied == implied and abs(implied - 1.0) <= 0.25
            if gap <= thr and jump > thr and implied_ok:
                # skip long-halt resumes (>10 natural days), same as validate
                try:
                    day_gap = (pd.Timestamp(ds) - pd.Timestamp(prev_ds)).days
                except Exception:
                    day_gap = 1
                if day_gap <= 10:
                    hits.append({
                        "symbol": symbol,
                        "date": ds,
                        "gap": round(gap, 4),
                        "close_jump": round(jump, 4),
                        "threshold": thr,
                        "prev_close": round(prev, 2),
                        "open": round(o, 2),
                        "close": round(c, 2),
                    })
        prev = c or None
        prev_ds = ds
    return hits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--symbols-file", default=None)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--tail", type=int, default=120,
                        help="only scan the last N rows per file (0 = full)")
    args = parser.parse_args(argv)
    symbols: list[str] | None = list(args.symbols or [])
    if args.symbols_file:
        symbols.extend(
            line.strip() for line in Path(args.symbols_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    files = (
        [STOCK_DIR / f"{s}.csv" for s in symbols]
        if symbols
        else sorted(STOCK_DIR.glob("*.csv"))
    )
    all_hits: list[dict] = []
    scanned = 0
    for path in files:
        if not path.is_file():
            print(f"[skip] missing {path.name}")
            continue
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except Exception as exc:
            print(f"[skip] {path.name}: {type(exc).__name__}")
            continue
        if args.tail and len(df) > args.tail:
            df = df.tail(args.tail)
        hits = detect_frame(path.stem, df)
        scanned += 1
        for hit in hits:
            all_hits.append(hit)
            print(
                f"SUSPECT {hit['symbol']} {hit['date']} "
                f"gap={hit['gap']} jump={hit['close_jump']} thr={hit['threshold']}"
            )
    print(f"DETECT-DONE scanned={scanned} suspects={len(all_hits)}")
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps({"suspects": all_hits}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
