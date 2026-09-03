# -*- coding: utf-8 -*-
"""Read-only full-market price-health scan (no refetch, no writes except report)."""
from __future__ import annotations
import json, sys, datetime
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from price_guards import implied_deviation_flags, overnight_gap_flags, validate_daily_bars
ROOT = Path(__file__).resolve().parent.parent
STOCK_DIR = ROOT / "data" / "stocks"
def scan_one(path):
    try:
        df = pd.read_csv(path, usecols=["date", "open", "high", "low", "close", "volume", "amount"])
    except Exception as exc:
        return {"symbol": path.stem, "error": type(exc).__name__}
    df = df.tail(250).reset_index(drop=True)
    if len(df) < 30:
        return {"symbol": path.stem, "rows": len(df), "hits": 0}
    dev = implied_deviation_flags(df).fillna(False)
    gap = overnight_gap_flags(df, symbol=path.stem).fillna(False)
    joint = gap
    out = {"symbol": path.stem, "rows": len(df), "hits": int(joint.sum())}
    if out["hits"] > 0:
        try:
            _, _, reasons = validate_daily_bars(df, source="scan", symbol=path.stem)
        except Exception as exc:
            return {"symbol": path.stem, "rows": len(df), "hits": int(joint.sum()), "error": type(exc).__name__}
        out["confirmed"] = len(reasons)
        out["dates"] = [r["date"] for r in reasons][:10]
    return out
def main():
    files = sorted(STOCK_DIR.glob("[0-9]*.csv"))
    hits, errors, total = [], 0, 0
    for i, path in enumerate(files):
        total += 1
        r = scan_one(path)
        if r.get("error"):
            errors += 1
        elif r.get("confirmed", r.get("hits", 0)) > 0:
            hits.append(r)
        if (i + 1) % 1000 == 0:
            print("scan progress " + str(i + 1) + "/" + str(len(files)), flush=True)
    hits.sort(key=lambda x: x.get("confirmed", 0), reverse=True)
    ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    by_date = {}
    only_recent = 0
    for h in hits:
        ds = h.get("dates", [])
        for d in ds:
            by_date[d] = by_date.get(d, 0) + 1
        if ds and min(ds) >= "2026-08-27":
            only_recent += 1
    rep = {"scanned": total, "errors": errors, "suspect_symbols": len(hits), "only_recent_event": only_recent, "by_date": dict(sorted(by_date.items())), "symbols": [h["symbol"] for h in hits], "top": hits[:30]}
    outp = ROOT / "logs" / "codex-runs" / ("scan-" + ts + ".json")
    outp.write_text(json.dumps(rep, ensure_ascii=True, indent=2) + chr(10), encoding="utf-8")
    print("SCAN-DONE scanned=" + str(total) + " errors=" + str(errors) + " suspects=" + str(len(hits)))
    for h in hits[:30]:
        print("SUSPECT " + h["symbol"] + " confirmed=" + str(h.get("confirmed", 0)) + " dates=" + str(h.get("dates", []))[:160])
    print("REPORT " + str(outp))
if __name__ == "__main__":
    main()
