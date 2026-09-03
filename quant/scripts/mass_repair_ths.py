# -*- coding: utf-8 -*-
"""Batch repair 2026-08-27+ corrupted OHLCV rows from THS referee."""
from __future__ import annotations
import json, sys, shutil, time, datetime
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from price_guards import validate_daily_bars, fetch_ths_daily
ROOT = Path(__file__).resolve().parent.parent
STOCK_DIR = ROOT / "data" / "stocks"
QDIR = STOCK_DIR / "quarantine"
START = "2026-08-27"
def repair_one(sym, ths_cache, ts):
    csv = STOCK_DIR / (sym + ".csv")
    metap = STOCK_DIR / (sym + ".meta.json")
    if not csv.exists():
        return "missing"
    shutil.copy2(csv, QDIR / (sym + ".pre-massrepair-" + ts + ".csv"))
    df = pd.read_csv(csv)
    _, bad, reasons = validate_daily_bars(df.tail(400), source="massrepair", symbol=sym)
    recent_bad = [r["date"] for r in reasons if r["date"] >= START]
    if not recent_bad:
        return "clean"
    if sym in ths_cache:
        ths = ths_cache[sym]
    else:
        try:
            ths = fetch_ths_daily(sym)
        except Exception:
            ths = None
        ths_cache[sym] = ths
        time.sleep(0.3)
    if ths is None or len(ths) == 0:
        df2 = df[~df["date"].isin(recent_bad)].reset_index(drop=True)
        df2.to_csv(csv, index=False, encoding="utf-8-sig")
        return "quarantined_no_ths:" + str(len(recent_bad))
    tmap = {r["date"]: r for _, r in ths.iterrows()}
    anchor = sorted([d for d in df["date"].astype(str) if d < min(recent_bad) and d in tmap])
    pfac, vfac = 1.0, 1.0
    if anchor:
        ad = anchor[-1]
        fc = float(df.loc[df["date"] == ad, "close"].iloc[0])
        tc = float(tmap[ad]["close"])
        fv = float(df.loc[df["date"] == ad, "volume"].iloc[0])
        tv = float(tmap[ad]["volume"])
        if tc:
            pfac = fc / tc
        if tv:
            vr = fv / tv
            vfac = 0.01 if 0.005 < vr < 0.02 else (100.0 if 50 < vr < 200 else 1.0)
    if not (0.2 < pfac < 5.0):
        pfac = 1.0
    fixed, missing = 0, []
    for ds in recent_bad:
        if ds not in tmap:
            missing.append(ds)
            continue
        t = tmap[ds]
        mask = df["date"] == ds
        df.loc[mask, "open"] = t["open"] * pfac
        df.loc[mask, "high"] = t["high"] * pfac
        df.loc[mask, "low"] = t["low"] * pfac
        df.loc[mask, "close"] = t["close"] * pfac
        df.loc[mask, "volume"] = t["volume"] * vfac
        df.loc[mask, "amount"] = t["amount"]
        df.loc[mask, "turnover"] = t["turnover"]
        fixed += 1
    lastd = str(df["date"].max())
    added = 0
    for _, t in ths.iterrows():
        if t["date"] > lastd and t["date"] >= START:
            df.loc[len(df)] = [t["date"], t["open"] * pfac, t["high"] * pfac, t["low"] * pfac, t["close"] * pfac, t["volume"] * vfac, t["amount"], t["turnover"]]
            added += 1
    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv(csv, index=False, encoding="utf-8-sig")
    try:
        meta = json.loads(metap.read_text(encoding="utf-8"))
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        meta["rows"] = int(len(df))
        meta["data_end_date"] = str(df["date"].max())[:10]
        meta["repaired_from"] = "ths-massrepair"
        meta["repaired_range"] = START + ".." + str(df["date"].max())[:10]
        meta["repaired_at"] = now
        metap.write_text(json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + chr(10), encoding="utf-8")
    except Exception:
        pass
    _, bad2, _ = validate_daily_bars(df.tail(400), source="verify", symbol=sym)
    res2 = [r for r in bad2["date"] if str(r) >= START] if len(bad2) else []
    if res2:
        return "residual:" + str(len(res2))
    return "fixed:" + str(fixed) + "+added:" + str(added)
def main():
    scan = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    syms = scan["symbols"]
    ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    QDIR.mkdir(parents=True, exist_ok=True)
    ths_cache, counts = {}, {}
    for i, sym in enumerate(syms):
        try:
            st = repair_one(sym, ths_cache, ts)
        except Exception as exc:
            st = "error:" + type(exc).__name__
        key = st.split(":")[0]
        counts[key] = counts.get(key, 0) + 1
        print("SYM-RESULT " + sym + " " + st, flush=True)
        if (i + 1) % 50 == 0:
            print("repair progress " + str(i + 1) + "/" + str(len(syms)) + " " + json.dumps(counts, ensure_ascii=True), flush=True)
    print("REPAIR-DONE " + json.dumps({"total": len(syms), "counts": counts}, ensure_ascii=True))
if __name__ == "__main__":
    main()
