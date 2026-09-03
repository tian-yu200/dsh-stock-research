# -*- coding: utf-8 -*-
import sys, time
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from price_guards import validate_daily_bars, fetch_ths_daily
ROOT = Path(__file__).resolve().parent.parent
START = "2026-05-01"
syms = ["001393"]
for sym in syms:
    csv = ROOT / "data" / "stocks" / (sym + ".csv")
    df = pd.read_csv(csv)
    _, bad, reasons = validate_daily_bars(df.tail(400), source="r8", symbol=sym)
    rb = [r["date"] for r in reasons if r["date"] >= START]
    print(sym + " bad=" + str(rb), flush=True)
    if not rb:
        continue
    ths = fetch_ths_daily(sym)
    tmap = {r["date"]: r for _, r in ths.iterrows()}
    have = set(df["date"].astype(str))
    anch = sorted([d for d in have if d < min(rb) and d in tmap])
    ad = anch[-1]
    pfac = float(df.loc[df["date"] == ad, "close"].iloc[0]) / float(tmap[ad]["close"])
    print(sym + " anchor=" + ad + " pfac=" + str(round(pfac, 4)), flush=True)
    for ds in rb:
        t = tmap[ds]
        mask = df["date"] == ds
        df.loc[mask, "open"] = t["open"] * pfac
        df.loc[mask, "high"] = t["high"] * pfac
        df.loc[mask, "low"] = t["low"] * pfac
        df.loc[mask, "close"] = t["close"] * pfac
        df.loc[mask, "volume"] = t["volume"]
        df.loc[mask, "amount"] = t["amount"]
        df.loc[mask, "turnover"] = t["turnover"]
    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv(csv, index=False, encoding="utf-8-sig")
    _, bad2, _ = validate_daily_bars(df.tail(400), source="r8v", symbol=sym)
    print(sym + " residual=" + str([str(d) for d in bad2["date"]] if len(bad2) else []), flush=True)
    time.sleep(1.0)
