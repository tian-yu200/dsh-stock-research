# -*- coding: utf-8 -*-
import sys, json, time
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from price_guards import validate_daily_bars, fetch_ths_daily
ROOT = Path(__file__).resolve().parent.parent
START = "2026-08-27"
syms = ["300256", "301289", "603950", "600182", "600721", "688419", "688448", "920090", "301520", "601958", "603137"]
for sym in syms:
    csv = ROOT / "data" / "stocks" / (sym + ".csv")
    df = pd.read_csv(csv)
    _, bad, reasons = validate_daily_bars(df.tail(400), source="pass2", symbol=sym)
    rb = [r["date"] for r in reasons if r["date"] >= START]
    print(sym + " bad=" + str(rb), flush=True)
    if not rb:
        continue
    try:
        ths = fetch_ths_daily(sym)
    except Exception:
        ths = None
    tmap = {r["date"]: r for _, r in ths.iterrows()} if ths is not None and len(ths) else {}
    fixed, dropped = 0, []
    for ds in rb:
        if ds in tmap:
            t = tmap[ds]
            mask = df["date"] == ds
            for col in ["open", "high", "low", "close", "volume", "amount", "turnover"]:
                df.loc[mask, col] = t[col]
            fixed += 1
        else:
            dropped.append(ds)
    if dropped:
        df = df[~df["date"].isin(dropped)].reset_index(drop=True)
    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv(csv, index=False, encoding="utf-8-sig")
    _, bad2, _ = validate_daily_bars(df.tail(400), source="verify2", symbol=sym)
    r2 = [str(d) for d in bad2["date"]] if len(bad2) else []
    print(sym + " fixed=" + str(fixed) + " dropped=" + str(dropped) + " residual=" + str(r2), flush=True)
    time.sleep(0.3)
