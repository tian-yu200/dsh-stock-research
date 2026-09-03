# -*- coding: utf-8 -*-
import sys, time
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from price_guards import validate_daily_bars, fetch_ths_daily
ROOT = Path(__file__).resolve().parent.parent
START = "2026-08-27"
syms = ["603205", "688590", "601618", "301448", "605133", "688685", "920045", "920021"]
for sym in syms:
    csv = ROOT / "data" / "stocks" / (sym + ".csv")
    df = pd.read_csv(csv)
    _, bad, reasons = validate_daily_bars(df.tail(400), source="r5", symbol=sym)
    rb = [r["date"] for r in reasons if r["date"] >= START]
    old = [r["date"] for r in reasons if r["date"] < START]
    print(sym + " recent_bad=" + str(rb) + " old_bad=" + str(old[:5]), flush=True)
    if not rb:
        continue
    try:
        ths = fetch_ths_daily(sym)
    except Exception as exc:
        print(sym + " ths-" + type(exc).__name__, flush=True)
        continue
    if ths is None or not len(ths):
        print(sym + " ths-empty", flush=True)
        continue
    tmap = {r["date"]: r for _, r in ths.iterrows()}
    have = set(df["date"].astype(str))
    anch = sorted([d for d in have if d < min(rb) and d in tmap])
    pfac, vfac = 1.0, 1.0
    if anch:
        ad = anch[-1]
        try:
            tc = float(tmap[ad]["close"])
            tv = float(tmap[ad]["volume"])
            fc = float(df.loc[df["date"] == ad, "close"].iloc[0])
            fv = float(df.loc[df["date"] == ad, "volume"].iloc[0])
            pfac = fc / tc if tc else 1.0
            vr = fv / tv if tv else 1.0
            vfac = 0.01 if 0.005 < vr < 0.02 else (100.0 if 50 < vr < 200 else 1.0)
        except Exception as exc:
            print(sym + " anchor-" + type(exc).__name__, flush=True)
    print(sym + " pfac=" + str(round(pfac, 4)) + " vfac=" + str(vfac), flush=True)
    if not (0.2 < pfac < 5.0):
        print(sym + " absurd-factor-skip", flush=True)
        continue
    for ds in rb:
        if ds not in tmap:
            print(sym + " " + ds + " no-ths-date-drop", flush=True)
            df = df[df["date"] != ds]
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
    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv(csv, index=False, encoding="utf-8-sig")
    _, bad2, _ = validate_daily_bars(df.tail(400), source="r5v", symbol=sym)
    print(sym + " residual=" + str([str(d) for d in bad2["date"]] if len(bad2) else []), flush=True)
    time.sleep(1.0)
