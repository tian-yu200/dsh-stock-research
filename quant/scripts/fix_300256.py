# -*- coding: utf-8 -*-
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from price_guards import fetch_ths_daily, validate_daily_bars
csv = Path(__file__).resolve().parents[1] / "data" / "stocks" / "300256.csv"
df = pd.read_csv(csv)
ths = fetch_ths_daily("300256")
anchor = "2026-08-27"
f_close = float(df.loc[df["date"] == anchor, "close"].iloc[0])
t_close = float(ths.loc[ths["date"] == anchor, "close"].iloc[0])
fac = f_close / t_close
print("factor=" + str(round(fac, 4)))
tmap = {r["date"]: r for _, r in ths.iterrows()}
for ds in ["2026-08-28", "2026-08-31", "2026-09-01", "2026-09-02", "2026-09-03"]:
    if ds not in tmap:
        continue
    t = tmap[ds]
    mask = df["date"] == ds
    if not mask.any():
        df.loc[len(df)] = [ds, t["open"] * fac, t["high"] * fac, t["low"] * fac, t["close"] * fac, t["volume"] / 100.0, t["amount"], t["turnover"]]
    else:
        df.loc[mask, "open"] = t["open"] * fac
        df.loc[mask, "high"] = t["high"] * fac
        df.loc[mask, "low"] = t["low"] * fac
        df.loc[mask, "close"] = t["close"] * fac
        df.loc[mask, "volume"] = t["volume"] / 100.0
        df.loc[mask, "amount"] = t["amount"]
        df.loc[mask, "turnover"] = t["turnover"]
df = df.sort_values("date").reset_index(drop=True)
df.to_csv(csv, index=False, encoding="utf-8-sig")
_, bad, _ = validate_daily_bars(df.tail(400), source="v300256", symbol="300256")
print("residual=" + str([str(d) for d in bad["date"]] if len(bad) else []))
print(df[df["date"] >= "2026-08-26"][["date", "close"]].to_string(index=False))
