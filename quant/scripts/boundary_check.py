# -*- coding: utf-8 -*-
import glob, os, json, datetime
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
bs = glob.glob(str(ROOT / "data" / "stocks" / "quarantine" / "*.pre-massrepair-20260903T203*.csv"))
syms = sorted(set(os.path.basename(b)[:6] for b in bs))
worst, over15, total = [], 0, 0
for sym in syms:
    try:
        df = pd.read_csv(ROOT / "data" / "stocks" / (sym + ".csv")).tail(12).reset_index(drop=True)
    except Exception:
        continue
    total += 1
    mg, mdate = 0.0, ""
    for i in range(1, len(df)):
        try:
            o = float(df.loc[i, "open"])
            p = float(df.loc[i - 1, "close"])
            g = abs(o / p - 1.0) if p else 0.0
        except Exception:
            g = 0.0
        if g > mg:
            mg, mdate = g, str(df.loc[i, "date"])
    end = str(df["date"].iloc[-1])
    if mg > 0.15:
        over15 += 1
        worst.append({"symbol": sym, "max_gap": round(mg, 4), "date": mdate, "end": end})
worst.sort(key=lambda x: -x["max_gap"])
ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
outp = str(ROOT / "logs" / "codex-runs" / ("boundary-" + ts + ".json"))
open(outp, "w", encoding="utf-8").write(json.dumps({"total": total, "over15": over15, "worst": worst[:40]}, ensure_ascii=True, indent=2))
print("BOUNDARY total=" + str(total) + " over15=" + str(over15))
print("REPORT " + outp)
