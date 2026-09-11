# -*- coding: utf-8 -*-
import glob, os, json
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
bs = glob.glob(str(ROOT / "data" / "stocks" / "quarantine" / "*.pre-massrepair-20260903T203*.csv"))
no = []
for b in bs:
    sym = os.path.basename(b)[:6]
    try:
        end = str(pd.read_csv(ROOT / "data" / "stocks" / (sym + ".csv"), usecols=["date"])["date"].max())
    except Exception:
        end = "read-error"
    if end < "2026-08-27":
        no.append(sym)
print("backups=" + str(len(bs)) + " stale=" + str(len(no)))
print(json.dumps(no, ensure_ascii=True))
