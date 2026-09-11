# -*- coding: utf-8 -*-
import glob, os, json
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
CORE = ["2026-08-28", "2026-08-31", "2026-09-01", "2026-09-02"]
out = []
files = sorted(glob.glob(str(ROOT / "data" / "stocks" / "[0-9]*.csv")))
for p in files:
    try:
        ds = set(pd.read_csv(p, usecols=["date"])["date"].astype(str))
    except Exception:
        continue
    miss = [d for d in CORE if d not in ds]
    if miss:
        out.append({"symbol": os.path.basename(p)[:6], "missing": miss, "end": sorted(ds)[-1] if ds else ""})
print("HOLES total=" + str(len(out)))
open(ROOT / "logs" / "codex-runs" / "holes.json", "w").write(json.dumps(out, ensure_ascii=True))
print(json.dumps([o["symbol"] for o in out[:40]], ensure_ascii=True))
