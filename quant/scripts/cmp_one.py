# -*- coding: utf-8 -*-
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from price_guards import fetch_ths_daily
sym = sys.argv[1]
df = pd.read_csv("D:/dsh-股票/quant/data/stocks/" + sym + ".csv")
print(df[df["date"] >= "2026-08-20"][["date", "open", "close", "volume", "amount"]].to_string(index=False))
t = fetch_ths_daily(sym)
print(t[t["date"] >= "2026-08-20"][["date", "open", "close", "volume", "amount"]].to_string(index=False) if t is not None else "ths-none")
