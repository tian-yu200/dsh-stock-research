# -*- coding: utf-8 -*-
"""一次性探测：对比单标的接口与全市场批量接口的耗时，用于抓取管线优化决策。"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("NO_PROXY", "*")

import akshare as ak


def timed(label, fn):
    t0 = time.time()
    try:
        df = fn()
        rows = 0 if df is None else len(df)
        print(f"{label}: {time.time() - t0:.1f}s rows={rows}", flush=True)
    except Exception as exc:
        print(f"{label}: FAILED {time.time() - t0:.1f}s {type(exc).__name__}: {exc}", flush=True)


timed("fund_single_fullhist(000001)", lambda: ak.fund_open_fund_info_em(symbol="000001", indicator="单位净值走势"))
timed("stock_hist_incr(600909, 1month)", lambda: ak.stock_zh_a_hist(symbol="600909", period="daily", start_date="20260801", end_date="20260829", adjust="hfq"))
timed("etf_hist_incr(510300, 1month)", lambda: ak.fund_etf_hist_em(symbol="510300", period="daily", start_date="20260801", end_date="20260829", adjust="qfq"))
timed("fund_daily_bulk_all_funds", ak.fund_open_fund_daily_em)
timed("stock_spot_bulk_all", ak.stock_zh_a_spot_em)
timed("etf_spot_bulk_all", ak.fund_etf_spot_em)
print("probe done", flush=True)
