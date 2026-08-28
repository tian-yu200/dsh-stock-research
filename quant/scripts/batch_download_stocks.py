#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股全市场日线批量下载（无超时限制）
写入目录：D:\\dsh-股票\\quant\\data\\stocks\\
用法：python batch_download_stocks.py
依赖：pip install akshare pandas
"""

import os, sys, time, gc
from pathlib import Path
from datetime import datetime

try:
    import akshare as ak
except ImportError:
    os.system("pip install akshare")
    import akshare as ak

# ── 配置 ──────────────────────────────────────────
OUT = Path(r"D:\dsh-股票\quant\data\stocks")
OUT.mkdir(parents=True, exist_ok=True)
START_DATE   = "20100101"   # 历史起始日
FRESH_HOURS  = 12           # 文件 < 此时间则跳过
SLEEP_SEC    = 0.12         # 请求间隔
LOG_EVERY    = 50           # 进度打印间隔

# ── 获取全市场代码 ────────────────────────────────
def get_codes():
    codes = set()
    # 沪深 A 股
    try:
        df = ak.stock_info_a_code_name()
        codes.update(df['code'].astype(str).str.zfill(6).tolist())
    except Exception as e:
        print(f"[warn] stock_info_a_code_name: {e}")
    # 北交所
    try:
        df_bj = ak.stock_info_bj_name_code()
        codes.update(df_bj['代码'].astype(str).str.zfill(6).tolist())
    except Exception as e:
        print(f"[warn] 北交所: {e}")
    return sorted(codes)

# ── 下载单只 ──────────────────────────────────────
def is_fresh(p):
    if not p.exists():
        return False
    return (time.time() - p.stat().st_mtime) / 3600 < FRESH_HOURS

def download_one(code):
    out = OUT / f"{code}.csv"
    if is_fresh(out):
        return "skip"
    end = datetime.now().strftime("%Y%m%d")
    # 主源：东财前复权
    try:
        df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                start_date=START_DATE, end_date=end, adjust="qfq")
        if df is not None and len(df) > 0:
            df.to_csv(out, index=False, encoding="utf-8-sig")
            return "ok"
    except Exception:
        pass
    # 备源：东财不复权
    try:
        df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                start_date=START_DATE, end_date=end)
        if df is not None and len(df) > 0:
            df.to_csv(out, index=False, encoding="utf-8-sig")
            return "ok"
    except Exception:
        pass
    # 备源：新浪
    try:
        if code[0] in "67":
            prefix = "sh"
        elif code.startswith("920") or code.startswith("8"):
            prefix = "bj"
        else:
            prefix = "sz"
        df = ak.stock_zh_a_daily(symbol=f"{prefix}{code}", adjust="qfq")
        if df is not None and len(df) > 0:
            df.to_csv(out, index=False, encoding="utf-8-sig")
            return "ok"
    except Exception:
        pass
    return "fail"

# ── 主流程 ────────────────────────────────────────
def main():
    codes = get_codes()
    n = len(codes)
    print(f"全 A 代码数：{n}")
    print(f"输出目录：{OUT}")
    print("=" * 60)
    ok = sk = fl = 0
    t0 = time.time()
    for i, c in enumerate(codes):
        r = download_one(c)
        if r == "ok":   ok += 1
        elif r == "skip": sk += 1
        else:           fl += 1; print(f"  ✗ {c}")
        if (i + 1) % LOG_EVERY == 0 or i + 1 == n:
            el = time.time() - t0
            rt = (i + 1) / el if el else 0
            eta = (n - i - 1) / rt if rt else 0
            print(f"{i+1}/{n} ({(i+1)/n*100:.1f}%)  ✓{ok} ⊟{sk} ✗{fl}  "
                  f"{rt:.1f}/s  ETA {eta/60:.0f}min")
        time.sleep(SLEEP_SEC)
        if (i + 1) % 500 == 0:
            gc.collect()
    el = time.time() - t0
    print("=" * 60)
    print(f"完成：{n} | ✓{ok} | ⊟{sk} | ✗{fl} | {el/60:.1f}min")
    if fl:
        print(f"失败 {fl} 只可重跑本脚本自动重试。")

if __name__ == "__main__":
    main()