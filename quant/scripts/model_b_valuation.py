# -*- coding: utf-8 -*-
"""模型 B：市盈率（估值）+ 产业逻辑卡。离线独立模块，不进主链路。

隔离约定（抄 stability_check.py 模板）：
- 只读独立配置（命令行参数，不读 research.* 配置节）；
- 重 import（akshare/pandas）延迟到真实运行时，--dry-run 保持瞬时；
- 不碰 research.py / data_layer.py / qlib_model.py / qlib_export.py，
  不进 qlib 数据集构建层（FEATURE_COLUMNS 一列不加）；
- 输出只写 logs/codex-runs/modelB-cache/（JSON）与本报告（md）。

PE 口径合规（硬约束）：全部 PE/PB/PEG 取 stock_value_em 厂商计算字段
（市值口径、分子分母同基准）；模块内任何地方不得出现 hfq价/EPS 自算。
历史分位数口径：主窗口近 2 年（2024-09-19→2026-09-18，自然日），
分位 = 窗口内 PE(TTM) <= 现值的样本占比；另附全历史分位备查。

用法：
  python model_b_valuation.py --dry-run   # 只打印计划，不联网
  python model_b_valuation.py             # 真实抓取并落盘
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
import sys
import urllib.parse
from pathlib import Path

QUANT_ROOT = Path(__file__).resolve().parents[1]
if str(QUANT_ROOT) not in sys.path:
    sys.path.insert(0, str(QUANT_ROOT))

SYMBOLS = ("000783", "002952", "600522")
CACHE_DIR = QUANT_ROOT / "logs" / "codex-runs" / "modelB-cache"
WINDOW_START = "2024-09-19"
DATA_END = "2026-09-18"
# 东财个股行业/地域字段直读（akshare 板块表在 Innodb 代理下不可达，见报告）。
EASTMONEY_STOCK_GET = "https://push2.eastmoney.com/api/qt/stock/get"


def _prefix(sym: str) -> str:
    return ("sh" if sym.startswith(("6", "9")) or sym.startswith("688") else "sz") + sym


def _secid(sym: str) -> str:
    return ("1" if sym.startswith("6") else "0") + "." + sym


def fetch_industry(sym: str) -> dict:
    """东财 f127（行业）/f128（地域）单次直取；失败即标缺，不重试。

    说明：python 直连（urllib/requests）在本机出口抖动/代理阻断下不稳定，
    改走 curl.exe（2026-09-20 实测可达）；仍失败则记录原文并标缺。
    """
    import json as _json
    import shutil

    url = EASTMONEY_STOCK_GET + "?secid=" + _secid(sym) + "&fields=f127,f128"
    at = _dt.datetime.now().isoformat()
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if not curl:
        return {"status": "missing", "error": "no-curl-binary", "url": url, "at": at}
    try:
        out = subprocess.run([curl, "-s", "-m", "20", url],
                             capture_output=True, text=True, timeout=30)
        data = _json.loads((out.stdout or "").strip() or "{}").get("data", {})
        if not data or not data.get("f127"):
            return {"status": "missing",
                    "error": ("empty rc=%s stderr=%s" % (out.returncode, out.stderr or ""))[:200],
                    "url": url, "at": at}
        return {"status": "ok", "industry": data.get("f127"), "region": data.get("f128"),
                "url": url, "at": at}
    except Exception as e:  # noqa: BLE001 - record verbatim, no retry
        return {"status": "missing", "error": type(e).__name__ + ": " + str(e)[:200],
                "url": url, "at": at}


def build_cards() -> dict:
    import akshare as ak
    import pandas as pd

    cards = {}
    for sym in SYMBOLS:
        card: dict = {"symbol": sym, "errors": []}
        try:
            vdf = ak.stock_value_em(symbol=sym)
        except Exception as e:  # noqa: BLE001
            card["errors"].append({"iface": "stock_value_em",
                                   "error": type(e).__name__ + ": " + str(e)[:200]})
            cards[sym] = card
            continue
        vdf.columns = [str(c) for c in vdf.columns]
        # akshare eastmoney 估值表列序固定：日期/收盘/涨跌幅/市值/流通市值/
        # 总股本/流通股本/PE(TTM)/PE(静)/市净率/PB?/PEG/…——按位置+名双保险取值
        cols = list(vdf.columns)

        def _col(names: list, fallback: int) -> str:
            for n in names:
                if n in cols:
                    return n
            return cols[fallback]

        c_date = _col(["日期"], 0)
        c_pe = _col(["PE(TTM)"], 7)
        c_pe_s = _col(["PE(静)"], 8)
        c_pb = _col(["市净率"], 9)
        c_peg = _col(["PEG值"], 10)
        c_mkt = _col(["市值"], 3)
        c_shr = _col(["总股本"], 5)
        last = vdf.iloc[-1]
        card["asof"] = str(last[c_date])[:10]
        card["pe_ttm"] = float(last[c_pe])
        card["pe_static"] = float(last[c_pe_s])
        card["pb"] = float(last[c_pb])
        card["peg"] = float(last[c_peg])
        card["mktcap_yuan"] = float(last[c_mkt])
        card["shares_total"] = float(last[c_shr])
        card["src_fields"] = {"iface": "ak.stock_value_em",
                              "cols": {"date": c_date, "pe_ttm": c_pe,
                                       "pe_static": c_pe_s, "pb": c_pb,
                                       "peg": c_peg, "mktcap": c_mkt,
                                       "shares": c_shr}}
        hist = pd.to_numeric(vdf[c_pe], errors="coerce").dropna()
        dates = pd.to_datetime(vdf[c_date], errors="coerce")
        win = hist[(dates >= WINDOW_START) & (dates <= DATA_END)]
        card["pe_quantile_2y"] = float((win <= card["pe_ttm"]).mean()) if len(win) else None
        card["pe_quantile_full"] = float((hist <= card["pe_ttm"]).mean())
        card["pe_window"] = {"start": WINDOW_START, "end": DATA_END, "n": int(len(win)),
                             "hist_n": int(len(hist))}
        try:
            fdf = ak.stock_financial_analysis_indicator(symbol=sym)
            fdf.columns = [str(c) for c in fdf.columns]
            fcols = list(fdf.columns)
            frow = fdf.iloc[-1]
            c_roe = next((c for c in fcols if "净资产收益率" in c), fcols[11])
            c_eps = next((c for c in fcols if "摊薄每股收益" in c), None)
            card["fin_latest"] = {
                "date": str(frow.iloc[0]),
                "roe_pct": float(frow[c_roe]) if str(frow[c_roe]) not in ("", "--", "nan") else None,
                "eps_diluted": float(frow[c_eps]) if c_eps and str(frow[c_eps]) not in ("", "--", "nan") else None}
            card["fin_src"] = {"iface": "ak.stock_financial_analysis_indicator",
                               "cols": {"roe": c_roe, "eps": c_eps}}
        except Exception as e:  # noqa: BLE001
            card["errors"].append({"iface": "stock_financial_analysis_indicator",
                                   "error": type(e).__name__ + ": " + str(e)[:200]})
        card["industry"] = fetch_industry(sym)
        cards[sym] = card
    return {"at": _dt.datetime.now().isoformat(), "data_end": DATA_END, "cards": cards}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.dry_run:
        print("modelB plan: symbols=%s ifaces=[stock_value_em,"
              "stock_financial_analysis_indicator,eastmoney-f127] out=%s" % (SYMBOLS, CACHE_DIR))
        return 0
    # Heavy imports only on real runs so --dry-run stays instant.
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = build_cards()
    (CACHE_DIR / "modelB_cards.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote %s" % (CACHE_DIR / "modelB_cards.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
