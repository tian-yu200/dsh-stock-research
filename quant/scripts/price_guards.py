# -*- coding: utf-8 -*-
"""Price-scale guards against corrupted OHLC segments.
Catches cases like 603826 2026-08-28 before bad rows reach CSV caches.
"""
from __future__ import annotations
import urllib.request
from pathlib import Path
import pandas as pd
IMPLIED_DEVIATION_THRESHOLD = 0.25
OVERNIGHT_GAP_THRESHOLD = 0.30
THS_TIMEOUT_SECONDS = 20
THS_CONFLICT_THRESHOLD = 0.20
THS_LOOKBACK_DAYS = 5
def _num(s):
    return pd.to_numeric(s, errors="coerce")
def implied_ratio(frame):
    close = _num(frame.get("close"))
    vol = _num(frame.get("volume"))
    amt = _num(frame.get("amount"))
    implied = amt / vol.replace(0, pd.NA)
    return implied / close.replace(0, pd.NA)
def gap_threshold_for(symbol=""):
    s = str(symbol or "")
    if s[:1] in ("4", "8") or s[:2] == "92":
        return 0.33
    if s[:1] == "3" or s.startswith("688"):
        return 0.23
    if s:
        return 0.13
    return OVERNIGHT_GAP_THRESHOLD
def implied_deviation_flags(frame, *, threshold=IMPLIED_DEVIATION_THRESHOLD, window=60, min_rows=20):
    try:
        ratio = pd.to_numeric(implied_ratio(frame), errors="coerce")
        if ratio.dropna().empty:
            return pd.Series(False, index=frame.index)
        med = ratio.rolling(window, min_periods=min_rows).median().shift(1)
        dev = (ratio - med).abs() / med
        return (dev.fillna(0) > threshold).fillna(False)
    except Exception:
        return pd.Series(False, index=frame.index)
def overnight_gap_flags(frame, *, threshold=None, symbol=""):
    if threshold is None:
        threshold = gap_threshold_for(symbol)
    try:
        op = _num(frame.get("open"))
        prev = _num(frame.get("close")).shift(1)
        gap = (op - prev).abs() / prev
        flags = (gap.fillna(0) > threshold).fillna(False)
        flags.iloc[0] = False
        return flags
    except Exception:
        return pd.Series(False, index=frame.index)
    return flags
def validate_daily_bars(frame, *, source="", symbol=""):
    work = frame.copy().reset_index(drop=True)
    try:
        work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    except Exception:
        pass
    kept_ratio = []
    keep = [True] * len(work)
    reasons = []
    prev_close, prev_date, prev_vol, prev_amt = None, None, None, None
    for i in range(len(work)):
        row = work.loc[i]
        o = float(row.get("open") or 0)
        c = float(row.get("close") or 0)
        v = float(row.get("volume") or 0)
        a = float(row.get("amount") or 0)
        ds = str(row.get("date"))
        implied = (a / v) if v else None
        ratio = (implied / c) if (implied and c) else None
        if i == 0 or i < 10:
            if ratio and ratio == ratio:
                kept_ratio.append(ratio)
            prev_close, prev_date, prev_vol, prev_amt = c or None, ds, v or None, a or None
            continue
        gap = (o / prev_close - 1.0) if (o and prev_close) else 0.0
        med = float(pd.Series(kept_ratio[-60:]).median()) if len(kept_ratio) >= 20 else None
        dev = abs(ratio / med - 1.0) if (ratio and med) else 0.0
        thr = gap_threshold_for(symbol)
        big_gap = abs(gap) > thr
        verdict, why = "keep", []
        if big_gap:
            try:
                day_gap = (pd.Timestamp(ds) - pd.Timestamp(prev_date)).days if prev_date else 1
            except Exception:
                day_gap = 1
            if day_gap > 10:
                why.append("halt_resume_unjudged")
            elif gap > 0:
                verdict, why = "quarantine", ["gap_up_scale_break"]
            else:
                pf = (o / prev_close) if prev_close else 0.0
                exp_vr = (prev_close / o) if o else 0.0
                vr = (v / prev_vol) if (v and prev_vol) else 0.0
                ar = (a / prev_amt) if (a and prev_amt) else 0.0
                exdiv = (0.25 <= pf <= 1.0) and (0.5 * exp_vr <= vr <= 2.0 * exp_vr) and (0.6 <= ar <= 1.67)
                if exdiv:
                    why.append("exdiv_confirmed")
                else:
                    verdict, why = "quarantine", ["gap_down_unconfirmed"]
        if verdict == "quarantine":
            keep[i] = False
            reasons.append({"date": ds, "symbol": symbol, "source": source, "reasons": why, "gap": round(gap, 4), "deviation": round(dev, 4)})
            continue
        if ratio and ratio == ratio:
            kept_ratio.append(ratio)
        prev_close, prev_date, prev_vol, prev_amt = c or prev_close, ds, v or prev_vol, a or prev_amt
    keep_s = pd.Series(keep, index=work.index)
    clean = work.loc[keep_s].reset_index(drop=True)
    quar = work.loc[~keep_s].reset_index(drop=True)
    return clean, quar, reasons
def quarantine_path(data_dir, symbol):
    d = Path(data_dir) / "quarantine"
    d.mkdir(parents=True, exist_ok=True)
    return d / (symbol + ".quarantine.csv")
def write_quarantine(data_dir, symbol, bad, reasons):
    if bad is None or len(bad) == 0:
        return None
    out = bad.copy()
    out["quarantine_reasons"] = [",".join(r["reasons"]) for r in reasons]
    path = quarantine_path(data_dir, symbol)
    if path.exists():
        try:
            old = pd.read_csv(path)
            out = pd.concat([old, out], ignore_index=True).drop_duplicates(subset=["date"], keep="last")
        except Exception:
            pass
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return path
def fetch_ths_daily(symbol, *, timeout=THS_TIMEOUT_SECONDS):
    mk = "bj_" if str(symbol)[:1] in ("4", "8") else "hs_"
    url = "http://d.10jqka.com.cn/v2/line/" + mk + str(symbol) + "/01/last.js"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Referer": "http://q.10jqka.com.cn/"})
    raw = urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8", errors="replace")
    mark = chr(34) + "data" + chr(34) + ":" + chr(34)
    if mark not in raw:
        return None
    s = raw.split(mark, 1)[1]
    s = s.rsplit(chr(34), 2)[0]
    rows = []
    for line in s.split(";"):
        c = line.split(",")
        if len(c) < 7 or not c[0].isdigit():
            continue
        d = c[0]
        try:
            if not c[1] or not c[4]:
                continue
            rows.append({"date": d[0:4] + "-" + d[4:6] + "-" + d[6:8], "open": float(c[1]), "high": float(c[2]), "low": float(c[3]), "close": float(c[4]), "volume": float(c[5]), "amount": float(c[6]), "turnover": (float(c[7]) / 100.0) if len(c) > 7 and c[7] else 0.0})
        except Exception:
            continue
    if not rows:
        return None
    return pd.DataFrame(rows)
def ths_crosscheck(symbol, frame, *, threshold=THS_CONFLICT_THRESHOLD, lookback=THS_LOOKBACK_DAYS):
    result = {"checked": False, "conflict": False, "common": 0, "median_ratio": None, "detail": ""}
    try:
        ths = fetch_ths_daily(symbol)
    except Exception as exc:
        result["detail"] = "ths_unreachable:" + type(exc).__name__
        return result
    if ths is None or len(ths) == 0:
        result["detail"] = "ths_empty"
        return result
    a = frame.copy()
    a["date"] = pd.to_datetime(a["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    common = sorted(set(a["date"].dropna()) & set(ths["date"]))
    if not common:
        result["detail"] = "no_common_dates"
        return result
    ratios = []
    for ds in common[-lookback:]:
        c1 = float(a.loc[a["date"] == ds, "close"].iloc[0])
        c2 = float(ths.loc[ths["date"] == ds, "close"].iloc[0])
        if c1:
            ratios.append(c2 / c1)
    if not ratios:
        result["detail"] = "no_comparable_close"
        return result
    med = float(pd.Series(ratios).median())
    result.update({"checked": True, "common": len(common), "median_ratio": med})
    if abs(med - 1.0) > threshold:
        result["conflict"] = True
        result["detail"] = "ths_close_ratio=" + str(round(med, 4))
    return result
def ths_patch_recent(symbol, frame, *, threshold=THS_CONFLICT_THRESHOLD):
    try:
        ths = fetch_ths_daily(symbol)
    except Exception:
        return frame, {"patched": 0, "detail": "ths_unreachable"}
    if ths is None or len(ths) == 0:
        return frame, {"patched": 0, "detail": "ths_empty"}
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    patched = 0
    for _, t in ths.iterrows():
        mask = out["date"] == t["date"]
        if not mask.any():
            continue
        cur = float(out.loc[mask, "close"].iloc[0])
        if cur and abs(float(t["close"]) / cur - 1.0) > threshold:
            for col in ["open", "high", "low", "close", "volume", "amount", "turnover"]:
                out.loc[mask, col] = t[col]
            patched += 1
    return out, {"patched": patched, "detail": "ths_patch_recent"}
