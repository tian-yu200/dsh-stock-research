# -*- coding: utf-8 -*-
"""Price-scale guards against corrupted OHLC segments.
Catches cases like 603826 2026-08-28 before bad rows reach CSV caches.
"""
from __future__ import annotations
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
import pandas as pd
IMPLIED_DEVIATION_THRESHOLD = 0.25
OVERNIGHT_GAP_THRESHOLD = 0.30
THS_TIMEOUT_SECONDS = 20
THS_CONFLICT_THRESHOLD = 0.20
THS_LOOKBACK_DAYS = 5
# T1b fail-closed THS patch: conversion must be proven within this tolerance
# (rootcause-0928 implied-factor ratio was 0.83-0.84 -> refused at 0.15).
THS_FACTOR_TOLERANCE = 0.15
# Same-segment local evidence window (rows each side of the target row).
THS_SEGMENT_HALF_WINDOW = 10
THS_SEGMENT_MIN_ROWS = 5
# Hypothesis grid for THS /01 raw fields vs local UNIT_CONTRACT
# (volume=shares, turnover=decimal-fraction): volume lots->shares, amount
# yuan->yuan (or thousand-yuan->yuan), turnover parser output -> decimal.
THS_VOLUME_SCALES = (1.0, 100.0)
THS_AMOUNT_SCALES = (1.0, 0.001)
THS_TURNOVER_SCALES = (0.01, 1.0, 0.0001)
THS_CONFLICT_LOG = Path(__file__).resolve().parents[1] / "logs" / "ths-conflicts.log"
# Rotation cap: when the conflict log exceeds this size it is renamed to
# <name>.1 (replacing any old .1) before the next append, inside the same
# silent try -- the evidence file must never grow unbounded.
THS_CONFLICT_LOG_MAX_BYTES = 5 * 1024 * 1024
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
        work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.normalize()
    except Exception:
        pass
    row_count = len(work)

    def _column_values(name: str) -> list:
        # 逐行 .loc 访问是全市场扫描的吞吐瓶颈（31k 标的 × 数千行），
        # 这里一次性转成纯 Python 标量列表；NaN 保持 NaN（真值），与原
        # `float(row.get(col) or 0)` 的 `or` 语义逐分支等价。
        if name in work.columns:
            return pd.to_numeric(work[name], errors="coerce").tolist()
        return [None] * row_count

    opens = _column_values("open")
    closes = _column_values("close")
    volumes = _column_values("volume")
    amounts = _column_values("amount")

    if "date" in work.columns:
        date_col = work["date"]
        if pd.api.types.is_datetime64_any_dtype(date_col):
            ds_list = date_col.dt.strftime("%Y-%m-%d").fillna("NaT").tolist()
        else:
            ds_list = [
                pd.Timestamp(value).date().isoformat() if pd.notna(value) else str(value)
                for value in date_col
            ]
    else:
        ds_list = ["None"] * row_count

    kept_ratio = []
    keep = [True] * row_count
    reasons = []
    prev_close, prev_date, prev_vol, prev_amt = None, None, None, None
    thr = gap_threshold_for(symbol)
    for i in range(row_count):
        o = float(opens[i] or 0)
        c = float(closes[i] or 0)
        v = float(volumes[i] or 0)
        a = float(amounts[i] or 0)
        ds = ds_list[i]
        implied = (a / v) if v else None
        ratio = (implied / c) if (implied and c) else None
        if i == 0 or i < 10:
            if ratio and ratio == ratio:
                kept_ratio.append(ratio)
            prev_close, prev_date, prev_vol, prev_amt = c or None, ds, v or None, a or None
            continue
        gap = (o / prev_close - 1.0) if (o and prev_close) else 0.0
        med = median(kept_ratio[-60:]) if len(kept_ratio) >= 20 else None
        dev = abs(ratio / med - 1.0) if (ratio and med) else 0.0
        big_gap = abs(gap) > thr
        # 开盘缺失（新浪通道已知形态：open=0/NaN）时 gap 恒为 0 或 NaN，
        # 跳变检测被整体绕过——2026-09-02 基准断裂正是借这条通道进入。
        # 此时退化为 close-to-close 检测（同一阈值、同一停牌豁免），
        # 开盘正常的行判定路径完全不变。
        open_missing = (not o) or o != o
        if prev_close and c and open_missing:
            close_jump = abs(c / prev_close - 1.0)
            if close_jump != close_jump:
                close_jump = 0.0
        else:
            close_jump = 0.0
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
        elif close_jump > thr:
            try:
                day_gap = (pd.Timestamp(ds) - pd.Timestamp(prev_date)).days if prev_date else 1
            except Exception:
                day_gap = 1
            if day_gap > 10:
                why.append("halt_resume_unjudged")
            else:
                verdict, why = "quarantine", ["close_jump_scale_break"]
        if verdict == "quarantine":
            keep[i] = False
            reasons.append({"date": ds, "symbol": symbol, "source": source, "reasons": why, "gap": round(close_jump if close_jump else gap, 4), "deviation": round(dev, 4)})
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
def _append_ths_conflict(record):
    """Tiny durable trace: one JSON line to quant/logs/ths-conflicts.log.

    Fetch worker stdout/stderr is DEVNULL (run_research_worker.py), so the
    conflict evidence must be appended to a file, never printed. Any failure
    is swallowed silently: logging must never break the fetch path.
    """
    try:
        line = json.dumps(record, ensure_ascii=True, default=str)
        path = Path(THS_CONFLICT_LOG)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        try:
            if path.exists() and path.stat().st_size > THS_CONFLICT_LOG_MAX_BYTES:
                path.replace(path.with_name(path.name + ".1"))
        except Exception:
            pass
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


def _ths_segment_evidence(out, pos):
    """(implied_factor_ref, shares_ref) from same-segment local rows.

    implied factor = close*volume/amount on the LOCAL series: the anchor that
    a converted THS row must reproduce.
    shares_ref = volume/turnover is only the local series' own
    volume-to-turnover scale, used to pin the THS kv/kt ratio against local
    convention -- it is NOT the share count. UNIT_CONTRACT (market_data.py)
    declares volume=shares, turnover=decimal-fraction, but on real repo data
    volume/turnover = total shares / 100 (000001 2026-09-23:
    906726/0.0047 = 1.93e8 vs ~19.3e9 true shares), i.e. stored volume
    behaves as 手 despite the declared contract. So neither leg proves
    absolute units: combined they are a relative convention check that is
    fail-closed -- on real THS /01 rows (unadjusted basis) the implied-factor
    leg fails and _ths_conversion_proof returns None by design; local scale
    errors are remediated by full single-source refetch, never by patching.
    The target row itself (pos) is excluded so its own corruption cannot
    become its own reference.
    """
    try:
        lo = max(0, int(pos) - THS_SEGMENT_HALF_WINDOW)
        hi = min(len(out), int(pos) + THS_SEGMENT_HALF_WINDOW + 1)
        seg = out.iloc[lo:hi]
        c = _num(seg.get("close"))
        v = _num(seg.get("volume"))
        a = _num(seg.get("amount"))
        tv = _num(seg.get("turnover"))
        keep = pd.Series(True, index=seg.index)
        if pos - lo < len(keep):
            keep.iloc[pos - lo] = False
        ok = keep & (c > 0) & (v > 0) & (a > 0)
        factors = ((c * v) / a)[ok]
        factors = factors.dropna()
        if int(factors.count()) < THS_SEGMENT_MIN_ROWS:
            return None, None
        shares = (v / tv.where(tv > 0))[ok].dropna()
        if int(shares.count()) < THS_SEGMENT_MIN_ROWS:
            return float(factors.median()), None
        return float(factors.median()), float(shares.median())
    except Exception:
        return None, None


def _ths_conversion_proof(t, out, pos, *, tolerance=THS_FACTOR_TOLERANCE):
    """Prove a THS row convertible, or return None (fail-closed).

    Proofs (all required):
    1. implied factor of the converted THS row (close*volume/amount, THS
       volume/amount scaled to the local UNIT_CONTRACT) matches the local
       same-segment factor within tolerance -- this pins the adjustment
       basis AND the volume/amount units together;
    2. converted volume/turnover reproduces the local shares-ref scale
       within tolerance -- an independent pin on the THS volume/turnover
       ratio against local convention (a relative check only; ref_shares is
       not the share count, see _ths_segment_evidence);
    3. re-check: implied factor of the fully converted row vs the local
       reference (the requirement's literal check).
    """
    try:
        c = float(t["close"])
        v = float(t["volume"])
        a = float(t["amount"])
        tv = float(t["turnover"])
        o = float(t["open"])
        h = float(t["high"])
        lo_ = float(t["low"])
    except (KeyError, TypeError, ValueError):
        return None
    if not all(x == x and x > 0 for x in (c, v, a, tv, o, h, lo_)):
        return None
    ref_factor, ref_shares = _ths_segment_evidence(out, pos)
    if not ref_factor or not ref_shares:
        return None
    for kv in THS_VOLUME_SCALES:
        for ka in THS_AMOUNT_SCALES:
            iths = (c * (v * kv)) / (a * ka)
            if not (iths > 0) or abs(iths / ref_factor - 1.0) > tolerance:
                continue
            for kt in THS_TURNOVER_SCALES:
                shares_ths = (v * kv) / (tv * kt)
                if abs(shares_ths / ref_shares - 1.0) > tolerance:
                    continue
                basis = ref_factor / iths
                patched_implied = ((c * basis) * (v * kv)) / (a * ka)
                if abs(patched_implied / ref_factor - 1.0) > tolerance:
                    continue
                return {
                    "volume": kv,
                    "amount": ka,
                    "turnover": kt,
                    "basis": basis,
                    "ref_factor": ref_factor,
                    "ref_shares": ref_shares,
                }
    return None


def ths_patch_recent(symbol, frame, *, threshold=THS_CONFLICT_THRESHOLD):
    """Re-check recent rows against THS -- but never overwrite blindly.

    The THS /01 channel is unadjusted and its raw fields do not carry the
    local unit contract (volume in shares, turnover as a decimal). Writing
    those 7 columns over local hfq rows produced 0.84x close seams
    (rootcause-0928, ~183 symbols). Now each deviating row must first pass
    _ths_conversion_proof; proven rows are written exactly converted
    (OHLC x basis, volume/amount/turnover x unit scales), unproven rows are
    REFUSED: local row kept and the conflict appended as a JSON line to
    quant/logs/ths-conflicts.log (durable, outside DEVNULL stdout/stderr).
    """
    try:
        ths = fetch_ths_daily(symbol)
    except Exception:
        return frame, {"patched": 0, "refused": 0, "detail": "ths_unreachable"}
    if ths is None or len(ths) == 0:
        return frame, {"patched": 0, "refused": 0, "detail": "ths_empty"}
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    patched = 0
    refused = 0
    for _, t in ths.iterrows():
        mask = out["date"] == t["date"]
        if not mask.any():
            continue
        pos = int(mask.to_numpy().nonzero()[0][0])
        cur = float(out.iloc[pos]["close"]) if "close" in out.columns else 0.0
        try:
            ths_close = float(t["close"])
        except (TypeError, ValueError):
            continue
        if not (cur and cur > 0 and ths_close > 0):
            continue
        if abs(ths_close / cur - 1.0) <= threshold:
            continue
        proof = _ths_conversion_proof(t, out, pos)
        record = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "symbol": str(symbol),
            "date": str(t["date"]),
            "close_local": cur,
            "close_ths": ths_close,
            "ratio": round(ths_close / cur, 6),
        }
        if proof is None:
            record.update({"action": "refused", "reason": "basis_or_units_unproven"})
            _append_ths_conflict(record)
            refused += 1
            continue
        for col, scale in (("open", "basis"), ("high", "basis"), ("low", "basis"), ("close", "basis")):
            if col in out.columns:
                out.loc[mask, col] = float(t[col]) * proof[scale]
        if "volume" in out.columns:
            out.loc[mask, "volume"] = float(t["volume"]) * proof["volume"]
        if "amount" in out.columns:
            out.loc[mask, "amount"] = float(t["amount"]) * proof["amount"]
        if "turnover" in out.columns:
            out.loc[mask, "turnover"] = float(t["turnover"]) * proof["turnover"]
        record.update({
            "action": "patched",
            "scales": {k: proof[k] for k in ("volume", "amount", "turnover", "basis")},
        })
        _append_ths_conflict(record)
        patched += 1
    return out, {"patched": patched, "refused": refused, "detail": "ths_patch_recent"}


# --- 单位守卫（2026-08-28 volume/turnover 口径切换事故后增设） ---
# 与价格守卫同级、语义独立：只看 volume/turnover 的日环比（r = V/T），close 不动
# 的纯单位跳变也能独立发射。现有价格函数一律未动（不变量 :27 的 A/B 零差异要求）。
# 单位契约见 market_data.UNIT_CONTRACT：volume=股，turnover=小数。
UNIT_JUMP_HI = 50.0
UNIT_JUMP_LO = 0.02
UNIT_BIG_HI = 2000.0
UNIT_BIG_LO = 0.0005
# amount 协同带：v/t 跳变落在 [30,50)（或反向）时，只看 v/t 无法区分
# “真实放量”（amount 同步）与“单位错”（amount 不动），此时以 amount 裁决。
# 50x 以上直接发射的行为不变。
UNIT_BAND_HI = 30.0
UNIT_BAND_LO = 1.0 / 30.0
UNIT_AMT_SYNC = 10.0
# 停牌豁免（与价格守卫对称，但已收紧，见 R1）：自然日 gap >10 天的首行，
# 仅当该行量价自洽（隐含均价 amount/volume 落在近 60 日中位的 X 倍带内）才豁免。
# 纯 gap 判定无法区分真停牌与数据缺口（实证 000875：11 天 gap 实为缺行，
# 且该行 volume 低估约 100 倍）；无法评估（隐含 NaN / 历史不足）时沿用旧行为豁免。
UNIT_HALT_DAYS = 10
UNIT_SELFCHECK_X = 5.0


def _unit_band(jump):
    try:
        j = float(jump)
    except (TypeError, ValueError):
        return "unit_jump_other"
    if UNIT_JUMP_HI <= j <= 500.0:
        return "unit_up_x100"
    if UNIT_BIG_HI <= j <= 200000.0:
        return "unit_up_x10000"
    if 0.002 <= j <= UNIT_JUMP_LO:
        return "unit_down_x100"
    if 0.000005 <= j <= UNIT_BIG_LO:
        return "unit_down_x10000"
    if UNIT_BAND_HI <= j < UNIT_JUMP_HI:
        return "unit_up_30_50_amtflat"
    if UNIT_JUMP_LO < j <= UNIT_BAND_LO:
        return "unit_down_30_50_amtflat"
    return "unit_jump_other"


def detect_unit_breaks(frame, *, source="", symbol=""):
    """Flag rows whose volume/turnover day-over-day ratio jumps unit-scale.

    r = volume/turnover per row; jump = r_i / r_{i-1}; suspect when
    jump >= UNIT_JUMP_HI (50) or jump <= UNIT_JUMP_LO (0.02), covering x100 /
    x10000 in both directions. Close is never consulted, so pure unit switches
    fire independently of price breaks. A level-consistent history (e.g. 000783
    hand-denominated throughout) has jump ~= 1 and is never flagged.
    Narrow band [30,50) (and mirror): amount corroborates — fire only when
    amount is flat (|day-ratio| < 10x), i.e. money didn't move; a synced
    amount move means real trading and stays quiet. Missing amount never
    fires in this band. Rows after a >10-natural-day gap are exempt only when
    amount/volume self-consistent (implied price within X of trailing median,
    R1); unevaluable rows keep the old exempt behavior.
    Returns (flags Series aligned to frame.index, reasons list shaped like
    validate_daily_bars reasons for write_quarantine).
    """
    flags = pd.Series(False, index=frame.index)
    reasons = []
    try:
        vol = pd.to_numeric(frame.get("volume"), errors="coerce")
        tov = pd.to_numeric(frame.get("turnover"), errors="coerce")
        amt = pd.to_numeric(frame.get("amount"), errors="coerce")
        valid = vol.gt(0) & tov.gt(0)
        ratio = vol / tov.where(valid, pd.NA)
        prev = ratio.shift(1)
        prev_valid = valid.shift(1, fill_value=False)
        # valid/prev_valid guarantee finite positive ratio and prev: no inf/0 risk.
        jump = ratio / prev
        hit_big = valid & prev_valid & jump.notna() & ((jump >= UNIT_JUMP_HI) | (jump <= UNIT_JUMP_LO))
        amt_valid = amt.gt(0)
        amt_prev_valid = amt_valid.shift(1, fill_value=False)
        aratio = amt / amt.shift(1)
        in_band = ((jump >= UNIT_BAND_HI) & (jump < UNIT_JUMP_HI)) | ((jump > UNIT_JUMP_LO) & (jump <= UNIT_BAND_LO))
        amt_flat = amt_valid & amt_prev_valid & aratio.notna() & (aratio < UNIT_AMT_SYNC) & (aratio > 1.0 / UNIT_AMT_SYNC)
        hit_band = valid & prev_valid & jump.notna() & in_band & amt_flat
        hit = hit_big | hit_band
        try:
            gap_days = (pd.to_datetime(frame.get("date"), errors="coerce") -
                        pd.to_datetime(frame.get("date"), errors="coerce").shift(1)).dt.days
            gap_mask = gap_days.fillna(0) > UNIT_HALT_DAYS
            # R1: gap 首行豁免须量价自洽。implied = amount/volume（与 close 无关，
            # hfq 安全）；偏离近 60 日中位超过 X 倍则拒绝豁免（如 000875 偏离 97x）。
            # 只在有肯定证据（可计算且偏离）时拒绝；无法评估则沿用旧行为豁免，
            # 避免误伤真停牌。已知上限（R2）：amount 若被同比例污染，
            # implied 保持自洽，豁免仍会生效。
            implied = amt / vol.where(vol.gt(0), pd.NA)
            implied = implied.where(amt.gt(0), pd.NA)
            trail_med = implied.rolling(60, min_periods=5).median().shift(1)
            dev = pd.to_numeric(implied / trail_med, errors="coerce")
            inconsistent = dev.notna() & ((dev < 1.0 / UNIT_SELFCHECK_X) | (dev > UNIT_SELFCHECK_X))
            hit = hit & ~(gap_mask & ~inconsistent)
        except Exception:
            pass
        flags = hit.fillna(False).astype(bool)
        dates = frame.get("date")
        for idx in flags.index[flags]:
            try:
                ds = pd.Timestamp(dates.loc[idx]).date().isoformat()
            except Exception:
                ds = str(dates.loc[idx]) if dates is not None else str(idx)
            reasons.append({"date": ds, "symbol": symbol, "source": source,
                            "reasons": [_unit_band(jump.loc[idx])],
                            "gap": 0.0, "deviation": 0.0})
    except Exception:
        return pd.Series(False, index=frame.index), []
    return flags, reasons
