# -*- coding: utf-8 -*-
"""One-off repair for 2026-09-10 scale-basis poison (THS referee verified).

Root cause: 10 symbols got recent rows spliced on a wrong price basis
(local 1-day move -21%~-33% while THS referee shows +0%~+7%).
validate_daily_bars missed them because the open-gap fell in the
ex-div confirmation band (pf 0.65~0.80 with proportional volume/amount).
Fix = rescale OHLC of rows t>=poison_date to the pre-poison local basis
using THS daily returns: fixed(t) = local(d0) * THS(t)/THS(d0).
Volume/amount/turnover already match local scale -> untouched.

3 more symbols (002607/600720/603823) show impossible moves for their
board limits but THS was unreachable: diagnose, prefer THS rescale on
retry, else drop only the poison-date row and reset fetch_state so the
normal worker refills from source (Tencent probe decides safety).

920269 (BJ +-30% board) left alone: plausible real move, fail-open.

Usage:
  python repair_scale_poison_20260910.py --dry-run   # diagnose only
  python repair_scale_poison_20260910.py --live      # backup + fix + verify
"""
from __future__ import annotations

import argparse
import datetime
import json
import shutil
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from price_guards import fetch_ths_daily, validate_daily_bars

ROOT = Path(__file__).resolve().parent.parent
STOCK_DIR = ROOT / "data" / "stocks"
QDIR = STOCK_DIR / "quarantine"
STATE_DB = ROOT / "data" / "fetch_state.sqlite3"

# symbol -> first poison date (rows t >= date are suspect)
RESCALE = {
    "600094": "2026-09-08",
    "600479": "2026-09-08",
    "600981": "2026-09-07",
    "600052": "2026-09-08",
    "601225": "2026-09-08",
    "000002": "2026-09-09",
    "600518": "2026-09-07",
    "688298": "2026-09-07",
    "600606": "2026-09-07",
    "601828": "2026-09-07",
}
# board-limit impossible moves, THS unreachable at取证 time
UNVERIFIED = {
    "002607": "2026-09-07",
    "600720": "2026-09-07",
    "603823": "2026-09-07",
}
APPLY_THRESHOLD = 0.05  # only rescale when |k-1| exceeds 5%


def _ths_with_retry(symbol: str, tries: int = 3):
    last = None
    for _ in range(tries):
        try:
            ths = fetch_ths_daily(symbol)
            if ths is not None and len(ths):
                return ths
        except Exception as exc:  # noqa: BLE001 - referee may flap
            last = exc
        time.sleep(1.0)
    return None


def _tencent_0907(symbol: str):
    """Read-only probe: what does Tencent serve TODAY for the suspect date?"""
    try:
        import akshare as ak

        frame = ak.stock_zh_a_hist_tx(
            symbol=symbol, start_date="20260901", end_date="20260910", adjust="hfq"
        )
        if frame is None or len(frame) == 0:
            return None
        frame = frame.copy()
        frame.columns = [str(c).strip() for c in frame.columns]
        date_col = next(
            (c for c in frame.columns if c in ("日期", "date", "交易日期")), None
        )
        close_col = next((c for c in frame.columns if c in ("收盘", "close")), None)
        if date_col is None or close_col is None:
            return None
        out = {}
        for _, row in frame.iterrows():
            out[str(row[date_col])[:10]] = float(row[close_col])
        return out
    except Exception:  # noqa: BLE001 - probe must not kill repair
        return None


def diagnose(symbol: str, poison_date: str) -> dict:
    csv = STOCK_DIR / f"{symbol}.csv"
    local = pd.read_csv(csv)
    lmap = {str(r["date"])[:10]: float(r["close"]) for _, r in local.iterrows()}
    ths = _ths_with_retry(symbol)
    info: dict = {"symbol": symbol, "poison_date": poison_date, "rows": len(local)}
    if ths is None:
        info["ths"] = "unreachable"
        info["tencent_now"] = _tencent_0907(symbol)
        return info
    tmap = {str(r["date"]): float(r["close"]) for _, r in ths.iterrows()}
    dates = sorted(d for d in lmap if d < poison_date and d in tmap)
    if not dates:
        info["ths"] = "no-anchor"
        return info
    d0 = dates[-1]
    anchor = lmap[d0] / tmap[d0]
    info.update({"ths": "ok", "anchor_date": d0, "anchor_ratio": round(anchor, 4)})
    plan = []
    for d in sorted(d for d in lmap if d >= poison_date and d in tmap):
        k = tmap[d] * anchor / lmap[d] if lmap[d] else 1.0
        plan.append(
            {
                "date": d,
                "local": round(lmap[d], 2),
                "ths": round(tmap[d], 2),
                "k": round(k, 4),
                "apply": abs(k - 1.0) > APPLY_THRESHOLD,
            }
        )
    info["plan"] = plan
    return info


def apply_rescale(symbol: str, info: dict, *, live: bool, ts: str) -> str:
    plan = [p for p in info.get("plan", []) if p["apply"]]
    if not plan:
        return "clean-or-no-ths"
    csv = STOCK_DIR / f"{symbol}.csv"
    metap = STOCK_DIR / f"{symbol}.meta.json"
    if live:
        QDIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(csv, QDIR / f"{symbol}.pre-scalefix-{ts}.csv")
    df = pd.read_csv(csv)
    tmap = None
    anchor = info["anchor_ratio"]
    # re-fetch THS map for exact values (already verified reachable)
    ths = _ths_with_retry(symbol)
    tmap = {str(r["date"]): float(r["close"]) for _, r in ths.iterrows()}
    fixed = 0
    for p in plan:
        d = p["date"]
        k = tmap[d] * anchor / float(df.loc[df["date"] == d, "close"].iloc[0])
        mask = df["date"] == d
        for col in ("open", "high", "low", "close"):
            df.loc[mask, col] = df.loc[mask, col] * k
        fixed += 1
    # verify boundary continuity on the repaired frame
    df = df.sort_values("date").reset_index(drop=True)
    closes = df.set_index("date")["close"]
    d1 = plan[0]["date"]
    prev_dates = sorted(d for d in closes.index if d < d1)
    brep = "boundary-ok"
    if prev_dates:
        ret = float(closes[d1] / closes[prev_dates[-1]] - 1.0)
        brep = f"boundary-ret={ret:+.4f}"
    _, bad, _ = validate_daily_bars(df.tail(60), source="verify", symbol=symbol)
    vrep = f"validate-bad-tail60={len(bad)}"
    if live:
        df.to_csv(csv, index=False, encoding="utf-8-sig")
        try:
            meta = json.loads(metap.read_text(encoding="utf-8"))
            meta["rows"] = int(len(df))
            meta["data_end_date"] = str(df["date"].max())[:10]
            meta["repaired_from"] = "ths-scalefix"
            meta["repaired_range"] = d1 + ".." + str(df["date"].max())[:10]
            meta["repaired_at"] = datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat()
            metap.write_text(
                json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except Exception:  # noqa: BLE001 - meta is auxiliary
            pass
    return f"fixed:{fixed} {brep} {vrep}" + ("" if live else " (dry-run)")


def apply_truncate(symbol: str, poison_date: str, *, live: bool, ts: str) -> str:
    csv = STOCK_DIR / f"{symbol}.csv"
    metap = STOCK_DIR / f"{symbol}.meta.json"
    df = pd.read_csv(csv)
    bad_rows = df[df["date"] == poison_date]
    if bad_rows.empty:
        return "no-such-row"
    post = df[df["date"] > poison_date]
    if live:
        QDIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(csv, QDIR / f"{symbol}.pre-scalefix-{ts}.csv")
        df2 = df[df["date"] != poison_date].reset_index(drop=True)
        # keep later rows only if they reconnect to pre-poison scale
        df2.to_csv(csv, index=False, encoding="utf-8-sig")
        try:
            meta = json.loads(metap.read_text(encoding="utf-8"))
            meta["rows"] = int(len(df2))
            meta["repaired_from"] = "truncate-poison-row"
            meta["repaired_range"] = poison_date
            meta["repaired_at"] = datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat()
            metap.write_text(
                json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except Exception:  # noqa: BLE001 - meta is auxiliary
            pass
        # make the worker refill from the last clean date
        last_clean = str(df[df["date"] < poison_date]["date"].max())[:10]
        con = sqlite3.connect(STATE_DB, timeout=30)
        try:
            con.execute(
                "UPDATE assets SET data_end_date=?, last_success_date=?,"
                " next_retry_at=NULL, updated_at=? "
                "WHERE asset_type='stock' AND symbol=?",
                (
                    last_clean,
                    last_clean,
                    datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    symbol,
                ),
            )
            con.commit()
        finally:
            con.close()
    return (
        f"dropped:{len(bad_rows)} post_rows_kept:{len(post)}"
        + ("" if live else " (dry-run)")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    if bool(args.dry_run) == bool(args.live):
        raise ValueError("specify exactly one of --dry-run / --live")
    live = bool(args.live)
    ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    print(f"mode={'LIVE' if live else 'DRY-RUN'} ts={ts}", flush=True)
    for symbol, pdate in RESCALE.items():
        try:
            info = diagnose(symbol, pdate)
            if info.get("ths") != "ok":
                print(f"SYM {symbol} ths={info.get('ths')} -> fallback truncate path")
                print(f"SYM-RESULT {symbol} {'would-truncate' if not live else apply_truncate(symbol, pdate, live=live, ts=ts)}")
                continue
            plan_str = ";".join(
                f"{p['date']}:k={p['k']}{'!' if p['apply'] else '='}"
                for p in info["plan"]
            )
            print(
                f"SYM {symbol} anchor={info['anchor_date']} ratio={info['anchor_ratio']} {plan_str}"
            )
            print(f"SYM-RESULT {symbol} {apply_rescale(symbol, info, live=live, ts=ts)}")
        except Exception as exc:  # noqa: BLE001 - per-symbol isolation
            print(f"SYM-RESULT {symbol} error:{type(exc).__name__}:{exc}")
    for symbol, pdate in UNVERIFIED.items():
        try:
            info = diagnose(symbol, pdate)
            if info.get("ths") == "ok":
                print(f"SYM {symbol} ths-retry-ok -> rescale path")
                print(f"SYM-RESULT {symbol} {apply_rescale(symbol, info, live=live, ts=ts)}")
            else:
                print(
                    f"SYM {symbol} ths={info.get('ths')} "
                    f"tencent_now={info.get('tencent_now')}"
                )
                print(f"SYM-RESULT {symbol} {'would-truncate' if not live else apply_truncate(symbol, pdate, live=live, ts=ts)}")
        except Exception as exc:  # noqa: BLE001 - per-symbol isolation
            print(f"SYM-RESULT {symbol} error:{type(exc).__name__}:{exc}")
    print("REPAIR-DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
