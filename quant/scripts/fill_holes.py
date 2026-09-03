# -*- coding: utf-8 -*-
import json, time
from pathlib import Path
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from price_guards import fetch_ths_daily, validate_daily_bars
ROOT = Path(__file__).resolve().parent.parent
holefile = sys.argv[1] if len(sys.argv) > 1 else "D:/dsh-股票/quant/logs/codex-runs/holes.json"
holes = json.loads(Path(holefile).read_text(encoding="utf-8"))
counts, filled_total = {}, 0
for i, h in enumerate(holes):
    sym, miss = h["symbol"], h["missing"]
    try:
        csv = ROOT / "data" / "stocks" / (sym + ".csv")
        df = pd.read_csv(csv)
        ths = fetch_ths_daily(sym, timeout=10)
        time.sleep(3.0)
        if ths is None or not len(ths):
            counts["no_ths"] = counts.get("no_ths", 0) + 1
            continue
        tmap = {r["date"]: r for _, r in ths.iterrows()}
        have = set(df["date"].astype(str))
        cands = sorted([d for d in miss if d not in have and d in tmap])
        if not cands:
            counts["nothing_to_fill"] = counts.get("nothing_to_fill", 0) + 1
            continue
        def _tof(x):
            try:
                return float(x)
            except Exception:
                return 0.0
        anch = sorted([d for d in have if d < cands[0] and d in tmap])
        pfac, vfac = 1.0, 1.0
        if anch:
            ad = anch[-1]
            tc = _tof(tmap[ad]["close"])
            tv = _tof(tmap[ad]["volume"])
            frow = df.loc[df["date"] == ad]
            if tc and len(frow):
                pfac = _tof(frow["close"].iloc[0]) / tc
            if tv and len(frow):
                vr = _tof(frow["volume"].iloc[0]) / tv
                vfac = 0.01 if 0.005 < vr < 0.02 else (100.0 if 50 < vr < 200 else 1.0)
        if not (0.2 < pfac < 5.0):
            counts["absurd_factor"] = counts.get("absurd_factor", 0) + 1
            continue
        n = 0
        for ds in cands:
            t = tmap[ds]
            df.loc[len(df)] = [ds, t["open"] * pfac, t["high"] * pfac, t["low"] * pfac, t["close"] * pfac, t["volume"] * vfac, t["amount"], t["turnover"]]
            n += 1
        df = df.sort_values("date").reset_index(drop=True)
        df.to_csv(csv, index=False, encoding="utf-8-sig")
        _, bad, _ = validate_daily_bars(df.tail(400), source="fillverify", symbol=sym)
        r2 = [str(d) for d in bad["date"]] if len(bad) else []
        if r2:
            counts["residual"] = counts.get("residual", 0) + 1
        else:
            counts["filled"] = counts.get("filled", 0) + 1
            filled_total += n
    except Exception as exc:
        counts["error"] = counts.get("error", 0) + 1
        print("FILL-ERROR " + str(sym) + " " + type(exc).__name__ + " " + str(exc)[:160], flush=True)
    if (i + 1) % 100 == 0:
        print("fill progress " + str(i + 1) + "/" + str(len(holes)) + " " + json.dumps(counts, ensure_ascii=True), flush=True)
print("FILL-DONE " + json.dumps({"total": len(holes), "rows_filled": filled_total, "counts": counts}, ensure_ascii=True))
