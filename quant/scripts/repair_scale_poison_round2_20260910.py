# -*- coding: utf-8 -*-
"""One-off repair round-2 (2026-09-10): listing-day basis + fresh-listing row.

Two new THS-verified poisons from detector round-3:
- 688377 (科创次新, rows=1498): 2026-09-08/09 +23.9%/+25.6% over local
  09-03. THS agrees 09-01..09-03 but shows 09-08 18.57 vs local 23.17.
  The 09-03 local close is fine; only 09-08/09 are wrong-basis splices.
  Fix = THS-return rescale of the 2 rows to pre-poison local basis.
- 920090 (北交所, rows=1975): same shape, 09-08/09 +33.9%/+33.6% over
  local 09-03; THS 09-08 3.61 vs local 4.74. Same fix.
- 603448 (次新, rows=3, starts 2026-09-07): THS agrees exactly
  (83.48/70.01/68.22). True listing volatility, NO fix. Detector
  false-positive on fresh listings (no 60-row implied median yet).

Usage: --dry-run / --live (backup + fix + verify), same contract as
repair_scale_poison_20260910.py.
"""
from __future__ import annotations

import argparse
import datetime
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from price_guards import fetch_ths_daily, validate_daily_bars

ROOT = Path(__file__).resolve().parent.parent
STOCK_DIR = ROOT / "data" / "stocks"
QDIR = STOCK_DIR / "quarantine"

TARGETS = {
    "688377": ("2026-09-03", ["2026-09-08", "2026-09-09"]),
    "920090": ("2026-09-03", ["2026-09-08", "2026-09-09"]),
}
APPLY_THRESHOLD = 0.05


def _ths_with_retry(symbol: str, tries: int = 3):
    for _ in range(tries):
        try:
            ths = fetch_ths_daily(symbol)
            if ths is not None and len(ths):
                return ths
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1.0)
    return None


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
    for symbol, (anchor_d, fix_dates) in TARGETS.items():
        try:
            csv = STOCK_DIR / f"{symbol}.csv"
            metap = STOCK_DIR / f"{symbol}.meta.json"
            df = pd.read_csv(csv, encoding="utf-8-sig")
            lmap = {str(r["date"])[:10]: float(r["close"]) for _, r in df.iterrows()}
            ths = _ths_with_retry(symbol)
            if ths is None:
                print(f"SYM-RESULT {symbol} error:ths-unreachable")
                continue
            tmap = {str(r["date"]): float(r["close"]) for _, r in ths.iterrows()}
            anchor = lmap[anchor_d] / tmap[anchor_d]
            plan = []
            for d in fix_dates:
                k = tmap[d] * anchor / lmap[d]
                plan.append((d, round(k, 4), abs(k - 1.0) > APPLY_THRESHOLD))
            print(f"SYM {symbol} anchor={anchor_d} ratio={anchor:.4f} " +
                  ";".join(f"{d}:k={k}{'!' if a else '='}" for d, k, a in plan))
            todo = [(d, tmap[d] * anchor / lmap[d]) for d, _, a in plan for _ in [0] if a]
            # recompute exact k from live frame (same values, kept explicit)
            todo = []
            for d in fix_dates:
                k = tmap[d] * anchor / float(df.loc[df["date"] == d, "close"].iloc[0])
                if abs(k - 1.0) > APPLY_THRESHOLD:
                    todo.append((d, k))
            if live and todo:
                QDIR.mkdir(parents=True, exist_ok=True)
                shutil.copy2(csv, QDIR / f"{symbol}.pre-scalefix-{ts}.csv")
                for d, k in todo:
                    mask = df["date"] == d
                    for col in ("open", "high", "low", "close"):
                        df.loc[mask, col] = df.loc[mask, col] * k
                df = df.sort_values("date").reset_index(drop=True)
                df.to_csv(csv, index=False, encoding="utf-8-sig")
                try:
                    meta = json.loads(metap.read_text(encoding="utf-8"))
                    meta["rows"] = int(len(df))
                    meta["data_end_date"] = str(df["date"].max())[:10]
                    meta["repaired_from"] = "ths-scalefix-round2"
                    meta["repaired_range"] = todo[0][0] + ".." + todo[-1][0]
                    meta["repaired_at"] = datetime.datetime.now(
                        datetime.timezone.utc).isoformat()
                    metap.write_text(
                        json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
                except Exception:  # noqa: BLE001
                    pass
            _, bad, _ = validate_daily_bars(df.tail(60) if not (live and todo) else
                                            pd.read_csv(csv, encoding="utf-8-sig").tail(60),
                                            source="verify", symbol=symbol)
            # boundary check vs anchor day
            cur = pd.read_csv(csv, encoding="utf-8-sig") if (live and todo) else df
            closes = cur.set_index("date")["close"]
            d1 = fix_dates[0]
            prevs = sorted(d for d in closes.index if d < d1)
            brep = f"boundary-ret={float(closes[d1] / closes[prevs[-1]] - 1.0):+.4f}" if prevs else "no-prev"
            print(f"SYM-RESULT {symbol} fixed:{len(todo)} {brep} "
                  f"validate-bad-tail60={len(bad)}" + ("" if live else " (dry-run)"))
        except Exception as exc:  # noqa: BLE001
            print(f"SYM-RESULT {symbol} error:{type(exc).__name__}:{exc}")
    print("REPAIR-DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
