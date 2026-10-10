# -*- coding: utf-8 -*-
"""One-off quarantine normaliser (T1).

Normalises ``quant/data/*/quarantine/*.quarantine.csv`` files:
  * date cells -> ``YYYY-MM-DD`` (format only; no value changes);
  * exact duplicate rows (identical in every column) -> dropped;
  * rows sharing a date but differing in other columns are all kept and counted.

Default is ``--dry-run`` (bounded counts only, no writes). ``--apply`` backs each
file up as ``<file>.pre-normalize.bak`` before rewriting. The T1 agent does NOT
run this against production; T2 owns that.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

DEFAULT_ROOT = Path("D:/dsh-gupiao/quant/data")


def _date_str(value):
    if value is None or (isinstance(value, float) and value != value):
        return value
    text = str(value).strip()
    if not text or text.lower() in {"nan", "nat"}:
        return value
    try:
        return pd.Timestamp(text).strftime("%Y-%m-%d")
    except Exception:
        return text[:10] if len(text) >= 10 else text


def normalise_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return (normalised frame, stats) for one quarantine table."""
    out = frame.copy()
    stats = {"rows_in": int(len(out)), "exact_dupes": 0, "date_conflicts": 0}
    mixed = 0
    if "date" in out.columns:
        raw = out["date"].astype(str)
        normalised = out["date"].map(_date_str).astype(str)
        mixed = int((raw != normalised).sum())
        out["date"] = normalised
    before = len(out)
    out = out.drop_duplicates(keep="first")
    stats["exact_dupes"] = before - len(out)
    if "date" in out.columns:
        dup_dates = out["date"].duplicated(keep=False)
        if bool(dup_dates.any()):
            # 同日期但其余列不同：全部保留，仅计数。
            differing = out.loc[dup_dates].groupby("date").size()
            stats["date_conflicts"] = int(sum(1 for n in differing if n > 1))
    stats["mixed_format_rows"] = mixed
    stats["rows_out"] = int(len(out))
    return out, stats


def iter_quarantine_files(root: Path):
    for path in sorted(root.glob("**/quarantine/*.quarantine.csv")):
        yield path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Normalise quarantine CSV files.")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--apply", action="store_true",
                        help="rewrite files (default: dry-run, no writes)")
    args = parser.parse_args(argv)
    root = Path(args.root)
    files = list(iter_quarantine_files(root))
    totals = {"files": 0, "rows": 0, "exact_dupes": 0, "date_conflicts": 0,
              "mixed_format_rows": 0, "written": 0}
    for path in files:
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        out, stats = normalise_frame(frame)
        totals["files"] += 1
        totals["rows"] += stats["rows_in"]
        totals["exact_dupes"] += stats["exact_dupes"]
        totals["date_conflicts"] += stats["date_conflicts"]
        totals["mixed_format_rows"] += stats["mixed_format_rows"]
        if args.apply and (stats["exact_dupes"] or stats["mixed_format_rows"]):
            backup = path.with_name(path.name + ".pre-normalize.bak")
            if not backup.exists():
                backup.write_bytes(path.read_bytes())
            out.to_csv(path, index=False, encoding="utf-8-sig")
            totals["written"] += 1
    print(json.dumps({"mode": "apply" if args.apply else "dry-run",
                      "root": str(root), **totals}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
