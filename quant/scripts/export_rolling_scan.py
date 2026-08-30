"""Export the resumable rolling-scan state without mutating the database."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


SCORE_COLUMNS = [
    "asset_type",
    "symbol",
    "data_end_date",
    "rows",
    "name",
    "quality_status",
    "analyzed_at",
    "trend_score",
    "return_20d",
    "return_60d",
    "return_252d",
    "volatility_60d_annualized",
    "max_drawdown",
    "current_drawdown",
    "ma20",
    "ma60",
    "ma200",
    "average_amount_20d",
    "average_volume_20d",
]

EVENT_COLUMNS = [
    "seq",
    "created_at",
    "job_id",
    "event",
    "asset_type",
    "symbol",
    "rank",
    "score",
    "reason",
]


def _atomic_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=columns,
                extrasaction="ignore",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise
    return len(rows)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=True, indent=2)
            handle.write("\n")
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _rank_candidates(
    rows: list[dict[str, Any]],
    *,
    as_of: date,
    pool_size: int,
    min_rows: int,
    min_amount: float,
    stale_after_days: int,
) -> list[dict[str, Any]]:
    eligible: list[dict[str, Any]] = []
    for row in rows:
        if row.get("quality_status") != "ok":
            continue
        try:
            row_count = int(row["rows"])
        except (TypeError, ValueError):
            continue
        try:
            data_date = date.fromisoformat(str(row["data_end_date"]))
        except (TypeError, ValueError):
            continue
        try:
            amount = float(row["average_amount_20d"])
            score = float(row["trend_score"])
        except (TypeError, ValueError):
            continue
        if row_count < min_rows:
            continue
        if (as_of - data_date).days > stale_after_days:
            continue
        if amount < min_amount:
            continue
        eligible.append(row)

    eligible.sort(
        key=lambda row: (-float(row["trend_score"]), str(row["symbol"]))
    )
    candidates: list[dict[str, Any]] = []
    for rank, row in enumerate(eligible[:pool_size], start=1):
        item = dict(row)
        item["rank"] = rank
        candidates.append(item)
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--cutoff", required=True)
    parser.add_argument("--as-of", required=True)
    args = parser.parse_args()

    cutoff = args.cutoff
    as_of = date.fromisoformat(args.as_of)
    generated_at = datetime.now(timezone.utc).isoformat()

    with sqlite3.connect(args.db) as connection:
        connection.row_factory = sqlite3.Row
        current_rows = [
            dict(row)
            for row in connection.execute(
                """
                SELECT asset_type, symbol, data_end_date, rows, name,
                       quality_status, analyzed_at, trend_score,
                       return_20d, return_60d, return_252d,
                       volatility_60d_annualized, max_drawdown,
                       current_drawdown, ma20, ma60, ma200,
                       average_amount_20d, average_volume_20d
                FROM scores
                ORDER BY
                    CASE WHEN trend_score IS NULL THEN 1 ELSE 0 END,
                    trend_score DESC, asset_type, symbol
                """
            )
        ]
        main_rows = [
            row for row in current_rows if str(row["analyzed_at"]) < cutoff
        ]
        usable_rows = [
            row for row in main_rows if row["quality_status"] == "ok"
        ]
        events = [
            dict(row)
            for row in connection.execute(
                """
                SELECT seq, created_at, job_id, event, asset_type,
                       symbol, rank, score, reason
                FROM pool_events
                ORDER BY seq
                """
            )
        ]

    main_rows.sort(
        key=lambda row: (
            row["trend_score"] is None,
            -(float(row["trend_score"]) if row["trend_score"] is not None else 0.0),
            str(row["asset_type"]),
            str(row["symbol"]),
        )
    )
    usable_rows.sort(
        key=lambda row: (
            -(float(row["trend_score"]) if row["trend_score"] is not None else 0.0),
            str(row["asset_type"]),
            str(row["symbol"]),
        )
    )

    candidates: list[dict[str, Any]] = []
    for asset_type in ("stock", "etf"):
        candidates.extend(
            _rank_candidates(
                [row for row in usable_rows if row["asset_type"] == asset_type],
                as_of=as_of,
                pool_size=30,
                min_rows=120,
                min_amount=30_000_000.0,
                stale_after_days=15,
            )
        )
    candidates.sort(key=lambda row: (str(row["asset_type"]), int(row["rank"])))

    audit_rows = []
    for row in current_rows:
        item = dict(row)
        item["record_scope"] = (
            "main_scan" if str(row["analyzed_at"]) < cutoff else "post_timeout_test"
        )
        audit_rows.append(item)

    stem = "rolling_scan_20260828"
    paths = {
        "audit_csv": args.out_dir / f"{stem}_scores_current_audit.csv",
        "main_csv": args.out_dir / f"{stem}_scores_main_scan.csv",
        "usable_csv": args.out_dir / f"{stem}_scores_usable.csv",
        "candidates_csv": args.out_dir / f"{stem}_candidates_top30.csv",
        "events_csv": args.out_dir / f"{stem}_pool_events.csv",
    }

    audit_count = _atomic_csv(
        paths["audit_csv"],
        SCORE_COLUMNS + ["record_scope"],
        audit_rows,
    )
    main_count = _atomic_csv(paths["main_csv"], SCORE_COLUMNS, main_rows)
    usable_count = _atomic_csv(paths["usable_csv"], SCORE_COLUMNS, usable_rows)
    candidate_count = _atomic_csv(
        paths["candidates_csv"],
        ["rank"] + SCORE_COLUMNS,
        candidates,
    )
    event_count = _atomic_csv(paths["events_csv"], EVENT_COLUMNS, events)

    manifest = {
        "generated_at_utc": generated_at,
        "source_db": str(args.db.resolve()),
        "data_as_of": args.as_of,
        "main_scan_cutoff_exclusive_utc": cutoff,
        "main_scan_definition": (
            "scores rows analyzed before the timeout-test cutoff; "
            "five later rows are retained only in the audit CSV"
        ),
        "counts": {
            "current_score_rows": audit_count,
            "main_scan_score_rows": main_count,
            "main_scan_usable_rows": usable_count,
            "candidate_rows": candidate_count,
            "pool_event_rows": event_count,
            "post_timeout_test_rows_excluded_from_main": audit_count - main_count,
        },
        "candidate_rules": {
            "asset_types": ["stock", "etf"],
            "pool_size_per_asset_type": 30,
            "min_rows": 120,
            "min_average_amount_20d": 30000000.0,
            "stale_after_days": 15,
            "sort": "trend_score descending, symbol ascending",
        },
        "files": {key: str(path.resolve()) for key, path in paths.items()},
    }
    manifest_path = args.out_dir / f"{stem}_manifest.json"
    _atomic_json(manifest_path, manifest)

    print(
        json.dumps(
            {
                "status": "ok",
                "manifest": str(manifest_path.resolve()),
                "counts": manifest["counts"],
                "candidate_counts": {
                    asset_type: sum(
                        1 for row in candidates if row["asset_type"] == asset_type
                    )
                    for asset_type in ("stock", "etf")
                },
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
