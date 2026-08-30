from __future__ import annotations

import copy
import json
import pathlib
import sqlite3
import tempfile
import unittest
from contextlib import ExitStack
from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from assistant import paths, research
from assistant.config import DEFAULTS
from fetch_state import FetchState
from fetch_utils import cached_end_date, merge_daily_frames


class FetchStateTests(unittest.TestCase):
    def test_release_leases_returns_only_owned_claims_to_pending(self) -> None:
        with tempfile.TemporaryDirectory(prefix="fetch-release-", dir=ROOT / "tests") as directory:
            state = FetchState(pathlib.Path(directory) / "state.sqlite3")
            state.register_assets("stock", ["000001", "000002"])
            claimed = state.claim_due(
                "stock",
                symbols=["000001", "000002"],
                limit=2,
                target_date="2026-08-26",
                owner="worker-a",
            )
            self.assertEqual(len(claimed), 2)

            self.assertEqual(
                state.release_leases(
                    "stock",
                    ["000001", "000002"],
                    owner="worker-b",
                ),
                0,
            )
            self.assertEqual(
                state.release_leases(
                    "stock",
                    ["000001", "000002"],
                    owner="worker-a",
                ),
                2,
            )
            reclaimed = state.claim_due(
                "stock",
                symbols=["000001", "000002"],
                limit=2,
                target_date="2026-08-26",
                owner="worker-c",
            )
            self.assertEqual(
                [item["symbol"] for item in reclaimed],
                ["000001", "000002"],
            )

    def test_cache_backfill_excludes_assets_with_persisted_end_dates(self) -> None:
        with tempfile.TemporaryDirectory(prefix="fetch-state-", dir=ROOT / "tests") as directory:
            state = FetchState(pathlib.Path(directory) / "state.sqlite3")
            state.register_assets(
                "otc_fund",
                ["000001", "000002"],
                data_end_dates={"000001": "2026-08-26"},
            )

            pending = state.assets_needing_cache_backfill(
                "otc_fund",
                ["000001", "000002", "000003"],
            )

        self.assertEqual(pending, ["000002", "000003"])

    def test_claim_success_and_retry_backoff_are_persistent(self) -> None:
        with tempfile.TemporaryDirectory(prefix="fetch-state-", dir=ROOT / "tests") as directory:
            state = FetchState(pathlib.Path(directory) / "state.sqlite3")
            state.register_assets(
                "stock",
                ["000001", "600519"],
                data_end_dates={"600519": "2026-08-26"},
            )

            claimed = state.claim_due(
                "stock",
                symbols=["000001", "600519"],
                limit=10,
                target_date="2026-08-26",
                owner="worker-a",
            )
            self.assertEqual([item["symbol"] for item in claimed], ["000001"])

            state.mark_success(
                "stock",
                "000001",
                data_end_date="2026-08-26",
                source="test",
                rows=10,
                refresh_interval_seconds=86400,
                owner="worker-a",
            )
            progress = state.progress(
                ["stock"],
                symbols={"stock": ["000001", "600519"]},
                target_date="2026-08-26",
            )
            self.assertEqual(progress["completed"], 2)
            self.assertEqual(progress["coverage"], 1.0)
            self.assertEqual(
                state.claim_due(
                    "stock",
                    symbols=["000001"],
                    limit=10,
                    target_date="2026-08-26",
                    owner="worker-b",
                ),
                [],
            )

            state.mark_failure(
                "stock",
                "000001",
                error="temporary",
                backoff_base_seconds=1,
                backoff_max_seconds=2,
                owner="worker-a",
            )
            self.assertEqual(
                state.get_job("missing"),
                None,
            )

    def test_new_target_date_bypasses_success_refresh_once(self) -> None:
        with tempfile.TemporaryDirectory(prefix="fetch-target-", dir=ROOT / "tests") as directory:
            state = FetchState(pathlib.Path(directory) / "state.sqlite3")
            state.register_assets("stock", ["000001", "000002"])
            previous_attempt = datetime(2026, 8, 26, 12, tzinfo=timezone.utc)
            with patch("fetch_state.utc_now", return_value=previous_attempt):
                state.mark_success(
                    "stock",
                    "000001",
                    data_end_date="2026-08-26",
                    source="test",
                    rows=10,
                    refresh_interval_seconds=86400,
                )

            previous_claim = state.claim_due(
                "stock",
                symbols=["000002"],
                target_date="2026-08-26",
                owner="worker-old",
                now=previous_attempt,
            )
            self.assertEqual(len(previous_claim), 1)
            state.mark_failure(
                "stock",
                "000002",
                error="temporary",
                backoff_base_seconds=86400,
                backoff_max_seconds=86400,
                owner="worker-old",
            )

            new_target_attempt = datetime(2026, 8, 27, 1, tzinfo=timezone.utc)
            claimed = state.claim_due(
                "stock",
                symbols=["000001", "000002"],
                target_date="2026-08-27",
                owner="worker-new",
                now=new_target_attempt,
            )
            self.assertEqual([item["symbol"] for item in claimed], ["000001"])

            with patch("fetch_state.utc_now", return_value=new_target_attempt):
                state.mark_success(
                    "stock",
                    "000001",
                    data_end_date="2026-08-26",
                    source="test",
                    rows=10,
                    refresh_interval_seconds=86400,
                    verified_through="2026-08-27",
                    owner="worker-new",
                )
            self.assertEqual(
                state.claim_due(
                    "stock",
                    symbols=["000001"],
                    target_date="2026-08-27",
                    owner="worker-repeat",
                    now=datetime(2026, 8, 27, 2, tzinfo=timezone.utc),
                ),
                [],
            )
            next_target_claim = state.claim_due(
                "stock",
                symbols=["000001"],
                target_date="2026-08-28",
                owner="worker-next",
                now=datetime(2026, 8, 28, 2, tzinfo=timezone.utc),
            )
            self.assertEqual([item["symbol"] for item in next_target_claim], ["000001"])

    def test_job_claim_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory(prefix="fetch-job-", dir=ROOT / "tests") as directory:
            state = FetchState(pathlib.Path(directory) / "state.sqlite3")
            state.create_job(
                "20260826T000000Z-00000001",
                kind="research",
                asset_scope="stock",
                metadata={"asset_types": ["stock"]},
            )
            self.assertTrue(state.claim_job("20260826T000000Z-00000001"))
            self.assertFalse(state.claim_job("20260826T000000Z-00000001"))

    def test_progress_uses_target_date_per_asset_type(self) -> None:
        with tempfile.TemporaryDirectory(prefix="fetch-targets-", dir=ROOT / "tests") as directory:
            state = FetchState(pathlib.Path(directory) / "state.sqlite3")
            state.register_assets("etf", ["510300"])
            state.register_assets("otc_fund", ["110020"])
            state.mark_success(
                "etf",
                "510300",
                data_end_date="2026-08-26",
                source="test",
                rows=1,
                refresh_interval_seconds=86400,
            )
            state.mark_success(
                "otc_fund",
                "110020",
                data_end_date="2026-08-25",
                source="test",
                rows=1,
                refresh_interval_seconds=86400,
            )

            progress = state.progress(
                ["etf", "otc_fund"],
                symbols={"etf": ["510300"], "otc_fund": ["110020"]},
                target_date={"etf": "2026-08-26", "otc_fund": "2026-08-25"},
            )

            self.assertEqual(progress["completed"], 2)
            self.assertEqual(progress["status"], "complete")
            self.assertEqual(
                progress["target_dates"],
                {"etf": "2026-08-26", "otc_fund": "2026-08-25"},
            )

    def test_quarantined_asset_is_persistent_and_not_claimed_again(self) -> None:
        with tempfile.TemporaryDirectory(prefix="fetch-quarantine-", dir=ROOT / "tests") as directory:
            state = FetchState(pathlib.Path(directory) / "state.sqlite3")
            state.register_assets("otc_fund", ["000001"])
            claimed = state.claim_due(
                "otc_fund",
                symbols=["000001"],
                target_date="2026-08-26",
                owner="worker-a",
            )
            self.assertEqual(len(claimed), 1)

            state.mark_quarantined(
                "otc_fund",
                "000001",
                error="invalid NAV rows",
                owner="worker-a",
            )

            self.assertEqual(
                state.claim_due(
                    "otc_fund",
                    symbols=["000001"],
                    target_date="2026-08-26",
                    owner="worker-b",
                ),
                [],
            )
            progress = state.progress(
                ["otc_fund"],
                symbols={"otc_fund": ["000001"]},
                target_date="2026-08-26",
            )
            self.assertEqual(progress["failed"], 1)
            self.assertEqual(progress["pending"], 1)

    def test_quarantine_details_are_persisted_and_matching_rows_can_be_requeued(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="fetch-quarantine-details-", dir=ROOT / "tests"
        ) as directory:
            state = FetchState(pathlib.Path(directory) / "state.sqlite3")
            state.register_assets("stock", ["000001", "000002"])
            state.claim_due(
                "stock",
                symbols=["000001", "000002"],
                limit=2,
                target_date="2026-08-26",
                owner="worker-a",
            )
            anomalies = {"volume": {"conflicting_rows": 2}}
            state.mark_quarantined(
                "stock",
                "000001",
                error="schema conflict",
                failure_class="schema_conflict",
                field_anomalies=anomalies,
                owner="worker-a",
            )
            with state._connect() as connection:
                row = connection.execute(
                    "SELECT status, failure_class, failure_details_json FROM assets "
                    "WHERE asset_type = ? AND symbol = ?",
                    ("stock", "000001"),
                ).fetchone()
            self.assertEqual(row["status"], "quarantined")
            self.assertEqual(row["failure_class"], "schema_conflict")
            self.assertEqual(json.loads(row["failure_details_json"]), anomalies)
            self.assertEqual(
                state.requeue_quarantined(
                    "stock", symbols=["000001"], failure_class="schema_conflict"
                ),
                1,
            )
            claimed = state.claim_due(
                "stock",
                symbols=["000001"],
                target_date="2026-08-26",
                owner="worker-b",
            )
            self.assertEqual([item["symbol"] for item in claimed], ["000001"])
            with state._connect() as connection:
                row = connection.execute(
                    "SELECT failure_class, failure_details_json FROM assets "
                    "WHERE asset_type = ? AND symbol = ?",
                    ("stock", "000001"),
                ).fetchone()
            self.assertIsNone(row["failure_class"])
            self.assertIsNone(row["failure_details_json"])

    def test_old_assets_table_is_migrated_without_rebuilding_it(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="fetch-migrate-", dir=ROOT / "tests"
        ) as directory:
            database = pathlib.Path(directory) / "state.sqlite3"
            connection = sqlite3.connect(database)
            connection.execute(
                """
                CREATE TABLE assets (
                    asset_type TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    data_end_date TEXT,
                    last_success_date TEXT,
                    last_attempt_at TEXT,
                    next_retry_at TEXT,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'pending',
                    source TEXT,
                    last_error TEXT,
                    rows INTEGER,
                    lease_owner TEXT,
                    lease_until TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (asset_type, symbol)
                )
                """,
            )
            connection.execute(
                "INSERT INTO assets (asset_type, symbol, updated_at) VALUES (?, ?, ?)",
                ("stock", "000001", "2026-08-26T00:00:00+00:00"),
            )
            connection.commit()
            connection.close()
            FetchState(database)
            connection = sqlite3.connect(database)
            columns = {row[1] for row in connection.execute("PRAGMA table_info(assets)")}
            preserved = connection.execute(
                "SELECT symbol, status FROM assets WHERE asset_type = ?", ("stock",)
            ).fetchone()
            connection.close()
        self.assertEqual(preserved, ("000001", "pending"))
        self.assertIn("failure_class", columns)
        self.assertIn("failure_details_json", columns)
class FetchUtilityTests(unittest.TestCase):
    def test_cached_end_date_and_incremental_merge(self) -> None:
        with tempfile.TemporaryDirectory(prefix="fetch-utils-", dir=ROOT / "tests") as directory:
            path = pathlib.Path(directory) / "000001.csv"
            pd.DataFrame(
                {
                    "date": ["2026-08-25"],
                    "open": [1],
                    "high": [2],
                    "low": [0.5],
                    "close": [1.5],
                    "volume": [10],
                }
            ).to_csv(path, index=False)
            self.assertEqual(cached_end_date(path), "2026-08-25")
            incoming = pd.DataFrame(
                {
                    "date": pd.to_datetime(["2026-08-25", "2026-08-26"]),
                    "open": [1.1, 1.2],
                    "high": [2.1, 2.2],
                    "low": [0.6, 0.7],
                    "close": [1.6, 1.7],
                    "volume": [11, 12],
                }
            )
            merged = merge_daily_frames(path, incoming, source="test")
            self.assertEqual(len(merged), 2)
            self.assertEqual(merged.iloc[-1]["close"], 1.7)


class QueuedResearchTests(unittest.TestCase):
    def test_full_universe_research_returns_queued_job(self) -> None:
        with tempfile.TemporaryDirectory(prefix="queued-research-", dir=ROOT / "tests") as directory:
            root = pathlib.Path(directory)
            data_dir = root / "data"
            results_dir = root / "results"
            runs_dir = results_dir / "runs"
            logs_dir = root / "logs"
            state_db = root / "fetch_state.sqlite3"
            for path in (data_dir, results_dir, runs_dir, logs_dir):
                path.mkdir(parents=True, exist_ok=True)
            manifest_path = data_dir / "universe.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "complete": False,
                        "stock_universe_complete": True,
                        "stock": [
                            {
                                "symbol": "600519",
                                "name": "test",
                                "asset_type": "stock",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            config = copy.deepcopy(DEFAULTS)
            config["fetch"]["enabled"] = True
            config["fetch"]["state_db"] = str(state_db)
            config["fetch"]["background_worker_enabled"] = False
            config["paths"]["data_dir"] = str(data_dir)
            config["paths"]["results_dir"] = str(results_dir)
            config["paths"]["logs_dir"] = str(logs_dir)
            with ExitStack() as stack:
                stack.enter_context(patch.object(research, "DATA_DIR", data_dir))
                stack.enter_context(patch.object(research, "UNIVERSE_PATH", manifest_path))
                stack.enter_context(patch.object(research, "ASSISTANT_RESULTS_DIR", results_dir))
                stack.enter_context(patch.object(research, "RUNS_DIR", runs_dir))
                stack.enter_context(patch.object(paths, "RUNS_DIR", runs_dir))
                stack.enter_context(patch.object(research, "load_config", return_value=config))
                stack.enter_context(
                    patch.object(
                        research,
                        "_now",
                        return_value=datetime(2026, 8, 26, 10, 0, tzinfo=timezone.utc),
                    )
                )
                stack.enter_context(
                    patch.object(
                        research,
                        "_run_id",
                        return_value="20260826T100000Z-00000001",
                    )
                )
                result = research.run_research("daily", "stock")

            self.assertEqual(result["status"], "queued")
            self.assertEqual(result["job_id"], "20260826T100000Z-00000001")
            self.assertEqual(result["completed"] + result["pending"], 1)
            self.assertEqual(
                json.loads(
                    (
                        runs_dir
                        / "2026"
                        / "08"
                        / "26"
                        / "20260826T100000Z-00000001"
                        / "status.json"
                    ).read_text(encoding="utf-8")
                )["status"],
                "queued",
            )


if __name__ == "__main__":
    unittest.main()
