from __future__ import annotations

import copy
import json
import pathlib
import sqlite3
import tempfile
import unittest
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from assistant import paths, research
from assistant.config import DEFAULTS
from fetch_state import FetchState, iso_now
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

    def test_cache_reconciliation_clears_failure_after_csv_advances(self) -> None:
        with tempfile.TemporaryDirectory(prefix="fetch-reconcile-", dir=ROOT / "tests") as directory:
            state = FetchState(pathlib.Path(directory) / "state.sqlite3")
            state.register_assets(
                "stock",
                ["000001"],
                data_end_dates={"000001": "2026-08-26"},
            )
            claimed = state.claim_due(
                "stock",
                symbols=["000001"],
                target_date="2026-08-27",
                owner="worker-a",
            )
            self.assertEqual(len(claimed), 1)
            state.mark_failure(
                "stock",
                "000001",
                error="AttributeError: 'str' object has no attribute 'date'",
                failure_class="upstream_error",
                backoff_base_seconds=120,
                backoff_max_seconds=120,
                owner="worker-a",
            )

            reconciled = state.reconcile_cached_assets(
                "stock",
                {"000001": "2026-08-27"},
            )

            self.assertEqual(reconciled, 1)
            self.assertEqual(
                state.claim_due(
                    "stock",
                    symbols=["000001"],
                    target_date="2026-08-27",
                    owner="worker-b",
                ),
                [],
            )
            progress = state.progress(
                ["stock"],
                symbols={"stock": ["000001"]},
                target_date="2026-08-27",
            )
            self.assertEqual(progress["completed"], 1)
            connection = sqlite3.connect(state.path)
            try:
                row = connection.execute(
                    """
                    SELECT data_end_date, last_success_date, status, last_error,
                           failure_class, next_retry_at
                    FROM assets WHERE asset_type = 'stock' AND symbol = '000001'
                    """
                ).fetchone()
            finally:
                connection.close()
            self.assertEqual(
                row,
                ("2026-08-27", "2026-08-27", "success", None, None, None),
            )

    def test_cache_reconciliation_keeps_newer_failure_state(self) -> None:
        with tempfile.TemporaryDirectory(prefix="fetch-reconcile-", dir=ROOT / "tests") as directory:
            state = FetchState(pathlib.Path(directory) / "state.sqlite3")
            state.register_assets(
                "stock",
                ["000001"],
                data_end_dates={"000001": "2026-08-26"},
            )
            claimed = state.claim_due(
                "stock",
                symbols=["000001"],
                target_date="2026-08-27",
                owner="worker-a",
            )
            self.assertEqual(len(claimed), 1)
            state.mark_failure(
                "stock",
                "000001",
                error="temporary upstream failure",
                failure_class="upstream_error",
                backoff_base_seconds=120,
                backoff_max_seconds=120,
                owner="worker-a",
            )

            reconciled = state.reconcile_cached_assets(
                "stock",
                {"000001": "2026-08-26"},
            )

            self.assertEqual(reconciled, 0)
            connection = sqlite3.connect(state.path)
            try:
                row = connection.execute(
                    """
                    SELECT data_end_date, last_success_date, status, last_error
                    FROM assets WHERE asset_type = 'stock' AND symbol = '000001'
                    """
                ).fetchone()
            finally:
                connection.close()
            self.assertEqual(
                row,
                (
                    "2026-08-26",
                    "2026-08-26",
                    "failed",
                    "temporary upstream failure",
                ),
            )

    def test_lifecycle_snapshot_stops_claims_and_revives_active_symbols(self) -> None:
        with tempfile.TemporaryDirectory(prefix="fetch-lifecycle-", dir=ROOT / "tests") as directory:
            state = FetchState(pathlib.Path(directory) / "state.sqlite3")
            state.register_assets("stock", ["000001", "000002"])
            connection = sqlite3.connect(state.path)
            try:
                connection.execute(
                    """
                    UPDATE assets
                    SET status = 'failed', failure_class = 'empty_window', attempt_count = 2
                    WHERE asset_type = 'stock' AND symbol = '000001'
                    """
                )
                connection.commit()
            finally:
                connection.close()

            applied = state.apply_lifecycle_snapshot(
                "stock",
                {
                    "complete": True,
                    "candidates": ["000001", "000002"],
                    "active": ["000002"],
                    "inactive_candidates": ["000001"],
                    "sources": {"exchange": 1, "eastmoney": 1, "sina": 1},
                },
            )

            self.assertEqual(applied, {"inactivated": 1, "revived": 0})
            self.assertEqual(
                state.claim_due(
                    "stock",
                    symbols=["000001"],
                    target_date="2026-08-27",
                    owner="worker-a",
                ),
                [],
            )
            progress = state.progress(
                ["stock"],
                symbols={"stock": ["000001"]},
                target_date="2026-08-27",
            )
            self.assertEqual(progress["completed"], 1)
            self.assertEqual(progress["pending"], 0)

            revived = state.apply_lifecycle_snapshot(
                "stock",
                {
                    "complete": True,
                    "candidates": ["000001"],
                    "active": ["000001"],
                    "inactive_candidates": [],
                    "sources": {"exchange": 1, "eastmoney": 1, "sina": 1},
                },
            )

            self.assertEqual(revived, {"inactivated": 0, "revived": 1})
            claimed = state.claim_due(
                "stock",
                symbols=["000001"],
                target_date="2026-08-27",
                owner="worker-b",
            )
            self.assertEqual([row["symbol"] for row in claimed], ["000001"])

    def test_incomplete_lifecycle_snapshot_cannot_inactivate_symbols(self) -> None:
        with tempfile.TemporaryDirectory(prefix="fetch-lifecycle-", dir=ROOT / "tests") as directory:
            state = FetchState(pathlib.Path(directory) / "state.sqlite3")
            state.register_assets("stock", ["000001"])
            connection = sqlite3.connect(state.path)
            try:
                connection.execute(
                    """
                    UPDATE assets
                    SET status = 'failed', failure_class = 'empty_window', attempt_count = 2
                    WHERE asset_type = 'stock' AND symbol = '000001'
                    """
                )
                connection.commit()
            finally:
                connection.close()

            applied = state.apply_lifecycle_snapshot(
                "stock",
                {
                    "complete": False,
                    "candidates": ["000001"],
                    "active": [],
                    "inactive_candidates": ["000001"],
                },
            )

            self.assertEqual(applied, {"inactivated": 0, "revived": 0})
            claimed = state.claim_due(
                "stock",
                symbols=["000001"],
                target_date="2026-08-27",
                owner="worker-a",
            )
            self.assertEqual([row["symbol"] for row in claimed], ["000001"])

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

    def test_claim_due_prefilters_cross_border_before_limit(self) -> None:
        # 2026-09-09 空转回归：头部 3 个跨境 + limit=3 时旧逻辑返回 []；
        # 预过滤后应跳过头部认领到第 4 个国内标的，且被过滤行不加租约。
        import fetch_state as fetch_state_module

        with tempfile.TemporaryDirectory(prefix="fetch-cross-", dir=ROOT / "tests") as directory:
            state = FetchState(pathlib.Path(directory) / "state.sqlite3")
            state.register_assets("etf", ["158000", "158003", "159010", "159011"])
            names = {
                "etf": {
                    "158000": "港股通金融ETF鹏华",
                    "158003": "港股通医疗ETF嘉实",
                    "159010": "港股通科技ETF博时",
                    "159011": "养殖ETF华安",
                }
            }
            with (
                patch.object(fetch_state_module, "_should_filter_progress", return_value=True),
                patch.object(fetch_state_module, "_load_manifest_names_f", return_value=names),
            ):
                claimed = state.claim_due(
                    "etf",
                    symbols=["158000", "158003", "159010", "159011"],
                    limit=3,
                    target_date="2026-08-27",
                    owner="worker-a",
                )
            self.assertEqual([item["symbol"] for item in claimed], ["159011"])
            connection = sqlite3.connect(state.path)
            try:
                rows = {
                    row[0]: (row[1], row[2])
                    for row in connection.execute(
                        "SELECT symbol, status, attempt_count FROM assets WHERE asset_type = 'etf'"
                    ).fetchall()
                }
            finally:
                connection.close()
            self.assertEqual(rows["159011"][0], "running")
            for skipped in ("158000", "158003", "159010"):
                self.assertEqual(rows[skipped], ("pending", 0))

    def test_claim_due_cross_border_stock_never_filtered_and_missing_name_fail_open(self) -> None:
        import fetch_state as fetch_state_module

        with tempfile.TemporaryDirectory(prefix="fetch-cross-", dir=ROOT / "tests") as directory:
            state = FetchState(pathlib.Path(directory) / "state.sqlite3")
            state.register_assets("stock", ["600001"])
            state.register_assets("etf", ["159011"])
            names = {"etf": {}, "otc_fund": {}}
            with (
                patch.object(fetch_state_module, "_should_filter_progress", return_value=True),
                patch.object(fetch_state_module, "_load_manifest_names_f", return_value=names),
            ):
                # stock 永不过滤：名称含亚太也不应被拦（此处直接按 symbol 认领）。
                stock_claimed = state.claim_due(
                    "stock",
                    symbols=["600001"],
                    limit=1,
                    target_date="2026-08-27",
                    owner="worker-a",
                )
                # 缺名 fail-open：manifest 无名仍可认领。
                etf_claimed = state.claim_due(
                    "etf",
                    symbols=["159011"],
                    limit=1,
                    target_date="2026-08-27",
                    owner="worker-b",
                )
            self.assertEqual([item["symbol"] for item in stock_claimed], ["600001"])
            self.assertEqual([item["symbol"] for item in etf_claimed], ["159011"])

    def test_claim_due_cross_border_disabled_by_default(self) -> None:
        import fetch_state as fetch_state_module

        with tempfile.TemporaryDirectory(prefix="fetch-cross-", dir=ROOT / "tests") as directory:
            state = FetchState(pathlib.Path(directory) / "state.sqlite3")
            state.register_assets("etf", ["158000"])
            names = {"etf": {"158000": "港股通金融ETF鹏华"}}
            with (
                patch.object(fetch_state_module, "_should_filter_progress", return_value=False),
                patch.object(fetch_state_module, "_load_manifest_names_f", return_value=names),
            ):
                claimed = state.claim_due(
                    "etf",
                    symbols=["158000"],
                    limit=1,
                    target_date="2026-08-27",
                    owner="worker-a",
                )
            self.assertEqual([item["symbol"] for item in claimed], ["158000"])


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

    def test_advance_due_for_stale_only_wakes_lagging_success_rows(self) -> None:
        """收盘后补数：只强制"数据落后且还在睡"的成功标的，其余不动。"""
        with tempfile.TemporaryDirectory(prefix="fetch-postclose-", dir=ROOT / "tests") as directory:
            db_path = pathlib.Path(directory) / "state.sqlite3"
            state = FetchState(db_path)
            state.register_assets("stock", ["000001", "000002", "000003", "000004"])
            state.register_assets("etf", ["510300"])
            # 000001：落后且在睡（next_retry 在未来）→ 应被强制到期
            state.mark_success("stock", "000001", data_end_date="2026-09-03",
                               source="test", rows=10, refresh_interval_seconds=86400)
            # 000002：数据已是最新 → 不动
            state.mark_success("stock", "000002", data_end_date="2026-09-05",
                               source="test", rows=10, refresh_interval_seconds=86400)
            # 000003：落后但持有未过期租约（在途）→ 不动
            state.mark_success("stock", "000003", data_end_date="2026-09-03",
                               source="test", rows=10, refresh_interval_seconds=86400)
            with state._connect() as connection:
                # lease_until 与 next_retry_at 一样用 ISO 字符串比较，
                # 测试里必须写同格式，不能用 SQLite datetime('now')
                connection.execute(
                    "UPDATE assets SET lease_owner = 'w1', lease_until = ? "
                    "WHERE symbol = '000003'",
                    (iso_now(datetime.now(timezone.utc) + timedelta(minutes=10)),),
                )
            # 000004：失败状态 → 不动（走失败重试路径）
            state.mark_failure("stock", "000004", error="x", backoff_base_seconds=60,
                               backoff_max_seconds=60)
            # 510300：其他资产类型的落后标的 → 默认不在范围内
            state.mark_success("etf", "510300", data_end_date="2026-09-01",
                               source="test", rows=10, refresh_interval_seconds=86400)

            forced = state.advance_due_for_stale(before_date="2026-09-05", asset_types=("stock",))
            self.assertEqual(forced, 1)

            with state._connect() as connection:
                connection.row_factory = sqlite3.Row
                rows = {
                    row["symbol"]: dict(row)
                    for row in connection.execute(
                        "SELECT symbol, status, next_retry_at FROM assets"
                    )
                }
            self.assertIsNone(rows["000001"]["next_retry_at"])
            self.assertIsNotNone(rows["000002"]["next_retry_at"])
            self.assertIsNotNone(rows["000003"]["next_retry_at"])
            self.assertIsNotNone(rows["000004"]["next_retry_at"])
            self.assertIsNotNone(rows["510300"]["next_retry_at"])

            # 已到期的行不再重复计数
            self.assertEqual(
                state.advance_due_for_stale(before_date="2026-09-05", asset_types=("stock",)),
                0,
            )

            # 不限资产类型时，落后的 ETF 也会被唤醒
            self.assertEqual(state.advance_due_for_stale(before_date="2026-09-05"), 1)

    def test_advance_due_for_stale_empty_asset_types_is_noop(self) -> None:
        with tempfile.TemporaryDirectory(prefix="fetch-postclose2-", dir=ROOT / "tests") as directory:
            state = FetchState(pathlib.Path(directory) / "state.sqlite3")
            state.register_assets("stock", ["000001"])
            self.assertEqual(
                state.advance_due_for_stale(before_date="2026-09-05", asset_types=[]),
                0,
            )

    def _row(self, state: FetchState, symbol: str) -> dict:
        with state._connect() as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT symbol, status, data_end_date, last_success_date, "
                "next_retry_at, lease_until FROM assets WHERE symbol = ?",
                (symbol,),
            ).fetchone()
        return dict(row)

    def test_advance_due_for_stale_demotes_verified_through_so_same_day_reclaimable(self) -> None:
        """T16 (2026-10-12, hermes A2): 复刻"盘中 bar 被 T15 守卫丢弃"后的状态——
        mark_success(verified_through=D) 使 last_success=D 而 data_end 仍 D-1，
        claim_due(target=D) 被 MAX(...) < D 拒绝，当日完整 bar 只能等次日。
        advance_due_for_stale 必须把该行 last_success 回拨到 data_end 并清计时器，
        使 claim_due(target=D) 立即重新认领；不改 claim_due / mark_success 语义。
        """
        with tempfile.TemporaryDirectory(prefix="t16-verified-", dir=ROOT / "tests") as directory:
            state = FetchState(pathlib.Path(directory) / "state.sqlite3")
            state.register_assets("stock", ["600519"])
            state.mark_success(
                "stock", "600519",
                data_end_date="2026-10-09",
                verified_through="2026-10-12",
                source="test", rows=100,
                refresh_interval_seconds=86400,
            )
            before = self._row(state, "600519")
            self.assertEqual(before["data_end_date"], "2026-10-09")
            self.assertEqual(before["last_success_date"], "2026-10-12")
            self.assertIsNotNone(before["next_retry_at"])

            forced = state.advance_due_for_stale(
                before_date="2026-10-12", asset_types=("stock",)
            )
            self.assertEqual(forced, 1)

            after = self._row(state, "600519")
            self.assertIsNone(after["next_retry_at"])
            # 核心：last_success 回拨到 data_end，MAX < target 恢复
            self.assertEqual(after["last_success_date"], "2026-10-09")

            claimed = state.claim_due(
                "stock", symbols=["600519"], limit=5,
                target_date="2026-10-12", owner="w",
            )
            self.assertEqual([c["symbol"] for c in claimed], ["600519"])

            # 幂等：第二次不重复计数（无实际变化）
            self.assertEqual(
                state.advance_due_for_stale(
                    before_date="2026-10-12", asset_types=("stock",)
                ),
                0,
            )

    def test_advance_due_for_stale_includes_row_already_due_but_blocked_by_last_success(self) -> None:
        """T16: 计时器本就已到期、但只被 last_success>=target 挡住的行，
        旧谓词 (next_retry_at IS NOT NULL AND next_retry_at > now) 永远匹配不到，
        连强制都不发生；新语义必须计入并使其可认领。"""
        with tempfile.TemporaryDirectory(prefix="t16-dued-only-", dir=ROOT / "tests") as directory:
            state = FetchState(pathlib.Path(directory) / "state.sqlite3")
            state.register_assets("stock", ["000001"])
            state.mark_success(
                "stock", "000001",
                data_end_date="2026-10-09",
                verified_through="2026-10-12",
                source="test", rows=100,
                refresh_interval_seconds=86400,
            )
            past = iso_now(datetime.now(timezone.utc) - timedelta(hours=1))
            with state._connect() as connection:
                connection.execute(
                    "UPDATE assets SET next_retry_at = ? WHERE symbol = '000001'",
                    (past,),
                )
            self.assertIsNotNone(self._row(state, "000001")["next_retry_at"])

            forced = state.advance_due_for_stale(
                before_date="2026-10-12", asset_types=("stock",)
            )
            self.assertEqual(forced, 1)
            self.assertIsNone(self._row(state, "000001")["next_retry_at"])
            self.assertEqual(
                self._row(state, "000001")["last_success_date"], "2026-10-09"
            )
            claimed = state.claim_due(
                "stock", symbols=["000001"], limit=5,
                target_date="2026-10-12", owner="w",
            )
            self.assertEqual([c["symbol"] for c in claimed], ["000001"])

    def test_advance_due_for_stale_untouched_rows_and_count(self) -> None:
        """T16: data_end>=before、租约在途、failed/quarantined/inactive 行一律不动；
        返回值 = 实际被改动的行数，改完再调用返回 0。"""
        with tempfile.TemporaryDirectory(prefix="t16-untouched-", dir=ROOT / "tests") as directory:
            state = FetchState(pathlib.Path(directory) / "state.sqlite3")
            state.register_assets(
                "stock",
                ["000001", "000002", "000003", "000004", "000005", "600519"],
            )
            # 000001：data_end 已到 before（不动）
            state.mark_success(
                "stock", "000001",
                data_end_date="2026-10-12", source="test", rows=10,
                refresh_interval_seconds=86400,
            )
            # 000002：落后但租约在途（不动）
            state.mark_success(
                "stock", "000002",
                data_end_date="2026-10-09", verified_through="2026-10-12",
                source="test", rows=10, refresh_interval_seconds=86400,
            )
            with state._connect() as connection:
                connection.execute(
                    "UPDATE assets SET lease_owner = 'w1', lease_until = ? "
                    "WHERE symbol = '000002'",
                    (iso_now(datetime.now(timezone.utc) + timedelta(minutes=10)),),
                )
            # 000003：failed（不动）
            state.mark_failure(
                "stock", "000003", error="x",
                backoff_base_seconds=60, backoff_max_seconds=60,
            )
            # 000004：quarantined、000005：inactive（均不动）
            with state._connect() as connection:
                connection.execute(
                    "UPDATE assets SET status = 'quarantined' WHERE symbol = '000004'"
                )
                connection.execute(
                    "UPDATE assets SET status = 'inactive' WHERE symbol = '000005'"
                )
            # 600519：落后、租约释放、计时器在未来（应被强制）
            state.mark_success(
                "stock", "600519",
                data_end_date="2026-10-09", verified_through="2026-10-12",
                source="test", rows=10, refresh_interval_seconds=86400,
            )
            untouched = {
                s: self._row(state, s)
                for s in ("000001", "000002", "000003", "000004", "000005")
            }

            forced = state.advance_due_for_stale(
                before_date="2026-10-12", asset_types=("stock",)
            )
            self.assertEqual(forced, 1)
            for symbol, before_row in untouched.items():
                self.assertEqual(self._row(state, symbol), before_row, symbol)
            self.assertIsNone(self._row(state, "600519")["next_retry_at"])
            self.assertEqual(
                self._row(state, "600519")["last_success_date"], "2026-10-09"
            )
            # 幂等：没有实际改动的行不重复计数
            self.assertEqual(
                state.advance_due_for_stale(
                    before_date="2026-10-12", asset_types=("stock",)
                ),
                0,
            )


if __name__ == "__main__":
    unittest.main()
