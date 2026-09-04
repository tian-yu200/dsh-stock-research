from __future__ import annotations

from datetime import datetime, timezone
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import fetch_stock_data as worker
from data_source import NoDailyBarsError


class _FakeLimiter:
    def wait(self) -> None:
        return None


class _FetchStateSpy:
    def __init__(self) -> None:
        self.successes: list[tuple[str, dict]] = []
        self.failures: list[tuple[str, dict]] = []
        self.quarantines: list[tuple[str, dict]] = []

    def mark_success(self, kind: str, symbol: str, **kwargs) -> None:
        self.successes.append((symbol, kwargs))

    def mark_failure(self, kind: str, symbol: str, **kwargs) -> None:
        self.failures.append((symbol, kwargs))

    def mark_quarantined(self, kind: str, symbol: str, **kwargs) -> None:
        self.quarantines.append((symbol, kwargs))


class _LifecycleStateSpy:
    def __init__(self, candidates: list[str]) -> None:
        self.candidates = candidates
        self.applied: list[dict] = []

    def lifecycle_candidates(self, asset_type: str, symbols: list[str]) -> list[str]:
        return self.candidates

    def apply_lifecycle_snapshot(self, asset_type: str, snapshot: dict) -> dict[str, int]:
        self.applied.append(snapshot)
        inactivated = (
            len(snapshot.get("inactive_candidates", []))
            if snapshot.get("complete") is True
            else 0
        )
        return {"inactivated": inactivated, "revived": 0}


class _ClaimStateSpy:
    def __init__(self) -> None:
        self.requests: list[list[str]] = []

    def claim_due(self, kind: str, *, symbols: list[str], **kwargs) -> list[dict]:
        self.requests.append(list(symbols))
        return [{"symbol": symbols[0]}] if symbols else []


def _bars(date: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [date],
            "open": [1.0],
            "high": [1.1],
            "low": [0.9],
            "close": [1.05],
            "volume": [100],
            "amount": [100],
            "turnover": [1],
        }
    )


class StockWorkerTests(unittest.TestCase):
    def test_worker_does_not_reclaim_symbols_attempted_in_same_run(self) -> None:
        state = _ClaimStateSpy()
        attempted: set[str] = set()
        codes = ["920001", "920002"]

        first = worker._claim_unattempted(
            state,
            codes,
            attempted,
            limit=1,
            target_date="2026-09-02",
            lease_seconds=1800,
            owner="test-owner",
        )
        second = worker._claim_unattempted(
            state,
            codes,
            attempted,
            limit=1,
            target_date="2026-09-02",
            lease_seconds=1800,
            owner="test-owner",
        )
        exhausted = worker._claim_unattempted(
            state,
            codes,
            attempted,
            limit=1,
            target_date="2026-09-02",
            lease_seconds=1800,
            owner="test-owner",
        )

        self.assertEqual([item["symbol"] for item in first], ["920001"])
        self.assertEqual([item["symbol"] for item in second], ["920002"])
        self.assertEqual(exhausted, [])
        self.assertEqual(
            state.requests,
            [["920001", "920002"], ["920002"]],
        )


    def _fetch(
        self,
        path: pathlib.Path,
        state: _FetchStateSpy,
        *,
        response,
        target: str = "2026-08-27",
        patch_fetch: bool = True,
    ):
        fetch_patch = patch.object(
            worker,
            "fetch_a_stock_daily",
            side_effect=response if isinstance(response, BaseException) else None,
            return_value=None if isinstance(response, BaseException) else response,
        )
        with patch.object(worker, "_data_path", return_value=path):
            if patch_fetch:
                with fetch_patch:
                    return worker._fetch_one(
                        {"symbol": "000016", "data_end_date": None},
                        state=state,
                        limiter=_FakeLimiter(),
                        target=target,
                        retries=3,
                        backoff_base=0,
                        backoff_max=0,
                        lease_owner="test-owner",
                        deadline=None,
                    )
            return worker._fetch_one(
                {"symbol": "000016", "data_end_date": None},
                state=state,
                limiter=_FakeLimiter(),
                target=target,
                retries=3,
                backoff_base=0,
                backoff_max=0,
                lease_owner="test-owner",
                deadline=None,
            )

    def test_confirmed_empty_window_with_cache_marks_verified_target(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="stock-worker-", dir=ROOT / "tests"
        ) as directory:
            path = pathlib.Path(directory) / "000016.csv"
            _bars("2026-08-26").to_csv(path, index=False)
            state = _FetchStateSpy()
            result = self._fetch(
                path,
                state,
                response=NoDailyBarsError("000016", ["腾讯", "新浪"]),
            )
            self.assertEqual(worker.cached_end_date(path), "2026-08-26")

        self.assertEqual(result[0:2], ("000016", True))
        self.assertEqual(result[2]["source"], "verified-empty-window")
        self.assertEqual(result[2]["data_end_date"], "2026-08-26")
        self.assertEqual(state.successes[0][1]["data_end_date"], "2026-08-26")
        self.assertEqual(state.successes[0][1]["verified_through"], "2026-08-27")

    def test_confirmed_empty_window_without_cache_is_failure(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="stock-worker-", dir=ROOT / "tests"
        ) as directory:
            path = pathlib.Path(directory) / "000016.csv"
            state = _FetchStateSpy()
            result = self._fetch(
                path,
                state,
                response=NoDailyBarsError("000016", ["腾讯", "新浪"]),
            )

        self.assertEqual(result[0:2], ("000016", False))
        self.assertEqual(state.successes, [])
        self.assertEqual(state.failures[0][0], "000016")
        self.assertIn("NoDailyBarsError", state.failures[0][1]["error"])
        self.assertEqual(state.failures[0][1]["backoff_base_seconds"], 86400)
        self.assertEqual(state.failures[0][1]["backoff_max_seconds"], 86400)

    def test_nonempty_response_keeps_real_end_and_verifies_target(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="stock-worker-", dir=ROOT / "tests"
        ) as directory:
            path = pathlib.Path(directory) / "000016.csv"
            state = _FetchStateSpy()
            incoming = _bars("2026-08-26")
            incoming.attrs["akshare_source"] = "腾讯"
            result = self._fetch(path, state, response=incoming)

            self.assertEqual(worker.cached_end_date(path), "2026-08-26")

        self.assertEqual(result[0:2], ("000016", True))
        self.assertEqual(state.successes[0][1]["data_end_date"], "2026-08-26")
        self.assertEqual(state.successes[0][1]["verified_through"], "2026-08-27")

    def test_schema_error_is_quarantined_without_network_retries(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="stock-worker-", dir=ROOT / "tests"
        ) as directory:
            path = pathlib.Path(directory) / "000016.csv"
            state = _FetchStateSpy()
            from market_data import DailyBarSchemaError

            error = DailyBarSchemaError(
                "conflicting duplicate values for volume",
                failure_class="schema_conflict",
                field_anomalies={"volume": {"conflicting_rows": 1}},
            )
            with patch.object(worker, "fetch_a_stock_daily", side_effect=error) as fetch:
                result = self._fetch(path, state, response=error, patch_fetch=False)

        self.assertEqual(result[0:2], ("000016", False))
        self.assertEqual(fetch.call_count, 1)
        self.assertEqual(state.failures, [])
        self.assertEqual(len(state.quarantines), 1)
        self.assertEqual(
            state.quarantines[0][1]["failure_class"], "schema_conflict"
        )
        self.assertEqual(
            state.quarantines[0][1]["field_anomalies"],
            {"volume": {"conflicting_rows": 1}},
        )

    def test_cached_lifecycle_snapshot_defers_candidates_it_did_not_check(self) -> None:
        state = _LifecycleStateSpy(["000001", "000002"])
        snapshot = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "complete": True,
            "candidates": ["000001", "000003"],
            "active": [],
            "unresolved_candidates": [],
            "inactive_candidates": ["000001", "000003"],
            "sources": {"exchange": 5000, "eastmoney": 5000, "sina": 5000},
            "errors": {},
        }

        with patch.object(worker, "discover_stock_lifecycle") as discover:
            result = worker._reconcile_stock_lifecycle(
                state,
                ["000001", "000002"],
                {"stock_lifecycle": snapshot},
            )

        discover.assert_not_called()
        self.assertEqual(len(state.applied), 1)
        self.assertEqual(result["candidate_count"], 2)
        self.assertEqual(result["checked_count"], 1)
        self.assertEqual(result["deferred_count"], 1)
        self.assertTrue(result["cached"])
        self.assertEqual(state.applied[0]["candidates"], ["000001"])
        self.assertEqual(state.applied[0]["inactive_candidates"], ["000001"])

    def test_lifecycle_reconciliation_skips_catalogs_without_candidates(self) -> None:
        state = _LifecycleStateSpy([])

        with patch.object(worker, "discover_stock_lifecycle") as discover:
            result = worker._reconcile_stock_lifecycle(state, ["000001"], {})

        discover.assert_not_called()
        self.assertEqual(state.applied, [])
        self.assertEqual(result["candidate_count"], 0)
        self.assertEqual(result["checked_count"], 0)
        self.assertEqual(result["deferred_count"], 0)


if __name__ == "__main__":
    unittest.main()
