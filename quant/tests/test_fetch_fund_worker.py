from __future__ import annotations

import pathlib
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "scripts"))

import fetch_fund_data as worker
from fetch_utils import FundNavValidationError, merge_otc_frames


class _FakeLimiter:
    def wait(self) -> None:
        return None


class _FetchStateSpy:
    def __init__(self) -> None:
        self.claim_limits: list[int] = []
        self.claim_kinds: list[str] = []
        self.claim_count = 0
        self.failures: list[str] = []
        self.quarantines: list[str] = []
        self.registered: list[tuple[str, list[str], dict[str, str | None]]] = []

    def reclaim_orphaned_leases(self) -> int:
        return 0

    def assets_needing_cache_backfill(self, kind: str, symbols):
        return list(symbols)

    def register_assets(self, kind: str, symbols, *, data_end_dates=None) -> int:
        values = list(symbols)
        self.registered.append((kind, values, dict(data_end_dates or {})))
        return len(values)

    def claim_due(self, kind: str, *, limit: int, **kwargs):
        self.claim_limits.append(limit)
        self.claim_kinds.append(kind)
        self.claim_count += 1
        if self.claim_count == 1:
            return [
                {"symbol": f"{index:06d}", "data_end_date": None}
                for index in range(limit)
            ]
        return []

    def progress(self, *args, **kwargs):
        return {
            "total": 10,
            "completed": 3,
            "pending": 7,
            "failed": 0,
            "stale": 7,
            "coverage": 0.3,
            "target_date": "2026-08-26",
            "by_asset_type": {},
            "status": "partial",
        }

    def mark_failure(self, kind: str, symbol: str, **kwargs) -> None:
        self.failures.append(symbol)

    def mark_quarantined(self, kind: str, symbol: str, **kwargs) -> None:
        self.quarantines.append(symbol)


class FundWorkerTests(unittest.TestCase):
    def test_register_pool_reads_only_assets_missing_persisted_dates(self) -> None:
        state = _FetchStateSpy()
        state.assets_needing_cache_backfill = lambda kind, symbols: ["000002"]
        with (
            tempfile.TemporaryDirectory(prefix="fund-register-", dir=ROOT / "tests") as directory,
            patch.object(worker, "_path", return_value=pathlib.Path(directory) / "fund.csv"),
            patch.object(worker, "cached_end_date", return_value="2026-08-26") as cached,
        ):
            path = pathlib.Path(directory) / "fund.csv"
            path.touch()
            worker._register_pool(state, "otc_fund", ["000001", "000002"])

        cached.assert_called_once_with(path)
        self.assertEqual(
            state.registered,
            [("otc_fund", ["000002"], {"000002": "2026-08-26"})],
        )

    def test_main_claims_only_one_worker_wave_at_a_time(self) -> None:
        state = _FetchStateSpy()
        manifest = {
            "complete": True,
            "etf_universe_complete": True,
            "otc_fund_universe_complete": True,
            "etf": [],
            "otc_fund": [],
        }
        with (
            patch.object(worker, "FetchState", return_value=state),
            patch.object(worker, "_manifest_complete", return_value=True),
            patch.object(
                worker,
                "discover_pools",
                return_value=([], [f"{index:06d}" for index in range(10)]),
            ),
            patch.object(worker, "cached_end_date", return_value=None),
            patch.object(worker, "install_request_timeout"),
            patch.object(worker, "load_universe_manifest", return_value=manifest),
            patch.object(
                worker,
                "_fetch_one",
                side_effect=lambda kind, item, **kwargs: (
                    kind,
                    item["symbol"],
                    True,
                    {"interface": "test"},
                    None,
                ),
            ),
        ):
            exit_code = worker.main(
                [
                    "--asset-type",
                    "otc_fund",
                    "--workers",
                    "3",
                    "--batch-size",
                    "200",
                    "--time-budget",
                    "10",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertTrue(state.claim_limits)
        self.assertLessEqual(max(state.claim_limits), 3)
        self.assertEqual(set(state.claim_kinds), {"otc_fund"})

    def test_fetch_one_does_not_retry_after_deadline(self) -> None:
        state = _FetchStateSpy()
        with (
            patch.object(worker, "fetch_otc_fund_nav", side_effect=TimeoutError("slow")) as fetch,
            patch.object(worker.time, "monotonic", side_effect=[99.0, 101.0]),
        ):
            result = worker._fetch_one(
                "otc_fund",
                {"symbol": "009216", "data_end_date": None},
                state=state,
                limiter=_FakeLimiter(),
                target="2026-08-26",
                retries=3,
                backoff_base=0,
                backoff_max=0,
                lease_owner="test-owner",
                deadline=100.0,
            )

        self.assertFalse(result[2])
        self.assertEqual(fetch.call_count, 1)
        self.assertEqual(state.failures, ["009216"])

    def test_invalid_nav_response_is_not_retried_and_is_quarantined(self) -> None:
        state = _FetchStateSpy()
        invalid = pd.DataFrame(
            {
                "净值日期": ["2026-08-25", "2026-08-26"],
                "单位净值": ["1.0", "--"],
            }
        )
        with tempfile.TemporaryDirectory(prefix="fund-worker-", dir=ROOT / "tests") as directory:
            path = pathlib.Path(directory) / "fund_009216.csv"
            with (
                patch.object(worker, "_path", return_value=path),
                patch.object(worker, "fetch_otc_fund_nav", return_value=invalid) as fetch,
            ):
                result = worker._fetch_one(
                    "otc_fund",
                    {"symbol": "009216", "data_end_date": None},
                    state=state,
                    limiter=_FakeLimiter(),
                    target="2026-08-26",
                    retries=3,
                    backoff_base=0,
                    backoff_max=0,
                    lease_owner="test-owner",
                    deadline=None,
                )

        self.assertFalse(result[2])
        self.assertEqual(fetch.call_count, 1)
        self.assertEqual(state.quarantines, ["009216"])
        self.assertEqual(state.failures, [])
        self.assertIn("invalid_navs=1", result[4] or "")


class FundNavValidationTests(unittest.TestCase):
    def test_validation_error_contains_row_diagnostics(self) -> None:
        incoming = pd.DataFrame(
            {
                "净值日期": ["bad-date", "2026-08-26"],
                "单位净值": ["1.0", "--"],
            }
        )
        with tempfile.TemporaryDirectory(prefix="fund-nav-", dir=ROOT / "tests") as directory:
            path = pathlib.Path(directory) / "fund.csv"
            with self.assertRaises(FundNavValidationError) as raised:
                merge_otc_frames(path, incoming, source="test")

        self.assertEqual(raised.exception.total_rows, 2)
        self.assertEqual(raised.exception.invalid_dates, 1)
        self.assertEqual(raised.exception.invalid_navs, 1)
        self.assertTrue(raised.exception.samples)


if __name__ == "__main__":
    unittest.main()
