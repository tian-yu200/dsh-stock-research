from __future__ import annotations

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

    def mark_success(self, kind: str, symbol: str, **kwargs) -> None:
        self.successes.append((symbol, kwargs))

    def mark_failure(self, kind: str, symbol: str, **kwargs) -> None:
        self.failures.append((symbol, kwargs))


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
    def _fetch(
        self,
        path: pathlib.Path,
        state: _FetchStateSpy,
        *,
        response,
        target: str = "2026-08-27",
    ):
        fetch_patch = patch.object(
            worker,
            "fetch_a_stock_daily",
            side_effect=response if isinstance(response, BaseException) else None,
            return_value=None if isinstance(response, BaseException) else response,
        )
        with patch.object(worker, "_data_path", return_value=path), fetch_patch:
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


if __name__ == "__main__":
    unittest.main()
