from __future__ import annotations

import pathlib
import sys
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import data_source
from data_source import _BREAKER_STATE


def _bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2026-08-26", "2026-08-27"],
            "open": [1.0, 1.01],
            "high": [1.1, 1.11],
            "low": [0.9, 0.91],
            "close": [1.05, 1.06],
            "volume": [100, 110],
        }
    )


class SourceBreakerTests(unittest.TestCase):
    def setUp(self) -> None:
        _BREAKER_STATE.clear()

    def tearDown(self) -> None:
        _BREAKER_STATE.clear()

    def test_breaker_disables_source_after_consecutive_same_type_failures(self) -> None:
        calls = {"东财": 0, "腾讯": 0}

        def em(symbol: str, start: str, end: str) -> pd.DataFrame:
            calls["东财"] += 1
            raise AttributeError("'str' object has no attribute 'date'")

        def tx(symbol: str, start: str, end: str) -> pd.DataFrame:
            calls["腾讯"] += 1
            return _bars()

        sources = [("东财", em), ("腾讯", tx)]
        with patch.object(data_source, "_SOURCES", sources), patch.object(
            data_source, "_SOURCE_BREAKER_THRESHOLD", 3
        ):
            for _ in range(4):
                data_source.fetch_a_stock_daily("000001")

        self.assertEqual(calls["东财"], 3)
        self.assertEqual(calls["腾讯"], 4)
        self.assertFalse(data_source._breaker_allows("东财"))
        self.assertTrue(data_source._breaker_allows("腾讯"))

    def test_success_resets_consecutive_counter(self) -> None:
        flip = {"up": False}

        def em(symbol: str, start: str, end: str) -> pd.DataFrame:
            if not flip["up"]:
                raise ConnectionError("boom")
            return _bars()

        with patch.object(data_source, "_SOURCES", [("东财", em)]):
            for _ in range(2):
                with self.assertRaises(RuntimeError):
                    data_source.fetch_a_stock_daily("000001")
            self.assertEqual(_BREAKER_STATE["东财"]["count"], 2)
            flip["up"] = True
            data_source.fetch_a_stock_daily("000001")
            self.assertNotIn("东财", _BREAKER_STATE)

    def test_breaker_resets_on_error_type_change(self) -> None:
        flip = {"kind": 0}

        def em(symbol: str, start: str, end: str) -> pd.DataFrame:
            if flip["kind"] == 0:
                raise ConnectionError("boom")
            raise TimeoutError("slow")

        with patch.object(data_source, "_SOURCES", [("东财", em)]), patch.object(
            data_source, "_SOURCE_BREAKER_THRESHOLD", 3
        ):
            for _ in range(2):
                with self.assertRaises(RuntimeError):
                    data_source.fetch_a_stock_daily("000001")
            self.assertEqual(_BREAKER_STATE["东财"]["count"], 2)
            flip["kind"] = 1
            with self.assertRaises(RuntimeError):
                data_source.fetch_a_stock_daily("000001")
            self.assertEqual(_BREAKER_STATE["东财"]["count"], 1)
            self.assertFalse(_BREAKER_STATE["东财"]["disabled"])

    def test_all_sources_failed_carries_failure_class_and_deterministic_flag(self) -> None:
        def em(symbol: str, start: str, end: str) -> pd.DataFrame:
            raise ConnectionError("boom")

        def tx(symbol: str, start: str, end: str) -> pd.DataFrame:
            raise TimeoutError("slow")

        def sina(symbol: str, start: str, end: str) -> pd.DataFrame:
            raise AttributeError("'str' object has no attribute 'date'")

        sources = [("东财", em), ("腾讯", tx), ("新浪", sina)]
        with patch.object(data_source, "_SOURCES", sources):
            with self.assertRaises(RuntimeError) as ctx:
                data_source.fetch_a_stock_daily("000001")

        self.assertEqual(ctx.exception.failure_class, "all_sources_failed")
        self.assertTrue(getattr(ctx.exception, "deterministic", False))
        self.assertIn("'str' object has no attribute 'date'", str(ctx.exception))

    def test_all_sources_failed_stays_transient_for_network_errors(self) -> None:
        def em(symbol: str, start: str, end: str) -> pd.DataFrame:
            raise ConnectionError("boom")

        def tx(symbol: str, start: str, end: str) -> pd.DataFrame:
            raise TimeoutError("slow")

        sources = [("东财", em), ("腾讯", tx)]
        with patch.object(data_source, "_SOURCES", sources):
            with self.assertRaises(RuntimeError) as ctx:
                data_source.fetch_a_stock_daily("000001")

        self.assertEqual(ctx.exception.failure_class, "all_sources_failed")
        self.assertFalse(getattr(ctx.exception, "deterministic", False))


if __name__ == "__main__":
    unittest.main()
