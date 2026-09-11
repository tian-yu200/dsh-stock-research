from __future__ import annotations

import pathlib
import sys
import unittest
import urllib.error
from unittest.mock import patch

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import data_source


class SourcesForTests(unittest.TestCase):
    def test_ths_appended_only_for_920(self) -> None:
        self.assertEqual(len(data_source._sources_for("000001")), 3)
        self.assertEqual(len(data_source._sources_for("600519")), 3)
        chain = data_source._sources_for("920000")
        self.assertEqual(len(chain), 4)
        self.assertEqual(chain[-1][0], "同花顺")

    def test_rebase_pinned_stocks_match_tencent_recheck(self) -> None:
        # 2026-09-10 腾讯复查：这 5 只仍 serve 新基座，需抓取侧钉住复核。
        self.assertEqual(
            data_source.REBASE_PINNED_STOCKS,
            frozenset({"600094", "600479", "600052", "601225", "000002"}),
        )


class ThsSourceTests(unittest.TestCase):
    def test_ths_parses_payload_and_filters_window(self) -> None:
        lines = [
            "20260901,10,11,9,10.5,1000,10500.00,1.5,,,0",
            "20260902,10.5,10.8,10.2,10.6,1100,,2.0,,,0",
            "20260903,10.6,10.9,10.4,10.7,1200,12840.00,,,0",
            "not-a-row",
        ]

        def fake_lines(path: str) -> list[str] | None:
            return lines if path.endswith("2026.js") or path.endswith("last.js") else None

        with patch.object(data_source, "_ths_get_lines", side_effect=fake_lines):
            df = data_source._ths("920000", "20260901", "20991231")

        self.assertEqual(len(df), 3)
        self.assertEqual(str(df["date"].iloc[0].date()), "2026-09-01")
        self.assertEqual(str(df["date"].iloc[-1].date()), "2026-09-03")
        self.assertTrue(pd.isna(df["amount"].iloc[1]))
        self.assertEqual(float(df["volume"].iloc[0]), 1000.0)
        self.assertEqual(df.attrs["akshare_source"], "同花顺")
        self.assertEqual(df.attrs["adjustment"], "hfq")

    def test_ths_raises_empty_when_no_payload(self) -> None:
        with patch.object(data_source, "_ths_get_lines", return_value=None):
            with self.assertRaises(data_source.EmptyDailyBarResponse):
                data_source._ths("920000", "20200101", "20991231")

    def test_ths_get_lines_404_returns_none(self) -> None:
        def fake_urlopen(req, timeout):
            raise urllib.error.HTTPError(req.full_url, 404, "nf", None, None)

        with patch.object(data_source.urllib.request, "urlopen", fake_urlopen):
            self.assertIsNone(data_source._ths_get_lines("hs_920000/02/1999.js"))

    def test_ths_get_lines_retries_5xx(self) -> None:
        calls = {"n": 0}

        class _Resp:
            def read(self) -> bytes:
                return b'"data":"20260904,1,2,1,2,100,200.00,1,,,0";'

        def fake_urlopen(req, timeout):
            calls["n"] += 1
            if calls["n"] < 3:
                raise urllib.error.HTTPError(req.full_url, 504, "gw", None, None)
            return _Resp()

        with patch.object(
            data_source.urllib.request, "urlopen", fake_urlopen
        ), patch.object(data_source.time, "sleep"):
            lines = data_source._ths_get_lines("hs_920000/02/last.js")

        self.assertEqual(calls["n"], 3)
        self.assertEqual(lines[0].split(",")[0], "20260904")


if __name__ == "__main__":
    unittest.main()
