from __future__ import annotations

import pathlib
import sys
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import data_source


def bars(prices):
    return pd.DataFrame({
        "date": pd.date_range("2026-07-13", periods=len(prices)),
        "open": prices, "high": prices, "low": prices, "close": prices,
        "volume": [1000] * len(prices), "amount": [1000 * p for p in prices],
    })


class EtfAdjustmentProvenanceTests(unittest.TestCase):
    def test_sina_raw_is_replaced_by_explicit_qfq_series(self):
        with patch.object(data_source.ak, "fund_etf_hist_em", side_effect=ConnectionError), \
             patch.object(data_source.ak, "fund_etf_hist_sina", return_value=bars([1.8, 1.8, .6])), \
             patch.object(data_source.ak, "stock_zh_a_hist_tx", return_value=bars([.6, .6, .6])) as tx:
            result = data_source.fetch_etf_daily("159320", "20260713", "20260715")
        self.assertEqual(result["close"].tolist(), [.6, .6, .6])
        self.assertEqual(tx.call_args.kwargs["adjust"], "qfq")
        self.assertEqual(tx.call_args.kwargs["symbol"], "sz159320")
        self.assertEqual(result.attrs["adjustment"], "qfq")
        self.assertEqual(result.attrs["akshare_interface"], "stock_zh_a_hist_tx")

    def test_unverified_sina_is_never_labelled_qfq(self):
        with patch.object(data_source.ak, "fund_etf_hist_em", side_effect=ConnectionError), \
             patch.object(data_source.ak, "fund_etf_hist_sina", return_value=bars([1.8, 1.8, .6])), \
             patch.object(data_source.ak, "stock_zh_a_hist_tx", side_effect=ConnectionError):
            result = data_source.fetch_etf_daily("159320", "20260713", "20260715")
        self.assertEqual(result.attrs["adjustment"], "unadjusted")
        self.assertEqual(result.attrs["akshare_interface"], "fund_etf_hist_sina")

    def test_empty_adjusted_response_does_not_validate_raw_sina(self):
        with patch.object(data_source.ak, "fund_etf_hist_em", side_effect=ConnectionError), \
             patch.object(data_source.ak, "fund_etf_hist_sina", return_value=bars([1.8, 1.8, .6])), \
             patch.object(data_source.ak, "stock_zh_a_hist_tx", return_value=pd.DataFrame()):
            result = data_source.fetch_etf_daily("159320", "20260713", "20260715")
        self.assertEqual(result.attrs["adjustment"], "unadjusted")

    def test_eastmoney_qfq_does_not_call_fallback(self):
        with patch.object(data_source.ak, "fund_etf_hist_em", return_value=bars([.6, .6, .6])), \
             patch.object(data_source.ak, "stock_zh_a_hist_tx") as tx:
            result = data_source.fetch_etf_daily("159320", "20260713", "20260715")
        tx.assert_not_called()
        self.assertEqual(result.attrs["adjustment"], "qfq")


if __name__ == "__main__":
    unittest.main()
