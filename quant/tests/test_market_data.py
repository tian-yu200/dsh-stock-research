from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from data_source import NoDailyBarsError, fetch_a_stock_daily, fetch_etf_daily
from fetch_fund_data import ETF_POOL, OTC_POOL, discover_pools
from fetch_stock_data import DEFAULT_POOL, discover_pool, is_allowed
from market_data import (
    DAILY_COLUMNS,
    DailyBarSchemaError,
    EmptyDailyBarResponse,
    normalize_daily_bars,
)
from universe import (
    discover_etf_universe,
    discover_otc_fund_universe,
    discover_stock_lifecycle,
    discover_stock_universe,
    discover_universe,
    infer_exchange,
)

class NormalizeDailyBarsTests(unittest.TestCase):
    def test_normalizes_chinese_columns_sorts_and_keeps_last_duplicate(self) -> None:
        frame = pd.DataFrame(
            {
                "日期": ["2024-01-03", "2024-01-02", "2024-01-03"],
                "开盘": [3, 1, 4],
                "最高": [4, 2, 5],
                "最低": [2, 0.5, 3],
                "收盘": [3.5, 1.5, 4.5],
                "成交量": [30, 10, 40],
            }
        )

        result = normalize_daily_bars(frame, source="test")

        self.assertEqual(list(result.columns), list(DAILY_COLUMNS))
        self.assertEqual(
            result["date"].dt.strftime("%Y-%m-%d").tolist(),
            ["2024-01-02", "2024-01-03"],
        )
        self.assertEqual(result.loc[1, "close"], 4.5)
        self.assertTrue(result["amount"].isna().all())
        self.assertTrue(result["turnover"].isna().all())

    def test_accepts_named_date_index(self) -> None:
        frame = pd.DataFrame(
            {
                "open": [1],
                "high": [2],
                "low": [0.5],
                "close": [1.5],
                "volume": [100],
            },
            index=pd.DatetimeIndex(["2024-01-02"], name="date"),
        )

        result = normalize_daily_bars(frame)

        self.assertEqual(result.loc[0, "date"], pd.Timestamp("2024-01-02"))

    def test_rejects_missing_required_fields(self) -> None:
        frame = pd.DataFrame({"date": ["2024-01-02"], "close": [1]})

        with self.assertRaisesRegex(ValueError, "missing required columns"):
            normalize_daily_bars(frame, source="broken")

    def test_drops_nonpositive_price_rows_and_records_anomaly(self) -> None:
        frame = pd.DataFrame(
            {
                "date": ["2019-05-22", "2019-05-23", "2019-05-24"],
                "open": [1.0, 0.0, 1.1],
                "high": [1.2, 0.0, 1.3],
                "low": [0.9, 0.0, 1.0],
                "close": [1.1, 0.0, 1.2],
                "volume": [100, 200, 300],
            }
        )

        result = normalize_daily_bars(frame, source="test")

        self.assertEqual(result["date"].dt.strftime("%Y-%m-%d").tolist(), ["2019-05-22", "2019-05-24"])
        self.assertEqual(result.attrs["field_anomalies"]["price_nonpositive"], {"dropped_rows": 1})

    def test_all_nonpositive_prices_raise_empty_response(self) -> None:
        frame = pd.DataFrame(
            {
                "date": ["2019-05-23", "2019-07-12"],
                "open": [0.0, 0.0],
                "high": [0.0, 0.0],
                "low": [0.0, 0.0],
                "close": [0.0, 0.0],
                "volume": [100, 200],
            }
        )

        with self.assertRaises(EmptyDailyBarResponse):
            normalize_daily_bars(frame, source="test")

    def test_merges_consistent_duplicate_alias_columns(self) -> None:
        frame = pd.DataFrame(
            {
                "日期": ["2024-01-02", "2024-01-03"],
                "date": ["2024-01-02", "2024-01-03"],
                "开盘": [1.0, 2.0],
                "open": [1.0, 2.0],
                "最高": [2.0, 3.0],
                "high": [2.0, 3.0],
                "最低": [0.5, 1.5],
                "low": [0.5, 1.5],
                "收盘": [1.5, 2.5],
                "close": [1.5, 2.5],
                "成交量": [100, 200],
                "volume": [100, 200],
            }
        )

        result = normalize_daily_bars(frame, source="duplicate-aliases")

        self.assertEqual(len(result), 2)
        self.assertEqual(result["close"].tolist(), [1.5, 2.5])

    def test_fills_empty_duplicate_column_from_value_column(self) -> None:
        frame = pd.DataFrame(
            {
                "date": ["2024-01-02"],
                "open": [1.0],
                "开盘": [None],
                "high": [2.0],
                "low": [0.5],
                "close": [1.5],
                "volume": [100],
            }
        )

        result = normalize_daily_bars(frame, source="duplicate-empty")

        self.assertEqual(result.loc[0, "open"], 1.0)

    def test_rejects_conflicting_duplicate_required_column_structurally(self) -> None:
        frame = pd.DataFrame(
            {
                "date": ["2024-01-02"],
                "open": [1.0],
                "high": [2.0],
                "low": [0.5],
                "close": [1.5],
                "volume": [100],
                "成交量": [101],
            }
        )

        with self.assertRaises(DailyBarSchemaError) as raised:
            normalize_daily_bars(frame, source="required-conflict")

        self.assertEqual(raised.exception.failure_class, "schema_conflict")
        self.assertIn("volume", raised.exception.field_anomalies)

    def test_accepts_matching_date_index_and_column_but_rejects_conflict(self) -> None:
        matching = pd.DataFrame(
            {
                "date": ["2024-01-02"],
                "open": [1.0],
                "high": [2.0],
                "low": [0.5],
                "close": [1.5],
                "volume": [100],
            },
            index=pd.DatetimeIndex(["2024-01-02"], name="date"),
        )

        result = normalize_daily_bars(matching, source="matching-date")

        self.assertEqual(result["date"].tolist(), [pd.Timestamp("2024-01-02")])

        conflicting = matching.copy()
        conflicting["date"] = ["2024-01-03"]

        with self.assertRaises(DailyBarSchemaError) as raised:
            normalize_daily_bars(conflicting, source="conflicting-date")

        self.assertEqual(raised.exception.failure_class, "schema_conflict")
        self.assertIn("date", raised.exception.field_anomalies)

    def test_records_optional_conflict_and_prefers_more_valid_values(self) -> None:
        frame = pd.DataFrame(
            {
                "date": ["2024-01-02", "2024-01-03"],
                "open": [1.0, 2.0],
                "high": [2.0, 3.0],
                "low": [0.5, 1.5],
                "close": [1.5, 2.5],
                "volume": [100, 200],
                "amount": [10.0, None],
                "成交额": [11.0, 20.0],
            }
        )

        result = normalize_daily_bars(frame, source="optional-conflict")

        self.assertEqual(result["amount"].tolist(), [11.0, 20.0])
        self.assertEqual(result.attrs["field_anomalies"]["amount"]["conflicting_rows"], 1)
    def test_rejects_empty_daily_bar_response_with_structured_error(self) -> None:
        with self.assertRaises(EmptyDailyBarResponse):
            normalize_daily_bars(pd.DataFrame(), source="empty-source")

    def test_classifies_two_independent_empty_sources(self) -> None:
        empty = EmptyDailyBarResponse("empty")
        with patch.object(
            __import__("data_source"),
            "_SOURCES",
            [("source-a", Mock(side_effect=empty)), ("source-b", Mock(side_effect=empty))],
        ):
            with self.assertRaises(NoDailyBarsError) as raised:
                fetch_a_stock_daily("000016", start="20260827", end="20260827")

        self.assertEqual(raised.exception.symbol, "000016")
        self.assertEqual(raised.exception.sources, ("source-a", "source-b"))

    def test_schema_failure_falls_back_to_next_source(self) -> None:
        module = __import__("data_source")
        schema_error = DailyBarSchemaError(
            "duplicate canonical columns",
            failure_class="schema_conflict",
            field_anomalies={"volume": {"conflicting_rows": 1}},
        )
        fallback = pd.DataFrame(
            {
                "date": ["2024-01-02"],
                "open": [1.0],
                "high": [2.0],
                "low": [0.5],
                "close": [1.5],
                "volume": [100],
            }
        )
        first = Mock(side_effect=schema_error)
        second = Mock(return_value=fallback)
        with patch.object(module, "_SOURCES", [("bad", first), ("backup", second)]):
            result = fetch_a_stock_daily("000016")
        self.assertIs(result, fallback)
        self.assertEqual(first.call_count, 1)
        self.assertEqual(second.call_count, 1)
        self.assertEqual(result.attrs["akshare_source"], "backup")

    def test_schema_failure_is_preserved_when_no_source_has_bars(self) -> None:
        module = __import__("data_source")
        schema_error = DailyBarSchemaError(
            "duplicate canonical columns",
            failure_class="schema_conflict",
            field_anomalies={"volume": {"conflicting_rows": 1}},
        )
        empty = EmptyDailyBarResponse("empty")
        with patch.object(
            module,
            "_SOURCES",
            [("bad", Mock(side_effect=schema_error)),
             ("empty-a", Mock(side_effect=empty)),
             ("empty-b", Mock(side_effect=empty))],
        ):
            with self.assertRaises(NoDailyBarsError):
                fetch_a_stock_daily("000016")
    def test_etf_fallback_drops_source_specific_columns_and_filters_dates(self) -> None:
        fallback = pd.DataFrame(
            {
                "open": [1, 2, 3],
                "high": [2, 3, 4],
                "low": [0.5, 1.5, 2.5],
                "close": [1.5, 2.5, 3.5],
                "volume": [100, 200, 300],
                "index": pd.to_datetime(
                    ["2024-01-01", "2024-01-02", "2024-01-03"]
                ),
                "postVol": [0, 0, 0],
                "postAmt": [0, 0, 0],
            }
        ).set_index("index")
        fallback.index.name = "date"

        with patch(
            "data_source.ak.fund_etf_hist_em", side_effect=RuntimeError("offline")
        ), patch("data_source.ak.fund_etf_hist_sina", return_value=fallback):
            result = fetch_etf_daily(
                "510300", start="20240102", end="20240102"
            )

        self.assertEqual(list(result.columns), list(DAILY_COLUMNS))
        self.assertEqual(len(result), 1)
        self.assertEqual(result.loc[0, "date"], pd.Timestamp("2024-01-02"))


class ResearchPoolDiscoveryTests(unittest.TestCase):
    def test_stock_pool_accepts_new_shenzhen_prefix(self) -> None:
        self.assertTrue(is_allowed("302132"))
        self.assertFalse(is_allowed("30213"))

    def test_stock_lifecycle_requires_all_catalogs_before_marking_inactive(self) -> None:
        class FakeAk:
            @staticmethod
            def stock_info_sh_name_code(**_kwargs) -> pd.DataFrame:
                return pd.DataFrame({"证券代码": ["600519"], "证券简称": ["A"]})

            @staticmethod
            def stock_info_sz_name_code(**_kwargs) -> pd.DataFrame:
                return pd.DataFrame({"A股代码": ["000001"], "A股简称": ["B"]})

            @staticmethod
            def stock_info_bj_name_code() -> pd.DataFrame:
                return pd.DataFrame({"证券代码": ["830001"], "证券简称": ["C"]})

            @staticmethod
            def stock_zh_a_spot_em() -> pd.DataFrame:
                return pd.DataFrame({"代码": ["000002"], "名称": ["D"]})

            @staticmethod
            def stock_zh_a_spot() -> pd.DataFrame:
                return pd.DataFrame({"代码": ["600519"], "名称": ["A"]})

        snapshot = discover_stock_lifecycle(
            ["000001", "000002", "000999"],
            FakeAk,
        )

        self.assertTrue(snapshot["complete"])
        self.assertEqual(snapshot["active"], ["000001"])
        self.assertEqual(snapshot["unresolved_candidates"], ["000002"])
        self.assertEqual(snapshot["inactive_candidates"], ["000999"])

    def test_stock_lifecycle_never_marks_inactive_when_a_catalog_fails(self) -> None:
        class IncompleteAk:
            @staticmethod
            def stock_info_sh_name_code(**_kwargs) -> pd.DataFrame:
                return pd.DataFrame({"证券代码": ["600519"]})

            @staticmethod
            def stock_info_sz_name_code(**_kwargs) -> pd.DataFrame:
                raise RuntimeError("exchange unavailable")

            @staticmethod
            def stock_info_bj_name_code() -> pd.DataFrame:
                return pd.DataFrame({"证券代码": ["830001"]})

            @staticmethod
            def stock_zh_a_spot_em() -> pd.DataFrame:
                return pd.DataFrame({"代码": ["000002"]})

            @staticmethod
            def stock_zh_a_spot() -> pd.DataFrame:
                return pd.DataFrame({"代码": ["600519"]})

        snapshot = discover_stock_lifecycle(["000999"], IncompleteAk)

        self.assertFalse(snapshot["complete"])
        self.assertEqual(snapshot["inactive_candidates"], [])
        self.assertEqual(snapshot["unresolved_candidates"], ["000999"])

    def test_stock_pool_discovers_valid_existing_files(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stocks-", dir=ROOT / "tests") as directory:
            data_dir = pathlib.Path(directory)
            for name in ("600519.csv", "000001.csv", "688001.csv", "notes.csv"):
                (data_dir / name).touch()

            self.assertEqual(discover_pool(data_dir), ["000001", "600519", "688001"])

    def test_stock_pool_uses_bootstrap_only_when_directory_is_empty(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stocks-", dir=ROOT / "tests") as directory:
            self.assertEqual(discover_pool(pathlib.Path(directory)), DEFAULT_POOL)

    def test_complete_universe_discovery_includes_star_and_bse(self) -> None:
        class FakeAk:
            @staticmethod
            def stock_info_a_code_name() -> pd.DataFrame:
                return pd.DataFrame(
                    {"code": ["600519", "688001", "830001"], "name": ["A", "B", "C"]}
                )

            @staticmethod
            def fund_etf_spot_em() -> pd.DataFrame:
                return pd.DataFrame(
                    {"代码": ["510300", "159915"], "名称": ["ETF A", "ETF B"]}
                )

            @staticmethod
            def fund_open_fund_daily_em() -> pd.DataFrame:
                return pd.DataFrame(
                    {"基金代码": ["110020", "001594", "510300"], "基金简称": ["A", "B", "ETF"]}
                )

        self.assertEqual(
            [item["symbol"] for item in discover_stock_universe(FakeAk)],
            ["600519", "688001", "830001"],
        )
        self.assertEqual(
            [item["symbol"] for item in discover_etf_universe(FakeAk)],
            ["159915", "510300"],
        )
        self.assertEqual(
            [item["symbol"] for item in discover_otc_fund_universe(
                FakeAk,
                exclude_symbols={"510300"},
            )],
            ["001594", "110020"],
        )
        snapshot = discover_universe(FakeAk)
        self.assertTrue(snapshot["complete"])
        self.assertTrue(snapshot["stock_universe_complete"])
        self.assertTrue(snapshot["etf_universe_complete"])
        self.assertTrue(snapshot["otc_fund_universe_complete"])
        self.assertEqual(infer_exchange("830001"), "BJ")

    def test_fund_pools_discover_etf_and_otc_files_independently(self) -> None:
        with tempfile.TemporaryDirectory(prefix="funds-", dir=ROOT / "tests") as directory:
            data_dir = pathlib.Path(directory)
            (data_dir / "etf_510300.csv").touch()
            (data_dir / "fund_110020.csv").touch()
            (data_dir / "etf_bad.csv").touch()

            etfs, funds = discover_pools(data_dir)

            self.assertEqual(etfs, ["510300"])
            self.assertEqual(funds, ["110020"])

    def test_fund_pools_use_bootstrap_when_no_valid_files_exist(self) -> None:
        with tempfile.TemporaryDirectory(prefix="funds-", dir=ROOT / "tests") as directory:
            etfs, funds = discover_pools(pathlib.Path(directory))

            self.assertEqual(etfs, ETF_POOL)
            self.assertEqual(funds, OTC_POOL)



class PriceGuardsTests(unittest.TestCase):
    def _clean_frame(self, n=70, price=15.0):
        from price_guards import validate_daily_bars
        dates = pd.bdate_range("2026-01-05", periods=n)
        closes = [price + 0.2 * ((i % 7) - 3) for i in range(n)]
        vols = [6000000 + 100000 * (i % 5) for i in range(n)]
        return pd.DataFrame({"date": [d.strftime("%Y-%m-%d") for d in dates], "open": closes, "high": [c + 0.2 for c in closes], "low": [c - 0.2 for c in closes], "close": closes, "volume": vols, "amount": [c * v for c, v in zip(closes, vols)], "turnover": [0.007] * n}), validate_daily_bars
    def test_clean_history_has_zero_quarantine(self):
        frame, validate = self._clean_frame()
        clean, bad, reasons = validate(frame, source="unit", symbol="TEST01")
        self.assertEqual(len(bad), 0)
        self.assertEqual(len(clean), len(frame))
    def test_scaled_segment_is_quarantined_with_chain(self):
        frame, validate = self._clean_frame()
        dirty = frame.copy()
        for i in (68, 69):
            dirty.at[i, "open"] = 39.29
            dirty.at[i, "high"] = 40.14
            dirty.at[i, "low"] = 38.96
            dirty.at[i, "close"] = 39.10
        clean, bad, reasons = validate(dirty, source="unit", symbol="TEST01")
        self.assertEqual(len(bad), 2)
        self.assertEqual([r["reasons"] for r in reasons], [["gap_up_scale_break"]] * 2)
        self.assertEqual(len(clean), len(frame) - 2)
    def test_ex_division_row_is_kept(self):
        frame, validate = self._clean_frame()
        adj = frame.copy()
        i = 69
        adj.at[i, "open"] = round(frame.at[i, "close"] / 2, 2)
        adj.at[i, "close"] = round(frame.at[i, "close"] / 2, 2)
        adj.at[i, "high"] = adj.at[i, "close"] + 0.1
        adj.at[i, "low"] = adj.at[i, "close"] - 0.1
        adj.at[i, "volume"] = frame.at[i, "volume"] * 2
        clean, bad, reasons = validate(adj, source="unit", symbol="TEST01")
        self.assertEqual(len(bad), 0)
        self.assertEqual(len(clean), len(frame))
    def test_quarantine_file_roundtrip(self):
        from price_guards import validate_daily_bars, write_quarantine
        frame, _ = self._clean_frame()
        dirty = frame.copy()
        dirty.at[69, "open"] = 39.29
        dirty.at[69, "close"] = 39.10
        dirty.at[69, "high"] = 40.14
        dirty.at[69, "low"] = 38.96
        _, bad, reasons = validate_daily_bars(dirty, source="unit", symbol="TEST01")
        self.assertEqual(len(bad), 1)
        with tempfile.TemporaryDirectory(prefix="quar-", dir=ROOT / "tests") as directory:
            path = write_quarantine(pathlib.Path(directory), "TEST01", bad, reasons)
            self.assertTrue(path is not None and path.exists())
            back = pd.read_csv(path)
            self.assertEqual(len(back), 1)
    def test_merge_hook_drops_corrupt_rows(self):
        from fetch_utils import merge_daily_frames
        from price_guards import validate_daily_bars
        frame, _ = self._clean_frame(n=40)
        with tempfile.TemporaryDirectory(prefix="merge-", dir=ROOT / "tests") as directory:
            target = pathlib.Path(directory) / "TEST01.csv"
            frame.to_csv(target, index=False)
            incoming = frame.tail(3).copy()
            incoming.at[incoming.index[-1], "open"] = 39.29
            incoming.at[incoming.index[-1], "close"] = 39.10
            merged = merge_daily_frames(target, incoming, source="unit TEST01")
            self.assertNotIn(39.10, list(merged["close"]))
            qpath = pathlib.Path(directory) / "quarantine" / "TEST01.quarantine.csv"
            self.assertTrue(qpath.exists())

    def test_stock_write_success_handles_guarded_date_strings(self):
        from fetch_stock_data import _write_success

        frame, _ = self._clean_frame(n=20)
        frame.at[frame.index[-1], "open"] = 39.29
        frame.at[frame.index[-1], "high"] = 40.14
        frame.at[frame.index[-1], "low"] = 38.96
        frame.at[frame.index[-1], "close"] = 39.10
        with tempfile.TemporaryDirectory(prefix="stock-write-", dir=ROOT / "tests") as directory:
            target = pathlib.Path(directory) / "TEST01.csv"

            metadata = _write_success("TEST01", frame, target)

            self.assertEqual(metadata["data_end_date"], "2026-01-29")
            self.assertEqual(metadata["rows"], 19)
            self.assertTrue(target.exists())
            self.assertTrue(target.with_suffix(".meta.json").exists())


    def test_exdiv_down_gap_is_kept(self):
        from price_guards import validate_daily_bars
        frame = pd.DataFrame({"date": ["2026-01-05", "2026-01-06", "2026-01-07"], "open": [10.0, 10.1, 5.05], "high": [10.2, 10.3, 5.2], "low": [9.8, 9.9, 4.9], "close": [10.1, 10.0, 5.0], "volume": [1000000, 1100000, 2200000], "amount": [10100000.0, 11000000.0, 11000000.0], "turnover": [0.01, 0.011, 0.011]})
        clean, bad, _ = validate_daily_bars(frame, source="unit", symbol="EXDIV")
        self.assertEqual(len(bad), 0)
        self.assertEqual(len(clean), 3)
    def test_halt_resume_gap_is_kept(self):
        from price_guards import validate_daily_bars
        frame = pd.DataFrame({"date": ["2026-01-05", "2026-04-10"], "open": [10.0, 15.0], "high": [10.2, 16.0], "low": [9.8, 14.0], "close": [10.1, 15.5], "volume": [1000000, 900000], "amount": [10100000.0, 13950000.0], "turnover": [0.01, 0.009]})
        clean, bad, _ = validate_daily_bars(frame, source="unit", symbol="HALT")
        self.assertEqual(len(bad), 0)
        self.assertEqual(len(clean), 2)


    def test_bj_limit_move_is_kept(self):
        from price_guards import validate_daily_bars
        frame = pd.DataFrame({"date": ["2026-01-12", "2026-01-13"], "open": [7.0, 9.12], "high": [7.2, 9.2], "low": [6.9, 8.8], "close": [7.05, 9.17], "volume": [73000000, 73000000], "amount": [511000000.0, 669000000.0], "turnover": [0.29, 0.29]})
        clean, bad, _ = validate_daily_bars(frame, source="unit", symbol="920001")
        self.assertEqual(len(bad), 0)
        self.assertEqual(len(clean), 2)


    def test_ipo_first_rows_are_kept(self):
        from price_guards import validate_daily_bars
        frame = pd.DataFrame({"date": ["2026-05-15", "2026-05-18", "2026-05-19"], "open": [120.0, 100.0, 92.0], "high": [150.0, 107.0, 100.28], "low": [120.0, 91.0, 90.21], "close": [120.67, 95.26, 93.6], "volume": [36246293, 24325958, 17715750], "amount": [4712710709.0, 2380860100.0, 1710343900.0], "turnover": [0.75, 0.5, 0.36]})
        clean, bad, _ = validate_daily_bars(frame, source="unit", symbol="001393")
        self.assertEqual(len(bad), 0)
        self.assertEqual(len(clean), 3)

if __name__ == "__main__":
    unittest.main()
