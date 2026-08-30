from __future__ import annotations

import copy
import json
import pathlib
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from assistant import read_api, research, rolling_scan
from assistant.config import DEFAULTS
from fetch_state import FetchState
import run_research_worker as worker


END_DATE = "2026-08-28"
SYMBOL_DRIFTS = {
    "600000": 0.010,
    "600001": 0.008,
    "600002": 0.006,
    "600003": 0.004,
    "600004": 0.002,
    "600005": 0.001,
}


def _write_csv(stocks_dir: pathlib.Path, symbol: str, drift: float) -> None:
    stocks_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.bdate_range(end=END_DATE, periods=260)
    closes = [10.0 * (1.0 + drift) ** index for index in range(260)]
    frame = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "close": closes,
            "volume": 1_000_000.0,
            "amount": 50_000_000.0,
        }
    )
    frame.to_csv(stocks_dir / f"{symbol}.csv", index=False)


def _write_etf_csv(funds_dir: pathlib.Path, symbol: str, drift: float) -> None:
    funds_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.bdate_range(end=END_DATE, periods=260)
    closes = [10.0 * (1.0 + drift) ** index for index in range(260)]
    frame = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "close": closes,
            "volume": 1_000_000.0,
            "amount": 50_000_000.0,
        }
    )
    frame.to_csv(funds_dir / f"etf_{symbol}.csv", index=False)


def _make_state(root: pathlib.Path, symbols: list[str]) -> FetchState:
    state = FetchState(root / "fetch_state.sqlite3")
    state.register_assets("stock", symbols)
    for symbol in symbols:
        state.mark_success(
            "stock",
            symbol,
            data_end_date=END_DATE,
            source="test",
            rows=260,
            refresh_interval_seconds=86400,
        )
    return state


def _config(root: pathlib.Path, **overrides) -> dict:
    config = copy.deepcopy(DEFAULTS)
    config["fetch"]["state_db"] = str(root / "fetch_state.sqlite3")
    config["rolling_scan"] = {
        **copy.deepcopy(rolling_scan.DEFAULT_SETTINGS),
        "state_db": str(root / "rolling_scan.sqlite3"),
        "snapshot_path": str(root / "rolling_scan.json"),
        "min_sample": 3,
        "pool_size": 2,
        "min_rows": 30,
        "min_average_amount_20d": 1.0,
        "stale_after_days": 15,
        "max_symbols_per_pass": 100,
        **overrides,
    }
    return config


def _execute_sql(root: pathlib.Path, statement: str) -> None:
    connection = sqlite3.connect(root / "rolling_scan.sqlite3")
    try:
        connection.execute(statement)
        connection.commit()
    finally:
        connection.close()


class RollingScanTests(unittest.TestCase):
    def test_pass_scores_symbols_and_publishes_pool(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rolling-", dir=ROOT / "tests") as directory:
            root = pathlib.Path(directory)
            for symbol, drift in SYMBOL_DRIFTS.items():
                _write_csv(root / "stocks", symbol, drift)
            _make_state(root, list(SYMBOL_DRIFTS))
            config = _config(root)
            with patch.object(research, "DATA_DIR", root), patch.object(
                research, "UNIVERSE_PATH", root / "universe.json"
            ):
                summary = rolling_scan.run_rolling_pass("job-1", config=config)

            self.assertEqual(summary["status"], "ok")
            self.assertEqual(summary["analyzed_this_pass"], 6)
            self.assertEqual(summary["pool_status"], "active")
            self.assertEqual(summary["pool_size"], 2)
            self.assertEqual(summary["events_added"], 2)
            self.assertEqual(summary["coverage"]["total"], 6)
            self.assertEqual(summary["coverage"]["success"], 6)

            snapshot = json.loads((root / "rolling_scan.json").read_text(encoding="utf-8"))
            self.assertTrue(snapshot["preliminary"])
            self.assertEqual(
                [item["symbol"] for item in snapshot["pool"]],
                ["600000", "600001"],
            )
            self.assertEqual(snapshot["pool"][0]["rank"], 1)
            self.assertEqual(len(snapshot["score_table"]), 6)
            self.assertIn("disclaimer", snapshot)

            # 数据未变化时重复执行不应重算，也不应重复产生事件。
            with patch.object(research, "DATA_DIR", root), patch.object(
                research, "UNIVERSE_PATH", root / "universe.json"
            ):
                repeat = rolling_scan.run_rolling_pass("job-1", config=config)
            self.assertEqual(repeat["analyzed_this_pass"], 0)
            self.assertEqual(repeat["events_added"], 0)

    def test_accumulating_until_min_sample_then_enter_events(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rolling-", dir=ROOT / "tests") as directory:
            root = pathlib.Path(directory)
            for symbol, drift in SYMBOL_DRIFTS.items():
                _write_csv(root / "stocks", symbol, drift)
            _make_state(root, list(SYMBOL_DRIFTS))
            with patch.object(research, "DATA_DIR", root), patch.object(
                research, "UNIVERSE_PATH", root / "universe.json"
            ):
                early = rolling_scan.run_rolling_pass(
                    "job-1", config=_config(root, min_sample=100)
                )
            self.assertEqual(early["pool_status"], "accumulating")
            self.assertEqual(early["pool_size"], 0)
            self.assertEqual(early["events_added"], 0)

            with patch.object(research, "DATA_DIR", root), patch.object(
                research, "UNIVERSE_PATH", root / "universe.json"
            ):
                activated = rolling_scan.run_rolling_pass("job-1", config=_config(root))
            self.assertEqual(activated["pool_status"], "active")
            self.assertEqual(activated["events_added"], 2)
            store = rolling_scan.RollingScanStore(root / "rolling_scan.sqlite3")
            events = store.recent_events(10)
            self.assertTrue(all(event["event"] == "enter" for event in events))

    def test_rank_drop_emits_exit_and_replacement_enter(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rolling-", dir=ROOT / "tests") as directory:
            root = pathlib.Path(directory)
            for symbol, drift in SYMBOL_DRIFTS.items():
                _write_csv(root / "stocks", symbol, drift)
            _make_state(root, list(SYMBOL_DRIFTS))
            config = _config(root)
            with patch.object(research, "DATA_DIR", root), patch.object(
                research, "UNIVERSE_PATH", root / "universe.json"
            ):
                rolling_scan.run_rolling_pass("job-1", config=config)

            _execute_sql(root, "UPDATE scores SET trend_score = -9 WHERE symbol = '600000'")
            with patch.object(research, "DATA_DIR", root), patch.object(
                research, "UNIVERSE_PATH", root / "universe.json"
            ):
                summary = rolling_scan.run_rolling_pass("job-1", config=config)

            self.assertEqual(summary["events_added"], 2)
            store = rolling_scan.RollingScanStore(root / "rolling_scan.sqlite3")
            events = {event["symbol"]: event for event in store.recent_events(10)}
            self.assertEqual(events["600000"]["event"], "exit")
            self.assertIn("跌出前", events["600000"]["reason"])
            self.assertEqual(events["600002"]["event"], "enter")
            snapshot = json.loads((root / "rolling_scan.json").read_text(encoding="utf-8"))
            self.assertEqual(
                [item["symbol"] for item in snapshot["pool"]],
                ["600001", "600002"],
            )

    def test_stale_data_exits_with_reason(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rolling-", dir=ROOT / "tests") as directory:
            root = pathlib.Path(directory)
            for symbol, drift in SYMBOL_DRIFTS.items():
                _write_csv(root / "stocks", symbol, drift)
            _make_state(root, list(SYMBOL_DRIFTS))
            config = _config(root)
            with patch.object(research, "DATA_DIR", root), patch.object(
                research, "UNIVERSE_PATH", root / "universe.json"
            ):
                rolling_scan.run_rolling_pass("job-1", config=config)

            # 模拟该票多日未刷新成功：状态库与得分库的截止日一致地停在旧日期，
            # 因此不会触发重算，只在排名时因数据过期被剔除。
            state = FetchState(root / "fetch_state.sqlite3")
            state.mark_success(
                "stock",
                "600000",
                data_end_date="2026-08-01",
                source="test",
                rows=260,
                refresh_interval_seconds=86400,
            )
            _execute_sql(root, "UPDATE scores SET data_end_date = '2026-08-01' WHERE symbol = '600000'")
            with patch.object(research, "DATA_DIR", root), patch.object(
                research, "UNIVERSE_PATH", root / "universe.json"
            ):
                summary = rolling_scan.run_rolling_pass("job-1", config=config)

            store = rolling_scan.RollingScanStore(root / "rolling_scan.sqlite3")
            events = {event["symbol"]: event for event in store.recent_events(10)}
            self.assertEqual(events["600000"]["event"], "exit")
            self.assertIn("数据过期", events["600000"]["reason"])
            self.assertEqual(summary["pool_size"], 2)

    def test_missing_csv_recorded_as_error_without_failing_pass(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rolling-", dir=ROOT / "tests") as directory:
            root = pathlib.Path(directory)
            for symbol, drift in SYMBOL_DRIFTS.items():
                _write_csv(root / "stocks", symbol, drift)
            symbols = list(SYMBOL_DRIFTS) + ["600006"]
            _make_state(root, symbols)
            config = _config(root)
            with patch.object(research, "DATA_DIR", root), patch.object(
                research, "UNIVERSE_PATH", root / "universe.json"
            ):
                summary = rolling_scan.run_rolling_pass("job-1", config=config)

            self.assertEqual(summary["status"], "ok")
            self.assertEqual(summary["analyzed_this_pass"], 7)
            store = rolling_scan.RollingScanStore(root / "rolling_scan.sqlite3")
            scored = store.scored_end_dates()
            self.assertIn(("stock", "600006"), scored)
            rows = {
                row["symbol"]: row
                for row in store.all_scores()
            }
            self.assertTrue(rows["600006"]["quality_status"].startswith("error:"))
            self.assertIsNone(rows["600006"]["trend_score"])

    def test_read_api_filters_rolling_snapshot(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rolling-", dir=ROOT / "tests") as directory:
            root = pathlib.Path(directory)
            for symbol, drift in SYMBOL_DRIFTS.items():
                _write_csv(root / "stocks", symbol, drift)
            _make_state(root, list(SYMBOL_DRIFTS))
            config = _config(root)
            with patch.object(research, "DATA_DIR", root), patch.object(
                research, "UNIVERSE_PATH", root / "universe.json"
            ), patch.object(research, "load_config", return_value=config):
                rolling_scan.run_rolling_pass("job-1", config=config)

            with patch.object(research, "ASSISTANT_RESULTS_DIR", root):
                with patch.object(research, "load_config", return_value=config):
                    result = read_api.get_rolling_scan(symbol="600001")
                self.assertEqual(result["status"], "ok")
                self.assertEqual(result["pool"][0]["symbol"], "600001")
                self.assertEqual(result["score_table"][0]["symbol"], "600001")

                with patch.object(research, "load_config", return_value=config):
                    everything = read_api.get_rolling_scan(limit=100)
                self.assertEqual(len(everything["score_table"]), 6)

                with patch.object(research, "load_config", return_value=config):
                    missing = read_api.get_rolling_scan(symbol="600006")
                self.assertEqual(missing["pool"], [])
                self.assertEqual(missing["score_table"], [])

            empty_config = copy.deepcopy(config)
            empty_config["rolling_scan"]["snapshot_path"] = str(
                root / "empty" / "rolling_scan.json"
            )
            with patch.object(research, "ASSISTANT_RESULTS_DIR", root / "empty"):
                with patch.object(research, "load_config", return_value=empty_config):
                    empty = read_api.get_rolling_scan()
            self.assertEqual(empty["status"], "no_rolling_scan")

    def test_read_api_uses_custom_snapshot_path(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="rolling-custom-path-",
            dir=ROOT / "tests",
        ) as directory:
            root = pathlib.Path(directory)
            for symbol, drift in SYMBOL_DRIFTS.items():
                _write_csv(root / "stocks", symbol, drift)
            _make_state(root, list(SYMBOL_DRIFTS))
            custom_path = root / "custom" / "snapshot.json"
            config = _config(root, snapshot_path=str(custom_path))
            with patch.object(research, "DATA_DIR", root), patch.object(
                research, "UNIVERSE_PATH", root / "universe.json"
            ):
                rolling_scan.run_rolling_pass("job-1", config=config)

            with patch.object(research, "load_config", return_value=config):
                result = read_api.get_rolling_scan()
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["pool"][0]["symbol"], "600000")
            self.assertTrue(custom_path.exists())

    def test_rolling_scan_rejects_external_fetch_state_path(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="rolling-external-state-",
            dir=ROOT / "tests",
        ) as directory:
            root = pathlib.Path(directory)
            config = _config(root)
            external_state = ROOT.parent / "outside-rolling-state.sqlite3"
            config["fetch"]["state_db"] = str(external_state)
            with self.assertRaises(ValueError):
                rolling_scan.run_rolling_pass("job-1", config=config)
            self.assertFalse(external_state.exists())

    def test_per_type_pools_and_pool_type_priority(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rolling-", dir=ROOT / "tests") as directory:
            root = pathlib.Path(directory)
            for symbol, drift in SYMBOL_DRIFTS.items():
                _write_csv(root / "stocks", symbol, drift)
            _write_etf_csv(root / "funds", "510300", 0.012)
            _write_etf_csv(root / "funds", "510500", 0.006)
            state = FetchState(root / "fetch_state.sqlite3")
            state.register_assets("stock", list(SYMBOL_DRIFTS))
            for symbol in SYMBOL_DRIFTS:
                state.mark_success(
                    "stock", symbol,
                    data_end_date=END_DATE, source="test", rows=260,
                    refresh_interval_seconds=86400,
                )
            state.register_assets("etf", ["510300", "510500"])
            state.register_assets("otc_fund", ["000001"])
            for asset_type, symbols in (
                ("etf", ["510300", "510500"]),
                ("otc_fund", ["000001"]),
            ):
                for symbol in symbols:
                    state.mark_success(
                        asset_type, symbol,
                        data_end_date=END_DATE, source="test", rows=260,
                        refresh_interval_seconds=86400,
                    )

            # 优先级：单波名额先给池内类型，场外基金排队在后。
            config = _config(root, max_symbols_per_pass=4)
            with patch.object(research, "DATA_DIR", root), patch.object(
                research, "UNIVERSE_PATH", root / "universe.json"
            ):
                summary = rolling_scan.run_rolling_pass("job-1", config=config)
            self.assertEqual(summary["analyzed_this_pass"], 4)
            store = rolling_scan.RollingScanStore(root / "rolling_scan.sqlite3")
            scored_types = {row["asset_type"] for row in store.all_scores()}
            self.assertEqual(scored_types, {"stock", "etf"})

            # 每类各自开池：股票和 ETF 各出 top-2，互不挤占。
            config = _config(root, max_symbols_per_pass=100, min_sample=2)
            with patch.object(research, "DATA_DIR", root), patch.object(
                research, "UNIVERSE_PATH", root / "universe.json"
            ):
                summary = rolling_scan.run_rolling_pass("job-1", config=config)
            self.assertEqual(summary["pool_status"], "active")
            self.assertEqual(summary["eligible_by_type"], {"stock": 6, "etf": 2})
            snapshot = json.loads((root / "rolling_scan.json").read_text(encoding="utf-8"))
            pool = snapshot["pool"]
            self.assertEqual(
                [(item["asset_type"], item["symbol"]) for item in pool],
                [
                    ("stock", "600000"),
                    ("stock", "600001"),
                    ("etf", "510300"),
                    ("etf", "510500"),
                ],
            )
            self.assertEqual(len(snapshot["score_table"]), 8)
            enters = [
                event for event in snapshot["recent_events"] if event["event"] == "enter"
            ]
            self.assertEqual(len(enters), 4)

    def test_worker_hook_isolates_failures_and_respects_enabled_flag(self) -> None:
        config = _config(pathlib.Path(ROOT / "tests"))
        config["rolling_scan"] = {"enabled": False}
        status_value: dict = {}
        worker._run_rolling_scan("job-1", config, status_value)
        self.assertNotIn("rolling_scan", status_value)

        with tempfile.TemporaryDirectory(prefix="rolling-", dir=ROOT / "tests") as directory:
            root = pathlib.Path(directory)
            config = _config(root)
            config["rolling_scan"]["state_db"] = str(root)  # 目录路径，必然打开失败
            status_value = {}
            worker._run_rolling_scan("job-1", config, status_value)
        self.assertEqual(status_value["rolling_scan"]["status"], "failed")
        self.assertIn("error", status_value["rolling_scan"])

    def test_compact_status_keeps_only_rolling_summary(self) -> None:
        compact = worker._compact_status(
            {
                "job_id": "job",
                "status": "running",
                "rolling_scan": {
                    "status": "ok",
                    "pool_status": "active",
                    "analyzed_this_pass": 100,
                    "pool_size": 30,
                    "events_added": 4,
                    "coverage": {"by_asset_type": {"stock": {"total": 1}}},
                    "snapshot_path": "x" * 500,
                },
            }
        )
        self.assertIn("rolling_scan", compact)
        self.assertNotIn("coverage", compact["rolling_scan"])
        self.assertNotIn("snapshot_path", compact["rolling_scan"])
        self.assertEqual(compact["rolling_scan"]["pool_size"], 30)


if __name__ == "__main__":
    unittest.main()
