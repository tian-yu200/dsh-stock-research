"""阶段 0 快照回归测试：锁定 ``run_research`` 整条输出链的行为。

这是 research.py 重构（docs/重构计划.md §3）的安全网，约定：

- 数据来自 ``tests/fixtures/research_snapshot`` 的固定小型数据集
  （3 股票 + 2 ETF + 2 场外基金；``etf_510300.csv`` 兼作基准），
  测试只复制到 ``quant/tests`` 下的临时目录，不重新生成。
- ``_run_qlib_stock_model`` 打桩返回固定指标与分数；编排层测试不训练真模型。
- 断言全部逐字段对比（不整份 JSON 字符串对比），失败信息可读。
- 对随真实时钟变化的字段（如 ``age_days``）只断言派生布尔值，不断言数值。

重构期间的每个提交都必须保持本文件全绿；若行为确需变更，
先改快照、说明理由，再动实现。
"""
from __future__ import annotations

import copy
import json
import pathlib
import shutil
import sys
import tempfile
import unittest
from contextlib import ExitStack
from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from assistant import notifications, paths, research
from assistant.config import DEFAULTS, config_snapshot

FIXTURE_DIR = ROOT / "tests" / "fixtures" / "research_snapshot"
RUN_ID = "20260825T100000Z-a1b2c3d4"
FIXED_NOW_ISO = "2026-08-25T10:00:00+00:00"
DATA_END = "2026-08-25"
STOCKS = ("000001", "300750", "600519")
ETFS = ("159915", "510300")
FUNDS = ("110020", "161725")

# 与真实模型相同的键结构；数值取在 gate 阈值内的固定值，
# 使个股门槛全绿（见 test_failed_stock_gate... 中对阈值的反向用例）。
STOCK_MODEL_STUB = {
    "model_available": True,
    "model_name": "Qlib Alpha158 + LightGBM",
    "qlib_version": "snapshot-stub",
    "provider": "snapshot-stub-provider",
    "data_end_date": DATA_END,
    "latest_signal_date": DATA_END,
    "valid": {"observations": 120, "mean_rank_ic": 0.04, "rank_ic_ir": 0.40},
    "test": {"observations": 60, "mean_rank_ic": 0.05, "rank_ic_ir": 0.50},
    "mean_rank_ic": 0.05,
    "test_rank_ic": 0.05,
    "rank_ic_ir": 0.50,
    "top_bottom_excess_return": 0.01,
    "forward_group_metrics": {
        "group_count": 5,
        "observations": 60,
        "valid_days": 12,
        "group_returns": {"1": -0.001, "5": 0.012},
        "top_group_return": 0.012,
        "bottom_group_return": -0.001,
        "top_bottom_excess_return": 0.01,
    },
    "net_annualized_return": 0.12,
    "test_max_drawdown": -0.10,
    "max_drawdown": -0.10,
    "transaction_cost_bps": 10.0,
    "mean_daily_turnover": 0.45,
    "benchmark": {
        "name": "equal_weight_stock_universe",
        "annualized_return": 0.05,
        "max_drawdown": -0.15,
        "excess_annualized_return": 0.07,
    },
    "training_best_iteration": 180,
    "scores": {"600519": 1.5, "000001": 1.2, "300750": 0.9},
    "leakage_check": {"passed": True, "label_horizon_days": 20},
}

_SIGNAL_BASE_KEYS = {
    "symbol", "name", "latest_value", "return_20d", "return_60d", "return_252d",
    "volatility_60d_annualized", "max_drawdown", "max_drawdown_252d",
    "max_drawdown_peak_date", "max_drawdown_trough_date", "max_drawdown_duration_days",
    "max_drawdown_recovery_date", "max_drawdown_recovery_days", "current_drawdown",
    "current_drawdown_peak_date", "current_drawdown_duration_days",
    "current_drawdown_recovered", "ma20", "ma60", "ma200", "trend_score",
    "series_type", "benchmark_symbol", "benchmark_data_status",
    "relative_return_20d_vs_benchmark", "relative_return_60d_vs_benchmark",
    "relative_return_252d_vs_benchmark", "peer_comparison_status",
    "peer_comparison_period_days", "peer_count", "peer_average_return_252d",
    "excess_return_252d_vs_peer_mean", "peer_return_252d_percentile",
    "peer_rank_252d", "peer_universe_return_252d",
}
_LIQUIDITY_KEYS = {
    "liquidity_window_days", "average_amount_20d", "amount_data_status",
    "average_volume_20d", "volume_data_status", "liquidity_status",
}
STOCK_SIGNAL_KEYS = _SIGNAL_BASE_KEYS | _LIQUIDITY_KEYS | {"qlib_score", "model_rank"}
ETF_SIGNAL_KEYS = _SIGNAL_BASE_KEYS | _LIQUIDITY_KEYS
FUND_SIGNAL_KEYS = set(_SIGNAL_BASE_KEYS)

STOCK_CHECK_KEYS = {
    "data_quality_and_model_available", "mean_rank_ic_threshold",
    "rank_ic_ir_threshold", "top_bottom_excess_return_threshold",
    "test_max_drawdown_threshold", "fit_quality_threshold",
    "stability_ic_std_threshold",
    "data_quality_passed", "freshness_passed",
    "refresh_passed", "leakage_check_passed", "full_universe_screen_passed",
    "positive_signal_passed", "risk_constraints_passed",
}
PATH_CHECK_KEYS = {
    "data_quality_passed", "freshness_passed", "refresh_passed",
    "risk_constraints_passed",
}
PORTFOLIO_CHECK_KEYS = {
    "max_single_weight", "known_asset_types", "stock_exposure", "etf_exposure",
    "otc_fund_exposure", "min_cash_weight", "max_daily_turnover",
}
RISK_RESULT_KEYS = {
    "passed", "checks", "requested_weights", "target_weights",
    "category_exposure", "cash_weight", "turnover", "unknown_symbols",
    "failure_reasons",
}
RUN_FILE_KEYS = {
    "manifest", "data_quality", "data_facts", "model_signals", "rule_gates",
    "decision_support", "model_metadata", "run_config", "summary", "logs",
}


def _fixture_close(symbol: str) -> pd.Series:
    if symbol in STOCKS:
        path = FIXTURE_DIR / "stocks" / f"{symbol}.csv"
    elif symbol in ETFS:
        path = FIXTURE_DIR / "funds" / f"etf_{symbol}.csv"
    else:
        path = FIXTURE_DIR / "funds" / f"fund_{symbol}.csv"
    frame = pd.read_csv(path)
    series = pd.Series(
        pd.to_numeric(frame["close"], errors="coerce").to_numpy(),
        index=pd.to_datetime(frame["date"], errors="coerce"),
    )
    return series.dropna().sort_index()


def _relative_return_vs_benchmark(close: pd.Series, period: int) -> float:
    benchmark = _fixture_close("510300")
    return float(
        (close.iloc[-1] / close.iloc[-period - 1])
        / (benchmark.iloc[-1] / benchmark.iloc[-period - 1])
        - 1.0
    )


class ResearchSnapshotTests(unittest.TestCase):
    """整链快照：一次全市场同步运行产出的每个落盘文件逐字段锁定。"""

    def setUp(self) -> None:
        # fixed_path() 只放行 quant/ 内的路径，临时目录必须建在 tests/ 下
        # （与 AssistantStateMachineTests 相同约定）。
        self.temporary = tempfile.TemporaryDirectory(
            prefix="research-snapshot-",
            dir=ROOT / "tests",
        )
        self.temp_root = pathlib.Path(self.temporary.name)
        self.data_dir = self.temp_root / "data"
        self.results_dir = self.temp_root / "results" / "assistant"
        self.runs_dir = self.results_dir / "runs"
        self.logs_dir = self.temp_root / "logs"
        self.provider_dir = self.temp_root / "qlib_data" / "akshare_data"
        self.lock_path = self.logs_dir / "research.lock"
        self.notifications_path = self.logs_dir / "notifications.json"
        for directory in (
            self.data_dir / "stocks",
            self.data_dir / "funds",
            self.results_dir,
            self.logs_dir,
            self.provider_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        shutil.copytree(FIXTURE_DIR / "stocks", self.data_dir / "stocks", dirs_exist_ok=True)
        shutil.copytree(FIXTURE_DIR / "funds", self.data_dir / "funds", dirs_exist_ok=True)

        self.config = copy.deepcopy(DEFAULTS)
        self.config["paths"] = {
            "data_dir": str(self.data_dir),
            "qlib_provider": str(self.provider_dir),
            "results_dir": str(self.results_dir),
            "logs_dir": str(self.logs_dir),
        }
        self.config["fetch"]["enabled"] = False
        self.config["fetch"]["max_data_age_days"] = 100000
        self.config["fetch"]["state_db"] = str(self.temp_root / "fetch_state.sqlite3")

        self.heartbeats: list[tuple[str, str]] = []
        real_touch = research._touch_status

        def _record_heartbeat(status_path, phase):
            real_touch(status_path, phase)
            record = json.loads(pathlib.Path(status_path).read_text(encoding="utf-8"))
            self.heartbeats.append((str(phase), str(record.get("status"))))

        self.patches = ExitStack()
        # UNIVERSE_PATH 指向临时目录中不存在的文件 → 走"无权威清单、按目录发现"分支。
        for module, name, value in (
            (research, "DATA_DIR", self.data_dir),
            (research, "ASSISTANT_RESULTS_DIR", self.results_dir),
            (research, "RUNS_DIR", self.runs_dir),
            (research, "LOCK_PATH", self.lock_path),
            (research, "QLIB_PROVIDER_DIR", self.provider_dir),
            (research, "UNIVERSE_PATH", self.data_dir / "universe.json"),
            (paths, "RUNS_DIR", self.runs_dir),
            (notifications, "NOTIFICATIONS_PATH", self.notifications_path),
        ):
            self.patches.enter_context(patch.object(module, name, value))
        self.patches.enter_context(
            patch.object(research, "load_config", return_value=self.config)
        )
        self.patches.enter_context(
            patch.object(research, "_now", return_value=datetime(2026, 8, 25, 10, 0, tzinfo=timezone.utc))
        )
        self.patches.enter_context(
            patch.object(research, "_touch_status", _record_heartbeat)
        )

    def tearDown(self) -> None:
        self.patches.close()
        self.temporary.cleanup()

    # ---------- 小工具 ----------

    def _run_dir(self) -> pathlib.Path:
        return self.runs_dir / "2026" / "08" / "25" / RUN_ID

    def _read_json(self, path) -> object:
        with pathlib.Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _assert_same_path(self, actual, expected) -> None:
        self.assertEqual(
            pathlib.Path(str(actual)).resolve(),
            pathlib.Path(expected).resolve(),
        )

    def _expected_weights(self) -> tuple[float, float, float]:
        """按现行等权切分公式独立推导预期权重（锁死该公式）。"""
        risk = self.config["risk"]
        budget = min(
            float(risk["max_daily_turnover"]),
            1.0 - float(risk["min_cash_weight"]),
        ) / 3.0
        stock_weight = min(budget / 3.0, float(risk["max_single_weight"]))
        path_weight = min(budget, float(risk["max_single_weight"]))
        return budget, stock_weight, path_weight

    def _assert_signal_core(self, signal: dict, symbol: str) -> None:
        close = _fixture_close(symbol)
        rising = symbol in {"000001", "600519", "510300", "110020"}
        self.assertEqual(signal["symbol"], symbol)
        self.assertIsNone(signal["name"])
        self.assertEqual(
            signal["series_type"], "net_value" if symbol in FUNDS else "daily_close"
        )
        self.assertEqual(signal["benchmark_symbol"], "510300")
        self.assertEqual(signal["benchmark_data_status"], "available")
        self.assertEqual(signal["peer_comparison_status"], "available")
        self.assertEqual(signal["peer_comparison_period_days"], 252)
        self.assertEqual(signal["peer_count"], 2 if symbol in STOCKS else 1)
        expected_rank = {"600519": 1, "000001": 1, "300750": 3,
                         "510300": 1, "159915": 2, "110020": 1, "161725": 2}[symbol]
        self.assertEqual(signal["peer_rank_252d"], expected_rank)
        self.assertAlmostEqual(
            signal["latest_value"], float(close.iloc[-1]), places=12
        )
        self.assertAlmostEqual(
            signal["return_20d"],
            float(close.iloc[-1] / close.iloc[-21] - 1.0),
            places=12,
        )
        if rising:
            self.assertEqual(signal["max_drawdown"], 0.0)
            self.assertEqual(signal["current_drawdown"], 0.0)
            self.assertTrue(signal["current_drawdown_recovered"])
            self.assertEqual(signal["max_drawdown_duration_days"], 0)
            self.assertEqual(signal["max_drawdown_recovery_days"], 0)
            self.assertEqual(signal["current_drawdown_duration_days"], 0)
            self.assertGreater(signal["trend_score"], 0.0)
        else:
            self.assertAlmostEqual(signal["max_drawdown"], -0.5, places=12)
            self.assertAlmostEqual(signal["current_drawdown"], -0.5, places=12)
            self.assertFalse(signal["current_drawdown_recovered"])
            self.assertLess(signal["trend_score"], 0.0)
        if symbol == "510300":
            for period in (20, 60, 252):
                self.assertEqual(
                    signal[f"relative_return_{period}d_vs_benchmark"], 0.0
                )
        else:
            for period in (20, 60, 252):
                self.assertAlmostEqual(
                    signal[f"relative_return_{period}d_vs_benchmark"],
                    _relative_return_vs_benchmark(close, period),
                    places=12,
                )

    def _assert_success_decision_support(self, decision_support: dict) -> None:
        _, stock_weight, path_weight = self._expected_weights()
        self.assertEqual(
            set(decision_support),
            {
                "allocation_basis", "hypothetical_capital", "broker_connected",
                "auto_order", "asset_paths", "target_weights", "cash_weight",
                "portfolio_gate", "publication",
                # 2026-09-11 P0-3：分类别发布后 publication_blocked 恒存在
                # （全量发布时为 False），使调用方无需区分"键缺失"与"未阻断"。
                "publication_blocked",
            },
        )
        self.assertFalse(decision_support["publication_blocked"])
        self.assertEqual(decision_support["publication"]["state"], "full")
        self.assertTrue(decision_support["publication"]["published"])
        self.assertEqual(decision_support["allocation_basis"], "基于假设资金/配置的目标权重")
        self.assertEqual(decision_support["hypothetical_capital"], 100000.0)
        self.assertFalse(decision_support["broker_connected"])
        self.assertFalse(decision_support["auto_order"])

        asset_paths = decision_support["asset_paths"]
        self.assertEqual(set(asset_paths), {"stock", "etf", "otc_fund"})

        stock_decision = asset_paths["stock"]
        self.assertEqual(
            set(stock_decision),
            {"status", "action", "target_weights", "cash_weight", "reason"},
        )
        self.assertEqual(stock_decision["status"], "candidate")
        self.assertEqual(stock_decision["action"], "观察/候选")
        self.assertEqual(
            set(stock_decision["target_weights"]), set(STOCKS)
        )
        for symbol in STOCKS:
            self.assertAlmostEqual(
                stock_decision["target_weights"][symbol], stock_weight, places=12
            )
        self.assertAlmostEqual(stock_decision["cash_weight"], 1.0 - 3.0 * stock_weight, places=9)
        self.assertEqual(
            stock_decision["reason"],
            "Qlib 模型和风险门槛均通过；仍需人工确认基本面与交易条件。",
        )

        etf_decision = asset_paths["etf"]
        self.assertEqual(
            set(etf_decision),
            {"status", "action", "target_weights", "cash_weight", "reason"},
        )
        self.assertEqual(etf_decision["status"], "candidate")
        self.assertEqual(etf_decision["action"], "配置候选")
        self.assertEqual(etf_decision["target_weights"], {"510300": path_weight})
        self.assertAlmostEqual(etf_decision["cash_weight"], 1.0 - path_weight, places=9)
        self.assertEqual(
            etf_decision["reason"],
            "技术趋势、相对表现和风险约束通过；需结合费用、流动性和个人期限确认。",
        )

        fund_decision = asset_paths["otc_fund"]
        self.assertEqual(
            set(fund_decision),
            {
                "status", "action", "target_weights", "cash_weight", "reason",
                "no_intraday_price_or_exact_shares",
            },
        )
        self.assertEqual(fund_decision["status"], "candidate")
        self.assertEqual(fund_decision["target_weights"], {"110020": path_weight})
        self.assertTrue(fund_decision["no_intraday_price_or_exact_shares"])

        combined = dict(decision_support["target_weights"])
        self.assertEqual(
            set(combined), set(STOCKS) | {"510300", "110020"}
        )
        for symbol in STOCKS:
            self.assertAlmostEqual(combined[symbol], stock_weight, places=12)
        self.assertAlmostEqual(combined["510300"], path_weight, places=12)
        self.assertAlmostEqual(combined["110020"], path_weight, places=12)
        self.assertAlmostEqual(decision_support["cash_weight"], 0.8, places=9)
        self.assertTrue(decision_support["portfolio_gate"]["passed"])

    # ---------- 用例 1：全市场成功链快照 ----------

    def test_full_universe_success_snapshot(self) -> None:
        with patch.object(
            research, "_run_id", return_value=RUN_ID
        ), patch.object(
            research,
            "_run_qlib_stock_model",
            return_value=copy.deepcopy(STOCK_MODEL_STUB),
        ) as model:
            result = research.run_research("on_demand", "all", None, _force_sync=True)

        model.assert_called_once_with(self.config)
        run_dir = self._run_dir()

        # --- 返回值信封 ---
        self.assertEqual(
            set(result),
            {
                "run_id", "status", "failure_reasons", "data_end_date",
                "result_path", "current_path", "notification_id", "data_facts",
                "model_signals", "rule_gates", "decision_support",
            },
        )
        self.assertEqual(result["run_id"], RUN_ID)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["failure_reasons"], [])
        self.assertEqual(result["data_end_date"], DATA_END)
        self._assert_same_path(result["result_path"], run_dir)
        self._assert_same_path(
            result["current_path"], self.results_dir / "current.json"
        )

        # --- status.json 心跳生命周期与终态 ---
        self.assertEqual(
            self.heartbeats,
            [("refreshed", "running"), ("scanned", "running"),
             ("signals", "running"), ("gates", "running")],
        )
        status = self._read_json(run_dir / "status.json")
        self.assertEqual(
            set(status),
            {"run_id", "status", "finished_at", "data_end_date", "result_path",
             "failure_reasons", "depth"},
        )
        self.assertEqual(status["run_id"], RUN_ID)
        self.assertEqual(status["status"], "success")
        self.assertEqual(status["depth"], "standard")
        self.assertEqual(status["finished_at"], FIXED_NOW_ISO)
        self.assertEqual(status["data_end_date"], DATA_END)
        self.assertEqual(status["failure_reasons"], [])
        self._assert_same_path(status["result_path"], run_dir)

        # --- manifest.json ---
        manifest = self._read_json(run_dir / "manifest.json")
        self.assertEqual(
            set(manifest),
            {
                "run_id", "status", "started_at", "finished_at", "mode", "depth",
                "asset_scope", "asset_types", "analyzed_types",
                "requested_symbols", "data_sources", "data_end_date",
                "retry_count", "failure_reasons", "publication", "config_snapshot",
                "paths", "stage_timings",
            },
        )
        self.assertEqual(manifest["status"], "success")
        self.assertEqual(manifest["depth"], "standard")
        self.assertTrue(manifest["stage_timings"])
        self.assertEqual(manifest["started_at"], FIXED_NOW_ISO)
        self.assertEqual(manifest["finished_at"], FIXED_NOW_ISO)
        self.assertEqual(manifest["mode"], "on_demand")
        self.assertEqual(manifest["asset_scope"], "all")
        self.assertEqual(manifest["asset_types"], ["stock", "etf", "otc_fund"])
        self.assertEqual(manifest["analyzed_types"], ["stock", "etf", "otc_fund"])
        self.assertEqual(manifest["requested_symbols"], [])
        self.assertEqual(
            manifest["data_sources"], ["AkShare cached CSV", "Qlib provider"]
        )
        self.assertEqual(manifest["retry_count"], 2)
        self.assertEqual(manifest["failure_reasons"], [])
        self.assertEqual(manifest["config_snapshot"], config_snapshot(self.config))
        self.assertEqual(set(manifest["paths"]), RUN_FILE_KEYS)
        expected_targets = {
            name: run_dir / f"{name}.json"
            for name in (
                "manifest", "data_quality", "data_facts", "model_signals",
                "rule_gates", "decision_support", "model_metadata",
            )
        }
        expected_targets["run_config"] = run_dir / "run_config.yaml"
        expected_targets["summary"] = run_dir / "summary.md"
        expected_targets["logs"] = run_dir / "logs" / "research.log"
        for name in RUN_FILE_KEYS:
            self._assert_same_path(manifest["paths"][name], expected_targets[name])

        # --- data_quality.json ---
        data_quality = self._read_json(run_dir / "data_quality.json")
        self.assertTrue(data_quality["passed"])
        self.assertEqual(data_quality["data_end_date"], DATA_END)
        self.assertEqual(data_quality["asset_count"], 7)
        self.assertEqual(data_quality["eligible_asset_count"], 7)
        self.assertEqual(data_quality["coverage_ratio"], 1.0)
        self.assertEqual(data_quality["minimum_coverage_ratio"], 0.95)
        self.assertEqual(data_quality["failed_assets"], [])
        self.assertEqual(data_quality["stale_assets"], [])
        self.assertFalse(data_quality["stale"])
        self.assertFalse(data_quality["all_sources_failed"])
        self.assertEqual(data_quality["max_allowed_age_days"], 100000)
        self.assertEqual(
            data_quality["refresh"],
            {"attempted": False, "passed": True, "results": []},
        )
        expected_keys = (
            {f"stock:{s}" for s in STOCKS}
            | {f"etf:{e}" for e in ETFS}
            | {f"otc_fund:{f}" for f in FUNDS}
        )
        self.assertEqual(set(data_quality["asset_freshness"]), expected_keys)
        for item in data_quality["asset_freshness"].values():
            self.assertEqual(item["data_end_date"], DATA_END)
            self.assertFalse(item["stale"])
        self.assertEqual(
            set(data_quality["eligible_assets"]),
            expected_keys,
        )
        self.assertFalse(data_quality["universe"]["complete"])
        self.assertIsNone(data_quality["universe"]["source"])
        self.assertIsNone(data_quality["universe"]["generated_at"])
        self.assertEqual(
            data_quality["universe"]["counts"],
            {"stock": 3, "etf": 2, "otc_fund": 2},
        )
        self.assertEqual(data_quality["universe"]["authoritative_asset_types"], [])
        self.assertTrue(data_quality["time_leakage_check"]["passed"])

        # --- data_facts.json ---
        facts = self._read_json(run_dir / "data_facts.json")
        self.assertEqual(len(facts), 7)
        by_symbol = {item["symbol"]: item for item in facts}
        self.assertEqual(set(by_symbol), set(STOCKS) | set(ETFS) | set(FUNDS))
        for symbol, fact in by_symbol.items():
            self.assertEqual(fact["quality_status"], "pass")
            self.assertEqual(fact["issues"], [])
            self.assertEqual(fact["rows"], 300)
            self.assertEqual(fact["data_end_date"], DATA_END)
            self.assertEqual(fact["duplicate_rows"], 0)
            self.assertFalse(fact["freshness"]["stale"])
            self.assertEqual(
                set(fact["checks"]),
                {
                    "schema", "duplicate_dates", "adjustment_or_nav", "liquidity",
                }
                | ({"price_consistency"} if symbol in STOCKS or symbol in ETFS else set())
                | ({"ohlc_values"} if symbol in STOCKS or symbol in ETFS else {"nav_values"}),
            )
            adjustment = fact["checks"]["adjustment_or_nav"]
            self.assertEqual(adjustment["status"], "assumed")
            self.assertTrue(adjustment["passed"])
        for symbol in STOCKS:
            self.assertEqual(by_symbol[symbol]["adjustment"], "hfq")
            self.assertEqual(by_symbol[symbol]["source"], "AkShare cached daily bars")
            self.assertEqual(by_symbol[symbol]["optional_missing_values"], 0)
        for symbol in ETFS:
            self.assertEqual(by_symbol[symbol]["adjustment"], "qfq")
        for symbol in FUNDS:
            self.assertEqual(
                by_symbol[symbol]["adjustment"], "unit_nav_no_price_adjustment"
            )
            self.assertEqual(
                by_symbol[symbol]["source"], "AkShare fund NAV CSV cache"
            )
            self.assertNotIn("optional_missing_values", by_symbol[symbol])
        self.assertEqual(result["data_facts"], facts)

        # --- model_signals.json ---
        model_signals = self._read_json(run_dir / "model_signals.json")
        self.assertEqual(set(model_signals), {"stock", "etf", "otc_fund"})
        for asset_type, analysis in model_signals.items():
            self.assertEqual(analysis["asset_type"], asset_type)
            self.assertEqual(
                analysis["method"],
                "trend_return_volatility_drawdown_relative_benchmark_peer_comparison",
            )
            self.assertEqual(analysis["data_end_date"], DATA_END)
        stock_signals = model_signals["stock"]["signals"]
        self.assertEqual(
            [item["symbol"] for item in stock_signals],
            ["600519", "000001", "300750"],
        )
        self.assertEqual([item["qlib_score"] for item in stock_signals], [1.5, 1.2, 0.9])
        self.assertEqual([item["model_rank"] for item in stock_signals], [1, 2, 3])
        self.assertEqual(
            [item["symbol"] for item in model_signals["etf"]["signals"]],
            ["510300", "159915"],
        )
        self.assertEqual(
            [item["symbol"] for item in model_signals["otc_fund"]["signals"]],
            ["110020", "161725"],
        )
        expected_signal_keys = {
            "stock": STOCK_SIGNAL_KEYS, "etf": ETF_SIGNAL_KEYS,
            "otc_fund": FUND_SIGNAL_KEYS,
        }
        for asset_type, analysis in model_signals.items():
            for signal in analysis["signals"]:
                self.assertEqual(set(signal), expected_signal_keys[asset_type])
                self._assert_signal_core(signal, signal["symbol"])
        self.assertEqual(result["model_signals"], model_signals)

        # --- rule_gates.json ---
        rule_gates = self._read_json(run_dir / "rule_gates.json")
        _, stock_weight, path_weight = self._expected_weights()
        self.assertEqual(set(rule_gates), {"stock", "etf", "otc_fund", "portfolio"})

        stock_gate = rule_gates["stock"]
        self.assertEqual(
            set(stock_gate), {"passed", "checks", "model_metrics", "risk"}
        )
        self.assertTrue(stock_gate["passed"])
        self.assertEqual(set(stock_gate["checks"]), STOCK_CHECK_KEYS)
        self.assertTrue(all(stock_gate["checks"].values()))
        self.assertEqual(stock_gate["model_metrics"], STOCK_MODEL_STUB)
        stock_risk = stock_gate["risk"]
        self.assertEqual(set(stock_risk), RISK_RESULT_KEYS)
        self.assertTrue(stock_risk["passed"])
        self.assertEqual(stock_risk["target_weights"], stock_risk["requested_weights"])
        for symbol in STOCKS:
            self.assertAlmostEqual(stock_risk["target_weights"][symbol], stock_weight, places=12)
        self.assertAlmostEqual(
            stock_risk["category_exposure"]["stock"], 3.0 * stock_weight, places=12
        )
        self.assertEqual(stock_risk["category_exposure"]["etf"], 0.0)
        self.assertEqual(stock_risk["category_exposure"]["otc_fund"], 0.0)
        self.assertAlmostEqual(stock_risk["turnover"], 3.0 * stock_weight, places=9)
        self.assertEqual(stock_risk["unknown_symbols"], [])
        self.assertEqual(stock_risk["failure_reasons"], [])

        for asset_type, symbol in (("etf", "510300"), ("otc_fund", "110020")):
            gate = rule_gates[asset_type]
            self.assertEqual(set(gate), {"passed", "checks", "risk"})
            self.assertTrue(gate["passed"])
            self.assertEqual(set(gate["checks"]), PATH_CHECK_KEYS)
            self.assertTrue(all(gate["checks"].values()))
            self.assertEqual(gate["risk"]["target_weights"], {symbol: path_weight})
            self.assertTrue(gate["risk"]["passed"])

        portfolio = rule_gates["portfolio"]
        self.assertEqual(set(portfolio), RISK_RESULT_KEYS)
        self.assertTrue(portfolio["passed"])
        self.assertEqual(set(portfolio["checks"]), PORTFOLIO_CHECK_KEYS)
        self.assertTrue(all(portfolio["checks"].values()))
        self.assertAlmostEqual(
            portfolio["category_exposure"]["stock"], 3.0 * stock_weight, places=12
        )
        self.assertAlmostEqual(portfolio["category_exposure"]["etf"], path_weight, places=12)
        self.assertAlmostEqual(
            portfolio["category_exposure"]["otc_fund"], path_weight, places=12
        )
        self.assertAlmostEqual(portfolio["cash_weight"], 0.8, places=9)
        self.assertAlmostEqual(portfolio["turnover"], 0.2, places=9)
        self.assertEqual(portfolio["unknown_symbols"], [])
        self.assertEqual(result["rule_gates"], rule_gates)

        # --- decision_support.json ---
        decision_support = self._read_json(run_dir / "decision_support.json")
        self._assert_success_decision_support(decision_support)
        self.assertEqual(
            decision_support["portfolio_gate"], rule_gates["portfolio"]
        )
        self.assertEqual(result["decision_support"], decision_support)

        # --- model_metadata.json ---
        model_metadata = self._read_json(run_dir / "model_metadata.json")
        self.assertEqual(
            set(model_metadata),
            {"model_version", "qlib_provider", "requested_mode", "stock_model"},
        )
        self.assertEqual(model_metadata["model_version"], "assistant-1")
        self.assertEqual(model_metadata["requested_mode"], "on_demand")
        self.assertEqual(model_metadata["stock_model"], STOCK_MODEL_STUB)

        # --- run_config.yaml / summary.md / research.log ---
        import yaml

        with (run_dir / "run_config.yaml").open("r", encoding="utf-8") as handle:
            self.assertEqual(yaml.safe_load(handle), config_snapshot(self.config))
        summary = (run_dir / "summary.md").read_text(encoding="utf-8")
        self.assertIn(f"- 运行 ID: `{RUN_ID}`", summary)
        self.assertIn("- 状态: `success`", summary)
        self.assertIn("- 个股模型门槛: `通过`", summary)
        log_text = (run_dir / "logs" / "research.log").read_text(encoding="utf-8")
        self.assertEqual(
            log_text.splitlines(),
            [
                f"started {FIXED_NOW_ISO}",
                "mode=on_demand",
                "asset_types=('stock', 'etf', 'otc_fund')",
                # 2026-09-11 整改同步：regime 检测块（16:56 加入 research.py）
                # 会在 asset_types 之后写 market_regime / retrain_recommended，
                # 成功发布时再追加 regime_state_saved。旧夹具缺少这三行。
                "market_regime=bull_low_vol",
                "retrain_recommended(advisory): 首次训练",
                "analyzed_types=('stock', 'etf', 'otc_fund')",
                "data refresh skipped by configuration or DSH_STOCK_SKIP_FETCH=1",
                # 2026-09-11 整改(P0-3 第一步)：显式暴露各类别的发布就绪度。
                "publication_readiness: ready=['stock', 'etf', 'otc_fund'] blocked=[]",
                "regime_state_saved: bull_low_vol",
            ],
        )

        # --- current.json 全字段 ---
        current = self._read_json(self.results_dir / "current.json")
        self.assertEqual(
            set(current),
            {"run_id", "status", "updated_at", "data_end_date", "result_path",
             "paths", "decision_support", "rule_gates"},
        )
        self.assertEqual(current["run_id"], RUN_ID)
        self.assertEqual(current["status"], "success")
        self.assertEqual(current["updated_at"], FIXED_NOW_ISO)
        self.assertEqual(current["data_end_date"], DATA_END)
        self._assert_same_path(current["result_path"], run_dir)
        self.assertEqual(set(current["paths"]), RUN_FILE_KEYS)
        self.assertEqual(current["decision_support"], decision_support)
        self.assertEqual(current["rule_gates"], rule_gates)

        # --- 通知 ---
        notifications_page = research.list_notifications()
        self.assertEqual(notifications_page["count"], 1)
        record = notifications_page["notifications"][0]
        self.assertEqual(record["notification_id"], result["notification_id"])
        self.assertEqual(record["kind"], "research_success")
        self.assertEqual(record["run_id"], RUN_ID)

        # --- 只读 API 往返 ---
        snapshot = research.get_current_snapshot()
        self.assertEqual(snapshot["run_id"], RUN_ID)
        filtered = research.get_current_snapshot("etf", "SH510300")
        self.assertEqual(filtered["status"], "success")
        self.assertEqual(filtered["requested_asset_type"], "etf")
        self.assertEqual(filtered["requested_symbol"], "510300")
        self.assertEqual(
            filtered["decision_support"], decision_support["asset_paths"]["etf"]
        )
        self.assertEqual(len(filtered["signals"]), 1)
        self.assertEqual(filtered["signals"][0]["symbol"], "510300")
        self.assertEqual(filtered["signals"][0]["asset_type"], "etf")
        missing = research.get_current_snapshot("stock", "600999")
        self.assertEqual(missing["status"], "not_found_or_stale")
        self.assertEqual(missing["signals"], [])
        self.assertNotIn("decision_support", missing)
        with self.assertRaisesRegex(ValueError, "asset_type must be one of"):
            research.get_current_snapshot("bond", None)

        run_status = research.get_run_status(RUN_ID)
        self.assertEqual(
            set(run_status),
            {"run_id", "status", "finished_at", "data_end_date", "result_path",
             "depth"},
        )
        self.assertEqual(run_status["status"], "success")
        self.assertEqual(run_status["finished_at"], FIXED_NOW_ISO)
        self._assert_same_path(run_status["result_path"], run_dir)

    # ---------- 用例 2：P0-3 分类别发布——个股门槛失败，ETF/基金结论仍落盘 ----------

    def test_failed_stock_gate_still_publishes_ready_asset_types(self) -> None:
        sentinel = {
            "run_id": "sentinel-success",
            "status": "success",
            "decision_support": {"target_weights": {"OLD": 0.10}},
        }
        paths.atomic_write_json(self.results_dir / "current.json", sentinel)
        blocked_model = {
            **copy.deepcopy(STOCK_MODEL_STUB),
            "mean_rank_ic": 0.001,
            "test_rank_ic": 0.001,
        }

        with patch.object(
            research, "_run_id", return_value=RUN_ID
        ), patch.object(
            research, "_run_qlib_stock_model", return_value=blocked_model
        ) as model:
            result = research.run_research("on_demand", "all", None, _force_sync=True)

        model.assert_called_once_with(self.config)
        run_dir = self._run_dir()

        # P0-3：个股被拦但 ETF/基金就绪 → 三态状态为 partial，不再是整体 failed。
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["failure_reasons"], ["stock_model_or_risk_gate_failed"])
        self.assertEqual(
            self.heartbeats,
            [("refreshed", "running"), ("scanned", "running"),
             ("signals", "running"), ("gates", "running")],
        )

        rule_gates = self._read_json(run_dir / "rule_gates.json")
        stock_gate = rule_gates["stock"]
        self.assertFalse(stock_gate["passed"])
        self.assertEqual(set(stock_gate["checks"]), STOCK_CHECK_KEYS)
        self.assertFalse(stock_gate["checks"]["mean_rank_ic_threshold"])
        self.assertTrue(stock_gate["checks"]["data_quality_and_model_available"])
        self.assertTrue(stock_gate["checks"]["rank_ic_ir_threshold"])
        self.assertTrue(stock_gate["checks"]["top_bottom_excess_return_threshold"])
        self.assertTrue(stock_gate["checks"]["test_max_drawdown_threshold"])
        self.assertTrue(stock_gate["checks"]["data_quality_passed"])
        self.assertTrue(stock_gate["checks"]["freshness_passed"])
        self.assertTrue(stock_gate["checks"]["refresh_passed"])
        self.assertTrue(stock_gate["checks"]["leakage_check_passed"])
        self.assertTrue(stock_gate["checks"]["full_universe_screen_passed"])
        self.assertTrue(stock_gate["checks"]["positive_signal_passed"])
        self.assertTrue(stock_gate["checks"]["risk_constraints_passed"])
        self.assertEqual(stock_gate["model_metrics"], blocked_model)
        self.assertEqual(stock_gate["risk"]["target_weights"], {})
        self.assertEqual(stock_gate["risk"]["cash_weight"], 1.0)

        _, _, path_weight = self._expected_weights()
        decision_support = self._read_json(run_dir / "decision_support.json")
        self.assertEqual(
            set(decision_support),
            {
                "allocation_basis", "hypothetical_capital", "broker_connected",
                "auto_order", "asset_paths", "target_weights", "cash_weight",
                "portfolio_gate", "publication_blocked", "failure_reasons",
                "publication",
            },
        )
        # P0-3：仍有类别就绪 → 不算被阻断；失败原因完整保留以便追踪。
        self.assertFalse(decision_support["publication_blocked"])
        self.assertEqual(
            decision_support["failure_reasons"], ["stock_model_or_risk_gate_failed"]
        )
        publication = decision_support["publication"]
        self.assertEqual(publication["state"], "partial")
        self.assertTrue(publication["published"])
        self.assertEqual(publication["ready_asset_types"], ["etf", "otc_fund"])
        self.assertEqual(publication["blocked_asset_types"], ["stock"])
        self.assertEqual(
            publication["blocked_reasons"]["stock"], ["mean_rank_ic_threshold"]
        )
        asset_paths = decision_support["asset_paths"]
        self.assertEqual(
            asset_paths["stock"],
            {
                "status": "cash_or_observe",
                "action": "现金",
                "target_weights": {},
                "cash_weight": 1.0,
                "allocation_basis": "基于假设资金/配置的目标权重",
                "hypothetical_capital": 100000.0,
                "reason": "个股模型或风险门槛未通过，不生成个股买入目标权重。",
                "broker_connected": False,
                "auto_order": False,
            },
        )
        self.assertEqual(asset_paths["etf"]["status"], "candidate")
        self.assertEqual(asset_paths["etf"]["target_weights"], {"510300": path_weight})
        self.assertEqual(asset_paths["otc_fund"]["status"], "candidate")
        self.assertEqual(asset_paths["otc_fund"]["target_weights"], {"110020": path_weight})
        self.assertTrue(decision_support["portfolio_gate"]["passed"])
        self.assertAlmostEqual(
            decision_support["target_weights"]["510300"], path_weight, places=12
        )
        self.assertAlmostEqual(
            decision_support["target_weights"]["110020"], path_weight, places=12
        )
        self.assertAlmostEqual(decision_support["cash_weight"], 1.0 - 2.0 * path_weight, places=9)
        self.assertEqual(result["decision_support"], decision_support)

        status = self._read_json(run_dir / "status.json")
        self.assertEqual(status["status"], "partial")
        self.assertEqual(status["failure_reasons"], ["stock_model_or_risk_gate_failed"])
        self.assertEqual(status["finished_at"], FIXED_NOW_ISO)
        manifest = self._read_json(run_dir / "manifest.json")
        self.assertEqual(manifest["status"], "partial")
        self.assertEqual(manifest["failure_reasons"], ["stock_model_or_risk_gate_failed"])
        self.assertEqual(set(manifest["paths"]), RUN_FILE_KEYS)

        # P0-3 的核心：就绪类别的结论要真的落盘，不再被"全有全无"挡在门外。
        current = self._read_json(self.results_dir / "current.json")
        self.assertNotEqual(current, sentinel)
        self.assertEqual(current["run_id"], RUN_ID)
        self.assertEqual(current["status"], "partial")
        published = current["decision_support"]
        self.assertNotIn("OLD", published["target_weights"])
        self.assertAlmostEqual(
            published["target_weights"]["510300"], path_weight, places=12
        )
        self.assertAlmostEqual(
            published["target_weights"]["110020"], path_weight, places=12
        )
        # 被拦类别（stock）在落盘快照里必须零权重。
        self.assertEqual(published["asset_paths"]["stock"]["target_weights"], {})
        self.assertEqual(published["publication"]["blocked_asset_types"], ["stock"])

        notifications_page = research.list_notifications()
        self.assertEqual(notifications_page["count"], 1)
        record = notifications_page["notifications"][0]
        self.assertEqual(record["kind"], "research_partial")
        self.assertEqual(record["run_id"], RUN_ID)
        self.assertEqual(record["notification_id"], result["notification_id"])

        run_status = research.get_run_status(RUN_ID)
        self.assertEqual(run_status["status"], "partial")

        # P0-3：被拦类别经只读 API 查询时必须是 blocked 且**不带权重**——
        # 否则界面会把未发布类别的陈旧结论当成新结论，正是本项要修的故障。
        blocked_view = research.get_current_snapshot("stock", None)
        self.assertEqual(blocked_view["status"], "blocked")
        self.assertNotIn("decision_support", blocked_view)
        self.assertEqual(
            blocked_view["publication"]["blocked_asset_types"], ["stock"]
        )
        # 已就绪类别照常返回结论与信号。
        ready_view = research.get_current_snapshot("etf", "SH510300")
        self.assertEqual(ready_view["status"], "partial")
        self.assertEqual(ready_view["publication"]["state"], "partial")
        self.assertEqual(len(ready_view["signals"]), 1)
        self.assertAlmostEqual(
            ready_view["decision_support"]["target_weights"]["510300"],
            path_weight,
            places=12,
        )


    # ---------- 用例 3：定向研究即使模型通过也不产生权重（海选原则） ----------

    def test_directed_stock_run_withholds_weights_pending_full_screen(self) -> None:
        sentinel = {
            "run_id": "sentinel-success",
            "status": "success",
            "decision_support": {"target_weights": {"OLD": 0.10}},
        }
        paths.atomic_write_json(self.results_dir / "current.json", sentinel)

        with patch.object(
            research, "_run_id", return_value=RUN_ID
        ), patch.object(
            research, "_run_qlib_stock_model",
            return_value=copy.deepcopy(STOCK_MODEL_STUB),
        ):
            result = research.run_research("on_demand", "stock", ["600519"], _force_sync=True)

        run_dir = self._run_dir()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failure_reasons"], ["stock_model_or_risk_gate_failed"])

        rule_gates = self._read_json(run_dir / "rule_gates.json")
        stock_gate = rule_gates["stock"]
        self.assertFalse(stock_gate["passed"])
        self.assertEqual(set(stock_gate["checks"]), STOCK_CHECK_KEYS)
        self.assertFalse(stock_gate["checks"]["full_universe_screen_passed"])
        self.assertTrue(stock_gate["checks"]["mean_rank_ic_threshold"])

        # 信号照常输出：定向研究保留验信号的能力，只是不能变成推荐权重。
        model_signals = self._read_json(run_dir / "model_signals.json")
        stock_signals = model_signals["stock"]["signals"]
        self.assertEqual([signal["symbol"] for signal in stock_signals], ["600519"])
        self.assertAlmostEqual(stock_signals[0]["qlib_score"], 1.5)

        decision_support = self._read_json(run_dir / "decision_support.json")
        stock_decision = decision_support["asset_paths"]["stock"]
        self.assertEqual(stock_decision["status"], "cash_or_observe")
        self.assertEqual(stock_decision["target_weights"], {})
        self.assertEqual(stock_decision["cash_weight"], 1.0)
        self.assertIn("海选", stock_decision["reason"])

        current = self._read_json(self.results_dir / "current.json")
        self.assertEqual(current, sentinel)


if __name__ == "__main__":
    unittest.main()
