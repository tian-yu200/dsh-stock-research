from __future__ import annotations

import copy
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import pandas as pd
import qlib.data

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from assistant import qlib_model
from assistant.config import DEFAULTS


class SingleQueryQlibDataLoaderTests(unittest.TestCase):
    def test_loads_all_groups_with_one_features_query(self) -> None:
        index = pd.MultiIndex.from_product(
            [["000001", "000002"], pd.to_datetime(["2024-01-02", "2024-01-03"])],
            names=["instrument", "datetime"],
        )
        provider = Mock()
        provider.instruments.return_value = ["000001", "000002"]
        provider.features.return_value = pd.DataFrame(
            {
                "feature_expr": [1.0, 2.0, 3.0, 4.0],
                "label_expr": [0.1, 0.2, 0.3, 0.4],
                "$close": [10.0, 11.0, 20.0, 21.0],
            },
            index=index,
        )
        loader = qlib_model._SingleQueryQlibDataLoader(
            {
                "feature": (["feature_expr"], ["FEATURE0"]),
                "label": (["label_expr"], ["LABEL0"]),
                "backtest": (["$close"], ["BACKTEST_CLOSE"]),
            },
            data_api=provider,
        )

        result = loader.load("stocks", "2024-01-02", "2024-01-03")

        provider.instruments.assert_called_once_with("stocks", filter_pipe=None)
        provider.features.assert_called_once_with(
            ["000001", "000002"],
            ["feature_expr", "label_expr", "$close"],
            "2024-01-02",
            "2024-01-03",
            freq="day",
            inst_processors=[],
        )
        self.assertEqual(
            list(result.columns),
            [
                ("feature", "FEATURE0"),
                ("label", "LABEL0"),
                ("backtest", "BACKTEST_CLOSE"),
            ],
        )
        self.assertEqual(result.index.names, ["datetime", "instrument"])
        self.assertEqual(result.loc[(pd.Timestamp("2024-01-02"), "000001"), ("backtest", "BACKTEST_CLOSE")], 10.0)

    def test_supports_optional_backtest_fields_without_changing_model_groups(self) -> None:
        provider = Mock()
        provider.instruments.return_value = ["000001"]
        index = pd.MultiIndex.from_tuples(
            [("000001", pd.Timestamp("2024-01-02"))],
            names=["instrument", "datetime"],
        )
        provider.features.return_value = pd.DataFrame(
            {"feature_expr": [1.0], "$open": [9.0], "$close": [10.0]},
            index=index,
        )
        loader = qlib_model._SingleQueryQlibDataLoader(
            {
                "feature": (["feature_expr"], ["FEATURE0"]),
                "label": ([], []),
                "backtest": (["$open", "$close"], ["BACKTEST_OPEN", "BACKTEST_CLOSE"]),
            },
            data_api=provider,
        )

        result = loader.load("stocks", "2024-01-02", "2024-01-02")

        provider.features.assert_called_once()
        self.assertEqual(list(result["feature"].columns), ["FEATURE0"])
        self.assertEqual(list(result["backtest"].columns), ["BACKTEST_OPEN", "BACKTEST_CLOSE"])


class QlibBacktestReuseTests(unittest.TestCase):
    def test_model_reuses_handler_close_without_second_features_query(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="qlib-model-", dir=ROOT / "tests")
        self.addCleanup(temporary.cleanup)
        temp_root = pathlib.Path(temporary.name)
        provider_dir = temp_root / "provider"
        results_dir = temp_root / "results"
        logs_dir = temp_root / "logs"
        for directory in (provider_dir, results_dir, logs_dir):
            directory.mkdir(parents=True, exist_ok=True)

        config = copy.deepcopy(DEFAULTS)
        config["paths"].update(
            {
                "qlib_provider": str(provider_dir),
                "results_dir": str(results_dir),
                "logs_dir": str(logs_dir),
            }
        )
        config["dataset"]["segments"]["test"] = ["2024-01-02", "2024-01-04"]

        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        instruments = ["000001", "000002"]
        index = pd.MultiIndex.from_product(
            [dates, instruments],
            names=["datetime", "instrument"],
        )
        close_frame = pd.DataFrame(
            {"BACKTEST_CLOSE": [10.0, 20.0, 11.0, 18.0, 12.0, 19.0]},
            index=index,
        )
        handler = Mock()
        handler.fetch.return_value = close_frame
        dataset = Mock(handler=handler)
        prediction_index = pd.MultiIndex.from_product(
            [dates[:2], instruments],
            names=["datetime", "instrument"],
        )
        prediction = pd.Series([2.0, 1.0, 2.1, 1.1], index=prediction_index)
        labels = pd.Series([0.1, 0.0, 0.2, 0.1], index=prediction_index)
        model = Mock()
        model.model.best_iteration = 7
        data_provider = Mock()
        data_provider.calendar.return_value = list(dates)

        with patch.object(qlib_model, "init_qlib"), patch.object(
            qlib_model, "build_dataset", return_value=dataset
        ), patch.object(qlib_model, "train_lgbm", return_value=model), patch.object(
            qlib_model,
            "segment_prediction",
            side_effect=[(prediction.iloc[:0], labels.iloc[:0]), (prediction, labels)],
        ), patch.object(qlib.data.D, "_provider", data_provider):
            result = qlib_model._run_qlib_stock_model(config)

        data_provider.features.assert_not_called()
        handler.fetch.assert_called_once_with(
            selector=slice("2024-01-02", "2024-01-04"),
            level="datetime",
            col_set="backtest",
            data_key="raw",
        )
        self.assertTrue(result["model_available"])
        self.assertFalse(result["cache_hit"])
        self.assertEqual(
            result["cache_miss_reason"],
            "provider_source_fingerprint_missing",
        )


class QlibModelCacheTests(unittest.TestCase):
    def test_cache_hit_filters_scores_and_holds_provider_lock_through_publish(self) -> None:
        with tempfile.TemporaryDirectory(prefix="qlib-cache-", dir=ROOT / "tests") as directory:
            root = pathlib.Path(directory)
            provider_dir = root / "provider"
            results_dir = root / "results"
            logs_dir = root / "logs"
            for path in (provider_dir, results_dir, logs_dir):
                path.mkdir(parents=True, exist_ok=True)

            config = copy.deepcopy(DEFAULTS)
            config["paths"].update(
                {
                    "qlib_provider": str(provider_dir),
                    "results_dir": str(results_dir),
                    "logs_dir": str(logs_dir),
                }
            )
            fingerprint = {"digest": "provider-v1", "files": 2}
            qlib_model.atomic_write_json(
                provider_dir / "manifest.json",
                {"source_fingerprint": fingerprint},
            )
            provider_end = "2024-01-04"
            cache_key, identity = qlib_model._model_cache_identity(
                config,
                provider_fingerprint=fingerprint,
                provider_end=provider_end,
            )
            qlib_model.atomic_write_json(
                qlib_model._model_cache_path(config, cache_key),
                {
                    "schema": qlib_model.MODEL_CACHE_SCHEMA,
                    "cache_key": cache_key,
                    "identity": identity,
                    "result": {
                        "model_available": True,
                        "model_name": "Qlib Alpha158 + LightGBM",
                        "scores": {"002942": 0.8, "600519": 0.5},
                    },
                },
            )

            lock_state = {"active": False}
            events: list[str] = []
            lock = Mock()

            def acquire(*, timeout_seconds: float) -> Mock:
                self.assertGreaterEqual(timeout_seconds, 0.0)
                lock_state["active"] = True
                events.append("lock_acquired")
                return lock

            def release() -> None:
                self.assertTrue(lock_state["active"])
                events.append("lock_released")
                lock_state["active"] = False

            def initialize(provider: pathlib.Path, *, kernels: int) -> None:
                self.assertTrue(lock_state["active"])
                self.assertEqual(provider, provider_dir.resolve())
                self.assertEqual(kernels, 1)
                events.append("initialized")

            def progress(phase: str, **details: object) -> None:
                if phase != "provider_cache_check":
                    self.assertTrue(lock_state["active"])
                events.append(phase)

            lock.acquire.side_effect = acquire
            lock.release.side_effect = release
            data_provider = Mock()
            data_provider.calendar.return_value = list(
                pd.to_datetime(["2024-01-02", provider_end])
            )

            with patch.object(qlib_model, "ProviderLock", return_value=lock), patch.object(
                qlib_model, "init_qlib", side_effect=initialize
            ), patch.object(
                qlib_model, "build_dataset"
            ) as build_dataset, patch.object(qlib.data.D, "_provider", data_provider):
                result = qlib_model._run_qlib_stock_model(
                    config,
                    requested_symbols=["002942"],
                    progress_callback=progress,
                )

        self.assertEqual(result["scores"], {"002942": 0.8})
        self.assertTrue(result["cache_hit"])
        self.assertNotIn("cache_miss_reason", result)
        self.assertEqual(result["model_cache_key"], cache_key)
        build_dataset.assert_not_called()
        lock.acquire.assert_called_once()
        lock.release.assert_called_once_with()
        self.assertFalse(lock_state["active"])
        self.assertLess(events.index("lock_acquired"), events.index("initialized"))
        self.assertLess(events.index("publish"), events.index("lock_released"))


class LabelLeakageCheckTests(unittest.TestCase):
    """测试 _check_label_leakage 标签窗口泄漏检查。"""

    def test_no_leakage_when_label_window_before_valid_start(self) -> None:
        """正常情况：标签窗口完全在 valid 段之前。"""
        calendar = pd.to_datetime(
            [
                "2022-12-26",
                "2022-12-27",
                "2022-12-28",
                "2022-12-29",
                "2022-12-30",
                "2023-01-03",
                "2023-01-04",
            ]
        )
        result = qlib_model._check_label_leakage(
            calendar=calendar,
            fit_end_time="2022-12-28",
            label_horizon=2,
            valid_start="2023-01-03",
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["label_horizon_days"], 2)
        self.assertIsNone(result["reason"])

    def test_leakage_when_label_window_crosses_into_valid(self) -> None:
        """泄漏情况：标签窗口跨越进入 valid 段。"""
        # 构造足够长的日历，确保 fit_end + 20 个交易日能覆盖
        calendar = pd.date_range("2022-12-01", "2023-02-01", freq="B")  # 工作日
        result = qlib_model._check_label_leakage(
            calendar=calendar,
            fit_end_time="2022-12-28",
            label_horizon=20,
            valid_start="2023-01-03",
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["label_horizon_days"], 20)
        self.assertIn("label window crosses into valid segment", result["reason"])

    def test_leakage_when_label_window_exactly_at_valid_start(self) -> None:
        """边界情况：标签窗口末尾恰好是 valid_start。"""
        calendar = pd.to_datetime(
            [
                "2022-12-28",
                "2022-12-29",
                "2022-12-30",
                "2023-01-03",
            ]
        )
        result = qlib_model._check_label_leakage(
            calendar=calendar,
            fit_end_time="2022-12-28",
            label_horizon=3,
            valid_start="2023-01-03",
        )
        # fit_end_time=2022-12-28 (loc=0) + 3个交易日 = loc=3 = 2023-01-03
        # 2023-01-03 >= valid_start，应判定为泄漏
        self.assertFalse(result["passed"])
        self.assertIn("label window crosses into valid segment", result["reason"])

    def test_fail_closed_when_calendar_lookup_fails(self) -> None:
        """异常情况：日历查找失败时 fail-closed。"""
        calendar = pd.to_datetime(["2022-12-28", "2022-12-29"])
        result = qlib_model._check_label_leakage(
            calendar=calendar,
            fit_end_time="2023-01-03",  # 不在日历中
            label_horizon=20,
            valid_start="2023-01-10",
        )
        self.assertFalse(result["passed"])
        self.assertIn("leakage check failed", result["reason"])

    def test_fail_when_label_window_exceeds_calendar(self) -> None:
        """标签窗口超出日历末尾。"""
        calendar = pd.to_datetime(["2022-12-28", "2022-12-29", "2022-12-30"])
        result = qlib_model._check_label_leakage(
            calendar=calendar,
            fit_end_time="2022-12-28",
            label_horizon=100,
            valid_start="2023-01-03",
        )
        self.assertFalse(result["passed"])
        self.assertIn("exceeds calendar length", result["reason"])


if __name__ == "__main__":
    unittest.main()
