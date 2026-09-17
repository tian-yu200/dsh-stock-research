from __future__ import annotations

import copy
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
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

class MultiWindowStabilityTests(unittest.TestCase):
    """多窗口滚动一致性检查：以注入替身控制各窗口 IC，验证判定边界。"""

    @staticmethod
    def _dataset() -> Mock:
        dataset = Mock()
        dataset.segments = {"train": ["2020-01-01", "2022-01-01"], "test": ["2023-01-01", "2024-01-01"]}
        return dataset

    @staticmethod
    def _prediction_with_ic(rank_ic: float, days: int = 4, symbols: int = 4) -> tuple[pd.Series, pd.Series]:
        """构造 rank IC 近似等于 rank_ic 的分数/标签对。

        用完美单调的标签，再按给定 rank_ic 在分数上施加单调扰动；
        这里更简单地直接构造 Spearman 可精确计算的整数序列。
        """
        index = pd.MultiIndex.from_product(
            [pd.to_datetime([f"2024-01-{d:02d}" for d in range(1, days + 1)]), [f"00000{i}" for i in range(symbols)]],
            names=["datetime", "instrument"],
        )
        labels = pd.Series([0.0, 1.0, 2.0, 3.0] * days, index=index)
        if rank_ic >= 1.0:
            scores = [0.0, 1.0, 2.0, 3.0] * days
        elif rank_ic <= -1.0:
            scores = [3.0, 2.0, 1.0, 0.0] * days
        else:
            # 与标签部分同序：两两同序，其余反向，得到稳定的中间相关系数。
            scores = [0.0, 1.0, 3.0, 2.0] * days
        return pd.Series(scores, index=index), labels

    def test_multi_window_stability_marks_stable_when_ics_agree(self) -> None:
        """各窗口 IC 完全一致 → 标准差 0 → 判定稳定。"""
        dataset = self._dataset()
        prediction, labels = self._prediction_with_ic(0.8)
        windows = [("2020-01-01", "2021-01-01"), ("2021-01-01", "2022-01-01"), ("2022-01-01", "2023-01-01")]
        with patch.object(qlib_model, "train_lgbm", return_value=Mock()), patch.object(
            qlib_model, "segment_prediction", return_value=(prediction, labels)
        ):
            result = qlib_model.multi_window_stability_check(dataset, {"seed": 42}, windows)

        self.assertTrue(result["stable"])
        self.assertEqual(result["windows_used"], 3)
        self.assertEqual(result["threshold"], 0.02)
        self.assertAlmostEqual(result["ic_std"], 0.0, places=9)
        self.assertIn("within threshold", result["reason"])

    def test_multi_window_stability_marks_unstable_when_ics_diverge(self) -> None:
        """窗口间 IC 差异大 → 标准差超阈值 → 判定不稳定。"""
        dataset = self._dataset()
        strong, labels = self._prediction_with_ic(1.0)
        weak, _ = self._prediction_with_ic(-1.0)
        windows = [("2020-01-01", "2021-01-01"), ("2021-01-01", "2022-01-01")]
        with patch.object(qlib_model, "train_lgbm", return_value=Mock()), patch.object(
            qlib_model, "segment_prediction", side_effect=[(strong, labels), (weak, labels)]
        ):
            result = qlib_model.multi_window_stability_check(dataset, {"seed": 42}, windows)

        self.assertFalse(result["stable"])
        self.assertGreater(result["ic_std"], 0.02)
        self.assertIn("exceeds threshold", result["reason"])

    def test_multi_window_stability_insufficient_windows_is_not_stable(self) -> None:
        """窗口不足时不得因标准差为 0 而误判稳定。"""
        dataset = self._dataset()
        prediction, labels = self._prediction_with_ic(0.5)
        with patch.object(qlib_model, "train_lgbm", return_value=Mock()), patch.object(
            qlib_model, "segment_prediction", return_value=(prediction, labels)
        ):
            result = qlib_model.multi_window_stability_check(
                dataset, {"seed": 42}, [("2020-01-01", "2021-01-01")]
            )

        self.assertFalse(result["stable"])
        self.assertEqual(result["windows_used"], 1)
        self.assertTrue(np.isnan(result["ic_std"]))
        self.assertIn("insufficient usable windows", result["reason"])

    def test_multi_window_stability_isolates_window_failure(self) -> None:
        """单个窗口抛错时不中断，整体仍给出结论且记录该窗口的错误。"""
        dataset = self._dataset()
        prediction, labels = self._prediction_with_ic(0.6)
        windows = [("2020-01-01", "2021-01-01"), ("2021-01-01", "2022-01-01")]
        with patch.object(qlib_model, "train_lgbm", return_value=Mock()), patch.object(
            qlib_model,
            "segment_prediction",
            side_effect=[RuntimeError("boom"), (prediction, labels)],
        ):
            result = qlib_model.multi_window_stability_check(dataset, {"seed": 42}, windows)

        self.assertEqual(result["windows_requested"], 2)
        self.assertEqual(result["windows_used"], 1)
        self.assertIsNone(result["details"][0]["ic"])
        self.assertIn("RuntimeError: boom", result["details"][0]["error"])

    def test_window_dataset_rewrites_train_segment(self) -> None:
        """窗口裁剪只改 train 段，不动 test 段，也不改原对象。"""
        dataset = self._dataset()
        original = dict(dataset.segments)
        windowed = qlib_model._window_dataset(
            dataset, start_time="2019-01-01", end_time="2021-06-30"
        )

        self.assertEqual(windowed.segments["train"], ["2019-01-01", "2021-06-30"])
        self.assertEqual(windowed.segments["test"], original["test"])
        self.assertEqual(dataset.segments, original)

    def test_run_qlib_stock_model_reports_stability_disabled_by_default(self) -> None:
        """未配置 stability_windows 时不额外训练，结果为显式 disabled。"""
        config = copy.deepcopy(DEFAULTS)
        config["research"].pop("stability_windows", None)

        # 数据集替身必须提供真实 DataFrame 的 backtest 帧：_run_qlib_stock_model
        # 会执行 raw["BACKTEST_CLOSE"].unstack("instrument")，裸 Mock 会在下标处
        # 抛 TypeError。此处对齐既有 _run_qlib_stock_model 测试的构造方式。
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        instruments = ["000001", "000002"]
        backtest_index = pd.MultiIndex.from_product(
            [dates, instruments],
            names=["datetime", "instrument"],
        )
        close_frame = pd.DataFrame(
            {"BACKTEST_CLOSE": [10.0, 20.0, 11.0, 18.0, 12.0, 19.0]},
            index=backtest_index,
        )
        handler = Mock()
        handler.fetch.return_value = close_frame
        dataset = Mock(handler=handler)
        dataset.segments = {
            "train": ["2020-01-01", "2022-01-01"],
            "valid": ["2022-01-01", "2023-01-01"],
            "test": ["2024-01-02", "2024-01-04"],
        }
        config["dataset"]["segments"]["test"] = ["2024-01-02", "2024-01-04"]

        prediction, labels = self._prediction_with_ic(0.5)
        model = Mock()
        model.model.best_iteration = 5
        data_provider = Mock()
        data_provider.calendar.return_value = dates
        with patch.object(qlib_model, "multi_window_stability_check") as stability_mock, patch.object(
            qlib_model, "init_qlib"
        ), patch.object(qlib_model, "build_dataset", return_value=dataset), patch.object(
            qlib_model, "train_lgbm", return_value=model
        ), patch.object(
            qlib_model, "segment_prediction", return_value=(prediction, labels)
        ), patch.object(qlib.data.D, "_provider", data_provider):
            result = qlib_model._run_qlib_stock_model(config)

        stability_mock.assert_not_called()
        self.assertIn("stability", result)
        self.assertIsNone(result["stability"]["stable"])
        self.assertIn("disabled", result["stability"]["reason"])
