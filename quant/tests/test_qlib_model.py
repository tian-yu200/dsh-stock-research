from __future__ import annotations

import copy
import pathlib
import sys
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
        config = copy.deepcopy(DEFAULTS)
        config["paths"]["logs_dir"] = str(ROOT / "logs")
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


if __name__ == "__main__":
    unittest.main()
