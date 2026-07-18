"""Unit tests for the cache-first prediction contract."""

import os
import sys
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def _install_prediction_test_stubs():
    tensorflow_stub = types.ModuleType("tensorflow")
    tensorflow_stub.config = types.SimpleNamespace(
        list_physical_devices=lambda *_args, **_kwargs: [],
        experimental=types.SimpleNamespace(
            set_virtual_device_configuration=lambda *_args, **_kwargs: None,
            VirtualDeviceConfiguration=lambda **_kwargs: object(),
        ),
        set_visible_devices=lambda *_args, **_kwargs: None,
    )

    ml_builder_stub = types.ModuleType("ml_builder")
    for attribute_name in (
        "train_and_validate_models",
        "predict_future_price_changes",
        "analyze_prediction_performance",
        "calculate_predicted_profit",
        "plot_graph",
    ):
        setattr(ml_builder_stub, attribute_name, Mock(name=attribute_name))

    monte_carlo_stub = types.ModuleType("monte_carlo_sim")
    monte_carlo_stub.monte_carlo_analysis = Mock(name="monte_carlo_analysis")

    matplotlib_stub = types.ModuleType("matplotlib")
    matplotlib_stub.use = lambda *_args, **_kwargs: None

    pyplot_stub = types.ModuleType("matplotlib.pyplot")
    for attribute_name in ("figure", "plot", "savefig", "close", "title", "xlabel", "ylabel", "legend", "grid", "tight_layout"):
        setattr(pyplot_stub, attribute_name, Mock(name=f"plt_{attribute_name}"))

    sys.modules.setdefault("tensorflow", tensorflow_stub)
    sys.modules.setdefault("ml_builder", ml_builder_stub)
    sys.modules.setdefault("monte_carlo_sim", monte_carlo_stub)
    sys.modules.setdefault("matplotlib", matplotlib_stub)
    sys.modules.setdefault("matplotlib.pyplot", pyplot_stub)


_install_prediction_test_stubs()

from model_pipeline_preprocessing import PreparedModelingData
from prediction_cache_contract import (
    PredictionCacheInvalidatedError,
    PredictionCacheState,
    PredictionTrainingRequiredError,
    inspect_prediction_cache,
    require_prediction_cache,
)
import price_predictor


class TestPredictionCacheContract(unittest.TestCase):
    def test_inspect_prediction_cache_reports_missing_model_types(self):
        def fake_load_hyperparameters(*, ticker, model_type, max_age_days):
            self.assertEqual(ticker, "AAPL")
            self.assertEqual(max_age_days, 30)
            if model_type in {"rf", "xgb"}:
                return {"loaded": model_type}
            return None

        cache_state = inspect_prediction_cache(
            "AAPL",
            ["rf", "xgb", "ridge", "svr"],
            max_age_days=30,
            load_hyperparameters=fake_load_hyperparameters,
        )

        self.assertEqual(cache_state.cache_status, "training_required")
        self.assertEqual(cache_state.available_model_types, ["rf", "xgb"])
        self.assertEqual(cache_state.missing_model_types, ["ridge", "svr"])

    def test_require_prediction_cache_raises_training_required_error(self):
        with self.assertRaises(PredictionTrainingRequiredError) as context:
            require_prediction_cache(
                "AAPL",
                ["rf", "xgb"],
                max_age_days=30,
                load_hyperparameters=lambda **_kwargs: None,
            )

        self.assertEqual(context.exception.cache_status, "training_required")
        self.assertEqual(context.exception.missing_model_types, ["rf", "xgb"])


class TestPricePredictorCacheStatuses(unittest.TestCase):
    def setUp(self):
        self.raw_df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=260),
                "ticker": ["AAPL"] * 260,
                "close_Price": np.linspace(10.0, 20.0, 260),
                "open_Price": np.linspace(9.5, 19.5, 260),
                "high_Price": np.linspace(10.5, 20.5, 260),
                "low_Price": np.linspace(9.0, 19.0, 260),
            }
        )
        self.prepared = PreparedModelingData(
            stock_data_df=self.raw_df.copy(),
            scaler_x=Mock(name="scaler_x"),
            scaler_y=Mock(name="scaler_y"),
            x_training_dataset_df=pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], columns=["f1", "f2"]),
            x_val_dataset_df=pd.DataFrame([[5.0, 6.0]], columns=["f1", "f2"]),
            x_test_dataset_df=pd.DataFrame([[7.0, 8.0]], columns=["f1", "f2"]),
            x_prediction_dataset_df=pd.DataFrame([[9.0, 10.0]], columns=["f1", "f2"]),
            selected_features_model=Mock(),
            selected_features_list=["f1", "f2"],
            y_train_scaled=np.array([0.1, 0.2]),
            y_val_scaled=np.array([0.3]),
            y_test_scaled=np.array([0.4]),
            y_train_unscaled=np.array([1.1, 1.2]),
            y_val_unscaled=np.array([1.3]),
            y_test_unscaled=np.array([1.4]),
            rows_before_cleaning=260,
            rows_after_cleaning=255,
            min_rows_required=252,
        )

    @patch("price_predictor.db_interactions.import_stock_dataset")
    @patch("price_predictor.prepare_modeling_data")
    @patch(
        "price_predictor.require_prediction_cache",
        side_effect=PredictionTrainingRequiredError(
            ticker="AAPL",
            required_model_types=["rf", "xgb", "ridge", "svr"],
            missing_model_types=["xgb", "svr"],
        ),
    )
    def test_predict_single_stock_fails_fast_when_training_required(
        self,
        _mock_require_cache,
        mock_prepare,
        mock_import_dataset,
    ):
        result = price_predictor.predict_single_stock("AAPL", investment_years=3, time_steps=30)

        self.assertFalse(result["success"])
        self.assertEqual(result["cache_status"], "training_required")
        self.assertEqual(result["missing_model_types"], ["xgb", "svr"])
        mock_import_dataset.assert_not_called()
        mock_prepare.assert_not_called()

    @patch("price_predictor.ml_builder.predict_future_price_changes")
    @patch(
        "price_predictor.ml_builder.train_and_validate_models",
        side_effect=PredictionCacheInvalidatedError(
            ticker="AAPL",
            model_type="rf",
            reason="bad cached params",
            invalidated_count=1,
        ),
    )
    @patch("price_predictor.prepare_modeling_data")
    @patch("price_predictor.db_interactions.import_stock_dataset")
    @patch("price_predictor.require_prediction_cache")
    def test_predict_single_stock_returns_cache_invalidated_status(
        self,
        mock_require_cache,
        mock_import_dataset,
        mock_prepare,
        _mock_train_models,
        mock_predict_future,
    ):
        mock_require_cache.return_value = PredictionCacheState(
            ticker="AAPL",
            required_model_types=["rf", "xgb", "ridge", "svr"],
            available_model_types=["rf", "xgb", "ridge", "svr"],
            missing_model_types=[],
            max_age_days=30,
        )
        mock_import_dataset.return_value = self.raw_df.copy()
        mock_prepare.return_value = self.prepared

        result = price_predictor.predict_single_stock("AAPL", investment_years=3, time_steps=30)

        self.assertFalse(result["success"])
        self.assertEqual(result["cache_status"], "cache_invalidated")
        self.assertEqual(result["failing_model_type"], "rf")
        mock_predict_future.assert_not_called()

    @patch("price_predictor._save_run_summary")
    @patch("price_predictor.predict_single_stock")
    @patch("price_predictor.db_interactions.get_tickers_needing_prediction")
    @patch("price_predictor.get_blacklist_manager")
    @patch("price_predictor.configure_gpu", return_value=False)
    def test_run_predictions_reports_cache_status_counts(
        self,
        _mock_configure_gpu,
        mock_get_blacklist_manager,
        mock_prediction_needs,
        mock_predict_single_stock,
        mock_save_summary,
    ):
        mock_get_blacklist_manager.return_value.get_blacklist.return_value = []
        mock_prediction_needs.return_value = {
            "needs_prediction": ["AAPL", "MSFT"],
            "recently_predicted": [],
        }
        mock_predict_single_stock.side_effect = [
            {"success": True, "execution_time": 1.0, "cache_status": "cache_hit"},
            {
                "success": False,
                "execution_time": 0.5,
                "cache_status": "training_required",
                "error_message": "Missing cache",
            },
        ]

        result = price_predictor.run_predictions(
            max_prediction_age_days=1,
            investment_years=3,
            time_steps=30,
            max_stocks=2,
        )

        self.assertEqual(result["status_counts"], {"cache_hit": 1, "training_required": 1})
        summary_payload = mock_save_summary.call_args.args[0]
        self.assertEqual(summary_payload["status_counts"], {"cache_hit": 1, "training_required": 1})


if __name__ == "__main__":
    unittest.main()