"""Unit tests for the shared trainer/predictor preprocessing contract."""

import os
import sys
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def _install_wiring_test_stubs():
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


_install_wiring_test_stubs()

import model_pipeline_preprocessing
from model_pipeline_preprocessing import InsufficientDataError, PreparedModelingData
from prediction_cache_contract import PredictionCacheState
import model_trainer
import price_predictor


class _IdentityScaler:
    def inverse_transform(self, values):
        return np.asarray(values)


class TestModelPipelinePreprocessing(unittest.TestCase):
    def setUp(self):
        periods = 60
        self.raw_df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=periods),
                "ticker": ["AAPL"] * periods,
                "close_Price": np.linspace(10.0, 20.0, periods),
                "open_Price": np.linspace(9.5, 19.5, periods),
                "high_Price": np.linspace(10.5, 20.5, periods),
                "low_Price": np.linspace(9.0, 19.0, periods),
                "feature_signal": np.linspace(1.0, 3.0, periods),
                "feature_aux": np.linspace(5.0, 7.0, periods),
                "all_nan_feature": [np.nan] * periods,
            }
        )
        self.raw_df.loc[0, "feature_signal"] = np.nan
        self.raw_df.loc[1, "feature_aux"] = np.nan
        self.raw_df.loc[2, "close_Price"] = np.nan

    def test_prepare_modeling_data_applies_shared_cleaning_contract(self):
        captured = {}

        def fake_splitter(cleaned_df, test_size, validation_size):
            captured["cleaned_df"] = cleaned_df.copy()
            captured["test_size"] = test_size
            captured["validation_size"] = validation_size
            scaler = _IdentityScaler()
            return (
                scaler,
                scaler,
                np.array([[0.1, 0.2], [0.3, 0.4]]),
                np.array([[0.5, 0.6]]),
                np.array([[0.7, 0.8]]),
                np.array([0.01, 0.02]),
                np.array([0.03]),
                np.array([0.04]),
                np.array([[0.9, 1.0]]),
            )

        def fake_selector(feature_amount, _x_train, _x_val, _x_test, _y_train, _y_val, _y_test, _prediction_data, stock_df):
            captured["feature_amount"] = feature_amount
            captured["selector_stock_df"] = stock_df.copy()
            return (
                np.array([[10.0, 20.0], [30.0, 40.0]]),
                np.array([[50.0, 60.0]]),
                np.array([[70.0, 80.0]]),
                np.array([[90.0, 100.0]]),
                Mock(),
                ["feature_signal", "feature_aux"],
            )

        prepared = model_pipeline_preprocessing.prepare_modeling_data(
            self.raw_df,
            time_steps=2,
            validation_size=0.2,
            test_size=0.1,
            min_rows_floor=4,
            dataset_splitter=fake_splitter,
            feature_selector=fake_selector,
        )

        self.assertEqual(prepared.rows_before_cleaning, 60)
        self.assertEqual(prepared.rows_after_cleaning, 59)
        self.assertEqual(prepared.min_rows_required, 52)
        self.assertEqual(captured["test_size"], 0.1)
        self.assertEqual(captured["validation_size"], 0.2)
        self.assertNotIn("all_nan_feature", captured["cleaned_df"].columns)
        self.assertFalse(captured["cleaned_df"].isna().any().any())
        self.assertEqual(list(prepared.x_prediction_dataset_df.columns), ["feature_signal", "feature_aux"])
        self.assertEqual(captured["feature_amount"], 1)
        self.assertEqual(len(captured["selector_stock_df"]), 59)

    def test_prepare_modeling_data_uses_shared_minimum_row_rule(self):
        with self.assertRaises(InsufficientDataError) as context:
            model_pipeline_preprocessing.prepare_modeling_data(
                self.raw_df,
                time_steps=40,
                validation_size=0.2,
                test_size=0.1,
                min_rows_floor=4,
            )

        self.assertIn("need >= 90", str(context.exception))

    def test_prepare_modeling_data_caps_feature_count_by_train_samples(self):
        captured = {}

        def fake_splitter(_cleaned_df, _test_size, validation_size):
            self.assertEqual(validation_size, 0.2)
            scaler = _IdentityScaler()
            x_train = np.arange(60, dtype=float).reshape(20, 3)
            x_val = np.arange(12, dtype=float).reshape(4, 3)
            x_test = np.arange(12, dtype=float).reshape(4, 3)
            x_pred = np.arange(6, dtype=float).reshape(2, 3)
            return (
                scaler,
                scaler,
                x_train,
                x_val,
                x_test,
                np.linspace(0.01, 0.20, 20),
                np.linspace(0.21, 0.24, 4),
                np.linspace(0.25, 0.28, 4),
                x_pred,
            )

        def fake_selector(feature_amount, _x_train, _x_val, _x_test, _y_train, _y_val, _y_test, _prediction_data, _stock_df):
            captured["feature_amount"] = feature_amount
            return (
                np.array([[10.0], [20.0]]),
                np.array([[30.0]]),
                np.array([[40.0]]),
                np.array([[50.0]]),
                Mock(),
                ["feature_signal"],
            )

        model_pipeline_preprocessing.prepare_modeling_data(
            self.raw_df,
            time_steps=2,
            validation_size=0.2,
            test_size=0.1,
            min_rows_floor=4,
            dataset_splitter=fake_splitter,
            feature_selector=fake_selector,
        )

        self.assertEqual(captured["feature_amount"], 2)


class TestTrainerPredictorSharedPreprocessing(unittest.TestCase):
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

    @patch("model_trainer.ml_builder.train_and_validate_models", return_value=({}, {}, {}))
    @patch("model_trainer.prepare_modeling_data")
    @patch("model_trainer.db_interactions.import_stock_dataset")
    @patch("model_trainer.validate_data_availability", return_value={"valid": True, "missing_tables": [], "message": "OK"})
    def test_train_single_stock_uses_shared_preprocessing_helper(
        self,
        _mock_validate,
        mock_import_dataset,
        mock_prepare,
        mock_train_models,
    ):
        mock_import_dataset.return_value = self.raw_df.copy()
        mock_prepare.return_value = self.prepared

        result = model_trainer.train_single_stock("AAPL", time_steps=30, use_tcn=False, use_sequence_model=False)

        self.assertTrue(result["success"])
        mock_prepare.assert_called_once()
        self.assertEqual(mock_prepare.call_args.kwargs["time_steps"], 30)
        self.assertEqual(mock_prepare.call_args.kwargs["min_rows_floor"], model_trainer.get_data_config().min_rows_floor)
        self.assertIs(mock_train_models.call_args.kwargs["scaler_y"], self.prepared.scaler_y)
        np.testing.assert_array_equal(mock_train_models.call_args.kwargs["y_train_scaled"], self.prepared.y_train_scaled)

    @patch("price_predictor.db_interactions.export_monte_carlo_results")
    @patch("price_predictor.db_interactions.export_stock_prediction_extended")
    @patch("price_predictor.monte_carlo_sim.monte_carlo_analysis", return_value=(pd.DataFrame({"x": [1]}), pd.DataFrame({"y": [2]})))
    @patch("price_predictor.ml_builder.plot_graph")
    @patch("price_predictor.ml_builder.calculate_predicted_profit")
    @patch("price_predictor.save_prediction_graph")
    @patch("price_predictor.ml_builder.analyze_prediction_performance")
    @patch("price_predictor.ml_builder.predict_future_price_changes", return_value=pd.DataFrame({"close_Price": [10.0, 11.0]}))
    @patch("price_predictor.ml_builder.train_and_validate_models", return_value=({"rf": Mock()}, {}, {}))
    @patch("price_predictor.prepare_modeling_data")
    @patch("price_predictor.db_interactions.import_stock_dataset")
    @patch("price_predictor.require_prediction_cache")
    def test_predict_single_stock_uses_shared_preprocessing_helper(
        self,
        mock_require_cache,
        mock_import_dataset,
        mock_prepare,
        mock_train_models,
        mock_predict,
        _mock_analyze,
        _mock_save_graph,
        _mock_profit,
        _mock_plot,
        _mock_mc,
        _mock_export_predictions,
        _mock_export_mc,
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

        self.assertTrue(result["success"])
        self.assertEqual(result["cache_status"], "cache_hit")
        mock_prepare.assert_called_once()
        self.assertEqual(mock_prepare.call_args.kwargs["time_steps"], 30)
        self.assertEqual(mock_prepare.call_args.kwargs["min_rows_floor"], price_predictor.get_data_config().min_rows_floor)
        self.assertIs(mock_train_models.call_args.kwargs["scaler_y"], self.prepared.scaler_y)
        self.assertTrue(mock_train_models.call_args.kwargs["cache_only"])
        self.assertIs(mock_predict.call_args.kwargs["scaler_x"], self.prepared.scaler_x)
        self.assertEqual(mock_predict.call_args.kwargs["selected_features_list"], self.prepared.selected_features_list)
        self.assertEqual(
            mock_predict.call_args.kwargs["prediction_days"],
            30 * price_predictor.get_pred_config().prediction_days_multiplier,
        )
        self.assertEqual(mock_predict.call_args.kwargs["mc_iterations"], price_predictor.get_pred_config().mc_iterations)
        self.assertTrue(mock_predict.call_args.kwargs["historical_prediction_dataset_df"].equals(self.prepared.x_prediction_dataset_df))
        self.assertEqual(_mock_mc.call_args.args[-1], price_predictor.get_pred_config().sim_amount)

    @patch("price_predictor.db_interactions.export_monte_carlo_results")
    @patch("price_predictor.db_interactions.export_stock_prediction_extended", side_effect=KeyError("db write failed"))
    @patch("price_predictor.monte_carlo_sim.monte_carlo_analysis", return_value=(pd.DataFrame({"x": [1]}), pd.DataFrame({"y": [2]})))
    @patch("price_predictor.ml_builder.plot_graph")
    @patch("price_predictor.ml_builder.calculate_predicted_profit")
    @patch("price_predictor.save_prediction_graph")
    @patch("price_predictor.ml_builder.analyze_prediction_performance")
    @patch("price_predictor.ml_builder.predict_future_price_changes", return_value=pd.DataFrame({"close_Price": [10.0, 11.0]}))
    @patch("price_predictor.ml_builder.train_and_validate_models", return_value=({"rf": Mock()}, {}, {}))
    @patch("price_predictor.prepare_modeling_data")
    @patch("price_predictor.db_interactions.import_stock_dataset")
    @patch("price_predictor.require_prediction_cache")
    def test_predict_single_stock_ignores_normalized_prediction_export_failures(
        self,
        mock_require_cache,
        mock_import_dataset,
        mock_prepare,
        _mock_train_models,
        _mock_predict,
        _mock_analyze,
        _mock_save_graph,
        _mock_profit,
        _mock_plot,
        _mock_mc,
        mock_export_predictions,
        mock_export_mc,
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

        self.assertTrue(result["success"])
        self.assertEqual(result["cache_status"], "cache_hit")
        mock_export_predictions.assert_called_once()
        mock_export_mc.assert_called_once()



if __name__ == "__main__":
    unittest.main()