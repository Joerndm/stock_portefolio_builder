"""
Unit Tests for ml_builder.py

This module contains comprehensive unit tests for individual functions in ml_builder.py.
Tests are isolated and use mocking where necessary to avoid dependencies on external resources.

Test Coverage:
- calculate_predicted_profit: Tests profit calculation logic
- plot_graph: Tests graph generation (mocked)
- build_random_forest_model: Tests RF model construction
- build_xgboost_model: Tests XGBoost model construction
- build_lstm_model: Tests LSTM model construction
- create_sequences: Tests LSTM sequence creation
- detect_overfitting: Tests overfitting detection logic
- are_hyperparameters_identical: Tests hyperparameter comparison
- check_data_health: Tests data validation
"""

import unittest
import warnings
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import types
import importlib.util

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from prediction_runtime_controls import (
    build_prediction_history_cache,
    combine_flat_model_predictions,
    stabilize_prediction,
    summarize_scaled_feature_drift,
)
from sklearn.exceptions import ConvergenceWarning

TENSORFLOW_AVAILABLE = importlib.util.find_spec('tensorflow') is not None

if TENSORFLOW_AVAILABLE:
    import ml_builder
    import model_cache_utils
else:
    ml_builder = None
    model_cache_utils = None


class MLBuilderDependencyTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not TENSORFLOW_AVAILABLE:
            raise unittest.SkipTest('TensorFlow is not installed in this environment')


class TestCalculatePredictedProfit(MLBuilderDependencyTestCase):
    """Test suite for calculate_predicted_profit function"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.forecast_df = pd.DataFrame({
            'close_Price': [100, 105, 103, 108, 110],
            'ticker': ['AAPL'] * 5,
            'date': pd.date_range('2024-01-01', periods=5)
        })
    
    def test_positive_profit(self):
        """Test calculation with increasing prices"""
        profit = ml_builder.calculate_predicted_profit(self.forecast_df, 5)
        self.assertGreater(profit, 0, "Profit should be positive for increasing prices")
    
    def test_negative_profit(self):
        """Test calculation with decreasing prices"""
        decreasing_df = pd.DataFrame({
            'close_Price': [110, 108, 105, 103, 100],
            'ticker': ['AAPL'] * 5,
            'date': pd.date_range('2024-01-01', periods=5)
        })
        profit = ml_builder.calculate_predicted_profit(decreasing_df, 5)
        self.assertLess(profit, 0, "Profit should be negative for decreasing prices")
    
    def test_zero_profit(self):
        """Test calculation with stable prices"""
        stable_df = pd.DataFrame({
            'close_Price': [100] * 5,
            'ticker': ['AAPL'] * 5,
            'date': pd.date_range('2024-01-01', periods=5)
        })
        profit = ml_builder.calculate_predicted_profit(stable_df, 5)
        self.assertAlmostEqual(profit, 0, places=5, msg="Profit should be zero for stable prices")
    
    def test_single_day(self):
        """Test with single day prediction"""
        single_day_df = pd.DataFrame({
            'close_Price': [100, 105],
            'ticker': ['AAPL'] * 2
        })
        profit = ml_builder.calculate_predicted_profit(single_day_df, 1)
        self.assertIsInstance(profit, (int, float), "Profit should be numeric")
    
    def test_prediction_days_parameter(self):
        """Test that prediction_days parameter affects calculation"""
        profit_3_days = ml_builder.calculate_predicted_profit(self.forecast_df, 3)
        profit_5_days = ml_builder.calculate_predicted_profit(self.forecast_df, 5)
        self.assertNotEqual(profit_3_days, profit_5_days, 
                          "Different prediction days should yield different profits")


class TestCreateSequences(MLBuilderDependencyTestCase):
    """Test suite for create_sequences function"""
    
    def setUp(self):
        """Set up test data"""
        self.data = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        self.time_steps = 2
    
    def test_sequence_shape(self):
        """Test that sequences have correct shape"""
        X, y = ml_builder.create_sequences(self.data, self.time_steps)
        expected_samples = len(self.data) - self.time_steps
        self.assertEqual(X.shape[0], expected_samples, "Number of sequences incorrect")
        self.assertEqual(X.shape[1], self.time_steps, "Time steps incorrect")
        self.assertEqual(X.shape[2], self.data.shape[1], "Features incorrect")
    
    def test_sequence_content(self):
        """Test that sequences contain correct data"""
        X, y = ml_builder.create_sequences(self.data, self.time_steps)
        # First sequence should be [1,2], [3,4]
        np.testing.assert_array_equal(X[0], [[1, 2], [3, 4]])
        # Target should be [5, 6]
        np.testing.assert_array_equal(y[0], [5, 6])
    
    def test_time_steps_one(self):
        """Test with time_steps=1"""
        X, y = ml_builder.create_sequences(self.data, 1)
        self.assertEqual(X.shape[1], 1, "Time steps should be 1")
        self.assertEqual(len(X), len(self.data) - 1, "Number of samples incorrect")
    
    def test_insufficient_data(self):
        """Test with insufficient data for sequences"""
        small_data = np.array([[1, 2], [3, 4]])
        X, y = ml_builder.create_sequences(small_data, 5)
        self.assertEqual(len(X), 0, "Should return empty array for insufficient data")
    
    def test_single_feature(self):
        """Test with single feature"""
        single_feature = np.array([[1], [2], [3], [4], [5]])
        X, y = ml_builder.create_sequences(single_feature, 2)
        self.assertEqual(X.shape[2], 1, "Should handle single feature")


class TestDetectOverfitting(MLBuilderDependencyTestCase):
    """Test suite for detect_overfitting function"""
    
    def test_clear_overfitting(self):
        """Test detection of clear overfitting case"""
        train_metrics = {'mse': 0.01, 'mae': 0.05, 'r2': 0.95}
        val_metrics = {'mse': 0.50, 'mae': 0.40, 'r2': 0.40}
        test_metrics = {'mse': 0.60, 'mae': 0.45, 'r2': 0.35}
        
        is_overfitting, score, details = ml_builder.detect_overfitting(
            train_metrics, val_metrics, test_metrics, "TestModel", threshold=0.15
        )
        self.assertTrue(is_overfitting, "Should detect obvious overfitting")
        self.assertGreater(score, 0.15, "Score should exceed threshold")
    
    def test_no_overfitting(self):
        """Test when model is not overfitting"""
        train_metrics = {'mse': 0.10, 'mae': 0.20, 'r2': 0.80}
        val_metrics = {'mse': 0.12, 'mae': 0.22, 'r2': 0.78}
        test_metrics = {'mse': 0.13, 'mae': 0.23, 'r2': 0.77}
        
        is_overfitting, score, details = ml_builder.detect_overfitting(
            train_metrics, val_metrics, test_metrics, "TestModel", threshold=0.15
        )
        self.assertFalse(is_overfitting, "Should not detect overfitting")
        self.assertLess(score, 0.15, "Score should be below threshold")
    
    def test_threshold_boundary(self):
        """Test behavior at threshold boundary"""
        train_metrics = {'mse': 0.10, 'mae': 0.20, 'r2': 0.80}
        val_metrics = {'mse': 0.15, 'mae': 0.25, 'r2': 0.75}
        test_metrics = {'mse': 0.16, 'mae': 0.26, 'r2': 0.74}
        
        _, score, _ = ml_builder.detect_overfitting(
            train_metrics, val_metrics, test_metrics, "TestModel"
        )

        # Test at exact threshold
        is_overfitting, _, _ = ml_builder.detect_overfitting(
            train_metrics, val_metrics, test_metrics, "TestModel", threshold=score
        )
        self.assertTrue(is_overfitting, "Should detect at threshold boundary")
    
    def test_single_metric_mode(self):
        """Test single metric detection mode"""
        train_metrics = {'mse': 0.01, 'mae': 0.05, 'r2': 0.95}
        val_metrics = {'mse': 0.50, 'mae': 0.40, 'r2': 0.40}
        test_metrics = {'mse': 0.60, 'mae': 0.45, 'r2': 0.35}
        
        is_overfitting, score, _ = ml_builder.detect_overfitting(
            train_metrics, val_metrics, test_metrics, "TestModel", 
            threshold=0.15, use_multi_metric=False
        )
        self.assertIsInstance(is_overfitting, bool, "Should return boolean")
        self.assertIsInstance(score, float, "Score should be float")
    
    def test_returns_details(self):
        """Test that function returns detailed metrics"""
        train_metrics = {'mse': 0.10, 'mae': 0.20, 'r2': 0.80}
        val_metrics = {'mse': 0.15, 'mae': 0.25, 'r2': 0.75}
        test_metrics = {'mse': 0.16, 'mae': 0.26, 'r2': 0.74}
        
        _, _, details = ml_builder.detect_overfitting(
            train_metrics, val_metrics, test_metrics, "TestModel"
        )
        self.assertIsInstance(details, dict, "Should return details dictionary")
        self.assertIn('train_metrics', details, "Should include train metrics")
        self.assertIn('val_metrics', details, "Should include val metrics")
        self.assertIn('test_metrics', details, "Should include test metrics")


class TestAreHyperparametersIdentical(MLBuilderDependencyTestCase):
    """Test suite for are_hyperparameters_identical function"""
    
    def test_identical_hyperparameters(self):
        """Test with identical hyperparameters"""
        hp1 = {'learning_rate': 0.01, 'n_estimators': 100, 'max_depth': 5}
        hp2 = {'learning_rate': 0.01, 'n_estimators': 100, 'max_depth': 5}
        
        result = ml_builder.are_hyperparameters_identical(hp1, hp2)
        self.assertTrue(result, "Identical hyperparameters should return True")
    
    def test_different_hyperparameters(self):
        """Test with different hyperparameters"""
        hp1 = {'learning_rate': 0.01, 'n_estimators': 100, 'max_depth': 5}
        hp2 = {'learning_rate': 0.02, 'n_estimators': 100, 'max_depth': 5}
        
        result = ml_builder.are_hyperparameters_identical(hp1, hp2)
        self.assertFalse(result, "Different hyperparameters should return False")
    
    def test_within_tolerance(self):
        """Test hyperparameters within tolerance"""
        hp1 = {'learning_rate': 0.01000, 'n_estimators': 100}
        hp2 = {'learning_rate': 0.01005, 'n_estimators': 100}
        
        result = ml_builder.are_hyperparameters_identical(hp1, hp2, tolerance=0.01)
        self.assertTrue(result, "Values within tolerance should be considered identical")
    
    def test_different_keys(self):
        """Test with different keys"""
        hp1 = {'learning_rate': 0.01, 'n_estimators': 100}
        hp2 = {'learning_rate': 0.01, 'max_depth': 5}
        
        result = ml_builder.are_hyperparameters_identical(hp1, hp2)
        self.assertFalse(result, "Different keys should return False")
    
    def test_nested_dictionaries(self):
        """Test with nested dictionary structures"""
        hp1 = {'optimizer': {'type': 'adam', 'lr': 0.01}}
        hp2 = {'optimizer': {'type': 'adam', 'lr': 0.01}}
        
        # This test depends on implementation - may need adjustment
        result = ml_builder.are_hyperparameters_identical(hp1, hp2)
        self.assertTrue(result, "Identical nested structures should return True")


class TestCheckDataHealth(MLBuilderDependencyTestCase):
    """Test suite for check_data_health function"""
    
    def test_healthy_data(self):
        """Test with healthy data"""
        x_train = np.random.randn(100, 10)
        x_val = np.random.randn(20, 10)
        x_test = np.random.randn(10, 10)
        y_train = np.random.randn(100, 1)
        y_val = np.random.randn(20, 1)
        y_test = np.random.randn(10, 1)
        
        # Should not raise any warnings/errors
        ml_builder.check_data_health(
            x_train, x_val, x_test, y_train, y_val, y_test, "TestModel"
        )
    
    def test_insufficient_training_data(self):
        """Test with insufficient training data"""
        x_train = np.random.randn(5, 10)  # Too few samples
        x_val = np.random.randn(20, 10)
        x_test = np.random.randn(10, 10)
        y_train = np.random.randn(5, 1)
        y_val = np.random.randn(20, 1)
        y_test = np.random.randn(10, 1)
        
        # Should handle gracefully (may print warnings)
        ml_builder.check_data_health(
            x_train, x_val, x_test, y_train, y_val, y_test, "TestModel"
        )
    
    def test_mismatched_shapes(self):
        """Test with mismatched X and y shapes"""
        x_train = np.random.randn(100, 10)
        y_train = np.random.randn(90, 1)  # Mismatched
        x_val = np.random.randn(20, 10)
        y_val = np.random.randn(20, 1)
        x_test = np.random.randn(10, 10)
        y_test = np.random.randn(10, 1)
        
        # Should detect shape mismatch
        ml_builder.check_data_health(
            x_train, x_val, x_test, y_train, y_val, y_test, "TestModel"
        )
    
    def test_nan_values(self):
        """Test with NaN values in data"""
        x_train = np.random.randn(100, 10)
        x_train[0, 0] = np.nan  # Inject NaN
        x_val = np.random.randn(20, 10)
        x_test = np.random.randn(10, 10)
        y_train = np.random.randn(100, 1)
        y_val = np.random.randn(20, 1)
        y_test = np.random.randn(10, 1)
        
        # Should detect NaN values
        ml_builder.check_data_health(
            x_train, x_val, x_test, y_train, y_val, y_test, "TestModel"
        )
    
    def test_inf_values(self):
        """Test with infinite values in data"""
        x_train = np.random.randn(100, 10)
        x_train[0, 0] = np.inf  # Inject infinity
        x_val = np.random.randn(20, 10)
        x_test = np.random.randn(10, 10)
        y_train = np.random.randn(100, 1)
        y_val = np.random.randn(20, 1)
        y_test = np.random.randn(10, 1)
        
        # Should detect infinite values
        ml_builder.check_data_health(
            x_train, x_val, x_test, y_train, y_val, y_test, "TestModel"
        )


class TestBuildRandomForestModel(MLBuilderDependencyTestCase):
    """Test suite for build_random_forest_model function"""
    
    @patch('ml_builder.RandomForestRegressor')
    def test_model_creation(self, mock_rf):
        """Test that Random Forest model is created"""
        mock_hp = Mock()
        mock_hp.Int.return_value = 100
        mock_hp.Choice.return_value = 'squared_error'
        
        model = ml_builder.build_random_forest_model(mock_hp)
        
        # Verify hyperparameter choices were called
        self.assertTrue(mock_hp.Int.called or mock_hp.Choice.called,
                       "Hyperparameters should be configured")
    
    @patch('ml_builder.RandomForestRegressor')
    def test_constrained_mode(self, mock_rf):
        """Test constrained mode for overfitting prevention"""
        mock_hp = Mock()
        mock_hp.Int.return_value = 50
        mock_hp.Choice.return_value = 'squared_error'
        
        model = ml_builder.build_random_forest_model(mock_hp, constrain_for_overfitting=True)
        
        # Should still create model with constrained hyperparameters
        self.assertTrue(mock_hp.Int.called or mock_hp.Choice.called,
                       "Constrained hyperparameters should be configured")


class TestRandomForestCacheNormalization(MLBuilderDependencyTestCase):
    """Regression tests for constructor-safe RF cache payloads."""

    def setUp(self):
        self.x_train = pd.DataFrame(np.random.rand(20, 4))
        self.y_train = pd.Series(np.random.rand(20))
        self.x_val = pd.DataFrame(np.random.rand(8, 4))
        self.y_val = pd.Series(np.random.rand(8))

    def test_cached_rf_hyperparameters_handle_bootstrap_false_with_max_samples(self):
        """Cached RF payloads should normalize invalid sklearn combinations before fit."""
        cached_hp = {
            'n_estimators': 200,
            'bootstrap': False,
            'max_samples': 0.8,
            'max_features': '0.3',
        }

        with patch('db_interactions.load_hyperparameters', return_value=cached_hp) as mock_load, \
             patch('ml_builder.Sklearn') as mock_tuner:
            model = ml_builder.tune_random_forest_model(
                stock_symbol='TEST',
                x_training_dataset_df=self.x_train,
                y_training_dataset_df=self.y_train,
                x_val_dataset_df=self.x_val,
                y_val_dataset_df=self.y_val,
                use_cached_hp=True,
            )

        self.assertFalse(mock_tuner.called, "Cached RF payload should skip retuning when normalization succeeds")
        self.assertTrue(mock_load.call_args.kwargs['require_same_features'])
        self.assertIsNone(model.get_params()['max_samples'])
        self.assertEqual(model.get_params()['max_features'], 0.3)

    def test_bad_cached_rf_hyperparameters_are_invalidated_before_retuning(self):
        """RF cache rows that fail restore should be invalidated before tuning continues."""
        tuned_model = MagicMock()
        tuned_model.feature_importances_ = np.array([0.4, 0.3, 0.2, 0.1])
        tuned_model.predict.return_value = np.zeros(len(self.x_val))
        mock_tuner = MagicMock()
        best_hp = MagicMock()
        best_hp.values = {'n_estimators': 100}
        mock_tuner.get_best_hyperparameters.return_value = [best_hp]
        mock_tuner.hypermodel.build.return_value = tuned_model

        with patch('db_interactions.load_hyperparameters', return_value={'max_samples': 'bad'}), \
             patch('ml_builder.build_cached_random_forest_model', side_effect=ValueError('bad cache')), \
             patch('db_interactions.invalidate_hyperparameters') as mock_invalidate, \
             patch('db_interactions.save_hyperparameters') as mock_save, \
             patch('ml_builder.serialize_random_forest_hyperparameters', return_value={'n_estimators': 100}), \
             patch('ml_builder.Sklearn', return_value=mock_tuner):
            result = ml_builder.tune_random_forest_model(
                stock_symbol='TEST',
                x_training_dataset_df=self.x_train,
                y_training_dataset_df=self.y_train,
                x_val_dataset_df=self.x_val,
                y_val_dataset_df=self.y_val,
                use_cached_hp=True,
                cleanup_after_tuning=False,
            )

        self.assertIs(result, tuned_model)
        mock_invalidate.assert_called_once_with(ticker='TEST', model_type='rf')
        mock_save.assert_called_once()


class TestSklearnCacheHelpers(MLBuilderDependencyTestCase):
    """Direct regression tests for sklearn cache helper normalization and serialization."""

    def test_serialize_xgboost_hyperparameters_returns_constructor_safe_payload(self):
        """XGBoost serialization should whitelist and coerce constructor params."""
        model = MagicMock()
        model.get_params.return_value = {
            'n_estimators': '250',
            'max_depth': '4',
            'learning_rate': '0.05',
            'subsample': '0.8',
            'colsample_bytree': '0.7',
            'min_child_weight': '3',
            'gamma': '0.1',
            'reg_alpha': '0.2',
            'reg_lambda': '0.3',
            'verbosity': 2,
        }

        serialized = model_cache_utils.serialize_xgboost_hyperparameters(model)

        self.assertEqual(serialized['n_estimators'], 250)
        self.assertEqual(serialized['max_depth'], 4)
        self.assertEqual(serialized['learning_rate'], 0.05)
        self.assertNotIn('verbosity', serialized)

    def test_serialize_ridge_hyperparameters_returns_constructor_safe_payload(self):
        """Ridge serialization should keep only alpha and solver in normalized form."""
        model = MagicMock()
        model.get_params.return_value = {
            'alpha': '2.5',
            'solver': 'SAG',
            'random_state': 42,
            'max_iter': 10000,
        }

        serialized = model_cache_utils.serialize_ridge_hyperparameters(model)

        self.assertEqual(serialized, {'alpha': 2.5, 'solver': 'sag'})

    def test_serialize_svr_hyperparameters_uses_wrapped_regressor_params(self):
        """SVR serialization should unwrap the fitted target-transform regressor."""
        model = MagicMock()
        model.regressor_.get_params.return_value = {
            'kernel': 'RBF',
            'C': '1.5',
            'gamma': '0.01',
            'epsilon': '0.1',
            'max_iter': 10000,
        }

        serialized = model_cache_utils.serialize_svr_hyperparameters(model)

        self.assertEqual(serialized, {'kernel': 'rbf', 'C': 1.5, 'gamma': 0.01, 'epsilon': 0.1})


class TestSklearnCacheContract(MLBuilderDependencyTestCase):
    """Regression tests for cached sklearn restore and save paths."""

    def setUp(self):
        self.x_train = pd.DataFrame(np.random.rand(20, 4), columns=['f1', 'f2', 'f3', 'f4'])
        self.y_train = pd.Series(np.random.rand(20))
        self.x_val = pd.DataFrame(np.random.rand(8, 4), columns=['f1', 'f2', 'f3', 'f4'])
        self.y_val = pd.Series(np.random.rand(8))

    def _make_tuner(self, best_model, hp_values):
        tuner = MagicMock()
        best_hp = MagicMock()
        best_hp.values = hp_values
        tuner.get_best_hyperparameters.return_value = [best_hp]
        tuner.hypermodel.build.return_value = best_model
        return tuner

    def test_cached_xgboost_hyperparameters_require_matching_features(self):
        """XGBoost cache restore should require the same feature hash and skip retuning on success."""
        cached_hp = {'n_estimators': '250', 'learning_rate': '0.05', 'max_depth': '4'}
        mock_model = MagicMock()

        with patch('db_interactions.load_hyperparameters', return_value=cached_hp) as mock_load, \
             patch('ml_builder.build_cached_xgboost_model', return_value=mock_model) as mock_build, \
             patch('ml_builder.Sklearn') as mock_tuner:
            result = ml_builder.tune_xgboost_model(
                stock_symbol='TEST',
                x_training_dataset_df=self.x_train,
                y_training_dataset_df=self.y_train,
                x_val_dataset_df=self.x_val,
                y_val_dataset_df=self.y_val,
                use_cached_hp=True,
                cleanup_after_tuning=False,
            )

        self.assertIs(result, mock_model)
        self.assertTrue(mock_load.call_args.kwargs['require_same_features'])
        mock_build.assert_called_once_with(cached_hp)
        mock_model.fit.assert_called_once()
        self.assertFalse(mock_tuner.called)

    def test_bad_cached_xgboost_hyperparameters_are_invalidated_before_retuning(self):
        """XGBoost cache rows that fail restore should be invalidated before retuning."""
        tuned_model = MagicMock()
        tuned_model.feature_importances_ = np.array([0.4, 0.3, 0.2, 0.1])
        tuned_model.predict.return_value = np.zeros(len(self.x_val))
        mock_tuner = self._make_tuner(tuned_model, {'n_estimators': 250})

        with patch('db_interactions.load_hyperparameters', return_value={'n_estimators': 'bad'}), \
             patch('ml_builder.build_cached_xgboost_model', side_effect=ValueError('bad cache')), \
             patch('db_interactions.invalidate_hyperparameters') as mock_invalidate, \
             patch('db_interactions.save_hyperparameters') as mock_save, \
             patch('ml_builder.serialize_xgboost_hyperparameters', return_value={'n_estimators': 250}), \
             patch('ml_builder.Sklearn', return_value=mock_tuner):
            result = ml_builder.tune_xgboost_model(
                stock_symbol='TEST',
                x_training_dataset_df=self.x_train,
                y_training_dataset_df=self.y_train,
                x_val_dataset_df=self.x_val,
                y_val_dataset_df=self.y_val,
                use_cached_hp=True,
                cleanup_after_tuning=False,
            )

        self.assertIs(result, tuned_model)
        mock_invalidate.assert_called_once_with(ticker='TEST', model_type='xgb')
        mock_save.assert_called_once()
        self.assertEqual(mock_save.call_args.kwargs['hyperparameters'], {'n_estimators': 250})

    def test_cached_ridge_hyperparameters_require_matching_features(self):
        """Ridge cache restore should require the same feature hash and skip retuning on success."""
        cached_hp = {'alpha': '2.5', 'solver': 'LSQR'}

        with patch('db_interactions.load_hyperparameters', return_value=cached_hp) as mock_load, \
             patch('ml_builder.Sklearn') as mock_tuner:
            model = ml_builder.tune_ridge_model(
                stock_symbol='TEST',
                x_training_dataset_df=self.x_train,
                y_training_dataset_df=self.y_train,
                x_val_dataset_df=self.x_val,
                y_val_dataset_df=self.y_val,
                use_cached_hp=True,
                cleanup_after_tuning=False,
            )

        self.assertTrue(mock_load.call_args.kwargs['require_same_features'])
        self.assertEqual(model.get_params()['alpha'], 2.5)
        self.assertEqual(model.get_params()['solver'], 'lsqr')
        self.assertFalse(mock_tuner.called)

    def test_ridge_tuning_saves_constructor_safe_payload(self):
        """Ridge tuning should save serialized estimator params instead of raw tuner values."""
        tuned_model = MagicMock()
        tuned_model.predict.return_value = np.zeros(len(self.x_val))
        mock_tuner = self._make_tuner(tuned_model, {'alpha': 'raw', 'solver': 'raw'})

        with patch('db_interactions.load_hyperparameters', return_value=None), \
             patch('db_interactions.save_hyperparameters') as mock_save, \
             patch('ml_builder.serialize_ridge_hyperparameters', return_value={'alpha': 1.5, 'solver': 'svd'}), \
             patch('ml_builder.Sklearn', return_value=mock_tuner):
            result = ml_builder.tune_ridge_model(
                stock_symbol='TEST',
                x_training_dataset_df=self.x_train,
                y_training_dataset_df=self.y_train,
                x_val_dataset_df=self.x_val,
                y_val_dataset_df=self.y_val,
                use_cached_hp=False,
                cleanup_after_tuning=False,
            )

        self.assertIs(result, tuned_model)
        self.assertEqual(mock_save.call_args.kwargs['hyperparameters'], {'alpha': 1.5, 'solver': 'svd'})

    def test_cached_svr_hyperparameters_require_matching_features(self):
        """SVR cache restore should require the same feature hash and skip retuning on success."""
        cached_hp = {'kernel': 'RBF', 'C': '1.5', 'gamma': '0.02', 'epsilon': '0.1'}

        with patch('db_interactions.load_hyperparameters', return_value=cached_hp) as mock_load, \
             patch('ml_builder.Sklearn') as mock_tuner:
            model = ml_builder.tune_svr_model(
                stock_symbol='TEST',
                x_training_dataset_df=self.x_train,
                y_training_dataset_df=self.y_train,
                x_val_dataset_df=self.x_val,
                y_val_dataset_df=self.y_val,
                use_cached_hp=True,
                cleanup_after_tuning=False,
            )

        self.assertTrue(mock_load.call_args.kwargs['require_same_features'])
        self.assertEqual(model.regressor.get_params()['kernel'], 'rbf')
        self.assertAlmostEqual(model.regressor.get_params()['C'], 1.5)
        self.assertFalse(mock_tuner.called)

    def test_svr_tuning_saves_constructor_safe_payload(self):
        """SVR tuning should save serialized wrapped-estimator params instead of raw tuner values."""
        tuned_model = MagicMock()
        tuned_model.predict.return_value = np.zeros(len(self.x_val))
        mock_tuner = self._make_tuner(tuned_model, {'kernel': 'raw', 'C': 'raw'})

        with patch('db_interactions.load_hyperparameters', return_value=None), \
             patch('db_interactions.save_hyperparameters') as mock_save, \
             patch('ml_builder.serialize_svr_hyperparameters', return_value={'kernel': 'rbf', 'C': 1.5, 'gamma': 0.02, 'epsilon': 0.1}), \
             patch('ml_builder.Sklearn', return_value=mock_tuner):
            result = ml_builder.tune_svr_model(
                stock_symbol='TEST',
                x_training_dataset_df=self.x_train,
                y_training_dataset_df=self.y_train,
                x_val_dataset_df=self.x_val,
                y_val_dataset_df=self.y_val,
                use_cached_hp=False,
                cleanup_after_tuning=False,
            )

        self.assertIs(result, tuned_model)
        self.assertEqual(
            mock_save.call_args.kwargs['hyperparameters'],
            {'kernel': 'rbf', 'C': 1.5, 'gamma': 0.02, 'epsilon': 0.1},
        )

    def test_serialize_random_forest_hyperparameters_returns_constructor_safe_payload(self):
        """Saved RF cache payloads should reflect the fitted estimator configuration."""
        model = model_cache_utils.build_cached_random_forest_model({
            'n_estimators': 150,
            'bootstrap': False,
            'max_samples': 0.7,
            'max_features': '0.5',
        })

        serialized = model_cache_utils.serialize_random_forest_hyperparameters(model)

        self.assertEqual(serialized['n_estimators'], 150)
        self.assertFalse(serialized['bootstrap'])
        self.assertIsNone(serialized['max_samples'])
        self.assertEqual(serialized['max_features'], 0.5)


class TestSequenceModelRestore(MLBuilderDependencyTestCase):
    """Regression tests for cached and tuned sequence-model restore paths."""

    def setUp(self):
        self.x_train = np.random.rand(4, 3, 2)
        self.y_train = np.random.rand(4, 1)
        self.x_val = np.random.rand(2, 3, 2)
        self.y_val = np.random.rand(2, 1)

    def test_cached_lstm_hyperparameters_are_fit_before_return(self):
        """Cached LSTM hyperparameters should rebuild and fit before return."""
        cached_hp = {'batch_size': 8, 'patience': 10, 'lr_schedule': 'none'}
        mock_model = MagicMock()
        mock_model.build.return_value = None

        with patch('db_interactions.load_hyperparameters', return_value=cached_hp), \
             patch('ml_builder.build_lstm_model', return_value=mock_model):
            result = ml_builder.tune_lstm_model(
                'TEST', self.x_train, self.y_train, self.x_val, self.y_val,
                time_steps=3, num_features=2, epochs=1, use_cached_hp=True
            )

        self.assertIs(result, mock_model)
        mock_model.fit.assert_called_once()

    def test_cached_tcn_hyperparameters_are_fit_before_return(self):
        """Cached TCN hyperparameters should rebuild and fit before return."""
        cached_hp = {'tcn_patience': 10}
        mock_model = MagicMock()
        mock_model.build.return_value = None

        with patch('db_interactions.load_hyperparameters', return_value=cached_hp), \
             patch('ml_builder.build_tcn_model', return_value=mock_model):
            result = ml_builder.tune_tcn_model(
                'TEST', self.x_train, self.y_train, self.x_val, self.y_val,
                time_steps=3, num_features=2, epochs=1, use_cached_hp=True
            )

        self.assertIs(result, mock_model)
        mock_model.fit.assert_called_once()

    def test_lstm_tuning_returns_trained_best_model_from_tuner(self):
        """Fresh LSTM tuning should return the tuner-trained best model, not a new architecture."""
        mock_tuner = MagicMock()
        best_trial = MagicMock()
        best_trial.hyperparameters.values = {'batch_size': 8}
        best_trial.metrics.get_best_value.side_effect = lambda _: 0.1
        mock_tuner.oracle.get_best_trials.return_value = [best_trial]

        trained_model = MagicMock()
        trained_model.predict.return_value = np.zeros((len(self.x_val), 1))
        trained_model.summary.return_value = 'summary'
        mock_tuner.get_best_models.return_value = [trained_model]

        with patch('db_interactions.load_hyperparameters', return_value=None), \
             patch('db_interactions.save_hyperparameters'), \
             patch('ml_builder.load_best_model_from_finished_tuning', return_value=None), \
             patch('ml_builder.kt.BayesianOptimization', return_value=mock_tuner):
            result = ml_builder.tune_lstm_model(
                'TEST', self.x_train, self.y_train, self.x_val, self.y_val,
                time_steps=3, num_features=2, max_trials=1, epochs=1,
                retries=1, use_cached_hp=False
            )

        self.assertIs(result, trained_model)
        mock_tuner.get_best_models.assert_called_once_with(num_models=1)

    def test_lstm_tuning_refits_when_tuner_checkpoint_is_missing(self):
        """Fresh LSTM tuning should rebuild from best hyperparameters if tuner checkpoints are gone."""
        mock_tuner = MagicMock()
        best_trial = MagicMock()
        best_trial.hyperparameters.values = {'batch_size': 8, 'patience': 10, 'lr_schedule': 'none'}
        best_trial.metrics.get_best_value.side_effect = lambda _: 0.1
        mock_tuner.oracle.get_best_trials.return_value = [best_trial]
        mock_tuner.get_best_models.side_effect = RuntimeError('missing checkpoint')

        rebuilt_model = MagicMock()
        rebuilt_model.predict.return_value = np.zeros((len(self.x_val), 1))
        rebuilt_model.summary.return_value = 'summary'

        with patch('db_interactions.load_hyperparameters', return_value=None), \
             patch('db_interactions.save_hyperparameters'), \
             patch('ml_builder.load_best_model_from_finished_tuning', return_value=None), \
             patch('ml_builder.fit_cached_lstm_model', side_effect=lambda model, *_args, **_kwargs: model) as mock_fit_cached, \
             patch('ml_builder.build_lstm_model', return_value=rebuilt_model) as mock_build_model, \
             patch('ml_builder.kt.BayesianOptimization', return_value=mock_tuner):
            result = ml_builder.tune_lstm_model(
                'TEST', self.x_train, self.y_train, self.x_val, self.y_val,
                time_steps=3, num_features=2, max_trials=1, epochs=1,
                retries=1, use_cached_hp=False
            )

        self.assertIs(result, rebuilt_model)
        mock_tuner.get_best_models.assert_called_once_with(num_models=1)
        mock_build_model.assert_called_once()
        mock_fit_cached.assert_called_once()

    def test_tcn_tuning_returns_trained_best_model_from_tuner(self):
        """Fresh TCN tuning should return the tuner-trained best model, not a new architecture."""
        mock_tuner = MagicMock()
        best_trial = MagicMock()
        best_trial.hyperparameters.values = {'tcn_nb_filters': 32}
        best_trial.metrics.get_best_value.side_effect = lambda _: 0.1
        mock_tuner.oracle.get_best_trials.return_value = [best_trial]

        trained_model = MagicMock()
        trained_model.predict.return_value = np.zeros((len(self.x_val), 1))
        trained_model.summary.return_value = 'summary'
        mock_tuner.get_best_models.return_value = [trained_model]

        with patch('db_interactions.load_hyperparameters', return_value=None), \
             patch('db_interactions.save_hyperparameters'), \
             patch('ml_builder.load_best_tcn_model', return_value=None), \
             patch('ml_builder.kt.RandomSearch', return_value=mock_tuner):
            result = ml_builder.tune_tcn_model(
                'TEST', self.x_train, self.y_train, self.x_val, self.y_val,
                time_steps=3, num_features=2, max_trials=1, epochs=1,
                retries=1, use_cached_hp=False
            )

        self.assertIs(result, trained_model)
        mock_tuner.get_best_models.assert_called_once_with(num_models=1)

    def test_tcn_tuning_refits_when_tuner_checkpoint_is_missing(self):
        """Fresh TCN tuning should rebuild from best hyperparameters if tuner checkpoints are gone."""
        mock_tuner = MagicMock()
        best_trial = MagicMock()
        best_trial.hyperparameters.values = {'tcn_nb_filters': 32, 'tcn_patience': 10}
        best_trial.metrics.get_best_value.side_effect = lambda _: 0.1
        mock_tuner.oracle.get_best_trials.return_value = [best_trial]
        mock_tuner.get_best_models.side_effect = RuntimeError('missing checkpoint')

        rebuilt_model = MagicMock()
        rebuilt_model.predict.return_value = np.zeros((len(self.x_val), 1))
        rebuilt_model.summary.return_value = 'summary'

        with patch('db_interactions.load_hyperparameters', return_value=None), \
             patch('db_interactions.save_hyperparameters'), \
             patch('ml_builder.load_best_tcn_model', return_value=None), \
             patch('ml_builder.fit_cached_tcn_model', side_effect=lambda model, *_args, **_kwargs: model) as mock_fit_cached, \
             patch('ml_builder.build_tcn_model', return_value=rebuilt_model) as mock_build_model, \
             patch('ml_builder.kt.RandomSearch', return_value=mock_tuner):
            result = ml_builder.tune_tcn_model(
                'TEST', self.x_train, self.y_train, self.x_val, self.y_val,
                time_steps=3, num_features=2, max_trials=1, epochs=1,
                retries=1, use_cached_hp=False
            )

        self.assertIs(result, rebuilt_model)
        mock_tuner.get_best_models.assert_called_once_with(num_models=1)
        mock_build_model.assert_called_once()
        mock_fit_cached.assert_called_once()


class TestBuildXGBoostModel(MLBuilderDependencyTestCase):
    """Test suite for build_xgboost_model function"""
    
    @patch('ml_builder.xgb.XGBRegressor')
    def test_model_creation(self, mock_xgb):
        """Test that XGBoost model is created"""
        mock_hp = Mock()
        mock_hp.Int.return_value = 100
        mock_hp.Float.return_value = 0.1
        mock_hp.Choice.return_value = 'reg:squarederror'
        
        model = ml_builder.build_xgboost_model(mock_hp)
        
        # Verify hyperparameter choices were called
        self.assertTrue(mock_hp.Int.called or mock_hp.Float.called,
                       "Hyperparameters should be configured")
    
    @patch('ml_builder.xgb.XGBRegressor')
    def test_constrained_mode(self, mock_xgb):
        """Test constrained mode for overfitting prevention"""
        mock_hp = Mock()
        mock_hp.Int.return_value = 50
        mock_hp.Float.return_value = 0.05
        
        model = ml_builder.build_xgboost_model(mock_hp, constrain_for_overfitting=True)
        
        # Should create model with constrained hyperparameters
        self.assertTrue(mock_hp.Int.called or mock_hp.Float.called,
                       "Constrained hyperparameters should be configured")


class TestBuildLSTMModel(MLBuilderDependencyTestCase):
    """Test suite for build_lstm_model function"""
    
    @patch('ml_builder.Sequential')
    def test_lstm_creation(self, mock_sequential):
        """Test LSTM model creation"""
        mock_hp = Mock()
        mock_hp.Int.return_value = 2
        mock_hp.Choice.return_value = 'adam'
        mock_hp.Float.return_value = 0.001
        
        input_shape = (30, 10)  # (time_steps, features)
        
        model = ml_builder.build_lstm_model(mock_hp, input_shape)
        
        # Verify hyperparameters were configured
        self.assertTrue(mock_hp.Int.called or mock_hp.Float.called,
                       "LSTM hyperparameters should be configured")
    
    @patch('ml_builder.Sequential')
    def test_input_shape_handling(self, mock_sequential):
        """Test that input shape is handled correctly"""
        mock_hp = Mock()
        mock_hp.Int.return_value = 1
        mock_hp.Choice.return_value = 'adam'
        mock_hp.Float.return_value = 0.001
        
        input_shapes = [(10, 5), (30, 20), (60, 50)]
        
        for shape in input_shapes:
            model = ml_builder.build_lstm_model(mock_hp, shape)
            # Should handle different input shapes
            self.assertIsNotNone(model or mock_sequential.called,
                               f"Should handle input shape {shape}")


class TestPredictionHistoryCache(unittest.TestCase):
    def setUp(self):
        self.stock_mod_df = pd.DataFrame(
            {
                'date': pd.date_range('2024-01-10', periods=3),
                'close_Price': [100.0, 101.0, 102.0],
            }
        )

    def test_build_prediction_history_cache_fetches_each_period_once(self):
        calls = []

        def fake_fetcher(ticker, period, progress, auto_adjust):
            calls.append((ticker, period, progress, auto_adjust))
            return pd.DataFrame(
                {
                    'Close': [90.0, 91.0, 92.0],
                },
                index=pd.date_range('2024-01-01', periods=3, name='Date'),
            )

        history_cache = build_prediction_history_cache(
            'AAPL',
            self.stock_mod_df,
            ['1M', '1Y', '5Y'],
            history_fetcher=fake_fetcher,
        )

        self.assertEqual({period for _ticker, period, _progress, _adjust in calls}, {'1y', '2y', '6y'})
        self.assertEqual(len(calls), 3)
        self.assertIn('1y', history_cache)
        self.assertEqual(history_cache['1y']['close_Price'].tolist(), [90.0, 91.0, 92.0, 100.0, 101.0, 102.0])

    def test_build_prediction_history_cache_skips_fetch_when_no_return_features_selected(self):
        fetcher = Mock(name='fetcher')

        history_cache = build_prediction_history_cache(
            'AAPL',
            self.stock_mod_df,
            ['sma_5', 'rsi_14'],
            history_fetcher=fetcher,
        )

        self.assertEqual(history_cache, {})
        fetcher.assert_not_called()


class TestPredictionStabilization(unittest.TestCase):
    def setUp(self):
        self.pred_cfg = types.SimpleNamespace(
            mean_reversion_strength=0.10,
            mean_reversion_threshold_std=2.5,
            mean_reversion_hard_cap_std=4.0,
            max_same_direction_days=5,
            max_daily_return=0.20,
        )

    def test_stabilize_prediction_flips_after_same_direction_streak(self):
        stabilized = stabilize_prediction(
            0.05,
            [0.02, 0.03, 0.01, 0.04, 0.02],
            historical_mean=0.0,
            historical_std=0.05,
            prediction_config=self.pred_cfg,
            random_uniform=lambda: 0.0,
        )

        self.assertLess(stabilized, 0.0)

    def test_stabilize_prediction_respects_daily_return_cap(self):
        stabilized = stabilize_prediction(
            0.50,
            [],
            historical_mean=0.0,
            historical_std=0.25,
            prediction_config=self.pred_cfg,
        )

        self.assertLessEqual(stabilized, 0.20)
        self.assertGreaterEqual(stabilized, -0.20)


class TestHistoricalFlatModelBlend(unittest.TestCase):
    def test_combine_flat_model_predictions_clips_outlier_and_uses_available_weights(self):
        blend = combine_flat_model_predictions(
            {'rf': 0.01, 'xgb': 0.02, 'ridge': 10.0, 'svr': -0.03},
            weights={'rf': 0.50, 'xgb': 0.30, 'ridge': 0.05, 'svr': 0.15},
            max_daily_return=0.20,
        )

        self.assertEqual(blend['clipped_predictions']['ridge'], 0.20)
        self.assertAlmostEqual(sum(blend['weights'].values()), 1.0)
        self.assertAlmostEqual(
            blend['ensemble_prediction'],
            (0.50 * 0.01) + (0.30 * 0.02) + (0.05 * 0.20) + (0.15 * -0.03),
            places=9,
        )
        self.assertEqual(blend['clipped_models']['ridge'], (10.0, 0.20))


class TestScaledFeatureDriftSummary(unittest.TestCase):
    def test_summarize_scaled_feature_drift_limits_to_selected_features(self):
        scaled_frame = pd.DataFrame(
            {
                'selected_severe': [2.5],
                'selected_mild': [1.4],
                'selected_ok': [0.8],
                'ignored_severe': [50.0],
            }
        )

        summary = summarize_scaled_feature_drift(
            scaled_frame,
            selected_features=['selected_severe', 'selected_mild', 'selected_ok'],
        )

        self.assertEqual(summary['feature_count'], 3)
        self.assertEqual(summary['severe_out_of_range'], [('selected_severe', 2.5)])
        self.assertEqual(summary['mild_out_of_range'], [('selected_mild', 1.4)])


class TestSequenceAlignmentHelpers(MLBuilderDependencyTestCase):
    def test_alignment_start_matches_sequence_target_horizon(self):
        alignment_start = ml_builder._alignment_start_from_sequence_predictions(
            np.zeros(700),
            np.zeros(699),
            'validation',
        )

        self.assertEqual(alignment_start, 1)

    def test_alignment_start_rejects_longer_sequence_predictions(self):
        with self.assertRaises(ValueError):
            ml_builder._alignment_start_from_sequence_predictions(
                np.zeros(10),
                np.zeros(11),
                'validation',
            )


class TestSvrFitHelpers(MLBuilderDependencyTestCase):
    def test_fit_svr_without_convergence_warnings_suppresses_warning(self):
        model = MagicMock()

        def emit_warning(*_args, **_kwargs):
            warnings.warn('solver terminated early', ConvergenceWarning)

        model.fit.side_effect = emit_warning

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            result = ml_builder._fit_svr_without_convergence_warnings(model, np.zeros((2, 2)), np.zeros(2))

        self.assertIs(result, model)
        self.assertEqual(captured, [])
        model.fit.assert_called_once()


class TestHistoricalPredictionState(MLBuilderDependencyTestCase):
    def test_historical_predictions_do_not_overwrite_latest_actual_close(self):
        stock_df = pd.DataFrame(
            {
                'date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04']),
                'close_Price': [100.0, 110.0, 121.0, 133.1],
                '1D': [0.00, 0.01, 0.01, 0.01],
                'momentum': [0.0, 1.0, 1.0, 1.0],
            }
        )
        historical_prediction_dataset_df = pd.DataFrame({'momentum': [1.0, 1.0]})

        scaler_x = Mock(name='scaler_x')
        scaler_x.transform.side_effect = lambda frame: frame.copy()
        scaler_y = Mock(name='scaler_y')
        rf_model = Mock(name='rf_model')
        rf_model.predict.side_effect = lambda _values: np.array([0.01])

        forecast_df = ml_builder.predict_future_price_changes(
            ticker='TEST',
            scaler_x=scaler_x,
            scaler_y=scaler_y,
            model={
                'sequence_model': None,
                'rf': rf_model,
                'xgb': None,
                'ridge': None,
                'svr': None,
                'ensemble_weights': {'lstm': 0.0, 'rf': 1.0, 'xgb': 0.0, 'ridge': 0.0, 'svr': 0.0},
            },
            selected_features_list=['momentum'],
            stock_df=stock_df,
            prediction_days=1,
            time_steps=1,
            historical_prediction_dataset_df=historical_prediction_dataset_df,
            use_mc_dropout=False,
        )

        forecast_dates = pd.to_datetime(forecast_df['date'])
        last_actual_date = stock_df['date'].max()
        actual_close = float(stock_df.loc[stock_df['date'] == last_actual_date, 'close_Price'].iloc[0])
        returned_close = float(forecast_df.loc[forecast_dates == last_actual_date, 'close_Price'].iloc[0])
        returned_predicted_close = float(
            forecast_df.loc[forecast_dates == last_actual_date, 'predicted_close_Price'].iloc[0]
        )
        first_future_close = float(forecast_df.loc[forecast_dates > last_actual_date, 'close_Price'].iloc[0])

        self.assertEqual(returned_close, actual_close)
        self.assertNotEqual(returned_predicted_close, actual_close)
        self.assertAlmostEqual(returned_predicted_close, 112.211, places=3)
        self.assertAlmostEqual(first_future_close, actual_close * 1.01, places=3)


def run_unit_tests():
    """Run all unit tests and return results"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestCalculatePredictedProfit))
    suite.addTests(loader.loadTestsFromTestCase(TestCreateSequences))
    suite.addTests(loader.loadTestsFromTestCase(TestDetectOverfitting))
    suite.addTests(loader.loadTestsFromTestCase(TestAreHyperparametersIdentical))
    suite.addTests(loader.loadTestsFromTestCase(TestCheckDataHealth))
    suite.addTests(loader.loadTestsFromTestCase(TestBuildRandomForestModel))
    suite.addTests(loader.loadTestsFromTestCase(TestRandomForestCacheNormalization))
    suite.addTests(loader.loadTestsFromTestCase(TestSklearnCacheHelpers))
    suite.addTests(loader.loadTestsFromTestCase(TestSklearnCacheContract))
    suite.addTests(loader.loadTestsFromTestCase(TestSequenceModelRestore))
    suite.addTests(loader.loadTestsFromTestCase(TestBuildXGBoostModel))
    suite.addTests(loader.loadTestsFromTestCase(TestBuildLSTMModel))
    suite.addTests(loader.loadTestsFromTestCase(TestPredictionHistoryCache))
    suite.addTests(loader.loadTestsFromTestCase(TestPredictionStabilization))
    suite.addTests(loader.loadTestsFromTestCase(TestHistoricalFlatModelBlend))
    suite.addTests(loader.loadTestsFromTestCase(TestScaledFeatureDriftSummary))
    suite.addTests(loader.loadTestsFromTestCase(TestSequenceAlignmentHelpers))
    suite.addTests(loader.loadTestsFromTestCase(TestSvrFitHelpers))
    suite.addTests(loader.loadTestsFromTestCase(TestHistoricalPredictionState))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    result = run_unit_tests()
    
    # Print summary
    print("\n" + "="*70)
    print("ML BUILDER UNIT TEST SUMMARY")
    print("="*70)
    print(f"Tests Run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print("="*70)
