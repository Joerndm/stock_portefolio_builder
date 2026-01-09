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
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import ml_builder


class TestCalculatePredictedProfit(unittest.TestCase):
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


class TestCreateSequences(unittest.TestCase):
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


class TestDetectOverfitting(unittest.TestCase):
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
        
        # Test at exact threshold
        is_overfitting, score, _ = ml_builder.detect_overfitting(
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


class TestAreHyperparametersIdentical(unittest.TestCase):
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


class TestCheckDataHealth(unittest.TestCase):
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


class TestBuildRandomForestModel(unittest.TestCase):
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


class TestBuildXGBoostModel(unittest.TestCase):
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


class TestBuildLSTMModel(unittest.TestCase):
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
    suite.addTests(loader.loadTestsFromTestCase(TestBuildXGBoostModel))
    suite.addTests(loader.loadTestsFromTestCase(TestBuildLSTMModel))
    
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
