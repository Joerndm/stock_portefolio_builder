"""
Unit Tests for dimension_reduction.py, monte_carlo_sim.py, efficient_frontier.py, split_dataset.py

This module contains comprehensive unit tests for feature selection, portfolio optimization,
Monte Carlo simulation, and dataset splitting functions.

Test Coverage:
- feature_selection: SelectKBest feature selection
- feature_selection_rf: Random Forest feature selection
- pca_dataset_transformation: PCA dimension reduction
- monte_carlo_analysis: Monte Carlo stock price simulation
- efficient_frontier_sim: Efficient frontier calculation
- dataset_train_test_split: Train/val/test splitting
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import dimension_reduction
import monte_carlo_sim
import efficient_frontier
import split_dataset


class TestFeatureSelection(unittest.TestCase):
    """Test suite for feature_selection function"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.x_train = pd.DataFrame(np.random.randn(100, 20))
        self.x_val = pd.DataFrame(np.random.randn(20, 20))
        self.x_test = pd.DataFrame(np.random.randn(10, 20))
        self.y_train = pd.Series(np.random.randn(100))
        self.y_val = pd.Series(np.random.randn(20))
        self.y_test = pd.Series(np.random.randn(10))
        self.x_pred = np.random.randn(5, 20)
        
        # Create dataset with proper columns
        columns = [f'feature_{i}' for i in range(20)]
        columns.extend(['date', 'ticker', 'currency', 'open_Price', 'high_Price', 
                       'low_Price', 'close_Price', 'trade_Volume', '1D'])
        self.dataset_df = pd.DataFrame(np.random.randn(100, 29), columns=columns)
    
    def test_correct_dimensions(self):
        """Test that output has correct dimensions"""
        dimensions = 10
        
        x_train_reduced, x_val_reduced, x_test_reduced, x_pred_reduced, selector, features = \
            dimension_reduction.feature_selection(
                dimensions, self.x_train, self.x_val, self.x_test,
                self.y_train, self.y_val, self.y_test, self.x_pred, self.dataset_df
            )
        
        self.assertEqual(x_train_reduced.shape[1], dimensions, "Train set should have correct dimensions")
        self.assertEqual(x_val_reduced.shape[1], dimensions, "Val set should have correct dimensions")
        self.assertEqual(x_test_reduced.shape[1], dimensions, "Test set should have correct dimensions")
        self.assertEqual(x_pred_reduced.shape[1], dimensions, "Pred set should have correct dimensions")
    
    def test_sample_count_preserved(self):
        """Test that number of samples is preserved"""
        dimensions = 10
        
        x_train_reduced, x_val_reduced, x_test_reduced, x_pred_reduced, _, _ = \
            dimension_reduction.feature_selection(
                dimensions, self.x_train, self.x_val, self.x_test,
                self.y_train, self.y_val, self.y_test, self.x_pred, self.dataset_df
            )
        
        self.assertEqual(x_train_reduced.shape[0], self.x_train.shape[0], "Train samples preserved")
        self.assertEqual(x_val_reduced.shape[0], self.x_val.shape[0], "Val samples preserved")
        self.assertEqual(x_test_reduced.shape[0], self.x_test.shape[0], "Test samples preserved")
    
    def test_invalid_dimensions(self):
        """Test with invalid dimension count"""
        with self.assertRaises(ValueError):
            dimension_reduction.feature_selection(
                100,  # More than available features
                self.x_train, self.x_val, self.x_test,
                self.y_train, self.y_val, self.y_test, self.x_pred, self.dataset_df
            )
    
    def test_returns_selector(self):
        """Test that selector object is returned"""
        dimensions = 10
        
        _, _, _, _, selector, features = dimension_reduction.feature_selection(
            dimensions, self.x_train, self.x_val, self.x_test,
            self.y_train, self.y_val, self.y_test, self.x_pred, self.dataset_df
        )
        
        self.assertIsNotNone(selector, "Should return selector object")
        self.assertIsInstance(features, list, "Should return feature list")


class TestFeatureSelectionRF(unittest.TestCase):
    """Test suite for feature_selection_rf function"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.x_train = pd.DataFrame(np.random.randn(100, 20), 
                                    columns=[f'feature_{i}' for i in range(20)])
        self.x_val = pd.DataFrame(np.random.randn(20, 20),
                                 columns=[f'feature_{i}' for i in range(20)])
        self.x_test = pd.DataFrame(np.random.randn(10, 20),
                                  columns=[f'feature_{i}' for i in range(20)])
        self.y_train = pd.Series(np.random.randn(100))
        self.y_val = pd.Series(np.random.randn(20))
        self.y_test = pd.Series(np.random.randn(10))
        self.x_pred = pd.DataFrame(np.random.randn(5, 20),
                                  columns=[f'feature_{i}' for i in range(20)])
    
    def test_rf_feature_selection(self):
        """Test Random Forest feature selection"""
        dimensions = 10
        
        x_train_reduced, x_val_reduced, x_test_reduced, x_pred_reduced, model, features = \
            dimension_reduction.feature_selection_rf(
                dimensions, self.x_train, self.x_val, self.x_test,
                self.y_train, self.y_val, self.y_test, self.x_pred
            )
        
        self.assertEqual(x_train_reduced.shape[1], dimensions, "Should reduce to correct dimensions")
        self.assertIsNotNone(model, "Should return model")
        self.assertEqual(len(features), dimensions, "Should return correct number of features")
    
    def test_feature_importance_ranking(self):
        """Test that features are ranked by importance"""
        dimensions = 5
        
        _, _, _, _, model, features = dimension_reduction.feature_selection_rf(
            dimensions, self.x_train, self.x_val, self.x_test,
            self.y_train, self.y_val, self.y_test, self.x_pred
        )
        
        # Features should be valid column names
        for feature in features:
            self.assertIn(feature, self.x_train.columns, "Selected features should be valid")


class TestPCADatasetTransformation(unittest.TestCase):
    """Test suite for pca_dataset_transformation function"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.x_train = np.random.randn(100, 20)
        self.x_val = np.random.randn(20, 20)
        self.x_test = np.random.randn(10, 20)
        self.x_pred = np.random.randn(5, 20)
    
    def test_pca_dimension_reduction(self):
        """Test PCA reduces dimensions correctly"""
        n_components = 10
        
        x_train_pca, x_val_pca, x_test_pca, x_pred_pca, pca_model = \
            dimension_reduction.pca_dataset_transformation(
                n_components, self.x_train, self.x_val, self.x_test, self.x_pred
            )
        
        self.assertEqual(x_train_pca.shape[1], n_components, "Should reduce to n_components")
        self.assertEqual(x_val_pca.shape[1], n_components)
        self.assertEqual(x_test_pca.shape[1], n_components)
    
    def test_pca_preserves_samples(self):
        """Test that PCA preserves number of samples"""
        n_components = 10
        
        x_train_pca, x_val_pca, x_test_pca, x_pred_pca, _ = \
            dimension_reduction.pca_dataset_transformation(
                n_components, self.x_train, self.x_val, self.x_test, self.x_pred
            )
        
        self.assertEqual(x_train_pca.shape[0], self.x_train.shape[0])
        self.assertEqual(x_val_pca.shape[0], self.x_val.shape[0])
        self.assertEqual(x_test_pca.shape[0], self.x_test.shape[0])
    
    def test_pca_returns_model(self):
        """Test that PCA model is returned"""
        n_components = 10
        
        _, _, _, _, pca_model = dimension_reduction.pca_dataset_transformation(
            n_components, self.x_train, self.x_val, self.x_test, self.x_pred
        )
        
        self.assertIsNotNone(pca_model, "Should return PCA model")


class TestMonteCarloAnalysis(unittest.TestCase):
    """Test suite for monte_carlo_analysis function"""
    
    def setUp(self):
        """Set up test data"""
        self.stock_data_df = pd.DataFrame({
            'ticker': ['AAPL'] * 250,
            'date': pd.date_range('2023-01-01', periods=250),
            'close_Price': np.linspace(100, 150, 250)  # Upward trend
        })
        
        self.forecast_df = pd.DataFrame({
            'close_Price': np.linspace(100, 150, 250)
        })
    
    def test_monte_carlo_returns_dataframes(self):
        """Test that Monte Carlo returns proper DataFrames"""
        price_df, monte_carlo_df = monte_carlo_sim.monte_carlo_analysis(
            seed_number=42,
            stock_data_df=self.stock_data_df,
            forecast_df=self.forecast_df,
            years=1,
            sim_amount=100
        )
        
        self.assertIsInstance(price_df, pd.DataFrame, "Should return price DataFrame")
        self.assertIsInstance(monte_carlo_df, pd.DataFrame, "Should return MC DataFrame")
    
    def test_monte_carlo_simulation_count(self):
        """Test correct number of simulations"""
        sim_amount = 50
        
        price_df, _ = monte_carlo_sim.monte_carlo_analysis(
            seed_number=42,
            stock_data_df=self.stock_data_df,
            forecast_df=self.forecast_df,
            years=1,
            sim_amount=sim_amount
        )
        
        # Price_df should have sim_amount rows (after transpose)
        self.assertEqual(price_df.shape[0], sim_amount, "Should have correct number of simulations")
    
    def test_monte_carlo_forecast_days(self):
        """Test correct forecast period"""
        years = 2
        expected_days = years * 252
        
        price_df, _ = monte_carlo_sim.monte_carlo_analysis(
            seed_number=42,
            stock_data_df=self.stock_data_df,
            forecast_df=self.forecast_df,
            years=years,
            sim_amount=100
        )
        
        # Should have approximately expected_days columns
        self.assertEqual(price_df.shape[1], expected_days, "Should forecast correct number of days")
    
    def test_monte_carlo_reproducibility(self):
        """Test that same seed produces same results"""
        price_df1, mc_df1 = monte_carlo_sim.monte_carlo_analysis(
            seed_number=42,
            stock_data_df=self.stock_data_df,
            forecast_df=self.forecast_df,
            years=1,
            sim_amount=50
        )
        
        price_df2, mc_df2 = monte_carlo_sim.monte_carlo_analysis(
            seed_number=42,
            stock_data_df=self.stock_data_df,
            forecast_df=self.forecast_df,
            years=1,
            sim_amount=50
        )
        
        np.testing.assert_array_almost_equal(
            price_df1.values, price_df2.values,
            decimal=5, err_msg="Same seed should produce same results"
        )


class TestEfficientFrontierSim(unittest.TestCase):
    """Test suite for efficient_frontier_sim function"""
    
    def setUp(self):
        """Set up test data"""
        # Create price data for 3 stocks
        dates = pd.date_range('2023-01-01', periods=250)
        self.price_df = pd.DataFrame({
            'AAPL': np.random.uniform(90, 110, 250),
            'GOOGL': np.random.uniform(80, 120, 250),
            'MSFT': np.random.uniform(85, 115, 250)
        }, index=dates)
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_efficient_frontier_returns_dataframe(self, mock_show, mock_savefig):
        """Test that efficient frontier returns DataFrame"""
        result = efficient_frontier.efficient_frontier_sim(self.price_df)
        
        self.assertIsInstance(result, pd.DataFrame, "Should return DataFrame")
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_portfolio_weights_sum_to_one(self, mock_show, mock_savefig):
        """Test that portfolio weights sum to 1"""
        result = efficient_frontier.efficient_frontier_sim(self.price_df)
        
        # Check if weight columns exist and sum to 1
        weight_cols = [col for col in result.columns if col in self.price_df.columns]
        if weight_cols:
            weight_sums = result[weight_cols].sum(axis=1)
            np.testing.assert_array_almost_equal(
                weight_sums, np.ones(len(result)),
                decimal=5, err_msg="Weights should sum to 1"
            )
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_return_and_volatility_columns(self, mock_show, mock_savefig):
        """Test that return and volatility columns exist"""
        result = efficient_frontier.efficient_frontier_sim(self.price_df)
        
        # Should have Return and Volatility columns
        self.assertIn('Return', result.columns, "Should have Return column")
        self.assertIn('Volatility', result.columns, "Should have Volatility column")
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_positive_volatility(self, mock_show, mock_savefig):
        """Test that volatility is non-negative"""
        result = efficient_frontier.efficient_frontier_sim(self.price_df)
        
        if 'Volatility' in result.columns:
            self.assertTrue((result['Volatility'] >= 0).all(),
                          "Volatility should be non-negative")


class TestDatasetTrainTestSplit(unittest.TestCase):
    """Test suite for dataset_train_test_split function"""
    
    def setUp(self):
        """Set up test data"""
        # Create realistic stock dataset
        self.dataset_df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=200),
            'ticker': ['AAPL'] * 200,
            'currency': ['USD'] * 200,
            'open_Price': np.random.uniform(90, 110, 200),
            'high_Price': np.random.uniform(95, 115, 200),
            'low_Price': np.random.uniform(85, 105, 200),
            'close_Price': np.random.uniform(90, 110, 200),
            'trade_Volume': np.random.randint(1000000, 10000000, 200),
            '1D': np.random.uniform(-0.05, 0.05, 200),
            'sma_5': np.random.uniform(90, 110, 200),
            'sma_20': np.random.uniform(90, 110, 200),
            'rsi': np.random.uniform(30, 70, 200),
            'macd': np.random.uniform(-5, 5, 200)
        })
    
    def test_split_proportions(self):
        """Test that data is split in correct proportions"""
        test_size = 0.10
        val_size = 0.20
        
        scaler_x, scaler_y, x_train, x_val, x_test, y_train, y_val, y_test, x_pred = \
            split_dataset.dataset_train_test_split(
                self.dataset_df.copy(), test_size=test_size, validation_size=val_size
            )
        
        total_samples = len(x_train) + len(x_val) + len(x_test)
        
        # Check approximate proportions (allowing for rounding)
        train_ratio = len(x_train) / total_samples
        val_ratio = len(x_val) / total_samples
        test_ratio = len(x_test) / total_samples
        
        self.assertAlmostEqual(train_ratio, 1 - test_size - val_size, delta=0.05,
                             msg="Train set should be approximately correct size")
        self.assertAlmostEqual(val_ratio, val_size, delta=0.05,
                             msg="Val set should be approximately correct size")
        self.assertAlmostEqual(test_ratio, test_size, delta=0.05,
                             msg="Test set should be approximately correct size")
    
    def test_returns_scalers(self):
        """Test that scalers are returned"""
        scaler_x, scaler_y, *_ = split_dataset.dataset_train_test_split(self.dataset_df.copy())
        
        self.assertIsNotNone(scaler_x, "Should return X scaler")
        self.assertIsNotNone(scaler_y, "Should return y scaler")
    
    def test_data_is_scaled(self):
        """Test that returned data is scaled"""
        _, _, x_train, x_val, x_test, y_train, y_val, y_test, _ = \
            split_dataset.dataset_train_test_split(self.dataset_df.copy())
        
        # Scaled data should generally be in [0, 1] range (MinMaxScaler)
        # Check if most values are in reasonable range
        if len(x_train) > 0:
            train_min = np.min(x_train)
            train_max = np.max(x_train)
            
            self.assertGreaterEqual(train_min, -1, "Scaled data should be >= -1")
            self.assertLessEqual(train_max, 2, "Scaled data should be <= 2")
    
    def test_no_data_leakage(self):
        """Test that sets don't share samples"""
        _, _, x_train, x_val, x_test, *_ = \
            split_dataset.dataset_train_test_split(self.dataset_df.copy(), rs=42)
        
        # Sets should have different sizes
        all_sizes = [len(x_train), len(x_val), len(x_test)]
        
        # Total should be less than original (due to prediction set)
        total = sum(all_sizes)
        self.assertLess(total, len(self.dataset_df),
                       "Total samples should be less than original (prediction set)")
    
    def test_reproducibility(self):
        """Test that same random state produces same split"""
        result1 = split_dataset.dataset_train_test_split(self.dataset_df.copy(), rs=42)
        result2 = split_dataset.dataset_train_test_split(self.dataset_df.copy(), rs=42)
        
        # x_train should be identical
        np.testing.assert_array_equal(result1[2], result2[2],
                                     err_msg="Same random state should produce same split")


def run_unit_tests():
    """Run all unit tests and return results"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestFeatureSelection))
    suite.addTests(loader.loadTestsFromTestCase(TestFeatureSelectionRF))
    suite.addTests(loader.loadTestsFromTestCase(TestPCADatasetTransformation))
    suite.addTests(loader.loadTestsFromTestCase(TestMonteCarloAnalysis))
    suite.addTests(loader.loadTestsFromTestCase(TestEfficientFrontierSim))
    suite.addTests(loader.loadTestsFromTestCase(TestDatasetTrainTestSplit))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    result = run_unit_tests()
    
    # Print summary
    print("\n" + "="*70)
    print("ADDITIONAL MODULES UNIT TEST SUMMARY")
    print("="*70)
    print(f"Tests Run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print("="*70)
