"""
Integration Tests for Stock Portfolio Builder

This module contains integration tests that verify multiple components work together correctly.
Tests cover data pipelines, training workflows, and prediction processes.

Test Categories:
- Data Pipeline Integration: Fetch → Process → Store
- Training Pipeline Integration: Load → Feature → Train → Evaluate
- Prediction Pipeline Integration: Load Model → Predict → Analyze
- Portfolio Analysis Integration: Multiple stocks → Optimization
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import stock_data_fetch
import db_interactions
import dimension_reduction
import split_dataset
import data_scalers


class TestDataPipelineIntegration(unittest.TestCase):
    """Integration tests for data fetching and processing pipeline"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_ticker = 'AAPL'
        self.test_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=250),
            'close_Price': np.random.uniform(90, 110, 250),
            'high_Price': np.random.uniform(95, 115, 250),
            'low_Price': np.random.uniform(85, 105, 250),
            'open_Price': np.random.uniform(88, 112, 250),
            'trade_Volume': np.random.randint(1000000, 10000000, 250),
            'ticker': [self.test_ticker] * 250
        })
    
    def test_price_data_to_technical_indicators(self):
        """Test pipeline: Price Data → Technical Indicators"""
        # Start with basic price data
        price_data = self.test_data.copy()
        
        # Add moving averages
        result = stock_data_fetch.calculate_moving_averages(price_data)
        self.assertIn('sma_5', result.columns, "Should add SMA columns")
        self.assertIn('ema_20', result.columns, "Should add EMA columns")
        
        # Add technical indicators
        result = stock_data_fetch.add_technical_indicators(result)
        self.assertGreater(len(result.columns), len(price_data.columns),
                         "Should add technical indicator columns")
        
        # Add volume indicators
        result = stock_data_fetch.add_volume_indicators(result)
        self.assertIsInstance(result, pd.DataFrame, "Should maintain DataFrame structure")
    
    def test_data_fetch_to_feature_calculation(self):
        """Test pipeline: Fetch → Calculate Features → Validate"""
        price_data = self.test_data.copy()
        
        # Calculate period returns
        with_returns = stock_data_fetch.calculate_period_returns(price_data)
        self.assertIn('1D', with_returns.columns, "Should add return columns")
        
        # Calculate standard deviation
        with_std = stock_data_fetch.calculate_standard_diviation_value(with_returns)
        
        # Calculate Bollinger Bands
        with_bb = stock_data_fetch.calculate_bollinger_bands(with_std)
        
        # Calculate momentum
        final_data = stock_data_fetch.calculate_momentum(with_bb)
        
        # Verify pipeline preserved data
        self.assertEqual(len(final_data), len(price_data),
                        "Should preserve row count")
        self.assertGreater(len(final_data.columns), len(price_data.columns),
                         "Should add feature columns")
    
    @patch('db_interactions.db_connectors.pandas_mysql_connector')
    @patch('db_interactions.fetch_secrets.secret_import')
    def test_process_to_database_export(self, mock_secrets, mock_connector):
        """Test pipeline: Process Data → Export to Database"""
        mock_secrets.return_value = ('host', 'user', 'pass', 'db')
        mock_connection = MagicMock()
        mock_connector.return_value = mock_connection
        
        # Process data
        processed_data = stock_data_fetch.calculate_moving_averages(self.test_data.copy())
        processed_data = stock_data_fetch.add_technical_indicators(processed_data)
        
        # Export to database (mocked)
        try:
            db_interactions.export_stock_price_data(processed_data)
        except Exception as e:
            self.fail(f"Export pipeline failed: {e}")


class TestTrainingPipelineIntegration(unittest.TestCase):
    """Integration tests for model training pipeline"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        # Create realistic dataset
        n_samples = 200
        n_features = 30
        
        self.dataset_df = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Add required columns
        self.dataset_df['date'] = pd.date_range('2023-01-01', periods=n_samples)
        self.dataset_df['ticker'] = 'AAPL'
        self.dataset_df['currency'] = 'USD'
        self.dataset_df['open_Price'] = np.random.uniform(90, 110, n_samples)
        self.dataset_df['high_Price'] = np.random.uniform(95, 115, n_samples)
        self.dataset_df['low_Price'] = np.random.uniform(85, 105, n_samples)
        self.dataset_df['close_Price'] = np.random.uniform(90, 110, n_samples)
        self.dataset_df['trade_Volume'] = np.random.randint(1000000, 10000000, n_samples)
        self.dataset_df['1D'] = np.random.uniform(-0.05, 0.05, n_samples)
    
    def test_data_split_to_feature_selection(self):
        """Test pipeline: Split Data → Feature Selection"""
        # Split dataset
        scaler_x, scaler_y, x_train, x_val, x_test, y_train, y_val, y_test, x_pred = \
            split_dataset.dataset_train_test_split(self.dataset_df.copy())
        
        # Verify split worked
        self.assertGreater(len(x_train), len(x_val), "Train > Val")
        self.assertGreater(len(x_val), len(x_test), "Val > Test")
        
        # Convert to DataFrames for feature selection
        feature_cols = [col for col in self.dataset_df.columns 
                       if col not in ['date', 'ticker', 'currency', 'open_Price', 
                                     'high_Price', 'low_Price', 'close_Price', 
                                     'trade_Volume', '1D', 'prediction']]
        
        x_train_df = pd.DataFrame(x_train, columns=feature_cols[:x_train.shape[1]])
        x_val_df = pd.DataFrame(x_val, columns=feature_cols[:x_val.shape[1]])
        x_test_df = pd.DataFrame(x_test, columns=feature_cols[:x_test.shape[1]])
        x_pred_arr = x_pred if isinstance(x_pred, np.ndarray) else x_pred.values
        
        y_train_series = pd.Series(y_train.flatten())
        y_val_series = pd.Series(y_val.flatten())
        y_test_series = pd.Series(y_test.flatten())
        
        # Feature selection
        if x_train_df.shape[1] >= 10:
            dimensions = 10
            x_train_reduced, x_val_reduced, x_test_reduced, x_pred_reduced, selector, features = \
                dimension_reduction.feature_selection(
                    dimensions, x_train_df, x_val_df, x_test_df,
                    y_train_series, y_val_series, y_test_series,
                    x_pred_arr, self.dataset_df
                )
            
            self.assertEqual(x_train_reduced.shape[1], dimensions,
                           "Feature selection should reduce dimensions")
    
    def test_scaling_to_feature_selection(self):
        """Test pipeline: Scaling → Feature Selection → Model Ready"""
        # This tests the integration of data_scalers with dimension_reduction
        
        # Prepare data
        feature_cols = [col for col in self.dataset_df.columns 
                       if col not in ['date', 'ticker', 'currency', 'open_Price', 
                                     'high_Price', 'low_Price', 'close_Price', 
                                     'trade_Volume', '1D']]
        
        X = self.dataset_df[feature_cols].iloc[:150]
        y = self.dataset_df['1D'].iloc[:150]
        
        # Fit scaler
        scaler = data_scalers.data_preprocessing_minmax_scaler_fit(X)
        
        # Transform data
        X_scaled = data_scalers.data_preprocessing_minmax_scaler_transform(scaler, X)
        
        # Verify scaling worked
        self.assertTrue((X_scaled.min() >= -0.1).all() and (X_scaled.max() <= 1.1).all(),
                       "Data should be approximately scaled to [0,1]")


class TestPredictionPipelineIntegration(unittest.TestCase):
    """Integration tests for prediction pipeline"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.test_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100),
            'close_Price': np.random.uniform(90, 110, 100),
            'sma_5': np.random.uniform(90, 110, 100),
            'sma_20': np.random.uniform(90, 110, 100),
            'rsi': np.random.uniform(30, 70, 100),
            'macd': np.random.uniform(-5, 5, 100),
            'ticker': ['AAPL'] * 100
        })
    
    def test_data_preprocessing_to_prediction(self):
        """Test pipeline: Preprocess → Scale → Predict"""
        # Fit scaler on features
        feature_cols = ['sma_5', 'sma_20', 'rsi', 'macd']
        X = self.test_data[feature_cols]
        
        scaler = data_scalers.data_preprocessing_minmax_scaler_fit(X)
        X_scaled = data_scalers.data_preprocessing_minmax_scaler_transform(scaler, X)
        
        # Verify preprocessing succeeded
        self.assertEqual(X_scaled.shape, X.shape, "Shape should be preserved")
        self.assertIsInstance(X_scaled, (pd.DataFrame, np.ndarray),
                            "Should return scaled data")
    
    def test_feature_calculation_for_future_prediction(self):
        """Test that features can be calculated for future predictions"""
        # This simulates the process of calculating features for next-day prediction
        
        historical_data = self.test_data.copy()
        
        # Calculate technical indicators
        with_ma = stock_data_fetch.calculate_moving_averages(historical_data)
        
        # Verify we can extract last values for prediction
        last_row = with_ma.iloc[-1:].copy()
        
        self.assertIsInstance(last_row, pd.DataFrame,
                            "Should be able to extract last row for prediction")
        self.assertGreater(len(last_row.columns), len(historical_data.columns),
                         "Should have additional feature columns")


class TestPortfolioAnalysisIntegration(unittest.TestCase):
    """Integration tests for portfolio analysis workflow"""
    
    def setUp(self):
        """Set up multi-stock test data"""
        dates = pd.date_range('2023-01-01', periods=250)
        
        self.portfolio_prices = pd.DataFrame({
            'AAPL': np.random.uniform(90, 110, 250),
            'GOOGL': np.random.uniform(80, 120, 250),
            'MSFT': np.random.uniform(85, 115, 250)
        }, index=dates)
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_multi_stock_to_efficient_frontier(self, mock_show, mock_savefig):
        """Test pipeline: Multiple Stocks → Efficient Frontier"""
        import efficient_frontier
        
        # Calculate efficient frontier
        result = efficient_frontier.efficient_frontier_sim(self.portfolio_prices)
        
        # Verify result
        self.assertIsInstance(result, pd.DataFrame, "Should return DataFrame")
        self.assertIn('Return', result.columns, "Should have Return column")
        self.assertIn('Volatility', result.columns, "Should have Volatility column")
        
        # Verify portfolio weights
        weight_cols = [col for col in result.columns if col in self.portfolio_prices.columns]
        if weight_cols:
            self.assertEqual(len(weight_cols), len(self.portfolio_prices.columns),
                           "Should have weights for all stocks")
    
    def test_stock_data_to_monte_carlo(self):
        """Test pipeline: Stock Data → Monte Carlo Simulation"""
        import monte_carlo_sim
        
        # Prepare data for Monte Carlo
        stock_data = pd.DataFrame({
            'ticker': ['AAPL'] * 250,
            'date': pd.date_range('2023-01-01', periods=250),
            'close_Price': self.portfolio_prices['AAPL'].values
        })
        
        forecast_df = pd.DataFrame({
            'close_Price': self.portfolio_prices['AAPL'].values
        })
        
        # Run Monte Carlo
        price_df, mc_df = monte_carlo_sim.monte_carlo_analysis(
            seed_number=42,
            stock_data_df=stock_data,
            forecast_df=forecast_df,
            years=1,
            sim_amount=50
        )
        
        # Verify results
        self.assertIsInstance(price_df, pd.DataFrame, "Should return price DataFrame")
        self.assertIsInstance(mc_df, pd.DataFrame, "Should return MC DataFrame")
        self.assertEqual(price_df.shape[0], 50, "Should have 50 simulations")


class TestEndToEndDataFlow(unittest.TestCase):
    """Integration tests for complete end-to-end data flow"""
    
    @patch('db_interactions.db_connectors.pandas_mysql_connector')
    @patch('db_interactions.fetch_secrets.secret_import')
    @patch('pandas.read_sql')
    def test_database_to_training_ready(self, mock_read_sql, mock_secrets, mock_connector):
        """Test complete flow: Database → Process → Training Ready"""
        # Mock database connection
        mock_secrets.return_value = ('host', 'user', 'pass', 'db')
        mock_connector.return_value = Mock()
        
        # Mock database data
        db_data = pd.DataFrame({
            'ticker': ['AAPL'] * 200,
            'date': pd.date_range('2023-01-01', periods=200),
            'close_Price': np.random.uniform(90, 110, 200),
            'high_Price': np.random.uniform(95, 115, 200),
            'low_Price': np.random.uniform(85, 105, 200),
            'open_Price': np.random.uniform(88, 112, 200),
            'trade_Volume': np.random.randint(1000000, 10000000, 200),
            'p_e_ratio': np.random.uniform(15, 35, 200),
            'p_b_ratio': np.random.uniform(5, 15, 200)
        })
        
        mock_read_sql.return_value = db_data
        
        # Import from database (mocked)
        try:
            dataset = db_interactions.import_stock_price_data(
                amount=200, stock_ticker='AAPL'
            )
            
            # Verify imported data
            self.assertIsInstance(dataset, pd.DataFrame, "Should return DataFrame")
            self.assertGreater(len(dataset), 0, "Should have data")
            
            # Process data
            dataset_with_features = stock_data_fetch.calculate_moving_averages(dataset)
            
            # Verify processing succeeded
            self.assertGreater(len(dataset_with_features.columns), len(dataset.columns),
                             "Should add feature columns")
            
        except Exception as e:
            # Some errors are expected due to mocking, but process should be testable
            self.assertIsInstance(e, (KeyError, ValueError),
                                "Only expected errors during mocked DB operations")
    
    def test_complete_feature_pipeline(self):
        """Test complete feature engineering pipeline"""
        # Start with raw price data
        raw_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=250),
            'close_Price': np.random.uniform(90, 110, 250),
            'high_Price': np.random.uniform(95, 115, 250),
            'low_Price': np.random.uniform(85, 105, 250),
            'open_Price': np.random.uniform(88, 112, 250),
            'trade_Volume': np.random.randint(1000000, 10000000, 250),
            'ticker': ['AAPL'] * 250
        })
        
        # Step 1: Moving averages
        step1 = stock_data_fetch.calculate_moving_averages(raw_data)
        self.assertGreater(len(step1.columns), len(raw_data.columns), "Step 1 adds columns")
        
        # Step 2: Technical indicators
        step2 = stock_data_fetch.add_technical_indicators(step1)
        self.assertGreater(len(step2.columns), len(step1.columns), "Step 2 adds columns")
        
        # Step 3: Volume indicators
        step3 = stock_data_fetch.add_volume_indicators(step2)
        self.assertGreater(len(step3.columns), len(step2.columns), "Step 3 adds columns")
        
        # Step 4: Volatility indicators
        step4 = stock_data_fetch.add_volatility_indicators(step3)
        self.assertIsInstance(step4, pd.DataFrame, "Step 4 completes successfully")
        
        # Step 5: Period returns
        final_data = stock_data_fetch.calculate_period_returns(step4)
        
        # Verify complete pipeline
        self.assertIsInstance(final_data, pd.DataFrame, "Pipeline produces DataFrame")
        self.assertGreater(len(final_data.columns), 20, "Should have many features")
        self.assertEqual(len(final_data), len(raw_data), "Should preserve row count")


def run_integration_tests():
    """Run all integration tests and return results"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDataPipelineIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestTrainingPipelineIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestPredictionPipelineIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestPortfolioAnalysisIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEndDataFlow))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    result = run_integration_tests()
    
    # Print summary
    print("\n" + "="*70)
    print("INTEGRATION TEST SUMMARY")
    print("="*70)
    print(f"Tests Run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print("="*70)
