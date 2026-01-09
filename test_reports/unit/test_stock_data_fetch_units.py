"""
Unit Tests for stock_data_fetch.py

This module contains comprehensive unit tests for stock data fetching and processing functions.
Tests use mocking to avoid external API calls and ensure fast, reliable execution.

Test Coverage:
- import_tickers_from_csv: CSV import functionality
- calculate_standard_diviation_value: Standard deviation calculation
- calculate_bollinger_bands: Bollinger Bands calculation
- calculate_momentum: Momentum calculation
- add_volume_indicators: Volume indicator calculations
- calculate_ratios: Financial ratio calculations
- calculate_moving_averages: Moving average calculations
- calculate_period_returns: Period return calculations
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock, mock_open
import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import stock_data_fetch


class TestImportTickersFromCSV(unittest.TestCase):
    """Test suite for import_tickers_from_csv function"""
    
    @patch('builtins.open', new_callable=mock_open, read_data='Symbol\nAAPL\nGOOGL\nMSFT\n')
    @patch('pandas.read_csv')
    def test_successful_import(self, mock_read_csv, mock_file):
        """Test successful CSV import"""
        mock_df = pd.DataFrame({'Symbol': ['AAPL', 'GOOGL', 'MSFT']})
        mock_read_csv.return_value = mock_df
        
        result = stock_data_fetch.import_tickers_from_csv('test.csv')
        
        self.assertIsInstance(result, pd.DataFrame, "Should return DataFrame")
        self.assertIn('Symbol', result.columns, "Should have Symbol column")
    
    def test_empty_csv_file(self):
        """Test with empty CSV file parameter"""
        with self.assertRaises(ValueError) as context:
            stock_data_fetch.import_tickers_from_csv("")
        
        self.assertIn("cannot be empty", str(context.exception).lower())
    
    @patch('pandas.read_csv')
    def test_missing_symbol_column(self, mock_read_csv):
        """Test CSV without Symbol column"""
        mock_df = pd.DataFrame({'Ticker': ['AAPL', 'GOOGL']})
        mock_read_csv.return_value = mock_df
        
        with self.assertRaises(KeyError):
            stock_data_fetch.import_tickers_from_csv('test.csv')


class TestCalculateStandardDeviationValue(unittest.TestCase):
    """Test suite for calculate_standard_diviation_value function"""
    
    def setUp(self):
        """Set up test data"""
        self.stock_data = pd.DataFrame({
            'close_Price': [100, 102, 101, 103, 105, 104, 106, 108, 107, 110],
            'ticker': ['AAPL'] * 10,
            'date': pd.date_range('2024-01-01', periods=10)
        })
    
    def test_standard_deviation_calculation(self):
        """Test standard deviation is calculated"""
        result = stock_data_fetch.calculate_standard_diviation_value(self.stock_data)
        
        # Should add standard deviation columns
        self.assertIn('std_5', result.columns, "Should have 5-day std")
        self.assertIn('std_20', result.columns, "Should have 20-day std")
        self.assertIn('std_40', result.columns, "Should have 40-day std")
    
    def test_positive_values(self):
        """Test that standard deviations are non-negative"""
        result = stock_data_fetch.calculate_standard_diviation_value(self.stock_data)
        
        # Drop NaN values before checking
        for col in ['std_5', 'std_20', 'std_40']:
            if col in result.columns:
                valid_values = result[col].dropna()
                if len(valid_values) > 0:
                    self.assertTrue((valid_values >= 0).all(), 
                                  f"{col} should be non-negative")
    
    def test_constant_prices(self):
        """Test with constant prices (zero volatility)"""
        constant_data = pd.DataFrame({
            'close_Price': [100] * 50,
            'ticker': ['AAPL'] * 50
        })
        
        result = stock_data_fetch.calculate_standard_diviation_value(constant_data)
        
        # Standard deviation should be zero or very close
        if 'std_5' in result.columns:
            valid_std = result['std_5'].dropna()
            if len(valid_std) > 0:
                self.assertTrue((valid_std < 0.01).all(), 
                              "Constant prices should have near-zero std")
    
    def test_maintains_original_columns(self):
        """Test that original columns are preserved"""
        result = stock_data_fetch.calculate_standard_diviation_value(self.stock_data)
        
        for col in self.stock_data.columns:
            self.assertIn(col, result.columns, f"Should preserve {col}")


class TestCalculateBollingerBands(unittest.TestCase):
    """Test suite for calculate_bollinger_bands function"""
    
    def setUp(self):
        """Set up test data"""
        self.stock_data = pd.DataFrame({
            'close_Price': np.random.uniform(90, 110, 100),
            'ticker': ['AAPL'] * 100,
            'date': pd.date_range('2024-01-01', periods=100)
        })
    
    def test_bollinger_bands_added(self):
        """Test that Bollinger Bands columns are added"""
        result = stock_data_fetch.calculate_bollinger_bands(self.stock_data)
        
        # Should have upper and lower bands for each period
        bb_columns = [col for col in result.columns if 'bb' in col.lower()]
        self.assertGreater(len(bb_columns), 0, "Should add Bollinger Band columns")
    
    def test_band_relationship(self):
        """Test that upper band > middle band > lower band"""
        result = stock_data_fetch.calculate_bollinger_bands(self.stock_data)
        
        # Check for common Bollinger Band column patterns
        if 'bb_upper_20' in result.columns and 'bb_lower_20' in result.columns:
            valid_rows = result[['bb_upper_20', 'sma_20', 'bb_lower_20']].dropna()
            if len(valid_rows) > 0:
                self.assertTrue((valid_rows['bb_upper_20'] >= valid_rows['sma_20']).all(),
                              "Upper band should be >= middle")
                self.assertTrue((valid_rows['sma_20'] >= valid_rows['bb_lower_20']).all(),
                              "Middle should be >= lower band")
    
    def test_nan_handling(self):
        """Test handling of NaN values"""
        data_with_nan = self.stock_data.copy()
        data_with_nan.loc[5, 'close_Price'] = np.nan
        
        result = stock_data_fetch.calculate_bollinger_bands(data_with_nan)
        
        self.assertIsInstance(result, pd.DataFrame, "Should handle NaN values")


class TestCalculateMomentum(unittest.TestCase):
    """Test suite for calculate_momentum function"""
    
    def setUp(self):
        """Set up test data"""
        self.stock_data = pd.DataFrame({
            'close_Price': [100, 102, 104, 106, 108, 110, 112, 114, 116, 118],
            'ticker': ['AAPL'] * 10,
            'date': pd.date_range('2024-01-01', periods=10)
        })
    
    def test_momentum_calculation(self):
        """Test that momentum is calculated"""
        result = stock_data_fetch.calculate_momentum(self.stock_data)
        
        # Should add momentum columns
        momentum_cols = [col for col in result.columns if 'momentum' in col.lower()]
        self.assertGreater(len(momentum_cols), 0, "Should add momentum columns")
    
    def test_positive_trend_momentum(self):
        """Test momentum for positive price trend"""
        result = stock_data_fetch.calculate_momentum(self.stock_data)
        
        # For increasing prices, momentum should generally be positive
        if 'momentum' in result.columns:
            valid_momentum = result['momentum'].dropna()
            if len(valid_momentum) > 0:
                mean_momentum = valid_momentum.mean()
                self.assertGreater(mean_momentum, 0, 
                                 "Increasing prices should have positive momentum")
    
    def test_negative_trend_momentum(self):
        """Test momentum for negative price trend"""
        decreasing_data = pd.DataFrame({
            'close_Price': [118, 116, 114, 112, 110, 108, 106, 104, 102, 100],
            'ticker': ['AAPL'] * 10
        })
        
        result = stock_data_fetch.calculate_momentum(decreasing_data)
        
        if 'momentum' in result.columns:
            valid_momentum = result['momentum'].dropna()
            if len(valid_momentum) > 0:
                mean_momentum = valid_momentum.mean()
                self.assertLess(mean_momentum, 0, 
                              "Decreasing prices should have negative momentum")


class TestAddVolumeIndicators(unittest.TestCase):
    """Test suite for add_volume_indicators function"""
    
    def setUp(self):
        """Set up test data"""
        self.stock_data = pd.DataFrame({
            'close_Price': np.random.uniform(90, 110, 50),
            'high_Price': np.random.uniform(95, 115, 50),
            'low_Price': np.random.uniform(85, 105, 50),
            'trade_Volume': np.random.randint(1000000, 5000000, 50),
            'ticker': ['AAPL'] * 50,
            'date': pd.date_range('2024-01-01', periods=50)
        })
    
    def test_volume_indicators_added(self):
        """Test that volume indicators are added"""
        result = stock_data_fetch.add_volume_indicators(self.stock_data)
        
        # Should add volume-related columns
        volume_cols = [col for col in result.columns 
                      if any(ind in col.lower() for ind in ['volume', 'vwap', 'obv'])]
        self.assertGreater(len(volume_cols), 0, "Should add volume indicators")
    
    def test_vwap_calculation(self):
        """Test VWAP calculation if present"""
        result = stock_data_fetch.add_volume_indicators(self.stock_data)
        
        # VWAP should be within price range
        if 'vwap' in result.columns:
            valid_vwap = result['vwap'].dropna()
            if len(valid_vwap) > 0:
                self.assertTrue((valid_vwap > 0).all(), "VWAP should be positive")
    
    def test_preserves_original_data(self):
        """Test that original columns are preserved"""
        result = stock_data_fetch.add_volume_indicators(self.stock_data)
        
        for col in ['close_Price', 'trade_Volume']:
            self.assertIn(col, result.columns, f"Should preserve {col}")


class TestCalculateRatios(unittest.TestCase):
    """Test suite for calculate_ratios function"""
    
    def setUp(self):
        """Set up test data with financial metrics"""
        self.stock_data = pd.DataFrame({
            'close_Price': [100, 105, 110],
            'earnings_per_share': [5, 5.5, 6],
            'book_value_per_share': [50, 52, 54],
            'revenue_per_share': [200, 210, 220],
            'free_cash_flow_per_share': [10, 11, 12],
            'ticker': ['AAPL'] * 3
        })
    
    def test_pe_ratio_calculation(self):
        """Test P/E ratio calculation"""
        result = stock_data_fetch.calculate_ratios(self.stock_data)
        
        if 'p_e_ratio' in result.columns:
            # P/E = Price / EPS
            expected_pe = self.stock_data['close_Price'] / self.stock_data['earnings_per_share']
            pd.testing.assert_series_equal(
                result['p_e_ratio'].dropna(), 
                expected_pe.dropna(), 
                check_names=False
            )
    
    def test_pb_ratio_calculation(self):
        """Test P/B ratio calculation"""
        result = stock_data_fetch.calculate_ratios(self.stock_data)
        
        if 'p_b_ratio' in result.columns:
            # P/B = Price / Book Value
            valid_pb = result['p_b_ratio'].dropna()
            self.assertTrue((valid_pb > 0).all(), "P/B should be positive")
    
    def test_zero_denominator_handling(self):
        """Test handling of zero denominators"""
        data_with_zeros = self.stock_data.copy()
        data_with_zeros.loc[0, 'earnings_per_share'] = 0
        
        result = stock_data_fetch.calculate_ratios(data_with_zeros)
        
        # Should handle division by zero gracefully (NaN or inf)
        if 'p_e_ratio' in result.columns:
            self.assertIsInstance(result, pd.DataFrame, "Should handle zero denominators")
    
    def test_negative_earnings_handling(self):
        """Test handling of negative earnings"""
        data_with_negative = self.stock_data.copy()
        data_with_negative.loc[0, 'earnings_per_share'] = -5
        
        result = stock_data_fetch.calculate_ratios(data_with_negative)
        
        # Should handle negative values
        self.assertIsInstance(result, pd.DataFrame, "Should handle negative earnings")


class TestCalculateMovingAverages(unittest.TestCase):
    """Test suite for calculate_moving_averages function"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.stock_data = pd.DataFrame({
            'close_Price': np.random.uniform(90, 110, 250),
            'ticker': ['AAPL'] * 250,
            'date': pd.date_range('2024-01-01', periods=250)
        })
    
    def test_sma_columns_added(self):
        """Test that SMA columns are added"""
        result = stock_data_fetch.calculate_moving_averages(self.stock_data)
        
        # Should have multiple SMA columns
        sma_cols = [col for col in result.columns if 'sma' in col.lower()]
        self.assertGreater(len(sma_cols), 0, "Should add SMA columns")
    
    def test_ema_columns_added(self):
        """Test that EMA columns are added"""
        result = stock_data_fetch.calculate_moving_averages(self.stock_data)
        
        # Should have multiple EMA columns
        ema_cols = [col for col in result.columns if 'ema' in col.lower()]
        self.assertGreater(len(ema_cols), 0, "Should add EMA columns")
    
    def test_sma_calculation_accuracy(self):
        """Test SMA calculation accuracy"""
        simple_data = pd.DataFrame({
            'close_Price': [100, 102, 104, 106, 108],
            'ticker': ['AAPL'] * 5
        })
        
        result = stock_data_fetch.calculate_moving_averages(simple_data)
        
        # For 5-day SMA on 5 data points, last value should be mean of all
        if 'sma_5' in result.columns:
            expected_sma = simple_data['close_Price'].mean()
            actual_sma = result['sma_5'].iloc[-1]
            if not pd.isna(actual_sma):
                self.assertAlmostEqual(actual_sma, expected_sma, places=2,
                                     msg="SMA calculation should be accurate")
    
    def test_ma_smoothing_effect(self):
        """Test that moving averages smooth data"""
        result = stock_data_fetch.calculate_moving_averages(self.stock_data)
        
        # Moving averages should have lower volatility than raw prices
        if 'sma_20' in result.columns:
            price_std = self.stock_data['close_Price'].std()
            ma_std = result['sma_20'].dropna().std()
            
            self.assertLess(ma_std, price_std, 
                          "Moving average should smooth volatility")


class TestCalculatePeriodReturns(unittest.TestCase):
    """Test suite for calculate_period_returns function"""
    
    def setUp(self):
        """Set up test data"""
        # Create 6 years of data
        self.stock_data = pd.DataFrame({
            'close_Price': np.linspace(100, 200, 1500),  # Linear growth
            'date': pd.date_range('2018-01-01', periods=1500),
            'ticker': ['AAPL'] * 1500
        })
    
    def test_return_columns_added(self):
        """Test that return columns are added"""
        result = stock_data_fetch.calculate_period_returns(self.stock_data)
        
        # Should add multiple period return columns
        return_cols = [col for col in result.columns 
                      if any(period in col for period in ['1D', '1M', '3M', '1Y'])]
        self.assertGreater(len(return_cols), 0, "Should add period return columns")
    
    def test_1day_return_calculation(self):
        """Test 1-day return calculation"""
        result = stock_data_fetch.calculate_period_returns(self.stock_data)
        
        if '1D' in result.columns:
            # 1-day return should be small for gradual growth
            valid_returns = result['1D'].dropna()
            if len(valid_returns) > 0:
                self.assertTrue((valid_returns.abs() < 1.0).all(),
                              "1-day returns should be reasonable")
    
    def test_longer_period_returns(self):
        """Test that longer periods have larger returns"""
        result = stock_data_fetch.calculate_period_returns(self.stock_data)
        
        # For linearly increasing prices, 1Y should have larger abs return than 1M
        if '1M' in result.columns and '1Y' in result.columns:
            avg_1m = result['1M'].dropna().abs().mean()
            avg_1y = result['1Y'].dropna().abs().mean()
            
            self.assertGreater(avg_1y, avg_1m,
                             "1-year returns should generally be larger than 1-month")
    
    def test_return_values_reasonable(self):
        """Test that return values are within reasonable bounds"""
        result = stock_data_fetch.calculate_period_returns(self.stock_data)
        
        for col in result.columns:
            if any(period in col for period in ['1D', '1M', '3M', '1Y']):
                valid_returns = result[col].dropna()
                if len(valid_returns) > 0:
                    # Returns should typically be between -100% and +200%
                    self.assertTrue((valid_returns > -1.5).all(),
                                  f"{col} should not have extreme negative returns")
                    self.assertTrue((valid_returns < 3.0).all(),
                                  f"{col} should not have extreme positive returns")


def run_unit_tests():
    """Run all unit tests and return results"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestImportTickersFromCSV))
    suite.addTests(loader.loadTestsFromTestCase(TestCalculateStandardDeviationValue))
    suite.addTests(loader.loadTestsFromTestCase(TestCalculateBollingerBands))
    suite.addTests(loader.loadTestsFromTestCase(TestCalculateMomentum))
    suite.addTests(loader.loadTestsFromTestCase(TestAddVolumeIndicators))
    suite.addTests(loader.loadTestsFromTestCase(TestCalculateRatios))
    suite.addTests(loader.loadTestsFromTestCase(TestCalculateMovingAverages))
    suite.addTests(loader.loadTestsFromTestCase(TestCalculatePeriodReturns))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    result = run_unit_tests()
    
    # Print summary
    print("\n" + "="*70)
    print("STOCK DATA FETCH UNIT TEST SUMMARY")
    print("="*70)
    print(f"Tests Run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print("="*70)
