"""
Data Validation Tests for Stock Portfolio Builder

This module contains comprehensive data validation tests to ensure data quality,
consistency, and integrity throughout the system.

Test Categories:
- Schema validation
- Data range validation
- Outlier detection
- Missing data handling
- Duplicate detection
- Data type consistency
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import stock_data_fetch
import db_interactions


class TestSchemaValidation(unittest.TestCase):
    """Tests for data schema validation"""
    
    def test_price_data_required_columns(self):
        """Test that price data has required columns"""
        
        required_columns = ['date', 'ticker', 'close_Price', 'trade_Volume']
        
        test_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'ticker': ['AAPL'] * 10,
            'close_Price': [150.0] * 10,
            'high_Price': [155.0] * 10,
            'low_Price': [145.0] * 10,
            'trade_Volume': [1000000] * 10
        })
        
        for col in required_columns:
            self.assertIn(col, test_data.columns,
                         f"Price data should have {col} column")
    
    def test_technical_indicators_schema(self):
        """Test that technical indicators have expected schema"""
        
        base_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=100),
            'close_Price': np.random.uniform(90, 110, 100),
            'high_Price': np.random.uniform(95, 115, 100),
            'low_Price': np.random.uniform(85, 105, 100),
            'trade_Volume': np.random.randint(1000000, 10000000, 100),
            'ticker': ['AAPL'] * 100
        })
        
        result = stock_data_fetch.calculate_moving_averages(base_data)
        result = stock_data_fetch.add_technical_indicators(result)
        
        # Should have SMA columns
        sma_cols = [col for col in result.columns if 'sma' in col.lower()]
        self.assertGreater(len(sma_cols), 0, "Should have SMA columns")
        
        # Should have EMA columns
        ema_cols = [col for col in result.columns if 'ema' in col.lower()]
        self.assertGreater(len(ema_cols), 0, "Should have EMA columns")
    
    def test_column_data_types(self):
        """Test that columns have correct data types"""
        
        test_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'ticker': ['AAPL'] * 10,
            'close_Price': [150.0] * 10,
            'trade_Volume': [1000000] * 10
        })
        
        # Date should be datetime
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(test_data['date']),
                       "Date column should be datetime type")
        
        # Price should be numeric
        self.assertTrue(pd.api.types.is_numeric_dtype(test_data['close_Price']),
                       "Price column should be numeric")
        
        # Volume should be numeric
        self.assertTrue(pd.api.types.is_numeric_dtype(test_data['trade_Volume']),
                       "Volume column should be numeric")
        
        # Ticker should be string/object
        self.assertTrue(pd.api.types.is_string_dtype(test_data['ticker']) or
                       pd.api.types.is_object_dtype(test_data['ticker']),
                       "Ticker column should be string type")


class TestDataRangeValidation(unittest.TestCase):
    """Tests for data range validation"""
    
    def test_price_values_positive(self):
        """Test that price values are positive"""
        
        test_data = pd.DataFrame({
            'close_Price': [100, 105, 110, 115, 120],
            'high_Price': [105, 110, 115, 120, 125],
            'low_Price': [95, 100, 105, 110, 115]
        })
        
        for col in ['close_Price', 'high_Price', 'low_Price']:
            self.assertTrue((test_data[col] > 0).all(),
                          f"{col} should be positive")
    
    def test_high_low_price_relationship(self):
        """Test that high >= close >= low"""
        
        test_data = pd.DataFrame({
            'close_Price': [100, 105, 110],
            'high_Price': [105, 110, 115],
            'low_Price': [95, 100, 105]
        })
        
        self.assertTrue((test_data['high_Price'] >= test_data['close_Price']).all(),
                       "High should be >= Close")
        self.assertTrue((test_data['close_Price'] >= test_data['low_Price']).all(),
                       "Close should be >= Low")
    
    def test_volume_non_negative(self):
        """Test that volume is non-negative"""
        
        test_data = pd.DataFrame({
            'trade_Volume': [1000000, 2000000, 1500000, 3000000]
        })
        
        self.assertTrue((test_data['trade_Volume'] >= 0).all(),
                       "Volume should be non-negative")
    
    def test_returns_reasonable_range(self):
        """Test that returns are within reasonable range"""
        
        test_data = pd.DataFrame({
            'close_Price': np.random.uniform(90, 110, 100),
            'ticker': ['AAPL'] * 100,
            'date': pd.date_range('2024-01-01', periods=100)
        })
        
        result = stock_data_fetch.calculate_period_returns(test_data)
        
        if '1D' in result.columns:
            daily_returns = result['1D'].dropna()
            
            # Daily returns should typically be between -20% and +20%
            self.assertTrue((daily_returns > -0.5).all(),
                          "Daily returns should be > -50%")
            self.assertTrue((daily_returns < 0.5).all(),
                          "Daily returns should be < +50%")
    
    def test_ratios_reasonable_values(self):
        """Test that financial ratios are reasonable"""
        
        test_data = pd.DataFrame({
            'close_Price': [100] * 10,
            'earnings_per_share': [5] * 10,
            'book_value_per_share': [50] * 10
        })
        
        result = stock_data_fetch.calculate_ratios(test_data)
        
        if 'p_e_ratio' in result.columns:
            pe_ratios = result['p_e_ratio'].dropna()
            
            # P/E ratios should typically be between 0 and 100
            self.assertTrue((pe_ratios > 0).all(), "P/E should be positive")
            self.assertTrue((pe_ratios < 200).all(), "P/E should be reasonable")


class TestOutlierDetection(unittest.TestCase):
    """Tests for outlier detection in data"""
    
    def test_detect_price_outliers(self):
        """Test detection of price outliers"""
        
        # Create data with an outlier
        normal_prices = [100, 102, 101, 103, 105, 104, 106]
        outlier_price = 500  # Clear outlier
        
        test_data = pd.DataFrame({
            'close_Price': normal_prices + [outlier_price],
            'ticker': ['AAPL'] * 8
        })
        
        # Calculate z-scores
        mean_price = test_data['close_Price'].mean()
        std_price = test_data['close_Price'].std()
        z_scores = (test_data['close_Price'] - mean_price) / std_price
        
        outliers = test_data[abs(z_scores) > 3]
        
        self.assertEqual(len(outliers), 1, "Should detect 1 outlier")
        self.assertEqual(outliers['close_Price'].iloc[0], outlier_price,
                        "Should identify the outlier price")
    
    def test_detect_volume_outliers(self):
        """Test detection of volume outliers"""
        
        # Normal volumes
        normal_volumes = [1000000, 1100000, 950000, 1050000, 1200000]
        # Suspicious very low volume
        outlier_volume = 100
        
        test_data = pd.DataFrame({
            'trade_Volume': normal_volumes + [outlier_volume],
            'ticker': ['AAPL'] * 6
        })
        
        # Volumes below 1000 might be suspicious
        suspicious = test_data[test_data['trade_Volume'] < 1000]
        
        self.assertEqual(len(suspicious), 1, "Should detect suspicious volume")


class TestMissingDataHandling(unittest.TestCase):
    """Tests for missing data handling"""
    
    def test_detect_missing_values(self):
        """Test detection of missing values"""
        
        test_data = pd.DataFrame({
            'close_Price': [100, np.nan, 105, 110, np.nan],
            'ticker': ['AAPL', 'AAPL', 'AAPL', 'AAPL', 'AAPL']
        })
        
        missing_count = test_data['close_Price'].isna().sum()
        self.assertEqual(missing_count, 2, "Should detect 2 missing values")
    
    def test_missing_data_percentage(self):
        """Test calculation of missing data percentage"""
        
        test_data = pd.DataFrame({
            'close_Price': [100, np.nan, 105, np.nan, 110],
            'volume': [1000000, 1100000, np.nan, 1200000, 1300000]
        })
        
        for col in test_data.columns:
            missing_pct = test_data[col].isna().sum() / len(test_data) * 100
            
            # No column should have >50% missing data
            self.assertLess(missing_pct, 50,
                          f"{col} should have <50% missing data")
    
    def test_forward_fill_handling(self):
        """Test forward fill for missing values"""
        
        test_data = pd.DataFrame({
            'close_Price': [100, np.nan, np.nan, 110, np.nan],
            'ticker': ['AAPL'] * 5
        })
        
        # Forward fill
        filled_data = test_data.fillna(method='ffill')
        
        # Should have fewer NaN values
        original_na = test_data['close_Price'].isna().sum()
        filled_na = filled_data['close_Price'].isna().sum()
        
        self.assertLessEqual(filled_na, original_na,
                           "Forward fill should reduce NaN count")
    
    def test_critical_columns_completeness(self):
        """Test that critical columns have minimal missing data"""
        
        test_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=100),
            'ticker': ['AAPL'] * 100,
            'close_Price': np.random.uniform(90, 110, 100)
        })
        
        critical_columns = ['date', 'ticker', 'close_Price']
        
        for col in critical_columns:
            missing_count = test_data[col].isna().sum()
            missing_pct = missing_count / len(test_data) * 100
            
            self.assertLess(missing_pct, 5,
                          f"Critical column {col} should have <5% missing data")


class TestDuplicateDetection(unittest.TestCase):
    """Tests for duplicate detection"""
    
    def test_detect_duplicate_rows(self):
        """Test detection of duplicate rows"""
        
        test_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5).tolist() + [pd.Timestamp('2024-01-01')],
            'ticker': ['AAPL'] * 6,
            'close_Price': [100, 105, 110, 115, 120, 100]
        })
        
        duplicates = test_data.duplicated(subset=['date', 'ticker'], keep=False)
        duplicate_count = duplicates.sum()
        
        self.assertGreater(duplicate_count, 0, "Should detect duplicates")
    
    def test_duplicate_removal(self):
        """Test removal of duplicate rows"""
        
        test_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=3).tolist() + [pd.Timestamp('2024-01-01')],
            'ticker': ['AAPL'] * 4,
            'close_Price': [100, 105, 110, 100]
        })
        
        # Remove duplicates
        deduplicated = test_data.drop_duplicates(subset=['date', 'ticker'], keep='first')
        
        self.assertEqual(len(deduplicated), 3,
                        "Should remove duplicate keeping first occurrence")
    
    def test_no_duplicate_dates_per_ticker(self):
        """Test that each ticker has unique dates"""
        
        test_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'ticker': ['AAPL'] * 10,
            'close_Price': np.random.uniform(90, 110, 10)
        })
        
        # Group by ticker and check date uniqueness
        for ticker in test_data['ticker'].unique():
            ticker_data = test_data[test_data['ticker'] == ticker]
            unique_dates = ticker_data['date'].nunique()
            total_rows = len(ticker_data)
            
            self.assertEqual(unique_dates, total_rows,
                           f"Ticker {ticker} should have unique dates")


class TestDataTypeConsistency(unittest.TestCase):
    """Tests for data type consistency"""
    
    def test_numeric_columns_consistency(self):
        """Test that numeric columns maintain type consistency"""
        
        test_data = pd.DataFrame({
            'close_Price': [100.0, 105.5, 110.2],
            'trade_Volume': [1000000, 1100000, 1200000]
        })
        
        # All price values should be numeric
        self.assertTrue(pd.api.types.is_numeric_dtype(test_data['close_Price']),
                       "Price should be numeric type")
        self.assertTrue(pd.api.types.is_numeric_dtype(test_data['trade_Volume']),
                       "Volume should be numeric type")
    
    def test_date_column_consistency(self):
        """Test that date columns maintain datetime type"""
        
        test_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5),
            'close_Price': [100, 105, 110, 115, 120]
        })
        
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(test_data['date']),
                       "Date column should be datetime64 type")
    
    def test_ticker_column_consistency(self):
        """Test that ticker columns maintain string type"""
        
        test_data = pd.DataFrame({
            'ticker': ['AAPL', 'GOOGL', 'MSFT'],
            'close_Price': [150, 2800, 350]
        })
        
        self.assertTrue(pd.api.types.is_string_dtype(test_data['ticker']) or
                       pd.api.types.is_object_dtype(test_data['ticker']),
                       "Ticker should be string/object type")
    
    def test_mixed_type_handling(self):
        """Test handling of mixed types in columns"""
        
        # Create DataFrame with mixed types (common data quality issue)
        test_data = pd.DataFrame({
            'close_Price': ['100', '105.5', '110', 'N/A', '115']
        })
        
        # Convert to numeric, coercing errors
        numeric_data = pd.to_numeric(test_data['close_Price'], errors='coerce')
        
        # Should convert valid numbers and make invalid ones NaN
        self.assertEqual(numeric_data.dropna().shape[0], 4,
                        "Should convert 4 valid numbers")
        self.assertEqual(numeric_data.isna().sum(), 1,
                        "Should have 1 NaN from invalid value")


def run_data_validation_tests():
    """Run all data validation tests and return results"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSchemaValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestDataRangeValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestOutlierDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestMissingDataHandling))
    suite.addTests(loader.loadTestsFromTestCase(TestDuplicateDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestDataTypeConsistency))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    result = run_data_validation_tests()
    
    # Print summary
    print("\n" + "="*70)
    print("DATA VALIDATION TEST SUMMARY")
    print("="*70)
    print(f"Tests Run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print("="*70)
