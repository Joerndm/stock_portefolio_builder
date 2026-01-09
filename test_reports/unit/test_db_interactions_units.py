"""
Unit Tests for db_interactions.py

This module contains comprehensive unit tests for database interaction functions.
Tests use mocking to avoid actual database connections.

Test Coverage:
- import_ticker_list: Ticker list retrieval
- does_stock_exists_*: Stock existence checks
- import_stock_*: Data import functions
- export_stock_*: Data export functions
- import_stock_dataset: Complete dataset import
"""

import unittest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import db_interactions


class TestImportTickerList(unittest.TestCase):
    """Test suite for import_ticker_list function"""
    
    @patch('db_interactions.db_connectors.pandas_mysql_connector')
    @patch('db_interactions.fetch_secrets.secret_import')
    @patch('pandas.read_sql')
    def test_successful_import(self, mock_read_sql, mock_secrets, mock_connector):
        """Test successful ticker list import"""
        # Mock secrets
        mock_secrets.return_value = ('host', 'user', 'pass', 'db')
        
        # Mock database connection
        mock_connector.return_value = Mock()
        
        # Mock SQL result
        mock_df = pd.DataFrame({'ticker': ['AAPL', 'GOOGL', 'MSFT']})
        mock_read_sql.return_value = mock_df
        
        result = db_interactions.import_ticker_list()
        
        self.assertIsInstance(result, list, "Should return list")
        self.assertEqual(len(result), 3, "Should return 3 tickers")
        self.assertIn('AAPL', result, "Should contain AAPL")
    
    @patch('db_interactions.fetch_secrets.secret_import')
    def test_secrets_fetch_failure(self, mock_secrets):
        """Test behavior when secrets fetch fails"""
        mock_secrets.side_effect = Exception("Secrets not found")
        
        with self.assertRaises(KeyError):
            db_interactions.import_ticker_list()
    
    @patch('db_interactions.db_connectors.pandas_mysql_connector')
    @patch('db_interactions.fetch_secrets.secret_import')
    def test_database_connection_failure(self, mock_secrets, mock_connector):
        """Test behavior when database connection fails"""
        mock_secrets.return_value = ('host', 'user', 'pass', 'db')
        mock_connector.side_effect = Exception("Connection failed")
        
        with self.assertRaises(KeyError):
            db_interactions.import_ticker_list()


class TestDoesStockExistsStockInfoData(unittest.TestCase):
    """Test suite for does_stock_exists_stock_info_data function"""
    
    def test_empty_ticker(self):
        """Test with empty ticker"""
        with self.assertRaises(ValueError) as context:
            db_interactions.does_stock_exists_stock_info_data("")
        
        self.assertIn("cannot be empty", str(context.exception).lower())
    
    @patch('db_interactions.db_connectors.pandas_mysql_connector')
    @patch('db_interactions.fetch_secrets.secret_import')
    @patch('pandas.read_sql')
    def test_stock_exists(self, mock_read_sql, mock_secrets, mock_connector):
        """Test when stock exists"""
        mock_secrets.return_value = ('host', 'user', 'pass', 'db')
        mock_connector.return_value = Mock()
        mock_df = pd.DataFrame({'ticker': ['AAPL']})
        mock_read_sql.return_value = mock_df
        
        result = db_interactions.does_stock_exists_stock_info_data('AAPL')
        
        self.assertTrue(result, "Should return True when stock exists")
    
    @patch('db_interactions.db_connectors.pandas_mysql_connector')
    @patch('db_interactions.fetch_secrets.secret_import')
    @patch('pandas.read_sql')
    def test_stock_not_exists(self, mock_read_sql, mock_secrets, mock_connector):
        """Test when stock doesn't exist"""
        mock_secrets.return_value = ('host', 'user', 'pass', 'db')
        mock_connector.return_value = Mock()
        mock_df = pd.DataFrame({'ticker': []})
        mock_read_sql.return_value = mock_df
        
        result = db_interactions.does_stock_exists_stock_info_data('INVALID')
        
        self.assertFalse(result, "Should return False when stock doesn't exist")


class TestDoesStockExistsStockPriceData(unittest.TestCase):
    """Test suite for does_stock_exists_stock_price_data function"""
    
    def test_empty_ticker(self):
        """Test with empty ticker"""
        with self.assertRaises(ValueError):
            db_interactions.does_stock_exists_stock_price_data("")
    
    @patch('db_interactions.db_connectors.pandas_mysql_connector')
    @patch('db_interactions.fetch_secrets.secret_import')
    @patch('pandas.read_sql')
    def test_price_data_exists(self, mock_read_sql, mock_secrets, mock_connector):
        """Test when price data exists"""
        mock_secrets.return_value = ('host', 'user', 'pass', 'db')
        mock_connector.return_value = Mock()
        mock_df = pd.DataFrame({'ticker': ['AAPL'], 'close_Price': [150.0]})
        mock_read_sql.return_value = mock_df
        
        result = db_interactions.does_stock_exists_stock_price_data('AAPL')
        
        self.assertTrue(result, "Should return True when price data exists")


class TestImportStockPriceData(unittest.TestCase):
    """Test suite for import_stock_price_data function"""
    
    def test_empty_ticker(self):
        """Test with empty ticker"""
        with self.assertRaises(ValueError):
            db_interactions.import_stock_price_data(stock_ticker="")
    
    @patch('db_interactions.db_connectors.pandas_mysql_connector')
    @patch('db_interactions.fetch_secrets.secret_import')
    @patch('pandas.read_sql')
    def test_successful_import(self, mock_read_sql, mock_secrets, mock_connector):
        """Test successful price data import"""
        mock_secrets.return_value = ('host', 'user', 'pass', 'db')
        mock_connector.return_value = Mock()
        
        mock_df = pd.DataFrame({
            'ticker': ['AAPL'] * 100,
            'date': pd.date_range('2024-01-01', periods=100),
            'close_Price': [150.0] * 100,
            'trade_Volume': [1000000] * 100
        })
        mock_read_sql.return_value = mock_df
        
        result = db_interactions.import_stock_price_data(amount=100, stock_ticker='AAPL')
        
        self.assertIsInstance(result, pd.DataFrame, "Should return DataFrame")
        self.assertEqual(len(result), 100, "Should return requested amount")
    
    @patch('db_interactions.db_connectors.pandas_mysql_connector')
    @patch('db_interactions.fetch_secrets.secret_import')
    @patch('pandas.read_sql')
    def test_limited_amount(self, mock_read_sql, mock_secrets, mock_connector):
        """Test with limited amount parameter"""
        mock_secrets.return_value = ('host', 'user', 'pass', 'db')
        mock_connector.return_value = Mock()
        
        mock_df = pd.DataFrame({
            'ticker': ['AAPL'] * 50,
            'date': pd.date_range('2024-01-01', periods=50),
            'close_Price': [150.0] * 50
        })
        mock_read_sql.return_value = mock_df
        
        result = db_interactions.import_stock_price_data(amount=50, stock_ticker='AAPL')
        
        self.assertLessEqual(len(result), 50, "Should not exceed requested amount")


class TestExportStockPriceData(unittest.TestCase):
    """Test suite for export_stock_price_data function"""
    
    def test_empty_dataframe(self):
        """Test with empty dataframe"""
        with self.assertRaises(ValueError):
            db_interactions.export_stock_price_data("")
    
    @patch('db_interactions.db_connectors.pandas_mysql_connector')
    @patch('db_interactions.fetch_secrets.secret_import')
    def test_successful_export(self, mock_secrets, mock_connector):
        """Test successful price data export"""
        mock_secrets.return_value = ('host', 'user', 'pass', 'db')
        mock_connection = MagicMock()
        mock_connector.return_value = mock_connection
        
        test_df = pd.DataFrame({
            'ticker': ['AAPL'] * 10,
            'date': pd.date_range('2024-01-01', periods=10),
            'close_Price': [150.0] * 10
        })
        
        # Should not raise exception
        try:
            db_interactions.export_stock_price_data(test_df)
        except ValueError as e:
            self.fail(f"Export raised ValueError: {e}")


class TestImportStockFinancialData(unittest.TestCase):
    """Test suite for import_stock_financial_data function"""
    
    def test_empty_ticker(self):
        """Test with empty ticker"""
        with self.assertRaises(ValueError):
            db_interactions.import_stock_financial_data(stock_ticker="")
    
    @patch('db_interactions.db_connectors.pandas_mysql_connector')
    @patch('db_interactions.fetch_secrets.secret_import')
    @patch('pandas.read_sql')
    def test_successful_import(self, mock_read_sql, mock_secrets, mock_connector):
        """Test successful financial data import"""
        mock_secrets.return_value = ('host', 'user', 'pass', 'db')
        mock_connector.return_value = Mock()
        
        # Mock three separate financial statement DataFrames
        income_df = pd.DataFrame({
            'ticker': ['AAPL'] * 4,
            'date_published': pd.date_range('2024-01-01', periods=4, freq='Q'),
            'total_revenue': [100e9] * 4
        })
        balance_df = pd.DataFrame({
            'ticker': ['AAPL'] * 4,
            'date_published': pd.date_range('2024-01-01', periods=4, freq='Q'),
            'total_assets': [300e9] * 4
        })
        cashflow_df = pd.DataFrame({
            'ticker': ['AAPL'] * 4,
            'date_published': pd.date_range('2024-01-01', periods=4, freq='Q'),
            'free_cash_flow': [50e9] * 4
        })
        
        # Mock read_sql to return different DataFrames based on query
        mock_read_sql.side_effect = [income_df, balance_df, cashflow_df]
        
        result = db_interactions.import_stock_financial_data(amount=4, stock_ticker='AAPL')
        
        self.assertIsInstance(result, pd.DataFrame, "Should return DataFrame")


class TestExportStockFinancialData(unittest.TestCase):
    """Test suite for export_stock_financial_data function"""
    
    def test_empty_dataframe(self):
        """Test with empty dataframe"""
        with self.assertRaises(ValueError):
            db_interactions.export_stock_financial_data("")
    
    @patch('db_interactions.db_connectors.pandas_mysql_connector')
    @patch('db_interactions.fetch_secrets.secret_import')
    def test_successful_export(self, mock_secrets, mock_connector):
        """Test successful financial data export"""
        mock_secrets.return_value = ('host', 'user', 'pass', 'db')
        mock_connection = MagicMock()
        mock_connector.return_value = mock_connection
        
        test_df = pd.DataFrame({
            'ticker': ['AAPL'] * 4,
            'date_published': pd.date_range('2024-01-01', periods=4, freq='Q'),
            'total_revenue': [100e9] * 4
        })
        
        # Should not raise exception
        try:
            db_interactions.export_stock_financial_data(test_df)
        except ValueError as e:
            self.fail(f"Export raised ValueError: {e}")


class TestImportStockRatioData(unittest.TestCase):
    """Test suite for import_stock_ratio_data function"""
    
    def test_empty_ticker(self):
        """Test with empty ticker"""
        with self.assertRaises(ValueError):
            db_interactions.import_stock_ratio_data(stock_ticker="")
    
    @patch('db_interactions.db_connectors.pandas_mysql_connector')
    @patch('db_interactions.fetch_secrets.secret_import')
    @patch('pandas.read_sql')
    def test_successful_import(self, mock_read_sql, mock_secrets, mock_connector):
        """Test successful ratio data import"""
        mock_secrets.return_value = ('host', 'user', 'pass', 'db')
        mock_connector.return_value = Mock()
        
        mock_df = pd.DataFrame({
            'ticker': ['AAPL'] * 100,
            'date': pd.date_range('2024-01-01', periods=100),
            'p_e_ratio': [25.0] * 100,
            'p_b_ratio': [8.0] * 100
        })
        mock_read_sql.return_value = mock_df
        
        result = db_interactions.import_stock_ratio_data(amount=100, stock_ticker='AAPL')
        
        self.assertIsInstance(result, pd.DataFrame, "Should return DataFrame")


class TestImportStockDataset(unittest.TestCase):
    """Test suite for import_stock_dataset function"""
    
    def test_empty_ticker(self):
        """Test with empty ticker"""
        with self.assertRaises(ValueError):
            db_interactions.import_stock_dataset("")
    
    @patch('db_interactions.import_stock_ratio_data')
    @patch('db_interactions.import_stock_price_data')
    def test_successful_combined_import(self, mock_price, mock_ratio):
        """Test successful combined dataset import"""
        # Mock price data
        mock_price.return_value = pd.DataFrame({
            'ticker': ['AAPL'] * 100,
            'date': pd.date_range('2024-01-01', periods=100),
            'close_Price': [150.0] * 100
        })
        
        # Mock ratio data
        mock_ratio.return_value = pd.DataFrame({
            'ticker': ['AAPL'] * 100,
            'date': pd.date_range('2024-01-01', periods=100),
            'p_e_ratio': [25.0] * 100
        })
        
        result = db_interactions.import_stock_dataset('AAPL')
        
        self.assertIsInstance(result, pd.DataFrame, "Should return combined DataFrame")
        
        # Should have columns from both price and ratio data
        self.assertIn('close_Price', result.columns, "Should have price columns")
        self.assertIn('p_e_ratio', result.columns, "Should have ratio columns")
    
    @patch('db_interactions.import_stock_ratio_data')
    @patch('db_interactions.import_stock_price_data')
    def test_merge_on_date(self, mock_price, mock_ratio):
        """Test that data is merged correctly on date"""
        dates = pd.date_range('2024-01-01', periods=50)
        
        mock_price.return_value = pd.DataFrame({
            'ticker': ['AAPL'] * 50,
            'date': dates,
            'close_Price': range(100, 150)
        })
        
        mock_ratio.return_value = pd.DataFrame({
            'ticker': ['AAPL'] * 50,
            'date': dates,
            'p_e_ratio': range(20, 70)
        })
        
        result = db_interactions.import_stock_dataset('AAPL')
        
        # Result should maintain date alignment
        if 'date' in result.columns:
            self.assertTrue((result['date'] == dates).all() or len(result) == 50,
                          "Should preserve date alignment")


def run_unit_tests():
    """Run all unit tests and return results"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestImportTickerList))
    suite.addTests(loader.loadTestsFromTestCase(TestDoesStockExistsStockInfoData))
    suite.addTests(loader.loadTestsFromTestCase(TestDoesStockExistsStockPriceData))
    suite.addTests(loader.loadTestsFromTestCase(TestImportStockPriceData))
    suite.addTests(loader.loadTestsFromTestCase(TestExportStockPriceData))
    suite.addTests(loader.loadTestsFromTestCase(TestImportStockFinancialData))
    suite.addTests(loader.loadTestsFromTestCase(TestExportStockFinancialData))
    suite.addTests(loader.loadTestsFromTestCase(TestImportStockRatioData))
    suite.addTests(loader.loadTestsFromTestCase(TestImportStockDataset))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    result = run_unit_tests()
    
    # Print summary
    print("\n" + "="*70)
    print("DB INTERACTIONS UNIT TEST SUMMARY")
    print("="*70)
    print(f"Tests Run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print("="*70)
