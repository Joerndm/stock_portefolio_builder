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
    
    @patch('db_interactions.db_connectors.pandas_mysql_connector')
    @patch('db_interactions.fetch_secrets.secret_import')
    @patch('pandas.read_sql')
    def test_empty_ticker_returns_global_recent_rows(self, mock_read_sql, mock_secrets, mock_connector):
        """Empty ticker should return recent rows across all tickers."""
        mock_secrets.return_value = ('host', 'user', 'pass', 'db')
        mock_connector.return_value = Mock()
        mock_read_sql.return_value = pd.DataFrame({'ticker': ['AAPL'], 'close_Price': [150.0]})

        result = db_interactions.import_stock_price_data(stock_ticker="")

        self.assertIsInstance(result, pd.DataFrame)
    
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
    
    @patch('pandas.DataFrame.to_sql')
    @patch('db_interactions.db_connectors.pandas_mysql_connector')
    @patch('db_interactions.fetch_secrets.secret_import')
    def test_successful_export(self, mock_secrets, mock_connector, mock_to_sql):
        """Test successful price data export"""
        mock_secrets.return_value = ('host', 'user', 'pass', 'db')
        mock_connection = MagicMock()
        mock_connector.return_value = mock_connection
        mock_connection.begin.return_value.__enter__.return_value = MagicMock()
        
        test_df = pd.DataFrame({
            'ticker': ['AAPL'] * 60,
            'date': pd.date_range('2024-01-01', periods=60),
            'close_Price': [150.0] * 60,
            'open_Price': [149.0] * 60,
            'high_Price': [151.0] * 60,
            'low_Price': [148.5] * 60,
            'trade_Volume': [1000000] * 60,
        })
        
        # Should not raise exception
        try:
            db_interactions.export_stock_price_data(test_df)
        except ValueError as e:
            self.fail(f"Export raised ValueError: {e}")

        mock_to_sql.assert_called_once()

    @patch('db_interactions.db_connectors.pandas_mysql_connector')
    @patch('db_interactions.fetch_secrets.secret_import')
    def test_rejects_multi_ticker_export_batch(self, mock_secrets, mock_connector):
        """Export batches should only contain one ticker."""
        mock_secrets.return_value = ('host', 'user', 'pass', 'db')
        mock_connector.return_value = MagicMock()

        test_df = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT'],
            'date': pd.to_datetime(['2024-01-01', '2024-01-01']),
            'close_Price': [150.0, 250.0],
            'open_Price': [149.0, 249.0],
            'high_Price': [151.0, 251.0],
            'low_Price': [148.5, 248.5],
            'trade_Volume': [1000000, 2000000],
        })

        with self.assertRaises(KeyError) as context:
            db_interactions.export_stock_price_data(test_df)

        self.assertIn('exactly one unique ticker', str(context.exception))

    @patch('db_interactions.db_connectors.pandas_mysql_connector')
    @patch('db_interactions.fetch_secrets.secret_import')
    def test_rejects_duplicate_date_ticker_rows(self, mock_secrets, mock_connector):
        """Export batches should reject duplicate primary-key rows before writing."""
        mock_secrets.return_value = ('host', 'user', 'pass', 'db')
        mock_connector.return_value = MagicMock()

        test_df = pd.DataFrame({
            'ticker': ['AAPL', 'AAPL'],
            'date': pd.to_datetime(['2024-01-01', '2024-01-01']),
            'close_Price': [150.0, 151.0],
            'open_Price': [149.0, 150.0],
            'high_Price': [151.0, 152.0],
            'low_Price': [148.5, 149.5],
            'trade_Volume': [1000000, 1000000],
        })

        with self.assertRaises(KeyError) as context:
            db_interactions.export_stock_price_data(test_df)

        self.assertIn('duplicate (date, ticker) rows', str(context.exception))


class TestGetNewestFinancialDate(unittest.TestCase):
    """Focused tests for newest financial date lookup."""

    @patch('db_interactions.db_connectors.pandas_mysql_connector')
    @patch('db_interactions.fetch_secrets.secret_import')
    @patch('pandas.read_sql')
    def test_prefers_newer_quarterly_date_when_requested(self, mock_read_sql, mock_secrets, mock_connector):
        """Quarterly lookups should use the ticker filter and override older annual data."""
        mock_secrets.return_value = ('host', 'user', 'pass', 'db')
        mock_connector.return_value = MagicMock()

        def fake_read_sql(sql, con, params=None, **_kwargs):
            sql_text = str(sql)
            if 'stock_income_stmt_data' in sql_text:
                self.assertEqual(params, {'ticker': 'AAPL'})
                return pd.DataFrame({'newest_date': [pd.Timestamp('2024-12-31')]})
            if 'stock_income_stmt_quarterly' in sql_text:
                self.assertEqual(params, {'ticker': 'AAPL'})
                return pd.DataFrame({'newest_date': [pd.Timestamp('2025-03-31')]})
            raise AssertionError(f'Unexpected SQL: {sql_text}')

        mock_read_sql.side_effect = fake_read_sql

        newest_date, source = db_interactions.get_newest_financial_date('AAPL', include_quarterly=True)

        self.assertEqual(str(newest_date), '2025-03-31')
        self.assertEqual(source, 'quarterly')

    @patch('db_interactions.db_connectors.pandas_mysql_connector')
    @patch('db_interactions.fetch_secrets.secret_import')
    @patch('pandas.read_sql')
    def test_returns_annual_date_when_quarterly_is_disabled(self, mock_read_sql, mock_secrets, mock_connector):
        """Annual-only lookups should not depend on quarterly queries."""
        mock_secrets.return_value = ('host', 'user', 'pass', 'db')
        mock_connector.return_value = MagicMock()
        mock_read_sql.return_value = pd.DataFrame({'newest_date': [pd.Timestamp('2024-12-31')]})

        newest_date, source = db_interactions.get_newest_financial_date('AAPL', include_quarterly=False)

        self.assertEqual(str(newest_date), '2024-12-31')
        self.assertEqual(source, 'annual')
        mock_read_sql.assert_called_once()


class TestImportStockFinancialData(unittest.TestCase):
    """Test suite for import_stock_financial_data function"""
    
    @patch('db_interactions.db_connectors.pandas_mysql_connector')
    @patch('db_interactions.fetch_secrets.secret_import')
    @patch('pandas.read_sql')
    def test_empty_ticker_returns_recent_global_financial_rows(self, mock_read_sql, mock_secrets, mock_connector):
        """Empty ticker should return recent merged financial rows across all tickers."""
        mock_secrets.return_value = ('host', 'user', 'pass', 'db')
        mock_connector.return_value = Mock()

        income_df = pd.DataFrame({
            'financial_Statement_Date': pd.to_datetime(['2024-01-01']),
            'ticker': ['AAPL'],
            'revenue': [100.0],
        })
        balance_df = pd.DataFrame({
            'financial_Statement_Date': pd.to_datetime(['2024-01-01']),
            'ticker': ['AAPL'],
            'total_Assets': [300.0],
        })
        cashflow_df = pd.DataFrame({
            'financial_Statement_Date': pd.to_datetime(['2024-01-01']),
            'ticker': ['AAPL'],
            'free_Cash_Flow': [50.0],
        })
        mock_read_sql.side_effect = [income_df, balance_df, cashflow_df]

        result = db_interactions.import_stock_financial_data(stock_ticker="")

        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('date', result.columns)
    
    @patch('db_interactions.does_stock_exists_stock_income_stmt_data')
    @patch('db_interactions.db_connectors.pandas_mysql_connector')
    @patch('db_interactions.fetch_secrets.secret_import')
    @patch('pandas.read_sql')
    def test_successful_import(self, mock_read_sql, mock_secrets, mock_connector, mock_exists):
        """Test successful financial data import"""
        mock_secrets.return_value = ('host', 'user', 'pass', 'db')
        mock_connector.return_value = Mock()
        mock_exists.return_value = True
        
        # Mock three separate financial statement DataFrames
        income_df = pd.DataFrame({
            'ticker': ['AAPL'] * 4,
            'financial_Statement_Date': pd.to_datetime(['2024-03-31', '2024-06-30', '2024-09-30', '2024-12-31']),
            'revenue': [100e9] * 4
        })
        balance_df = pd.DataFrame({
            'ticker': ['AAPL'] * 4,
            'financial_Statement_Date': pd.to_datetime(['2024-03-31', '2024-06-30', '2024-09-30', '2024-12-31']),
            'total_Assets': [300e9] * 4
        })
        cashflow_df = pd.DataFrame({
            'ticker': ['AAPL'] * 4,
            'financial_Statement_Date': pd.to_datetime(['2024-03-31', '2024-06-30', '2024-09-30', '2024-12-31']),
            'free_Cash_Flow': [50e9] * 4
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
    
    @patch('pandas.DataFrame.to_sql')
    @patch('db_interactions.db_connectors.pandas_mysql_connector')
    @patch('db_interactions.fetch_secrets.secret_import')
    def test_successful_export(self, mock_secrets, mock_connector, mock_to_sql):
        """Test successful financial data export"""
        mock_secrets.return_value = ('host', 'user', 'pass', 'db')
        mock_connection = MagicMock()
        mock_connector.return_value = mock_connection
        mock_connection.begin.return_value.__enter__.return_value = MagicMock()
        
        test_df = pd.DataFrame({
            'date': pd.to_datetime(['2024-03-31', '2024-06-30', '2024-09-30', '2024-12-31']),
            'ticker': ['AAPL'] * 4,
            'date_published': pd.to_datetime(['2024-04-15', '2024-07-15', '2024-10-15', '2025-01-15']),
            'revenue': [100e9] * 4,
            'total_Assets': [300e9] * 4,
            'free_Cash_Flow': [50e9] * 4,
        })
        
        # Should not raise exception
        try:
            db_interactions.export_stock_financial_data(test_df)
        except ValueError as e:
            self.fail(f"Export raised ValueError: {e}")

        self.assertEqual(mock_to_sql.call_count, 3)


class TestImportStockRatioData(unittest.TestCase):
    """Test suite for import_stock_ratio_data function"""
    
    @patch('db_interactions.db_connectors.pandas_mysql_connector')
    @patch('db_interactions.fetch_secrets.secret_import')
    @patch('pandas.read_sql')
    def test_empty_ticker_returns_global_recent_ratio_rows(self, mock_read_sql, mock_secrets, mock_connector):
        """Empty ticker should return recent ratio rows across all tickers."""
        mock_secrets.return_value = ('host', 'user', 'pass', 'db')
        mock_connector.return_value = Mock()
        mock_read_sql.return_value = pd.DataFrame({'ticker': ['AAPL'], 'p_e': [25.0]})

        result = db_interactions.import_stock_ratio_data(stock_ticker="")

        self.assertIsInstance(result, pd.DataFrame)
    
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
            'p_e': [25.0] * 100,
            'p_b': [8.0] * 100
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
    
    @patch('db_interactions.does_stock_exists_stock_ratio_data', return_value=True)
    @patch('db_interactions.does_stock_exists_stock_cash_flow_data', return_value=True)
    @patch('db_interactions.does_stock_exists_stock_balancesheet_data', return_value=True)
    @patch('db_interactions.does_stock_exists_stock_income_stmt_data', return_value=True)
    @patch('db_interactions.does_stock_exists_stock_price_data', return_value=True)
    @patch('db_interactions.does_stock_exists_stock_info_data', return_value=True)
    @patch('db_interactions.db_connectors.pandas_mysql_connector')
    @patch('db_interactions.fetch_secrets.secret_import')
    @patch('pandas.read_sql')
    def test_successful_combined_import(
        self,
        mock_read_sql,
        mock_secrets,
        mock_connector,
        _mock_info,
        _mock_price_exists,
        _mock_income_exists,
        _mock_balance_exists,
        _mock_cash_exists,
        _mock_ratio_exists,
    ):
        """Test successful combined dataset import"""
        mock_secrets.return_value = ('host', 'user', 'pass', 'db')
        mock_connector.return_value = Mock()

        dates = pd.date_range('2024-01-01', periods=100)
        price_df = pd.DataFrame({
            'ticker': ['AAPL'] * 100,
            'date': dates,
            'close_Price': [150.0] * 100,
            'open_Price': [149.0] * 100,
            'high_Price': [151.0] * 100,
            'low_Price': [148.0] * 100,
        })
        vix_df = pd.DataFrame({
            'ticker': ['^VIX'] * 100,
            'date': dates,
            'open_Price': [20.0] * 100,
        })
        financial_dates = pd.to_datetime(['2024-03-31', '2024-06-30', '2024-09-30', '2024-12-31'])
        income_df = pd.DataFrame({
            'financial_Statement_Date': financial_dates,
            'date_published': financial_dates,
            'ticker': ['AAPL'] * 4,
            'revenue': [100.0, 110.0, 120.0, 130.0],
        })
        balance_df = pd.DataFrame({
            'financial_Statement_Date': financial_dates,
            'date_published': financial_dates,
            'ticker': ['AAPL'] * 4,
            'book_Value_Per_Share': [10.0, 10.5, 11.0, 11.5],
        })
        cashflow_df = pd.DataFrame({
            'financial_Statement_Date': financial_dates,
            'date_published': financial_dates,
            'ticker': ['AAPL'] * 4,
            'free_Cash_Flow': [50.0, 52.0, 54.0, 56.0],
        })
        ratio_df = pd.DataFrame({
            'ticker': ['AAPL'] * 100,
            'date': dates,
            'p_e': [25.0] * 100,
        })
        mock_read_sql.side_effect = [price_df, vix_df, income_df, balance_df, cashflow_df, ratio_df]
        
        result = db_interactions.import_stock_dataset('AAPL')
        
        self.assertIsInstance(result, pd.DataFrame, "Should return combined DataFrame")
        
        # Should have columns from both price and ratio data
        self.assertIn('close_Price', result.columns, "Should have price columns")
        self.assertIn('p_e', result.columns, "Should have ratio columns")
    
    @patch('db_interactions.does_stock_exists_stock_ratio_data', return_value=True)
    @patch('db_interactions.does_stock_exists_stock_cash_flow_data', return_value=True)
    @patch('db_interactions.does_stock_exists_stock_balancesheet_data', return_value=True)
    @patch('db_interactions.does_stock_exists_stock_income_stmt_data', return_value=True)
    @patch('db_interactions.does_stock_exists_stock_price_data', return_value=True)
    @patch('db_interactions.does_stock_exists_stock_info_data', return_value=True)
    @patch('db_interactions.db_connectors.pandas_mysql_connector')
    @patch('db_interactions.fetch_secrets.secret_import')
    @patch('pandas.read_sql')
    def test_merge_on_date(
        self,
        mock_read_sql,
        mock_secrets,
        mock_connector,
        _mock_info,
        _mock_price_exists,
        _mock_income_exists,
        _mock_balance_exists,
        _mock_cash_exists,
        _mock_ratio_exists,
    ):
        """Test that data is merged correctly on date"""
        dates = pd.date_range('2024-01-01', periods=50)
        mock_secrets.return_value = ('host', 'user', 'pass', 'db')
        mock_connector.return_value = Mock()

        price_df = pd.DataFrame({
            'ticker': ['AAPL'] * 50,
            'date': dates,
            'close_Price': range(100, 150),
            'open_Price': range(99, 149),
            'high_Price': range(101, 151),
            'low_Price': range(98, 148),
        })
        vix_df = pd.DataFrame({
            'ticker': ['^VIX'] * 50,
            'date': dates,
            'open_Price': [20.0] * 50,
        })
        financial_dates = pd.to_datetime(['2024-01-15', '2024-02-15'])
        income_df = pd.DataFrame({
            'financial_Statement_Date': financial_dates,
            'date_published': financial_dates,
            'ticker': ['AAPL'] * 2,
            'revenue': [100.0, 110.0],
        })
        balance_df = pd.DataFrame({
            'financial_Statement_Date': financial_dates,
            'date_published': financial_dates,
            'ticker': ['AAPL'] * 2,
            'book_Value_Per_Share': [10.0, 11.0],
        })
        cashflow_df = pd.DataFrame({
            'financial_Statement_Date': financial_dates,
            'date_published': financial_dates,
            'ticker': ['AAPL'] * 2,
            'free_Cash_Flow': [50.0, 55.0],
        })
        ratio_df = pd.DataFrame({
            'ticker': ['AAPL'] * 50,
            'date': dates,
            'p_e': range(20, 70)
        })
        mock_read_sql.side_effect = [price_df, vix_df, income_df, balance_df, cashflow_df, ratio_df]
        
        result = db_interactions.import_stock_dataset('AAPL')
        
        # Result should maintain date alignment
        if 'date' in result.columns:
            self.assertEqual(len(result), 50, "Should preserve one row per price date")
            self.assertEqual(list(result['date']), list(dates), "Should preserve date alignment")


class TestHyperparameterCacheRows(unittest.TestCase):
    """Regression tests for cached hyperparameter row metadata access."""

    def test_empty_ticker_raises_value_error(self):
        with self.assertRaises(ValueError):
            db_interactions.get_hyperparameter_cache_rows("")

    @patch('db_interactions.db_connectors.pandas_mysql_connector')
    @patch('db_interactions.fetch_secrets.secret_import')
    @patch('pandas.read_sql')
    def test_get_hyperparameter_cache_rows_parses_json_and_filters_model_types(self, mock_read_sql, mock_secrets, mock_connector):
        mock_secrets.return_value = ('host', 'user', 'pass', 'db')
        mock_connector.return_value = Mock()
        mock_read_sql.return_value = pd.DataFrame(
            {
                'ticker': ['AAPL', 'AAPL'],
                'model_type': ['RF', 'ridge'],
                'hyperparameters': ['{"n_estimators": 100}', '{"alpha": 2.5, "solver": "auto"}'],
                'tuning_date': [pd.Timestamp('2024-01-01'), pd.Timestamp('2024-01-02')],
                'feature_hash': ['hash-a', 'hash-b'],
                'num_features': [10, 11],
                'num_trials': [20, 5],
                'best_score': [0.1, 0.2],
                'tuning_time_seconds': [12.5, 3.0],
                'val_mse': [0.02, 0.03],
                'val_r2': [0.6, 0.5],
                'val_mae': [0.1, 0.2],
                'is_constrained': [False, True],
                'is_valid': [True, True],
            }
        )

        rows = db_interactions.get_hyperparameter_cache_rows('AAPL', model_types=['ridge'])

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['model_type'], 'ridge')
        self.assertEqual(rows[0]['hyperparameters'], {'alpha': 2.5, 'solver': 'auto'})


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
    suite.addTests(loader.loadTestsFromTestCase(TestHyperparameterCacheRows))
    
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
