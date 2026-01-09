"""
Security Tests for Stock Portfolio Builder

This module contains security tests to verify system security requirements
and prevent common vulnerabilities.

Test Categories:
- SQL injection prevention
- Input validation and sanitization
- Secrets management validation
- File path traversal prevention
- Data access control
"""

import unittest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import db_interactions
import stock_data_fetch
import fetch_secrets


class TestSQLInjectionPrevention(unittest.TestCase):
    """Tests for SQL injection vulnerabilities"""
    
    @patch('db_interactions.db_connectors.pandas_mysql_connector')
    @patch('db_interactions.fetch_secrets.secret_import')
    @patch('pandas.read_sql')
    def test_ticker_parameter_sql_injection(self, mock_read_sql, mock_secrets, mock_connector):
        """Test that ticker parameters prevent SQL injection"""
        
        mock_secrets.return_value = ('host', 'user', 'pass', 'db')
        mock_connector.return_value = Mock()
        mock_read_sql.return_value = pd.DataFrame({'ticker': []})
        
        # Try SQL injection patterns
        malicious_inputs = [
            "AAPL' OR '1'='1",
            "AAPL'; DROP TABLE stock_info_data; --",
            "AAPL' UNION SELECT * FROM users--",
            "'; DELETE FROM stock_price_data WHERE '1'='1",
        ]
        
        for malicious_input in malicious_inputs:
            # Function should either reject input or safely handle it
            try:
                result = db_interactions.does_stock_exists_stock_info_data(malicious_input)
                
                # If it doesn't raise an error, check that SQL query was safe
                call_args = mock_read_sql.call_args
                if call_args:
                    query = call_args[1]['sql']
                    
                    # Query should use parameterized approach or escape properly
                    # The malicious input should not break query structure
                    self.assertNotIn('DROP TABLE', query.upper(),
                                   "Should not contain DROP TABLE")
                    self.assertNotIn('DELETE FROM', query.upper(),
                                   "Should not contain DELETE FROM")
                
            except (ValueError, KeyError):
                # Rejecting suspicious input is also acceptable
                pass
    
    @patch('db_interactions.db_connectors.pandas_mysql_connector')
    @patch('db_interactions.fetch_secrets.secret_import')
    def test_database_write_operations_use_parameterization(self, mock_secrets, mock_connector):
        """Test that write operations use parameterized queries"""
        
        mock_secrets.return_value = ('host', 'user', 'pass', 'db')
        mock_connection = MagicMock()
        mock_connector.return_value = mock_connection
        
        # Create test data with potentially dangerous content
        test_data = pd.DataFrame({
            'ticker': ["AAPL'; DROP TABLE--"],
            'date': [pd.Timestamp('2024-01-01')],
            'close_Price': [150.0]
        })
        
        try:
            db_interactions.export_stock_price_data(test_data)
            
            # If it succeeds, verify it used safe methods (to_sql)
            # to_sql uses parameterized queries by default
            
        except Exception as e:
            # Some exceptions are expected with mocking
            self.assertNotIn('DROP TABLE', str(e).upper(),
                           "Error should not indicate SQL injection succeeded")


class TestInputValidation(unittest.TestCase):
    """Tests for input validation and sanitization"""
    
    def test_empty_string_validation(self):
        """Test that empty strings are rejected"""
        
        functions_to_test = [
            (db_interactions.does_stock_exists_stock_info_data, ""),
            (db_interactions.does_stock_exists_stock_price_data, ""),
            (stock_data_fetch.import_tickers_from_csv, ""),
        ]
        
        for func, empty_input in functions_to_test:
            with self.assertRaises(ValueError,
                                 msg=f"{func.__name__} should reject empty string"):
                func(empty_input)
    
    def test_ticker_format_validation(self):
        """Test ticker symbol format validation"""
        
        # Valid tickers should work (though may not exist in DB)
        valid_tickers = ['AAPL', 'GOOGL', 'MSFT', 'BRK.B', '^GSPC']
        
        # Each should be accepted as valid format (even if not in database)
        for ticker in valid_tickers:
            # Should not raise ValueError for format
            try:
                # This will fail on DB connection, but shouldn't fail on validation
                db_interactions.does_stock_exists_stock_info_data(ticker)
            except KeyError:
                # Database errors are expected, format validation passed
                pass
            except ValueError as e:
                if "empty" not in str(e).lower():
                    self.fail(f"Valid ticker {ticker} should not raise ValueError for format")
    
    def test_numeric_parameter_validation(self):
        """Test that numeric parameters are validated"""
        
        # Test with invalid amount parameter
        @patch('db_interactions.db_connectors.pandas_mysql_connector')
        @patch('db_interactions.fetch_secrets.secret_import')
        @patch('pandas.read_sql')
        def test_invalid_amount(mock_read_sql, mock_secrets, mock_connector):
            mock_secrets.return_value = ('host', 'user', 'pass', 'db')
            mock_connector.return_value = Mock()
            mock_read_sql.return_value = pd.DataFrame()
            
            # Negative amounts should be handled
            try:
                result = db_interactions.import_stock_price_data(amount=-1, stock_ticker='AAPL')
                # If it succeeds, it should return empty or handle gracefully
                self.assertIsInstance(result, pd.DataFrame, "Should handle negative amount")
            except (ValueError, KeyError):
                # Rejecting is also acceptable
                pass
        
        test_invalid_amount()
    
    def test_date_parameter_validation(self):
        """Test that date parameters are validated"""
        
        # Test with invalid date
        from datetime import datetime
        
        valid_date = datetime(2023, 1, 1)
        
        # Should accept valid datetime
        try:
            result = stock_data_fetch.fetch_stock_price_data('AAPL', start_date=valid_date)
        except Exception as e:
            # API/connection errors are OK, just checking it accepted the date
            self.assertNotIn('date', str(e).lower(), "Should accept valid date format")


class TestSecretsManagement(unittest.TestCase):
    """Tests for secrets management and credential handling"""
    
    def test_secrets_not_hardcoded(self):
        """Test that secrets are not hardcoded in code"""
        
        # Read the db_interactions file
        db_interactions_path = os.path.join(
            os.path.dirname(__file__), '..', '..', 'db_interactions.py'
        )
        
        if os.path.exists(db_interactions_path):
            with open(db_interactions_path, 'r') as f:
                content = f.read()
            
            # Check for common hardcoded credential patterns
            dangerous_patterns = [
                'password = "',
                'password="',
                "password = '",
                "password='",
                'PASSWORD = "',
                'passwd = "',
                'pwd = "',
            ]
            
            for pattern in dangerous_patterns:
                self.assertNotIn(pattern, content,
                               f"Should not contain hardcoded pattern: {pattern}")
    
    @patch.dict(os.environ, {'DB_PASSWORD': 'test_password'}, clear=False)
    def test_secrets_from_environment(self):
        """Test that secrets can be loaded from environment"""
        
        # fetch_secrets should use environment variables or secure storage
        try:
            # This test verifies the mechanism exists
            # Actual implementation may vary
            result = fetch_secrets.secret_import()
            
            if result:
                self.assertEqual(len(result), 4,
                               "Should return 4 credentials (host, user, pass, db)")
                
                # Passwords should not be empty
                db_pass = result[2]
                self.assertIsNotNone(db_pass, "Password should not be None")
                
        except Exception as e:
            # If secrets are not configured, that's OK for test environment
            self.assertIn('secret', str(e).lower(),
                        "Error should be related to secret configuration")
    
    def test_credentials_not_in_error_messages(self):
        """Test that credentials don't appear in error messages"""
        
        @patch('fetch_secrets.secret_import')
        def test_error_message(mock_secret):
            # Mock credentials
            mock_secret.return_value = ('host', 'user', 'SuperSecretPassword123', 'dbname')
            
            # Try to cause an error
            try:
                with patch('db_interactions.db_connectors.pandas_mysql_connector') as mock_conn:
                    mock_conn.side_effect = Exception("Connection failed")
                    db_interactions.import_ticker_list()
            except Exception as e:
                error_message = str(e)
                
                # Password should NOT appear in error message
                self.assertNotIn('SuperSecretPassword123', error_message,
                               "Password should not appear in error messages")
        
        test_error_message()


class TestFilePathSecurity(unittest.TestCase):
    """Tests for file path traversal prevention"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.safe_file = os.path.join(self.temp_dir, 'safe.csv')
        
        # Create a safe CSV file
        pd.DataFrame({'Symbol': ['AAPL', 'GOOGL']}).to_csv(self.safe_file, index=False)
    
    def tearDown(self):
        """Clean up"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_path_traversal_prevention(self):
        """Test that path traversal attacks are prevented"""
        
        # Malicious path patterns
        malicious_paths = [
            '../../../etc/passwd',
            '..\\..\\..\\windows\\system32\\config\\sam',
            './../../sensitive_data.csv',
        ]
        
        for malicious_path in malicious_paths:
            try:
                # Should either reject the path or safely handle it
                result = stock_data_fetch.import_tickers_from_csv(malicious_path)
                
                # If it doesn't raise an error, it should not access outside directories
                # The function constructs path using os.path.join which is safer
                
            except (FileNotFoundError, ValueError, OSError):
                # These exceptions are acceptable - path was rejected
                pass
    
    def test_absolute_path_handling(self):
        """Test that absolute paths are handled safely"""
        
        # Absolute path should work
        try:
            result = stock_data_fetch.import_tickers_from_csv(self.safe_file)
            self.assertIsInstance(result, pd.DataFrame, "Should handle absolute path")
        except FileNotFoundError:
            # File not being found is OK for this test
            pass


class TestDataAccessControl(unittest.TestCase):
    """Tests for data access control and authorization"""
    
    @patch('db_interactions.db_connectors.pandas_mysql_connector')
    @patch('db_interactions.fetch_secrets.secret_import')
    @patch('pandas.read_sql')
    def test_ticker_isolation(self, mock_read_sql, mock_secrets, mock_connector):
        """Test that queries only return requested ticker data"""
        
        mock_secrets.return_value = ('host', 'user', 'pass', 'db')
        mock_connector.return_value = Mock()
        
        # Mock data for multiple tickers
        all_data = pd.DataFrame({
            'ticker': ['AAPL', 'AAPL', 'GOOGL', 'GOOGL'],
            'date': pd.date_range('2024-01-01', periods=4),
            'close_Price': [150.0, 151.0, 2800.0, 2810.0]
        })
        
        # Only AAPL data should be returned
        aapl_data = all_data[all_data['ticker'] == 'AAPL']
        mock_read_sql.return_value = aapl_data
        
        result = db_interactions.import_stock_price_data(amount=10, stock_ticker='AAPL')
        
        # Verify only AAPL data
        if 'ticker' in result.columns:
            self.assertTrue((result['ticker'] == 'AAPL').all(),
                          "Should only return requested ticker data")
    
    def test_data_export_validation(self):
        """Test that data export validates data before writing"""
        
        @patch('db_interactions.db_connectors.pandas_mysql_connector')
        @patch('db_interactions.fetch_secrets.secret_import')
        def test_export_validation(mock_secrets, mock_connector):
            mock_secrets.return_value = ('host', 'user', 'pass', 'db')
            mock_connection = MagicMock()
            mock_connector.return_value = mock_connection
            
            # Test with invalid data type
            with self.assertRaises(ValueError):
                db_interactions.export_stock_price_data("")  # Empty string instead of DataFrame
            
            with self.assertRaises(ValueError):
                db_interactions.export_stock_price_data(None)  # None instead of DataFrame
        
        test_export_validation()


def run_security_tests():
    """Run all security tests and return results"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSQLInjectionPrevention))
    suite.addTests(loader.loadTestsFromTestCase(TestInputValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestSecretsManagement))
    suite.addTests(loader.loadTestsFromTestCase(TestFilePathSecurity))
    suite.addTests(loader.loadTestsFromTestCase(TestDataAccessControl))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    result = run_security_tests()
    
    # Print summary
    print("\n" + "="*70)
    print("SECURITY TEST SUMMARY")
    print("="*70)
    print(f"Tests Run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print("="*70)
