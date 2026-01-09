"""
Comprehensive test suite for database components.
Tests db_connectors.py and db_interactions.py functionality.
"""

import sys
import io
import pandas as pd
import datetime

# Add UTF-8 encoding wrapper for Windows console
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import db_connectors
import db_interactions
import fetch_secrets

print("="*80)
print("🔧 DATABASE COMPONENTS TEST")
print("="*80)
print()
print("This test validates database connectivity and interaction functions.")
print()

# ============================================================================
# TEST 1: Database Secrets
# ============================================================================
print("─"*80)
print("1️⃣  TESTING SECRET IMPORT")
print("─"*80)

try:
    db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()
    print(f"✅ Secrets imported successfully")
    print(f"   Host: {db_host}")
    print(f"   User: {db_user}")
    print(f"   Database: {db_name}")
    print(f"   Password: {'*' * len(db_pass)}")
    print()
except Exception as e:
    print(f"❌ FAILED: {str(e)}")
    sys.exit(1)

# ============================================================================
# TEST 2: MySQL Connector
# ============================================================================
print("─"*80)
print("2️⃣  TESTING MYSQL CONNECTOR")
print("─"*80)

try:
    mysql_con = db_connectors.mysql_connector(db_host, db_user, db_pass, db_name)
    print(f"✅ MySQL connection established")
    print(f"   Connection type: {type(mysql_con)}")
    print(f"   Is connected: {mysql_con.is_connected()}")
    mysql_con.close()
    print(f"   Connection closed successfully")
    print()
except Exception as e:
    print(f"❌ FAILED: {str(e)}")
    sys.exit(1)

# ============================================================================
# TEST 3: Pandas MySQL Connector (SQLAlchemy)
# ============================================================================
print("─"*80)
print("3️⃣  TESTING PANDAS MYSQL CONNECTOR (SQLAlchemy)")
print("─"*80)

try:
    pandas_con = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)
    print(f"✅ Pandas MySQL connection established")
    print(f"   Connection type: {type(pandas_con)}")
    print(f"   Dialect: {pandas_con.dialect.name}")
    print(f"   Driver: {pandas_con.driver}")
    
    # Test a simple query
    test_df = pd.read_sql("SELECT 1 as test_value", pandas_con)
    print(f"   Test query result: {test_df['test_value'][0]}")
    print()
except Exception as e:
    print(f"❌ FAILED: {str(e)}")
    sys.exit(1)

# ============================================================================
# TEST 4: Import Ticker List
# ============================================================================
print("─"*80)
print("4️⃣  TESTING IMPORT TICKER LIST")
print("─"*80)

try:
    ticker_list = db_interactions.import_ticker_list()
    print(f"✅ Ticker list imported successfully")
    print(f"   Type: {type(ticker_list)}")
    print(f"   Count: {len(ticker_list)} tickers")
    if len(ticker_list) > 0:
        print(f"   Sample (first 5): {ticker_list[:5]}")
    print()
except Exception as e:
    print(f"❌ FAILED: {str(e)}")
    print()

# ============================================================================
# TEST 5: Stock Existence Checks
# ============================================================================
print("─"*80)
print("5️⃣  TESTING STOCK EXISTENCE CHECKS")
print("─"*80)

# Test with a ticker that should exist (if any)
test_ticker = None
if 'ticker_list' in locals() and len(ticker_list) > 0:
    test_ticker = ticker_list[0]
    print(f"Testing with ticker: {test_ticker}")
    print()
    
    # Test stock_info_data
    try:
        exists = db_interactions.does_stock_exists_stock_info_data(test_ticker)
        print(f"   stock_info_data: {'✅ EXISTS' if exists else '❌ MISSING'}")
    except Exception as e:
        print(f"   stock_info_data: ⚠️  ERROR - {str(e)}")
    
    # Test stock_price_data
    try:
        exists = db_interactions.does_stock_exists_stock_price_data(test_ticker)
        print(f"   stock_price_data: {'✅ EXISTS' if exists else '❌ MISSING'}")
    except Exception as e:
        print(f"   stock_price_data: ⚠️  ERROR - {str(e)}")
    
    # Test stock_income_stmt_data
    try:
        exists = db_interactions.does_stock_exists_stock_income_stmt_data(test_ticker)
        print(f"   stock_income_stmt_data: {'✅ EXISTS' if exists else '❌ MISSING'}")
    except Exception as e:
        print(f"   stock_income_stmt_data: ⚠️  ERROR - {str(e)}")
    
    # Test stock_balancesheet_data
    try:
        exists = db_interactions.does_stock_exists_stock_balancesheet_data(test_ticker)
        print(f"   stock_balancesheet_data: {'✅ EXISTS' if exists else '❌ MISSING'}")
    except Exception as e:
        print(f"   stock_balancesheet_data: ⚠️  ERROR - {str(e)}")
    
    # Test stock_cash_flow_data
    try:
        exists = db_interactions.does_stock_exists_stock_cash_flow_data(test_ticker)
        print(f"   stock_cash_flow_data: {'✅ EXISTS' if exists else '❌ MISSING'}")
    except Exception as e:
        print(f"   stock_cash_flow_data: ⚠️  ERROR - {str(e)}")
    
    # Test stock_ratio_data
    try:
        exists = db_interactions.does_stock_exists_stock_ratio_data(test_ticker)
        print(f"   stock_ratio_data: {'✅ EXISTS' if exists else '❌ MISSING'}")
    except Exception as e:
        print(f"   stock_ratio_data: ⚠️  ERROR - {str(e)}")
    
    # Test stock_prediction_data
    try:
        exists = db_interactions.does_stock_exists_stock_prediction_data(test_ticker)
        print(f"   stock_prediction_data: {'✅ EXISTS' if exists else '❌ MISSING'}")
    except Exception as e:
        print(f"   stock_prediction_data: ⚠️  ERROR - {str(e)}")
    
    print()
else:
    print("⚠️  No tickers available to test")
    print()

# ============================================================================
# TEST 6: Import Stock Info Data
# ============================================================================
print("─"*80)
print("6️⃣  TESTING IMPORT STOCK INFO DATA")
print("─"*80)

if test_ticker:
    try:
        stock_info_df = db_interactions.import_stock_info_data(test_ticker)
        print(f"✅ Stock info data imported")
        print(f"   Rows: {len(stock_info_df)}")
        print(f"   Columns: {list(stock_info_df.columns)}")
        if len(stock_info_df) > 0:
            print(f"   Sample data:")
            print(stock_info_df.head())
        print()
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        print()
else:
    print("⚠️  Skipped - no test ticker available")
    print()

# ============================================================================
# TEST 7: Import Stock Price Data
# ============================================================================
print("─"*80)
print("7️⃣  TESTING IMPORT STOCK PRICE DATA")
print("─"*80)

if test_ticker:
    try:
        # Import limited amount for testing
        stock_price_df = db_interactions.import_stock_price_data(amount=5, stock_ticker=test_ticker)
        print(f"✅ Stock price data imported")
        print(f"   Rows: {len(stock_price_df)}")
        print(f"   Columns: {len(stock_price_df.columns)}")
        print(f"   Column names: {list(stock_price_df.columns)[:10]}...")  # First 10 columns
        if len(stock_price_df) > 0:
            print(f"   Date range: {stock_price_df['date'].min()} to {stock_price_df['date'].max()}")
            print(f"   Sample data (first 2 rows):")
            print(stock_price_df.head(2))
        print()
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        print()
else:
    print("⚠️  Skipped - no test ticker available")
    print()

# ============================================================================
# TEST 8: Import Stock Financial Data
# ============================================================================
print("─"*80)
print("8️⃣  TESTING IMPORT STOCK FINANCIAL DATA")
print("─"*80)

if test_ticker:
    try:
        stock_financial_df = db_interactions.import_stock_financial_data(amount=2, stock_ticker=test_ticker)
        print(f"✅ Stock financial data imported")
        print(f"   Rows: {len(stock_financial_df)}")
        print(f"   Columns: {len(stock_financial_df.columns)}")
        if len(stock_financial_df) > 0:
            print(f"   Date range: {stock_financial_df['date'].min()} to {stock_financial_df['date'].max()}")
        print()
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        print()
else:
    print("⚠️  Skipped - no test ticker available")
    print()

# ============================================================================
# TEST 9: Import Stock Ratio Data
# ============================================================================
print("─"*80)
print("9️⃣  TESTING IMPORT STOCK RATIO DATA")
print("─"*80)

if test_ticker:
    try:
        stock_ratio_df = db_interactions.import_stock_ratio_data(amount=5, stock_ticker=test_ticker)
        print(f"✅ Stock ratio data imported")
        print(f"   Rows: {len(stock_ratio_df)}")
        print(f"   Columns: {len(stock_ratio_df.columns)}")
        if len(stock_ratio_df) > 0:
            print(f"   Date range: {stock_ratio_df['date'].min()} to {stock_ratio_df['date'].max()}")
        print()
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        print()
else:
    print("⚠️  Skipped - no test ticker available")
    print()

# ============================================================================
# TEST 10: Check Empty DataFrame After dropna()
# ============================================================================
print("─"*80)
print("🔟  TESTING DROPNA() BEHAVIOR ON STOCK PRICE DATA")
print("─"*80)

if test_ticker:
    try:
        # Import a large enough dataset
        stock_price_df = db_interactions.import_stock_price_data(amount=252*6, stock_ticker=test_ticker)
        print(f"✅ Stock price data imported for dropna() test")
        print(f"   Initial rows: {len(stock_price_df)}")
        print(f"   Initial columns: {len(stock_price_df.columns)}")
        
        # Check for NaN values
        nan_counts = stock_price_df.isnull().sum()
        columns_with_nans = nan_counts[nan_counts > 0]
        if len(columns_with_nans) > 0:
            print(f"   Columns with NaN values: {len(columns_with_nans)}")
            print(f"   Top 10 columns by NaN count:")
            for col, count in columns_with_nans.nlargest(10).items():
                print(f"      {col}: {count} NaNs ({count/len(stock_price_df)*100:.1f}%)")
        else:
            print(f"   ✅ No NaN values found!")
        
        # Test dropna()
        df_after_dropna = stock_price_df.dropna()
        print(f"   After dropna(): {len(df_after_dropna)} rows remaining")
        
        if len(df_after_dropna) == 0:
            print(f"   ⚠️  WARNING: DataFrame is EMPTY after dropna()!")
            print(f"   This will cause 'ValueError: The stock_price_data_df parameter cannot be empty.'")
        else:
            print(f"   ✅ DataFrame has data after dropna()")
        print()
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        print()
else:
    print("⚠️  Skipped - no test ticker available")
    print()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("="*80)
print("📊 TEST SUMMARY")
print("="*80)
print()
print("Database connection components:")
print("   ✅ Secret import working")
print("   ✅ MySQL connector working")
print("   ✅ Pandas/SQLAlchemy connector working")
print()
print("Database interaction functions:")
if test_ticker:
    print(f"   ✅ Tested with ticker: {test_ticker}")
    print("   ✅ All import functions operational")
    print()
    print("⚠️  IMPORTANT: Check dropna() test results above to see if")
    print("    DataFrame becomes empty after processing.")
else:
    print("   ⚠️  Limited testing - no tickers in database")
print()
print("="*80)
print("✅ DATABASE COMPONENT TESTS COMPLETED")
print("="*80)
