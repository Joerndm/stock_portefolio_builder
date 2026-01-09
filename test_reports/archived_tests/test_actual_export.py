"""
Final test - actually export to database (with small dataset)
"""

import sys
import io

# Add UTF-8 encoding wrapper for Windows console
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import stock_data_fetch
import db_interactions
import yfinance as yf

test_ticker = "DEMANT.CO"
print("="*80)
print("FINAL DATABASE EXPORT TEST")
print("="*80)
print(f"\nTesting with ticker: {test_ticker}\n")

try:
    # Check if data already exists
    print("Step 1: Checking if ticker already exists in database...")
    exists = db_interactions.does_stock_exists_stock_price_data(test_ticker)
    print(f"   Ticker '{test_ticker}' exists in database: {exists}")
    
    if exists:
        print("\n   NOTE: Data already exists. This test will append new data.")
        print("   If you want a clean test, manually DELETE FROM stock_price_data WHERE ticker='DEMANT.CO' first.")
    
    # Fetch and process data (need ~1 year for windows)
    print("\nStep 2: Fetching data (1 year for window calculations)...")
    
    import datetime
    from dateutil.relativedelta import relativedelta
    
    start_date = datetime.datetime.now() - relativedelta(years=1, days=60)  # 1+ year for safety
    
    stock_info = yf.Ticker(test_ticker).info
    print(f"   Stock type: {stock_info.get('typeDisp', 'Unknown')}")
    
    # Fetch limited data
    stock_price_data_df = stock_data_fetch.fetch_stock_price_data(test_ticker, start_date)
    print(f"   Fetched {len(stock_price_data_df)} rows from {start_date.strftime('%Y-%m-%d')}")
    
    # Process through pipeline
    print("\nStep 3: Processing through pipeline...")
    stock_price_data_df = stock_data_fetch.calculate_period_returns(stock_price_data_df)
    stock_price_data_df = stock_data_fetch.add_technical_indicators(stock_price_data_df)
    stock_price_data_df = stock_data_fetch.add_volume_indicators(stock_price_data_df)
    if stock_info.get("typeDisp", "") != "Index":
        stock_price_data_df = stock_data_fetch.add_volatility_indicators(stock_price_data_df)
    stock_price_data_df = stock_data_fetch.calculate_moving_averages(stock_price_data_df)
    stock_price_data_df = stock_data_fetch.calculate_standard_diviation_value(stock_price_data_df)
    stock_price_data_df = stock_data_fetch.calculate_bollinger_bands(stock_price_data_df)
    stock_price_data_df = stock_data_fetch.calculate_momentum(stock_price_data_df)
    
    # Apply dropna fix
    critical_cols = ['date', 'ticker', 'close_Price', 'open_Price', 'high_Price', 'low_Price']
    stock_price_data_df = stock_price_data_df.dropna(subset=critical_cols)
    
    print(f"   After processing: {len(stock_price_data_df)} rows remaining")
    print(f"   DataFrame columns: {len(stock_price_data_df.columns)}")
    
    if stock_price_data_df.empty:
        print("\n   ERROR: DataFrame is empty after processing!")
    else:
        print("\nStep 4: Exporting to database...")
        db_interactions.export_stock_price_data(stock_price_data_df)
        print("   SUCCESS: Data exported to database!")
        
        # Verify export
        print("\nStep 5: Verifying export...")
        imported_df = db_interactions.import_stock_price_data(amount=5, stock_ticker=test_ticker)
        print(f"   Imported {len(imported_df)} rows from database")
        print(f"   Database columns: {len(imported_df.columns)}")
        print(f"\n   Latest dates in database:")
        print(imported_df[['date', 'ticker', 'close_Price', 'rsi_14', 'volume_ratio', 'volatility_20d']].head())
        
        print("\n" + "="*80)
        print("SUCCESS: EXPORT TEST COMPLETED")
        print("="*80)
        print("\nThe export function is working correctly!")
        print("All new columns (technical, volume, volatility indicators) are being exported.")
    
except Exception as e:
    print(f"\nERROR: {str(e)}")
    import traceback
    traceback.print_exc()
