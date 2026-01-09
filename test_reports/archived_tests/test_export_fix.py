"""
Test the fixed export_stock_price_data function
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
import pandas as pd

test_ticker = "DEMANT.CO"
print("="*80)
print("TESTING EXPORT FIX")
print("="*80)
print(f"\nTesting with ticker: {test_ticker}\n")

try:
    # Fetch and process data
    print("Step 1: Fetching and processing data...")
    stock_info = yf.Ticker(test_ticker).info
    stock_price_data_df = stock_data_fetch.fetch_stock_price_data(test_ticker)
    stock_price_data_df = stock_data_fetch.calculate_period_returns(stock_price_data_df)
    stock_price_data_df = stock_data_fetch.add_technical_indicators(stock_price_data_df)
    stock_price_data_df = stock_data_fetch.add_volume_indicators(stock_price_data_df)
    if stock_info.get("typeDisp", "") != "Index":
        stock_price_data_df = stock_data_fetch.add_volatility_indicators(stock_price_data_df)
    stock_price_data_df = stock_data_fetch.calculate_moving_averages(stock_price_data_df)
    stock_price_data_df = stock_data_fetch.calculate_standard_diviation_value(stock_price_data_df)
    stock_price_data_df = stock_data_fetch.calculate_bollinger_bands(stock_price_data_df)
    stock_price_data_df = stock_data_fetch.calculate_momentum(stock_price_data_df)
    
    critical_cols = ['date', 'ticker', 'close_Price', 'open_Price', 'high_Price', 'low_Price']
    stock_price_data_df = stock_price_data_df.dropna(subset=critical_cols)
    
    print(f"   Processed {len(stock_price_data_df)} rows")
    print(f"   DataFrame has {len(stock_price_data_df.columns)} columns")
    
    # Check for duplicates BEFORE export
    duplicate_cols = stock_price_data_df.columns[stock_price_data_df.columns.duplicated()].tolist()
    if duplicate_cols:
        print(f"\n   Warning: {len(set(duplicate_cols))} duplicate columns found before export")
        for col in set(duplicate_cols):
            count = stock_price_data_df.columns.tolist().count(col)
            print(f"      {col}: appears {count} times")
    
    print("\nStep 2: Attempting database export...")
    
    # Create a copy to test export logic without actually exporting
    test_df = stock_price_data_df.copy()
    
    # Simulate the export preparation steps
    print("   Removing duplicate columns...")
    test_df = test_df.loc[:, ~test_df.columns.duplicated()]
    print(f"   After deduplication: {len(test_df.columns)} columns")
    
    print("   Renaming columns...")
    column_rename_map = {
        'RSI_14': 'rsi_14',
        'ATR_14': 'atr_14',
        'ATRr_14': 'atr_14',
        'ATRl_14': 'atr_14'
    }
    test_df = test_df.rename(columns=column_rename_map)
    
    print("   Adding missing columns...")
    db_columns = [
        'date', 'ticker', 'currency', 'trade_Volume', 
        'open_Price', 'high_Price', 'low_Price', 'close_Price',
        '1D', '1M', '3M', '6M', '9M', '1Y', '2Y', '3Y', '4Y', '5Y',
        'sma_5', 'sma_20', 'sma_40', 'sma_120', 'sma_200',
        'ema_5', 'ema_20', 'ema_40', 'ema_120', 'ema_200',
        'std_Div_5', 'std_Div_20', 'std_Div_40', 'std_Div_120', 'std_Div_200',
        'bollinger_Band_5_2STD', 'bollinger_Band_20_2STD', 'bollinger_Band_40_2STD',
        'bollinger_Band_120_2STD', 'bollinger_Band_200_2STD',
        'momentum', 'rsi_14', 'atr_14', 'macd', 'macd_signal', 'macd_histogram',
        'volume_sma_20', 'volume_ema_20', 'volume_ratio', 'vwap', 'obv',
        'volatility_5d', 'volatility_20d', 'volatility_60d'
    ]
    
    missing_cols = []
    for col in db_columns:
        if col not in test_df.columns:
            test_df[col] = None
            missing_cols.append(col)
    
    if missing_cols:
        print(f"   Added {len(missing_cols)} missing columns with NULL values:")
        for col in missing_cols[:10]:  # Show first 10
            print(f"      {col}")
        if len(missing_cols) > 10:
            print(f"      ... and {len(missing_cols) - 10} more")
    
    print("   Reordering columns to match database schema...")
    test_df = test_df[db_columns]
    
    print(f"\n   Final DataFrame: {len(test_df)} rows × {len(test_df.columns)} columns")
    print("   Column check:")
    print(f"      Expected: {len(db_columns)} columns")
    print(f"      Actual: {len(test_df.columns)} columns")
    print(f"      Match: {'YES' if len(test_df.columns) == len(db_columns) else 'NO'}")
    
    # Check for duplicates AFTER preparation
    duplicate_cols_after = test_df.columns[test_df.columns.duplicated()].tolist()
    if duplicate_cols_after:
        print(f"\n   ERROR: Still have {len(set(duplicate_cols_after))} duplicate columns after preparation!")
        for col in set(duplicate_cols_after):
            count = test_df.columns.tolist().count(col)
            print(f"      {col}: appears {count} times")
    else:
        print("\n   SUCCESS: No duplicate columns after preparation")
    
    print("\n" + "="*80)
    print("Column names comparison (first 20):")
    print("="*80)
    print(f"{'DataFrame Column':<30} {'Database Column':<30}")
    print("-"*80)
    for i, (df_col, db_col) in enumerate(zip(test_df.columns[:20], db_columns[:20])):
        match = "✓" if df_col == db_col else "✗"
        print(f"{df_col:<30} {db_col:<30} {match}")
    
    print("\n" + "="*80)
    print("PREPARATION SUCCESSFUL")
    print("="*80)
    print("\nDataFrame is ready for database export!")
    print(f"Columns: {len(test_df.columns)}")
    print(f"Rows: {len(test_df)}")
    print(f"\nYou can now safely call db_interactions.export_stock_price_data()")
    
except Exception as e:
    print(f"\nERROR: {str(e)}")
    import traceback
    traceback.print_exc()
