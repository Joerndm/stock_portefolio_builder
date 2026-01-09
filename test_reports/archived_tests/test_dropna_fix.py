"""
Quick test to verify dropna() fix - fetches data for one ticker
"""

import sys
import io

# Add UTF-8 encoding wrapper for Windows console
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import stock_data_fetch
import yfinance as yf

print("="*80)
print("🧪 TESTING DROPNA() FIX")
print("="*80)
print()

# Test with DEMANT.CO (regular stock with volatility)
test_ticker = "DEMANT.CO"
print(f"Testing with ticker: {test_ticker}")
print()

try:
    stock_info = yf.Ticker(test_ticker).info
    print(f"Stock type: {stock_info.get('typeDisp', 'Unknown')}")
    print()
    
    print("─"*80)
    print("STEP 1: Fetching stock price data")
    print("─"*80)
    stock_price_data_df = stock_data_fetch.fetch_stock_price_data(test_ticker)
    print(f"✅ Initial rows: {len(stock_price_data_df)}")
    print()
    
    print("─"*80)
    print("STEP 2: Calculating period returns")
    print("─"*80)
    stock_price_data_df = stock_data_fetch.calculate_period_returns(stock_price_data_df)
    print(f"✅ Rows: {len(stock_price_data_df)}")
    print()
    
    print("─"*80)
    print("STEP 3: Adding technical indicators")
    print("─"*80)
    stock_price_data_df = stock_data_fetch.add_technical_indicators(stock_price_data_df)
    print(f"✅ Rows: {len(stock_price_data_df)}")
    print()
    
    print("─"*80)
    print("STEP 4: Adding volume indicators")
    print("─"*80)
    stock_price_data_df = stock_data_fetch.add_volume_indicators(stock_price_data_df)
    print(f"✅ Rows: {len(stock_price_data_df)}")
    print()
    
    print("─"*80)
    print("STEP 5: Adding volatility indicators")
    print("─"*80)
    if stock_info.get("typeDisp", "") != "Index":
        stock_price_data_df = stock_data_fetch.add_volatility_indicators(stock_price_data_df)
        print(f"✅ Rows: {len(stock_price_data_df)}")
    else:
        print("⏭️  Skipped (index ticker)")
    print()
    
    print("─"*80)
    print("STEP 6: Calculating moving averages")
    print("─"*80)
    stock_price_data_df = stock_data_fetch.calculate_moving_averages(stock_price_data_df)
    print(f"✅ Rows: {len(stock_price_data_df)}")
    print()
    
    print("─"*80)
    print("STEP 7: Calculating standard deviation")
    print("─"*80)
    stock_price_data_df = stock_data_fetch.calculate_standard_diviation_value(stock_price_data_df)
    print(f"✅ Rows: {len(stock_price_data_df)}")
    print()
    
    print("─"*80)
    print("STEP 8: Calculating Bollinger Bands")
    print("─"*80)
    stock_price_data_df = stock_data_fetch.calculate_bollinger_bands(stock_price_data_df)
    print(f"✅ Rows: {len(stock_price_data_df)}")
    print()
    
    print("─"*80)
    print("STEP 9: Calculating momentum")
    print("─"*80)
    stock_price_data_df = stock_data_fetch.calculate_momentum(stock_price_data_df)
    print(f"✅ Rows: {len(stock_price_data_df)}")
    print()
    
    print("─"*80)
    print("STEP 10: dropna() with subset (CRITICAL TEST)")
    print("─"*80)
    print(f"Before dropna(): {len(stock_price_data_df)} rows")
    
    # Check for NaN values
    nan_counts = stock_price_data_df.isnull().sum()
    columns_with_nans = nan_counts[nan_counts > 0]
    if len(columns_with_nans) > 0:
        print(f"Columns with NaN values: {len(columns_with_nans)}")
        print(f"Top 10 columns by NaN count:")
        for col, count in columns_with_nans.nlargest(10).items():
            print(f"   {col}: {count} NaNs ({count/len(stock_price_data_df)*100:.1f}%)")
    else:
        print("No NaN values found!")
    print()
    
    # Apply the fixed dropna() logic
    critical_cols = ['date', 'ticker', 'close_Price', 'open_Price', 'high_Price', 'low_Price']
    stock_price_data_df = stock_price_data_df.dropna(subset=critical_cols)
    
    print(f"After dropna(subset=critical_cols): {len(stock_price_data_df)} rows")
    print()
    
    if stock_price_data_df.empty:
        print("❌ FAILED: DataFrame is EMPTY after dropna()")
        print("   This would cause: ValueError: The stock_price_data_df parameter cannot be empty.")
    else:
        print("✅ SUCCESS: DataFrame has data after dropna()!")
        print(f"   Total columns: {len(stock_price_data_df.columns)}")
        print(f"   Data is ready for database export")
        
        # Show how many calculated features still have NaN (this is OK)
        nan_after = stock_price_data_df.isnull().sum().sum()
        if nan_after > 0:
            print(f"   Note: {nan_after} NaN values remain in calculated features (this is expected)")
        print()
        
        print("📋 Sample data (last 3 rows):")
        print(stock_price_data_df[['date', 'ticker', 'close_Price', 'RSI_14', 'volume_ratio', 'volatility_20d']].tail(3))
    
    print()
    print("="*80)
    print("✅ DROPNA() FIX TEST COMPLETED")
    print("="*80)
    
except Exception as e:
    print(f"❌ ERROR: {str(e)}")
    import traceback
    traceback.print_exc()
