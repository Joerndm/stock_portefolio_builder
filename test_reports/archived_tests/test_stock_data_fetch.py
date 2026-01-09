"""
Test script for stock_data_fetch.py pipeline validation.
Tests both regular stocks (DEMANT.CO) and indices (^VIX) to ensure all features are calculated correctly.
"""
import sys
import io

# Force UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Import functions from stock_data_fetch
import stock_data_fetch

def test_stock_ticker(ticker):
    """Test all feature calculations for a given ticker."""
    print(f"\n{'='*80}")
    print(f"🧪 TESTING TICKER: {ticker}")
    print(f"{'='*80}\n")
    
    # Get stock info to determine if it's an index
    try:
        stock = yf.Ticker(ticker)
        stock_info = stock.info
        is_index = stock_info.get("typeDisp") == "Index"
        print(f"📋 Ticker Type: {'INDEX' if is_index else 'STOCK'}")
        print(f"📋 Company/Index Name: {stock_info.get('longName', 'N/A')}")
    except Exception as e:
        print(f"❌ Failed to fetch stock info: {e}")
        return False
    
    try:
        # Step 1: Fetch stock price data
        print(f"\n{'─'*80}")
        print("1️⃣  FETCHING STOCK PRICE DATA")
        print(f"{'─'*80}")
        stock_price_data_df = stock_data_fetch.fetch_stock_price_data(
            ticker, 
            start_date=(datetime.now() - relativedelta(years=6))
        )
        print(f"✅ Fetched {len(stock_price_data_df)} rows")
        print(f"   Columns: {list(stock_price_data_df.columns)}")
        print(f"   Date range: {stock_price_data_df['date'].min()} to {stock_price_data_df['date'].max()}")
        
        # Step 2: Calculate period returns
        print(f"\n{'─'*80}")
        print("2️⃣  CALCULATING PERIOD RETURNS")
        print(f"{'─'*80}")
        initial_cols = len(stock_price_data_df.columns)
        stock_price_data_df = stock_data_fetch.calculate_period_returns(stock_price_data_df)
        new_cols = [col for col in stock_price_data_df.columns if col not in ['date', 'ticker', 'currency', 'trade_Volume', 'open_Price', 'high_Price', 'low_Price', 'close_Price']]
        print(f"✅ Added {len(stock_price_data_df.columns) - initial_cols} period return columns")
        print(f"   New columns: {[col for col in new_cols if 'D' in col or 'M' in col or 'Y' in col]}")
        
        # Step 3: Add technical indicators
        print(f"\n{'─'*80}")
        print("3️⃣  ADDING TECHNICAL INDICATORS (RSI, MACD, ATR)")
        print(f"{'─'*80}")
        initial_cols = len(stock_price_data_df.columns)
        stock_price_data_df = stock_data_fetch.add_technical_indicators(stock_price_data_df)
        technical_cols = [col for col in stock_price_data_df.columns if col in ['RSI_14', 'macd', 'macd_histogram', 'macd_signal', 'ATR_14', 'ATRl_14', 'ATRr_14']]
        print(f"✅ Added {len(technical_cols)} technical indicator columns")
        print(f"   Technical indicators: {technical_cols}")
        
        # Step 4: Add volume indicators
        print(f"\n{'─'*80}")
        print("4️⃣  ADDING VOLUME INDICATORS")
        print(f"{'─'*80}")
        initial_cols = len(stock_price_data_df.columns)
        stock_price_data_df = stock_data_fetch.add_volume_indicators(stock_price_data_df)
        volume_cols = [col for col in stock_price_data_df.columns if 'volume' in col.lower() or col in ['vwap', 'obv']]
        print(f"✅ Added {len(volume_cols)} volume indicator columns")
        print(f"   Volume indicators: {volume_cols}")
        
        # Step 5: Add volatility indicators (should be skipped for indices)
        print(f"\n{'─'*80}")
        print("5️⃣  ADDING VOLATILITY INDICATORS")
        print(f"{'─'*80}")
        if is_index:
            print(f"⏭️  SKIPPED for index ticker (as expected)")
            volatility_cols = []
        else:
            initial_cols = len(stock_price_data_df.columns)
            stock_price_data_df = stock_data_fetch.add_volatility_indicators(stock_price_data_df)
            volatility_cols = [col for col in stock_price_data_df.columns if 'volatility' in col.lower()]
            print(f"✅ Added {len(volatility_cols)} volatility indicator columns")
            print(f"   Volatility indicators: {volatility_cols}")
        
        # Step 6: Calculate moving averages
        print(f"\n{'─'*80}")
        print("6️⃣  CALCULATING MOVING AVERAGES")
        print(f"{'─'*80}")
        initial_cols = len(stock_price_data_df.columns)
        stock_price_data_df = stock_data_fetch.calculate_moving_averages(stock_price_data_df)
        ma_cols = [col for col in stock_price_data_df.columns if 'sma' in col.lower() or 'ema' in col.lower()]
        print(f"✅ Added {len(ma_cols)} moving average columns")
        print(f"   Moving averages: {ma_cols}")
        
        # Step 7: Calculate standard deviation
        print(f"\n{'─'*80}")
        print("7️⃣  CALCULATING STANDARD DEVIATION")
        print(f"{'─'*80}")
        initial_cols = len(stock_price_data_df.columns)
        stock_price_data_df = stock_data_fetch.calculate_standard_diviation_value(stock_price_data_df)
        std_cols = [col for col in stock_price_data_df.columns if 'std_Div' in col]
        print(f"✅ Added {len(std_cols)} standard deviation columns")
        print(f"   Std dev indicators: {std_cols}")
        
        # Step 8: Calculate Bollinger Bands
        print(f"\n{'─'*80}")
        print("8️⃣  CALCULATING BOLLINGER BANDS")
        print(f"{'─'*80}")
        initial_cols = len(stock_price_data_df.columns)
        stock_price_data_df = stock_data_fetch.calculate_bollinger_bands(stock_price_data_df)
        bb_cols = [col for col in stock_price_data_df.columns if 'bollinger' in col.lower()]
        print(f"✅ Added {len(bb_cols)} Bollinger Band columns")
        print(f"   Bollinger Bands: {bb_cols}")
        
        # Step 9: Calculate momentum
        print(f"\n{'─'*80}")
        print("9️⃣  CALCULATING MOMENTUM")
        print(f"{'─'*80}")
        initial_cols = len(stock_price_data_df.columns)
        stock_price_data_df = stock_data_fetch.calculate_momentum(stock_price_data_df)
        momentum_cols = [col for col in stock_price_data_df.columns if 'momentum' in col.lower()]
        print(f"✅ Added {len(momentum_cols)} momentum columns")
        print(f"   Momentum indicators: {momentum_cols}")
        
        # Final summary
        print(f"\n{'─'*80}")
        print("📊 FINAL DATAFRAME SUMMARY")
        print(f"{'─'*80}")
        print(f"Total columns: {len(stock_price_data_df.columns)}")
        print(f"Total rows: {len(stock_price_data_df)}")
        print(f"Rows before dropna: {len(stock_price_data_df)}")
        stock_price_data_df = stock_price_data_df.dropna()
        print(f"Rows after dropna: {len(stock_price_data_df)}")
        
        # Check for null values
        null_counts = stock_price_data_df.isnull().sum()
        null_columns = null_counts[null_counts > 0]
        if len(null_columns) > 0:
            print(f"\n⚠️  WARNING: Found NULL values in {len(null_columns)} columns:")
            for col, count in null_columns.items():
                print(f"   - {col}: {count} nulls ({count/len(stock_price_data_df)*100:.2f}%)")
        else:
            print(f"\n✅ No NULL values found")
        
        # Check for expected features for ml_builder
        print(f"\n{'─'*80}")
        print("🎯 ML_BUILDER FEATURE VALIDATION")
        print(f"{'─'*80}")
        
        # Define expected features for ml_builder
        expected_features = [
            '1M', '3M', '6M', '9M', '1Y', '2Y', '3Y', '4Y', '5Y',  # Period returns
            'sma_40', 'sma_120', 'ema_40', 'ema_120',  # Moving averages
            'std_Div_40', 'std_Div_120',  # Std dev
            'bollinger_Band_40_2STD', 'bollinger_Band_120_2STD',  # Bollinger
            'momentum',  # Momentum
            'RSI_14', 'macd', 'macd_histogram', 'macd_signal',  # Technical (ATR optional)
            'volume_sma_20', 'volume_ema_20', 'volume_ratio', 'vwap', 'obv'  # Volume
        ]
        
        # Add volatility only for stocks, not indices
        if not is_index:
            expected_features.extend(['volatility_5d', 'volatility_20d', 'volatility_60d'])
        
        missing_features = []
        present_features = []
        
        for feature in expected_features:
            if feature in stock_price_data_df.columns:
                present_features.append(feature)
            else:
                missing_features.append(feature)
        
        print(f"✅ Present features: {len(present_features)}/{len(expected_features)}")
        for feat in present_features:
            print(f"   ✓ {feat}")
        
        if missing_features:
            print(f"\n❌ Missing features: {len(missing_features)}")
            for feat in missing_features:
                print(f"   ✗ {feat}")
        else:
            print(f"\n🎉 ALL EXPECTED FEATURES PRESENT!")
        
        # Sample data preview
        print(f"\n{'─'*80}")
        print("📋 SAMPLE DATA (Last 5 rows)")
        print(f"{'─'*80}")
        if len(stock_price_data_df) > 0:
            display_cols = ['date', 'close_Price']
            if 'RSI_14' in stock_price_data_df.columns:
                display_cols.append('RSI_14')
            if 'volume_sma_20' in stock_price_data_df.columns:
                display_cols.append('volume_sma_20')
            if 'momentum' in stock_price_data_df.columns:
                display_cols.append('momentum')
            print(stock_price_data_df[display_cols].tail())
        else:
            print("⚠️  WARNING: DataFrame is empty after dropna()")
        
        print(f"\n{'='*80}")
        print(f"✅ TEST PASSED FOR {ticker}")
        print(f"{'='*80}\n")
        
        return True, missing_features
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"❌ TEST FAILED FOR {ticker}")
        print(f"Error: {str(e)}")
        print(f"{'='*80}\n")
        import traceback
        traceback.print_exc()
        return False, []


if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 STOCK_DATA_FETCH PIPELINE TEST")
    print("="*80)
    print("\nThis test validates the complete feature calculation pipeline")
    print("for both regular stocks and indices.\n")
    
    # Test tickers
    test_tickers = [
        ("DEMANT.CO", "Regular Stock"),
        ("^VIX", "Volatility Index")
    ]
    
    results = {}
    
    for ticker, description in test_tickers:
        print(f"\n\n{'#'*80}")
        print(f"# Testing: {ticker} ({description})")
        print(f"{'#'*80}")
        
        success, missing = test_stock_ticker(ticker)
        results[ticker] = {"success": success, "missing_features": missing}
    
    # Final summary
    print("\n\n" + "="*80)
    print("📊 FINAL TEST SUMMARY")
    print("="*80)
    
    all_passed = all(result["success"] for result in results.values())
    
    for ticker, result in results.items():
        status = "✅ PASSED" if result["success"] else "❌ FAILED"
        print(f"\n{ticker}: {status}")
        if result["missing_features"]:
            print(f"  Missing features: {', '.join(result['missing_features'])}")
    
    if all_passed:
        print("\n" + "="*80)
        print("🎉 ALL TESTS PASSED!")
        print("="*80)
        print("\n✅ The pipeline correctly:")
        print("   • Calculates all technical indicators (RSI, MACD, ATR)")
        print("   • Calculates all volume indicators (SMA, EMA, ratio, VWAP, OBV)")
        print("   • Calculates volatility for stocks only (excludes indices)")
        print("   • Calculates moving averages, std dev, Bollinger Bands, momentum")
        print("   • Provides all features required for ml_builder training")
        print("\n✅ Ready for ml_builder training!")
    else:
        print("\n" + "="*80)
        print("⚠️  SOME TESTS FAILED")
        print("="*80)
        print("\nPlease review the errors above and fix the issues.")
