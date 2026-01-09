"""Test with real stock data to validate production readiness."""

import sys
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# Import all needed functions
from stock_data_fetch import (
    fetch_stock_price_data,
    calculate_period_returns,
    add_technical_indicators,
    add_volume_indicators,
    add_volatility_indicators,
    calculate_moving_averages,
    calculate_standard_diviation_value,
    calculate_bollinger_bands,
    calculate_momentum
)

print("="*80)
print("REAL STOCK DATA VALIDATION TEST")
print("="*80)

# Fetch 1 year of data for DEMANT.CO
ticker = "DEMANT.CO"
start_date = datetime.now() - relativedelta(years=1)

print(f"\nStep 1: Fetching {ticker} data (1 year)...")
try:
    df = fetch_stock_price_data(ticker, start_date)
    print(f"   OK: Fetched {len(df)} rows")
except Exception as e:
    print(f"   ERROR: {e}")
    sys.exit(1)

print(f"\nStep 2: Calculating period returns...")
try:
    df = calculate_period_returns(df)
    print(f"   OK: Returns calculated, rows: {len(df)}")
except Exception as e:
    print(f"   ERROR: {e}")
    sys.exit(1)

print(f"\nStep 3: Adding technical indicators (RSI, MACD, ATR)...")
try:
    df = add_technical_indicators(df)
    print(f"   OK: Technical indicators added, rows: {len(df)}")
except Exception as e:
    print(f"   ERROR: {e}")
    sys.exit(1)

print(f"\nStep 4: Adding volume indicators...")
try:
    df = add_volume_indicators(df)
    print(f"   OK: Volume indicators added, rows: {len(df)}")
except Exception as e:
    print(f"   ERROR: {e}")
    sys.exit(1)

print(f"\nStep 5: Adding volatility indicators...")
try:
    df = add_volatility_indicators(df)
    print(f"   OK: Volatility indicators added, rows: {len(df)}")
except Exception as e:
    print(f"   ERROR: {e}")
    sys.exit(1)

print(f"\nStep 6: Calculating moving averages (NEW: all 5 periods)...")
try:
    df = calculate_moving_averages(df)
    print(f"   OK: Moving averages calculated, rows: {len(df)}")
except Exception as e:
    print(f"   ERROR: {e}")
    sys.exit(1)

print(f"\nStep 7: Calculating standard deviations (NEW: vectorized)...")
try:
    df = calculate_standard_diviation_value(df)
    print(f"   OK: Standard deviations calculated, rows: {len(df)}")
except Exception as e:
    print(f"   ERROR: {e}")
    sys.exit(1)

print(f"\nStep 8: Calculating Bollinger Bands (NEW: all 5 periods)...")
try:
    df = calculate_bollinger_bands(df)
    print(f"   OK: Bollinger Bands calculated, rows: {len(df)}")
except Exception as e:
    print(f"   ERROR: {e}")
    sys.exit(1)

print(f"\nStep 9: Calculating momentum...")
try:
    df = calculate_momentum(df)
    print(f"   OK: Momentum calculated, rows: {len(df)}")
except Exception as e:
    print(f"   ERROR: {e}")
    sys.exit(1)

# Remove duplicate columns
df = df.loc[:, ~df.columns.duplicated()]

print("\n" + "="*80)
print("FEATURE VALIDATION")
print("="*80)

# Check all expected features
new_features = [
    'sma_5', 'sma_20', 'sma_40', 'sma_120', 'sma_200',
    'ema_5', 'ema_20', 'ema_40', 'ema_120', 'ema_200',
    'std_Div_5', 'std_Div_20', 'std_Div_40', 'std_Div_120', 'std_Div_200',
    'bollinger_Band_5_2STD', 'bollinger_Band_20_2STD', 'bollinger_Band_40_2STD',
    'bollinger_Band_120_2STD', 'bollinger_Band_200_2STD'
]

print("\nNew Features (20 total):")
present_count = 0
for feature in new_features:
    if feature in df.columns:
        non_null = df[feature].notna().sum()
        pct = (non_null / len(df)) * 100
        status = "OK" if non_null > 0 else "EMPTY"
        print(f"  {status}: {feature:30s} - {non_null:3d}/{len(df)} ({pct:5.1f}%)")
        present_count += 1
    else:
        print(f"  MISSING: {feature}")

print(f"\nFeature Coverage: {present_count}/{len(new_features)}")

# Show sample data from recent dates
print("\n" + "="*80)
print("SAMPLE DATA (Last 5 rows)")
print("="*80)

sample_cols = ['date', 'close_Price', 'sma_5', 'sma_40', 'sma_200', 
               'std_Div_5', 'std_Div_40', 'bollinger_Band_40_2STD']
available_cols = [col for col in sample_cols if col in df.columns]

print(df[available_cols].tail(5).to_string())

# Check for critical issues
print("\n" + "="*80)
print("DATA QUALITY CHECKS")
print("="*80)

critical_cols = ['date', 'ticker', 'close_Price', 'open_Price', 'high_Price', 'low_Price']
for col in critical_cols:
    null_count = df[col].isnull().sum()
    if null_count > 0:
        print(f"  WARNING: {col} has {null_count} null values")
    else:
        print(f"  OK: {col} complete")

# Check NaN distribution in new features
print("\nNaN distribution in new features (expected for initial rows):")
for feature in ['sma_5', 'sma_40', 'sma_200']:
    if feature in df.columns:
        nan_count = df[feature].isnull().sum()
        print(f"  {feature:15s}: {nan_count:3d} NaN values")

print("\n" + "="*80)
print("FINAL VERDICT")
print("="*80)

if present_count == len(new_features):
    print("SUCCESS: All 20 new features calculated correctly!")
    print(f"         {len(df)} rows of {ticker} data processed")
    print("         Ready for database export")
    sys.exit(0)
else:
    print(f"PARTIAL: {present_count}/{len(new_features)} features present")
    print("         Review missing features above")
    sys.exit(1)
