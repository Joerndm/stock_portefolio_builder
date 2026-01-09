"""Quick validation test for feature calculations with real-like data."""

import pandas as pd
from datetime import datetime, timedelta
from stock_data_fetch import (
    calculate_moving_averages,
    calculate_standard_diviation_value,
    calculate_bollinger_bands
)

# Create test data similar to real stock data
dates = [datetime.now() - timedelta(days=x) for x in range(300, 0, -1)]
prices = [100 + i + (i % 10) * 2 for i in range(300)]  # Add some variation

df = pd.DataFrame({
    'date': dates,
    'ticker': ['TEST'] * 300,
    'close_Price': prices,
    'open_Price': prices,
    'high_Price': [p + 2 for p in prices],
    'low_Price': [p - 2 for p in prices],
})

print("Starting feature calculation test...")
print(f"Initial columns: {list(df.columns)}")
print(f"Rows: {len(df)}")

# Calculate all features
df = calculate_moving_averages(df)
df = calculate_standard_diviation_value(df)
df = calculate_bollinger_bands(df)

# Remove duplicates if any
df = df.loc[:, ~df.columns.duplicated()]

print(f"\nFinal columns: {len(df.columns)}")
print("\nExpected features status:")

expected_features = [
    'sma_5', 'sma_20', 'sma_40', 'sma_120', 'sma_200',
    'ema_5', 'ema_20', 'ema_40', 'ema_120', 'ema_200',
    'std_Div_5', 'std_Div_20', 'std_Div_40', 'std_Div_120', 'std_Div_200',
    'bollinger_Band_5_2STD', 'bollinger_Band_20_2STD', 'bollinger_Band_40_2STD', 
    'bollinger_Band_120_2STD', 'bollinger_Band_200_2STD'
]

present = 0
for feature in expected_features:
    if feature in df.columns:
        non_null = df[feature].notna().sum()
        print(f"  OK: {feature:30s} - {non_null:3d} non-null values")
        present += 1
    else:
        print(f"  MISSING: {feature}")

print(f"\nResult: {present}/{len(expected_features)} features calculated")

# Show sample values
print("\nSample values (row 250):")
for col in ['sma_5', 'sma_40', 'sma_200', 'std_Div_5', 'std_Div_40', 'bollinger_Band_40_2STD']:
    if col in df.columns:
        val = df.loc[250, col]
        print(f"  {col:30s}: {val:.2f}" if pd.notna(val) else f"  {col:30s}: NaN")

print("\n=== TEST COMPLETE ===")
if present == len(expected_features):
    print("SUCCESS: All 20 features calculated correctly!")
else:
    print(f"PARTIAL: {present}/{len(expected_features)} features present")
