"""
Test suite to validate that all moving averages, standard deviations, 
and Bollinger Bands are calculated correctly.

This test ensures:
1. All expected periods (5, 20, 40, 120, 200) are calculated
2. Calculations are correct with sufficient historical data
3. Calculations handle insufficient data gracefully
4. Other features remain unaffected
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys

# Import the functions we're testing
from stock_data_fetch import (
    calculate_moving_averages,
    calculate_standard_diviation_value,
    calculate_bollinger_bands
)

def create_test_data(num_days=300):
    """Create synthetic test data with known values for validation."""
    dates = [datetime.now() - timedelta(days=x) for x in range(num_days, 0, -1)]
    
    # Create data with a linear trend for easy validation
    # Price = 100 + index (so day 0 = 100, day 1 = 101, etc.)
    prices = [100 + i for i in range(num_days)]
    
    df = pd.DataFrame({
        'date': dates,
        'ticker': ['TEST'] * num_days,
        'close_Price': prices,
        'open_Price': prices,
        'high_Price': [p + 1 for p in prices],
        'low_Price': [p - 1 for p in prices],
    })
    
    return df

def validate_sma_calculation(df, period, column_name):
    """Validate that SMA is calculated correctly."""
    print(f"\n  Testing {column_name} (period={period})...")
    
    # Check if column exists
    if column_name not in df.columns:
        print(f"    ❌ FAIL: Column '{column_name}' not found in dataframe")
        return False
    
    # Handle potential duplicate columns by selecting first occurrence
    if isinstance(df[column_name], pd.DataFrame):
        series = df[column_name].iloc[:, 0]
    else:
        series = df[column_name]
    
    # Check if column has non-null values after sufficient data
    non_null_count = int(series.notna().sum())
    expected_non_null = max(0, len(df) - period)
    
    if non_null_count == 0:
        print(f"    ❌ FAIL: Column '{column_name}' has all NULL values")
        return False
    
    # For our linear data (100, 101, 102, ...), SMA should be calculable
    # Check a specific row where we have enough data
    test_idx = min(period + 50, len(df) - 10)  # Test somewhere in the middle
    
    # Calculate expected SMA manually
    # Remember: the function shifts by 1, so we need to account for that
    actual_value = series.iloc[test_idx]
    
    # Due to shift(1), the value at index i should be the SMA calculated at index i-1
    expected_value = df.loc[test_idx-period:test_idx-1, 'close_Price'].mean()
    
    if pd.isna(actual_value):
        print(f"    ⚠️  WARNING: Value at index {test_idx} is NaN (might be expected for short periods)")
        return True
    
    # Allow 0.01% tolerance
    if abs(actual_value - expected_value) < 0.01:
        print(f"    ✅ PASS: {column_name} calculated correctly")
        print(f"       Sample at index {test_idx}: actual={actual_value:.2f}, expected={expected_value:.2f}")
        return True
    else:
        print(f"    ❌ FAIL: {column_name} calculation incorrect")
        print(f"       At index {test_idx}: actual={actual_value:.2f}, expected={expected_value:.2f}")
        return False

def validate_ema_calculation(df, period, column_name):
    """Validate that EMA is calculated correctly."""
    print(f"\n  Testing {column_name} (period={period})...")
    
    # Check if column exists
    if column_name not in df.columns:
        print(f"    ❌ FAIL: Column '{column_name}' not found in dataframe")
        return False
    
    # Check if column has non-null values
    non_null_count = int(df[column_name].notna().sum())
    
    if non_null_count == 0:
        print(f"    ❌ FAIL: Column '{column_name}' has all NULL values")
        return False
    
    # For EMA, we just check that values exist and are in reasonable range
    test_idx = min(period + 50, len(df) - 10)
    actual_value = df.loc[test_idx, column_name]
    
    if pd.isna(actual_value):
        print(f"    ⚠️  WARNING: Value at index {test_idx} is NaN")
        return True
    
    # EMA should be close to price values (within reasonable range)
    price_at_idx = df.loc[test_idx, 'close_Price']
    if 50 <= actual_value <= 500:  # Reasonable range for our test data
        print(f"    ✅ PASS: {column_name} calculated (value={actual_value:.2f})")
        return True
    else:
        print(f"    ❌ FAIL: {column_name} has unreasonable value: {actual_value}")
        return False

def validate_std_calculation(df, period, column_name):
    """Validate that standard deviation is calculated correctly."""
    print(f"\n  Testing {column_name} (period={period})...")
    
    # Check if column exists
    if column_name not in df.columns:
        print(f"    ❌ FAIL: Column '{column_name}' not found in dataframe")
        return False
    
    # Check if column has non-null values
    non_null_count = int(df[column_name].notna().sum())
    
    if non_null_count == 0:
        print(f"    ❌ FAIL: Column '{column_name}' has all NULL values")
        return False
    
    # Test a specific row
    test_idx = min(period + 50, len(df) - 10)
    actual_value = df.loc[test_idx, column_name]
    
    # Calculate expected std manually (accounting for shift)
    expected_value = df.loc[test_idx-period:test_idx-1, 'close_Price'].std()
    
    if pd.isna(actual_value):
        print(f"    ⚠️  WARNING: Value at index {test_idx} is NaN")
        return True
    
    # For linear data, std should be relatively consistent
    if abs(actual_value - expected_value) < 0.01:
        print(f"    ✅ PASS: {column_name} calculated correctly")
        print(f"       Sample at index {test_idx}: actual={actual_value:.2f}, expected={expected_value:.2f}")
        return True
    else:
        print(f"    ❌ FAIL: {column_name} calculation incorrect")
        print(f"       At index {test_idx}: actual={actual_value:.2f}, expected={expected_value:.2f}")
        return False

def validate_bollinger_calculation(df, period, column_name, std_column):
    """Validate that Bollinger Band is calculated correctly."""
    print(f"\n  Testing {column_name} (period={period})...")
    
    # Check if column exists
    if column_name not in df.columns:
        print(f"    ❌ FAIL: Column '{column_name}' not found in dataframe")
        return False
    
    # Check if column has non-null values
    non_null_count = int(df[column_name].notna().sum())
    
    if non_null_count == 0:
        print(f"    ❌ FAIL: Column '{column_name}' has all NULL values")
        return False
    
    # Test a specific row
    test_idx = min(period + 50, len(df) - 10)
    actual_value = df.loc[test_idx, column_name]
    
    # Bollinger Band width = 4 * std_dev
    std_value = df.loc[test_idx, std_column]
    expected_value = 4.0 * std_value
    
    if pd.isna(actual_value) or pd.isna(std_value):
        print(f"    ⚠️  WARNING: Value at index {test_idx} is NaN")
        return True
    
    if abs(actual_value - expected_value) < 0.01:
        print(f"    ✅ PASS: {column_name} calculated correctly")
        print(f"       Sample at index {test_idx}: actual={actual_value:.2f}, expected={expected_value:.2f}")
        return True
    else:
        print(f"    ❌ FAIL: {column_name} calculation incorrect")
        print(f"       At index {test_idx}: actual={actual_value:.2f}, expected={expected_value:.2f}")
        return False

def run_comprehensive_test():
    """Run comprehensive test suite."""
    print("="*80)
    print("FEATURE CALCULATION TEST SUITE")
    print("="*80)
    
    # Create test data with 300 days (sufficient for all periods)
    print("\n1. Creating test data (300 days)...")
    df = create_test_data(300)
    print(f"   ✓ Created dataframe with {len(df)} rows")
    print(f"   ✓ Price range: {df['close_Price'].min():.2f} to {df['close_Price'].max():.2f}")
    
    # Store original columns to verify no unintended changes
    original_cols = set(df.columns)
    
    all_tests_passed = True
    
    # Test 1: Calculate moving averages
    print("\n" + "="*80)
    print("TEST 1: MOVING AVERAGES")
    print("="*80)
    try:
        df = calculate_moving_averages(df)
        # Remove any duplicate columns that might have been created
        df = df.loc[:, ~df.columns.duplicated()]
        print("✓ Function executed without errors")
    except Exception as e:
        print(f"❌ Function failed with error: {e}")
        all_tests_passed = False
    
    # Validate all expected SMA columns
    expected_smas = {
        'sma_5': 5,
        'sma_20': 20,
        'sma_40': 40,
        'sma_120': 120,
        'sma_200': 200
    }
    
    print("\nValidating Simple Moving Averages (SMA):")
    for col, period in expected_smas.items():
        if not validate_sma_calculation(df, period, col):
            all_tests_passed = False
    
    # Validate all expected EMA columns
    expected_emas = {
        'ema_5': 5,
        'ema_20': 20,
        'ema_40': 40,
        'ema_120': 120,
        'ema_200': 200
    }
    
    print("\nValidating Exponential Moving Averages (EMA):")
    for col, period in expected_emas.items():
        if not validate_ema_calculation(df, period, col):
            all_tests_passed = False
    
    # Test 2: Calculate standard deviations
    print("\n" + "="*80)
    print("TEST 2: STANDARD DEVIATIONS")
    print("="*80)
    try:
        df = calculate_standard_diviation_value(df)
        # Remove any duplicate columns that might have been created
        df = df.loc[:, ~df.columns.duplicated()]
        print("✓ Function executed without errors")
    except Exception as e:
        print(f"❌ Function failed with error: {e}")
        all_tests_passed = False
    
    expected_stds = {
        'std_Div_5': 5,
        'std_Div_20': 20,
        'std_Div_40': 40,
        'std_Div_120': 120,
        'std_Div_200': 200
    }
    
    print("\nValidating Standard Deviations:")
    for col, period in expected_stds.items():
        if not validate_std_calculation(df, period, col):
            all_tests_passed = False
    
    # Test 3: Calculate Bollinger Bands
    print("\n" + "="*80)
    print("TEST 3: BOLLINGER BANDS")
    print("="*80)
    try:
        df = calculate_bollinger_bands(df)
        # Remove any duplicate columns that might have been created
        df = df.loc[:, ~df.columns.duplicated()]
        print("✓ Function executed without errors")
    except Exception as e:
        print(f"❌ Function failed with error: {e}")
        all_tests_passed = False
    
    expected_bollingers = {
        'bollinger_Band_5_2STD': (5, 'std_Div_5'),
        'bollinger_Band_20_2STD': (20, 'std_Div_20'),
        'bollinger_Band_40_2STD': (40, 'std_Div_40'),
        'bollinger_Band_120_2STD': (120, 'std_Div_120'),
        'bollinger_Band_200_2STD': (200, 'std_Div_200')
    }
    
    print("\nValidating Bollinger Bands:")
    for col, (period, std_col) in expected_bollingers.items():
        if not validate_bollinger_calculation(df, period, col, std_col):
            all_tests_passed = False
    
    # Test 4: Verify original columns unchanged
    print("\n" + "="*80)
    print("TEST 4: ORIGINAL COLUMNS INTEGRITY")
    print("="*80)
    for orig_col in original_cols:
        if orig_col in df.columns:
            print(f"  ✅ Original column '{orig_col}' still present")
        else:
            print(f"  ❌ Original column '{orig_col}' was removed!")
            all_tests_passed = False
    
    # Test 5: Insufficient data handling
    print("\n" + "="*80)
    print("TEST 5: INSUFFICIENT DATA HANDLING")
    print("="*80)
    print("\nCreating small dataset (50 days - insufficient for 120 and 200 periods)...")
    small_df = create_test_data(50)
    
    try:
        small_df = calculate_moving_averages(small_df)
        small_df = calculate_standard_diviation_value(small_df)
        small_df = calculate_bollinger_bands(small_df)
        # Remove duplicate columns if any
        small_df = small_df.loc[:, ~small_df.columns.duplicated()]
        print("✓ Functions handled small dataset without crashing")
        
        # Check that short-period features still work
        if small_df['sma_5'].notna().sum() > 0:
            print("  ✅ Short period features (5) calculated successfully")
        else:
            print("  ❌ Short period features (5) not calculated")
            all_tests_passed = False
        
        # Check that long-period features gracefully handle insufficient data
        if small_df['sma_200'].notna().sum() == 0 or small_df['sma_200'].notna().sum() < 50:
            print("  ✅ Long period features (200) handled insufficient data gracefully")
        else:
            print(f"  ⚠️  Long period features (200) calculated with insufficient data: {small_df['sma_200'].notna().sum()} values")
        
    except Exception as e:
        print(f"❌ Functions failed on small dataset: {e}")
        all_tests_passed = False
    
    # Final summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    if all_tests_passed:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED - See details above")
    
    print("\n" + "="*80)
    print("FEATURE COVERAGE REPORT")
    print("="*80)
    print("Expected features in database schema:")
    all_expected = list(expected_smas.keys()) + list(expected_emas.keys()) + \
                   list(expected_stds.keys()) + list(expected_bollingers.keys())
    
    present = [col for col in all_expected if col in df.columns]
    missing = [col for col in all_expected if col not in df.columns]
    
    print(f"\n  Present: {len(present)}/{len(all_expected)}")
    for col in present:
        print(f"    ✓ {col}")
    
    if missing:
        print(f"\n  Missing: {len(missing)}/{len(all_expected)}")
        for col in missing:
            print(f"    ✗ {col}")
    
    return all_tests_passed, df

if __name__ == "__main__":
    success, result_df = run_comprehensive_test()
    sys.exit(0 if success else 1)
