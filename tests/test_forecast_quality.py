"""
Test script to verify forecast quality after implementing prediction stabilization fixes:
- Fix #1: Prediction noise/uncertainty
- Fix #2: Mean reversion constraint  
- Fix #3: Momentum feature trap fix
- Fix #4: TCN implementation (replacing LSTM)
- Fix #5: Monte Carlo Dropout for uncertainty estimation

Success criteria:
1. Std dev of last 90 days > 1%
2. Unique daily return values > 50
3. No constant predictions > 5 consecutive days
"""
import pandas as pd
import numpy as np
import os

def count_max_consecutive_same(series, tolerance=0.0001):
    """Count max consecutive same values (within tolerance)."""
    values = series.round(6).values
    max_count = 1
    current_count = 1
    for i in range(1, len(values)):
        if abs(values[i] - values[i-1]) < tolerance:
            current_count += 1
            max_count = max(max_count, current_count)
        else:
            current_count = 1
    return max_count

def count_max_same_direction(returns_series):
    """Count max consecutive days with same direction."""
    directions = np.sign(returns_series.values)
    max_count = 1
    current_count = 1
    for i in range(1, len(directions)):
        if directions[i] == directions[i-1]:
            current_count += 1
            max_count = max(max_count, current_count)
        else:
            current_count = 1
    return max_count

def analyze_forecast(forecast_path):
    """Analyze forecast quality and check success criteria."""
    
    if not os.path.exists(forecast_path):
        print(f"ERROR: {forecast_path} not found. Run ml_builder.py first.")
        return False
        
    df = pd.read_excel(forecast_path)
    
    print("=" * 70)
    print("FORECAST QUALITY ANALYSIS (Post-Fix #1-3 and implementation of TCN model)")
    print("=" * 70)
    
    # Basic info
    print(f"\nDate range: {df['date'].min()} to {df['date'].max()}")
    print(f"Total rows: {len(df)}")
    
    # Analyze 1D returns distribution
    returns = df['1D'].dropna()
    print(f"\n--- Overall 1D Returns Statistics ---")
    print(f"  Mean:   {returns.mean()*100:.4f}%")
    print(f"  Median: {returns.median()*100:.4f}%")
    print(f"  Std:    {returns.std()*100:.4f}%")
    print(f"  Min:    {returns.min()*100:.4f}%")
    print(f"  Max:    {returns.max()*100:.4f}%")
    
    # Check last 90 days specifically (future predictions)
    last_90 = df.tail(90)
    last_90_returns = last_90['1D'].dropna()
    unique_last_90 = last_90_returns.round(6).nunique()
    
    print(f"\n--- Future Predictions (Last 90 days) ---")
    print(f"  Unique 1D values: {unique_last_90}")
    print(f"  Mean return: {last_90_returns.mean()*100:.4f}%")
    print(f"  Std deviation: {last_90_returns.std()*100:.4f}%")
    
    # === SUCCESS CRITERIA VALIDATION ===
    print("\n" + "=" * 70)
    print("SUCCESS CRITERIA CHECK")
    print("=" * 70)
    
    # Criterion 1: Std dev of last 90 days > 1%
    std_last_90 = last_90_returns.std() * 100
    criterion_1_pass = std_last_90 > 1.0
    print(f"\n1. Std dev of last 90 days > 1%")
    print(f"   Result: {std_last_90:.4f}%")
    print(f"   Status: {'✅ PASS' if criterion_1_pass else '❌ FAIL'}")
    
    # Criterion 2: Unique daily values > 50
    criterion_2_pass = unique_last_90 > 50
    print(f"\n2. Unique daily return values > 50")
    print(f"   Result: {unique_last_90}")
    print(f"   Status: {'✅ PASS' if criterion_2_pass else '❌ FAIL'}")
    
    # Criterion 3: No constant predictions > 5 consecutive days
    max_consecutive = count_max_consecutive_same(last_90_returns)
    criterion_3_pass = max_consecutive <= 5
    print(f"\n3. No constant predictions > 5 consecutive days")
    print(f"   Max consecutive same-value days: {max_consecutive}")
    print(f"   Status: {'✅ PASS' if criterion_3_pass else '❌ FAIL'}")
    
    # Summary
    all_pass = criterion_1_pass and criterion_2_pass and criterion_3_pass
    print("\n" + "=" * 70)
    print(f"OVERALL: {'✅ ALL CRITERIA PASS' if all_pass else '❌ CRITERIA FAILED'}")
    print("=" * 70)
    
    # Additional analysis
    print(f"\n--- Mode Collapse Detection ---")
    rolling_std = returns.rolling(20).std()
    constant_mask = rolling_std < 0.0001  # Near-zero variance
    
    if constant_mask.any():
        first_constant = constant_mask.idxmax()
        print(f"⚠️  Constant prediction starting at row {first_constant}:")
        print(f"    Date: {df.loc[first_constant, 'date']}")
        print(f"    Value: {df.loc[first_constant, '1D']*100:.4f}%")
        print(f"    Duration: {len(df) - first_constant} days")
    else:
        print("✅ No constant predictions detected (good!)")
    
    # Check consecutive same-direction days (overall)
    max_same_dir = count_max_same_direction(returns)
    print(f"\n--- Direction Analysis ---")
    print(f"Max consecutive same-direction days (entire dataset): {max_same_dir}")
    if max_same_dir > 15:
        print("   Note: This may include historical market data with real trends")
    
    # Check future predictions specifically
    max_same_dir_future = count_max_same_direction(last_90_returns)
    print(f"Max consecutive same-direction days (future only): {max_same_dir_future}")
    if max_same_dir_future > 15:
        print("⚠️  WARNING: Future predictions may have directional bias")
    else:
        print("✅ Future predictions have good directional balance")
    
    # Show comparison: first 90 days vs last 90 days
    first_90 = df.head(90)
    first_90_returns = first_90['1D'].dropna()
    print(f"\n--- Comparison: Historical vs Future ---")
    print(f"{'Metric':<30} {'Historical (First 90)':<25} {'Future (Last 90)':<25}")
    print("-" * 80)
    print(f"{'Unique 1D values':<30} {first_90_returns.round(6).nunique():<25} {unique_last_90:<25}")
    print(f"{'Std deviation':<30} {first_90_returns.std()*100:>23.4f}% {last_90_returns.std()*100:>23.4f}%")
    print(f"{'Mean return':<30} {first_90_returns.mean()*100:>23.4f}% {last_90_returns.mean()*100:>23.4f}%")
    
    # Show sample of last 10 predictions
    print(f"\n--- Last 10 Predictions ---")
    print(df[['date', '1D', 'close_Price']].tail(10).to_string())
    
    return all_pass

if __name__ == "__main__":
    forecast_path = "generated_forecasts/forecast_DEMANT.CO.xlsx"
    analyze_forecast(forecast_path)
