"""Deeper analysis of forecast patterns"""
import pandas as pd
import numpy as np

# Load forecast data
df = pd.read_excel('generated_forecasts/forecast_DEMANT.CO.xlsx')

print("=" * 70)
print("DETAILED FORECAST PATTERN ANALYSIS")
print("=" * 70)

# Identify where predictions become "stuck" (same value repeated)
print("\n1. CHECKING FOR REPEATED/STUCK PREDICTIONS")
print("-" * 70)

# Find unique 1D values and their counts
unique_counts = df['1D'].value_counts()
print(f"Total unique daily return values: {len(unique_counts)}")
print(f"\nMost repeated values:")
print(unique_counts.head(10))

# Find the transition point where predictions become constant
const_threshold = 0.000001  # Values within this are "same"
for i in range(1, len(df)):
    if abs(df['1D'].iloc[i] - df['1D'].iloc[i-1]) < const_threshold:
        start_const = i - 1
        # Find how long it stays constant
        const_length = 1
        for j in range(i, len(df)):
            if abs(df['1D'].iloc[j] - df['1D'].iloc[start_const]) < const_threshold:
                const_length += 1
            else:
                break
        
        if const_length > 5:  # Only report if stuck for >5 days
            print(f"\nConstant prediction starting at row {start_const}:")
            print(f"  Date: {df['date'].iloc[start_const]}")
            print(f"  Value: {df['1D'].iloc[start_const]*100:.4f}%")
            print(f"  Duration: {const_length} days")
            break

# Check the last 90 days (prediction period)
print("\n2. ANALYZING LAST 90 DAYS (FUTURE PREDICTIONS)")
print("-" * 70)
last_90 = df.tail(90)
print(f"Date range: {last_90['date'].iloc[0]} to {last_90['date'].iloc[-1]}")
print(f"Unique 1D values in last 90 days: {last_90['1D'].nunique()}")
print(f"Mean return: {last_90['1D'].mean()*100:.6f}%")
print(f"Std deviation: {last_90['1D'].std()*100:.6f}%")
print(f"\nIf std dev is very low, predictions are 'mode collapsed'")

# Check first 90 days for comparison
print("\n3. ANALYZING FIRST 90 DAYS (LIKELY HISTORICAL)")
print("-" * 70)
first_90 = df.head(90)
print(f"Date range: {first_90['date'].iloc[0]} to {first_90['date'].iloc[-1]}")
print(f"Unique 1D values in first 90 days: {first_90['1D'].nunique()}")
print(f"Mean return: {first_90['1D'].mean()*100:.6f}%")
print(f"Std deviation: {first_90['1D'].std()*100:.6f}%")

# Find where the forecast starts showing issues
print("\n4. FINDING TRANSITION POINT")
print("-" * 70)
rolling_std = df['1D'].rolling(20).std()
for i in range(50, len(df)):
    if rolling_std.iloc[i] < 0.001:  # Very low variance
        print(f"Low variance detected starting around row {i}")
        print(f"  Date: {df['date'].iloc[i]}")
        print(f"  20-day rolling std: {rolling_std.iloc[i]*100:.6f}%")
        break
else:
    print("No sudden variance collapse detected")

# Check if this is the prediction vs actual boundary
print("\n5. CHECKING DATA TYPES AND BOUNDARIES")
print("-" * 70)
# The forecast_df likely has historical actuals + future predictions
# Let's see if there's a pattern change

# Count how many days have "realistic" variance
realistic_days = (df['1D'].rolling(5).std() > 0.001).sum()
constant_days = len(df) - realistic_days
print(f"Days with normal variance: {realistic_days}")
print(f"Days with near-constant predictions: {constant_days}")

# Show the last 20 rows to see the pattern
print("\n6. DETAILED VIEW OF LAST 20 PREDICTIONS")
print("-" * 70)
for idx, row in df.tail(20).iterrows():
    print(f"{row['date'].strftime('%Y-%m-%d')}: Price={row['close_Price']:.2f}, 1D={row['1D']*100:.4f}%")

# Calculate what a realistic prediction range should be
print("\n7. EXPECTED REALISTIC RANGES")
print("-" * 70)
historical_std = first_90['1D'].std()
print(f"Based on first 90 days, typical daily std: {historical_std*100:.4f}%")
print(f"Expected 1D range (95% CI): [{-2*historical_std*100:.2f}%, +{2*historical_std*100:.2f}%]")
print(f"Expected 90-day return range: [{-2*historical_std*np.sqrt(90)*100:.2f}%, +{2*historical_std*np.sqrt(90)*100:.2f}%]")
print(f"\nActual 90-day future return: {((df['close_Price'].iloc[-1]/df['close_Price'].iloc[-91])-1)*100:.2f}%")
