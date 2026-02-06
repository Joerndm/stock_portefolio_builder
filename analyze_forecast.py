"""Quick analysis of forecast results"""
import pandas as pd
import numpy as np

# Load forecast data
df = pd.read_excel('generated_forecasts/forecast_DEMANT.CO.xlsx')

print("=" * 60)
print("FORECAST ANALYSIS - DEMANT.CO")
print("=" * 60)

print(f"\nDate range: {df['date'].min()} to {df['date'].max()}")
print(f"Total rows: {len(df)}")

print("\n" + "=" * 60)
print("FIRST 10 DAYS (Historical Predictions if applicable)")
print("=" * 60)
print(df[['date', 'close_Price', '1D']].head(10).to_string())

print("\n" + "=" * 60)
print("LAST 10 DAYS (Future Predictions)")
print("=" * 60)
print(df[['date', 'close_Price', '1D']].tail(10).to_string())

print("\n" + "=" * 60)
print("DAILY RETURN STATISTICS")
print("=" * 60)
print(f"1D Return mean: {df['1D'].mean()*100:.4f}%")
print(f"1D Return std:  {df['1D'].std()*100:.4f}%")
print(f"1D Return min:  {df['1D'].min()*100:.4f}%")
print(f"1D Return max:  {df['1D'].max()*100:.4f}%")
print(f"Median return:  {df['1D'].median()*100:.4f}%")

print("\n" + "=" * 60)
print("PRICE TRAJECTORY")
print("=" * 60)
print(f"Start price: {df['close_Price'].iloc[0]:.2f}")
print(f"End price:   {df['close_Price'].iloc[-1]:.2f}")
print(f"Total return: {((df['close_Price'].iloc[-1]/df['close_Price'].iloc[0])-1)*100:.2f}%")

# Calculate if predictions seem realistic
days = len(df)
daily_return = df['1D'].mean()
annualized_return = (1 + daily_return) ** 252 - 1

print("\n" + "=" * 60)
print("REASONABLENESS CHECK")
print("=" * 60)
print(f"Prediction horizon: {days} days")
print(f"Average daily return: {daily_return*100:.4f}%")
print(f"Implied annualized return: {annualized_return*100:.2f}%")

# Check for unrealistic patterns
consecutive_same_sign = 0
max_consecutive = 0
prev_sign = None
for ret in df['1D']:
    current_sign = 1 if ret > 0 else -1
    if prev_sign == current_sign:
        consecutive_same_sign += 1
        max_consecutive = max(max_consecutive, consecutive_same_sign)
    else:
        consecutive_same_sign = 1
    prev_sign = current_sign

print(f"Max consecutive same-direction days: {max_consecutive}")

# Check variance over time
first_half = df['1D'].iloc[:len(df)//2].std()
second_half = df['1D'].iloc[len(df)//2:].std()
print(f"Std dev first half:  {first_half*100:.4f}%")
print(f"Std dev second half: {second_half*100:.4f}%")

# Typical stock benchmarks
print("\n" + "=" * 60)
print("BENCHMARK COMPARISON")
print("=" * 60)
print("Typical S&P 500: 0.04% daily mean, 1.0% daily std")
print("Typical volatile stock: 0.06% daily mean, 2.0% daily std")
print("Typical Danish stock: 0.03% daily mean, 1.5% daily std")
print(f"\nYour predictions: {daily_return*100:.4f}% daily mean, {df['1D'].std()*100:.4f}% daily std")

if abs(daily_return) > 0.01:  # >1% daily is unusual
    print("\n⚠️ WARNING: Average daily return seems high")
if df['1D'].std() < 0.005:  # <0.5% std is too smooth
    print("\n⚠️ WARNING: Predictions may be too smooth (low variance)")
if max_consecutive > 10:
    print("\n⚠️ WARNING: Too many consecutive same-direction days")
