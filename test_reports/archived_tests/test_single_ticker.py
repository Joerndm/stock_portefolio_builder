"""Test script to fetch and calculate features for a single ticker."""
import stock_data_fetch as sdf
import datetime
from dateutil.relativedelta import relativedelta

# Test with one ticker
ticker = 'DEMANT.CO'
print(f'Fetching data for {ticker}...')
df = sdf.fetch_stock_price_data(ticker, datetime.datetime.now() - relativedelta(years=1))
print(f'Fetched {len(df)} rows')

df = sdf.calculate_period_returns(df)
df = sdf.add_technical_indicators(df)
df = sdf.add_volume_indicators(df)
df = sdf.add_volatility_indicators(df)
df = sdf.calculate_moving_averages(df)
df = sdf.calculate_standard_diviation_value(df)
df = sdf.calculate_bollinger_bands(df)
df = sdf.calculate_momentum(df)

print(f'After calculations: {len(df)} rows')
print(f'Columns: {len(df.columns)}')
print('\nNew features present:')
for col in ['sma_5', 'sma_20', 'sma_40', 'sma_120', 'sma_200',
            'ema_5', 'ema_20', 'ema_40', 'ema_120', 'ema_200',
            'std_Div_5', 'std_Div_20', 'std_Div_40', 'std_Div_120', 'std_Div_200',
            'bollinger_Band_5_2STD', 'bollinger_Band_20_2STD', 'bollinger_Band_40_2STD',
            'bollinger_Band_120_2STD', 'bollinger_Band_200_2STD']:
    if col in df.columns:
        non_null = df[col].notna().sum()
        pct = (non_null / len(df) * 100) if len(df) > 0 else 0
        print(f'  {col:30s}: {non_null:3d}/{len(df)} ({pct:.1f}%)')
    else:
        print(f'  {col:30s}: MISSING')

print('\nSample values (last 5 rows):')
print(df[['close_Price', 'sma_5', 'sma_40', 'sma_200', 'std_Div_40', 'bollinger_Band_40_2STD']].tail())
