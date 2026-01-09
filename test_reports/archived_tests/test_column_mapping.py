"""
Quick diagnostic to check column names and duplicates
"""

import pandas as pd
import stock_data_fetch
import yfinance as yf

test_ticker = "DEMANT.CO"
print("Fetching and processing data...")

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

print("\n" + "="*80)
print("DATAFRAME COLUMNS:")
print("="*80)
print(f"Total columns: {len(stock_price_data_df.columns)}")
print("\nColumn list:")
for i, col in enumerate(stock_price_data_df.columns, 1):
    print(f"{i:3d}. {col}")

# Check for duplicates
duplicate_cols = stock_price_data_df.columns[stock_price_data_df.columns.duplicated()].tolist()
if duplicate_cols:
    print("\n" + "="*80)
    print("⚠️  DUPLICATE COLUMNS FOUND:")
    print("="*80)
    for col in set(duplicate_cols):
        count = stock_price_data_df.columns.tolist().count(col)
        print(f"   {col}: appears {count} times")

print("\n" + "="*80)
print("DATABASE SCHEMA (from ddl.sql):")
print("="*80)
db_columns = [
    'date', 'ticker', 'currency', 'trade_Volume', 'open_Price', 'high_Price', 'low_Price', 'close_Price',
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
print(f"Total columns: {len(db_columns)}")

print("\n" + "="*80)
print("COLUMN NAME MAPPING NEEDED:")
print("="*80)

# Check which columns need mapping
dataframe_cols = set(stock_price_data_df.columns)
db_cols_set = set(db_columns)

print("\nColumns in DataFrame but NOT in database schema:")
for col in sorted(dataframe_cols - db_cols_set):
    print(f"   ❌ {col}")

print("\nColumns in database schema but NOT in DataFrame:")
for col in sorted(db_cols_set - dataframe_cols):
    print(f"   ⚠️  {col}")

print("\n" + "="*80)
print("SUGGESTED COLUMN NAME MAPPING:")
print("="*80)

# Suggest mappings
mappings = []
for df_col in dataframe_cols:
    if df_col not in db_cols_set:
        # Check for case-insensitive match
        for db_col in db_cols_set:
            if df_col.lower() == db_col.lower():
                mappings.append((df_col, db_col))
                print(f"   '{df_col}' → '{db_col}'")
                break

if not mappings:
    print("   No simple case mappings found")
