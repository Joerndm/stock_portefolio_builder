"""Debug script to understand CARL-B.CO validation failure."""
import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import db_connectors
import fetch_secrets
from ttm_financial_calculator import TTMFinancialCalculator

# Connect to DB
db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()
db_con = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)

# Get existing ratio data
existing = pd.read_sql(
    "SELECT date, p_s, p_e, p_b, p_fcf FROM stock_ratio_data WHERE ticker = 'CARL-B.CO' ORDER BY date",
    db_con
)
print("=== EXISTING DB DATA ===")
print(f"Total rows: {len(existing)}")
print(f"Date range: {existing['date'].min()} to {existing['date'].max()}")
print(f"P/S range: {existing['p_s'].min():.2f} to {existing['p_s'].max():.2f}")
print(f"P/E range: {existing['p_e'].min():.2f} to {existing['p_e'].max():.2f}")
print("\nFirst 5 rows:")
print(existing.head())
print("\nLast 5 rows:")
print(existing.tail())

# Get price data
price_data = pd.read_sql(
    "SELECT date, ticker, close_Price FROM stock_price_data WHERE ticker = 'CARL-B.CO' ORDER BY date",
    db_con
)
print("\n=== PRICE DATA ===")
print(f"Total rows: {len(price_data)}")
print(f"Date range: {price_data['date'].min()} to {price_data['date'].max()}")

# Calculate new ratios using TTM calculator
print("\n=== CALCULATING NEW RATIOS ===")
calculator = TTMFinancialCalculator()
new_df = calculator.calculate_ratios_with_source_tracking('CARL-B.CO', price_data, prefer_ttm=True)

print(f"Data source: {new_df['ratio_data_source'].iloc[0] if not new_df.empty else 'N/A'}")
print(f"Quarters available: {new_df['quarters_available'].iloc[0] if not new_df.empty else 'N/A'}")

if not new_df.empty:
    print(f"\nNew P/S range: {new_df['P/S'].min():.2f} to {new_df['P/S'].max():.2f}")
    print(f"New P/E range: {new_df['P/E'].min():.2f} to {new_df['P/E'].max():.2f}")
    print("\nFirst 5 rows of new data:")
    print(new_df[['date', 'P/S', 'P/E', 'P/B', 'P/FCF']].head())
    print("\nLast 5 rows of new data:")
    print(new_df[['date', 'P/S', 'P/E', 'P/B', 'P/FCF']].tail())

# Compare values
print("\n=== COMPARISON ===")
existing['date'] = pd.to_datetime(existing['date']).dt.date
new_df['date'] = pd.to_datetime(new_df['date']).dt.date

merged = pd.merge(
    existing[['date', 'p_s', 'p_e']].rename(columns={'p_s': 'existing_ps', 'p_e': 'existing_pe'}),
    new_df[['date', 'P/S', 'P/E']].rename(columns={'P/S': 'new_ps', 'P/E': 'new_pe'}),
    on='date',
    how='inner'
)
print(f"Matched rows: {len(merged)}")

if len(merged) > 0:
    # Check sample differences
    merged['ps_diff_pct'] = abs(merged['new_ps'] - merged['existing_ps']) / merged['existing_ps'].abs()
    merged['pe_diff_pct'] = abs(merged['new_pe'] - merged['existing_pe']) / merged['existing_pe'].abs()
    
    print("\nSample comparison (first 10 rows):")
    print(merged[['date', 'existing_ps', 'new_ps', 'ps_diff_pct', 'existing_pe', 'new_pe', 'pe_diff_pct']].head(10).to_string())
    
    print(f"\nP/S mean diff: {merged['ps_diff_pct'].mean():.2%}")
    print(f"P/E mean diff: {merged['pe_diff_pct'].mean():.2%}")
