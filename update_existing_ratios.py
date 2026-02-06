"""
Update ratio data for all existing tickers in the database.

This script will:
1. Find all tickers that have ratio data
2. Run process_ratio_data() for each, which will:
   - Recalculate if financial_date_used is NULL (legacy data)
   - Recalculate if new financial data is available
   - Add new days if already up to date
"""
import sys
import pandas as pd

# Mock pandas_ta to avoid import issues
class MockPandasTA:
    def __getattr__(self, name):
        return lambda *args, **kwargs: pd.Series()

sys.modules['pandas_ta'] = MockPandasTA()

import db_interactions
import fetch_secrets
import db_connectors
from stock_orchestrator import StockDataOrchestrator

# Connect to database
db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()
db_con = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)

# Get all tickers with ratio data
tickers_df = pd.read_sql("""
    SELECT DISTINCT ticker 
    FROM stock_ratio_data 
    ORDER BY ticker
""", db_con)

print(f"Found {len(tickers_df)} tickers with ratio data\n")

# Initialize orchestrator
orchestrator = StockDataOrchestrator()

# Process each ticker
results = {'success': 0, 'skipped': 0, 'errors': 0}

for idx, row in tickers_df.iterrows():
    ticker = row['ticker']
    print(f"\n[{idx+1}/{len(tickers_df)}] Processing {ticker}...")
    
    try:
        success, ratio_df = orchestrator.process_ratio_data(ticker, prefer_ttm=False)
        
        if success:
            if ratio_df is not None:
                print(f"   ✓ Updated {len(ratio_df)} rows")
                results['success'] += 1
            else:
                print(f"   ✓ Already up to date")
                results['skipped'] += 1
        else:
            print(f"   ⚠ Skipped (no data)")
            results['skipped'] += 1
            
    except Exception as e:
        print(f"   ✗ Error: {e}")
        results['errors'] += 1

print(f"\n{'='*50}")
print(f"SUMMARY")
print(f"{'='*50}")
print(f"Updated: {results['success']}")
print(f"Skipped: {results['skipped']}")
print(f"Errors:  {results['errors']}")
