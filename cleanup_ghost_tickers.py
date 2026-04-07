"""
Database Cleanup Script for Ghost and Renamed Tickers

Removes data for tickers that are blacklisted (delisted/invalid) but still
have residual records in the database, and handles renamed tickers.

Ghost tickers: CHR.CO, NZYM-B.CO — blacklisted but still in stock_info_data
Renamed tickers: SNDK→WDC, CI→CIGNA Group, HUM→acquired, ELV→ANTM→ELV
    These have price data but no financials (ticker changed, API returns nothing)

Usage:
    python cleanup_ghost_tickers.py              # Dry-run (show what would be deleted)
    python cleanup_ghost_tickers.py --execute    # Actually perform cleanup
"""
import argparse
import pandas as pd
from sqlalchemy import text as sa_text

import fetch_secrets
import db_connectors


# Tables to clean (in order to respect foreign key constraints)
TABLES_TO_CLEAN = [
    'stock_ratio_data',
    'stock_income_stmt_data',
    'stock_balancesheet_data',
    'stock_cash_flow_data',
    'stock_income_stmt_quarterly',
    'stock_balancesheet_quarterly',
    'stock_cashflow_quarterly',
    'stock_prediction_extended',
    'monte_carlo_results',
    'stock_price_data',
    'stock_info_data',
]

# Ghost tickers: blacklisted, no usable data
GHOST_TICKERS = ['CHR.CO', 'NZYM-B.CO']

# Renamed tickers: old symbols with orphaned data
RENAMED_TICKERS = ['SNDK', 'CI', 'HUM', 'ELV']


def cleanup_tickers(engine, tickers, execute=False):
    """Remove all data for the given tickers from all tables."""
    total_deleted = 0
    
    for ticker in tickers:
        print(f"\n{'─'*40}")
        print(f"Ticker: {ticker}")
        print(f"{'─'*40}")
        
        for table in TABLES_TO_CLEAN:
            try:
                count_query = sa_text(f"SELECT COUNT(*) as cnt FROM {table} WHERE ticker = :ticker")
                result = pd.read_sql(count_query, engine, params={"ticker": ticker})
                count = result['cnt'].iloc[0]
                
                if count > 0:
                    if execute:
                        delete_query = sa_text(f"DELETE FROM {table} WHERE ticker = :ticker")
                        with engine.begin() as conn:
                            conn.execute(delete_query, {"ticker": ticker})
                        print(f"  ✓ {table}: deleted {count} rows")
                    else:
                        print(f"  → {table}: {count} rows (would delete)")
                    total_deleted += count
            except Exception as e:
                if "doesn't exist" not in str(e):
                    print(f"  ⚠ {table}: error — {e}")
    
    return total_deleted


def main():
    parser = argparse.ArgumentParser(description='Clean up ghost/renamed tickers from database')
    parser.add_argument('--execute', action='store_true', help='Actually delete (default: dry-run)')
    args = parser.parse_args()
    
    db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()
    engine = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)
    
    mode = "EXECUTE" if args.execute else "DRY-RUN"
    print(f"\n{'='*60}")
    print(f"DATABASE CLEANUP — {mode}")
    print(f"{'='*60}")
    
    print(f"\n--- Ghost tickers (blacklisted, no data) ---")
    ghost_total = cleanup_tickers(engine, GHOST_TICKERS, args.execute)
    
    print(f"\n--- Renamed tickers (orphaned data) ---")
    renamed_total = cleanup_tickers(engine, RENAMED_TICKERS, args.execute)
    
    total = ghost_total + renamed_total
    print(f"\n{'='*60}")
    if args.execute:
        print(f"CLEANUP COMPLETE: {total} rows deleted")
    else:
        print(f"DRY-RUN COMPLETE: {total} rows would be deleted")
        print(f"Run with --execute to perform the actual cleanup")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
