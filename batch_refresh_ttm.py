"""
Batch Refresh: Quarterly TTM Data & Ratios

Finds tickers with NULL revenue_ttm (beyond the first 3 quarters) and
re-triggers the quarterly data fetch + TTM calculation + ratio recalculation
for each. This fixes the issues caused by the old min_periods=4 rule.

Also re-calculates ratios for tickers that had extreme values (now nulled)
so they are replaced with properly-clamped values.

Usage:
    python batch_refresh_ttm.py              # Dry-run (show affected tickers)
    python batch_refresh_ttm.py --execute    # Actually refresh
    python batch_refresh_ttm.py --execute --max 50   # Limit batch size
"""
import argparse
import sys
import pandas as pd
from sqlalchemy import text as sa_text

import fetch_secrets
import db_connectors


def find_tickers_with_null_ttm(engine):
    """Find tickers that have unexpected NULL revenue_ttm beyond the first 3 quarters."""
    query = sa_text("""
        SELECT ticker,
               COUNT(*) AS total_quarters,
               SUM(CASE WHEN revenue_ttm IS NULL THEN 1 ELSE 0 END) AS null_ttm,
               MIN(fiscal_quarter_end) AS earliest_quarter
        FROM stock_income_stmt_quarterly
        GROUP BY ticker
        HAVING total_quarters >= 4 AND null_ttm > 3
        ORDER BY null_ttm DESC
    """)
    return pd.read_sql(query, engine)


def find_tickers_with_null_ratios(engine):
    """Find tickers where ALL ratio values are NULL (likely need fresh calculation)."""
    query = sa_text("""
        SELECT ticker,
               COUNT(*) AS total_rows,
               SUM(CASE WHEN p_e IS NULL AND p_b IS NULL AND p_s IS NULL AND p_fcf IS NULL
                   THEN 1 ELSE 0 END) AS all_null_rows
        FROM stock_ratio_data
        GROUP BY ticker
        HAVING all_null_rows > total_rows * 0.5
        ORDER BY all_null_rows DESC
    """)
    return pd.read_sql(query, engine)


def refresh_ticker(ticker, orchestrator):
    """Re-fetch quarterly data and recalculate ratios for a ticker."""
    print(f"\n  Processing {ticker}...")
    try:
        # Force re-fetch quarterly data
        orchestrator._fetch_and_export_quarterly_data(ticker, force_fetch=True)
        
        # Recalculate ratios (delete existing and recompute)
        orchestrator.process_ratio_data(ticker, prefer_ttm=True)
        
        return True
    except Exception as e:
        print(f"  ✗ Error refreshing {ticker}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Batch refresh TTM data and ratios')
    parser.add_argument('--execute', action='store_true', help='Actually refresh (default: dry-run)')
    parser.add_argument('--max', type=int, default=0, help='Max tickers to process (0 = all)')
    args = parser.parse_args()
    
    db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()
    engine = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)
    
    print("=" * 60)
    print("BATCH TTM REFRESH")
    print("=" * 60)
    
    # Find tickers needing refresh
    print("\n1. Finding tickers with NULL revenue_ttm...")
    null_ttm_df = find_tickers_with_null_ttm(engine)
    print(f"   Found {len(null_ttm_df)} tickers with excessive NULL TTM values")
    if not null_ttm_df.empty:
        for _, row in null_ttm_df.head(10).iterrows():
            print(f"   {row['ticker']}: {row['null_ttm']}/{row['total_quarters']} quarters NULL")
        if len(null_ttm_df) > 10:
            print(f"   ... and {len(null_ttm_df) - 10} more")
    
    print("\n2. Finding tickers with >50% NULL ratios...")
    null_ratio_df = find_tickers_with_null_ratios(engine)
    print(f"   Found {len(null_ratio_df)} tickers with mostly-NULL ratios")
    
    # Combine unique tickers
    tickers_to_refresh = set()
    if not null_ttm_df.empty:
        tickers_to_refresh.update(null_ttm_df['ticker'].tolist())
    if not null_ratio_df.empty:
        tickers_to_refresh.update(null_ratio_df['ticker'].tolist())
    
    tickers_to_refresh = sorted(tickers_to_refresh)
    
    if args.max > 0:
        tickers_to_refresh = tickers_to_refresh[:args.max]
    
    print(f"\n   Total unique tickers to refresh: {len(tickers_to_refresh)}")
    
    if not args.execute:
        print("\n[DRY-RUN] No changes made. Run with --execute to refresh.")
        return
    
    # Import orchestrator for processing
    from stock_orchestrator import StockDataOrchestrator
    orchestrator = StockDataOrchestrator()
    
    success = 0
    failed = 0
    for i, ticker in enumerate(tickers_to_refresh, 1):
        print(f"\n[{i}/{len(tickers_to_refresh)}] {ticker}")
        if refresh_ticker(ticker, orchestrator):
            success += 1
        else:
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"BATCH REFRESH COMPLETE")
    print(f"  Success: {success}")
    print(f"  Failed: {failed}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
