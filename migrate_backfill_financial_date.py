"""
Migration: Backfill financial_date_used in stock_ratio_data

Many legacy ratio rows have NULL financial_date_used because this column was
added after the initial data was populated. This script fills it in by finding
the most recent financial_Statement_Date that precedes each ratio row's date.

This improves the orchestrator's ability to detect when new financial data
warrants a ratio recalculation (Scenario 2 vs 3 in process_ratio_data).

Usage:
    python migrate_backfill_financial_date.py              # Dry-run
    python migrate_backfill_financial_date.py --execute    # Actually update
"""
import argparse
import pandas as pd
from sqlalchemy import text as sa_text

import fetch_secrets
import db_connectors


def main():
    parser = argparse.ArgumentParser(description='Backfill financial_date_used in stock_ratio_data')
    parser.add_argument('--execute', action='store_true', help='Actually update (default: dry-run)')
    args = parser.parse_args()
    
    db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()
    engine = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)
    
    print("=" * 60)
    print("BACKFILL financial_date_used")
    print("=" * 60)
    
    # Count rows needing backfill
    count_q = sa_text("""
        SELECT COUNT(*) AS cnt,
               COUNT(DISTINCT ticker) AS tickers
        FROM stock_ratio_data
        WHERE financial_date_used IS NULL
    """)
    counts = pd.read_sql(count_q, engine)
    null_rows = int(counts['cnt'].iloc[0])
    null_tickers = int(counts['tickers'].iloc[0])
    
    print(f"\nRows with NULL financial_date_used: {null_rows:,}")
    print(f"Tickers affected: {null_tickers}")
    
    if null_rows == 0:
        print("\nNothing to backfill!")
        return
    
    if not args.execute:
        print(f"\n[DRY-RUN] Would update {null_rows:,} rows across {null_tickers} tickers.")
        print("Run with --execute to proceed.")
        
        # Show sample
        sample_q = sa_text("""
            SELECT ticker, COUNT(*) AS null_count
            FROM stock_ratio_data
            WHERE financial_date_used IS NULL
            GROUP BY ticker
            ORDER BY null_count DESC
            LIMIT 10
        """)
        sample = pd.read_sql(sample_q, engine)
        print("\nTop 10 tickers by NULL count:")
        for _, row in sample.iterrows():
            print(f"  {row['ticker']:12s}: {row['null_count']:,} rows")
        return
    
    # Execute backfill
    # Strategy: For each ratio row, find the latest financial statement date
    # from stock_income_stmt_data that is <= the ratio's date
    print("\nBackfilling...")
    
    # Get all affected tickers
    tickers_q = sa_text("""
        SELECT DISTINCT ticker FROM stock_ratio_data
        WHERE financial_date_used IS NULL
    """)
    tickers = pd.read_sql(tickers_q, engine)['ticker'].tolist()
    
    total_updated = 0
    for i, ticker in enumerate(tickers, 1):
        if i % 50 == 0 or i == len(tickers):
            print(f"  [{i}/{len(tickers)}] Processing {ticker}...")
        
        try:
            # Get financial statement dates for this ticker
            fin_dates_q = sa_text("""
                SELECT financial_Statement_Date AS fin_date
                FROM stock_income_stmt_data
                WHERE ticker = :ticker
                ORDER BY financial_Statement_Date ASC
            """)
            fin_dates = pd.read_sql(fin_dates_q, engine, params={"ticker": ticker})
            
            if fin_dates.empty:
                continue
            
            fin_dates_list = sorted(pd.to_datetime(fin_dates['fin_date']).dt.date.tolist())
            
            # For each financial date, update ratio rows between this date and the next
            for j, fin_date in enumerate(fin_dates_list):
                if j + 1 < len(fin_dates_list):
                    next_date = fin_dates_list[j + 1]
                    update_q = sa_text("""
                        UPDATE stock_ratio_data
                        SET financial_date_used = :fin_date
                        WHERE ticker = :ticker
                          AND date >= :fin_date
                          AND date < :next_date
                          AND financial_date_used IS NULL
                    """)
                    with engine.begin() as conn:
                        result = conn.execute(update_q, {
                            "fin_date": fin_date,
                            "next_date": next_date,
                            "ticker": ticker
                        })
                        total_updated += result.rowcount
                else:
                    # Last financial date — applies to all remaining ratio rows
                    update_q = sa_text("""
                        UPDATE stock_ratio_data
                        SET financial_date_used = :fin_date
                        WHERE ticker = :ticker
                          AND date >= :fin_date
                          AND financial_date_used IS NULL
                    """)
                    with engine.begin() as conn:
                        result = conn.execute(update_q, {
                            "fin_date": fin_date,
                            "ticker": ticker
                        })
                        total_updated += result.rowcount
        
        except Exception as e:
            print(f"  ⚠ Error processing {ticker}: {e}")
    
    # Check remaining NULLs (ratio rows before the earliest financial statement)
    remaining_q = sa_text("""
        SELECT COUNT(*) AS cnt FROM stock_ratio_data
        WHERE financial_date_used IS NULL
    """)
    remaining = pd.read_sql(remaining_q, engine)
    remaining_count = int(remaining['cnt'].iloc[0])
    
    print(f"\n{'='*60}")
    print(f"BACKFILL COMPLETE")
    print(f"  Updated: {total_updated:,} rows")
    print(f"  Remaining NULL: {remaining_count:,} rows")
    if remaining_count > 0:
        print(f"  (Remaining rows are before the earliest financial statement — expected)")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
