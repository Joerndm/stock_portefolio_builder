"""
Backfill Ratios for Tickers with Previously Extreme Values

After the epsilon guard fix, tickers that had extreme ratio values (now NULLed)
need their ratios recalculated with the corrected _safe_divide().

This script:
1. Finds tickers with a high percentage of NULL ratio values
2. Deletes their existing ratio data
3. Re-runs ratio calculation through the orchestrator

Usage:
    python backfill_ratios.py              # Dry-run
    python backfill_ratios.py --execute    # Actually recalculate
    python backfill_ratios.py --execute --max 30  # Limit batch size
"""
import argparse
import pandas as pd
from sqlalchemy import text as sa_text

import fetch_secrets
import db_connectors


def find_tickers_needing_backfill(engine):
    """
    Find tickers that need ratio recalculation.
    
    Criteria:
    - Tickers where >30% of any ratio column is NULL (likely had values
      NULLed by the extreme value cleanup or have gaps from old calculations)
    - Only tickers that have financial data (no point recalculating if no financials)
    """
    query = sa_text("""
        SELECT r.ticker,
               COUNT(*) AS total_rows,
               ROUND(SUM(CASE WHEN r.p_e IS NULL THEN 1 ELSE 0 END) / COUNT(*) * 100) AS pct_null_pe,
               ROUND(SUM(CASE WHEN r.p_b IS NULL THEN 1 ELSE 0 END) / COUNT(*) * 100) AS pct_null_pb,
               ROUND(SUM(CASE WHEN r.p_fcf IS NULL THEN 1 ELSE 0 END) / COUNT(*) * 100) AS pct_null_pfcf,
               ROUND(SUM(CASE WHEN r.financial_date_used IS NULL THEN 1 ELSE 0 END) / COUNT(*) * 100) AS pct_null_fdu
        FROM stock_ratio_data r
        INNER JOIN stock_income_stmt_data i ON r.ticker = i.ticker
        GROUP BY r.ticker
        HAVING pct_null_pe > 30 OR pct_null_pb > 30 OR pct_null_pfcf > 30 OR pct_null_fdu > 30
        ORDER BY pct_null_pe DESC
    """)
    return pd.read_sql(query, engine)


def delete_ratio_data(engine, ticker):
    """Delete all ratio data for a ticker so it can be recalculated."""
    with engine.begin() as conn:
        result = conn.execute(
            sa_text("DELETE FROM stock_ratio_data WHERE ticker = :ticker"),
            {"ticker": ticker}
        )
        return result.rowcount


def main():
    parser = argparse.ArgumentParser(description='Backfill ratios for tickers with NULL values')
    parser.add_argument('--execute', action='store_true', help='Actually recalculate')
    parser.add_argument('--max', type=int, default=0, help='Max tickers (0 = all)')
    args = parser.parse_args()
    
    db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()
    engine = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)
    
    print("=" * 60)
    print("RATIO BACKFILL")
    print("=" * 60)
    
    affected = find_tickers_needing_backfill(engine)
    print(f"\nFound {len(affected)} tickers needing ratio backfill:")
    
    for _, row in affected.head(20).iterrows():
        print(f"  {row['ticker']:12s} | {row['total_rows']:5d} rows | "
              f"PE:{row['pct_null_pe']:3.0f}% PB:{row['pct_null_pb']:3.0f}% "
              f"PFCF:{row['pct_null_pfcf']:3.0f}% FDU:{row['pct_null_fdu']:3.0f}% NULL")
    if len(affected) > 20:
        print(f"  ... and {len(affected) - 20} more")
    
    tickers = affected['ticker'].tolist()
    if args.max > 0:
        tickers = tickers[:args.max]
    
    if not args.execute:
        print(f"\n[DRY-RUN] Would recalculate ratios for {len(tickers)} tickers.")
        print("Run with --execute to proceed.")
        return
    
    from stock_orchestrator import StockDataOrchestrator
    orchestrator = StockDataOrchestrator()
    
    success = 0
    failed = 0
    for i, ticker in enumerate(tickers, 1):
        print(f"\n[{i}/{len(tickers)}] {ticker}")
        try:
            deleted = delete_ratio_data(engine, ticker)
            print(f"  Deleted {deleted} old ratio rows")
            
            ok, _ = orchestrator.process_ratio_data(ticker, prefer_ttm=True)
            if ok:
                success += 1
                print(f"  ✓ Ratios recalculated")
            else:
                failed += 1
                print(f"  ✗ Ratio calculation returned no data")
        except Exception as e:
            failed += 1
            print(f"  ✗ Error: {e}")
    
    print(f"\n{'='*60}")
    print(f"BACKFILL COMPLETE: {success} success, {failed} failed")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
