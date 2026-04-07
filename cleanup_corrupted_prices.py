"""
Cleanup corrupted stock price data caused by yfinance thread-safety issues.

Root cause: Parallel yf.download() calls in the stock orchestrator caused
data to be shuffled between tickers (yfinance shares session state across
threads). Prices from one ticker were written to another ticker's rows.

This script:
1. Detects corrupted rows by finding >40% day-over-day close price jumps
2. Deletes the corrupted rows from stock_price_data and stock_ratio_data
3. Deletes predictions, MC results, and hyperparameters for affected tickers
4. Deletes generated graphs and forecast files for affected tickers

After running this, re-run the stock orchestrator to re-fetch clean data,
then retrain models for affected tickers.

Usage:
    python cleanup_corrupted_prices.py              # Dry-run
    python cleanup_corrupted_prices.py --execute    # Actually delete
"""
import argparse
import os
import glob
import pandas as pd
from sqlalchemy import text as sa_text

import fetch_secrets
import db_connectors


def detect_corrupted_rows(engine, date_start='2026-01-01', date_end='2026-04-05', 
                          pct_threshold=0.40):
    """
    Find rows with suspiciously large day-over-day price jumps.
    Uses a LAG window function for efficiency instead of a correlated subquery.
    
    Returns DataFrame with ticker, date, price, prev_price, pct_change.
    """
    q = sa_text("""
        SELECT ticker, date, price, prev_price, 
               ABS((price - prev_price) / prev_price) AS pct_change
        FROM (
            SELECT ticker, date, close_Price AS price,
                   LAG(close_Price) OVER (PARTITION BY ticker ORDER BY date) AS prev_price
            FROM stock_price_data
            WHERE date BETWEEN DATE_SUB(:start, INTERVAL 5 DAY) AND :end
        ) sub
        WHERE prev_price IS NOT NULL
          AND date >= :start
          AND ABS((price - prev_price) / prev_price) > :threshold
        ORDER BY ticker, date
    """)
    return pd.read_sql(q, engine, params={
        'start': date_start, 'end': date_end, 'threshold': pct_threshold
    })


def find_corrupted_date_ranges(anomalies_df):
    """
    For each ticker, find the date range of corruption.
    A corruption "block" is a sequence of anomalous jumps — but we also need
    to include the dates BETWEEN the jump-in and jump-back since those rows
    all have wrong data.
    """
    delete_ranges = {}
    
    for ticker in anomalies_df['ticker'].unique():
        ticker_anomalies = anomalies_df[anomalies_df['ticker'] == ticker].sort_values('date')
        dates = ticker_anomalies['date'].tolist()
        
        if len(dates) >= 2:
            # Delete from first anomaly to last anomaly date
            delete_ranges[ticker] = (dates[0], dates[-1])
        elif len(dates) == 1:
            # Single anomaly — delete just that date
            delete_ranges[ticker] = (dates[0], dates[0])
    
    return delete_ranges


def main():
    parser = argparse.ArgumentParser(description='Cleanup corrupted stock price data')
    parser.add_argument('--execute', action='store_true', help='Actually delete (default: dry-run)')
    parser.add_argument('--start-date', default='2026-01-01', help='Start of scan range')
    parser.add_argument('--end-date', default='2026-04-05', help='End of scan range')
    parser.add_argument('--threshold', type=float, default=0.40, help='Min %% change to flag (default: 0.40)')
    args = parser.parse_args()
    
    db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()
    engine = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    graphs_dir = os.path.join(script_dir, "generated_graphs")
    forecasts_dir = os.path.join(script_dir, "generated_forecasts")
    
    print("=" * 60)
    print("FULL CORRUPTED DATA CLEANUP")
    print("=" * 60)
    
    # Step 1: Detect anomalies
    print(f"\nScanning {args.start_date} to {args.end_date} for >{args.threshold*100:.0f}% jumps...")
    anomalies = detect_corrupted_rows(engine, args.start_date, args.end_date, args.threshold)
    
    if len(anomalies) == 0:
        print("No anomalies found!")
        return
    
    affected_tickers = sorted(anomalies['ticker'].unique())
    print(f"Found {len(anomalies)} anomalous jumps across {len(affected_tickers)} tickers")
    
    # Step 2: Determine date ranges to delete per ticker
    delete_ranges = find_corrupted_date_ranges(anomalies)
    
    # Calculate total rows that will be deleted (price + ratio)
    total_price_rows = 0
    total_ratio_rows = 0
    
    for ticker, (d_start, d_end) in delete_ranges.items():
        count_q = sa_text("""
            SELECT 
                (SELECT COUNT(*) FROM stock_price_data 
                 WHERE ticker = :ticker AND date BETWEEN :start AND :end) AS price_rows,
                (SELECT COUNT(*) FROM stock_ratio_data 
                 WHERE ticker = :ticker AND date BETWEEN :start AND :end) AS ratio_rows
        """)
        counts = pd.read_sql(count_q, engine, params={
            'ticker': ticker, 'start': d_start, 'end': d_end
        })
        total_price_rows += int(counts['price_rows'].iloc[0])
        total_ratio_rows += int(counts['ratio_rows'].iloc[0])
    
    # Step 3: Count prediction/model artifacts
    prediction_count = 0
    mc_count = 0
    hp_count = 0
    for ticker in affected_tickers:
        pred_q = sa_text("SELECT COUNT(*) AS cnt FROM stock_prediction_extended WHERE ticker = :ticker")
        mc_q = sa_text("SELECT COUNT(*) AS cnt FROM monte_carlo_results WHERE ticker = :ticker")
        hp_q = sa_text("SELECT COUNT(*) AS cnt FROM model_hyperparameters WHERE ticker = :ticker")
        prediction_count += int(pd.read_sql(pred_q, engine, params={'ticker': ticker})['cnt'].iloc[0])
        mc_count += int(pd.read_sql(mc_q, engine, params={'ticker': ticker})['cnt'].iloc[0])
        hp_count += int(pd.read_sql(hp_q, engine, params={'ticker': ticker})['cnt'].iloc[0])
    
    # Step 4: Count graph/forecast files to delete
    files_to_delete = []
    for ticker in affected_tickers:
        patterns = [
            os.path.join(graphs_dir, f"future_stock_prediction_of_{ticker}.png"),
            os.path.join(graphs_dir, f"stock_prediction_of_{ticker}.png"),
            os.path.join(graphs_dir, f"Monte_Carlo_Sim_of_{ticker}.png"),
            os.path.join(forecasts_dir, f"forecast_{ticker}.xlsx"),
        ]
        for p in patterns:
            if os.path.exists(p):
                files_to_delete.append(p)
    
    # Print summary
    print(f"\n{'='*60}")
    print("CLEANUP SUMMARY")
    print(f"{'='*60}")
    print(f"  Tickers affected:        {len(affected_tickers)}")
    print(f"  stock_price_data rows:   {total_price_rows:,}")
    print(f"  stock_ratio_data rows:   {total_ratio_rows:,}")
    print(f"  prediction rows:         {prediction_count:,}")
    print(f"  monte_carlo rows:        {mc_count:,}")
    print(f"  hyperparameter rows:     {hp_count:,}")
    print(f"  graph/forecast files:    {len(files_to_delete)}")
    print(f"  TOTAL DB rows to delete: {total_price_rows + total_ratio_rows + prediction_count + mc_count + hp_count:,}")
    
    # Show sample
    print(f"\nSample (first 15 tickers):")
    for i, (ticker, (d_start, d_end)) in enumerate(sorted(delete_ranges.items())):
        if i >= 15:
            print(f"  ... and {len(delete_ranges) - 15} more")
            break
        print(f"  {ticker:15s}: {d_start} to {d_end}")
    
    if not args.execute:
        print(f"\n[DRY-RUN] Would delete {total_price_rows + total_ratio_rows + prediction_count + mc_count + hp_count:,} DB rows + {len(files_to_delete)} files.")
        print("Run with --execute to proceed.")
        return
    
    # ===== EXECUTE MODE =====
    print(f"\nDeleting corrupted price/ratio data...")
    deleted_price = 0
    deleted_ratio = 0
    
    for i, (ticker, (d_start, d_end)) in enumerate(sorted(delete_ranges.items()), 1):
        if i % 50 == 0 or i == len(delete_ranges):
            print(f"  [{i}/{len(delete_ranges)}] {ticker}...")
        
        with engine.begin() as conn:
            result = conn.execute(sa_text(
                "DELETE FROM stock_price_data WHERE ticker = :ticker AND date BETWEEN :start AND :end"
            ), {'ticker': ticker, 'start': d_start, 'end': d_end})
            deleted_price += result.rowcount
        
        with engine.begin() as conn:
            result = conn.execute(sa_text(
                "DELETE FROM stock_ratio_data WHERE ticker = :ticker AND date BETWEEN :start AND :end"
            ), {'ticker': ticker, 'start': d_start, 'end': d_end})
            deleted_ratio += result.rowcount
    
    # Delete predictions, MC results, hyperparameters for affected tickers
    print(f"\nDeleting predictions, MC results, and hyperparameters...")
    deleted_pred = 0
    deleted_mc = 0
    deleted_hp = 0
    
    for i, ticker in enumerate(affected_tickers, 1):
        if i % 50 == 0 or i == len(affected_tickers):
            print(f"  [{i}/{len(affected_tickers)}] {ticker}...")
        
        with engine.begin() as conn:
            r = conn.execute(sa_text(
                "DELETE FROM stock_prediction_extended WHERE ticker = :ticker"
            ), {'ticker': ticker})
            deleted_pred += r.rowcount
        
        with engine.begin() as conn:
            r = conn.execute(sa_text(
                "DELETE FROM monte_carlo_results WHERE ticker = :ticker"
            ), {'ticker': ticker})
            deleted_mc += r.rowcount
        
        with engine.begin() as conn:
            r = conn.execute(sa_text(
                "DELETE FROM model_hyperparameters WHERE ticker = :ticker"
            ), {'ticker': ticker})
            deleted_hp += r.rowcount
    
    # Delete graph/forecast files
    print(f"\nDeleting graph and forecast files...")
    deleted_files = 0
    for f in files_to_delete:
        try:
            os.remove(f)
            deleted_files += 1
        except OSError as e:
            print(f"  [WARN] Could not delete {f}: {e}")
    
    print(f"\n{'='*60}")
    print("CLEANUP COMPLETE")
    print(f"{'='*60}")
    print(f"  Deleted from stock_price_data:       {deleted_price:,}")
    print(f"  Deleted from stock_ratio_data:        {deleted_ratio:,}")
    print(f"  Deleted from stock_prediction_extended: {deleted_pred:,}")
    print(f"  Deleted from monte_carlo_results:     {deleted_mc:,}")
    print(f"  Deleted from model_hyperparameters:   {deleted_hp:,}")
    print(f"  Deleted files:                        {deleted_files}")
    print(f"  Tickers cleaned:                      {len(affected_tickers)}")
    print(f"\nNext steps:")
    print(f"  1. Re-fetch data:    python stock_orchestrator.py")
    print(f"  2. Retrain models:   python price_predictor.py")
    print(f"  3. Rebuild portfolio: python portfolio_builder.py")


if __name__ == '__main__':
    main()
