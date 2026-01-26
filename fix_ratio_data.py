"""
Fix Ratio Data Script

This script recalculates and updates ratio data (P/S, P/E, P/B, P/FCF) for all stocks
in the database that have NULL or missing ratio values.

The issue typically occurs when:
1. Financial data has zero or negative values
2. Ratios were calculated with invalid data
3. TTM calculations weren't available

This script:
1. Identifies tickers with NULL ratio values
2. Recalculates ratios using TTM data when available
3. Falls back to annual data when TTM is not available
4. Updates the database with corrected values

Usage:
    python fix_ratio_data.py [--ticker TICKER] [--force-recalc]

Arguments:
    --ticker TICKER    Only fix data for a specific ticker
    --force-recalc     Force recalculation for all tickers, not just those with NULL values
"""
import argparse
import datetime
import numpy as np
import pandas as pd
import sys

import fetch_secrets
import db_connectors
import db_interactions
from ttm_financial_calculator import calculate_ratios_ttm_with_fallback


def get_tickers_with_null_ratios(db_con) -> list:
    """Get list of tickers that have NULL values in ratio data."""
    query = """
    SELECT DISTINCT ticker 
    FROM stock_ratio_data 
    WHERE p_s IS NULL OR p_e IS NULL OR p_b IS NULL OR p_fcf IS NULL
    """
    df = pd.read_sql(query, db_con)
    return df['ticker'].tolist()


def get_all_ratio_tickers(db_con) -> list:
    """Get all tickers that have ratio data."""
    query = "SELECT DISTINCT ticker FROM stock_ratio_data"
    df = pd.read_sql(query, db_con)
    return df['ticker'].tolist()


def recalculate_ratios_for_ticker(ticker: str, db_con) -> pd.DataFrame:
    """
    Recalculate ratio data for a specific ticker.
    
    Uses TTM data when available, falls back to annual data.
    
    Args:
        ticker: Stock ticker symbol
        db_con: Database connection
        
    Returns:
        DataFrame with recalculated ratio data
    """
    print(f"\n  Recalculating ratios for {ticker}...")
    
    # Get financial data
    try:
        financial_df = db_interactions.import_stock_financial_data(stock_ticker=ticker)
        if financial_df is None or financial_df.empty:
            print(f"    ↳ No financial data available")
            return pd.DataFrame()
    except Exception as e:
        print(f"    ↳ Error getting financial data: {e}")
        return pd.DataFrame()
    
    # Get price data
    try:
        # Get the date range from financial data
        oldest_financial_date = financial_df['date'].min()
        
        query = f"""
        SELECT date, ticker, close_Price 
        FROM stock_price_data 
        WHERE ticker = '{ticker}' AND date >= '{oldest_financial_date}'
        ORDER BY date ASC
        """
        price_df = pd.read_sql(query, db_con)
        
        if price_df.empty:
            print(f"    ↳ No price data available")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"    ↳ Error getting price data: {e}")
        return pd.DataFrame()
    
    # Combine price and financial data
    try:
        from stock_data_fetch import combine_stock_data
        combined_df = combine_stock_data(price_df, financial_df)
    except Exception as e:
        print(f"    ↳ Error combining data: {e}")
        return pd.DataFrame()
    
    # Calculate ratios using TTM with fallback
    try:
        ratio_df = calculate_ratios_ttm_with_fallback(
            combined_df,
            symbol=ticker,
            prefer_ttm=True
        )
        
        # Extract ratio columns
        ratio_cols = ['date', 'ticker', 'P/S', 'P/E', 'P/B', 'P/FCF']
        
        available_cols = [col for col in ratio_cols if col in ratio_df.columns]
        ratio_df = ratio_df[available_cols].copy()
        
        # Rename columns to match database schema
        ratio_df = ratio_df.rename(columns={
            'P/S': 'p_s',
            'P/E': 'p_e',
            'P/B': 'p_b',
            'P/FCF': 'p_fcf'
        })
        
        # Replace inf values with NaN
        ratio_df = ratio_df.replace([np.inf, -np.inf], np.nan)
        
        # Drop rows where all ratio columns are NaN
        ratio_cols_db = ['p_s', 'p_e', 'p_b', 'p_fcf']
        ratio_df = ratio_df.dropna(subset=ratio_cols_db, how='all')
        
        print(f"    ✓ Calculated {len(ratio_df)} ratio records")
        print(f"    ↳ Non-null P/S: {ratio_df['p_s'].notna().sum()}, P/E: {ratio_df['p_e'].notna().sum()}, P/B: {ratio_df['p_b'].notna().sum()}, P/FCF: {ratio_df['p_fcf'].notna().sum()}")
        
        return ratio_df
        
    except Exception as e:
        print(f"    ↳ Error calculating ratios: {e}")
        return pd.DataFrame()


def update_ratio_data(ticker: str, ratio_df: pd.DataFrame, db_con):
    """
    Update ratio data in database for a specific ticker.
    
    Uses delete-then-insert pattern to avoid duplicates.
    """
    if ratio_df.empty:
        return False
    
    try:
        from sqlalchemy import text
        
        # Delete existing ratio data for this ticker
        with db_con.begin() as connection:
            connection.execute(text("""
                DELETE FROM stock_ratio_data WHERE ticker = :ticker
            """), {'ticker': ticker})
        
        # Insert new ratio data
        ratio_df.to_sql(
            name='stock_ratio_data',
            con=db_con,
            index=False,
            if_exists='append'
        )
        
        print(f"    ✓ Updated {len(ratio_df)} ratio records in database")
        return True
        
    except Exception as e:
        print(f"    ✗ Error updating database: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Fix NULL ratio data in the database')
    parser.add_argument('--ticker', type=str, help='Only fix data for a specific ticker')
    parser.add_argument('--force-recalc', action='store_true', 
                        help='Force recalculation for all tickers')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done without making changes')
    args = parser.parse_args()
    
    # Connect to database
    try:
        db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()
        db_con = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)
    except Exception as e:
        print(f"Failed to connect to database: {e}")
        sys.exit(1)
    
    # Get list of tickers to process
    if args.ticker:
        tickers = [args.ticker]
    elif args.force_recalc:
        tickers = get_all_ratio_tickers(db_con)
    else:
        tickers = get_tickers_with_null_ratios(db_con)
    
    if not tickers:
        print("No tickers found with NULL ratio values.")
        sys.exit(0)
    
    print(f"\n{'='*60}")
    print(f"Fixing Ratio Data for {len(tickers)} tickers")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE UPDATE'}")
    print(f"{'='*60}")
    
    success_count = 0
    error_count = 0
    
    for i, ticker in enumerate(tickers, 1):
        print(f"\n[{i}/{len(tickers)}] Processing {ticker}")
        
        # Recalculate ratios
        ratio_df = recalculate_ratios_for_ticker(ticker, db_con)
        
        if ratio_df.empty:
            error_count += 1
            continue
        
        # Update database
        if not args.dry_run:
            if update_ratio_data(ticker, ratio_df, db_con):
                success_count += 1
            else:
                error_count += 1
        else:
            print(f"    [DRY RUN] Would update {len(ratio_df)} records")
            success_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"  Total tickers processed: {len(tickers)}")
    print(f"  Successful updates: {success_count}")
    print(f"  Errors: {error_count}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
