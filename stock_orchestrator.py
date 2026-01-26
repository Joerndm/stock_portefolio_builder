"""
Stock Data Orchestrator Module

This is the main orchestration module for the stock data pipeline. It coordinates
all data fetching, processing, and database operations using a modular architecture.

Features:
    - Dynamic symbol fetching from multiple indices (C25, S&P 500, DAX40, etc.)
    - Comprehensive error handling with blacklist management
    - TTM (Trailing Twelve Months) financial data with annual fallback
    - Technical indicators calculation
    - Parallel processing support (optional)
    - Incremental updates for existing data
    - Database upsert operations to handle duplicates

Usage:
    python stock_orchestrator.py                   # Run full pipeline
    python stock_orchestrator.py --indices C25     # Specific index
    python stock_orchestrator.py --ticker AAPL     # Single ticker
    python stock_orchestrator.py --update-only     # Only update existing

Author: Stock Portfolio Builder
Last Modified: 2026
"""
import os
import sys
import time
import datetime
import argparse
import traceback
from typing import Dict, List, Optional, Tuple, Set
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
import yfinance as yf

# Local imports
from dynamic_index_fetcher import (
    dynamic_fetch_index_data,
    DynamicIndexFetcher,
    INDEX_CONFIGS,
    MARKET_INDICES
)
from blacklist_manager import (
    BlacklistManager,
    get_blacklist_manager,
    is_blacklisted,
    blacklist_ticker,
    filter_blacklisted
)
from enhanced_financial_fetcher import (
    QuarterlyFinancialFetcher,
    fetch_quarterly_financial_data
)
from ttm_financial_calculator import (
    TTMFinancialCalculator,
    calculate_ratios_ttm_with_fallback,
    get_financial_data_with_ttm_preference
)
import fetch_secrets
import db_connectors
import db_interactions


class StockDataOrchestrator:
    """
    Main orchestrator for the stock data pipeline.
    
    Handles the complete workflow of:
    1. Fetching stock symbols from indices
    2. Filtering blacklisted tickers
    3. Fetching and processing price data
    4. Calculating technical indicators
    5. Fetching financial data (quarterly/TTM preferred)
    6. Calculating financial ratios
    7. Exporting to database
    """
    
    # Default indices to fetch
    DEFAULT_INDICES = ['C25', 'SP500', 'DAX40', 'CAC40', 'FTSE100', 'AEX25', 'SMI', 'FTSEMIB', 'IBEX35', 'BEL20', 'ATX']

    
    # Error tracking for auto-blacklisting
    MAX_CONSECUTIVE_ERRORS = 3
    
    def __init__(self, indices: List[str] = None, include_market_indices: bool = True):
        """
        Initialize the StockDataOrchestrator.
        
        Args:
            indices: List of index codes to fetch (e.g., ['C25', 'SP500'])
            include_market_indices: Whether to include market indices like ^VIX
        """
        self.indices = indices or self.DEFAULT_INDICES
        self.include_market_indices = include_market_indices
        
        # Initialize components
        self.blacklist = get_blacklist_manager()
        self.quarterly_fetcher = QuarterlyFinancialFetcher()
        self.ttm_calculator = TTMFinancialCalculator()
        self.index_fetcher = DynamicIndexFetcher()
        
        # Database connection
        self.db_con = None
        self._connect_database()
        
        # Tracking
        self.processed_tickers = set()
        self.error_counts = {}
        self.processing_stats = {
            'total': 0,
            'success': 0,
            'skipped': 0,
            'errors': 0,
            'blacklisted': 0
        }
    
    def _connect_database(self):
        """Establish database connection."""
        try:
            db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()
            self.db_con = db_connectors.pandas_mysql_connector(
                db_host, db_user, db_pass, db_name
            )
            print("✓ Database connection established")
        except Exception as e:
            print(f"⚠️  Database connection failed: {e}")
            self.db_con = None
    
    def get_symbols(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch stock symbols from configured indices.
        
        Args:
            use_cache: Whether to use cached index data
            
        Returns:
            DataFrame with Symbol, Index, Exchange columns
        """
        print(f"\n{'='*60}")
        print(f"Fetching symbols from indices: {', '.join(self.indices)}")
        print(f"{'='*60}")
        
        symbols_df = dynamic_fetch_index_data(
            indices=self.indices,
            include_market_indices=self.include_market_indices,
            use_cache=use_cache
        )
        
        # Filter blacklisted tickers
        valid_symbols, blacklisted = self.blacklist.filter_tickers(
            symbols_df['Symbol'].tolist()
        )
        
        symbols_df = symbols_df[symbols_df['Symbol'].isin(valid_symbols)]
        self.processing_stats['blacklisted'] = len(blacklisted)
        
        print(f"\n✓ Total symbols after filtering: {len(symbols_df)}")
        if blacklisted:
            print(f"  ↳ Filtered {len(blacklisted)} blacklisted tickers")
        
        return symbols_df
    
    def _handle_ticker_error(self, ticker: str, error: Exception, 
                              error_type: str = "error") -> bool:
        """
        Handle errors for a ticker, potentially blacklisting if too many errors.
        
        Args:
            ticker: Stock ticker symbol
            error: The exception that occurred
            error_type: Type of error for categorization
            
        Returns:
            True if ticker should be skipped (blacklisted), False to retry
        """
        error_msg = str(error)
        
        # Track consecutive errors
        self.error_counts[ticker] = self.error_counts.get(ticker, 0) + 1
        
        # Check for specific error types that warrant immediate blacklisting
        immediate_blacklist_patterns = [
            "invalid or not found",
            "no data found",
            "symbol may be delisted",
            "404",
            "not available",
            "KeyError",
        ]
        
        should_blacklist = False
        blacklist_reason = 'error'
        
        # Check for immediate blacklist patterns
        for pattern in immediate_blacklist_patterns:
            if pattern.lower() in error_msg.lower():
                should_blacklist = True
                if "delisted" in error_msg.lower() or "not found" in error_msg.lower():
                    blacklist_reason = 'delisted'
                elif "no data" in error_msg.lower():
                    blacklist_reason = 'no_data'
                else:
                    blacklist_reason = 'invalid'
                break
        
        # Also blacklist if too many consecutive errors
        if self.error_counts[ticker] >= self.MAX_CONSECUTIVE_ERRORS:
            should_blacklist = True
            blacklist_reason = 'error'
        
        if should_blacklist:
            self.blacklist.add_ticker(
                ticker, 
                reason=blacklist_reason,
                details=error_msg[:200]
            )
            self.processing_stats['blacklisted'] += 1
            print(f"   ↳ Added {ticker} to blacklist (reason: {blacklist_reason})")
            return True
        
        return False
    
    def _is_index_ticker(self, ticker: str, stock_info: dict = None) -> bool:
        """
        Check if a ticker is a market index.
        
        Args:
            ticker: Stock ticker symbol
            stock_info: Optional yfinance info dict
            
        Returns:
            True if ticker is an index
        """
        # Quick check by prefix
        if ticker.startswith('^'):
            return True
        
        # Check against known market indices
        if ticker in MARKET_INDICES.values():
            return True
        
        # Check via stock info
        if stock_info and stock_info.get('typeDisp') == 'Index':
            return True
        
        return False
    
    def process_stock_info(self, ticker: str) -> Tuple[bool, Optional[pd.DataFrame]]:
        """
        Process and export basic stock information.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Tuple of (success, stock_info_df or None)
        """
        try:
            if db_interactions.does_stock_exists_stock_info_data(ticker):
                return True, None
            
            print(f"   → Fetching company info for {ticker}")
            
            # Use stock_data_fetch function
            from stock_data_fetch import fetch_stock_standard_data
            stock_info_df = fetch_stock_standard_data(ticker)
            
            db_interactions.export_stock_info_data(stock_info_df)
            return True, stock_info_df
            
        except Exception as e:
            print(f"   ✗ Error fetching stock info: {e}")
            if self._handle_ticker_error(ticker, e, "stock_info"):
                return False, None
            return False, None
    
    def process_price_data(self, ticker: str, stock_info: dict = None,
                           force_full: bool = False) -> Tuple[bool, Optional[pd.DataFrame]]:
        """
        Fetch and process price data with technical indicators.
        
        Args:
            ticker: Stock ticker symbol
            stock_info: yfinance info dict (to check if index)
            force_full: Force full historical data fetch
            
        Returns:
            Tuple of (success, price_data_df or None)
        """
        from stock_data_fetch import (
            fetch_stock_price_data,
            calculate_period_returns,
            add_technical_indicators,
            add_volume_indicators,
            add_volatility_indicators,
            calculate_moving_averages,
            calculate_standard_diviation_value,
            calculate_bollinger_bands,
            calculate_momentum
        )
        
        is_index = self._is_index_ticker(ticker, stock_info)
        
        try:
            # Check if we need full fetch or just update
            if not force_full and db_interactions.does_stock_exists_stock_price_data(ticker):
                return self._update_price_data(ticker, stock_info, is_index)
            
            print(f"   → Fetching full price history for {ticker}")
            
            # Fetch full historical data
            stock_price_data_df = fetch_stock_price_data(ticker)
            
            if stock_price_data_df.empty:
                print(f"   ✗ No price data available for {ticker}")
                return False, None
            
            # Calculate all indicators
            stock_price_data_df = calculate_period_returns(stock_price_data_df)
            stock_price_data_df = add_technical_indicators(stock_price_data_df)
            stock_price_data_df = add_volume_indicators(stock_price_data_df)
            
            # Skip volatility for indices
            if not is_index:
                stock_price_data_df = add_volatility_indicators(stock_price_data_df)
            
            stock_price_data_df = calculate_moving_averages(stock_price_data_df)
            stock_price_data_df = calculate_standard_diviation_value(stock_price_data_df)
            stock_price_data_df = calculate_bollinger_bands(stock_price_data_df)
            stock_price_data_df = calculate_momentum(stock_price_data_df)
            
            # Drop rows with NaN in critical columns
            critical_cols = ['date', 'ticker', 'close_Price', 'open_Price', 'high_Price', 'low_Price']
            stock_price_data_df = stock_price_data_df.dropna(subset=critical_cols)
            
            if stock_price_data_df.empty:
                print(f"   ✗ DataFrame empty after processing for {ticker}")
                return False, None
            
            # Export to database
            db_interactions.export_stock_price_data(stock_price_data_df)
            print(f"   ✓ Exported {len(stock_price_data_df)} price records")
            
            return True, stock_price_data_df
            
        except Exception as e:
            print(f"   ✗ Error processing price data: {e}")
            traceback.print_exc()
            if self._handle_ticker_error(ticker, e, "price_data"):
                return False, None
            return False, None
    
    def _update_price_data(self, ticker: str, stock_info: dict,
                           is_index: bool) -> Tuple[bool, Optional[pd.DataFrame]]:
        """
        Update existing price data with new records.
        
        Args:
            ticker: Stock ticker symbol
            stock_info: yfinance info dict
            is_index: Whether this is an index ticker
            
        Returns:
            Tuple of (success, updated_price_df or None)
        """
        from stock_data_fetch import (
            fetch_stock_price_data,
            calculate_period_returns,
            add_technical_indicators,
            add_volume_indicators,
            add_volatility_indicators,
            calculate_moving_averages,
            calculate_standard_diviation_value,
            calculate_bollinger_bands,
            calculate_momentum
        )
        from market_hours_utils import should_fetch_new_data
        
        try:
            # Get the last date in database
            stock_price_data_df = db_interactions.import_stock_price_data(
                amount=1, stock_ticker=ticker
            )
            last_db_date = stock_price_data_df.iloc[0]["date"]
            
            # Convert to date object
            if hasattr(last_db_date, 'date'):
                last_db_date = last_db_date.date()
            elif isinstance(last_db_date, str):
                last_db_date = datetime.datetime.strptime(last_db_date, "%Y-%m-%d").date()
            
            # Check if update needed
            should_fetch, new_date, reason = should_fetch_new_data(last_db_date, ticker)
            
            if not should_fetch:
                print(f"   ↳ {reason}")
                return True, None
            
            print(f"   → Updating price data from {new_date}")
            
            # Get enough historical data for indicator calculations
            stock_price_data_df = db_interactions.import_stock_price_data(
                amount=252 * 5 + 1, stock_ticker=ticker
            )
            stock_price_data_df["date"] = pd.to_datetime(stock_price_data_df["date"])
            
            # Fetch new data
            new_stock_price_data_df = fetch_stock_price_data(ticker, new_date)
            
            if new_stock_price_data_df.empty:
                print(f"   ↳ No new data available")
                return True, None
            
            # Combine old and new data, keeping only NEW rows for calculation context
            # First, remove any columns from db_data that are indicator columns
            indicator_cols = [
                'sma_5', 'sma_20', 'sma_40', 'sma_120', 'sma_200',
                'ema_5', 'ema_20', 'ema_40', 'ema_120', 'ema_200',
                'std_Div_5', 'std_Div_20', 'std_Div_40', 'std_Div_120', 'std_Div_200',
                'bollinger_Band_5_2STD', 'bollinger_Band_20_2STD', 'bollinger_Band_40_2STD',
                'bollinger_Band_120_2STD', 'bollinger_Band_200_2STD',
                'momentum', 'rsi_14', 'atr_14', 'macd', 'macd_signal', 'macd_histogram',
                'volume_sma_20', 'volume_ema_20', 'volume_ratio', 'vwap', 'obv',
                'volatility_5d', 'volatility_20d', 'volatility_60d',
                '1D', '1M', '3M', '6M', '9M', '1Y', '2Y', '3Y', '4Y', '5Y'
            ]
            
            # Drop indicator columns from DB data to force recalculation
            cols_to_drop = [c for c in indicator_cols if c in stock_price_data_df.columns]
            if cols_to_drop:
                stock_price_data_df = stock_price_data_df.drop(columns=cols_to_drop)
            
            # Concatenate, ensuring no duplicate columns
            combined_df = pd.concat(
                [stock_price_data_df, new_stock_price_data_df],
                axis=0,
                ignore_index=True
            )
            
            # Remove duplicate rows by date
            combined_df = combined_df.drop_duplicates(subset=['date'], keep='last')
            combined_df = combined_df.sort_values('date').reset_index(drop=True)
            
            # Recalculate all indicators
            combined_df = calculate_period_returns(combined_df)
            combined_df = add_technical_indicators(combined_df)
            combined_df = add_volume_indicators(combined_df)
            
            if not is_index:
                combined_df = add_volatility_indicators(combined_df)
            
            combined_df = calculate_moving_averages(combined_df)
            combined_df = calculate_standard_diviation_value(combined_df)
            combined_df = calculate_bollinger_bands(combined_df)
            combined_df = calculate_momentum(combined_df)
            
            # Keep only new rows for export
            combined_df = combined_df.loc[
                combined_df["date"] >= new_stock_price_data_df.loc[0, "date"]
            ]
            
            # Drop NaN in critical columns
            critical_cols = ['date', 'ticker', 'close_Price', 'open_Price', 'high_Price', 'low_Price']
            combined_df = combined_df.dropna(subset=critical_cols)
            
            if combined_df.empty:
                print(f"   ↳ No valid new data to export")
                return True, None
            
            # Export new data
            db_interactions.export_stock_price_data(combined_df)
            print(f"   ✓ Exported {len(combined_df)} new price records")
            
            return True, combined_df
            
        except Exception as e:
            print(f"   ✗ Error updating price data: {e}")
            traceback.print_exc()
            return False, None
    
    def process_financial_data(self, ticker: str, prefer_ttm: bool = True
                               ) -> Tuple[bool, Optional[pd.DataFrame]]:
        """
        Process and export financial statement data.
        
        Uses quarterly data with TTM calculations when available,
        falls back to annual data otherwise.
        
        Args:
            ticker: Stock ticker symbol
            prefer_ttm: Whether to prefer TTM over annual data
            
        Returns:
            Tuple of (success, financial_data_df or None)
        """
        from stock_data_fetch import fetch_stock_financial_data
        import numpy as np
        
        try:
            # Also fetch and export quarterly data when prefer_ttm is True
            if prefer_ttm:
                self._fetch_and_export_quarterly_data(ticker)
            
            # Check if financial data exists
            if db_interactions.does_stock_exists_stock_income_stmt_data(ticker):
                # Check if we need to update
                full_stock_financial_data_df = db_interactions.import_stock_financial_data(
                    stock_ticker=ticker
                )
                db_date = full_stock_financial_data_df.iloc[0]["date"]
                
                # Fetch latest to compare
                try:
                    new_financial_data_df = fetch_stock_financial_data(ticker)
                    if new_financial_data_df.empty:
                        print(f"   ↳ No new financial data available")
                        return True, full_stock_financial_data_df
                    
                    # Replace infinity values with NaN
                    new_financial_data_df = new_financial_data_df.replace([np.inf, -np.inf], np.nan)
                    
                    source_date = new_financial_data_df["date"].dt.date.iloc[-1]
                    
                    if db_date == source_date:
                        print(f"   ↳ Financial data is up to date")
                        return True, full_stock_financial_data_df
                    
                    # Filter to only new data
                    new_financial_data_df = new_financial_data_df.loc[
                        new_financial_data_df["date"].dt.date > db_date
                    ]
                    
                    if new_financial_data_df.empty:
                        print(f"   ↳ No new financial statements to export")
                        return True, full_stock_financial_data_df
                    
                    # Export new data
                    db_interactions.export_stock_financial_data(new_financial_data_df)
                    print(f"   ✓ Exported {len(new_financial_data_df)} new financial records")
                    return True, new_financial_data_df
                    
                except Exception as e:
                    print(f"   ↳ Could not update financial data: {e}")
                    return True, full_stock_financial_data_df
            
            # Fetch new financial data
            print(f"   → Fetching financial data for {ticker}")
            full_stock_financial_data_df = fetch_stock_financial_data(ticker)
            
            if full_stock_financial_data_df.empty:
                print(f"   ↳ No financial data available")
                return False, None
            
            # Replace infinity values with NaN
            full_stock_financial_data_df = full_stock_financial_data_df.replace([np.inf, -np.inf], np.nan)
            
            # Export to database
            db_interactions.export_stock_financial_data(full_stock_financial_data_df)
            print(f"   ✓ Exported {len(full_stock_financial_data_df)} financial records")
            
            return True, full_stock_financial_data_df
            
        except Exception as e:
            print(f"   ✗ Error processing financial data: {e}")
            # Don't blacklist for financial data errors - company might just not have reports
            return False, None
    
    def _fetch_and_export_quarterly_data(self, ticker: str) -> bool:
        """
        Fetch and export quarterly financial data for TTM calculations.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            True if quarterly data was successfully exported
        """
        try:
            print(f"   → Fetching quarterly data for {ticker}")
            
            # Fetch quarterly income statement
            quarterly_income = self.quarterly_fetcher.fetch_quarterly_income_statement(ticker)
            if not quarterly_income.empty:
                # Prepare for database export
                quarterly_income['ticker'] = ticker
                quarterly_income = quarterly_income.reset_index()
                quarterly_income = quarterly_income.rename(columns={'index': 'fiscal_quarter_end'})
                
                # Ensure fiscal_quarter_end is datetime
                quarterly_income['fiscal_quarter_end'] = pd.to_datetime(quarterly_income['fiscal_quarter_end'])
                
                try:
                    db_interactions.export_quarterly_income_stmt(quarterly_income)
                except Exception as e:
                    print(f"   ↳ Could not export quarterly income: {e}")
            
            # Fetch quarterly balance sheet
            quarterly_bs = self.quarterly_fetcher.fetch_quarterly_balance_sheet(ticker)
            if not quarterly_bs.empty:
                quarterly_bs['ticker'] = ticker
                quarterly_bs = quarterly_bs.reset_index()
                quarterly_bs = quarterly_bs.rename(columns={'index': 'fiscal_quarter_end'})
                quarterly_bs['fiscal_quarter_end'] = pd.to_datetime(quarterly_bs['fiscal_quarter_end'])
                
                try:
                    db_interactions.export_quarterly_balancesheet(quarterly_bs)
                except Exception as e:
                    print(f"   ↳ Could not export quarterly balance sheet: {e}")
            
            # Fetch quarterly cash flow
            quarterly_cf = self.quarterly_fetcher.fetch_quarterly_cash_flow(ticker)
            if not quarterly_cf.empty:
                quarterly_cf['ticker'] = ticker
                quarterly_cf = quarterly_cf.reset_index()
                quarterly_cf = quarterly_cf.rename(columns={'index': 'fiscal_quarter_end'})
                quarterly_cf['fiscal_quarter_end'] = pd.to_datetime(quarterly_cf['fiscal_quarter_end'])
                
                try:
                    db_interactions.export_quarterly_cashflow(quarterly_cf)
                except Exception as e:
                    print(f"   ↳ Could not export quarterly cash flow: {e}")
            
            return True
            
        except Exception as e:
            print(f"   ↳ Could not fetch quarterly data: {e}")
            return False
    
    def process_ratio_data(self, ticker: str, prefer_ttm: bool = True
                           ) -> Tuple[bool, Optional[pd.DataFrame]]:
        """
        Calculate and export financial ratios.
        
        Uses TTM data when available for more current metrics.
        
        Args:
            ticker: Stock ticker symbol
            prefer_ttm: Whether to prefer TTM calculations
            
        Returns:
            Tuple of (success, ratio_data_df or None)
        """
        from stock_data_fetch import combine_stock_data, calculate_ratios, drop_nan_values
        
        try:
            # Check current status
            ratio_exists = db_interactions.does_stock_exists_stock_ratio_data(ticker)
            
            if ratio_exists:
                # Check if update needed
                stock_ratio_data_df = db_interactions.import_stock_ratio_data(
                    stock_ticker=ticker
                )
                last_date = stock_ratio_data_df.iloc[0]["date"]
                
                if str(last_date) == datetime.datetime.now().strftime("%Y-%m-%d"):
                    print(f"   ↳ Ratio data is up to date")
                    return True, stock_ratio_data_df
                
                # Get financial data
                full_stock_financial_data_df = db_interactions.import_stock_financial_data(
                    amount=1, stock_ticker=ticker
                )
            else:
                # Get all financial data
                TABEL_NAME = "stock_income_stmt_data"
                query = f"""SELECT COUNT(financial_Statement_Date)
                            FROM {TABEL_NAME}
                            WHERE ticker = '{ticker}'"""
                entry_amount = pd.read_sql(sql=query, con=self.db_con)
                full_stock_financial_data_df = db_interactions.import_stock_financial_data(
                    amount=entry_amount.loc[0, entry_amount.columns[0]],
                    stock_ticker=ticker
                )
            
            if full_stock_financial_data_df is None or full_stock_financial_data_df.empty:
                print(f"   ↳ No financial data for ratio calculation")
                return False, None
            
            full_stock_financial_data_df = full_stock_financial_data_df.dropna(axis=1)
            
            if full_stock_financial_data_df.empty:
                print(f"   ↳ Financial data empty after dropna")
                return False, None
            
            # Get the date range for price data
            if ratio_exists:
                date = last_date
            else:
                date = full_stock_financial_data_df.iloc[0]["date"]
            
            # Get price data
            TABEL_NAME = "stock_price_data"
            query = f"""SELECT *
                        FROM {TABEL_NAME}
                        WHERE ticker = '{ticker}' AND date >= '{date}'"""
            stock_price_data_df = pd.read_sql(sql=query, con=self.db_con)
            
            if stock_price_data_df.empty:
                print(f"   ↳ No price data for ratio calculation")
                return False, None
            
            # Combine and calculate ratios
            combined_stock_data_df = combine_stock_data(
                stock_price_data_df, full_stock_financial_data_df
            )
            combined_stock_data_df = calculate_ratios(
                combined_stock_data_df,
                stock_symbol=ticker,
                prefer_ttm=prefer_ttm
            )
            
            # Extract ratio columns
            stock_ratio_data_df = combined_stock_data_df[
                ['date', 'ticker', 'P/S', 'P/E', 'P/B', 'P/FCF']
            ].copy()
            stock_ratio_data_df = stock_ratio_data_df.rename(columns={
                "P/S": "p_s", "P/E": "p_e", "P/B": "p_b", "P/FCF": "p_fcf"
            })
            stock_ratio_data_df = drop_nan_values(stock_ratio_data_df)
            
            if stock_ratio_data_df.empty:
                print(f"   ↳ No valid ratio data to export")
                return False, None
            
            # Export to database
            db_interactions.export_stock_ratio_data(stock_ratio_data_df)
            print(f"   ✓ Exported {len(stock_ratio_data_df)} ratio records")
            
            return True, stock_ratio_data_df
            
        except Exception as e:
            print(f"   ✗ Error processing ratio data: {e}")
            traceback.print_exc()
            return False, None
    
    def process_ticker(self, ticker: str, force_full: bool = False,
                       prefer_ttm: bool = True) -> bool:
        """
        Process a single ticker through the complete pipeline.
        
        Args:
            ticker: Stock ticker symbol
            force_full: Force full data fetch (not incremental)
            prefer_ttm: Prefer TTM over annual financial data
            
        Returns:
            True if processing succeeded, False otherwise
        """
        print(f"\n{'─'*50}")
        print(f"Processing: {ticker}")
        print(f"{'─'*50}")
        
        self.processing_stats['total'] += 1
        
        # Check blacklist
        if self.blacklist.is_blacklisted(ticker):
            print(f"   ↳ Skipped (blacklisted)")
            self.processing_stats['skipped'] += 1
            return False
        
        # Get stock info for type checking
        try:
            stock_info = yf.Ticker(ticker).info
            if not stock_info or stock_info.get('regularMarketPrice') is None:
                raise ValueError(f"No data available for {ticker}")
        except Exception as e:
            print(f"   ✗ Cannot access {ticker}: {e}")
            self._handle_ticker_error(ticker, e, "info")
            self.processing_stats['errors'] += 1
            return False
        
        is_index = self._is_index_ticker(ticker, stock_info)
        
        # Step 1: Stock Info
        success, _ = self.process_stock_info(ticker)
        if not success:
            self.processing_stats['errors'] += 1
            return False
        
        # Step 2: Price Data
        success, _ = self.process_price_data(ticker, stock_info, force_full)
        if not success:
            self.processing_stats['errors'] += 1
            return False
        
        # Step 3 & 4: Financial Data and Ratios (skip for indices)
        if not is_index:
            # Financial Data
            success, _ = self.process_financial_data(ticker, prefer_ttm)
            if success:
                # Ratio Data
                self.process_ratio_data(ticker, prefer_ttm)
        else:
            print(f"   ↳ Skipping financial data for index {ticker}")
        
        self.processed_tickers.add(ticker)
        self.processing_stats['success'] += 1
        return True
    
    def run(self, tickers: List[str] = None, force_full: bool = False,
            prefer_ttm: bool = True) -> Dict:
        """
        Run the complete data pipeline.
        
        Args:
            tickers: Optional list of specific tickers (None = fetch from indices)
            force_full: Force full data fetch for all tickers
            prefer_ttm: Prefer TTM financial data
            
        Returns:
            Dictionary with processing statistics
        """
        start_time = time.time()
        
        print("\n" + "=" * 60)
        print("STOCK DATA ORCHESTRATOR")
        print("=" * 60)
        print(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Indices: {', '.join(self.indices)}")
        print(f"TTM Preferred: {prefer_ttm}")
        print("=" * 60)
        
        # Get tickers
        if tickers is None:
            symbols_df = self.get_symbols()
            tickers = symbols_df['Symbol'].tolist()
        else:
            # Filter provided tickers against blacklist
            tickers = filter_blacklisted(tickers)
        
        # Also add existing database tickers if doing updates
        if not force_full:
            try:
                db_tickers = db_interactions.import_ticker_list()
                db_tickers = filter_blacklisted(db_tickers)
                # Combine with fetched tickers, removing duplicates
                all_tickers = list(set(tickers + db_tickers))
                print(f"\n✓ Combined {len(tickers)} index tickers + {len(db_tickers)} DB tickers = {len(all_tickers)} unique")
                tickers = all_tickers
            except Exception as e:
                print(f"⚠️  Could not fetch DB tickers: {e}")
        
        print(f"\nProcessing {len(tickers)} tickers...")
        
        # Process each ticker
        for i, ticker in enumerate(tickers, 1):
            try:
                print(f"\n[{i}/{len(tickers)}]", end="")
                self.process_ticker(ticker, force_full, prefer_ttm)
            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted by user")
                break
            except Exception as e:
                print(f"\n   ✗ Unexpected error: {e}")
                traceback.print_exc()
                self.processing_stats['errors'] += 1
        
        # Summary
        elapsed_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("PROCESSING COMPLETE")
        print("=" * 60)
        print(f"Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duration: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
        print(f"\nStatistics:")
        print(f"  Total processed: {self.processing_stats['total']}")
        print(f"  Successful:      {self.processing_stats['success']}")
        print(f"  Skipped:         {self.processing_stats['skipped']}")
        print(f"  Errors:          {self.processing_stats['errors']}")
        print(f"  Blacklisted:     {self.processing_stats['blacklisted']}")
        print("=" * 60)
        
        # Show blacklist summary
        bl_summary = self.blacklist.get_summary()
        if bl_summary['active'] > 0:
            print(f"\nBlacklist Summary ({bl_summary['active']} active):")
            for reason, count in bl_summary['by_reason'].items():
                print(f"  {reason}: {count}")
        
        return self.processing_stats


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description='Stock Data Orchestrator - Fetch and process stock market data'
    )
    parser.add_argument(
        '--indices', '-i',
        nargs='+',
        default=['C25', 'SP500'],
        help='Index codes to fetch (e.g., C25 SP500 DAX40)'
    )
    parser.add_argument(
        '--ticker', '-t',
        nargs='+',
        help='Specific ticker(s) to process'
    )
    parser.add_argument(
        '--update-only', '-u',
        action='store_true',
        help='Only update existing tickers in database'
    )
    parser.add_argument(
        '--force-full', '-f',
        action='store_true',
        help='Force full data fetch (not incremental)'
    )
    parser.add_argument(
        '--no-ttm',
        action='store_true',
        help='Use annual data instead of TTM'
    )
    parser.add_argument(
        '--no-market-indices',
        action='store_true',
        help='Exclude market indices (^VIX, ^GSPC, etc.)'
    )
    parser.add_argument(
        '--list-indices',
        action='store_true',
        help='List available indices and exit'
    )
    parser.add_argument(
        '--show-blacklist',
        action='store_true',
        help='Show blacklisted tickers and exit'
    )
    
    args = parser.parse_args()
    
    # Handle info commands
    if args.list_indices:
        print("\n" + "="*60)
        print("AVAILABLE STOCK INDICES")
        print("="*60)
        
        # Group indices by region
        regions = {
            'USA': ['SP500', 'NASDAQ100', 'DOW30', 'SP400', 'SP600', 'RUSSELL1000'],
            'Denmark': ['C25'],
            'Germany': ['DAX40'],
            'France': ['CAC40'],
            'UK': ['FTSE100'],
            'Netherlands': ['AEX25'],
            'Sweden': ['OMX30'],
            'Finland': ['OMXH25'],
            'Spain': ['IBEX35'],
            'Switzerland': ['SMI'],
            'Italy': ['FTSEMIB'],
            'Belgium': ['BEL20'],
            'Austria': ['ATX'],
            'Norway': ['OBX'],
            'Portugal': ['PSI20'],
            'Pan-European': ['STOXX50', 'STOXX600'],
        }
        
        fetcher = DynamicIndexFetcher()
        available = fetcher.get_available_indices()
        
        for region, codes in regions.items():
            valid_codes = [c for c in codes if c in available]
            if valid_codes:
                print(f"\n{region}:")
                for code in valid_codes:
                    print(f"  {code:12} {available[code]}")
        
        print("\n" + "-"*60)
        print("Market Indices (for benchmarking):")
        for name, symbol in fetcher.get_market_indices().items():
            print(f"  {name:12} {symbol}")
        
        print("\n" + "-"*60)
        print("Example commands:")
        print("  # Fetch all European major indices:")
        print("  python stock_orchestrator.py --indices DAX40 CAC40 FTSE100 AEX25 SMI FTSEMIB")
        print("")
        print("  # Fetch US + Scandinavia:")
        print("  python stock_orchestrator.py --indices SP500 NASDAQ100 C25 OMX30 OMXH25")
        print("")
        print("  # Fetch everything (large dataset!):")
        print("  python stock_orchestrator.py --indices SP500 NASDAQ100 DAX40 CAC40 FTSE100 C25 OMX30 SMI FTSEMIB AEX25")
        return
    
    if args.show_blacklist:
        manager = get_blacklist_manager()
        summary = manager.get_summary()
        print("\nBlacklist Summary:")
        print("-" * 40)
        print(f"Total entries: {summary['total']}")
        print(f"Active: {summary['active']}")
        print(f"By reason: {summary['by_reason']}")
        if summary['active'] > 0:
            print("\nActive blacklisted tickers:")
            for ticker in summary['active_tickers']:
                details = manager.get_blacklist_details(ticker)
                print(f"  {ticker}: {details.get('reason')} - {details.get('details', '')[:40]}")
        return
    
    # Run orchestrator
    orchestrator = StockDataOrchestrator(
        indices=args.indices,
        include_market_indices=not args.no_market_indices
    )
    
    if args.update_only:
        # Only process tickers already in database
        try:
            tickers = db_interactions.import_ticker_list()
            tickers = filter_blacklisted(tickers)
        except Exception as e:
            print(f"Error fetching database tickers: {e}")
            return
    else:
        tickers = args.ticker if args.ticker else None
    
    orchestrator.run(
        tickers=tickers,
        force_full=args.force_full,
        prefer_ttm=not args.no_ttm
    )


def run_with_indices(indices=None, tickers=None, force_full=False, prefer_ttm=True):
    """
    Run the orchestrator programmatically with specific indices.
    
    Use this function when calling from another module or in debugging mode.
    
    Args:
        indices: List of index codes (e.g., ['SP500', 'DAX40', 'CAC40'])
                Default: ['SP500', 'DAX40', 'CAC40', 'FTSE100', 'AEX25', 'SMI', 'FTSEMIB', 'IBEX35', 'BEL20', 'ATX']
        tickers: Optional list of specific tickers to process (overrides indices)
        force_full: Force full data fetch (not incremental)
        prefer_ttm: Prefer TTM financial data (default: True)
        
    Returns:
        Dictionary with processing statistics
        
    Example:
        # Run with default European/US indices:
        from stock_orchestrator import run_with_indices
        stats = run_with_indices()
        
        # Run with specific indices:
        stats = run_with_indices(indices=['C25', 'OMX30'])
        
        # Run specific tickers:
        stats = run_with_indices(tickers=['AAPL', 'MSFT', 'GOOGL'])
    """
    if indices is None:
        indices = ['SP500', 'DAX40', 'CAC40', 'FTSE100', 'AEX25', 'SMI', 'FTSEMIB', 'IBEX35', 'BEL20', 'ATX']
    
    orchestrator = StockDataOrchestrator(
        indices=indices,
        include_market_indices=True
    )
    
    return orchestrator.run(
        tickers=tickers,
        force_full=force_full,
        prefer_ttm=prefer_ttm
    )


if __name__ == "__main__":
    main()
