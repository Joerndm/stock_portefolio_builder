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
import json
import time
import datetime
import argparse
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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

# Cache file to track when Wikipedia was last fetched
MONTHLY_FETCH_CACHE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "monthly_index_fetch_cache.json"
)


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
    
    # Parallel processing defaults
    DEFAULT_MAX_WORKERS = 5  # Conservative to avoid API rate limits
    MAX_WORKERS_LIMIT = 10   # Upper limit to prevent overwhelming APIs
    
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
        self.failed_tickers = []  # List of (ticker, error_message) tuples
        self.error_counts = {}
        self.processing_stats = {
            'total': 0,
            'success': 0,
            'skipped': 0,
            'errors': 0,
            'blacklisted': 0
        }
        
        # Thread-safe lock for parallel processing
        self._stats_lock = threading.Lock()
        self._processed_lock = threading.Lock()
        
        # Monthly fetch tracking
        self._monthly_fetch_cache = self._load_monthly_fetch_cache()
    
    def _load_monthly_fetch_cache(self) -> Dict:
        """Load the monthly fetch cache from file."""
        if os.path.exists(MONTHLY_FETCH_CACHE_FILE):
            try:
                with open(MONTHLY_FETCH_CACHE_FILE, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}
    
    def _save_monthly_fetch_cache(self):
        """Save the monthly fetch cache to file."""
        try:
            with open(MONTHLY_FETCH_CACHE_FILE, 'w') as f:
                json.dump(self._monthly_fetch_cache, f, indent=2)
        except IOError as e:
            print(f"Warning: Could not save monthly fetch cache: {e}")
    
    def _should_fetch_from_wikipedia(self) -> bool:
        """
        Determine if we should fetch fresh data from Wikipedia.
        
        Returns True if:
        - No cache exists
        - Last fetch was in a different month
        - Cache is corrupted or invalid
        """
        if not self._monthly_fetch_cache:
            return True
        
        last_fetch_str = self._monthly_fetch_cache.get('last_wikipedia_fetch')
        if not last_fetch_str:
            return True
        
        try:
            last_fetch = datetime.datetime.fromisoformat(last_fetch_str)
            now = datetime.datetime.now()
            
            # Check if same month and year
            if last_fetch.year == now.year and last_fetch.month == now.month:
                return False  # Already fetched this month
            return True  # New month, should fetch
        except (ValueError, TypeError):
            return True  # Invalid date format, fetch fresh
    
    def _update_wikipedia_fetch_timestamp(self):
        """Update the timestamp of last Wikipedia fetch."""
        self._monthly_fetch_cache['last_wikipedia_fetch'] = datetime.datetime.now().isoformat()
        self._monthly_fetch_cache['indices_fetched'] = self.indices
        self._save_monthly_fetch_cache()
    
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
    
    def get_symbols(self, use_cache: bool = True, force_wikipedia: bool = False) -> pd.DataFrame:
        """
        Fetch stock symbols from configured indices.
        
        Uses a monthly caching strategy:
        - First run of each month: Fetch fresh from Wikipedia
        - Rest of the month: Use database tickers only (faster)
        
        Args:
            use_cache: Whether to use cached index data (24h cache)
            force_wikipedia: Force Wikipedia fetch regardless of monthly cache
            
        Returns:
            DataFrame with Symbol, Index, Exchange columns
        """
        print(f"\n{'='*60}")
        print(f"Fetching symbols from indices: {', '.join(self.indices)}")
        print(f"{'='*60}")
        
        should_fetch_wikipedia = force_wikipedia or self._should_fetch_from_wikipedia()
        
        if should_fetch_wikipedia:
            print("\n📡 Monthly Wikipedia fetch: Fetching fresh index constituents...")
            symbols_df = dynamic_fetch_index_data(
                indices=self.indices,
                include_market_indices=self.include_market_indices,
                use_cache=use_cache
            )
            
            # Update the monthly fetch timestamp
            self._update_wikipedia_fetch_timestamp()
            print(f"   ✓ Wikipedia fetch complete, cached for this month")
        else:
            # Use database tickers instead of Wikipedia
            print("\n📁 Using database tickers (Wikipedia already fetched this month)...")
            try:
                db_tickers = db_interactions.import_ticker_list()
                symbols_df = pd.DataFrame({
                    'Symbol': db_tickers,
                    'Index': 'Database',
                    'Exchange': 'various'
                })
                print(f"   ✓ Loaded {len(db_tickers)} tickers from database")
                
                # Add market indices if requested
                if self.include_market_indices:
                    market_indices_df = pd.DataFrame({
                        'Symbol': list(MARKET_INDICES.values()),
                        'Index': 'Market Index',
                        'Exchange': 'us'
                    })
                    symbols_df = pd.concat([symbols_df, market_indices_df], ignore_index=True)
                    symbols_df = symbols_df.drop_duplicates(subset=['Symbol'])
                    
            except Exception as e:
                print(f"   ⚠️  Could not load from database: {e}")
                print("   → Falling back to Wikipedia fetch...")
                symbols_df = dynamic_fetch_index_data(
                    indices=self.indices,
                    include_market_indices=self.include_market_indices,
                    use_cache=use_cache
                )
                self._update_wikipedia_fetch_timestamp()
        
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
    
    def force_wikipedia_refresh(self):
        """
        Force a fresh Wikipedia fetch, clearing the monthly cache.
        
        Use this when you want to update index constituents mid-month.
        """
        self._monthly_fetch_cache = {}
        self._save_monthly_fetch_cache()
        print("✓ Monthly cache cleared. Next run will fetch from Wikipedia.")
    
    def get_monthly_fetch_status(self) -> Dict:
        """
        Get information about the last Wikipedia fetch.
        
        Returns:
            Dictionary with last fetch date and status
        """
        if not self._monthly_fetch_cache:
            return {'status': 'never_fetched', 'last_fetch': None, 'will_fetch': True}
        
        last_fetch_str = self._monthly_fetch_cache.get('last_wikipedia_fetch')
        indices_fetched = self._monthly_fetch_cache.get('indices_fetched', [])
        
        try:
            last_fetch = datetime.datetime.fromisoformat(last_fetch_str)
            will_fetch = self._should_fetch_from_wikipedia()
            return {
                'status': 'cached',
                'last_fetch': last_fetch.strftime('%Y-%m-%d %H:%M:%S'),
                'last_fetch_month': last_fetch.strftime('%B %Y'),
                'indices_fetched': indices_fetched,
                'will_fetch_wikipedia': will_fetch
            }
        except (ValueError, TypeError):
            return {'status': 'invalid_cache', 'last_fetch': None, 'will_fetch': True}
    
    def _handle_ticker_error(self, ticker: str, error: Exception, 
                              error_type: str = "error") -> bool:
        """
        Handle errors for a ticker, potentially blacklisting if too many errors.
        
        If a ticker is blacklisted as 'delisted', also removes its data from the database
        to keep the database clean and prevent orphaned data.
        
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
            
            # If delisted, also remove from database to keep it clean
            if blacklist_reason == 'delisted':
                print(f"   ↳ Cleaning up database for delisted ticker {ticker}...")
                try:
                    deleted = self.blacklist.cleanup_database(ticker, self.db_con)
                    if deleted:
                        total_deleted = sum(deleted.values())
                        print(f"   ↳ Removed {total_deleted} records from database")
                except Exception as cleanup_error:
                    print(f"   ⚠️ Database cleanup failed: {cleanup_error}")
            
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
    
    def _fetch_and_export_quarterly_data(self, ticker: str, force_fetch: bool = False) -> bool:
        """
        Fetch and export quarterly financial data for TTM calculations.
        
        This method implements a smart caching approach:
        1. Check if data is fresh (fetched within 30 days AND recent quarter < 100 days old)
        2. If fresh: skip API fetch, use existing database data
        3. If stale: fetch from yfinance, merge, and export
        4. Update fetch metadata after successful export
        
        Args:
            ticker: Stock ticker symbol
            force_fetch: If True, bypass freshness check and always fetch from API
            
        Returns:
            True if quarterly data was successfully processed (either from cache or fresh fetch)
        """
        from enhanced_financial_fetcher import (
            transform_quarterly_to_db_schema,
            calculate_ttm_from_quarterly,
            merge_quarterly_data
        )
        
        try:
            print(f"   → Processing quarterly data for {ticker}")
            
            # Check if we should fetch from API or use cached data
            if not force_fetch:
                should_fetch, reason = db_interactions.should_fetch_quarterly_data(ticker)
                if not should_fetch:
                    print(f"   ↳ Using cached quarterly data ({reason})")
                    return True
                else:
                    print(f"   ↳ Fetching from API: {reason}")
            else:
                print(f"   ↳ Force fetch enabled - bypassing cache")
            
            # Track the latest quarter and count for metadata update
            latest_quarter_end = None
            total_quarters = 0
            
            # 1. Fetch quarterly income statement from yfinance
            quarterly_income = self.quarterly_fetcher.fetch_quarterly_income_statement(ticker)
            if not quarterly_income.empty:
                # Get existing database data
                db_income = db_interactions.import_quarterly_income_data(ticker)
                existing_quarters = len(db_income)
                
                # Transform yfinance data to database schema
                transformed_income = transform_quarterly_to_db_schema(
                    quarterly_income, ticker, statement_type='income'
                )
                
                # Merge with existing data (keeping unique quarters)
                merged_income = merge_quarterly_data(db_income, transformed_income, ticker)
                
                # Calculate TTM values if we have at least 4 quarters
                if len(merged_income) >= 4:
                    merged_income = calculate_ttm_from_quarterly(
                        merged_income, ticker, statement_type='income'
                    )
                
                # Export to database (delete-then-insert pattern already in export function)
                try:
                    if not merged_income.empty:
                        db_interactions.export_quarterly_income_stmt(merged_income)
                        new_quarters = len(merged_income) - existing_quarters
                        if new_quarters > 0:
                            print(f"   ↳ Added {new_quarters} new quarters (total: {len(merged_income)})")
                        
                        # Track for metadata
                        total_quarters = len(merged_income)
                        if 'fiscal_quarter_end' in merged_income.columns:
                            latest_quarter_end = merged_income['fiscal_quarter_end'].max()
                            # Convert to date if it's a datetime/timestamp
                            if hasattr(latest_quarter_end, 'date'):
                                latest_quarter_end = latest_quarter_end.date()
                            elif hasattr(latest_quarter_end, 'to_pydatetime'):
                                latest_quarter_end = latest_quarter_end.to_pydatetime().date()
                except Exception as e:
                    print(f"   ↳ Could not export quarterly income: {e}")
            
            # 2. Fetch quarterly balance sheet from yfinance
            quarterly_bs = self.quarterly_fetcher.fetch_quarterly_balance_sheet(ticker)
            if not quarterly_bs.empty:
                db_bs = db_interactions.import_quarterly_balancesheet_data(ticker)
                
                transformed_bs = transform_quarterly_to_db_schema(
                    quarterly_bs, ticker, statement_type='balancesheet'
                )
                
                merged_bs = merge_quarterly_data(db_bs, transformed_bs, ticker)
                
                try:
                    if not merged_bs.empty:
                        db_interactions.export_quarterly_balancesheet(merged_bs)
                except Exception as e:
                    print(f"   ↳ Could not export quarterly balance sheet: {e}")
            
            # 3. Fetch quarterly cash flow from yfinance
            quarterly_cf = self.quarterly_fetcher.fetch_quarterly_cash_flow(ticker)
            if not quarterly_cf.empty:
                db_cf = db_interactions.import_quarterly_cashflow_data(ticker)
                
                transformed_cf = transform_quarterly_to_db_schema(
                    quarterly_cf, ticker, statement_type='cashflow'
                )
                
                merged_cf = merge_quarterly_data(db_cf, transformed_cf, ticker)
                
                if len(merged_cf) >= 4:
                    merged_cf = calculate_ttm_from_quarterly(
                        merged_cf, ticker, statement_type='cashflow'
                    )
                
                try:
                    if not merged_cf.empty:
                        db_interactions.export_quarterly_cashflow(merged_cf)
                except Exception as e:
                    print(f"   ↳ Could not export quarterly cash flow: {e}")
            
            # Update fetch metadata after successful processing
            db_interactions.update_quarterly_fetch_metadata(
                ticker=ticker,
                last_quarter_end=latest_quarter_end,
                quarters_count=total_quarters
            )
            
            return True
            
        except Exception as e:
            print(f"   ↳ Could not fetch quarterly data: {e}")
            return False
    
    def process_ratio_data(self, ticker: str, prefer_ttm: bool = True
                           ) -> Tuple[bool, Optional[pd.DataFrame]]:
        """
        Calculate and export financial ratios with smart update logic.
        
        Logic:
        1. INITIAL POPULATION (no ratio data exists):
           - Get ALL financial statements for the ticker
           - Calculate ratios from the earliest financial statement date onwards
           - Store which financial date was used for each ratio calculation
        
        2. DAILY UPDATE (ratio data exists, no new financial reports):
           - Only calculate ratios for new dates since last calculation
           - Use the most recent financial data
        
        3. RECALCULATION (new financial report available):
           - Detect new financial data (annual or quarterly)
           - Delete ratio data from the new report's fiscal date onwards
           - Recalculate with the new financial data
        
        Args:
            ticker: Stock ticker symbol
            prefer_ttm: Whether to prefer TTM calculations
            
        Returns:
            Tuple of (success, ratio_data_df or None)
        """
        from stock_data_fetch import combine_stock_data, calculate_ratios, drop_nan_values
        from datetime import timedelta
        
        try:
            # Check current ratio data status
            ratio_exists = db_interactions.does_stock_exists_stock_ratio_data(ticker)
            
            # Get newest financial data date (annual + quarterly)
            newest_financial_date, fin_source = db_interactions.get_newest_financial_date(
                ticker, include_quarterly=prefer_ttm
            )
            
            if newest_financial_date is None:
                print(f"   [SKIP] No financial data available for ratio calculation")
                return False, None
            
            # Determine which scenario we're in
            if not ratio_exists:
                # ========================================
                # SCENARIO 1: INITIAL POPULATION
                # ========================================
                print(f"   [INIT] Initial ratio population for {ticker}")
                
                # Get ALL financial statement dates
                all_fin_dates = db_interactions.get_all_financial_dates(ticker, include_quarterly=prefer_ttm)
                
                if all_fin_dates.empty:
                    print(f"   [SKIP] No financial statement dates found")
                    return False, None
                
                # Start from the earliest financial statement date
                earliest_fin_date = all_fin_dates['date'].min()
                start_date = pd.to_datetime(earliest_fin_date).strftime('%Y-%m-%d')
                
                # Get ALL financial data
                query = f"""SELECT COUNT(financial_Statement_Date)
                            FROM stock_income_stmt_data
                            WHERE ticker = '{ticker}'"""
                entry_amount = pd.read_sql(sql=query, con=self.db_con)
                full_stock_financial_data_df = db_interactions.import_stock_financial_data(
                    amount=entry_amount.loc[0, entry_amount.columns[0]],
                    stock_ticker=ticker
                )
                
                recalculate_from = start_date
                
            else:
                # ========================================
                # SCENARIO 2 or 3: UPDATE or RECALCULATION
                # ========================================
                
                # Get the last ratio record's info
                last_ratio_date, last_fin_date_used = db_interactions.get_last_ratio_financial_date(ticker)
                
                # Check if new financial data is available
                needs_recalculation = False
                if last_fin_date_used is not None and newest_financial_date > last_fin_date_used:
                    needs_recalculation = True
                    print(f"   [RECALC] New financial data detected: {newest_financial_date} > {last_fin_date_used}")
                elif last_fin_date_used is None:
                    # Old ratio data without financial_date_used tracking - needs recalculation
                    needs_recalculation = True
                    print(f"   [RECALC] Legacy ratio data without financial_date_used tracking")
                
                if needs_recalculation:
                    # ========================================
                    # SCENARIO 3: RECALCULATION NEEDED
                    # ========================================
                    
                    all_fin_dates = db_interactions.get_all_financial_dates(ticker, include_quarterly=prefer_ttm)
                    
                    if last_fin_date_used is None:
                        # Legacy data - delete all and recalculate from earliest financial date
                        earliest_fin_date = all_fin_dates['date'].min()
                        recalculate_from = pd.to_datetime(earliest_fin_date).strftime('%Y-%m-%d')
                        print(f"   [RECALC] Starting from earliest financial date: {recalculate_from}")
                    else:
                        # Find the first financial date newer than what we've used
                        fin_dates_after_used = all_fin_dates[all_fin_dates['date'] > pd.to_datetime(last_fin_date_used)]
                        
                        if not fin_dates_after_used.empty:
                            # Recalculate from the first new financial date
                            first_new_fin_date = fin_dates_after_used['date'].min()
                            recalculate_from = pd.to_datetime(first_new_fin_date).strftime('%Y-%m-%d')
                        else:
                            # Fallback: start from day after last ratio
                            recalculate_from = (pd.to_datetime(last_ratio_date) + timedelta(days=1)).strftime('%Y-%m-%d')
                    
                    # Delete ratios from the recalculation start date onwards
                    deleted_count = db_interactions.delete_stock_ratio_data_from_date(
                        ticker, recalculate_from
                    )
                    print(f"   [DELETE] Deleted {deleted_count} ratio records from {recalculate_from}")
                    
                    # Get ALL financial data for proper forward-fill
                    query = f"""SELECT COUNT(financial_Statement_Date)
                                FROM stock_income_stmt_data
                                WHERE ticker = '{ticker}'"""
                    entry_amount = pd.read_sql(sql=query, con=self.db_con)
                    full_stock_financial_data_df = db_interactions.import_stock_financial_data(
                        amount=entry_amount.loc[0, entry_amount.columns[0]],
                        stock_ticker=ticker
                    )
                    
                else:
                    # ========================================
                    # SCENARIO 2: SIMPLE DAILY UPDATE
                    # ========================================
                    
                    # Check if already up to date
                    today = datetime.datetime.now().strftime("%Y-%m-%d")
                    if str(last_ratio_date) == today:
                        print(f"   [OK] Ratio data is up to date")
                        return True, None
                    
                    # Just calculate for new days
                    recalculate_from = (pd.to_datetime(last_ratio_date) + timedelta(days=1)).strftime('%Y-%m-%d')
                    print(f"   [UPDATE] Adding ratios from {recalculate_from}")
                    
                    # Only need the latest financial data
                    full_stock_financial_data_df = db_interactions.import_stock_financial_data(
                        amount=1, stock_ticker=ticker
                    )
            
            # Validate financial data
            if full_stock_financial_data_df is None or full_stock_financial_data_df.empty:
                print(f"   [SKIP] No financial data for ratio calculation")
                return False, None
            
            full_stock_financial_data_df = full_stock_financial_data_df.dropna(axis=1)
            
            if full_stock_financial_data_df.empty:
                print(f"   [SKIP] Financial data empty after dropna")
                return False, None
            
            # Get price data from the recalculate_from date
            query = f"""SELECT *
                        FROM stock_price_data
                        WHERE ticker = '{ticker}' AND date >= '{recalculate_from}'"""
            stock_price_data_df = pd.read_sql(sql=query, con=self.db_con)
            
            if stock_price_data_df.empty:
                print(f"   [SKIP] No price data for ratio calculation from {recalculate_from}")
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
            
            # Extract ratio columns and add financial_date_used
            ratio_columns = ['date', 'ticker', 'P/S', 'P/E', 'P/B', 'P/FCF']
            stock_ratio_data_df = combined_stock_data_df[ratio_columns].copy()
            stock_ratio_data_df = stock_ratio_data_df.rename(columns={
                "P/S": "p_s", "P/E": "p_e", "P/B": "p_b", "P/FCF": "p_fcf"
            })
            
            # Add financial_date_used column based on which financial data was applied
            # For each price date, find which financial statement date was used
            stock_ratio_data_df['financial_date_used'] = None
            stock_ratio_data_df['date'] = pd.to_datetime(stock_ratio_data_df['date'])
            fin_dates_sorted = sorted(full_stock_financial_data_df['date'].unique())
            
            for fin_date in fin_dates_sorted:
                fin_date_ts = pd.to_datetime(fin_date)
                mask = stock_ratio_data_df['date'] >= fin_date_ts
                stock_ratio_data_df.loc[mask, 'financial_date_used'] = fin_date
            
            stock_ratio_data_df = drop_nan_values(stock_ratio_data_df)
            
            if stock_ratio_data_df.empty:
                print(f"   [SKIP] No valid ratio data to export")
                return False, None
            
            # Export to database
            db_interactions.export_stock_ratio_data(stock_ratio_data_df)
            print(f"   [OK] Exported {len(stock_ratio_data_df)} ratio records")
            
            return True, stock_ratio_data_df
            
        except Exception as e:
            print(f"   [ERROR] Error processing ratio data: {e}")
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
            error_msg = f"Cannot access: {str(e)[:60]}"
            print(f"   ✗ {error_msg}")
            self._handle_ticker_error(ticker, e, "info")
            self._add_failed_ticker(ticker, error_msg)
            self.processing_stats['errors'] += 1
            return False
        
        is_index = self._is_index_ticker(ticker, stock_info)
        
        # Step 1: Stock Info
        success, _ = self.process_stock_info(ticker)
        if not success:
            self._add_failed_ticker(ticker, "Stock info processing failed")
            self.processing_stats['errors'] += 1
            return False
        
        # Step 2: Price Data
        success, _ = self.process_price_data(ticker, stock_info, force_full)
        if not success:
            self._add_failed_ticker(ticker, "Price data processing failed")
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
    
    def _update_stats(self, stat_key: str, increment: int = 1):
        """
        Thread-safe update of processing statistics.
        
        Args:
            stat_key: Key in processing_stats dict
            increment: Value to add (default 1)
        """
        with self._stats_lock:
            self.processing_stats[stat_key] += increment
    
    def _add_processed_ticker(self, ticker: str):
        """Thread-safe addition of processed ticker."""
        with self._processed_lock:
            self.processed_tickers.add(ticker)
    
    def _add_failed_ticker(self, ticker: str, error_msg: str):
        """Thread-safe addition of failed ticker with error message."""
        with self._processed_lock:
            self.failed_tickers.append((ticker, error_msg))
    
    def _process_ticker_worker(self, ticker: str, force_full: bool, 
                                prefer_ttm: bool, progress_info: str) -> Tuple[str, bool, Optional[str]]:
        """
        Worker function for parallel ticker processing.
        
        This is a wrapper around process_ticker that's safe for thread pool execution.
        Each thread gets its own database connection for thread safety.
        
        Args:
            ticker: Stock ticker symbol
            force_full: Force full data fetch
            prefer_ttm: Prefer TTM financial data
            progress_info: Progress string like "[1/100]"
            
        Returns:
            Tuple of (ticker, success, error_message or None)
        """
        try:
            # Print progress (thread-safe via GIL for simple prints)
            print(f"\n{progress_info} Starting: {ticker}")
            
            success = self.process_ticker(ticker, force_full, prefer_ttm)
            return (ticker, success, None)
            
        except Exception as e:
            error_msg = str(e)
            print(f"\n{progress_info} ✗ {ticker}: {error_msg}")
            self._update_stats('errors')
            self._add_failed_ticker(ticker, error_msg)
            return (ticker, False, error_msg)
    
    def run(self, tickers: List[str] = None, force_full: bool = False,
            prefer_ttm: bool = True, force_wikipedia: bool = False,
            parallel: bool = False, max_workers: int = None) -> Dict:
        """
        Run the complete data pipeline.
        
        Args:
            tickers: Optional list of specific tickers (None = fetch from indices)
            force_full: Force full data fetch for all tickers
            prefer_ttm: Prefer TTM financial data
            force_wikipedia: Force fresh Wikipedia fetch (ignore monthly cache)
            parallel: Enable parallel processing with ThreadPoolExecutor
            max_workers: Number of parallel workers (default: 5, max: 10)
            
        Returns:
            Dictionary with processing statistics
        """
        start_time = time.time()
        
        # Validate max_workers
        if max_workers is None:
            max_workers = self.DEFAULT_MAX_WORKERS
        max_workers = min(max(1, max_workers), self.MAX_WORKERS_LIMIT)
        
        print("\n" + "=" * 60)
        print("STOCK DATA ORCHESTRATOR")
        print("=" * 60)
        print(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Indices: {', '.join(self.indices)}")
        print(f"TTM Preferred: {prefer_ttm}")
        print(f"Parallel Mode: {'Enabled (' + str(max_workers) + ' workers)' if parallel else 'Disabled'}")
        print("=" * 60)
        
        # Get tickers
        if tickers is None:
            symbols_df = self.get_symbols(force_wikipedia=force_wikipedia)
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
        
        if parallel:
            self._run_parallel(tickers, force_full, prefer_ttm, max_workers)
        else:
            self._run_sequential(tickers, force_full, prefer_ttm)
        
        # Summary
        elapsed_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("PROCESSING COMPLETE")
        print("=" * 60)
        print(f"Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duration: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
        if parallel:
            print(f"Mode: Parallel ({max_workers} workers)")
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
        
        # Show failed tickers for troubleshooting
        if self.failed_tickers:
            print(f"\n" + "-" * 60)
            print(f"FAILED TICKERS ({len(self.failed_tickers)}):")
            print("-" * 60)
            for ticker, error_msg in self.failed_tickers:
                # Truncate long error messages
                if len(error_msg) > 80:
                    error_msg = error_msg[:77] + "..."
                print(f"  {ticker}: {error_msg}")
            print("-" * 60)
        
        # Add failed tickers to stats for programmatic access
        self.processing_stats['failed_tickers'] = self.failed_tickers.copy()
        
        return self.processing_stats
    
    def _run_sequential(self, tickers: List[str], force_full: bool, prefer_ttm: bool):
        """
        Run pipeline sequentially (original behavior).
        
        Args:
            tickers: List of ticker symbols
            force_full: Force full data fetch
            prefer_ttm: Prefer TTM financial data
        """
        for i, ticker in enumerate(tickers, 1):
            try:
                print(f"\n[{i}/{len(tickers)}]", end="")
                self.process_ticker(ticker, force_full, prefer_ttm)
            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted by user")
                break
            except Exception as e:
                error_msg = str(e)
                print(f"\n   ✗ Unexpected error: {error_msg}")
                traceback.print_exc()
                self._add_failed_ticker(ticker, f"Unexpected: {error_msg[:50]}")
                self.processing_stats['errors'] += 1
    
    def _run_parallel(self, tickers: List[str], force_full: bool, 
                      prefer_ttm: bool, max_workers: int):
        """
        Run pipeline with parallel processing using ThreadPoolExecutor.
        
        This provides significant speedup for I/O-bound operations like
        fetching data from yfinance and writing to the database.
        
        Args:
            tickers: List of ticker symbols
            force_full: Force full data fetch
            prefer_ttm: Prefer TTM financial data
            max_workers: Number of parallel workers
        """
        total = len(tickers)
        completed = 0
        
        print(f"\n🚀 Starting parallel processing with {max_workers} workers...")
        print(f"   (Note: Output may be interleaved due to parallel execution)\n")
        
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks with progress info
                future_to_ticker = {
                    executor.submit(
                        self._process_ticker_worker, 
                        ticker, 
                        force_full, 
                        prefer_ttm,
                        f"[{i}/{total}]"
                    ): ticker 
                    for i, ticker in enumerate(tickers, 1)
                }
                
                # Process results as they complete
                for future in as_completed(future_to_ticker):
                    ticker = future_to_ticker[future]
                    completed += 1
                    
                    try:
                        ticker_result, success, error_msg = future.result()
                        if success:
                            print(f"   ✓ {ticker_result} completed ({completed}/{total})")
                        elif error_msg:
                            print(f"   ✗ {ticker_result} failed: {error_msg[:50]}")
                    except Exception as e:
                        print(f"   ✗ {ticker} exception: {e}")
                        self._update_stats('errors')
                        
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user - waiting for running tasks to complete...")
            # ThreadPoolExecutor will wait for running tasks on context exit


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description='Stock Data Orchestrator - Fetch and process stock market data'
    )
    parser.add_argument(
        '--indices', '-i',
        nargs='+',
        default=['C25', 'SP500', 'DAX40', 'CAC40', 'FTSE100', 'AEX25', 'SMI', 'FTSEMIB', 'IBEX35', 'BEL20', 'ATX'],
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
    parser.add_argument(
        '--force-wikipedia',
        action='store_true',
        help='Force fresh Wikipedia fetch (ignore monthly cache)'
    )
    parser.add_argument(
        '--show-fetch-status',
        action='store_true',
        help='Show Wikipedia fetch status and exit'
    )
    parser.add_argument(
        '--no-parallel',
        action='store_true',
        help='Disable parallel processing (parallel is enabled by default)'
    )
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=5,
        help='Number of parallel workers (default: 5, max: 10)'
    )
    
    args = parser.parse_args()
    
    # Handle info commands
    if args.show_fetch_status:
        orchestrator = StockDataOrchestrator(indices=args.indices)
        status = orchestrator.get_monthly_fetch_status()
        print("\n" + "="*60)
        print("WIKIPEDIA FETCH STATUS")
        print("="*60)
        print(f"Status: {status['status']}")
        if status.get('last_fetch'):
            print(f"Last fetch: {status['last_fetch']}")
            print(f"Month: {status.get('last_fetch_month', 'N/A')}")
            print(f"Indices fetched: {', '.join(status.get('indices_fetched', []))}")
        print(f"Will fetch Wikipedia on next run: {'Yes' if status.get('will_fetch_wikipedia', status.get('will_fetch', True)) else 'No (using database)'}")
        print("="*60)
        print("\nTo force a fresh Wikipedia fetch, use: --force-wikipedia")
        return
    
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
        prefer_ttm=not args.no_ttm,
        force_wikipedia=args.force_wikipedia,
        parallel=not args.no_parallel,
        max_workers=args.workers
    )


def run_with_indices(indices=None, tickers=None, force_full=False, prefer_ttm=True, 
                     force_wikipedia=False, parallel=True, max_workers=5):
    """
    Run the orchestrator programmatically with specific indices.
    
    Use this function when calling from another module or in debugging mode.
    
    Args:
        indices: List of index codes (e.g., ['SP500', 'DAX40', 'CAC40'])
                Default: ['SP500', 'DAX40', 'CAC40', 'FTSE100', 'AEX25', 'SMI', 'FTSEMIB', 'IBEX35', 'BEL20', 'ATX']
        tickers: Optional list of specific tickers to process (overrides indices)
        force_full: Force full data fetch (not incremental)
        prefer_ttm: Prefer TTM financial data (default: True)
        force_wikipedia: Force fresh Wikipedia fetch (ignore monthly cache)
        parallel: Enable parallel processing for faster execution (default: True)
        max_workers: Number of parallel workers (default: 5, max: 10)
        
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
        
        # Force Wikipedia refresh:
        stats = run_with_indices(force_wikipedia=True)
        
        # Run with parallel processing (faster):
        stats = run_with_indices(parallel=True, max_workers=5)
        
        # Run specific tickers in parallel:
        stats = run_with_indices(tickers=['AAPL', 'MSFT', 'GOOGL', 'AMZN'], parallel=True)
    """
    if indices is None:
        indices = ['C25', 'SP500', 'DAX40', 'CAC40', 'FTSE100', 'AEX25', 'SMI', 'FTSEMIB', 'IBEX35', 'BEL20', 'ATX']
    
    orchestrator = StockDataOrchestrator(
        indices=indices,
        include_market_indices=True
    )
    
    return orchestrator.run(
        tickers=tickers,
        force_full=force_full,
        prefer_ttm=prefer_ttm,
        force_wikipedia=force_wikipedia,
        parallel=parallel,
        max_workers=max_workers
    )


if __name__ == "__main__":
    main()
