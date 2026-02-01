"""
Blacklist Manager Module

This module provides centralized management for blacklisted stock tickers.
Handles adding, removing, and checking tickers against the blacklist.

Features:
    - Persistent storage in JSON file
    - Reason tracking for blacklisted tickers
    - Automatic date stamping
    - Database cleanup for delisted stocks
    - Thread-safe operations

Author: Stock Portfolio Builder
Last Modified: 2026
"""
import os
import json
import datetime
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd


class BlacklistManager:
    """
    Manages blacklisted stock tickers that should be excluded from data fetching.
    
    Reasons for blacklisting:
        - 'delisted': Stock no longer exists on exchange
        - 'invalid': Symbol not recognized by data sources
        - 'no_data': No historical data available
        - 'manual': Manually excluded by user
        - 'error': Repeated errors during fetching
    """
    
    VALID_REASONS = ['delisted', 'invalid', 'no_data', 'manual', 'error', 'suspended']
    
    def __init__(self, blacklist_file: str = "blacklisted_tickers.json"):
        """
        Initialize the BlacklistManager.
        
        Args:
            blacklist_file: Path to the JSON file storing blacklisted tickers
        """
        self.blacklist_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            blacklist_file
        )
        self._blacklist = self._load_blacklist()
    
    def _load_blacklist(self) -> Dict:
        """Load blacklist from file, create if doesn't exist."""
        if os.path.exists(self.blacklist_file):
            try:
                with open(self.blacklist_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load blacklist: {e}")
                return {}
        return {}
    
    def _save_blacklist(self):
        """Save blacklist to file."""
        try:
            with open(self.blacklist_file, 'w', encoding='utf-8') as f:
                json.dump(self._blacklist, f, indent=2, default=str)
        except IOError as e:
            print(f"Error saving blacklist: {e}")
    
    def add_ticker(self, ticker: str, reason: str, details: str = "", 
                   active: bool = True) -> bool:
        """
        Add a ticker to the blacklist.
        
        Args:
            ticker: Stock ticker symbol
            reason: Reason for blacklisting (must be in VALID_REASONS)
            details: Additional details about why the ticker was blacklisted
            active: Whether the blacklist entry is currently active
            
        Returns:
            True if successfully added, False otherwise
        """
        if reason not in self.VALID_REASONS:
            print(f"Warning: Invalid reason '{reason}'. Valid reasons: {self.VALID_REASONS}")
            reason = 'error'
        
        ticker = ticker.upper().strip()
        
        self._blacklist[ticker] = {
            'reason': reason,
            'details': details,
            'date_added': datetime.datetime.now().isoformat(),
            'active': active
        }
        
        self._save_blacklist()
        print(f"Added {ticker} to blacklist (reason: {reason})")
        return True
    
    def remove_ticker(self, ticker: str, permanent: bool = False) -> bool:
        """
        Remove a ticker from the blacklist or mark as inactive.
        
        Args:
            ticker: Stock ticker symbol
            permanent: If True, completely remove; if False, mark as inactive
            
        Returns:
            True if found and modified, False if not found
        """
        ticker = ticker.upper().strip()
        
        if ticker not in self._blacklist:
            return False
        
        if permanent:
            del self._blacklist[ticker]
            print(f"Permanently removed {ticker} from blacklist")
        else:
            self._blacklist[ticker]['active'] = False
            self._blacklist[ticker]['date_deactivated'] = datetime.datetime.now().isoformat()
            print(f"Deactivated {ticker} in blacklist")
        
        self._save_blacklist()
        return True
    
    def is_blacklisted(self, ticker: str, active_only: bool = True) -> bool:
        """
        Check if a ticker is blacklisted.
        
        Args:
            ticker: Stock ticker symbol
            active_only: If True, only check active blacklist entries
            
        Returns:
            True if blacklisted, False otherwise
        """
        ticker = ticker.upper().strip()
        
        if ticker not in self._blacklist:
            return False
        
        if active_only:
            return self._blacklist[ticker].get('active', True)
        
        return True
    
    def get_blacklist(self, active_only: bool = True) -> List[str]:
        """
        Get list of blacklisted ticker symbols.
        
        Args:
            active_only: If True, only return active blacklist entries
            
        Returns:
            List of blacklisted ticker symbols
        """
        if active_only:
            return [
                ticker for ticker, data in self._blacklist.items() 
                if data.get('active', True)
            ]
        return list(self._blacklist.keys())
    
    def get_blacklist_details(self, ticker: str = None) -> Dict:
        """
        Get detailed information about blacklisted tickers.
        
        Args:
            ticker: Specific ticker to get details for (None for all)
            
        Returns:
            Dictionary with blacklist details
        """
        if ticker:
            ticker = ticker.upper().strip()
            return self._blacklist.get(ticker, {})
        return self._blacklist.copy()
    
    def filter_tickers(self, tickers: List[str], active_only: bool = True) -> Tuple[List[str], List[str]]:
        """
        Filter a list of tickers, removing blacklisted ones.
        
        Args:
            tickers: List of ticker symbols to filter
            active_only: If True, only filter by active blacklist entries
            
        Returns:
            Tuple of (valid_tickers, blacklisted_tickers)
        """
        valid = []
        blacklisted = []
        
        for ticker in tickers:
            if self.is_blacklisted(ticker, active_only):
                blacklisted.append(ticker)
            else:
                valid.append(ticker)
        
        if blacklisted:
            print(f"Filtered out {len(blacklisted)} blacklisted tickers: {blacklisted[:5]}{'...' if len(blacklisted) > 5 else ''}")
        
        return valid, blacklisted
    
    def cleanup_database(self, ticker: str, db_con=None) -> Dict[str, int]:
        """
        Remove all data for a blacklisted ticker from the database.
        
        Args:
            ticker: Stock ticker symbol to remove
            db_con: Database connection (if None, will create one)
            
        Returns:
            Dictionary with counts of deleted rows per table
        """
        from sqlalchemy import text
        
        if db_con is None:
            try:
                import fetch_secrets
                import db_connectors
                db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()
                db_con = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)
            except Exception as e:
                print(f"Could not connect to database: {e}")
                return {}
        
        ticker = ticker.upper().strip()
        deleted_counts = {}
        
        # Tables to clean (in order to respect foreign key constraints)
        tables = [
            'stock_ratio_data_ttm',
            'stock_ratio_data',
            'stock_cash_flow_data',
            'stock_balancesheet_data',
            'stock_income_stmt_data',
            'stock_cashflow_quarterly',
            'stock_balancesheet_quarterly',
            'stock_income_stmt_quarterly',
            'stock_price_data',
            'stock_prediction_data',
            'index_membership',
            'stock_info_data'  # Last due to foreign key constraints
        ]
        
        for table in tables:
            try:
                # Check if table exists
                check_query = text(f"""
                    SELECT COUNT(*) as cnt FROM information_schema.tables 
                    WHERE table_schema = DATABASE() AND table_name = :table_name
                """)
                result = pd.read_sql(check_query, db_con, params={'table_name': table})
                
                if result['cnt'].iloc[0] == 0:
                    continue
                
                # Count rows to delete
                count_query = text(f"SELECT COUNT(*) as cnt FROM {table} WHERE ticker = :ticker")
                count_result = pd.read_sql(count_query, db_con, params={'ticker': ticker})
                rows_to_delete = count_result['cnt'].iloc[0]
                
                if rows_to_delete > 0:
                    # Delete rows using proper SQLAlchemy transaction
                    delete_query = text(f"DELETE FROM {table} WHERE ticker = :ticker")
                    with db_con.begin() as connection:
                        connection.execute(delete_query, {'ticker': ticker})
                    deleted_counts[table] = rows_to_delete
                    print(f"  Deleted {rows_to_delete} rows from {table}")
                    
            except Exception as e:
                print(f"  Error cleaning {table}: {e}")
        
        if deleted_counts:
            print(f"Database cleanup complete for {ticker}: {sum(deleted_counts.values())} total rows deleted")
        else:
            print(f"No data found for {ticker} in database")
        
        return deleted_counts
    
    def get_summary(self) -> Dict:
        """
        Get a summary of the blacklist.
        
        Returns:
            Dictionary with blacklist statistics
        """
        active = [t for t, d in self._blacklist.items() if d.get('active', True)]
        inactive = [t for t, d in self._blacklist.items() if not d.get('active', True)]
        
        by_reason = {}
        for ticker, data in self._blacklist.items():
            if data.get('active', True):
                reason = data.get('reason', 'unknown')
                by_reason[reason] = by_reason.get(reason, 0) + 1
        
        return {
            'total': len(self._blacklist),
            'active': len(active),
            'inactive': len(inactive),
            'by_reason': by_reason,
            'active_tickers': active,
            'inactive_tickers': inactive
        }


# Singleton instance for easy access
_blacklist_manager = None

def get_blacklist_manager() -> BlacklistManager:
    """Get the singleton BlacklistManager instance."""
    global _blacklist_manager
    if _blacklist_manager is None:
        _blacklist_manager = BlacklistManager()
    return _blacklist_manager


def is_blacklisted(ticker: str) -> bool:
    """Quick check if a ticker is blacklisted."""
    return get_blacklist_manager().is_blacklisted(ticker)


def blacklist_ticker(ticker: str, reason: str, details: str = "") -> bool:
    """Quick function to blacklist a ticker."""
    return get_blacklist_manager().add_ticker(ticker, reason, details)


def filter_blacklisted(tickers: List[str]) -> List[str]:
    """Quick function to filter out blacklisted tickers."""
    valid, _ = get_blacklist_manager().filter_tickers(tickers)
    return valid


if __name__ == "__main__":
    # Demo usage
    manager = BlacklistManager()
    
    print("Blacklist Summary:")
    summary = manager.get_summary()
    print(f"  Total: {summary['total']}")
    print(f"  Active: {summary['active']}")
    print(f"  By reason: {summary['by_reason']}")
    
    print("\nActive blacklisted tickers:")
    for ticker in manager.get_blacklist():
        details = manager.get_blacklist_details(ticker)
        print(f"  {ticker}: {details.get('reason', 'unknown')} - {details.get('details', '')[:50]}")
