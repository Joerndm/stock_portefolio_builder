"""
Market Hours and Trading Day Utilities

This module provides utilities for handling market hours, trading days, and timezone
considerations when fetching stock data from multiple international exchanges.

Features:
    - Market hours definitions for major exchanges
    - Trading day validation (weekends, holidays)
    - Timezone-aware market status checks
    - Smart date adjustment for data fetching

Supported Exchanges:
    - US: NYSE, NASDAQ (9:30-16:00 ET)
    - Denmark: Nasdaq Copenhagen (9:00-17:00 CET)
    - Germany: Frankfurt/Xetra (9:00-17:30 CET)
    - UK: London Stock Exchange (8:00-16:30 GMT)
    - France: Euronext Paris (9:00-17:30 CET)
    - Sweden: Nasdaq Stockholm (9:00-17:30 CET)
    - And more...

Author: Stock Portfolio Builder
Last Modified: 2026
"""
import datetime
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo
from dateutil.relativedelta import relativedelta


# Market definitions with timezone and trading hours
MARKET_INFO = {
    'us': {
        'name': 'US Markets (NYSE/NASDAQ)',
        'timezone': 'America/New_York',
        'open_time': (9, 30),   # 9:30 AM
        'close_time': (16, 0),  # 4:00 PM
        'exchange_suffix': '',
    },
    'copenhagen': {
        'name': 'Nasdaq Copenhagen',
        'timezone': 'Europe/Copenhagen',
        'open_time': (9, 0),    # 9:00 AM
        'close_time': (17, 0),  # 5:00 PM
        'exchange_suffix': '.CO',
    },
    'stockholm': {
        'name': 'Nasdaq Stockholm',
        'timezone': 'Europe/Stockholm',
        'open_time': (9, 0),
        'close_time': (17, 30),
        'exchange_suffix': '.ST',
    },
    'helsinki': {
        'name': 'Nasdaq Helsinki',
        'timezone': 'Europe/Helsinki',
        'open_time': (10, 0),
        'close_time': (18, 30),
        'exchange_suffix': '.HE',
    },
    'oslo': {
        'name': 'Oslo Børs',
        'timezone': 'Europe/Oslo',
        'open_time': (9, 0),
        'close_time': (16, 20),
        'exchange_suffix': '.OL',
    },
    'frankfurt': {
        'name': 'Frankfurt Stock Exchange (Xetra)',
        'timezone': 'Europe/Berlin',
        'open_time': (9, 0),
        'close_time': (17, 30),
        'exchange_suffix': '.DE',
    },
    'paris': {
        'name': 'Euronext Paris',
        'timezone': 'Europe/Paris',
        'open_time': (9, 0),
        'close_time': (17, 30),
        'exchange_suffix': '.PA',
    },
    'london': {
        'name': 'London Stock Exchange',
        'timezone': 'Europe/London',
        'open_time': (8, 0),
        'close_time': (16, 30),
        'exchange_suffix': '.L',
    },
    'amsterdam': {
        'name': 'Euronext Amsterdam',
        'timezone': 'Europe/Amsterdam',
        'open_time': (9, 0),
        'close_time': (17, 30),
        'exchange_suffix': '.AS',
    },
    'madrid': {
        'name': 'Bolsa de Madrid',
        'timezone': 'Europe/Madrid',
        'open_time': (9, 0),
        'close_time': (17, 30),
        'exchange_suffix': '.MC',
    },
    'milan': {
        'name': 'Borsa Italiana',
        'timezone': 'Europe/Rome',
        'open_time': (9, 0),
        'close_time': (17, 30),
        'exchange_suffix': '.MI',
    },
    'zurich': {
        'name': 'SIX Swiss Exchange',
        'timezone': 'Europe/Zurich',
        'open_time': (9, 0),
        'close_time': (17, 30),
        'exchange_suffix': '.SW',
    },
}

# Common market holidays (fixed dates) - will be checked for each year
# Note: This is a simplified list. For production, use a proper holiday calendar library.
COMMON_HOLIDAYS = {
    'us': [
        (1, 1),    # New Year's Day
        (7, 4),    # Independence Day
        (12, 25),  # Christmas Day
        # Note: MLK Day, Presidents Day, Memorial Day, Labor Day, Thanksgiving are floating
    ],
    'copenhagen': [
        (1, 1),    # New Year's Day
        (12, 25),  # Christmas Day
        (12, 26),  # 2nd Christmas Day
        (12, 31),  # New Year's Eve (half day, but often closed)
    ],
    # Add more as needed
}


def get_exchange_from_ticker(ticker: str) -> str:
    """
    Determine the exchange from a ticker symbol based on its suffix.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'NOVO-B.CO', 'SAP.DE')
        
    Returns:
        Exchange key (e.g., 'us', 'copenhagen', 'frankfurt')
    """
    ticker = ticker.upper()
    
    # Check for market index symbols
    if ticker.startswith('^'):
        return 'us'  # Most market indices are US-based in yfinance
    
    # Check for exchange suffixes
    for exchange, info in MARKET_INFO.items():
        suffix = info['exchange_suffix']
        if suffix and ticker.endswith(suffix):
            return exchange
    
    # Default to US if no suffix
    return 'us'


def get_market_timezone(exchange: str) -> ZoneInfo:
    """
    Get the timezone for a specific exchange.
    
    Args:
        exchange: Exchange key (e.g., 'us', 'copenhagen')
        
    Returns:
        ZoneInfo object for the exchange's timezone
    """
    if exchange not in MARKET_INFO:
        exchange = 'us'  # Default to US
    return ZoneInfo(MARKET_INFO[exchange]['timezone'])


def is_market_open(exchange: str, check_time: Optional[datetime.datetime] = None) -> bool:
    """
    Check if a market is currently open.
    
    Args:
        exchange: Exchange key (e.g., 'us', 'copenhagen')
        check_time: Time to check (default: now). Should be timezone-aware or UTC.
        
    Returns:
        True if market is open, False otherwise
    """
    if exchange not in MARKET_INFO:
        exchange = 'us'
    
    market = MARKET_INFO[exchange]
    tz = ZoneInfo(market['timezone'])
    
    # Get current time in market's timezone
    if check_time is None:
        now = datetime.datetime.now(tz)
    else:
        now = check_time.astimezone(tz)
    
    # Check if weekend
    if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False
    
    # Check market hours
    open_hour, open_min = market['open_time']
    close_hour, close_min = market['close_time']
    
    market_open = now.replace(hour=open_hour, minute=open_min, second=0, microsecond=0)
    market_close = now.replace(hour=close_hour, minute=close_min, second=0, microsecond=0)
    
    return market_open <= now <= market_close


def has_market_closed_today(exchange: str, check_time: Optional[datetime.datetime] = None) -> bool:
    """
    Check if a market has already closed for today.
    
    Args:
        exchange: Exchange key (e.g., 'us', 'copenhagen')
        check_time: Time to check (default: now)
        
    Returns:
        True if market has closed today (data should be complete), False otherwise
    """
    if exchange not in MARKET_INFO:
        exchange = 'us'
    
    market = MARKET_INFO[exchange]
    tz = ZoneInfo(market['timezone'])
    
    if check_time is None:
        now = datetime.datetime.now(tz)
    else:
        now = check_time.astimezone(tz)
    
    # Weekend - no trading today
    if now.weekday() >= 5:
        return False  # Market didn't open today
    
    # Check if past closing time
    close_hour, close_min = market['close_time']
    market_close = now.replace(hour=close_hour, minute=close_min, second=0, microsecond=0)
    
    return now > market_close


def get_last_trading_day(exchange: str, reference_date: Optional[datetime.date] = None) -> datetime.date:
    """
    Get the last completed trading day for an exchange.
    
    This function returns the most recent date for which complete trading data
    should be available from the exchange.
    
    Args:
        exchange: Exchange key (e.g., 'us', 'copenhagen')
        reference_date: Reference date (default: today)
        
    Returns:
        Date of the last completed trading day
    """
    if exchange not in MARKET_INFO:
        exchange = 'us'
    
    market = MARKET_INFO[exchange]
    tz = ZoneInfo(market['timezone'])
    
    # Get current time in market's timezone
    now = datetime.datetime.now(tz)
    
    if reference_date is None:
        current_date = now.date()
    else:
        current_date = reference_date
    
    # Check if market has closed today
    if has_market_closed_today(exchange):
        last_trading_day = current_date
    else:
        # Market hasn't closed yet or didn't open today, go to previous day
        last_trading_day = current_date - datetime.timedelta(days=1)
    
    # Skip backwards over weekends
    while last_trading_day.weekday() >= 5:
        last_trading_day = last_trading_day - datetime.timedelta(days=1)
    
    return last_trading_day


def get_next_trading_day(exchange: str, reference_date: Optional[datetime.date] = None) -> datetime.date:
    """
    Get the next trading day for an exchange.
    
    Args:
        exchange: Exchange key (e.g., 'us', 'copenhagen')
        reference_date: Reference date (default: today)
        
    Returns:
        Date of the next trading day
    """
    if reference_date is None:
        reference_date = datetime.date.today()
    
    next_day = reference_date + datetime.timedelta(days=1)
    
    # Skip forward over weekends
    while next_day.weekday() >= 5:
        next_day = next_day + datetime.timedelta(days=1)
    
    return next_day


def should_fetch_new_data(
    last_db_date: datetime.date,
    ticker: str,
    check_time: Optional[datetime.datetime] = None
) -> Tuple[bool, datetime.date, str]:
    """
    Determine if new data should be fetched for a ticker.
    
    This is the main function to use before fetching data. It considers:
    - The last date in the database
    - The ticker's exchange timezone
    - Whether the market has closed
    - Weekends and trading days
    
    Args:
        last_db_date: Last date in the database for this ticker
        ticker: Stock ticker symbol
        check_time: Time to check (default: now)
        
    Returns:
        Tuple of (should_fetch, start_date, reason)
        - should_fetch: True if new data should be fetched
        - start_date: Date to start fetching from
        - reason: Human-readable explanation
    """
    exchange = get_exchange_from_ticker(ticker)
    last_trading_day = get_last_trading_day(exchange, check_time.date() if check_time else None)
    
    # If database is up to date with last trading day
    if last_db_date >= last_trading_day:
        return False, last_db_date, f"Data is up to date (last trading day: {last_trading_day})"
    
    # Calculate the next day to fetch from
    next_fetch_date = get_next_trading_day(exchange, last_db_date)
    
    # If next fetch date is in the future, nothing to fetch
    if next_fetch_date > last_trading_day:
        return False, last_db_date, f"Next trading day ({next_fetch_date}) is in the future"
    
    return True, next_fetch_date, f"Fetching data from {next_fetch_date} to {last_trading_day}"


def get_market_status_summary(tickers: List[str]) -> Dict[str, dict]:
    """
    Get a summary of market status for multiple tickers.
    
    Useful for determining when to run batch updates.
    
    Args:
        tickers: List of ticker symbols
        
    Returns:
        Dictionary with exchange status information
    """
    exchanges = set()
    for ticker in tickers:
        exchanges.add(get_exchange_from_ticker(ticker))
    
    summary = {}
    for exchange in exchanges:
        market = MARKET_INFO.get(exchange, MARKET_INFO['us'])
        tz = ZoneInfo(market['timezone'])
        now = datetime.datetime.now(tz)
        
        summary[exchange] = {
            'name': market['name'],
            'local_time': now.strftime('%Y-%m-%d %H:%M %Z'),
            'is_open': is_market_open(exchange),
            'has_closed_today': has_market_closed_today(exchange),
            'last_trading_day': get_last_trading_day(exchange),
            'next_trading_day': get_next_trading_day(exchange),
        }
    
    return summary


def get_optimal_fetch_time() -> str:
    """
    Suggest the optimal time to run data fetching for global markets.
    
    Returns:
        String with recommendation
    """
    # All major markets are closed after 22:00 UTC (US market closes at 21:00 UTC)
    return """
    Optimal fetch times (UTC):
    - 22:00-23:00 UTC: All major markets closed, complete daily data available
    - 06:00-08:00 UTC: Good for European markets only (US still has previous day)
    
    For mixed portfolios (US + Europe):
    - Run after 22:00 UTC to ensure all markets have complete data
    - Or run twice: Once at 18:00 UTC for Europe, once at 22:00 UTC for US
    """


# Convenience function for checking if today is a weekend
def is_weekend(date: Optional[datetime.date] = None) -> bool:
    """Check if a date is a weekend."""
    if date is None:
        date = datetime.date.today()
    return date.weekday() >= 5


# Demo/testing
if __name__ == "__main__":
    print("Market Hours Utility Demo")
    print("=" * 50)
    
    # Check some sample tickers
    test_tickers = ['AAPL', 'NOVO-B.CO', 'SAP.DE', '^GSPC']
    
    print("\nExchange detection:")
    for ticker in test_tickers:
        exchange = get_exchange_from_ticker(ticker)
        print(f"  {ticker}: {exchange} ({MARKET_INFO[exchange]['name']})")
    
    print("\nMarket status:")
    summary = get_market_status_summary(test_tickers)
    for exchange, info in summary.items():
        print(f"\n  {info['name']}:")
        print(f"    Local time: {info['local_time']}")
        print(f"    Is open: {info['is_open']}")
        print(f"    Has closed today: {info['has_closed_today']}")
        print(f"    Last trading day: {info['last_trading_day']}")
    
    print("\nOptimal fetch times:")
    print(get_optimal_fetch_time())
