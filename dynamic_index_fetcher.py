"""
Dynamic Index Fetcher Module

This module provides functionality to dynamically fetch stock symbols from various
stock exchanges and market indices around the world. It supports multiple data sources
and allows for easy extension to add new exchanges and indices.

Supported Indices:
    - Denmark: C25 (OMX Copenhagen 25)
    - USA: S&P 500, NASDAQ-100, Dow Jones 30
    - Europe: STOXX 600, DAX 40, CAC 40, FTSE 100, AEX 25, IBEX 35
    - Scandinavia: OMX Stockholm 30, OMX Helsinki 25

Data Sources:
    - Wikipedia (free, reliable for index constituents)
    - yfinance (for individual stock lookup and validation)
    - slickcharts.com (backup for S&P 500)

Features:
    - Multi-source data fetching with fallback options
    - Caching to reduce API calls
    - Automatic symbol suffix handling for international exchanges
    - Extensible architecture for adding new indices

Author: Stock Portfolio Builder
Last Modified: 2026
"""
import os
import json
import time
import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
import requests
from io import StringIO

# Exchange suffix mapping for yfinance compatibility
EXCHANGE_SUFFIXES = {
    'copenhagen': '.CO',
    'stockholm': '.ST',
    'helsinki': '.HE',
    'oslo': '.OL',
    'frankfurt': '.DE',
    'xetra': '.DE',
    'paris': '.PA',
    'london': '.L',
    'amsterdam': '.AS',
    'madrid': '.MC',
    'milan': '.MI',
    'zurich': '.SW',
    'brussels': '.BR',
    'vienna': '.VI',
    'lisbon': '.LS',
    'dublin': '.IR',
    'warsaw': '.WA',
    'prague': '.PR',
    'us': '',  # US stocks have no suffix
    'nyse': '',
    'nasdaq': '',
}

# Index configurations
INDEX_CONFIGS = {
    # Denmark
    'C25': {
        'name': 'OMX Copenhagen 25',
        'exchange': 'copenhagen',
        'wikipedia_url': 'https://en.wikipedia.org/wiki/OMX_Copenhagen_25',
        'table_match': 'Ticker',
        'ticker_column': 'Ticker',
        'fallback_symbols': [
            'NOVO-B.CO', 'MAERSK-B.CO', 'DSV.CO', 'NZYM-B.CO', 'VWS.CO',
            'CARL-B.CO', 'ORSTED.CO', 'COLO-B.CO', 'PNDORA.CO', 'DEMANT.CO',
            'GN.CO', 'GMAB.CO', 'TRYG.CO', 'RBREW.CO', 'DNORD.CO',
            'DANSKE.CO', 'JYSK.CO', 'ISS.CO', 'FLS.CO', 'CHR.CO',
            'AMBU-B.CO', 'NKT.CO', 'ROCK-B.CO', 'BAVA.CO', 'NETC.CO'
        ]
    },
    # USA
    'SP500': {
        'name': 'S&P 500',
        'exchange': 'us',
        'wikipedia_url': 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
        'table_match': 'Symbol',
        'ticker_column': 'Symbol',
        'slickcharts_url': 'https://www.slickcharts.com/sp500',
    },
    'NASDAQ100': {
        'name': 'NASDAQ-100',
        'exchange': 'us',
        'wikipedia_url': 'https://en.wikipedia.org/wiki/Nasdaq-100',
        'table_match': 'Ticker',
        'ticker_column': 'Ticker',
    },
    'DOW30': {
        'name': 'Dow Jones Industrial Average',
        'exchange': 'us',
        'wikipedia_url': 'https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average',
        'table_match': 'Symbol',
        'ticker_column': 'Symbol',
    },
    # Germany
    'DAX40': {
        'name': 'DAX 40',
        'exchange': 'frankfurt',
        'wikipedia_url': 'https://en.wikipedia.org/wiki/DAX',
        'table_match': 'Ticker',
        'ticker_column': 'Ticker symbol',
    },
    # France
    'CAC40': {
        'name': 'CAC 40',
        'exchange': 'paris',
        'wikipedia_url': 'https://en.wikipedia.org/wiki/CAC_40',
        'table_match': 'Ticker',
        'ticker_column': 'Ticker',
    },
    # UK
    'FTSE100': {
        'name': 'FTSE 100',
        'exchange': 'london',
        'wikipedia_url': 'https://en.wikipedia.org/wiki/FTSE_100_Index',
        'table_match': 'Ticker',
        'ticker_column': 'Ticker',
    },
    # Netherlands
    'AEX25': {
        'name': 'AEX 25',
        'exchange': 'amsterdam',
        'wikipedia_url': 'https://en.wikipedia.org/wiki/AEX_index',
        'table_match': 'Ticker',
        'ticker_column': 'Ticker',
    },
    # Sweden
    'OMX30': {
        'name': 'OMX Stockholm 30',
        'exchange': 'stockholm',
        'wikipedia_url': 'https://en.wikipedia.org/wiki/OMX_Stockholm_30',
        'table_match': 'Ticker',
        'ticker_column': 'Ticker',
    },
    # Spain
    'IBEX35': {
        'name': 'IBEX 35',
        'exchange': 'madrid',
        'wikipedia_url': 'https://en.wikipedia.org/wiki/IBEX_35',
        'table_match': 'Ticker',
        'ticker_column': 'Ticker',
    },
    # Finland
    'OMXH25': {
        'name': 'OMX Helsinki 25',
        'exchange': 'helsinki',
        'wikipedia_url': 'https://en.wikipedia.org/wiki/OMX_Helsinki_25',
        'table_match': 'Ticker',
        'ticker_column': 'Ticker',
    },
    # Switzerland
    'SMI': {
        'name': 'Swiss Market Index',
        'exchange': 'zurich',
        'wikipedia_url': 'https://en.wikipedia.org/wiki/Swiss_Market_Index',
        'table_match': 'Ticker',
        'ticker_column': 'Ticker',
    },
    # Italy
    'FTSEMIB': {
        'name': 'FTSE MIB',
        'exchange': 'milan',
        'wikipedia_url': 'https://en.wikipedia.org/wiki/FTSE_MIB',
        'table_match': 'Ticker',
        'ticker_column': 'Ticker',
    },
    # Belgium
    'BEL20': {
        'name': 'BEL 20',
        'exchange': 'brussels',
        'wikipedia_url': 'https://en.wikipedia.org/wiki/BEL_20',
        'table_match': 'Ticker',
        'ticker_column': 'Ticker',
    },
    # Austria
    'ATX': {
        'name': 'Austrian Traded Index',
        'exchange': 'vienna',
        'wikipedia_url': 'https://en.wikipedia.org/wiki/Austrian_Traded_Index',
        'table_match': 'Ticker',
        'ticker_column': 'Ticker',
    },
    # Norway
    'OBX': {
        'name': 'OBX Index',
        'exchange': 'oslo',
        'wikipedia_url': 'https://en.wikipedia.org/wiki/OBX_Index',
        'table_match': 'Ticker',
        'ticker_column': 'Ticker',
    },
    # Portugal
    'PSI20': {
        'name': 'PSI-20',
        'exchange': 'lisbon',
        'wikipedia_url': 'https://en.wikipedia.org/wiki/PSI-20',
        'table_match': 'Ticker',
        'ticker_column': 'Ticker',
    },
    # USA - Additional indices
    'SP400': {
        'name': 'S&P 400 Mid Cap',
        'exchange': 'us',
        'wikipedia_url': 'https://en.wikipedia.org/wiki/List_of_S%26P_400_companies',
        'table_match': 'Ticker',
        'ticker_column': 'Ticker symbol',
    },
    'SP600': {
        'name': 'S&P 600 Small Cap',
        'exchange': 'us',
        'wikipedia_url': 'https://en.wikipedia.org/wiki/List_of_S%26P_600_companies',
        'table_match': 'Ticker',
        'ticker_column': 'Ticker symbol',
    },
    'RUSSELL1000': {
        'name': 'Russell 1000',
        'exchange': 'us',
        'wikipedia_url': 'https://en.wikipedia.org/wiki/Russell_1000_Index',
        'table_match': 'Ticker',
        'ticker_column': 'Ticker',
    },
    # Pan-European
    'STOXX50': {
        'name': 'Euro Stoxx 50',
        'exchange': 'us',  # Mixed exchanges, symbols need individual handling
        'wikipedia_url': 'https://en.wikipedia.org/wiki/EURO_STOXX_50',
        'table_match': 'Ticker',
        'ticker_column': 'Ticker',
    },
    'STOXX600': {
        'name': 'STOXX Europe 600',
        'exchange': 'us',  # Mixed exchanges
        'wikipedia_url': 'https://en.wikipedia.org/wiki/STOXX_Europe_600',
        'table_match': 'Ticker',
        'ticker_column': 'Ticker',
    },
}

# Market indices that can be added to any symbol list
MARKET_INDICES = {
    'VIX': '^VIX',           # CBOE Volatility Index
    'SP500_INDEX': '^GSPC',   # S&P 500 Index
    'NASDAQ': '^IXIC',        # NASDAQ Composite
    'DOW': '^DJI',            # Dow Jones Industrial Average
    'STOXX50': '^STOXX50E',   # Euro Stoxx 50
    'FTSE': '^FTSE',          # FTSE 100 Index
    'DAX': '^GDAXI',          # DAX Index
    'CAC': '^FCHI',           # CAC 40 Index
}


class DynamicIndexFetcher:
    """
    A class to dynamically fetch stock symbols from various indices and exchanges.
    
    Attributes:
        cache_file (str): Path to the cache file for storing fetched symbols.
        cache_expiry_hours (int): Number of hours before cache expires.
        session (requests.Session): HTTP session for making requests.
    """
    
    def __init__(self, cache_file: str = "index_cache.json", cache_expiry_hours: int = 24):
        """
        Initialize the DynamicIndexFetcher.
        
        Args:
            cache_file: Path to the cache file.
            cache_expiry_hours: Hours before cache expires.
        """
        self.cache_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), cache_file)
        self.cache_expiry_hours = cache_expiry_hours
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self._cache = self._load_cache()
    
    def _load_cache(self) -> Dict:
        """Load cache from file if it exists."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}
    
    def _save_cache(self):
        """Save cache to file."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self._cache, f, indent=2, default=str)
        except IOError as e:
            print(f"Warning: Could not save cache: {e}")
    
    def _is_cache_valid(self, index_name: str) -> bool:
        """Check if cache is still valid for an index."""
        if index_name not in self._cache:
            return False
        
        cached_time = datetime.datetime.fromisoformat(self._cache[index_name].get('timestamp', '2000-01-01'))
        expiry_time = cached_time + datetime.timedelta(hours=self.cache_expiry_hours)
        return datetime.datetime.now() < expiry_time
    
    def _get_exchange_suffix(self, exchange: str) -> str:
        """Get the yfinance-compatible suffix for an exchange."""
        return EXCHANGE_SUFFIXES.get(exchange.lower(), '')
    
    def _clean_ticker(self, ticker: str, exchange: str) -> str:
        """Clean and format ticker symbol with appropriate exchange suffix."""
        if not ticker:
            return ''
        
        # Remove common unwanted characters and normalize
        ticker = ticker.strip().upper()
        
        # Handle tickers that already have exchange suffix (e.g., "NOVO B.CO")
        # We need to replace spaces before the suffix, not in the suffix
        for suffix in EXCHANGE_SUFFIXES.values():
            if suffix and ticker.endswith(suffix):
                # Extract the base ticker and clean it
                base = ticker[:-len(suffix)]
                base = base.strip().replace(' ', '-')
                return f"{base}{suffix}"
        
        # No suffix found, clean the ticker and add suffix if needed
        ticker = ticker.replace(' ', '-')
        
        if exchange != 'us':
            suffix = self._get_exchange_suffix(exchange)
            ticker = f"{ticker}{suffix}"
        
        return ticker
    
    def _fetch_wikipedia_table(self, url: str, table_match: str) -> Optional[pd.DataFrame]:
        """
        Fetch a table from Wikipedia that contains the specified column name.
        
        Args:
            url: Wikipedia page URL
            table_match: Column name to match for identifying the correct table
            
        Returns:
            DataFrame containing the table or None if not found
        """
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Parse all tables from the page
            tables = pd.read_html(StringIO(response.text))
            
            # Find the table containing the ticker column
            for table in tables:
                columns_lower = [str(col).lower() for col in table.columns]
                if any(table_match.lower() in col for col in columns_lower):
                    return table
            
            return None
        except Exception as e:
            print(f"Error fetching Wikipedia table from {url}: {e}")
            return None
    
    def fetch_index_constituents(self, index_name: str, use_cache: bool = True) -> List[str]:
        """
        Fetch constituents for a specific index.
        
        Args:
            index_name: Name of the index (e.g., 'C25', 'SP500')
            use_cache: Whether to use cached data if available
            
        Returns:
            List of ticker symbols
        """
        index_name = index_name.upper()
        
        if index_name not in INDEX_CONFIGS:
            raise ValueError(f"Unknown index: {index_name}. Available: {list(INDEX_CONFIGS.keys())}")
        
        # Check cache
        if use_cache and self._is_cache_valid(index_name):
            print(f"Using cached data for {index_name}")
            return self._cache[index_name]['symbols']
        
        config = INDEX_CONFIGS[index_name]
        symbols = []
        
        # Try Wikipedia first
        if 'wikipedia_url' in config:
            print(f"Fetching {config['name']} from Wikipedia...")
            table = self._fetch_wikipedia_table(config['wikipedia_url'], config['table_match'])
            
            if table is not None:
                # Find the ticker column (case-insensitive search)
                ticker_col = None
                for col in table.columns:
                    if config['ticker_column'].lower() in str(col).lower():
                        ticker_col = col
                        break
                
                if ticker_col:
                    raw_symbols = table[ticker_col].dropna().tolist()
                    symbols = [self._clean_ticker(str(s), config['exchange']) for s in raw_symbols if s]
                    symbols = [s for s in symbols if s]  # Remove empty strings
        
        # Use fallback if needed
        if not symbols and 'fallback_symbols' in config:
            print(f"Using fallback symbols for {index_name}")
            symbols = config['fallback_symbols']
        
        # Cache the results
        if symbols:
            self._cache[index_name] = {
                'symbols': symbols,
                'timestamp': datetime.datetime.now().isoformat(),
                'source': 'wikipedia' if symbols != config.get('fallback_symbols', []) else 'fallback'
            }
            self._save_cache()
        
        print(f"Fetched {len(symbols)} symbols for {index_name}")
        return symbols
    
    def fetch_multiple_indices(self, indices: List[str], use_cache: bool = True,
                               include_duplicates: bool = False) -> Dict[str, List[str]]:
        """
        Fetch constituents for multiple indices.
        
        Args:
            indices: List of index names
            use_cache: Whether to use cached data
            include_duplicates: Whether to include duplicate symbols across indices
            
        Returns:
            Dictionary mapping index names to their symbols
        """
        results = {}
        seen_symbols = set()
        
        for index in indices:
            try:
                symbols = self.fetch_index_constituents(index, use_cache)
                
                if not include_duplicates:
                    symbols = [s for s in symbols if s not in seen_symbols]
                    seen_symbols.update(symbols)
                
                results[index] = symbols
                time.sleep(1)  # Be respectful to servers
            except Exception as e:
                print(f"Error fetching {index}: {e}")
                results[index] = []
        
        return results
    
    def get_available_indices(self) -> Dict[str, str]:
        """Get a list of all available indices and their names."""
        return {key: config['name'] for key, config in INDEX_CONFIGS.items()}
    
    def get_market_indices(self) -> Dict[str, str]:
        """Get market index symbols (^VIX, ^GSPC, etc.)."""
        return MARKET_INDICES.copy()
    
    def validate_symbols(self, symbols: List[str]) -> Tuple[List[str], List[str]]:
        """
        Validate symbols using yfinance.
        
        Args:
            symbols: List of symbols to validate
            
        Returns:
            Tuple of (valid_symbols, invalid_symbols)
        """
        import yfinance as yf
        
        valid = []
        invalid = []
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                if info and info.get('regularMarketPrice') is not None:
                    valid.append(symbol)
                else:
                    invalid.append(symbol)
            except Exception:
                invalid.append(symbol)
            time.sleep(0.1)  # Rate limiting
        
        return valid, invalid


def dynamic_fetch_index_data(
    indices: List[str] = ['C25', 'SP500'],
    include_market_indices: bool = True,
    export_csv: bool = False,
    csv_filename: str = 'dynamic_symbol_list.csv',
    use_cache: bool = True,
    validate: bool = False
) -> pd.DataFrame:
    """
    Main function to dynamically fetch index data and return as DataFrame.
    
    This is the primary entry point for the module, designed to be compatible
    with the existing stock_data_fetch.py workflow.
    
    Args:
        indices: List of index names to fetch (e.g., ['C25', 'SP500', 'DAX40'])
        include_market_indices: Whether to include market indices (^VIX, ^GSPC, etc.)
        export_csv: Whether to export the symbols to a CSV file
        csv_filename: Name of the CSV file to export
        use_cache: Whether to use cached data
        validate: Whether to validate symbols with yfinance (slower but more reliable)
        
    Returns:
        DataFrame with columns: Symbol, Index, Exchange
        
    Example:
        >>> symbols_df = dynamic_fetch_index_data(
        ...     indices=['C25', 'SP500', 'DAX40'],
        ...     include_market_indices=True
        ... )
        >>> stock_tickers_list = symbols_df["Symbol"].tolist()
    """
    fetcher = DynamicIndexFetcher()
    
    # Fetch all requested indices
    index_data = fetcher.fetch_multiple_indices(indices, use_cache=use_cache)
    
    # Build DataFrame
    records = []
    
    for index_name, symbols in index_data.items():
        config = INDEX_CONFIGS.get(index_name, {})
        exchange = config.get('exchange', 'unknown')
        
        for symbol in symbols:
            records.append({
                'Symbol': symbol,
                'Index': index_name,
                'Exchange': exchange
            })
    
    # Add market indices if requested
    if include_market_indices:
        for name, symbol in MARKET_INDICES.items():
            records.append({
                'Symbol': symbol,
                'Index': 'MARKET_INDEX',
                'Exchange': 'us'
            })
    
    df = pd.DataFrame(records)
    
    # Remove duplicates based on Symbol
    df = df.drop_duplicates(subset=['Symbol'], keep='first')
    
    # Validate symbols if requested
    if validate:
        print("Validating symbols with yfinance...")
        valid_symbols, invalid_symbols = fetcher.validate_symbols(df['Symbol'].tolist())
        if invalid_symbols:
            print(f"Warning: {len(invalid_symbols)} invalid symbols found: {invalid_symbols[:10]}...")
        df = df[df['Symbol'].isin(valid_symbols)]
    
    # Export to CSV if requested
    if export_csv:
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), csv_filename)
        df.to_csv(csv_path, index=False)
        print(f"Exported {len(df)} symbols to {csv_path}")
    
    print(f"\nTotal symbols fetched: {len(df)}")
    return df


def add_custom_index(
    index_key: str,
    name: str,
    exchange: str,
    wikipedia_url: Optional[str] = None,
    table_match: str = 'Ticker',
    ticker_column: str = 'Ticker',
    fallback_symbols: Optional[List[str]] = None
):
    """
    Add a custom index configuration for future fetching.
    
    Args:
        index_key: Short key for the index (e.g., 'MIB40')
        name: Full name of the index
        exchange: Exchange key from EXCHANGE_SUFFIXES
        wikipedia_url: URL to Wikipedia page with constituents
        table_match: Column name to identify the correct table
        ticker_column: Column containing ticker symbols
        fallback_symbols: List of fallback symbols if Wikipedia fails
        
    Example:
        >>> add_custom_index(
        ...     'MIB40',
        ...     'FTSE MIB 40',
        ...     'milan',
        ...     'https://en.wikipedia.org/wiki/FTSE_MIB',
        ...     'Ticker',
        ...     'Ticker'
        ... )
    """
    INDEX_CONFIGS[index_key.upper()] = {
        'name': name,
        'exchange': exchange,
        'wikipedia_url': wikipedia_url,
        'table_match': table_match,
        'ticker_column': ticker_column,
        'fallback_symbols': fallback_symbols or []
    }
    print(f"Added custom index: {index_key}")


# Demo/testing code
if __name__ == "__main__":
    # Example usage
    print("Available indices:")
    fetcher = DynamicIndexFetcher()
    for key, name in fetcher.get_available_indices().items():
        print(f"  {key}: {name}")
    
    print("\n" + "="*50)
    print("Fetching C25 and S&P 500...")
    
    df = dynamic_fetch_index_data(
        indices=['C25', 'SP500'],
        include_market_indices=True,
        export_csv=True
    )
    
    print(f"\nSample of fetched data:")
    print(df.head(10))
    print(f"\nSymbols by index:")
    print(df.groupby('Index').size())
