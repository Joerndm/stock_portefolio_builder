"""
Stock Data Fetch Module

This module provides comprehensive functions for fetching, processing, and managing stock market data
using the yfinance library. It handles both price data and fundamental financial data for stocks and indices.

Main Features:
    - Import stock symbols from CSV files
    - Fetch real-time and historical stock price data
    - Calculate technical indicators (RSI, MACD, ATR, Bollinger Bands, Moving Averages)
    - Calculate volume-based indicators (Volume SMA/EMA, VWAP, OBV)
    - Calculate volatility metrics
    - Fetch and process fundamental financial data (income statements, balance sheets, cash flow)
    - Calculate financial ratios (P/E, P/B, P/S, P/FCF, ROE, ROA, etc.)
    - Calculate period returns (1D, 1M, 3M, 6M, 9M, 1Y, 2Y, 3Y, 4Y, 5Y)
    - Export/import data to/from Excel and CSV formats
    - Database integration for storing and retrieving stock data

Key Functions:
    - import_tickers_from_csv: Load stock symbols from CSV file
    - fetch_stock_standard_data: Get basic company information
    - fetch_stock_price_data: Retrieve historical price data
    - fetch_stock_financial_data: Get fundamental financial statements
    - calculate_period_returns: Compute returns over multiple time periods
    - add_technical_indicators: Add RSI, MACD, ATR indicators
    - add_volume_indicators: Add volume-based technical indicators
    - add_volatility_indicators: Calculate volatility metrics
    - calculate_moving_averages: Compute SMA and EMA (5, 20, 40, 120, 200 periods)
    - calculate_standard_diviation_value: Calculate rolling standard deviation
    - calculate_bollinger_bands: Compute Bollinger Bands
    - calculate_momentum: Calculate price momentum
    - calculate_ratios: Compute valuation ratios (P/E, P/B, P/S, P/FCF)
    - combine_stock_data: Merge price and financial data
    - export_to_excel: Export data to Excel file
    - import_excel: Import data from Excel file
    - convert_excel_to_csv: Convert Excel data to CSV format

Data Processing Pipeline:
    1. Import ticker symbols from CSV
    2. Fetch company info and price data
    3. Calculate technical indicators and returns
    4. Fetch fundamental financial data (for non-index securities)
    5. Calculate financial ratios
    6. Export to database or Excel/CSV

Dependencies:
    - yfinance: For fetching stock data from Yahoo Finance
    - pandas: For data manipulation and analysis
    - pandas_ta: For technical analysis indicators
    - dateutil: For date calculations
    - xlsxwriter: For Excel file export

Note:
    - All technical indicators are shifted by 1 period to avoid look-ahead bias
    - NaN values in calculated features are preserved for ML pipeline handling
    - Different calculation methods are used for banks, insurance, and biotechnology sectors
    - Index securities (e.g., ^VIX, ^GSPC) skip certain calculations like volatility of volatility

Author: Joern
Last Modified: 2026
"""
import os
import time
import datetime
import logging
import threading
from dateutil.relativedelta import relativedelta
import pandas as pd
import yfinance as yf
import pandas_ta as ta

# Suppress yfinance HTTP error noise (401 Invalid Crumb, etc.)
# Actual errors are still caught and handled by our retry logic
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

# Lock to serialize yf.download() calls — yfinance is NOT thread-safe.
# Concurrent calls can return data for the wrong ticker due to shared session state.
_yfinance_download_lock = threading.Lock()

from dynamic_index_fetcher import dynamic_fetch_index_data
from ttm_financial_calculator import (
    TTMFinancialCalculator,
    calculate_ratios_ttm_with_fallback,
    get_financial_data_with_ttm_preference
)

def import_tickers_from_csv(csv_file):
    """
    Imports stock symbols from a CSV file and returns a pandas DataFrame.

    The CSV file should have a column named 'Symbol' containing the stock symbols.

    Parameters:
    - csv_file (str): The path to the CSV file.

    Returns:
    pandas.DataFrame: A DataFrame containing the imported stock symbols.

    Raises:
    - ValueError: If the csv_file parameter is empty.
    - FileNotFoundError: If the CSV file does not exist.
    - KeyError: If the CSV file does not have a column named 'Symbol'.
    """
    # Check if the csv_file parameter is empty
    if csv_file == "":
        raise ValueError("The csv_file parameter cannot be empty.")

    try:
        # Read the CSV file from current position into a DataFrame
        my_path = os.path.abspath(__file__)
        path = os.path.dirname(my_path)
        import_location = os.path.join(path, csv_file)
        print(import_location)
        df = pd.read_csv(import_location)

        # Check if the 'Symbol' column exists in the DataFrame
        if 'Symbol' not in df.columns:
            raise KeyError("CSV file does not have a column named 'Symbol'.")

        # Return the DataFrame with stock symbols
        return df[['Symbol']]

    except FileNotFoundError as e:
        raise FileNotFoundError(f"CSV file '{csv_file}' does not exist.") from e

def fetch_stock_standard_data(stock_symbol = ""):
    """
    Fetches company information for the given stock symbols and returns a pandas DataFrame.
    
    The DataFrame will contain the stock symbol, company name, and industry.
    
    Parameters:
    - stock_symbols (list): A list of stock symbols.
    
    Returns:
    pandas.DataFrame: A DataFrame containing the fetched company information.
    
    Raises:
    - ValueError: If the stock_symbols parameter is empty.
    - KeyError: If any of the stock symbols are invalid or not found.
    - KeyError: If the stock_info cannot be transformed to a pandas DataFrame.
    """
    # Check if the stock_symbols parameter is empty
    if stock_symbol == "":
        raise ValueError("The stock_symbols parameter cannot be empty.")

    try:
        # Fetch the stock data for the symbol with retry for transient API errors
        symbol = stock_symbol
        stock_info = None
        last_error = None
        
        for attempt in range(3):
            try:
                stock_info = yf.Ticker(symbol).info
                if stock_info and stock_info.get("symbol"):
                    break
            except (KeyError, TypeError, Exception) as e:
                last_error = e
                if attempt < 2:
                    time.sleep(1 * (attempt + 1))
        
        if stock_info is None or not stock_info.get("symbol"):
            raise KeyError(f"Stock symbol '{symbol}' is invalid or not found."
                          + (f" Last error: {last_error}" if last_error else ""))
        
        if stock_info.get("typeDisp") == "Index":
            stock_info = {
                "ticker": stock_info["symbol"],
                "company_Name": stock_info.get("longName", symbol),
                "industry": "Index"
            }
        else:
            stock_info = {
                "ticker": stock_info["symbol"],
                "company_Name": stock_info.get("longName", symbol),
                "industry": stock_info.get("industry", "Unknown")
            }

    except KeyError as e:
        raise KeyError(f"Stock symbol '{symbol}' is invalid or not found.") from e

    try:
        # Create a DataFrame with the stock data
        stock_info_df = pd.DataFrame(
            stock_info,
            index=[0]
        )
        return stock_info_df

    except KeyError as e:
        raise KeyError("Could not transform stock_info to a pandas dataframe") from e

def fetch_stock_price_data(stock_ticker="", start_date=(datetime.datetime.now() - relativedelta(years=15))):
    """
    Fetches stock data using yfinance for the given stock symbols and returns a pandas DataFrame.

    The DataFrame will contain the date, stock name, stock symbol, opening price, currency, and trade_Volume.

    Parameters:
    - stock_tickers (list): A list of stock symbols.

    Returns:
    pandas.DataFrame: A DataFrame containing the fetched stock data.

    Raises:
    - ValueError: If the stock_tickers parameter is empty.
    - ValueError: If the start_date parameter is empty.
    - KeyError: If any of the stock symbols are invalid or not found.
    - KeyError: If the stock_price_data cannot be transformed to a pandas DataFrame.
    - KeyError: If the stock_info cannot be transformed to a pandas DataFrame.
    - KeyError: If the stock_price_data_df cannot be joined with stock_info_df.
    """
    # Check if the stock_tickers parameter is empty
    if stock_ticker == "":
        raise ValueError("The stock_tickers parameter cannot be empty.")

    # Check if the start_date parameter is empty
    if start_date == "":
        raise ValueError("The start_date parameter cannot be empty.")

    try:
        # Fetch the stock data for the ticker
        # Use lock to prevent yfinance thread-safety issues (shared session/crumb
        # can cause data for one ticker to be returned for another ticker's request)
        with _yfinance_download_lock:
            stock_price_data = yf.download(
                stock_ticker, start=start_date,
                auto_adjust=True, progress=False
            )

    except KeyError as e:
        raise KeyError(f"Stock ticker '{stock_ticker}' is invalid or not found.") from e

    try:
        stock_price_data_df = pd.DataFrame(stock_price_data)
        # Handle MultiIndex columns from yfinance (ticker level)
        if isinstance(stock_price_data_df.columns, pd.MultiIndex):
            # Validate: ensure yfinance returned data for the requested ticker
            ticker_levels = stock_price_data_df.columns.get_level_values(1).unique()
            if len(ticker_levels) == 1 and ticker_levels[0] != stock_ticker:
                raise ValueError(
                    f"yfinance returned data for '{ticker_levels[0]}' when "
                    f"'{stock_ticker}' was requested (thread-safety issue)"
                )
            stock_price_data_df = stock_price_data_df.droplevel(1, axis=1)
        # Deduplicate columns (parallel yfinance downloads can corrupt MultiIndex)
        stock_price_data_df = stock_price_data_df.loc[:, ~stock_price_data_df.columns.duplicated()]
        stock_price_data_df = stock_price_data_df.reset_index()
        stock_price_data_df = stock_price_data_df[["Date", "Open", "High", "Low", "Close", "Volume"]]
        stock_price_data_df = stock_price_data_df.rename(
                        columns={
                "Date": "date",
                "Open": "open_Price",
                "High": "high_Price",
                "Low": "low_Price",
                "Close": "close_Price",
                "Volume": "trade_Volume"
            }
        )
        stock_price_data_df = stock_price_data_df.rename_axis(None, axis=1)

    except KeyError as e:
        raise KeyError("Could not transform stock_price_data to a pandas dataframe") from e

    try:
        # Get ticker symbol and currency
        # Use the input parameter directly for the ticker name (avoids extra API call)
        # Try to get currency from yfinance .info, with retry and fallback
        ticker_symbol = stock_ticker
        currency = None
        
        for attempt in range(2):
            try:
                yf_info = yf.Ticker(stock_ticker).info
                if yf_info:
                    ticker_symbol = yf_info.get("symbol", stock_ticker)
                    currency = yf_info.get("currency")
                    if currency:
                        break
            except Exception:
                if attempt == 0:
                    time.sleep(1)  # Brief pause before retry
        
        # Fallback currency based on ticker suffix
        if not currency:
            suffix_currency_map = {
                '.CO': 'DKK', '.ST': 'SEK', '.HE': 'EUR', '.OL': 'NOK',
                '.L': 'GBp', '.PA': 'EUR', '.DE': 'EUR', '.AS': 'EUR',
                '.MI': 'EUR', '.MC': 'EUR', '.BR': 'EUR', '.VI': 'EUR',
                '.SW': 'CHF', '.TO': 'CAD', '.AX': 'AUD', '.HK': 'HKD',
                '.T': 'JPY', '.NS': 'INR', '.BO': 'INR',
            }
            for suffix, curr in suffix_currency_map.items():
                if stock_ticker.endswith(suffix):
                    currency = curr
                    break
            if not currency:
                currency = 'USD'  # Default for US tickers and unknowns
        
        stock_info = {
            "ticker": ticker_symbol,
            "currency": currency
        }

    except Exception as e:
        # Last resort: use input ticker and USD
        stock_info = {
            "ticker": stock_ticker,
            "currency": "USD"
        }

    try:
        # Create a DataFrame with the stock data
        stock_info_df = pd.DataFrame(
            stock_info,
            index=[0]
        )

    except KeyError as e:
        raise KeyError("Could not transform stock_info to a pandas dataframe") from e

    try:
        # Create a temporary DataFrame with the stock data joined with the stock_price_data_df and stock_info_df
        stock_price_data_df = stock_price_data_df.join(
            stock_info_df,
            how="cross"
        )
        # print("stock_price_data_df")
        # print(stock_price_data_df)
        return stock_price_data_df

    except KeyError as e:
        raise KeyError("Could not join stock_price_data_df with stock_info_df") from e

def calculate_period_returns(stock_price_data_df):
    """
    Calculates the period returns for the given stock data and returns a pandas DataFrame.

    The DataFrame will contain the date, stock name, stock symbol, and the period returns.

    Parameters:
    - stock_price_data_df (pandas.DataFrame): A DataFrame containing the stock data.

    Returns:
    pandas.DataFrame: A DataFrame containing the period returns.

    Raises:
    - ValueError: If the stock_price_data_df parameter is empty.
    - ValueError: If the stock_price_data_df parameter contains null values.
    - KeyError: If the periodic returns cannot be calculated from the specified DataFrame.
    - KeyError: If the rows cannot be shifted by 1.
    """
    # Check if the stock_price_data_df parameter is empty
    if stock_price_data_df.empty:
        raise ValueError("The stock_price_data_df parameter cannot be empty.")

    if stock_price_data_df["close_Price"].isnull().values.any():
        raise ValueError("""The stock_price_data_df parameter "close_Price" cannot contain null values.""")

    try:
        periods = {
            "1D": 1,
            "1M": 21,
            "3M": 63,
            "6M": 126,
            "9M": 189,
            "1Y": 252,
            "2Y": 504,
            "3Y": 756,
            "4Y": 1008,
            "5Y": 1260,
        }
        for label, period in periods.items():
            stock_price_data_df[label] = stock_price_data_df["close_Price"].pct_change(period)

    except KeyError as e:
        raise KeyError("Could not calculate periodic returns from specified dataframe.") from e

    try:
        # Shift the rows by 1 to avoid look-ahead bias
        # This ensures we predict TOMORROW'S return using TODAY's features
        stock_price_data_df[["1D", "1M", "3M", "6M", "9M", "1Y", "2Y", "3Y", "4Y", "5Y"]] = stock_price_data_df[["1D", "1M", "3M", "6M", "9M", "1Y", "2Y", "3Y", "4Y", "5Y"]].shift(periods=1)
        # Return the stock_price_data_df DataFrame
        return stock_price_data_df

    except KeyError as e:
        raise KeyError("Could not shift the rows by 1.") from e

def add_technical_indicators(stock_price_data_df):
    """
    Adds technical indicators (RSI, MACD, ATR) to the stock data DataFrame.

    Parameters:
    - stock_price_data_df (pd.DataFrame): DataFrame with stock price data. Must have columns: close_Price, high_Price, low_Price.

    Returns:
    - pd.DataFrame: DataFrame with added technical indicator columns.
    """
    # Check if the stock_price_data_df parameter is empty
    if stock_price_data_df.empty:
        raise ValueError("The stock_price_data_df parameter cannot be empty.")

    if stock_price_data_df["close_Price"].isnull().values.any():
        raise ValueError("""The stock_price_data_df parameter "close_Price" cannot contain null values.""")

    try:
        stock_price_data_df.ta.rsi(close="close_Price", length=14, append=True)  # RSI with default length of 14
        stock_price_data_df.ta.macd(close="close_Price", append=True)  # MACD with default configuration
        stock_price_data_df.ta.atr(high="high_Price", low="low_Price", close="close_Price", length=14, append=True)  # ATR with length 14
        # Don't drop all rows with any NaN - these will be handled by final dropna(subset=) in main pipeline
        # Technical indicators create expected NaN values at the beginning due to calculation windows
    
    except KeyError as e:
        raise KeyError("Could not calculate periodic returns from specified dataframe.") from e

    try:
        # Rename columns to match database schema
        rename_dict = {}
        if "MACDh_12_26_9" in stock_price_data_df.columns:
            rename_dict["MACDh_12_26_9"] = "macd_histogram"
        if "MACDs_12_26_9" in stock_price_data_df.columns:
            rename_dict["MACDs_12_26_9"] = "macd_signal"
        if "MACD_12_26_9" in stock_price_data_df.columns:
            rename_dict["MACD_12_26_9"] = "macd"
        
        if rename_dict:
            stock_price_data_df.rename(columns=rename_dict, inplace=True)
        
        # Only shift columns that exist in the dataframe
        columns_to_shift = []
        for col in ["RSI_14", "macd", "macd_histogram", "macd_signal", "ATRl_14", "ATRr_14", "ATR_14"]:
            if col in stock_price_data_df.columns:
                columns_to_shift.append(col)
        
        if columns_to_shift:
            stock_price_data_df[columns_to_shift] = stock_price_data_df[columns_to_shift].shift(periods=1)
        
        # Return the stock_price_data_df DataFrame
        return stock_price_data_df
    
    except KeyError as e:
        raise KeyError("Could not shift the rows by 1.") from e

def add_volume_indicators(stock_price_data_df):
    """
    Adds volume-based technical indicators to the stock data DataFrame.

    Parameters:
    - stock_price_data_df (pd.DataFrame): DataFrame with stock price data. Must have columns: close_Price, trade_Volume.

    Returns:
    - pd.DataFrame: DataFrame with added volume indicator columns (volume_sma_20, volume_ema_20, volume_ratio, vwap, obv).
    """
    # Check if the stock_price_data_df parameter is empty
    if stock_price_data_df.empty:
        raise ValueError("The stock_price_data_df parameter cannot be empty.")

    if stock_price_data_df["trade_Volume"].isnull().values.any():
        raise ValueError("""The stock_price_data_df parameter "trade_Volume" cannot contain null values.""")

    try:
        # Volume SMA (20-day)
        stock_price_data_df["volume_sma_20"] = stock_price_data_df["trade_Volume"].rolling(window=20).mean()
        
        # Volume EMA (20-day)
        stock_price_data_df["volume_ema_20"] = stock_price_data_df["trade_Volume"].ewm(span=20, adjust=False).mean()
        
        # Volume Ratio (current volume / 20-day SMA)
        stock_price_data_df["volume_ratio"] = stock_price_data_df["trade_Volume"] / stock_price_data_df["volume_sma_20"]
        
        # VWAP (Volume Weighted Average Price)
        # Calculate cumulative (price * volume) and cumulative volume
        stock_price_data_df["vwap"] = (stock_price_data_df["close_Price"] * stock_price_data_df["trade_Volume"]).cumsum() / stock_price_data_df["trade_Volume"].cumsum()
        
        # OBV (On-Balance Volume)
        # Calculate price change direction
        price_change = stock_price_data_df["close_Price"].diff()
        obv_direction = price_change.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        stock_price_data_df["obv"] = (stock_price_data_df["trade_Volume"] * obv_direction).cumsum()
        
        # Shift all volume indicators by 1 to avoid lookahead bias
        stock_price_data_df[["volume_sma_20", "volume_ema_20", "volume_ratio", "vwap", "obv"]] = stock_price_data_df[["volume_sma_20", "volume_ema_20", "volume_ratio", "vwap", "obv"]].shift(periods=1)
        
        return stock_price_data_df
    
    except KeyError as e:
        raise KeyError("Could not calculate volume indicators from specified dataframe.") from e

def add_volatility_indicators(stock_price_data_df):
    """
    Adds volatility indicators to the stock data DataFrame.

    Parameters:
    - stock_price_data_df (pd.DataFrame): DataFrame with stock price data. Must have column: close_Price.

    Returns:
    - pd.DataFrame: DataFrame with added volatility columns (volatility_5d, volatility_20d, volatility_60d).
    """
    # Check if the stock_price_data_df parameter is empty
    if stock_price_data_df.empty:
        raise ValueError("The stock_price_data_df parameter cannot be empty.")

    if stock_price_data_df["close_Price"].isnull().values.any():
        raise ValueError("""The stock_price_data_df parameter "close_Price" cannot contain null values.""")

    try:
        # Calculate returns for volatility calculation
        returns = stock_price_data_df["close_Price"].pct_change()
        
        # 5-day volatility (rolling standard deviation of returns)
        stock_price_data_df["volatility_5d"] = returns.rolling(window=5).std()
        
        # 20-day volatility
        stock_price_data_df["volatility_20d"] = returns.rolling(window=20).std()
        
        # 60-day volatility
        stock_price_data_df["volatility_60d"] = returns.rolling(window=60).std()
        
        # Shift all volatility indicators by 1 to avoid lookahead bias
        stock_price_data_df[["volatility_5d", "volatility_20d", "volatility_60d"]] = stock_price_data_df[["volatility_5d", "volatility_20d", "volatility_60d"]].shift(periods=1)
        
        return stock_price_data_df
    
    except KeyError as e:
        raise KeyError("Could not calculate volatility indicators from specified dataframe.") from e

def calculate_moving_averages(stock_price_data_df):
    """
    Calculates the moving averages for the given stock data and returns a pandas DataFrame.

    The DataFrame will contain the date, stock name, stock symbol, price, and the moving averages.
    Calculates SMA and EMA for periods: 5, 20, 40, 120, 200

    Parameters:
    - stock_price_data_df (pandas.DataFrame): A DataFrame containing the stock data.

    Returns:
    pandas.DataFrame: A DataFrame containing the moving averages.

    Raises:
    - ValueError: If the stock_price_data_df parameter is empty.
    - ValueError: If the stock_price_data_df parameter contains null values.
    - KeyError: If the moving averages cannot be calculated from the specified DataFrame.
    """
    # Check if the stock_price_data_df parameter is empty
    if stock_price_data_df.empty:
        raise ValueError("The stock_price_data_df parameter cannot be empty.")

    if stock_price_data_df["close_Price"].isnull().values.any():
        raise ValueError("""The stock_price_data_df parameter "close_Price" cannot contain null values.""")

    try:
        # Define all periods we need to calculate
        periods = [5, 20, 40, 120, 200]
        
        # Calculate Simple Moving Averages (SMA) using pandas-ta
        for period in periods:
            stock_price_data_df.ta.sma(close="close_Price", length=period, append=True)
        
        # Calculate Exponential Moving Averages (EMA) using pandas-ta
        for period in periods:
            stock_price_data_df.ta.ema(close="close_Price", length=period, append=True)
        
        # Remove any duplicate columns that pandas-ta might have created
        stock_price_data_df = stock_price_data_df.loc[:, ~stock_price_data_df.columns.duplicated()]
        
        # Rename columns to match database schema naming convention
        rename_map = {}
        for period in periods:
            # Only add to rename map if the columns exist
            if f'SMA_{period}' in stock_price_data_df.columns:
                rename_map[f'SMA_{period}'] = f'sma_{period}'
            if f'EMA_{period}' in stock_price_data_df.columns:
                rename_map[f'EMA_{period}'] = f'ema_{period}'
        
        stock_price_data_df.rename(columns=rename_map, inplace=True)
        
        # Shift all moving averages by 1 to avoid look-ahead bias
        # (using yesterday's MA to predict today)
        ma_columns = [f'sma_{p}' for p in periods if f'sma_{p}' in stock_price_data_df.columns] + \
                     [f'ema_{p}' for p in periods if f'ema_{p}' in stock_price_data_df.columns]
        stock_price_data_df[ma_columns] = stock_price_data_df[ma_columns].shift(1)
        
        print("Moving averages calculated successfully (periods: 5, 20, 40, 120, 200).")
        return stock_price_data_df

    except KeyError as e:
        raise KeyError("Could not calculate moving averages from specified DataFrame.") from e

def calculate_standard_diviation_value(stock_price_data_df):
    """
    Calculates the standard deviation of the stock price and returns a pandas DataFrame.

    The DataFrame will contain the date, stock name, stock symbol, price, and the standard deviation of the stock price.
    Calculates rolling standard deviation for periods: 5, 20, 40, 120, 200

    Parameters:
    - stock_price_data_df (pandas.DataFrame): A DataFrame containing the stock data.

    Returns:
    pandas.DataFrame: A DataFrame containing the standard deviation of the stock price.

    Raises:
    - ValueError: If the stock_price_data_df parameter is empty.
    - ValueError: If the stock_price_data_df parameter "close_Price" contains null values.
    - KeyError: If the standard deviation of the stock price cannot be calculated from the specified DataFrame.
    """
    # Checking if the combined_stock_price_data_df parameter is empty
    if stock_price_data_df.empty:
        raise ValueError("No stock data provided.")

    if stock_price_data_df["close_Price"].isnull().values.any():
        raise ValueError("""The stock_price_data_df parameter "close_Price" cannot contain null values.""")

    try:
        # Define all periods we need to calculate
        periods = [5, 20, 40, 120, 200]
        
        # Calculate rolling standard deviation using vectorized pandas operations
        # This is 10-100x faster than loop-based approach
        for period in periods:
            # Use rolling window to calculate std for each period
            # min_periods=period ensures we only calculate when we have enough data
            stock_price_data_df[f"std_Div_{period}"] = stock_price_data_df["close_Price"].rolling(
                window=period, 
                min_periods=period
            ).std()
        
        # Shift all std columns by 1 to avoid look-ahead bias
        std_columns = [f'std_Div_{p}' for p in periods]
        stock_price_data_df[std_columns] = stock_price_data_df[std_columns].shift(1)
        
        print(f"Standard deviation calculated successfully (periods: 5, 20, 40, 120, 200).")
        return stock_price_data_df

    except KeyError as e:
        raise KeyError("Could not calculate standard deviation of the stock price.") from e

def calculate_bollinger_bands(stock_price_data_df):
    """
    Calculates the Bollinger Bands for the given stock data and returns a pandas DataFrame.

    The DataFrame will contain the date, stock name, stock symbol, price, and the Bollinger Bands.
    Calculates Bollinger Bands (4*STD width) for periods: 5, 20, 40, 120, 200

    Parameters:
    - stock_price_data_df (pandas.DataFrame): A DataFrame containing the stock data.

    Returns:
    pandas.DataFrame: A DataFrame containing the Bollinger Bands.

    Raises:
    - ValueError: If the stock_price_data_df parameter is empty.
    - KeyError: If the Bollinger Bands cannot be calculated from the specified DataFrame.
    """
    # Checking if the stock_price_data_df parameter is empty
    if stock_price_data_df.empty:
        raise ValueError("No stock data provided.")

    # Note: We don't validate for NULL values here because:
    # 1. Rolling windows and shifts create expected NaNs at the beginning
    # 2. The final dropna() in the pipeline will clean these up
    # 3. Bollinger calculation handles NaN propagation naturally

    try:
        # Define all periods we need to calculate
        periods = [5, 20, 40, 120, 200]
        
        # Calculate the Bollinger Bands using vectorized operations
        # Bollinger Band width = (Upper Band - Lower Band) = (SMA + 2*STD) - (SMA - 2*STD) = 4*STD
        for period in periods:
            stock_price_data_df[f"bollinger_Band_{period}_2STD"] = 4.0 * stock_price_data_df[f"std_Div_{period}"]
        
        # Shift all Bollinger Band columns by 1 to avoid look-ahead bias
        bb_columns = [f'bollinger_Band_{p}_2STD' for p in periods]
        stock_price_data_df[bb_columns] = stock_price_data_df[bb_columns].shift(1)
        
        print("Bollinger Bands calculated successfully (periods: 5, 20, 40, 120, 200).")
        return stock_price_data_df

    except KeyError as e:
        raise KeyError("Could not calculate Bollinger Bands from specified DataFrame.") from e

def calculate_momentum(stock_price_data_df):
    """
    Calculates the momentum for the given stock data and returns a pandas DataFrame.

    The DataFrame will contain the date, stock name, stock symbol, price, and the momentum.

    Parameters:
    - stock_price_data_df (pandas.DataFrame): A DataFrame containing the stock data.

    Returns:
    pandas.DataFrame: A DataFrame containing the momentum.

    Raises:
    - ValueError: If the stock_price_data_df parameter is empty.
    - ValueError: If the stock_price_data_df parameter "close_Price" contains null values.
    - KeyError: If new columns cannot be created in the specified DataFrame.
    - KeyError: If the momentum cannot be calculated from the specified DataFrame.
    - KeyError: If the rows cannot be shifted by 1.
    """
    # Checking if the stock_price_data_df parameter is empty
    if stock_price_data_df.empty:
        raise ValueError("No stock data provided.")

    if stock_price_data_df.iloc[40:]["close_Price"].isnull().values.any():
        raise ValueError("""The stock_price_data_df parameter "close_Price" cannot contain null values.""")

    # Calculate the momentum for the given stock data
    try:
        # Create a new columns in stock_price_data_df called momentum
        stock_price_data_df["momentum"] = 0.0

    except KeyError as e:
        raise KeyError("Could not create new columns in the spicified dataframe") from e

    try:
        # Use vectorized operations to calculate momentum more efficiently
        stock_price_data_df["price_change"] = stock_price_data_df["close_Price"] - stock_price_data_df["close_Price"].shift(1)
        stock_price_data_df["momentum"] = 0.0
        
        # Create momentum tracking
        for i in range(1, len(stock_price_data_df)):
            prev_idx = stock_price_data_df.index[i-1]
            curr_idx = stock_price_data_df.index[i]
            
            if stock_price_data_df.loc[curr_idx, "price_change"] >= 0:  # Price went up or stayed same
                if stock_price_data_df.loc[prev_idx, "momentum"] <= 0:
                    stock_price_data_df.loc[curr_idx, "momentum"] = 1
                else:
                    stock_price_data_df.loc[curr_idx, "momentum"] = stock_price_data_df.loc[prev_idx, "momentum"] + 1
            else:  # Price went down
                if stock_price_data_df.loc[prev_idx, "momentum"] >= 0:
                    stock_price_data_df.loc[curr_idx, "momentum"] = -1
                else:
                    stock_price_data_df.loc[curr_idx, "momentum"] = stock_price_data_df.loc[prev_idx, "momentum"] - 1
                    
            # Create print statement per 250 rows processed
            if i % 250 == 0:
                print(f"Processed {i} rows, out of {len(stock_price_data_df)} rows.")
        
        # Drop the temporary price_change column
        stock_price_data_df = stock_price_data_df.drop(columns=["price_change"])
        
        print("Momentum calculated successfully.")

    except KeyError as e:
        raise KeyError("Could not calculate momentum from specified DataFrame.") from e

    try:
        stock_price_data_df["momentum"] = stock_price_data_df["momentum"].shift(1)
        # Return the stock_price_data_df DataFrame
        return stock_price_data_df

    except KeyError as e:
        raise KeyError("Could not shift the rows by 1.") from e


def _safe_growth(current, previous):
    """Calculate growth rate (current/previous - 1), returning NaN when previous is 0."""
    if previous == 0 or pd.isna(previous) or pd.isna(current):
        return float('nan')
    return (current / previous) - 1


# ─── Index-to-yfinance-symbol mapping for beta calculations ─────────
# Maps human-readable index codes (used in index_membership and UI)
# to the yfinance ticker symbols needed to download index price data.
INDEX_BETA_SYMBOLS = {
    'SP500': '^GSPC',
    'NASDAQ100': '^NDX',
    'DOW30': '^DJI',
    'C25': '^OMXC25',
    'DAX40': '^GDAXI',
    'CAC40': '^FCHI',
    'FTSE100': '^FTSE',
    'AEX25': '^AEX',
    'OMX30': '^OMX',
    'IBEX35': '^IBEX',
    'OMXH25': '^OMXH25',
    'STOXX600': '^STOXX',
    'STOXX50': '^STOXX50E',
    'SMI': '^SSMI',
    'FTSEMIB': 'FTSEMIB.MI',
    'BEL20': '^BFX',
    'ATX': '^ATX',
    'OBX': 'OBX.OL',
    'PSI20': '^PSI20',
}


def calculate_and_export_beta(stock_ticker, stock_price_df=None,
                               index_codes=None, years=5):
    """
    Calculate beta for a stock against one or more market indices and export to DB.
    
    Fetches index price history via yfinance, calculates rolling beta using
    technical_indicators.calculate_beta(), and exports results to stock_beta_data.
    
    Args:
        stock_ticker: Stock ticker symbol
        stock_price_df: Optional pre-loaded stock price DataFrame.
                        If None, fetched from the database.
        index_codes: List of index codes to compare against (e.g., ['SP500', 'C25']).
                     If None, uses all available indices.
        years: Years of history to use for beta calculation
        
    Returns:
        Dict mapping index_code → latest beta_252d value, or empty dict on failure
    """
    from technical_indicators import calculate_beta
    
    # Load stock prices if not provided
    if stock_price_df is None:
        try:
            import db_interactions
            stock_price_df = db_interactions.import_stock_price_data(
                amount=years * 252, stock_ticker=stock_ticker
            )
        except Exception as e:
            print(f"   ⚠️  Cannot load price data for {stock_ticker}: {e}")
            return {}
    
    if stock_price_df.empty or 'close_Price' not in stock_price_df.columns:
        return {}
    
    # Ensure date column
    if 'date' not in stock_price_df.columns and stock_price_df.index.name == 'date':
        stock_price_df = stock_price_df.reset_index()
    
    index_codes = index_codes or list(INDEX_BETA_SYMBOLS.keys())
    
    results = {}
    
    for idx_code in index_codes:
        yf_symbol = INDEX_BETA_SYMBOLS.get(idx_code)
        if not yf_symbol:
            continue
        
        try:
            # Fetch index price data
            start_date = datetime.datetime.now() - relativedelta(years=years)
            with _yfinance_download_lock:
                idx_data = yf.download(
                    yf_symbol, start=start_date,
                    auto_adjust=True, progress=False
                )
            
            if idx_data.empty:
                continue
            
            idx_df = pd.DataFrame(idx_data).reset_index()
            # Handle MultiIndex columns from yfinance
            if isinstance(idx_df.columns, pd.MultiIndex):
                idx_df = idx_df.droplevel(1, axis=1)
            idx_df = idx_df.loc[:, ~idx_df.columns.duplicated()]
            idx_df = idx_df.rename(columns={'Date': 'date', 'Close': 'close_Price'})
            
            if 'close_Price' not in idx_df.columns or idx_df.empty:
                continue
            
            # Calculate beta
            beta_df = calculate_beta(stock_price_df, idx_df)
            
            # Add metadata columns
            beta_df['ticker'] = stock_ticker
            beta_df['index_code'] = idx_code
            beta_df['index_symbol'] = yf_symbol
            
            # Drop rows where all beta values are NaN
            beta_cols = [c for c in beta_df.columns if c.startswith('beta_')]
            beta_df = beta_df.dropna(subset=beta_cols, how='all')
            
            if beta_df.empty:
                continue
            
            # Export to database
            try:
                import db_interactions
                db_interactions.export_stock_beta_data(beta_df)
            except Exception as e:
                print(f"   ⚠️  Could not export beta for {stock_ticker} vs {idx_code}: {e}")
            
            # Store latest beta for return
            latest = beta_df.iloc[-1]
            beta_252 = latest.get('beta_252d')
            if pd.notna(beta_252):
                results[idx_code] = float(beta_252)
                
        except Exception as e:
            print(f"   ⚠️  Beta calc failed for {stock_ticker} vs {idx_code}: {e}")
            continue
    
    if results:
        print(f"   ✓ Beta calculated for {stock_ticker} vs {list(results.keys())}")
    
    return results


def fetch_stock_financial_data(stock_symbol = ""):
    """
    Fetches financial stock data using yfinance and a list of stock symbols.

    The function fetches financial stock data using yfinance and a list of stock symbols and returns a DataFrame with the fetched data.

    Parameters:
    - stock_symbols (list): A list of stock symbols.

    Returns:
    - stock_data_df (pd.DataFrame): A DataFrame with the fetched data.

    Raises:
    - ValueError: If the stock_symbols parameter is empty.
    - KeyError: If any of the stock symbols in the list is invalid.
    """
    # Checking if the stock_symbols parameter is empty
    if stock_symbol == "":
        raise ValueError("No stock symbols provided.")

    try:
        symbol = stock_symbol
        # Fetching the financial stock data using yfinance (with retry for transient 401 errors)
        stock_data = yf.Ticker(symbol)
        stock_info = None
        for attempt in range(3):
            try:
                stock_info = stock_data.info
                if stock_info and stock_info.get("symbol"):
                    break
            except Exception:
                if attempt < 2:
                    time.sleep(1 * (attempt + 1))
                    stock_data = yf.Ticker(symbol)  # Fresh Ticker for retry
        if stock_info is None or not stock_info.get("symbol"):
            raise KeyError(f"Could not fetch financial info for '{symbol}'")
        if stock_info.get("typeDisp", "Equity") != "Index":
            print(f"length of income: {len(pd.DataFrame(stock_data.income_stmt).columns)}, length of balance: {len(pd.DataFrame(stock_data.balancesheet).columns)}, length of cashflow: {len(pd.DataFrame(stock_data.cashflow).columns)}")
            income_stmt = stock_data.income_stmt
            income_stmt_df = pd.DataFrame(income_stmt)
            if len(pd.DataFrame(stock_data.income_stmt).columns) > len(pd.DataFrame(stock_data.balancesheet).columns):
                column_amount = len(pd.DataFrame(stock_data.income_stmt).columns) - len(pd.DataFrame(stock_data.balancesheet).columns)
                income_stmt_df = income_stmt_df.drop(columns=income_stmt_df.columns[-column_amount])
            elif len(pd.DataFrame(stock_data.income_stmt).columns) > len(pd.DataFrame(stock_data.cashflow).columns):
                column_amount = len(pd.DataFrame(stock_data.income_stmt).columns) - len(pd.DataFrame(stock_data.cashflow).columns)
                income_stmt_df = income_stmt_df.drop(columns=income_stmt_df.columns[-column_amount])

            # Checking if the input DataFrame is empty
            if income_stmt_df.empty:
                raise ValueError("Input DataFrame is empty.")

            # Rotate the income_stmt_df dataframe
            income_stmt_df = income_stmt_df.transpose()
            income_stmt_df["Revenue growth"] = 0.0
            income_stmt_df["Gross Profit growth"] = 0.0
            income_stmt_df["Gross Margin"] = 0.0
            income_stmt_df["Gross Margin growth"] = 0.0
            income_stmt_df["Operating Earnings"] = 0.0
            income_stmt_df["Operating Margin"] = 0.0
            income_stmt_df["Operating Margin growth"] = 0.0
            income_stmt_df["Net Income growth"] = 0.0
            income_stmt_df["Net Income Margin"] = 0.0
            income_stmt_df["Net Income Margin growth"] = 0.0
            income_stmt_df["EPS"] = 0.0
            income_stmt_df["EPS growth"] = 0.0
            # Invert the rows in income_stmt_df dataframe
            income_stmt_df = income_stmt_df.iloc[::-1]
            # Reset the index of the income_stmt_df dataframe
            income_stmt_df = income_stmt_df.reset_index()
            # Rename the index column to Date
            income_stmt_df = income_stmt_df.rename(columns={"index": "Date"})
            # Fill the NaN values in the income_stmt_df dataframe with values from the previous row
            income_stmt_df = income_stmt_df.ffill()
            # Use stock_data to check if bank is part of the registered industry for the stock
            industry = stock_info.get("industry", "Unknown")
            # Broad financial-sector detection: banks, insurance, brokers,
            # asset managers, diversified financials, etc.
            _FINANCIAL_SECTOR_KEYWORDS = [
                "banks", "insurance", "capital markets", "financial services",
                "asset management", "diversified financial", "specialty finance",
                "investment banking", "brokerage", "credit services",
                "financial data", "financial exchange",
            ]
            _industry_lower = industry.lower()
            _is_financial_sector = any(kw in _industry_lower for kw in _FINANCIAL_SECTOR_KEYWORDS)
            # Check if bank is part of the registered industry for the stock
            if _is_financial_sector:
                # Check is "Gross Profit" is in the income_stmt_df dataframe
                if "Gross Profit" not in income_stmt_df.columns:
                    if "Operating Income" not in income_stmt_df.columns:
                        for index, row in income_stmt_df.iterrows():
                            income_stmt_df.loc[index, "Net Income Margin"] = income_stmt_df.loc[index, "Net Income Common Stockholders"] / income_stmt_df.loc[index, "Total Revenue"]
                            income_stmt_df.loc[index, "EPS"] = income_stmt_df.loc[index, "Net Income Common Stockholders"] / income_stmt_df.loc[index, "Diluted Average Shares"]
                            if index == 0:
                                income_stmt_df.loc[index, "Revenue growth"] = 0.0
                                income_stmt_df.loc[index, "Net Income growth"] = 0.0
                                income_stmt_df.loc[index, "Net Income Margin growth"] = 0.0
                                income_stmt_df.loc[index, "EPS growth"] = 0.0
                            else:
                                income_stmt_df.loc[index, "Revenue growth"] = _safe_growth(income_stmt_df.iloc[index]["Total Revenue"], income_stmt_df.iloc[index-1]["Total Revenue"])
                                income_stmt_df.loc[index, "Net Income growth"] = _safe_growth(income_stmt_df.iloc[index]["Net Income Common Stockholders"], income_stmt_df.iloc[index-1]["Net Income Common Stockholders"])
                                income_stmt_df.loc[index, "Net Income Margin growth"] = _safe_growth(income_stmt_df.iloc[index]["Net Income Margin"], income_stmt_df.iloc[index-1]["Net Income Margin"])
                                income_stmt_df.loc[index, "EPS growth"] = _safe_growth(income_stmt_df.iloc[index]["EPS"], income_stmt_df.iloc[index-1]["EPS"])
                    else:
                        for index, row in income_stmt_df.iterrows():
                            income_stmt_df.loc[index, "Operating Margin"] = income_stmt_df.loc[index, "Operating Income"] / income_stmt_df.loc[index, "Total Revenue"]
                            income_stmt_df.loc[index, "Net Income Margin"] = income_stmt_df.loc[index, "Net Income Common Stockholders"] / income_stmt_df.loc[index, "Total Revenue"]
                            income_stmt_df.loc[index, "EPS"] = income_stmt_df.loc[index, "Net Income Common Stockholders"] / income_stmt_df.loc[index, "Diluted Average Shares"]
                            if index == 0:
                                income_stmt_df.loc[index, "Revenue growth"] = 0.0
                                income_stmt_df.loc[index, "Operating Earnings growth"] = 0.0
                                income_stmt_df.loc[index, "Operating Margin growth"] = 0.0
                                income_stmt_df.loc[index, "Net Income growth"] = 0.0
                                income_stmt_df.loc[index, "Net Income Margin growth"] = 0.0
                                income_stmt_df.loc[index, "EPS growth"] = 0.0
                            else:
                                income_stmt_df.loc[index, "Revenue growth"] = _safe_growth(income_stmt_df.iloc[index]["Total Revenue"], income_stmt_df.iloc[index-1]["Total Revenue"])
                                income_stmt_df.loc[index, "Operating Earnings growth"] = _safe_growth(income_stmt_df.iloc[index]["Operating Income"], income_stmt_df.iloc[index-1]["Operating Income"])
                                income_stmt_df.loc[index, "Operating Margin growth"] = _safe_growth(income_stmt_df.iloc[index]["Operating Margin"], income_stmt_df.iloc[index-1]["Operating Margin"])
                                income_stmt_df.loc[index, "Net Income growth"] = _safe_growth(income_stmt_df.iloc[index]["Net Income Common Stockholders"], income_stmt_df.iloc[index-1]["Net Income Common Stockholders"])
                                income_stmt_df.loc[index, "Net Income Margin growth"] = _safe_growth(income_stmt_df.iloc[index]["Net Income Margin"], income_stmt_df.iloc[index-1]["Net Income Margin"])
                                income_stmt_df.loc[index, "EPS growth"] = _safe_growth(income_stmt_df.iloc[index]["EPS"], income_stmt_df.iloc[index-1]["EPS"])
                else:
                    for index, row in income_stmt_df.iterrows():
                        income_stmt_df.loc[index, "Gross Margin"] = income_stmt_df.loc[index, "Gross Profit"] / income_stmt_df.loc[index, "Total Revenue"]
                        income_stmt_df.loc[index, "Operating Margin"] = income_stmt_df.loc[index, "Operating Income"] / income_stmt_df.loc[index, "Total Revenue"]
                        income_stmt_df.loc[index, "Net Income Margin"] = income_stmt_df.loc[index, "Net Income Common Stockholders"] / income_stmt_df.loc[index, "Total Revenue"]
                        income_stmt_df.loc[index, "EPS"] = income_stmt_df.loc[index, "Net Income Common Stockholders"] / income_stmt_df.loc[index, "Diluted Average Shares"]
                        if index == 0:
                            income_stmt_df.loc[index, "Revenue growth"] = 0.0
                            income_stmt_df.loc[index, "Gross Profit growth"] = 0.0
                            income_stmt_df.loc[index, "Gross Margin growth"] = 0.0
                            income_stmt_df.loc[index, "Operating Earnings growth"] = 0.0
                            income_stmt_df.loc[index, "Operating Margin growth"] = 0.0
                            income_stmt_df.loc[index, "Net Income growth"] = 0.0
                            income_stmt_df.loc[index, "Net Income Margin growth"] = 0.0
                        else:
                            income_stmt_df.loc[index, "Revenue growth"] = _safe_growth(income_stmt_df.iloc[index]["Total Revenue"], income_stmt_df.iloc[index-1]["Total Revenue"])
                            income_stmt_df.loc[index, "Gross Profit growth"] = _safe_growth(income_stmt_df.iloc[index]["Gross Profit"], income_stmt_df.iloc[index-1]["Gross Profit"])
                            income_stmt_df.loc[index, "Gross Margin growth"] = _safe_growth(income_stmt_df.iloc[index]["Gross Margin"], income_stmt_df.iloc[index-1]["Gross Margin"])
                            income_stmt_df.loc[index, "Operating Earnings growth"] = _safe_growth(income_stmt_df.iloc[index]["Operating Income"], income_stmt_df.iloc[index-1]["Operating Income"])
                            income_stmt_df.loc[index, "Operating Margin growth"] = _safe_growth(income_stmt_df.iloc[index]["Operating Margin"], income_stmt_df.iloc[index-1]["Operating Margin"])
                            income_stmt_df.loc[index, "Net Income growth"] = _safe_growth(income_stmt_df.iloc[index]["Net Income Common Stockholders"], income_stmt_df.iloc[index-1]["Net Income Common Stockholders"])
                            income_stmt_df.loc[index, "Net Income Margin growth"] = _safe_growth(income_stmt_df.iloc[index]["Net Income Margin"], income_stmt_df.iloc[index-1]["Net Income Margin"])
            elif "biotechnology" in industry.lower():
                # Check is "Gross Profit" is in the income_stmt_df dataframe
                if "Gross Profit" not in income_stmt_df.columns:
                    for index, row in income_stmt_df.iterrows():
                        income_stmt_df.loc[index, "Operating Margin"] = income_stmt_df.loc[index, "Operating Income"] / income_stmt_df.loc[index, "Total Revenue"]
                        income_stmt_df.loc[index, "Net Income Margin"] = income_stmt_df.loc[index, "Net Income Common Stockholders"] / income_stmt_df.loc[index, "Total Revenue"]
                        income_stmt_df.loc[index, "EPS"] = income_stmt_df.loc[index, "Net Income Common Stockholders"] / income_stmt_df.loc[index, "Diluted Average Shares"]
                        if index == 0:
                            income_stmt_df.loc[index, "Revenue growth"] = 0.0
                            income_stmt_df.loc[index, "Operating Earnings growth"] = 0.0
                            income_stmt_df.loc[index, "Operating Margin growth"] = 0.0
                            income_stmt_df.loc[index, "Net Income growth"] = 0.0
                            income_stmt_df.loc[index, "Net Income Margin growth"] = 0.0
                            income_stmt_df.loc[index, "EPS growth"] = 0.0
                        else:
                            income_stmt_df.loc[index, "Revenue growth"] = _safe_growth(income_stmt_df.iloc[index]["Total Revenue"], income_stmt_df.iloc[index-1]["Total Revenue"])
                            income_stmt_df.loc[index, "Operating Earnings growth"] = _safe_growth(income_stmt_df.iloc[index]["Operating Income"], income_stmt_df.iloc[index-1]["Operating Income"])
                            income_stmt_df.loc[index, "Operating Margin growth"] = _safe_growth(income_stmt_df.iloc[index]["Operating Margin"], income_stmt_df.iloc[index-1]["Operating Margin"])
                            income_stmt_df.loc[index, "Net Income growth"] = _safe_growth(income_stmt_df.iloc[index]["Net Income Common Stockholders"], income_stmt_df.iloc[index-1]["Net Income Common Stockholders"])
                            income_stmt_df.loc[index, "Net Income Margin growth"] = _safe_growth(income_stmt_df.iloc[index]["Net Income Margin"], income_stmt_df.iloc[index-1]["Net Income Margin"])
                            income_stmt_df.loc[index, "EPS growth"] = _safe_growth(income_stmt_df.iloc[index]["EPS"], income_stmt_df.iloc[index-1]["EPS"])
                else:
                    for index, row in income_stmt_df.iterrows():
                        income_stmt_df.loc[index, "Gross Margin"] = income_stmt_df.loc[index, "Gross Profit"] / income_stmt_df.loc[index, "Total Revenue"]
                        income_stmt_df.loc[index, "Operating Margin"] = income_stmt_df.loc[index, "Operating Income"] / income_stmt_df.loc[index, "Total Revenue"]
                        income_stmt_df.loc[index, "Net Income Margin"] = income_stmt_df.loc[index, "Net Income Common Stockholders"] / income_stmt_df.loc[index, "Total Revenue"]
                        income_stmt_df.loc[index, "EPS"] = income_stmt_df.loc[index, "Net Income Common Stockholders"] / income_stmt_df.loc[index, "Diluted Average Shares"]
                        if index == 0:
                            income_stmt_df.loc[index, "Revenue growth"] = 0.0
                            income_stmt_df.loc[index, "Gross Profit growth"] = 0.0
                            income_stmt_df.loc[index, "Gross Margin growth"] = 0.0
                            income_stmt_df.loc[index, "Operating Earnings growth"] = 0.0
                            income_stmt_df.loc[index, "Operating Margin growth"] = 0.0
                            income_stmt_df.loc[index, "Net Income growth"] = 0.0
                            income_stmt_df.loc[index, "Net Income Margin growth"] = 0.0
                            income_stmt_df.loc[index, "EPS growth"] = 0.0
                        else:
                            income_stmt_df.loc[index, "Revenue growth"] = _safe_growth(income_stmt_df.iloc[index]["Total Revenue"], income_stmt_df.iloc[index-1]["Total Revenue"])
                            income_stmt_df.loc[index, "Gross Profit growth"] = _safe_growth(income_stmt_df.iloc[index]["Gross Profit"], income_stmt_df.iloc[index-1]["Gross Profit"])
                            income_stmt_df.loc[index, "Gross Margin growth"] = _safe_growth(income_stmt_df.iloc[index]["Gross Margin"], income_stmt_df.iloc[index-1]["Gross Margin"])
                            income_stmt_df.loc[index, "Operating Earnings growth"] = _safe_growth(income_stmt_df.iloc[index]["Operating Income"], income_stmt_df.iloc[index-1]["Operating Income"])
                            income_stmt_df.loc[index, "Operating Margin growth"] = _safe_growth(income_stmt_df.iloc[index]["Operating Margin"], income_stmt_df.iloc[index-1]["Operating Margin"])
                            income_stmt_df.loc[index, "Net Income growth"] = _safe_growth(income_stmt_df.iloc[index]["Net Income Common Stockholders"], income_stmt_df.iloc[index-1]["Net Income Common Stockholders"])
                            income_stmt_df.loc[index, "Net Income Margin growth"] = _safe_growth(income_stmt_df.iloc[index]["Net Income Margin"], income_stmt_df.iloc[index-1]["Net Income Margin"])
                            income_stmt_df.loc[index, "EPS growth"] = _safe_growth(income_stmt_df.iloc[index]["EPS"], income_stmt_df.iloc[index-1]["EPS"])
            else:
                if "Gross Profit" not in income_stmt_df.columns:
                    if "Operating Income" not in income_stmt_df.columns:
                        for index, row in income_stmt_df.iterrows():
                            income_stmt_df.loc[index, "Net Income Margin"] = income_stmt_df.loc[index, "Net Income Common Stockholders"] / income_stmt_df.loc[index, "Total Revenue"]
                            income_stmt_df.loc[index, "EPS"] = income_stmt_df.loc[index, "Net Income Common Stockholders"] / income_stmt_df.loc[index, "Diluted Average Shares"]
                            if index == 0:
                                income_stmt_df.loc[index, "Revenue growth"] = 0.0
                                income_stmt_df.loc[index, "Net Income growth"] = 0.0
                                income_stmt_df.loc[index, "Net Income Margin growth"] = 0.0
                                income_stmt_df.loc[index, "EPS growth"] = 0.0
                            else:
                                income_stmt_df.loc[index, "Revenue growth"] = _safe_growth(income_stmt_df.iloc[index]["Total Revenue"], income_stmt_df.iloc[index-1]["Total Revenue"])
                                income_stmt_df.loc[index, "Net Income growth"] = _safe_growth(income_stmt_df.iloc[index]["Net Income Common Stockholders"], income_stmt_df.iloc[index-1]["Net Income Common Stockholders"])
                                income_stmt_df.loc[index, "Net Income Margin growth"] = _safe_growth(income_stmt_df.iloc[index]["Net Income Margin"], income_stmt_df.iloc[index-1]["Net Income Margin"])
                                income_stmt_df.loc[index, "EPS growth"] = _safe_growth(income_stmt_df.iloc[index]["EPS"], income_stmt_df.iloc[index-1]["EPS"])
                    else:
                        for index, row in income_stmt_df.iterrows():
                            income_stmt_df.loc[index, "Operating Margin"] = income_stmt_df.loc[index, "Operating Income"] / income_stmt_df.loc[index, "Total Revenue"]
                            income_stmt_df.loc[index, "Net Income Margin"] = income_stmt_df.loc[index, "Net Income Common Stockholders"] / income_stmt_df.loc[index, "Total Revenue"]
                            income_stmt_df.loc[index, "EPS"] = income_stmt_df.loc[index, "Net Income Common Stockholders"] / income_stmt_df.loc[index, "Diluted Average Shares"]
                            if index == 0:
                                income_stmt_df.loc[index, "Revenue growth"] = 0.0
                                income_stmt_df.loc[index, "Operating Earnings growth"] = 0.0
                                income_stmt_df.loc[index, "Operating Margin growth"] = 0.0
                                income_stmt_df.loc[index, "Net Income growth"] = 0.0
                                income_stmt_df.loc[index, "Net Income Margin growth"] = 0.0
                                income_stmt_df.loc[index, "EPS growth"] = 0.0
                            else:
                                income_stmt_df.loc[index, "Revenue growth"] = _safe_growth(income_stmt_df.iloc[index]["Total Revenue"], income_stmt_df.iloc[index-1]["Total Revenue"])
                                income_stmt_df.loc[index, "Operating Earnings growth"] = _safe_growth(income_stmt_df.iloc[index]["Operating Income"], income_stmt_df.iloc[index-1]["Operating Income"])
                                income_stmt_df.loc[index, "Operating Margin growth"] = _safe_growth(income_stmt_df.iloc[index]["Operating Margin"], income_stmt_df.iloc[index-1]["Operating Margin"])
                                income_stmt_df.loc[index, "Net Income growth"] = _safe_growth(income_stmt_df.iloc[index]["Net Income Common Stockholders"], income_stmt_df.iloc[index-1]["Net Income Common Stockholders"])
                                income_stmt_df.loc[index, "Net Income Margin growth"] = _safe_growth(income_stmt_df.iloc[index]["Net Income Margin"], income_stmt_df.iloc[index-1]["Net Income Margin"])
                                income_stmt_df.loc[index, "EPS growth"] = _safe_growth(income_stmt_df.iloc[index]["EPS"], income_stmt_df.iloc[index-1]["EPS"])
                else:
                    for index, row in income_stmt_df.iterrows():
                        income_stmt_df.loc[index, "Gross Margin"] = income_stmt_df.loc[index, "Gross Profit"] / income_stmt_df.loc[index, "Total Revenue"]
                        income_stmt_df.loc[index, "Operating Margin"] = income_stmt_df.loc[index, "Operating Income"] / income_stmt_df.loc[index, "Total Revenue"]
                        income_stmt_df.loc[index, "Net Income Margin"] = income_stmt_df.loc[index, "Net Income Common Stockholders"] / income_stmt_df.loc[index, "Total Revenue"]
                        income_stmt_df.loc[index, "EPS"] = income_stmt_df.loc[index, "Net Income Common Stockholders"] / income_stmt_df.loc[index, "Diluted Average Shares"]
                        if index == 0:
                            income_stmt_df.loc[index, "Revenue growth"] = 0.0
                            income_stmt_df.loc[index, "Gross Profit growth"] = 0.0
                            income_stmt_df.loc[index, "Gross Margin growth"] = 0.0
                            income_stmt_df.loc[index, "Operating Earnings growth"] = 0.0
                            income_stmt_df.loc[index, "Operating Margin growth"] = 0.0
                            income_stmt_df.loc[index, "Net Income growth"] = 0.0
                            income_stmt_df.loc[index, "Net Income Margin growth"] = 0.0
                            income_stmt_df.loc[index, "EPS growth"] = 0.0
                        else:
                            income_stmt_df.loc[index, "Revenue growth"] = _safe_growth(income_stmt_df.iloc[index]["Total Revenue"], income_stmt_df.iloc[index-1]["Total Revenue"])
                            income_stmt_df.loc[index, "Gross Profit growth"] = _safe_growth(income_stmt_df.iloc[index]["Gross Profit"], income_stmt_df.iloc[index-1]["Gross Profit"])
                            income_stmt_df.loc[index, "Gross Margin growth"] = _safe_growth(income_stmt_df.iloc[index]["Gross Margin"], income_stmt_df.iloc[index-1]["Gross Margin"])
                            income_stmt_df.loc[index, "Operating Earnings growth"] = _safe_growth(income_stmt_df.iloc[index]["Operating Income"], income_stmt_df.iloc[index-1]["Operating Income"])
                            income_stmt_df.loc[index, "Operating Margin growth"] = _safe_growth(income_stmt_df.iloc[index]["Operating Margin"], income_stmt_df.iloc[index-1]["Operating Margin"])
                            income_stmt_df.loc[index, "Net Income growth"] = _safe_growth(income_stmt_df.iloc[index]["Net Income Common Stockholders"], income_stmt_df.iloc[index-1]["Net Income Common Stockholders"])
                            income_stmt_df.loc[index, "Net Income Margin growth"] = _safe_growth(income_stmt_df.iloc[index]["Net Income Margin"], income_stmt_df.iloc[index-1]["Net Income Margin"])
                            income_stmt_df.loc[index, "EPS growth"] = _safe_growth(income_stmt_df.iloc[index]["EPS"], income_stmt_df.iloc[index-1]["EPS"])

            balancesheet = stock_data.balancesheet
            balancesheet_df = pd.DataFrame(balancesheet)
            if len(pd.DataFrame(stock_data.balancesheet).columns) > len(pd.DataFrame(stock_data.income_stmt).columns):
                column_amount = len(pd.DataFrame(stock_data.balancesheet).columns) - len(pd.DataFrame(stock_data.income_stmt).columns)
                balancesheet_df = balancesheet_df.drop(columns=balancesheet_df.columns[-column_amount])
            elif len(pd.DataFrame(stock_data.balancesheet).columns) > len(pd.DataFrame(stock_data.cashflow).columns):
                column_amount = len(pd.DataFrame(stock_data.balancesheet).columns) - len(pd.DataFrame(stock_data.cashflow).columns)
                balancesheet_df = balancesheet_df.drop(columns=balancesheet_df.columns[-column_amount])

            # Checking if the input DataFrame is empty
            if balancesheet_df.empty:
                raise ValueError("Input DataFrame is empty.")

            # Rotate the balancesheet_df dataframe
            balancesheet_df = balancesheet_df.transpose()
            balancesheet_df["Total Assets growth"] = 0.0
            balancesheet_df["Current Assets growth"] = 0.0
            balancesheet_df["Cash and Cash Equivalents growth"] = 0.0
            balancesheet_df["Total Liabilities growth"] = 0.0
            balancesheet_df["Total Equity growth"] = 0.0
            balancesheet_df["Current Liabilities growth"] = 0.0
            balancesheet_df["Book Value"] = 0.0
            balancesheet_df["Book Value growth"] = 0.0
            balancesheet_df["Book Value per share"] = 0.0
            balancesheet_df["Book Value per share growth"] = 0.0
            balancesheet_df["Return on Assets"] = 0.0
            balancesheet_df["Return on Assets growth"] = 0.0
            balancesheet_df["Return on Equity"] = 0.0
            balancesheet_df["Return on Equity growth"] = 0.0
            balancesheet_df["Return on Invested Capital"] = 0.0
            balancesheet_df["Return on Invested Capital growth"] = 0.0
            balancesheet_df["Current Ratio"] = 0.0
            balancesheet_df["Current Ratio growth"] = 0.0
            balancesheet_df["Quick Ratio"] = 0.0
            balancesheet_df["Quick Ratio growth"] = 0.0
            balancesheet_df["Debt to Equity"] = 0.0
            balancesheet_df["Debt to Equity growth"] = 0.0
            # Invert the rows in balancesheet_df dataframe
            balancesheet_df = balancesheet_df.iloc[::-1]
            # Reset the index of the balancesheet_df dataframe
            balancesheet_df = balancesheet_df.reset_index()
            # Rename the index column to Date
            balancesheet_df = balancesheet_df.rename(columns={"index": "Date"})
            if _is_financial_sector:
                if "Current Assets" not in balancesheet_df.columns:
                    balancesheet_df["Current Assets"] = 0.0
                    # Safely rename Current Liabilities from available column
                    if "Derivative Product Liabilities" in balancesheet_df.columns:
                        balancesheet_df = balancesheet_df.rename(columns={"Derivative Product Liabilities": "Current Liabilities"})
                    elif "Current Liabilities" not in balancesheet_df.columns:
                        balancesheet_df["Current Liabilities"] = 0.0
                    for index, row in balancesheet_df.iterrows():
                        # Safely compute Current Assets from available columns
                        _cash = balancesheet_df.loc[index, "Cash And Cash Equivalents"] if "Cash And Cash Equivalents" in balancesheet_df.columns else 0
                        _recv = balancesheet_df.loc[index, "Receivables"] if "Receivables" in balancesheet_df.columns else 0
                        _trade = balancesheet_df.loc[index, "Trading Securities"] if "Trading Securities" in balancesheet_df.columns else 0
                        _fair = balancesheet_df.loc[index, "Financial Assets Designatedas Fair Value Through Profitor Loss Total"] if "Financial Assets Designatedas Fair Value Through Profitor Loss Total" in balancesheet_df.columns else 0
                        balancesheet_df.loc[index, "Current Assets"] = _cash + _recv + _trade + _fair
                        balancesheet_df.loc[index, "Book Value"] = balancesheet_df.loc[index, "Total Equity Gross Minority Interest"]
                        balancesheet_df.loc[index, "Book Value per share"] = balancesheet_df.loc[index, "Total Equity Gross Minority Interest"] / income_stmt_df.loc[index, "Diluted Average Shares"]
                        balancesheet_df.loc[index, "Return on Assets"] = income_stmt_df.loc[index, "Net Income Common Stockholders"] / balancesheet_df.loc[index, "Total Assets"]
                        balancesheet_df.loc[index, "Return on Equity"] = income_stmt_df.loc[index, "Net Income Common Stockholders"] / balancesheet_df.loc[index, "Total Equity Gross Minority Interest"]
                        balancesheet_df.loc[index, "Return on Invested Capital"] = income_stmt_df.loc[index, "Net Income Common Stockholders"] / (balancesheet_df.loc[index, "Total Equity Gross Minority Interest"] + balancesheet_df.loc[index, "Total Liabilities Net Minority Interest"])
                        balancesheet_df.loc[index, "Current Ratio"] = (balancesheet_df.loc[index, "Current Assets"]) / balancesheet_df.loc[index, "Current Liabilities"] if balancesheet_df.loc[index, "Current Liabilities"] != 0 else 0
                        _quick_cash = balancesheet_df.loc[index, "Cash And Cash Equivalents"] if "Cash And Cash Equivalents" in balancesheet_df.columns else 0
                        balancesheet_df.loc[index, "Quick Ratio"] = _quick_cash / balancesheet_df.loc[index, "Current Liabilities"] if balancesheet_df.loc[index, "Current Liabilities"] != 0 else 0
                        balancesheet_df.loc[index, "Debt to Equity"] = balancesheet_df.loc[index, "Total Liabilities Net Minority Interest"] / balancesheet_df.loc[index, "Total Equity Gross Minority Interest"]
                        if index == 0:
                            balancesheet_df.loc[index, "Total Assets growth"] = 0.0
                            balancesheet_df.loc[index, "Current Assets growth"] = 0.0
                            balancesheet_df.loc[index, "Cash and Cash Equivalents growth"] = 0.0
                            balancesheet_df.loc[index, "Total Liabilities growth"] = 0.0
                            balancesheet_df.loc[index, "Total Equity growth"] = 0.0
                            balancesheet_df.loc[index, "Current Liabilities growth"] = 0.0
                            balancesheet_df.loc[index, "Book Value growth"] = 0.0
                            balancesheet_df.loc[index, "Book Value per share growth"] = 0.0
                            balancesheet_df.loc[index, "Return on Assets growth"] = 0.0
                            balancesheet_df.loc[index, "Return on Equity growth"] = 0.0
                            balancesheet_df.loc[index, "Return on Invested Capital growth"] = 0.0
                            balancesheet_df.loc[index, "Current Ratio growth"] = 0.0
                            balancesheet_df.loc[index, "Quick Ratio growth"] = 0.0
                            balancesheet_df.loc[index, "Debt to Equity growth"] = 0.0
                        else:
                            balancesheet_df.loc[index, "Total Assets growth"] = _safe_growth(balancesheet_df.iloc[index]["Total Assets"], balancesheet_df.iloc[index-1]["Total Assets"])
                            balancesheet_df.loc[index, "Current Assets growth"] = _safe_growth(balancesheet_df.iloc[index]["Current Assets"], balancesheet_df.iloc[index-1]["Current Assets"])
                            balancesheet_df.loc[index, "Cash and Cash Equivalents growth"] = _safe_growth(balancesheet_df.iloc[index]["Cash And Cash Equivalents"], balancesheet_df.iloc[index-1]["Cash And Cash Equivalents"])
                            balancesheet_df.loc[index, "Total Liabilities growth"] = _safe_growth(balancesheet_df.iloc[index]["Total Liabilities Net Minority Interest"], balancesheet_df.iloc[index-1]["Total Liabilities Net Minority Interest"])
                            balancesheet_df.loc[index, "Total Equity growth"] = _safe_growth(balancesheet_df.iloc[index]["Total Equity Gross Minority Interest"], balancesheet_df.iloc[index-1]["Total Equity Gross Minority Interest"])
                            balancesheet_df.loc[index, "Current Liabilities growth"] = _safe_growth(balancesheet_df.iloc[index]["Current Liabilities"], balancesheet_df.iloc[index-1]["Current Liabilities"])
                            balancesheet_df.loc[index, "Book Value growth"] = _safe_growth(balancesheet_df.iloc[index]["Book Value"], balancesheet_df.iloc[index-1]["Book Value"])
                            balancesheet_df.loc[index, "Book Value per share growth"] = _safe_growth(balancesheet_df.iloc[index]["Book Value per share"], balancesheet_df.iloc[index-1]["Book Value per share"])
                            balancesheet_df.loc[index, "Return on Assets growth"] = _safe_growth(balancesheet_df.iloc[index]["Return on Assets"], balancesheet_df.iloc[index-1]["Return on Assets"])
                            balancesheet_df.loc[index, "Return on Equity growth"] = _safe_growth(balancesheet_df.iloc[index]["Return on Equity"], balancesheet_df.iloc[index-1]["Return on Equity"])
                            balancesheet_df.loc[index, "Return on Invested Capital growth"] = _safe_growth(balancesheet_df.iloc[index]["Return on Invested Capital"], balancesheet_df.iloc[index-1]["Return on Invested Capital"])
                            balancesheet_df.loc[index, "Current Ratio growth"] = _safe_growth(balancesheet_df.iloc[index]["Current Ratio"], balancesheet_df.iloc[index-1]["Current Ratio"])
                            balancesheet_df.loc[index, "Quick Ratio growth"] = _safe_growth(balancesheet_df.iloc[index]["Quick Ratio"], balancesheet_df.iloc[index-1]["Quick Ratio"])
                            balancesheet_df.loc[index, "Debt to Equity growth"] = _safe_growth(balancesheet_df.iloc[index]["Debt to Equity"], balancesheet_df.iloc[index-1]["Debt to Equity"])
                else:
                    for index, row in balancesheet_df.iterrows():
                        balancesheet_df.loc[index, "Book Value"] = balancesheet_df.loc[index, "Total Equity Gross Minority Interest"]
                        balancesheet_df.loc[index, "Book Value per share"] = balancesheet_df.loc[index, "Total Equity Gross Minority Interest"] / income_stmt_df.loc[index, "Diluted Average Shares"]
                        balancesheet_df.loc[index, "Return on Assets"] = income_stmt_df.loc[index, "Net Income Common Stockholders"] / balancesheet_df.loc[index, "Total Assets"]
                        balancesheet_df.loc[index, "Return on Equity"] = income_stmt_df.loc[index, "Net Income Common Stockholders"] / balancesheet_df.loc[index, "Total Equity Gross Minority Interest"]
                        _cl = balancesheet_df.loc[index, "Current Liabilities"]
                        balancesheet_df.loc[index, "Current Ratio"] = balancesheet_df.loc[index, "Current Assets"] / _cl if _cl != 0 else 0.0
                        _cash = balancesheet_df.loc[index, "Cash And Cash Equivalents"] if "Cash And Cash Equivalents" in balancesheet_df.columns else 0
                        balancesheet_df.loc[index, "Quick Ratio"] = _cash / _cl if _cl != 0 else 0.0
                        balancesheet_df.loc[index, "Debt to Equity"] = balancesheet_df.loc[index, "Total Liabilities Net Minority Interest"] / balancesheet_df.loc[index, "Total Equity Gross Minority Interest"]
                        if index == 0:
                            balancesheet_df.loc[index, "Total Assets growth"] = 0.0
                            balancesheet_df.loc[index, "Current Assets growth"] = 0.0
                            balancesheet_df.loc[index, "Cash and Cash Equivalents growth"] = 0.0
                            balancesheet_df.loc[index, "Total Liabilities growth"] = 0.0
                            balancesheet_df.loc[index, "Total Equity growth"] = 0.0
                            balancesheet_df.loc[index, "Current Liabilities growth"] = 0.0
                            balancesheet_df.loc[index, "Book Value growth"] = 0.0
                            balancesheet_df.loc[index, "Book Value per share growth"] = 0.0
                            balancesheet_df.loc[index, "Return on Assets growth"] = 0.0
                            balancesheet_df.loc[index, "Return on Equity growth"] = 0.0
                            balancesheet_df.loc[index, "Current Ratio growth"] = 0.0
                            balancesheet_df.loc[index, "Quick Ratio growth"] = 0.0
                            balancesheet_df.loc[index, "Debt to Equity growth"] = 0.0
                        else:
                            balancesheet_df.loc[index, "Total Assets growth"] = _safe_growth(balancesheet_df.iloc[index]["Total Assets"], balancesheet_df.iloc[index-1]["Total Assets"])
                            balancesheet_df.loc[index, "Current Assets growth"] = _safe_growth(balancesheet_df.iloc[index]["Current Assets"], balancesheet_df.iloc[index-1]["Current Assets"])
                            balancesheet_df.loc[index, "Cash and Cash Equivalents growth"] = _safe_growth(balancesheet_df.iloc[index]["Cash And Cash Equivalents"], balancesheet_df.iloc[index-1]["Cash And Cash Equivalents"])
                            balancesheet_df.loc[index, "Total Liabilities growth"] = _safe_growth(balancesheet_df.iloc[index]["Total Liabilities Net Minority Interest"], balancesheet_df.iloc[index-1]["Total Liabilities Net Minority Interest"])
                            balancesheet_df.loc[index, "Total Equity growth"] = _safe_growth(balancesheet_df.iloc[index]["Total Equity Gross Minority Interest"], balancesheet_df.iloc[index-1]["Total Equity Gross Minority Interest"])
                            balancesheet_df.loc[index, "Current Liabilities growth"] = _safe_growth(balancesheet_df.iloc[index]["Current Liabilities"], balancesheet_df.iloc[index-1]["Current Liabilities"])
                            balancesheet_df.loc[index, "Book Value growth"] = _safe_growth(balancesheet_df.iloc[index]["Book Value"], balancesheet_df.iloc[index-1]["Book Value"])
                            balancesheet_df.loc[index, "Book Value per share growth"] = _safe_growth(balancesheet_df.iloc[index]["Book Value per share"], balancesheet_df.iloc[index-1]["Book Value per share"])
                            balancesheet_df.loc[index, "Return on Assets growth"] = _safe_growth(balancesheet_df.iloc[index]["Return on Assets"], balancesheet_df.iloc[index-1]["Return on Assets"])
                            balancesheet_df.loc[index, "Return on Equity growth"] = _safe_growth(balancesheet_df.iloc[index]["Return on Equity"], balancesheet_df.iloc[index-1]["Return on Equity"])
                            balancesheet_df.loc[index, "Current Ratio growth"] = _safe_growth(balancesheet_df.iloc[index]["Current Ratio"], balancesheet_df.iloc[index-1]["Current Ratio"])
                            balancesheet_df.loc[index, "Quick Ratio growth"] = _safe_growth(balancesheet_df.iloc[index]["Quick Ratio"], balancesheet_df.iloc[index-1]["Quick Ratio"])
                            balancesheet_df.loc[index, "Debt to Equity growth"] = _safe_growth(balancesheet_df.iloc[index]["Debt to Equity"], balancesheet_df.iloc[index-1]["Debt to Equity"])
            else:
                for index, row in balancesheet_df.iterrows():
                    balancesheet_df.loc[index, "Book Value"] = balancesheet_df.loc[index, "Total Equity Gross Minority Interest"]
                    balancesheet_df.loc[index, "Book Value per share"] = balancesheet_df.loc[index, "Total Equity Gross Minority Interest"] / income_stmt_df.loc[index, "Diluted Average Shares"]
                    balancesheet_df.loc[index, "Return on Assets"] = income_stmt_df.loc[index, "Net Income Common Stockholders"] / balancesheet_df.loc[index, "Total Assets"]
                    balancesheet_df.loc[index, "Return on Equity"] = income_stmt_df.loc[index, "Net Income Common Stockholders"] / balancesheet_df.loc[index, "Total Equity Gross Minority Interest"]
                    balancesheet_df.loc[index, "Return on Invested Capital"] = income_stmt_df.loc[index, "Net Income Common Stockholders"] / (balancesheet_df.loc[index, "Total Equity Gross Minority Interest"] + balancesheet_df.loc[index, "Total Liabilities Net Minority Interest"])
                    _cl = balancesheet_df.loc[index, "Current Liabilities"]
                    balancesheet_df.loc[index, "Current Ratio"] = balancesheet_df.loc[index, "Current Assets"] / _cl if _cl != 0 else 0.0
                    _cash_equiv = balancesheet_df.loc[index, "Cash Cash Equivalents And Short Term Investments"] if "Cash Cash Equivalents And Short Term Investments" in balancesheet_df.columns else 0
                    balancesheet_df.loc[index, "Quick Ratio"] = _cash_equiv / _cl if _cl != 0 else 0.0
                    balancesheet_df.loc[index, "Debt to Equity"] = balancesheet_df.loc[index, "Total Liabilities Net Minority Interest"] / balancesheet_df.loc[index, "Total Equity Gross Minority Interest"]
                    if index == 0:
                        balancesheet_df.loc[index, "Total Assets growth"] = 0.0
                        balancesheet_df.loc[index, "Current Assets growth"] = 0.0
                        balancesheet_df.loc[index, "Cash and Cash Equivalents growth"] = 0.0
                        balancesheet_df.loc[index, "Total Equity growth"] = 0.0
                        balancesheet_df.loc[index, "Total Liabilities growth"] = 0.0
                        balancesheet_df.loc[index, "Current Liabilities growth"] = 0.0
                        balancesheet_df.loc[index, "Book Value growth"] = 0.0
                        balancesheet_df.loc[index, "Book Value per share growth"] = 0.0
                        balancesheet_df.loc[index, "Return on Assets growth"] = 0.0
                        balancesheet_df.loc[index, "Return on Equity growth"] = 0.0
                        balancesheet_df.loc[index, "Return on Invested Capital growth"] = 0.0
                        balancesheet_df.loc[index, "Current Ratio growth"] = 0.0
                        balancesheet_df.loc[index, "Quick Ratio growth"] = 0.0
                        balancesheet_df.loc[index, "Debt to Equity growth"] = 0.0
                    else:
                        balancesheet_df.loc[index, "Total Assets growth"] = _safe_growth(balancesheet_df.iloc[index]["Total Assets"], balancesheet_df.iloc[index-1]["Total Assets"])
                        balancesheet_df.loc[index, "Current Assets growth"] = _safe_growth(balancesheet_df.iloc[index]["Current Assets"], balancesheet_df.iloc[index-1]["Current Assets"])
                        balancesheet_df.loc[index, "Cash and Cash Equivalents growth"] = _safe_growth(balancesheet_df.iloc[index]["Cash Cash Equivalents And Short Term Investments"], balancesheet_df.iloc[index-1]["Cash Cash Equivalents And Short Term Investments"])
                        balancesheet_df.loc[index, "Total Liabilities growth"] = _safe_growth(balancesheet_df.iloc[index]["Total Liabilities Net Minority Interest"], balancesheet_df.iloc[index-1]["Total Liabilities Net Minority Interest"])
                        balancesheet_df.loc[index, "Total Equity growth"] = _safe_growth(balancesheet_df.iloc[index]["Total Equity Gross Minority Interest"], balancesheet_df.iloc[index-1]["Total Equity Gross Minority Interest"])
                        balancesheet_df.loc[index, "Current Liabilities growth"] = _safe_growth(balancesheet_df.iloc[index]["Current Liabilities"], balancesheet_df.iloc[index-1]["Current Liabilities"])
                        balancesheet_df.loc[index, "Book Value growth"] = _safe_growth(balancesheet_df.iloc[index]["Book Value"], balancesheet_df.iloc[index-1]["Book Value"])
                        balancesheet_df.loc[index, "Book Value per share growth"] = _safe_growth(balancesheet_df.iloc[index]["Book Value per share"], balancesheet_df.iloc[index-1]["Book Value per share"])
                        balancesheet_df.loc[index, "Return on Assets growth"] = _safe_growth(balancesheet_df.iloc[index]["Return on Assets"], balancesheet_df.iloc[index-1]["Return on Assets"])
                        balancesheet_df.loc[index, "Return on Equity growth"] = _safe_growth(balancesheet_df.iloc[index]["Return on Equity"], balancesheet_df.iloc[index-1]["Return on Equity"])
                        balancesheet_df.loc[index, "Return on Invested Capital growth"] = _safe_growth(balancesheet_df.iloc[index]["Return on Invested Capital"], balancesheet_df.iloc[index-1]["Return on Invested Capital"])
                        balancesheet_df.loc[index, "Current Ratio growth"] = _safe_growth(balancesheet_df.iloc[index]["Current Ratio"], balancesheet_df.iloc[index-1]["Current Ratio"])
                        balancesheet_df.loc[index, "Quick Ratio growth"] = _safe_growth(balancesheet_df.iloc[index]["Quick Ratio"], balancesheet_df.iloc[index-1]["Quick Ratio"])
                        balancesheet_df.loc[index, "Debt to Equity growth"] = _safe_growth(balancesheet_df.iloc[index]["Debt to Equity"], balancesheet_df.iloc[index-1]["Debt to Equity"])

            cashflow = stock_data.cashflow
            cashflow_df = pd.DataFrame(cashflow)
            if cashflow_df.empty or len(cashflow_df.columns) < 2:
                raise ValueError(f"Insufficient cash flow data for {symbol} (need at least 2 periods, got {len(cashflow_df.columns)})")
            if len(pd.DataFrame(stock_data.cashflow).columns) > len(pd.DataFrame(stock_data.income_stmt).columns):
                column_amount = len(pd.DataFrame(stock_data.cashflow).columns) - len(pd.DataFrame(stock_data.income_stmt).columns)
                cashflow_df = cashflow_df.drop(columns=cashflow_df.columns[-column_amount])
            elif len(pd.DataFrame(stock_data.cashflow).columns) > len(pd.DataFrame(stock_data.balancesheet).columns):
                column_amount = len(pd.DataFrame(stock_data.cashflow).columns) - len(pd.DataFrame(stock_data.balancesheet).columns)
                cashflow_df = cashflow_df.drop(columns=cashflow_df.columns[-column_amount])

            cashflow_df = cashflow_df.drop(columns=cashflow_df.columns[-1])
            # Checking if the input DataFrame is empty
            if cashflow_df.empty:
                raise ValueError("Input DataFrame is empty.")

            # Rotate the cashflow_df dataframe
            cashflow_df = cashflow_df.transpose()
            cashflow_df["Free Cash Flow growth"] = 0.0
            cashflow_df["Free Cash Flow per share"] = 0.0
            cashflow_df["Free Cash Flow per share growth"] = 0.0
            # Invert the rows in cashflow_df dataframe
            cashflow_df = cashflow_df.iloc[::-1]
            # Reset the index of the cashflow_df dataframe
            cashflow_df = cashflow_df.reset_index()
            # Rename the index column to Date
            cashflow_df = cashflow_df.rename(columns={"index": "Date"})
            for index, row in cashflow_df.iterrows():
                cashflow_df.loc[index, "Free Cash Flow per share"] = (cashflow_df.loc[index, "Free Cash Flow"] / income_stmt_df.loc[index, "Diluted Average Shares"])
                if index == 0:
                    cashflow_df.loc[index, "Free Cash Flow growth"] = 0.0
                    cashflow_df.loc[index, "Free Cash Flow per share growth"] = 0.0
                else:
                    cashflow_df.loc[index, "Free Cash Flow growth"] = _safe_growth(cashflow_df.iloc[index]["Free Cash Flow"], cashflow_df.iloc[index-1]["Free Cash Flow"])
                    cashflow_df.loc[index, "Free Cash Flow per share growth"] = _safe_growth(cashflow_df.iloc[index]["Free Cash Flow per share"], cashflow_df.iloc[index-1]["Free Cash Flow per share"])
            # Join income_stmt_df, balancesheet_df and cashflow_df dataframes on the Date column
            full_stock_financial_data_df = pd.merge(income_stmt_df, balancesheet_df, on="Date")
            full_stock_financial_data_df = pd.merge(full_stock_financial_data_df, cashflow_df, on="Date")
            # Drop row 0 from full_stock_financial_data_df
            full_stock_financial_data_df["Ticker"] = symbol
            full_stock_financial_data_df = full_stock_financial_data_df.drop([0])
            full_stock_financial_data_df = full_stock_financial_data_df.reset_index(drop=True)

            # --- Dynamic column mapper ---
            # Apply universal rename mapping: handles different raw column names from yfinance
            # If both source and target exist, drop source.
            # If multiple sources map to the same target, keep the first and drop the rest.
            _rename_map = {
                "Diluted Average Shares": "Amount of stocks",
                "Total Revenue": "Revenue",
                "Operating Income": "Operating Earnings",
                "Cash And Cash Equivalents": "Cash and Cash Equivalents",
                "Cash Cash Equivalents And Short Term Investments": "Cash and Cash Equivalents",
                "Total Liabilities Net Minority Interest": "Total Liabilities",
                "Total Equity Gross Minority Interest": "Total Equity",
            }
            _cols = set(full_stock_financial_data_df.columns)
            _to_drop = []
            _to_rename = {}
            _seen_targets = set()
            for src, tgt in _rename_map.items():
                if src not in _cols:
                    continue
                if tgt in _cols:
                    # Target already exists as a real column — drop the source
                    _to_drop.append(src)
                elif tgt in _seen_targets:
                    # Another source was already renamed to this target — drop this duplicate
                    _to_drop.append(src)
                else:
                    _to_rename[src] = tgt
                    _seen_targets.add(tgt)
            if _to_drop:
                full_stock_financial_data_df = full_stock_financial_data_df.drop(columns=_to_drop)
            full_stock_financial_data_df = full_stock_financial_data_df.rename(columns=_to_rename)

            # Desired output columns (superset) — only those present will be selected
            _desired_columns = [
                "Ticker", "Date", "Amount of stocks", "Revenue", "Revenue growth",
                "Gross Profit", "Gross Profit growth", "Gross Margin", "Gross Margin growth",
                "Operating Earnings", "Operating Earnings growth",
                "Operating Margin", "Operating Margin growth",
                "Net Income", "Net Income growth", "Net Income Margin", "Net Income Margin growth",
                "EPS", "EPS growth",
                "Total Assets", "Total Assets growth", "Current Assets", "Current Assets growth",
                "Cash and Cash Equivalents", "Cash and Cash Equivalents growth",
                "Total Liabilities", "Total Liabilities growth",
                "Total Equity", "Total Equity growth",
                "Current Liabilities", "Current Liabilities growth",
                "Book Value", "Book Value growth", "Book Value per share", "Book Value per share growth",
                "Return on Assets", "Return on Assets growth",
                "Return on Equity", "Return on Equity growth",
                "Return on Invested Capital", "Return on Invested Capital growth",
                "Current Ratio", "Current Ratio growth", "Quick Ratio", "Quick Ratio growth",
                "Debt to Equity", "Debt to Equity growth",
                "Free Cash Flow", "Free Cash Flow growth",
                "Free Cash Flow per share", "Free Cash Flow per share growth",
            ]
            available_columns = [c for c in _desired_columns if c in full_stock_financial_data_df.columns]
            full_stock_financial_data_df = full_stock_financial_data_df[available_columns]

            full_stock_financial_data_df = full_stock_financial_data_df.rename(columns={"Date": "date",
                "Ticker": "ticker", "Amount of stocks": "average_shares", "Revenue": "revenue", "Revenue growth": "revenue_Growth",
                "Gross Profit": "gross_Profit", "Gross Profit growth": "gross_Profit_Growth", "Gross Margin": "gross_Margin",
                "Gross Margin growth": "gross_Margin_Growth",
                "Operating Earnings": "operating_Earning", "Operating Earnings growth": "operating_Earning_Growth",
                "Operating Margin": "operating_Earning_Margin", "Operating Margin growth": "operating_Earning_Margin_Growth",
                "Net Income": "net_Income", "Net Income growth": "net_Income_Growth",
                "Net Income Margin": "net_Income_Margin", "Net Income Margin growth": "net_Income_Margin_Growth", "EPS": "eps",
                "EPS growth": "eps_Growth", "Total Assets": "total_Assets", "Total Assets growth": "total_Assets_Growth",
                "Current Assets": "current_Assets", "Current Assets growth": "current_Assets_Growth", "Cash and Cash Equivalents": "cash_And_Cash_Equivalents",
                "Cash and Cash Equivalents growth": "cash_And_Cash_Equivalents_Growth", "Total Equity": "equity", "Total Equity growth": "equity_Growth",
                "Total Liabilities": "liabilities", "Total Liabilities growth": "liabilities_Growth", "Current Liabilities": "current_Liabilities",
                "Current Liabilities growth": "current_Liabilities_Growth", "Book Value": "book_Value", "Book Value growth": "book_Value_Growth",
                "Book Value per share": "book_Value_Per_Share", "Book Value per share growth": "book_Value_Per_Share_Growth",
                "Return on Assets": "return_On_Assets", "Return on Assets growth": "return_On_Assets_Growth", "Return on Equity": "return_On_Equity",
                "Return on Equity growth": "return_On_Equity_Growth", "Current Ratio": "current_Ratio", "Current Ratio growth": "current_Ratio_Growth",
                "Quick Ratio": "quick_Ratio", "Quick Ratio growth": "quick_Ratio_Growth", "Debt to Equity": "debt_To_Equity",
                "Debt to Equity growth": "debt_To_Equity_Growth", "Free Cash Flow": "free_Cash_Flow", "Free Cash Flow growth": "free_Cash_Flow_Growth",
                "Free Cash Flow per share": "free_Cash_Flow_Per_Share", "Free Cash Flow per share growth": "free_Cash_Flow_Per_Share_Growth"
                })
            
            # Replace infinity values with NaN (MySQL cannot store inf values)
            import numpy as np
            full_stock_financial_data_df = full_stock_financial_data_df.replace([np.inf, -np.inf], np.nan)
            
            return full_stock_financial_data_df
    except KeyError as e:
        raise KeyError(f"Financial data processing failed for '{symbol}': missing data key {e}") from e

def combine_stock_data(stock_price_data_df, full_stock_financial_data_df):
    """
    Combines stock data with financial stock data.

    The function combines stock data with financial stock data and returns a DataFrame with the combined data.

    Parameters:
    - stock_price_data_df (pd.DataFrame): A DataFrame with stock data.
    - full_stock_financial_data_df (pd.DataFrame): A DataFrame with financial stock data.

    Returns:
    - combined_stock_data_df (pd.DataFrame): A DataFrame with the combined data.

    Raises:
    - ValueError: If the stock_price_data_df parameter is empty or if the full_stock_financial_data_df parameter is empty.
    - ValueError: Failed to combine stock data and financial stock data.
    """
    # Checking if the stock_price_data_df parameter is empty
    if stock_price_data_df.empty:
        raise ValueError("No stock data provided.")

    # Checking if the full_stock_financial_data_df parameter is empty
    if full_stock_financial_data_df.empty:
        raise ValueError("No financial stock data provided.")

    try:
        # Create a list of column names to copy from full_stock_financial_data_df to stock_price_data_df
        column_names = full_stock_financial_data_df.columns[2:]
        # Create a copy of stock_price_data_df
        combined_stock_data_df = stock_price_data_df.copy()
        # Add columns from full_stock_financial_data_df to stock_price_data_df
        for year in range(len(full_stock_financial_data_df["date"])):
            combined_stock_data_df.loc[combined_stock_data_df["date"] >= full_stock_financial_data_df.iloc[year]["date"], column_names] = full_stock_financial_data_df.iloc[year].values[2:]

        print("Stock data and financial stock data combined successfully.")
        return combined_stock_data_df

    except ValueError as e:
        raise ValueError(f"Error combining stock data: {e}") from e

def calculate_ratios(combined_stock_data_df, stock_symbol=None, prefer_ttm=True):
    """
    Calculates P/S, P/E, P/B and P/FCF ratios using TTM data when available.

    The function calculates the P/S, P/E, P/B and P/FCF ratios using:
    - TTM (Trailing Twelve Months) data when 4+ quarters are available (preferred)
    - Annual report data as fallback when insufficient quarterly data exists
    
    Adds metadata columns to track which data source was used:
    - ratio_data_source: 'ttm' or 'annual'
    - quarters_available: Number of quarterly reports available

    Parameters:
    - combined_stock_data_df (pd.DataFrame): A DataFrame with combined stock data.
    - stock_symbol (str, optional): Stock ticker symbol for TTM lookup. If None,
      will attempt to get from 'ticker' column in DataFrame.
    - prefer_ttm (bool): Whether to prefer TTM over annual data (default: True)

    Returns:
    - combined_stock_data_df (pd.DataFrame): A DataFrame with the ratios and source metadata added.

    Raises:
    - ValueError: If the combined_stock_data_df parameter is empty.
    - ValueError: Error calculating ratios.
    """
    # Checking if the combined_stock_data_df parameter is empty
    if combined_stock_data_df.empty:
        raise ValueError("No stock data provided.")
    
    # Get symbol from DataFrame if not provided
    if stock_symbol is None:
        if 'ticker' in combined_stock_data_df.columns:
            stock_symbol = combined_stock_data_df['ticker'].iloc[0]
        else:
            print("⚠️ No stock symbol provided, falling back to annual ratio calculation")
            prefer_ttm = False
    
    if prefer_ttm and stock_symbol:
        try:
            # Use TTM-enhanced ratio calculation
            result = calculate_ratios_ttm_with_fallback(
                combined_stock_data_df, 
                symbol=stock_symbol,
                prefer_ttm=True
            )
            print(f"Ratios calculated using {result['ratio_data_source'].iloc[0]} data for {stock_symbol}")
            return result
            
        except Exception as e:
            print(f"⚠️ TTM calculation failed for {stock_symbol}, falling back to annual: {e}")

    # Fallback to original annual-based calculation
    try:
        # Calculate the P/S ratio
        combined_stock_data_df["P/S"] = combined_stock_data_df["close_Price"] / (combined_stock_data_df["revenue"] / combined_stock_data_df["average_shares"])
        # Calculate the P/E ratio
        combined_stock_data_df["P/E"] = combined_stock_data_df["close_Price"] / combined_stock_data_df["eps"]
        # Calculate the P/B ratio
        combined_stock_data_df["P/B"] = combined_stock_data_df["close_Price"] / combined_stock_data_df["book_Value_Per_Share"]
        # Calculate the P/FCF ratio
        combined_stock_data_df["P/FCF"] = combined_stock_data_df["close_Price"] / combined_stock_data_df["free_Cash_Flow_Per_Share"]
        # Replace inf/-inf from zero-division (e.g. eps=0) with NaN
        combined_stock_data_df[["P/S", "P/E", "P/B", "P/FCF"]] = combined_stock_data_df[["P/S", "P/E", "P/B", "P/FCF"]].replace([np.inf, -np.inf], np.nan)
        print("Ratios have been calculated successfully using annual data, and added to the dataframe.")
        combined_stock_data_df[["P/S", "P/E", "P/B", "P/FCF"]] = combined_stock_data_df[["P/S", "P/E", "P/B", "P/FCF"]].shift(1)
        
        # Add source tracking for consistency
        combined_stock_data_df['ratio_data_source'] = 'annual'
        combined_stock_data_df['quarters_available'] = 0
        
        return combined_stock_data_df

    except ValueError as e:
        raise ValueError(f"Error calculating ratios: {e}") from e

def drop_nan_values(combined_stock_data_df):
    """
    Drops rows with NaN values in the given DataFrame.

    Parameters:
    - combined_stock_data_df (pd.DataFrame): A DataFrame with combined stock data.

    Returns:
    - combined_stock_data_df (pd.DataFrame): A DataFrame with the NaN values dropped.

    Raises:
    - ValueError: If the combined_stock_data_df parameter is empty.
    """
    # Checking if the combined_stock_data_df parameter is empty
    if combined_stock_data_df.empty:
        raise ValueError("No combined stock data provided.")

    # Drop rows with NaN values only in critical columns (not all columns) to preserve data
    # Allow calculated features to have NaN values (they will be handled by ml_builder)
    critical_cols = ['date', 'ticker', 'close_Price']
    available_critical_cols = [col for col in critical_cols if col in combined_stock_data_df.columns]
    combined_stock_data_df = combined_stock_data_df.dropna(subset=available_critical_cols)
    combined_stock_data_df = combined_stock_data_df.reset_index(drop=True)
    return combined_stock_data_df

def export_to_excel(dataframes, excel_file):
    """
    Exports the given dataframes to an Excel file with separate sheets.

    Each dataframe will be exported to a separate sheet in the Excel file.

    Parameters:
    - dataframes (dict): A dictionary of dataframes, where the keys are the sheet names and the values are the dataframes.
    - excel_file (str): The path to the Excel file to export.

    Raises:
    - ValueError: If the dataframes parameter is empty or if the excel_file parameter is empty.
    """
    try:
        # Check if the dataframes parameter is empty
        if not dataframes:
            raise ValueError("No dataframes provided.")

        # Check if the excel_file parameter is empty
        if not excel_file:
            raise ValueError("No Excel file path provided.")

        # Create a Pandas Excel writer object
        my_path = os.path.abspath(__file__)
        path = os.path.dirname(my_path)
        export_location = os.path.join(path, excel_file)
        writer = pd.ExcelWriter(export_location, engine='xlsxwriter')

        # Export each dataframe to a separate sheet in the Excel file
        for sheet_name, dataframe in dataframes.items():
            dataframe.to_excel(writer, sheet_name=sheet_name, index=False)

        # Save the Excel file
        writer.close()
        print("Dataframes have been successfully exported as xlsx file.")


    except ValueError as e:
        raise ValueError(f"Error exporting to Excel: {e}") from e

def import_excel(excel_file):
    """
    Imports the data from the given Excel file.

    Parameters:
    - excel_file (str): The path to the Excel file to import.

    Returns:
    - A dictionary of dataframes, where the keys are the sheet names and the values are the dataframes.

    Raises:
    - ValueError: If the excel_file parameter is empty.
    """
    try:
        # Check if the excel_file parameter is empty
        if not excel_file:
            raise ValueError("No Excel file path provided.")

        # Create a dictionary of dataframes to return
        dataframes = {}

        # Create a Pandas Excel file reader object
        my_path = os.path.abspath(__file__)
        path = os.path.dirname(my_path)
        import_location = os.path.join(path, excel_file)
        reader = pd.ExcelFile(import_location)

        # Import each sheet in the Excel file into a separate dataframe
        for sheet_name in reader.sheet_names:
            dataframes[sheet_name] = pd.read_excel(reader, sheet_name=sheet_name)

        # Return the dictionary of dataframes
        return dataframes

    except ValueError as e:
        raise ValueError(f"Error importing from Excel: {e}") from e

def convert_excel_to_csv(dataframe, file_name):
    """
    Converts the given Excel file to a CSV file.

    Parameters:
    - dataframe (dataframe): The dataframe to convert.
    - file_name (str): The name of the CSV file to create.

    Raises:
    - ValueError: If the dataframe parameter is empty or if the file_name parameter is empty.
    """
    try:
        # Check if the dataframe parameter is empty
        if dataframe.empty:
            raise ValueError("No dataframe provided.")

        # Check if the file_name parameter is empty
        if not file_name:
            raise ValueError("No file name provided.")

        # Convert the Excel file to a CSV file
        my_path = os.path.abspath(__file__)
        path = os.path.dirname(my_path)
        export_location = os.path.join(path, f"{file_name}.csv")
        dataframe.to_csv(export_location, sep=',', encoding='utf-8', index=False)

    except ValueError as e:
        raise ValueError(f"Error converting to CSV: {e}") from e

# Run the main function
if __name__ == "__main__":
    """
    Main execution block for stock data fetching.
    
    For the full modular orchestration with blacklist management, TTM support,
    and comprehensive error handling, use stock_orchestrator.py instead:
    
        python stock_orchestrator.py --indices C25 SP500 --include-market-indices --export-csv
        python stock_orchestrator.py --tickers NOVO-B.CO MAERSK-B.CO
        python stock_orchestrator.py --indices C25 --years 10
    
    This main block provides backward compatibility with the existing workflow.
    """
    import fetch_secrets
    import db_connectors
    import db_interactions
    from blacklist_manager import BlacklistManager
    
    db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()
    db_con = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)
    start_time = time.time()
    
    # Initialize blacklist manager
    blacklist = BlacklistManager()

    # Fetch C25 and S&P 500 dynamically
    symbols_df = dynamic_fetch_index_data(
        indices=['C25', 'SP500'],
        include_market_indices=True,  # Adds ^VIX, ^GSPC
        export_csv=True
    )
    stock_tickers_list = symbols_df["Symbol"].tolist()
    
    # Filter out blacklisted tickers
    stock_tickers_list = blacklist.filter_tickers(stock_tickers_list)
    print(f"Processing {len(stock_tickers_list)} tickers after filtering blacklist")

    # Process stock info data
    for ticker in stock_tickers_list:
        try:
            if db_interactions.does_stock_exists_stock_info_data(ticker) is False:
                print(f"Fetching stock info data for {ticker}")
                stock_info_data_df = fetch_stock_standard_data(ticker)
                db_interactions.export_stock_info_data(stock_info_data_df)
                print(f"✓ Stock info data exported for {ticker}")
            else:
                print(f"Stock info data already exists for {ticker}")
        except Exception as e:
            print(f"❌ Error processing stock info for {ticker}: {e}")
            if "No data found" in str(e) or "delisted" in str(e).lower():
                blacklist.add_ticker(ticker, f"Stock info error: {e}")
            continue

    # Process price, financial, and ratio data for all tickers in database
    ticker_list = db_interactions.import_ticker_list()
    ticker_list = blacklist.filter_tickers(ticker_list)
    
    for ticker in ticker_list:
        print(f"\n{'='*60}")
        print(f"Processing: {ticker}")
        print('='*60)
        
        try:
            stock_info = yf.Ticker(ticker).info
            is_index = stock_info.get("typeDisp") == "Index"
        except Exception as e:
            print(f"⚠️  Could not get stock info for {ticker}, assuming not an index: {e}")
            is_index = str(ticker).startswith('^') if isinstance(ticker, str) else False
        
        # Process price data
        try:
            if db_interactions.does_stock_exists_stock_price_data(ticker) is False:
                print("Stock price data does not exist - fetching new data")
                stock_price_data_df = fetch_stock_price_data(ticker)
                stock_price_data_df = calculate_period_returns(stock_price_data_df)
                stock_price_data_df = add_technical_indicators(stock_price_data_df)
                stock_price_data_df = add_volume_indicators(stock_price_data_df)
                if not is_index:
                    stock_price_data_df = add_volatility_indicators(stock_price_data_df)
                stock_price_data_df = calculate_moving_averages(stock_price_data_df)
                stock_price_data_df = calculate_standard_diviation_value(stock_price_data_df)
                stock_price_data_df = calculate_bollinger_bands(stock_price_data_df)
                stock_price_data_df = calculate_momentum(stock_price_data_df)
                
                # Drop rows with NaN only in critical columns
                critical_cols = ['date', 'ticker', 'close_Price', 'open_Price', 'high_Price', 'low_Price']
                stock_price_data_df = stock_price_data_df.dropna(subset=critical_cols)
                
                if not stock_price_data_df.empty:
                    db_interactions.export_stock_price_data(stock_price_data_df)
                    print(f"✓ Exported {len(stock_price_data_df)} price records")
                else:
                    print("⚠️  No valid price data to export")
            else:
                print("Checking for new price data...")
                stock_price_data_df = db_interactions.import_stock_price_data(stock_ticker=ticker)
                last_db_date = stock_price_data_df.iloc[0]["date"]
                
                if hasattr(last_db_date, 'date'):
                    last_db_date = last_db_date.date()
                elif isinstance(last_db_date, str):
                    last_db_date = datetime.datetime.strptime(last_db_date, "%Y-%m-%d").date()
                
                from market_hours_utils import should_fetch_new_data
                should_fetch, new_date, reason = should_fetch_new_data(last_db_date, ticker)
                print(f"  {reason}")
                
                if should_fetch:
                    print(f"Fetching new data from {new_date}")
                    stock_price_data_df = db_interactions.import_stock_price_data(amount=252*5+1, stock_ticker=ticker)
                    stock_price_data_df["date"] = pd.to_datetime(stock_price_data_df["date"])
                    new_stock_price_data_df = fetch_stock_price_data(ticker, new_date)
                    
                    if not new_stock_price_data_df.empty:
                        stock_price_data_df = pd.concat([stock_price_data_df, new_stock_price_data_df], axis=0, ignore_index=True)
                        stock_price_data_df = calculate_period_returns(stock_price_data_df)
                        stock_price_data_df = add_technical_indicators(stock_price_data_df)
                        stock_price_data_df = add_volume_indicators(stock_price_data_df)
                        if not is_index:
                            stock_price_data_df = add_volatility_indicators(stock_price_data_df)
                        stock_price_data_df = calculate_moving_averages(stock_price_data_df)
                        stock_price_data_df = calculate_standard_diviation_value(stock_price_data_df)
                        stock_price_data_df = calculate_bollinger_bands(stock_price_data_df)
                        stock_price_data_df = calculate_momentum(stock_price_data_df)
                        stock_price_data_df = stock_price_data_df.loc[stock_price_data_df["date"] >= new_stock_price_data_df.loc[0, "date"]]
                        
                        critical_cols = ['date', 'ticker', 'close_Price', 'open_Price', 'high_Price', 'low_Price']
                        stock_price_data_df = stock_price_data_df.dropna(subset=critical_cols)
                        
                        if not stock_price_data_df.empty:
                            db_interactions.export_stock_price_data(stock_price_data_df)
                            print(f"✓ Exported {len(stock_price_data_df)} new price records")
                    else:
                        print("No new data available")
        except ValueError as e:
            if "Columns must be same length" in str(e):
                print(f"⚠️  Skipping {ticker}: Column length mismatch (likely index ticker with different structure)")
            else:
                print(f"❌ Error fetching price data for {ticker}: {e}")
        except Exception as e:
            print(f"❌ Error processing price data for {ticker}: {e}")
            continue

        # Skip financial and ratio data for index tickers
        if is_index:
            print(f"Skipping financial/ratio data for index ticker {ticker}")
            continue

        # Process financial data
        try:
            if db_interactions.does_stock_exists_stock_income_stmt_data(ticker) is False:
                print("Financial data does not exist - fetching")
                full_stock_financial_data_df = fetch_stock_financial_data(ticker)
                db_interactions.export_stock_financial_data(full_stock_financial_data_df)
                print("✓ Financial data exported")
            else:
                print("Checking for new financial data...")
                full_stock_financial_data_df = db_interactions.import_stock_financial_data(stock_ticker=ticker)
                db_date = full_stock_financial_data_df.iloc[0]["date"]
                full_stock_financial_data_df = fetch_stock_financial_data(ticker)
                source_date = full_stock_financial_data_df["date"].dt.date.iloc[-1]
                if db_date != source_date:
                    full_stock_financial_data_df = full_stock_financial_data_df.loc[full_stock_financial_data_df["date"].dt.date > db_date]
                    db_interactions.export_stock_financial_data(full_stock_financial_data_df)
                    print("✓ New financial data exported")
                else:
                    print("Financial data is up to date")
        except Exception as e:
            print(f"❌ Error processing financial data for {ticker}: {e}")

        # Process ratio data
        try:
            if db_interactions.does_stock_exists_stock_ratio_data(ticker) is False:
                print("Ratio data does not exist - calculating")
                TABEL_NAME = "stock_income_stmt_data"
                quary = f"""SELECT COUNT(financial_Statement_Date) FROM {TABEL_NAME} WHERE ticker = '{ticker}'"""
                entry_amount = pd.read_sql(sql=quary, con=db_con)
                full_stock_financial_data_df = db_interactions.import_stock_financial_data(amount=entry_amount.loc[0, entry_amount.columns[0]], stock_ticker=ticker)
                full_stock_financial_data_df = full_stock_financial_data_df.dropna(axis=1)
                date = full_stock_financial_data_df.iloc[0]["date"]
                TABEL_NAME = "stock_price_data"
                quary = f"""SELECT * FROM {TABEL_NAME} WHERE ticker = '{ticker}' AND date >= '{date}'"""
                stock_price_data_df = pd.read_sql(sql=quary, con=db_con)
                combined_stock_data_df = combine_stock_data(stock_price_data_df, full_stock_financial_data_df)
                combined_stock_data_df = calculate_ratios(combined_stock_data_df)
                stock_ratio_data_df = combined_stock_data_df[['date', 'ticker', 'P/S', 'P/E', 'P/B', 'P/FCF']]
                stock_ratio_data_df = stock_ratio_data_df.rename(columns={"P/S": "p_s", "P/E": "p_e", "P/B": "p_b", "P/FCF": "p_fcf"})
                stock_ratio_data_df = drop_nan_values(stock_ratio_data_df)
                db_interactions.export_stock_ratio_data(stock_ratio_data_df)
                print("✓ Ratio data exported")
            else:
                print("Checking for new ratio data...")
                stock_ratio_data_df = db_interactions.import_stock_ratio_data(stock_ticker=ticker)
                date = stock_ratio_data_df.iloc[0]["date"]
                if str(date) != datetime.datetime.now().strftime("%Y-%m-%d"):
                    print("Calculating new ratio data")
                    full_stock_financial_data_df = db_interactions.import_stock_financial_data(amount=1, stock_ticker=ticker)
                    full_stock_financial_data_df = full_stock_financial_data_df.dropna(axis=1)
                    TABEL_NAME = "stock_price_data"
                    quary = f"""SELECT * FROM {TABEL_NAME} WHERE ticker = '{ticker}' AND date >= '{date}'"""
                    stock_price_data_df = pd.read_sql(sql=quary, con=db_con)
                    combined_stock_data_df = combine_stock_data(stock_price_data_df, full_stock_financial_data_df)
                    combined_stock_data_df = calculate_ratios(combined_stock_data_df)
                    stock_ratio_data_df = combined_stock_data_df[['date', 'ticker', 'P/S', 'P/E', 'P/B', 'P/FCF']]
                    stock_ratio_data_df = stock_ratio_data_df.rename(columns={"P/S": "p_s", "P/E": "p_e", "P/B": "p_b", "P/FCF": "p_fcf"})
                    stock_ratio_data_df = drop_nan_values(stock_ratio_data_df)
                    db_interactions.export_stock_ratio_data(stock_ratio_data_df)
                    print("✓ Ratio data exported")
                else:
                    print("Ratio data is up to date")
        except Exception as e:
            print(f"❌ Error processing ratio data for {ticker}: {e}")

    # Calculate the execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\n{'='*60}")
    print(f"Execution completed in {execution_time:.2f} seconds")
    blacklisted_tickers = blacklist.get_blacklist()
    print(f"Blacklisted tickers: {len(blacklisted_tickers)}")
    print('='*60)
