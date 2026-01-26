"""
Technical Indicators Module

This module provides functions for calculating technical analysis indicators
for stock price data. It separates indicator calculations from data fetching
for better modularity and testability.

Indicators:
    - Moving Averages (SMA, EMA) - 5, 20, 40, 120, 200 periods
    - Bollinger Bands (2 standard deviations)
    - RSI (Relative Strength Index) - 14 period
    - MACD (Moving Average Convergence Divergence)
    - ATR (Average True Range) - 14 period
    - Momentum
    - Volume indicators (Volume SMA/EMA, VWAP, OBV)
    - Volatility metrics (5, 20, 60 day)
    - Period returns (1D, 1M, 3M, 6M, 9M, 1Y, 2Y, 3Y, 4Y, 5Y)

All indicators are shifted by 1 period to avoid look-ahead bias in ML applications.

Author: Stock Portfolio Builder
Last Modified: 2026
"""
import pandas as pd
import numpy as np
import pandas_ta as ta


def calculate_period_returns(df: pd.DataFrame, price_col: str = 'close_Price') -> pd.DataFrame:
    """
    Calculate period returns for multiple time horizons.
    
    Args:
        df: DataFrame with price data
        price_col: Name of the price column
        
    Returns:
        DataFrame with added return columns
    """
    if df.empty:
        raise ValueError("DataFrame cannot be empty")
    
    if price_col not in df.columns:
        raise ValueError(f"Column '{price_col}' not found")
    
    if df[price_col].isnull().values.any():
        raise ValueError(f"Column '{price_col}' contains null values")
    
    # Define periods (in trading days)
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
        df[label] = df[price_col].pct_change(period)
    
    # Shift to avoid look-ahead bias
    return_cols = list(periods.keys())
    df[return_cols] = df[return_cols].shift(periods=1)
    
    return df


def calculate_moving_averages(df: pd.DataFrame, price_col: str = 'close_Price',
                              periods: list = None) -> pd.DataFrame:
    """
    Calculate Simple Moving Averages (SMA) and Exponential Moving Averages (EMA).
    
    Args:
        df: DataFrame with price data
        price_col: Name of the price column
        periods: List of periods for MA calculation (default: [5, 20, 40, 120, 200])
        
    Returns:
        DataFrame with added MA columns (sma_N, ema_N)
    """
    if df.empty:
        raise ValueError("DataFrame cannot be empty")
    
    if price_col not in df.columns:
        raise ValueError(f"Column '{price_col}' not found")
    
    periods = periods or [5, 20, 40, 120, 200]
    
    # Calculate SMAs
    for period in periods:
        df.ta.sma(close=price_col, length=period, append=True)
    
    # Calculate EMAs
    for period in periods:
        df.ta.ema(close=price_col, length=period, append=True)
    
    # Remove duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Rename to match database schema
    rename_map = {}
    for period in periods:
        if f'SMA_{period}' in df.columns:
            rename_map[f'SMA_{period}'] = f'sma_{period}'
        if f'EMA_{period}' in df.columns:
            rename_map[f'EMA_{period}'] = f'ema_{period}'
    
    df.rename(columns=rename_map, inplace=True)
    
    # Shift to avoid look-ahead bias
    ma_cols = [f'sma_{p}' for p in periods if f'sma_{p}' in df.columns] + \
              [f'ema_{p}' for p in periods if f'ema_{p}' in df.columns]
    df[ma_cols] = df[ma_cols].shift(1)
    
    return df


def calculate_standard_deviation(df: pd.DataFrame, price_col: str = 'close_Price',
                                  periods: list = None) -> pd.DataFrame:
    """
    Calculate rolling standard deviation.
    
    Args:
        df: DataFrame with price data
        price_col: Name of the price column
        periods: List of periods (default: [5, 20, 40, 120, 200])
        
    Returns:
        DataFrame with added std_Div_N columns
    """
    if df.empty:
        raise ValueError("DataFrame cannot be empty")
    
    periods = periods or [5, 20, 40, 120, 200]
    
    for period in periods:
        df[f"std_Div_{period}"] = df[price_col].rolling(
            window=period, min_periods=period
        ).std()
    
    # Shift to avoid look-ahead bias
    std_cols = [f'std_Div_{p}' for p in periods]
    df[std_cols] = df[std_cols].shift(1)
    
    return df


def calculate_bollinger_bands(df: pd.DataFrame, periods: list = None) -> pd.DataFrame:
    """
    Calculate Bollinger Band width (4 * STD).
    
    Note: Requires std_Div columns to be calculated first.
    
    Args:
        df: DataFrame with std_Div columns
        periods: List of periods (default: [5, 20, 40, 120, 200])
        
    Returns:
        DataFrame with added bollinger_Band_N_2STD columns
    """
    if df.empty:
        raise ValueError("DataFrame cannot be empty")
    
    periods = periods or [5, 20, 40, 120, 200]
    
    for period in periods:
        std_col = f"std_Div_{period}"
        if std_col in df.columns:
            df[f"bollinger_Band_{period}_2STD"] = 4.0 * df[std_col]
    
    # Shift to avoid look-ahead bias
    bb_cols = [f'bollinger_Band_{p}_2STD' for p in periods 
               if f'bollinger_Band_{p}_2STD' in df.columns]
    df[bb_cols] = df[bb_cols].shift(1)
    
    return df


def calculate_rsi_macd_atr(df: pd.DataFrame, close_col: str = 'close_Price',
                           high_col: str = 'high_Price', low_col: str = 'low_Price',
                           rsi_period: int = 14, atr_period: int = 14) -> pd.DataFrame:
    """
    Calculate RSI, MACD, and ATR indicators.
    
    Args:
        df: DataFrame with price data
        close_col: Name of close price column
        high_col: Name of high price column
        low_col: Name of low price column
        rsi_period: Period for RSI calculation
        atr_period: Period for ATR calculation
        
    Returns:
        DataFrame with added indicator columns
    """
    if df.empty:
        raise ValueError("DataFrame cannot be empty")
    
    # RSI
    df.ta.rsi(close=close_col, length=rsi_period, append=True)
    
    # MACD
    df.ta.macd(close=close_col, append=True)
    
    # ATR
    df.ta.atr(high=high_col, low=low_col, close=close_col, length=atr_period, append=True)
    
    # Rename MACD columns
    rename_dict = {}
    if "MACDh_12_26_9" in df.columns:
        rename_dict["MACDh_12_26_9"] = "macd_histogram"
    if "MACDs_12_26_9" in df.columns:
        rename_dict["MACDs_12_26_9"] = "macd_signal"
    if "MACD_12_26_9" in df.columns:
        rename_dict["MACD_12_26_9"] = "macd"
    if "RSI_14" in df.columns:
        rename_dict["RSI_14"] = "rsi_14"
    if "ATR_14" in df.columns:
        rename_dict["ATR_14"] = "atr_14"
    if "ATRr_14" in df.columns:
        rename_dict["ATRr_14"] = "atr_14"
    
    if rename_dict:
        df.rename(columns=rename_dict, inplace=True)
    
    # Shift to avoid look-ahead bias
    indicator_cols = ["rsi_14", "macd", "macd_histogram", "macd_signal", "atr_14"]
    cols_to_shift = [c for c in indicator_cols if c in df.columns]
    if cols_to_shift:
        df[cols_to_shift] = df[cols_to_shift].shift(periods=1)
    
    return df


def calculate_volume_indicators(df: pd.DataFrame, price_col: str = 'close_Price',
                                volume_col: str = 'trade_Volume') -> pd.DataFrame:
    """
    Calculate volume-based technical indicators.
    
    Indicators:
        - Volume SMA (20-day)
        - Volume EMA (20-day)
        - Volume Ratio (current / 20-day SMA)
        - VWAP (Volume Weighted Average Price)
        - OBV (On-Balance Volume)
    
    Args:
        df: DataFrame with price and volume data
        price_col: Name of close price column
        volume_col: Name of volume column
        
    Returns:
        DataFrame with added volume indicator columns
    """
    if df.empty:
        raise ValueError("DataFrame cannot be empty")
    
    if volume_col not in df.columns:
        raise ValueError(f"Column '{volume_col}' not found")
    
    # Volume SMA (20-day)
    df["volume_sma_20"] = df[volume_col].rolling(window=20).mean()
    
    # Volume EMA (20-day)
    df["volume_ema_20"] = df[volume_col].ewm(span=20, adjust=False).mean()
    
    # Volume Ratio
    df["volume_ratio"] = df[volume_col] / df["volume_sma_20"]
    
    # VWAP (cumulative)
    df["vwap"] = (df[price_col] * df[volume_col]).cumsum() / df[volume_col].cumsum()
    
    # OBV (On-Balance Volume)
    price_change = df[price_col].diff()
    obv_direction = price_change.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    df["obv"] = (df[volume_col] * obv_direction).cumsum()
    
    # Shift to avoid look-ahead bias
    volume_cols = ["volume_sma_20", "volume_ema_20", "volume_ratio", "vwap", "obv"]
    df[volume_cols] = df[volume_cols].shift(periods=1)
    
    return df


def calculate_volatility(df: pd.DataFrame, price_col: str = 'close_Price') -> pd.DataFrame:
    """
    Calculate volatility metrics.
    
    Args:
        df: DataFrame with price data
        price_col: Name of close price column
        
    Returns:
        DataFrame with added volatility columns
    """
    if df.empty:
        raise ValueError("DataFrame cannot be empty")
    
    # Calculate returns
    returns = df[price_col].pct_change()
    
    # Rolling volatility (standard deviation of returns)
    df["volatility_5d"] = returns.rolling(window=5).std()
    df["volatility_20d"] = returns.rolling(window=20).std()
    df["volatility_60d"] = returns.rolling(window=60).std()
    
    # Shift to avoid look-ahead bias
    volatility_cols = ["volatility_5d", "volatility_20d", "volatility_60d"]
    df[volatility_cols] = df[volatility_cols].shift(periods=1)
    
    return df


def calculate_momentum(df: pd.DataFrame, price_col: str = 'close_Price') -> pd.DataFrame:
    """
    Calculate momentum indicator.
    
    Momentum tracks consecutive up/down days:
    - Positive momentum: consecutive days of price increase
    - Negative momentum: consecutive days of price decrease
    
    Args:
        df: DataFrame with price data
        price_col: Name of close price column
        
    Returns:
        DataFrame with added momentum column
    """
    if df.empty:
        raise ValueError("DataFrame cannot be empty")
    
    # Initialize momentum column
    df["momentum"] = 0.0
    
    # Calculate price change direction
    df["_price_change"] = df[price_col].diff()
    
    # Vectorized momentum calculation
    momentum_values = [0.0]  # First value is 0
    
    for i in range(1, len(df)):
        price_change = df.iloc[i]["_price_change"]
        prev_momentum = momentum_values[i - 1]
        
        if price_change >= 0:  # Price went up or stayed same
            if prev_momentum <= 0:
                momentum_values.append(1)
            else:
                momentum_values.append(prev_momentum + 1)
        else:  # Price went down
            if prev_momentum >= 0:
                momentum_values.append(-1)
            else:
                momentum_values.append(prev_momentum - 1)
    
    df["momentum"] = momentum_values
    
    # Clean up temporary column
    df.drop(columns=["_price_change"], inplace=True)
    
    # Shift to avoid look-ahead bias
    df["momentum"] = df["momentum"].shift(1)
    
    return df


def add_all_technical_indicators(df: pd.DataFrame, 
                                  is_index: bool = False,
                                  verbose: bool = True) -> pd.DataFrame:
    """
    Add all technical indicators to a price DataFrame.
    
    This is the main function to use for the complete indicator suite.
    
    Args:
        df: DataFrame with OHLCV price data (columns: date, open_Price, 
            high_Price, low_Price, close_Price, trade_Volume, ticker)
        is_index: If True, skip volatility indicators (for index tickers)
        verbose: Print progress messages
        
    Returns:
        DataFrame with all technical indicators added
    """
    if df.empty:
        raise ValueError("DataFrame cannot be empty")
    
    required_cols = ['close_Price', 'high_Price', 'low_Price', 'trade_Volume']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    if verbose:
        print("Calculating technical indicators...")
    
    # Period returns
    df = calculate_period_returns(df)
    if verbose:
        print("  ✓ Period returns")
    
    # RSI, MACD, ATR
    df = calculate_rsi_macd_atr(df)
    if verbose:
        print("  ✓ RSI, MACD, ATR")
    
    # Volume indicators
    df = calculate_volume_indicators(df)
    if verbose:
        print("  ✓ Volume indicators")
    
    # Volatility (skip for indices)
    if not is_index:
        df = calculate_volatility(df)
        if verbose:
            print("  ✓ Volatility")
    
    # Moving averages
    df = calculate_moving_averages(df)
    if verbose:
        print("  ✓ Moving averages")
    
    # Standard deviation
    df = calculate_standard_deviation(df)
    if verbose:
        print("  ✓ Standard deviation")
    
    # Bollinger Bands
    df = calculate_bollinger_bands(df)
    if verbose:
        print("  ✓ Bollinger Bands")
    
    # Momentum
    df = calculate_momentum(df)
    if verbose:
        print("  ✓ Momentum")
    
    return df


if __name__ == "__main__":
    # Demo with sample data
    print("Technical Indicators Module - Demo")
    print("-" * 40)
    
    # Create sample data
    dates = pd.date_range(start='2020-01-01', periods=300, freq='D')
    np.random.seed(42)
    
    sample_df = pd.DataFrame({
        'date': dates,
        'ticker': 'DEMO',
        'open_Price': 100 + np.cumsum(np.random.randn(300) * 0.5),
        'high_Price': 100 + np.cumsum(np.random.randn(300) * 0.5) + 1,
        'low_Price': 100 + np.cumsum(np.random.randn(300) * 0.5) - 1,
        'close_Price': 100 + np.cumsum(np.random.randn(300) * 0.5),
        'trade_Volume': np.random.randint(1000000, 10000000, 300)
    })
    
    # Calculate all indicators
    result_df = add_all_technical_indicators(sample_df)
    
    print(f"\nResult DataFrame shape: {result_df.shape}")
    print(f"Columns: {list(result_df.columns)}")
    print(f"\nSample of last 5 rows:")
    print(result_df.tail())
