"""
Technical Pattern Detection for Stock Portfolio Builder.

Detects binary (0/1) technical analysis patterns from price and indicator data.
These signals capture market psychology and regime changes without suffering
from mode collapse (unlike continuous features).

All detection functions accept a DataFrame with at minimum 'close_Price' and
return integer columns (0 or 1) suitable for ML feature input.

Patterns implemented:
    - Golden Cross / Death Cross (SMA 50/200 crossover)
    - RSI Oversold / Overbought
    - MACD Crossover (bullish / bearish)
    - Bollinger Squeeze (low volatility regime)
    - Support / Resistance Break
    - Volume Spike
"""

import pandas as pd


# ── Cross-over helpers ─────────────────────────────────────────────────────

def _crossover(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    """True on the bar where *series_a* crosses above *series_b*."""
    return (series_a > series_b) & (series_a.shift(1) <= series_b.shift(1))


def _crossunder(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    """True on the bar where *series_a* crosses below *series_b*."""
    return (series_a < series_b) & (series_a.shift(1) >= series_b.shift(1))


# ── Pattern detectors ──────────────────────────────────────────────────────

def detect_golden_cross(df: pd.DataFrame, short: int = 50, long: int = 200,
                        signal_window: int = 20) -> pd.Series:
    """Detect Golden Cross: SMA(short) crosses above SMA(long).

    The signal stays active for *signal_window* bars after the cross.
    """
    sma_short = df["close_Price"].rolling(short, min_periods=short).mean()
    sma_long = df["close_Price"].rolling(long, min_periods=long).mean()

    cross = _crossover(sma_short, sma_long).astype(float)
    signal = cross.rolling(signal_window, min_periods=1).max()
    return signal.fillna(0).astype(int)


def detect_death_cross(df: pd.DataFrame, short: int = 50, long: int = 200,
                       signal_window: int = 20) -> pd.Series:
    """Detect Death Cross: SMA(short) crosses below SMA(long)."""
    sma_short = df["close_Price"].rolling(short, min_periods=short).mean()
    sma_long = df["close_Price"].rolling(long, min_periods=long).mean()

    cross = _crossunder(sma_short, sma_long).astype(float)
    signal = cross.rolling(signal_window, min_periods=1).max()
    return signal.fillna(0).astype(int)


def detect_rsi_oversold(df: pd.DataFrame, threshold: int = 30,
                        rsi_col: str = "rsi_14") -> pd.Series:
    """1 when RSI is below *threshold* (oversold territory)."""
    if rsi_col not in df.columns:
        return pd.Series(0, index=df.index)
    return (df[rsi_col] < threshold).astype(int)


def detect_rsi_overbought(df: pd.DataFrame, threshold: int = 70,
                          rsi_col: str = "rsi_14") -> pd.Series:
    """1 when RSI is above *threshold* (overbought territory)."""
    if rsi_col not in df.columns:
        return pd.Series(0, index=df.index)
    return (df[rsi_col] > threshold).astype(int)


def detect_macd_bullish_crossover(df: pd.DataFrame,
                                  macd_col: str = "macd",
                                  signal_col: str = "macd_signal",
                                  window: int = 10) -> pd.Series:
    """1 when MACD crosses above its signal line (bullish). Active for *window* bars."""
    if macd_col not in df.columns or signal_col not in df.columns:
        return pd.Series(0, index=df.index)
    cross = _crossover(df[macd_col], df[signal_col]).astype(float)
    return cross.rolling(window, min_periods=1).max().fillna(0).astype(int)


def detect_macd_bearish_crossover(df: pd.DataFrame,
                                  macd_col: str = "macd",
                                  signal_col: str = "macd_signal",
                                  window: int = 10) -> pd.Series:
    """1 when MACD crosses below its signal line (bearish). Active for *window* bars."""
    if macd_col not in df.columns or signal_col not in df.columns:
        return pd.Series(0, index=df.index)
    cross = _crossunder(df[macd_col], df[signal_col]).astype(float)
    return cross.rolling(window, min_periods=1).max().fillna(0).astype(int)


def detect_bollinger_squeeze(df: pd.DataFrame, period: int = 20,
                             percentile: float = 0.20,
                             threshold_window: int = 120) -> pd.Series:
    """1 when Bollinger Band width is in the lowest *percentile* of a recent window.

    A squeeze indicates low volatility, often preceding a breakout.
    """
    col = f"bollinger_Band_{period}_2STD"
    if col in df.columns:
        bw = df[col]
    else:
        std = df["close_Price"].rolling(period, min_periods=period).std()
        bw = 4 * std  # Upper - Lower = 4 * std (2-std bands)

    threshold_window = max(int(threshold_window), period)
    threshold = bw.rolling(
        threshold_window,
        min_periods=threshold_window,
    ).quantile(percentile)
    return (bw <= threshold).fillna(False).astype(int)


def detect_support_break(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """1 when price breaks below the rolling *lookback*-day low (support)."""
    rolling_low = df["close_Price"].rolling(lookback, min_periods=lookback).min()
    # Break = today's close < yesterday's rolling low
    return (df["close_Price"] < rolling_low.shift(1)).fillna(False).astype(int)


def detect_resistance_break(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """1 when price breaks above the rolling *lookback*-day high (resistance)."""
    rolling_high = df["close_Price"].rolling(lookback, min_periods=lookback).max()
    return (df["close_Price"] > rolling_high.shift(1)).fillna(False).astype(int)


def detect_volume_spike(df: pd.DataFrame, multiplier: float = 2.0,
                        period: int = 20,
                        volume_col: str = "trade_Volume") -> pd.Series:
    """1 when volume exceeds *multiplier* × the *period*-day average."""
    if volume_col not in df.columns:
        return pd.Series(0, index=df.index)
    vol_avg = df[volume_col].rolling(period, min_periods=period).mean()
    return (df[volume_col] > multiplier * vol_avg).fillna(False).astype(int)


# ── Convenience: add all patterns at once ──────────────────────────────────

PATTERN_COLUMNS = [
    "pat_golden_cross",
    "pat_death_cross",
    "pat_rsi_oversold",
    "pat_rsi_overbought",
    "pat_macd_bullish",
    "pat_macd_bearish",
    "pat_bollinger_squeeze",
    "pat_support_break",
    "pat_resistance_break",
    "pat_volume_spike",
]


def add_all_technical_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical pattern columns to *df* (in-place) and return it.

    New columns are prefixed with ``pat_`` to distinguish them from
    continuous technical indicators.
    """
    df["pat_golden_cross"] = detect_golden_cross(df)
    df["pat_death_cross"] = detect_death_cross(df)
    df["pat_rsi_oversold"] = detect_rsi_oversold(df)
    df["pat_rsi_overbought"] = detect_rsi_overbought(df)
    df["pat_macd_bullish"] = detect_macd_bullish_crossover(df)
    df["pat_macd_bearish"] = detect_macd_bearish_crossover(df)
    df["pat_bollinger_squeeze"] = detect_bollinger_squeeze(df)
    df["pat_support_break"] = detect_support_break(df)
    df["pat_resistance_break"] = detect_resistance_break(df)
    df["pat_volume_spike"] = detect_volume_spike(df)
    return df
