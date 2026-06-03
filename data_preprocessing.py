"""
Data Preprocessing Utilities for Stock Portfolio Builder.

Provides feature-level preprocessing steps that sit between raw data loading
and model training:

    1. Remove highly correlated features  (#9)
    2. Remove outlier samples              (#10)
    3. Walk-forward validation splits      (#11)

All functions are pure transforms — they accept arrays / DataFrames and return
cleaned versions without side-effects.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import zscore

logger = logging.getLogger(__name__)


# ── #9  Remove Correlated Features ─────────────────────────────────────────

def remove_correlated_features(
    df: pd.DataFrame,
    threshold: float = 0.85,
    protect_columns: List[str] | None = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Drop features whose pairwise Pearson correlation exceeds *threshold*.

    When two features are highly correlated the one that appears later in
    the column order is dropped (standard upper-triangle approach).

    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix (all numeric).
    threshold : float
        Absolute correlation above which to drop (default 0.85).
    protect_columns : list[str] | None
        Columns that should never be dropped even if correlated.

    Returns
    -------
    df_reduced : pd.DataFrame
        DataFrame with correlated columns removed.
    dropped : list[str]
        Names of the dropped columns.
    """
    if threshold <= 0 or threshold > 1:
        return df, []

    protect = set(protect_columns or [])
    corr_matrix = df.corr(numeric_only=True).abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
    )

    to_drop = []
    for col in upper.columns:
        if col in protect:
            continue
        if any(upper[col] > threshold):
            to_drop.append(col)

    if to_drop:
        logger.info("Removing %d correlated features (threshold=%.2f): %s",
                     len(to_drop), threshold, to_drop)
    return df.drop(columns=to_drop), to_drop


# ── #10  Outlier Removal ───────────────────────────────────────────────────

def remove_outliers(
    x: np.ndarray,
    y: np.ndarray,
    threshold: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove samples where any feature has an absolute z-score > *threshold*.

    Parameters
    ----------
    x : np.ndarray
        Feature matrix (n_samples, n_features).
    y : np.ndarray
        Target array (n_samples,).
    threshold : float
        Z-score cutoff (default 3.0). Set to 0 to disable.

    Returns
    -------
    x_clean, y_clean : np.ndarray
        Filtered arrays with outlier rows removed.
    """
    if threshold <= 0:
        return x, y

    z = np.abs(zscore(x, axis=0, nan_policy="omit"))
    # Replace NaN z-scores (constant columns) with 0 so they don't trigger removal
    z = np.nan_to_num(z, nan=0.0)
    mask = (z < threshold).all(axis=1)

    n_removed = (~mask).sum()
    if n_removed > 0:
        logger.info("Outlier removal: dropped %d / %d samples (z > %.1f)",
                     n_removed, len(x), threshold)
    return x[mask], y[mask]


# ── #11  Walk-Forward Validation ───────────────────────────────────────────

def walk_forward_splits(
    n_samples: int,
    min_train_years: int = 2,
    test_window_years: int = 1,
    step_years: int = 1,
    trading_days_per_year: int = 252,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate walk-forward (expanding window) train/test index pairs.

    Example with 5 years of data, min_train=2, test_window=1, step=1::

        Fold 1: Train [0 .. 504)  Test [504 .. 756)
        Fold 2: Train [0 .. 756)  Test [756 .. 1008)
        Fold 3: Train [0 .. 1008) Test [1008 .. 1260)

    Parameters
    ----------
    n_samples : int
        Total number of samples in the dataset.
    min_train_years : int
        Minimum training window in years.
    test_window_years : int
        Size of each test window in years.
    step_years : int
        How many years to advance between folds.
    trading_days_per_year : int
        Trading days per calendar year (default 252).

    Returns
    -------
    splits : list of (train_indices, test_indices)
        Each element is a tuple of numpy arrays.
    """
    min_train = min_train_years * trading_days_per_year
    test_win = test_window_years * trading_days_per_year
    step = step_years * trading_days_per_year

    splits = []
    train_end = min_train

    while train_end + test_win <= n_samples:
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(train_end, min(train_end + test_win, n_samples))
        splits.append((train_idx, test_idx))
        train_end += step

    if not splits:
        logger.warning(
            "Not enough data for walk-forward validation "
            "(need %d samples, have %d). Falling back to single split.",
            min_train + test_win, n_samples,
        )

    return splits
