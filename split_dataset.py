"""
Dataset splitting utilities for stock portfolio analysis.

This module provides functionality to split stock market datasets into training,
validation, and test sets with proper scaling. It handles both feature (x) and
target (y) data preparation for machine learning models, ensuring no data leakage
between sets.

Key features:
- Train/validation/test split with configurable proportions
- MinMax scaling for features and targets
- Separate prediction dataset for future forecasting
- Automatic handling of temporal data and missing values

Functions:
    dataset_train_test_split: Split and scale dataset into train/val/test sets
"""

import math
import os
import numpy as np
import pandas as pd

import data_scalers

def dataset_train_test_split(dataset_dataframe, test_size=0.10, validation_size=0.20, rs=1,
                             horizon=None):
    """
    Split the dataset chronologically into training, validation, and test data
    with proper scaling for both x and y values.

    The prediction target is the forward return over `horizon` trading days:
    prediction(t) = close_Price(t + horizon) / close_Price(t) - 1. Features in
    row t only contain information available at day t, so there is no same-row
    or look-ahead leakage between features and target. With horizon=1 (the
    default) this is the next trading day's return.

    The horizon is resolved in priority order:
      1. The explicit `horizon` argument, if given.
      2. The TARGET_HORIZON_DAYS environment variable, if set. This allows
         horizon experiments across the whole pipeline (trainer, scheduler,
         tests) without threading a parameter through every call site.
      3. Default: 1 (next-day return).

    IMPORTANT: hyperparameter caches are NOT keyed by horizon. Models trained
    on different horizons must not be mixed — when changing the horizon,
    retrain affected tickers from scratch (model_trainer.py --max-age 0) and
    treat their previous metrics as belonging to a different task.

    With horizon > 1, an embargo of horizon-1 rows is applied at the
    train/val and val/test boundaries: overlapping forward-return windows
    would otherwise leak future information across subsets. At horizon=1
    the embargo is zero and split sizes are unaffected.

    The input dataframe must be sorted by date in ascending order (oldest row
    first). If a 'date' column is present, this is verified and a ValueError
    is raised on violation.

    Parameters:
    - dataset_dataframe (pandas.DataFrame): The dataset to split, sorted by
      date ascending. Must contain a 'close_Price' column.
    - test_size (float): The fraction of rows used for the test set (default 0.10).
    - validation_size (float): The fraction of rows used for the validation set (default 0.20).
    - rs (int): DEPRECATED and unused. The split is chronological (no
      shuffling), so no random state is involved. Kept for backward
      compatibility with existing callers.
    - horizon (int or None): Forward-return horizon in trading days. None
      (default) resolves via the TARGET_HORIZON_DAYS environment variable,
      falling back to 1. Must be >= 1.

    Returns:
    - scaler_x: The fitted MinMaxScaler for x values (fit on training data only).
    - scaler_y: The fitted MinMaxScaler for y values (fit on training data only).
    - numpy.ndarray: The scaled training data (x).
    - numpy.ndarray: The scaled validation data (x).
    - numpy.ndarray: The scaled test data (x).
    - numpy.ndarray: The scaled training labels (y).
    - numpy.ndarray: The scaled validation labels (y).
    - numpy.ndarray: The scaled test labels (y).
    - pandas.DataFrame: The scaled prediction frame (most recent rows,
      reserved for downstream historical blending and future forecasting).

    Raises:
    - KeyError: If the dataset does not have the required columns.
    - ValueError: If the dataset is not sorted by date in ascending order.
    """

    try:
        # GUARD: the split and the forward-shifted target below assume the
        # dataframe is sorted in ascending chronological order. Verify while
        # the date column is still available, before it gets dropped.
        if "date" in dataset_dataframe.columns:
            date_series = pd.to_datetime(dataset_dataframe["date"])
            if not date_series.is_monotonic_increasing:
                raise ValueError(
                    "dataset_dataframe must be sorted by 'date' in ascending order "
                    "before splitting (oldest row first)."
                )

        # Drop the columns that are not needed
        drop_colum_list = ["date", "name", "date_published", "ticker", "currency", "financial_date_used"]
        for column in drop_colum_list:
            if column in dataset_dataframe.columns:
                dataset_dataframe = dataset_dataframe.drop([column], axis=1)

        train_data_df = dataset_dataframe.copy()
        forecast_out = int(math.ceil(0.05 * len(train_data_df)))

        # Resolve the target horizon: explicit arg > env var > default 1
        if horizon is None:
            env_horizon = os.environ.get("TARGET_HORIZON_DAYS", "").strip()
            horizon = int(env_horizon) if env_horizon else 1
        horizon = int(horizon)
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")

        # TARGET CONSTRUCTION (forward-shifted, no same-row leakage):
        # prediction(t) = close(t + horizon) / close(t) - 1
        # (the forward return over the next `horizon` trading days)
        #
        # Features in row t are computed from prices up to and including day t,
        # so the target must lie strictly in the future relative to row t.
        # NOTE: the stored "1D" column is a *lagged* return (pct_change(1)
        # shifted +1 in stock_data_fetch/technical_indicators) and is a valid
        # feature, but it must never be used as the target — that would ask
        # the model to reconstruct a past value already embedded in the
        # features (SMAs/EMAs contain close(t-1)), inflating test metrics.
        train_data_df["prediction"] = (
            train_data_df["close_Price"].pct_change(horizon).shift(-horizon)
        )

        # Exclude raw OHLCV columns that won't be available for future predictions
        exclude_cols = ["open_Price", "high_Price", "low_Price", "close_Price", "trade_Volume", "1D", "prediction"]
        x_all = train_data_df.drop(exclude_cols, axis=1, errors='ignore')

        # Reserve the most recent forecast_out rows as the prediction frame
        # used downstream for historical blending and future forecasting.
        # These rows stay out of train/val/test, preserving their
        # pseudo-out-of-sample role.
        x_Predictions = x_all.iloc[-forecast_out:].copy()
        train_data_df = train_data_df.iloc[:-forecast_out]

        # Drop rows with NaN targets (the final reserved row has no known
        # next-day return by construction; guard also covers edge cases)
        train_data_df = train_data_df.dropna(subset=["prediction"], axis=0, how="any")
        y = train_data_df["prediction"].values.reshape(-1, 1)  # Reshape for scaler

        # Align x with y (remove rows that were dropped from y)
        x = x_all.loc[train_data_df.index].copy()

        # TIME-BASED SPLIT: preserve chronological order (no shuffling)
        n = len(x)
        train_end = int(n * (1 - test_size - validation_size))
        val_end = int(n * (1 - test_size))

        # EMBARGO (purge) at split boundaries:
        # With horizon > 1, consecutive rows' targets overlap horizon-1 days,
        # so the last rows of one subset share future information with the
        # first rows of the next (e.g. the final training targets peek into
        # the validation window). Dropping horizon-1 rows at the end of train
        # and of val removes that boundary leakage. At horizon=1 the embargo
        # is 0 and the split is unchanged.
        embargo = horizon - 1

        x_train = x.iloc[:max(train_end - embargo, 0)]
        x_val = x.iloc[train_end:max(val_end - embargo, train_end)]
        x_test = x.iloc[val_end:]

        y_train = y[:max(train_end - embargo, 0)]
        y_val = y[train_end:max(val_end - embargo, train_end)]
        y_test = y[val_end:]

        # Fit x scaler on TRAINING data only (prevent data leakage)
        scaler_x = data_scalers.data_preprocessing_minmax_scaler_fit(x_train)
        scaler_x.set_output(transform="pandas")

        # Transform all x datasets using the scaler fit on training data
        x_train_scaled = data_scalers.data_preprocessing_minmax_scaler_transform(scaler_x, x_train)
        x_val_scaled = data_scalers.data_preprocessing_minmax_scaler_transform(scaler_x, x_val)
        x_test_scaled = data_scalers.data_preprocessing_minmax_scaler_transform(scaler_x, x_test)
        x_Predictions = data_scalers.data_preprocessing_minmax_scaler_transform(scaler_x, x_Predictions)

        # Fit y scaler on TRAINING data only (prevent data leakage)
        scaler_y = data_scalers.data_preprocessing_minmax_scaler_fit(y_train)

        # Transform all y datasets using the scaler fit on training data
        y_train = data_scalers.data_preprocessing_minmax_scaler_transform(scaler_y, y_train).flatten()
        y_val = data_scalers.data_preprocessing_minmax_scaler_transform(scaler_y, y_val).flatten()
        y_test = data_scalers.data_preprocessing_minmax_scaler_transform(scaler_y, y_test).flatten()

        return scaler_x, scaler_y, x_train_scaled, x_val_scaled, x_test_scaled, y_train, y_val, y_test, x_Predictions

    except KeyError as e:
        raise KeyError("Dataset does not have the required columns.") from e

# Run the main function
if __name__ == "__main__":
    import db_interactions  # local import: only needed for this demo block
    stock_data_df = db_interactions.import_stock_dataset("BAVA.CO")
    print(stock_data_df.info())
    print("stock_data_df")
    scaler_x, scaler_y, x_training_data, x_val_data, x_test_data, y_training_data, y_val_data, y_test_data, prediction_data = dataset_train_test_split(
        stock_data_df, test_size=0.20, validation_size=0.15, rs=1
    )

    print("\n=== TRAINING DATA ===")
    print("x_training_data shape:", x_training_data.shape)
    print("y_training_data shape:", y_training_data.shape)
    print("x_training_data:")
    print(x_training_data)
    # print column names in x_training_data
    print("Column names in x_training_data:", x_training_data.columns.tolist())
    print("y_training_data:")
    print(y_training_data)

    print("\n=== VALIDATION DATA ===")
    print("x_val_data shape:", x_val_data.shape)
    print("y_val_data shape:", y_val_data.shape)
    print("x_val_data:")
    print(x_val_data)
    print("y_val_data:")
    print(y_val_data)

    print("\n=== TEST DATA ===")
    print("x_test_data shape:", x_test_data.shape)
    print("y_test_data shape:", y_test_data.shape)
    print("x_test_data:")
    print(x_test_data)
    print("y_test_data:")
    print(y_test_data)

    print("\n=== PREDICTION DATA ===")
    print("prediction_data shape:", prediction_data.shape)
    print("prediction_data:")
    print(prediction_data)