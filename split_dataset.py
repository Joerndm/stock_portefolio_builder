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
import numpy as np
import pandas as pd

import db_interactions
import data_scalers

def dataset_train_test_split(dataset_dataframe, test_size=0.10, validation_size=0.20, rs=1):
    """
    Split the dataset into training, validation, and test data with proper scaling for both x and y values.

    Parameters:
    - dataset_dataframe (pandas.DataFrame): The dataset to split.
    - test_size (float): The size of the test data (default 0.20).
    - validation_size (float): The size of the validation data (default 0.15).
    - rs (int): The random state for the split.

    Returns:
    - scaler_x: The fitted MinMaxScaler for x values.
    - scaler_y: The fitted MinMaxScaler for y values.
    - numpy.ndarray: The scaled training data (x).
    - numpy.ndarray: The scaled validation data (x).
    - numpy.ndarray: The scaled test data (x).
    - numpy.ndarray: The scaled training labels (y).
    - numpy.ndarray: The scaled validation labels (y).
    - numpy.ndarray: The scaled test labels (y).
    - numpy.ndarray: The scaled prediction data (x for future predictions).

    Raises:
    - KeyError: If the dataset does not have the required columns.
    """

    try:
        # Drop the columns that are not needed
        drop_colum_list = ["date", "name", "date_published", "ticker", "currency", "financial_date_used"]
        for column in drop_colum_list:
            if column in dataset_dataframe.columns:
                dataset_dataframe = dataset_dataframe.drop([column], axis=1)

        train_data_df = dataset_dataframe.copy()
        forecast_out = int(math.ceil(0.05 * len(train_data_df)))
        train_data_df["prediction"] = train_data_df.iloc[0:-forecast_out]["1D"]

        # Exclude raw OHLCV columns that won't be available for future predictions
        exclude_cols = ["open_Price", "high_Price", "low_Price", "close_Price", "trade_Volume", "1D", "prediction"]
        x_all = train_data_df.drop(exclude_cols, axis=1, errors='ignore')

        # Separate prediction data (future data with no known y values)
        x_Predictions = x_all.iloc[-forecast_out:].copy()
        x = x_all.iloc[:-forecast_out].copy()

        # Drop rows with NaN values in prediction column
        train_data_df = train_data_df.dropna(subset=["prediction"], axis=0, how="any")
        y = train_data_df["prediction"].values.reshape(-1, 1)  # Reshape for scaler

        # Align x with y (remove rows that were dropped from y)
        x = x.loc[train_data_df.index].copy()

        # TIME-BASED SPLIT: preserve chronological order (no shuffling)
        n = len(x)
        train_end = int(n * (1 - test_size - validation_size))
        val_end = int(n * (1 - test_size))

        x_train = x.iloc[:train_end]
        x_val = x.iloc[train_end:val_end]
        x_test = x.iloc[val_end:]

        y_train = y[:train_end]
        y_val = y[train_end:val_end]
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
