"""Shared preprocessing contract for model training and prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

import dimension_reduction
import split_dataset


class InsufficientDataError(ValueError):
    """Raised when cleaned stock history is too short for the configured pipeline."""


@dataclass
class PreparedModelingData:
    """Prepared data contract shared by model_trainer and price_predictor."""

    stock_data_df: pd.DataFrame
    scaler_x: Any
    scaler_y: Any
    x_training_dataset_df: pd.DataFrame
    x_val_dataset_df: pd.DataFrame
    x_test_dataset_df: pd.DataFrame
    x_prediction_dataset_df: pd.DataFrame
    selected_features_model: Any
    selected_features_list: list[str]
    y_train_scaled: np.ndarray
    y_val_scaled: np.ndarray
    y_test_scaled: np.ndarray
    y_train_unscaled: np.ndarray
    y_val_unscaled: np.ndarray
    y_test_unscaled: np.ndarray
    rows_before_cleaning: int
    rows_after_cleaning: int
    min_rows_required: int

    @property
    def dropped_rows(self) -> int:
        return self.rows_before_cleaning - self.rows_after_cleaning


def required_min_rows(time_steps: int, min_rows_floor: int) -> int:
    """Return the shared minimum row requirement for training and prediction."""
    return max(min_rows_floor, time_steps + 50)


def recommended_feature_count(
    train_sample_count: int,
    available_feature_count: int,
    min_samples_per_feature: int = 10,
) -> int:
    """Cap selected features so the training split keeps a minimum sample/feature ratio."""
    if available_feature_count <= 0:
        return 0

    max_feature_count = max(1, train_sample_count // max(min_samples_per_feature, 1))
    return min(available_feature_count, max_feature_count)


def _clean_stock_dataset(stock_data_df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    cleaned_df = stock_data_df.copy()
    cleaned_df["date"] = pd.to_datetime(cleaned_df["date"])
    rows_before_cleaning = len(cleaned_df)

    cleaned_df = cleaned_df.dropna(axis=1, how="all")

    always_required = [
        "date",
        "ticker",
        "close_Price",
        "open_Price",
        "high_Price",
        "low_Price",
    ]
    critical_cols = [column for column in always_required if column in cleaned_df.columns]
    cleaned_df = cleaned_df.dropna(subset=critical_cols)

    feature_cols = [column for column in cleaned_df.columns if column not in ("date", "ticker")]
    if feature_cols:
        cleaned_df.loc[:, feature_cols] = cleaned_df[feature_cols].ffill().bfill()

    cleaned_df = cleaned_df.dropna(axis=0, how="any")
    cleaned_df = cleaned_df.dropna(axis=1, how="any")

    rows_after_cleaning = len(cleaned_df)
    return cleaned_df, rows_before_cleaning, rows_after_cleaning


def prepare_modeling_data(
    stock_data_df: pd.DataFrame,
    *,
    time_steps: int,
    validation_size: float,
    test_size: float,
    min_rows_floor: int,
    dataset_splitter: Callable[..., tuple[Any, ...]] = split_dataset.dataset_train_test_split,
    feature_selector: Callable[..., tuple[Any, ...]] = dimension_reduction.feature_selection_rf,
) -> PreparedModelingData:
    """Clean, split, scale, and reduce the feature space for both pipeline phases."""
    cleaned_df, rows_before_cleaning, rows_after_cleaning = _clean_stock_dataset(stock_data_df)
    min_rows_required = required_min_rows(time_steps, min_rows_floor)

    if len(cleaned_df) < min_rows_required:
        raise InsufficientDataError(
            f"Insufficient data: {len(cleaned_df)} rows after cleaning "
            f"(need >= {min_rows_required} for time_steps={time_steps})"
        )

    scaler_x, scaler_y, x_train_scaled, x_val_scaled, x_test_scaled, \
        y_train_scaled, y_val_scaled, y_test_scaled, x_predictions = dataset_splitter(
            cleaned_df,
            test_size,
            validation_size=validation_size,
        )

    y_train_unscaled = scaler_y.inverse_transform(y_train_scaled.reshape(-1, 1)).flatten()
    y_val_unscaled = scaler_y.inverse_transform(y_val_scaled.reshape(-1, 1)).flatten()
    y_test_unscaled = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()

    x_training_data = pd.DataFrame(x_train_scaled)
    x_val_data = pd.DataFrame(x_val_scaled)
    x_test_data = pd.DataFrame(x_test_scaled)
    y_training_data_df = pd.Series(y_train_unscaled)
    y_val_data_df = pd.Series(y_val_unscaled)
    y_test_data_df = pd.Series(y_test_unscaled)
    prediction_data = x_predictions

    feature_amount = recommended_feature_count(
        train_sample_count=len(x_training_data),
        available_feature_count=len(x_training_data.columns),
    )

    x_training_dataset, x_val_dataset, x_test_dataset, x_prediction_dataset, \
        selected_features_model, selected_features_list = feature_selector(
            feature_amount,
            x_training_data,
            x_val_data,
            x_test_data,
            y_training_data_df,
            y_val_data_df,
            y_test_data_df,
            prediction_data,
            cleaned_df,
        )

    x_training_dataset_df = pd.DataFrame(x_training_dataset, columns=selected_features_list)
    x_val_dataset_df = pd.DataFrame(x_val_dataset, columns=selected_features_list)
    x_test_dataset_df = pd.DataFrame(x_test_dataset, columns=selected_features_list)
    x_prediction_dataset_df = pd.DataFrame(x_prediction_dataset, columns=selected_features_list)

    return PreparedModelingData(
        stock_data_df=cleaned_df,
        scaler_x=scaler_x,
        scaler_y=scaler_y,
        x_training_dataset_df=x_training_dataset_df,
        x_val_dataset_df=x_val_dataset_df,
        x_test_dataset_df=x_test_dataset_df,
        x_prediction_dataset_df=x_prediction_dataset_df,
        selected_features_model=selected_features_model,
        selected_features_list=list(selected_features_list),
        y_train_scaled=np.asarray(y_train_scaled),
        y_val_scaled=np.asarray(y_val_scaled),
        y_test_scaled=np.asarray(y_test_scaled),
        y_train_unscaled=np.asarray(y_train_unscaled),
        y_val_unscaled=np.asarray(y_val_unscaled),
        y_test_unscaled=np.asarray(y_test_unscaled),
        rows_before_cleaning=rows_before_cleaning,
        rows_after_cleaning=rows_after_cleaning,
        min_rows_required=min_rows_required,
    )