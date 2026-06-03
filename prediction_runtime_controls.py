"""Helpers for recursive forecast stabilization and cached history lookups."""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd


RETURN_FEATURE_PERIODS = {
    "1M": (21, "1y"),
    "3M": (63, "1y"),
    "6M": (126, "1y"),
    "9M": (189, "1y"),
    "1Y": (252, "2y"),
    "2Y": (504, "3y"),
    "3Y": (756, "4y"),
    "4Y": (1008, "5y"),
    "5Y": (1260, "6y"),
}


def clip_scaled_features(scaled_features, min_value: float = -1.0, max_value: float = 2.0):
    """Clamp scaled features to a mild extrapolation band to limit OOD blowups."""
    if hasattr(scaled_features, "clip"):
        return scaled_features.clip(min_value, max_value)
    return np.clip(scaled_features, min_value, max_value)


def combine_flat_model_predictions(
    predictions: Mapping[str, float],
    *,
    weights: Mapping[str, float] | None = None,
    max_daily_return: float,
) -> dict[str, object]:
    """Clip flat-model returns and combine them with normalized available weights."""
    raw_predictions = {name: float(value) for name, value in predictions.items()}
    clipped_predictions = {
        name: float(np.clip(value, -max_daily_return, max_daily_return))
        for name, value in raw_predictions.items()
    }

    if weights is not None:
        active_weights = {
            name: float(weights.get(name, 0.0))
            for name in clipped_predictions
            if float(weights.get(name, 0.0)) > 0.0
        }
    else:
        active_weights = {}

    if active_weights:
        weight_sum = sum(active_weights.values())
        normalized_weights = {
            name: weight / weight_sum
            for name, weight in active_weights.items()
        }
    else:
        equal_weight = 1.0 / len(clipped_predictions) if clipped_predictions else 0.0
        normalized_weights = {
            name: equal_weight
            for name in clipped_predictions
        }

    ensemble_prediction = float(
        sum(normalized_weights[name] * clipped_predictions[name] for name in normalized_weights)
    )
    clipped_models = {
        name: (raw_predictions[name], clipped_predictions[name])
        for name in clipped_predictions
        if clipped_predictions[name] != raw_predictions[name]
    }
    return {
        "raw_predictions": raw_predictions,
        "clipped_predictions": clipped_predictions,
        "weights": normalized_weights,
        "ensemble_prediction": ensemble_prediction,
        "clipped_models": clipped_models,
    }


def summarize_scaled_feature_drift(
    scaled_features,
    *,
    feature_names: Sequence[str] | None = None,
    selected_features: Sequence[str] | None = None,
    warn_min: float = -0.1,
    warn_max: float = 1.1,
    clip_min: float = -1.0,
    clip_max: float = 2.0,
) -> dict[str, object]:
    """Summarize selected scaled features that drift outside training range or clip band."""
    if isinstance(scaled_features, pd.DataFrame):
        scaled_frame = scaled_features.copy()
    else:
        scaled_array = np.asarray(scaled_features)
        if scaled_array.ndim == 1:
            scaled_array = scaled_array.reshape(1, -1)
        resolved_feature_names = list(feature_names or [str(index) for index in range(scaled_array.shape[1])])
        scaled_frame = pd.DataFrame(scaled_array, columns=resolved_feature_names)

    if selected_features is not None:
        selected_columns = [feature for feature in selected_features if feature in scaled_frame.columns]
        scaled_frame = scaled_frame.loc[:, selected_columns]

    mild_out_of_range: list[tuple[str, float]] = []
    severe_out_of_range: list[tuple[str, float]] = []

    if scaled_frame.empty:
        return {
            "feature_count": 0,
            "mild_out_of_range": mild_out_of_range,
            "severe_out_of_range": severe_out_of_range,
        }

    numeric_row = scaled_frame.iloc[0].apply(pd.to_numeric, errors="coerce")
    for feature_name, value in numeric_row.items():
        if pd.isna(value):
            continue
        numeric_value = float(value)
        if numeric_value < warn_min or numeric_value > warn_max:
            target = (feature_name, numeric_value)
            if numeric_value < clip_min or numeric_value > clip_max:
                severe_out_of_range.append(target)
            else:
                mild_out_of_range.append(target)

    return {
        "feature_count": len(scaled_frame.columns),
        "mild_out_of_range": mild_out_of_range,
        "severe_out_of_range": severe_out_of_range,
    }


def required_history_periods(selected_features: Sequence[str]) -> list[str]:
    periods = {period for feature, (_days, period) in RETURN_FEATURE_PERIODS.items() if feature in selected_features}
    return sorted(periods)


def _normalize_downloaded_history(history_df) -> pd.DataFrame:
    if history_df is None:
        return pd.DataFrame(columns=["date", "close_Price"])

    normalized = pd.DataFrame(history_df).copy()
    if normalized.empty:
        return pd.DataFrame(columns=["date", "close_Price"])

    if isinstance(normalized.columns, pd.MultiIndex):
        if "Close" in normalized.columns.get_level_values(0):
            normalized = normalized["Close"]
        else:
            return pd.DataFrame(columns=["date", "close_Price"])

    if isinstance(normalized, pd.Series):
        normalized = normalized.to_frame(name="close_Price")

    if "Close" in normalized.columns:
        close_series = normalized["Close"]
    elif "close_Price" in normalized.columns:
        close_series = normalized["close_Price"]
    elif len(normalized.columns) == 1:
        close_series = normalized.iloc[:, 0]
    else:
        return pd.DataFrame(columns=["date", "close_Price"])

    normalized = close_series.reset_index()
    date_col = "Date" if "Date" in normalized.columns else normalized.columns[0]
    price_col = close_series.name if close_series.name in normalized.columns else normalized.columns[-1]
    normalized = normalized.rename(columns={date_col: "date", price_col: "close_Price"})
    normalized = normalized[["date", "close_Price"]]
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
    normalized["close_Price"] = pd.to_numeric(normalized["close_Price"], errors="coerce")
    normalized = normalized.dropna(subset=["date", "close_Price"])
    return normalized.reset_index(drop=True)


def build_prediction_history_cache(
    ticker: str,
    stock_mod_df: pd.DataFrame,
    selected_features: Sequence[str],
    history_fetcher: Callable[..., pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    periods = required_history_periods(selected_features)
    if not periods:
        return {}

    base_history = stock_mod_df[["date", "close_Price"]].copy()
    base_history["date"] = pd.to_datetime(base_history["date"], errors="coerce")
    base_history["close_Price"] = pd.to_numeric(base_history["close_Price"], errors="coerce")
    base_history = base_history.dropna(subset=["date", "close_Price"]).reset_index(drop=True)
    if base_history.empty:
        return {period: pd.DataFrame(columns=["date", "close_Price"]) for period in periods}

    start_date = base_history["date"].iloc[0]
    history_cache: dict[str, pd.DataFrame] = {}
    for period in periods:
        try:
            downloaded = history_fetcher(ticker, period=period, progress=False, auto_adjust=False)
            history_df = _normalize_downloaded_history(downloaded)
        except (ValueError, KeyError, ConnectionError, TimeoutError):
            history_df = pd.DataFrame(columns=["date", "close_Price"])

        if not history_df.empty:
            history_df = history_df.loc[history_df["date"] < start_date]
            history_df = pd.concat([history_df, base_history], ignore_index=True)
        else:
            history_df = base_history.copy()

        history_df = history_df.drop_duplicates(subset=["date"], keep="last")
        history_df = history_df.sort_values("date").reset_index(drop=True)
        history_cache[period] = history_df

    return history_cache


def apply_mean_reversion(
    prediction: float,
    historical_mean: float,
    historical_std: float,
    strength: float,
    threshold_std: float,
    hard_cap_std: float,
) -> float:
    if historical_std <= 0 or np.isnan(historical_std):
        return float(prediction)

    z_score = (prediction - historical_mean) / historical_std
    if abs(z_score) > threshold_std:
        reversion_factor = 1 - (strength * (abs(z_score) - threshold_std))
        reversion_factor = max(0.4, reversion_factor)
        prediction = historical_mean + (prediction - historical_mean) * reversion_factor

    max_prediction = historical_mean + hard_cap_std * historical_std
    min_prediction = historical_mean - hard_cap_std * historical_std
    return float(np.clip(prediction, min_prediction, max_prediction))


def apply_directional_balance(
    prediction: float,
    recent_predictions: Sequence[float],
    max_same_direction: int,
    random_uniform: Callable[[], float] = np.random.random,
) -> float:
    if len(recent_predictions) < max_same_direction:
        return float(prediction)

    recent_directions = [1 if value > 0 else -1 for value in recent_predictions[-max_same_direction:]]
    if not all(direction == recent_directions[0] for direction in recent_directions):
        return float(prediction)

    streak_length = max_same_direction
    for index in range(len(recent_predictions) - max_same_direction - 1, -1, -1):
        if (recent_predictions[index] > 0) == (recent_directions[0] > 0):
            streak_length += 1
        else:
            break

    base_correction = 0.50
    streak_bonus = min(0.30, (streak_length - max_same_direction) * 0.03)
    correction_strength = base_correction + streak_bonus

    if random_uniform() < correction_strength:
        dampening = max(0.3, 0.7 - (streak_length * 0.02))
        prediction = -prediction * dampening

    return float(prediction)


def stabilize_prediction(
    prediction: float,
    recent_predictions: Sequence[float],
    historical_mean: float,
    historical_std: float,
    prediction_config,
    random_uniform: Callable[[], float] = np.random.random,
) -> float:
    stabilized = apply_mean_reversion(
        prediction,
        historical_mean,
        historical_std,
        strength=prediction_config.mean_reversion_strength,
        threshold_std=prediction_config.mean_reversion_threshold_std,
        hard_cap_std=prediction_config.mean_reversion_hard_cap_std,
    )
    stabilized = apply_directional_balance(
        stabilized,
        recent_predictions,
        max_same_direction=prediction_config.max_same_direction_days,
        random_uniform=random_uniform,
    )
    return float(np.clip(stabilized, -prediction_config.max_daily_return, prediction_config.max_daily_return))