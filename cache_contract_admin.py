"""Admin helpers for reconciling cached flat-model hyperparameters with current features."""

from __future__ import annotations

import hashlib
import logging
from typing import Sequence

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import db_interactions
from model_cache_utils import (
    build_cached_random_forest_model,
    build_cached_ridge_model,
    build_cached_svr_model,
    build_cached_xgboost_model,
    serialize_random_forest_hyperparameters,
    serialize_ridge_hyperparameters,
    serialize_svr_hyperparameters,
    serialize_xgboost_hyperparameters,
)
from model_pipeline_preprocessing import InsufficientDataError, prepare_modeling_data
from pipeline_config import get_data_config

logger = logging.getLogger(__name__)

SUPPORTED_CACHE_MODEL_TYPES = ("rf", "xgb", "ridge", "svr")
CACHE_MODEL_BUILDERS = {
    "rf": build_cached_random_forest_model,
    "xgb": build_cached_xgboost_model,
    "ridge": build_cached_ridge_model,
    "svr": build_cached_svr_model,
}
CACHE_MODEL_SERIALIZERS = {
    "rf": serialize_random_forest_hyperparameters,
    "xgb": serialize_xgboost_hyperparameters,
    "ridge": serialize_ridge_hyperparameters,
    "svr": serialize_svr_hyperparameters,
}


def _feature_hash(feature_list: Sequence[str]) -> str:
    feature_str = ",".join(sorted(str(feature) for feature in feature_list))
    return hashlib.sha256(feature_str.encode()).hexdigest()[:64]


def _normalize_model_types(model_types: Sequence[str] | None) -> list[str]:
    if model_types is None:
        return list(SUPPORTED_CACHE_MODEL_TYPES)

    normalized = []
    for model_type in model_types:
        lowered = str(model_type).lower()
        if lowered not in SUPPORTED_CACHE_MODEL_TYPES:
            raise ValueError(
                f"Unsupported cache refresh model type: {model_type}. "
                f"Supported values: {', '.join(SUPPORTED_CACHE_MODEL_TYPES)}"
            )
        if lowered not in normalized:
            normalized.append(lowered)
    return normalized


def _normalize_tickers(tickers: Sequence[str]) -> list[str]:
    normalized = []
    seen = set()
    for ticker in tickers:
        resolved = str(ticker).strip()
        if not resolved or resolved in seen:
            continue
        seen.add(resolved)
        normalized.append(resolved)
    return normalized


def _evaluate_flat_model(model, x_val_df, y_val_series) -> dict[str, float | None]:
    predictions = np.asarray(model.predict(x_val_df)).reshape(-1)
    y_true = np.asarray(y_val_series).reshape(-1)
    r2_value = None
    if len(y_true) > 1:
        r2_value = float(r2_score(y_true, predictions))
    return {
        "mse": float(mean_squared_error(y_true, predictions)),
        "mae": float(mean_absolute_error(y_true, predictions)),
        "r2": r2_value,
    }


def refresh_ticker_cache_contract(
    ticker: str,
    *,
    model_types: Sequence[str] | None = None,
    time_steps: int | None = None,
    validate_prediction: bool = False,
) -> dict[str, object]:
    """Refresh cached flat-model rows for one ticker against the current feature contract."""
    data_cfg = get_data_config()
    resolved_model_types = _normalize_model_types(model_types)
    if time_steps is None:
        time_steps = data_cfg.time_steps

    result = {
        "ticker": ticker,
        "success": False,
        "feature_hash": None,
        "num_features": 0,
        "requested_model_types": resolved_model_types,
        "refreshed_models": [],
        "unchanged_models": [],
        "failed_models": {},
        "validation": None,
        "error_message": None,
    }

    try:
        stock_data_df = db_interactions.import_stock_dataset(ticker)
        prepared_data = prepare_modeling_data(
            stock_data_df,
            time_steps=time_steps,
            validation_size=data_cfg.validation_size,
            test_size=data_cfg.test_size,
            min_rows_floor=data_cfg.min_rows_floor,
        )
    except InsufficientDataError as exc:
        result["error_message"] = str(exc)
        return result

    selected_features = list(prepared_data.selected_features_list)
    current_hash = _feature_hash(selected_features)
    result["feature_hash"] = current_hash
    result["num_features"] = len(selected_features)

    cache_rows = db_interactions.get_hyperparameter_cache_rows(
        ticker,
        model_types=resolved_model_types,
        valid_only=True,
    )
    if not cache_rows:
        result["error_message"] = "No valid cached flat-model rows found"
        return result

    for row in cache_rows:
        model_type = row["model_type"]
        row_hash = row.get("feature_hash")
        row_num_features = row.get("num_features")
        if row_hash == current_hash and row_num_features == len(selected_features):
            result["unchanged_models"].append(model_type)
            continue

        try:
            model = CACHE_MODEL_BUILDERS[model_type](row.get("hyperparameters"))
            model.fit(prepared_data.x_training_dataset_df, prepared_data.y_train_unscaled)
            metrics = _evaluate_flat_model(
                model,
                prepared_data.x_val_dataset_df,
                prepared_data.y_val_unscaled,
            )
            serialized_hyperparameters = CACHE_MODEL_SERIALIZERS[model_type](model)
            save_ok = db_interactions.save_hyperparameters(
                ticker=ticker,
                model_type=model_type,
                hyperparameters=serialized_hyperparameters,
                num_trials=row.get("num_trials"),
                best_score=row.get("best_score"),
                tuning_time_seconds=row.get("tuning_time_seconds"),
                training_samples=len(prepared_data.x_training_dataset_df),
                num_features=len(selected_features),
                feature_list=selected_features,
                val_mse=metrics["mse"],
                val_r2=metrics["r2"],
                val_mae=metrics["mae"],
                is_constrained=bool(row.get("is_constrained", False)),
            )
            if not save_ok:
                raise RuntimeError(f"Failed to save refreshed {model_type} cache row")
            result["refreshed_models"].append(model_type)
        except Exception as exc:
            db_interactions.invalidate_hyperparameters(ticker=ticker, model_type=model_type)
            result["failed_models"][model_type] = f"{type(exc).__name__}: {exc}"
            logger.warning("[CACHE] Failed refreshing %s %s: %s", ticker, model_type, exc)

    if validate_prediction and not result["failed_models"]:
        import price_predictor

        result["validation"] = price_predictor.predict_single_stock(
            stock_symbol=ticker,
            time_steps=time_steps,
        )
        if not result["validation"].get("success", False):
            result["error_message"] = result["validation"].get("error_message") or "Prediction validation failed"

    validation_ok = True
    if validate_prediction:
        validation_ok = bool(result["validation"] and result["validation"].get("success", False))

    result["success"] = not result["failed_models"] and validation_ok and (
        bool(result["refreshed_models"]) or bool(result["unchanged_models"])
    )
    if not result["success"] and result["error_message"] is None and result["failed_models"]:
        result["error_message"] = "One or more cache rows failed to refresh"
    return result


def refresh_cache_contracts(
    tickers: Sequence[str],
    *,
    model_types: Sequence[str] | None = None,
    time_steps: int | None = None,
    validate_prediction: bool = False,
) -> dict[str, object]:
    """Refresh cached flat-model feature contracts for one or more tickers."""
    normalized_tickers = _normalize_tickers(tickers)
    if not normalized_tickers:
        raise ValueError("At least one ticker is required for cache contract refresh")

    results = {}
    successful_tickers = []
    failed_tickers = []
    for ticker in normalized_tickers:
        ticker_result = refresh_ticker_cache_contract(
            ticker,
            model_types=model_types,
            time_steps=time_steps,
            validate_prediction=validate_prediction,
        )
        results[ticker] = ticker_result
        if ticker_result["success"]:
            successful_tickers.append(ticker)
        else:
            failed_tickers.append(ticker)

    return {
        "requested_tickers": normalized_tickers,
        "successful_tickers": successful_tickers,
        "failed_tickers": failed_tickers,
        "results": results,
    }
