"""
Helpers for normalizing and rebuilding cached model hyperparameters.

This module keeps constructor-safe cache handling out of orchestration-heavy
training code.
"""

from __future__ import annotations

from typing import Any, Mapping

from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


RF_CACHE_KEYS = (
    'n_estimators',
    'max_depth',
    'min_samples_split',
    'min_samples_leaf',
    'criterion',
    'bootstrap',
    'max_features',
    'max_samples',
)

XGB_CACHE_KEYS = (
    'n_estimators',
    'max_depth',
    'learning_rate',
    'subsample',
    'colsample_bytree',
    'min_child_weight',
    'gamma',
    'reg_alpha',
    'reg_lambda',
)

RIDGE_CACHE_KEYS = (
    'alpha',
    'solver',
)

SVR_CACHE_KEYS = (
    'kernel',
    'C',
    'gamma',
    'epsilon',
)


def _load_keras_callback_types() -> tuple[Any, Any]:
    try:
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    except Exception as exc:
        raise ImportError(
            "TensorFlow/Keras callbacks are required to fit cached sequence models."
        ) from exc
    return EarlyStopping, ReduceLROnPlateau


def _coerce_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {'1', 'true', 'yes', 'y'}
    return bool(value)


def _coerce_int(value: Any) -> Any:
    if value in (None, ''):
        return None
    return int(value)


def _coerce_float(value: Any) -> Any:
    if value in (None, '', 'none', 'null'):
        return None
    return float(value)


def _coerce_float_or_keyword(value: Any, keywords: set[str] | None = None) -> Any:
    if value in (None, '', 'none', 'null'):
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {'none', 'null', ''}:
            return None
        if keywords and lowered in keywords:
            return lowered
        return float(lowered)
    return float(value)


def _coerce_choice(value: Any, default: str) -> str:
    if value in (None, ''):
        return default
    return str(value).strip().lower()


def _normalize_max_features(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {'sqrt', 'log2'}:
            return lowered
        if lowered in {'none', 'null', ''}:
            return None
        try:
            return float(lowered)
        except ValueError:
            return value
    return value


def _extract_estimator_params(model_or_params: Any, nested_attr: str | None = None) -> Mapping[str, Any]:
    if nested_attr and hasattr(model_or_params, nested_attr):
        nested = getattr(model_or_params, nested_attr)
        if hasattr(nested, 'get_params'):
            return nested.get_params(deep=False)

    if isinstance(model_or_params, Mapping):
        if nested_attr and nested_attr in model_or_params and hasattr(model_or_params[nested_attr], 'get_params'):
            return model_or_params[nested_attr].get_params(deep=False)
        return model_or_params

    if hasattr(model_or_params, 'get_params'):
        params = model_or_params.get_params(deep=False)
        if nested_attr and isinstance(params, Mapping) and nested_attr in params and hasattr(params[nested_attr], 'get_params'):
            return params[nested_attr].get_params(deep=False)
        return params

    return dict(model_or_params or {})


def normalize_random_forest_hyperparameters(hyperparameters: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return Random Forest hyperparameters in a constructor-safe form."""
    params = dict(hyperparameters or {})

    if 'max_sample' in params and 'max_samples' not in params:
        params['max_samples'] = params['max_sample']

    bootstrap = _coerce_bool(params.get('bootstrap', True), default=True)
    max_samples = _coerce_float(params.get('max_samples'))

    normalized = {
        'n_estimators': _coerce_int(params.get('n_estimators', 500)) or 500,
        'max_depth': _coerce_int(params.get('max_depth')),
        'min_samples_split': _coerce_int(params.get('min_samples_split', 5)) or 5,
        'min_samples_leaf': _coerce_int(params.get('min_samples_leaf', 2)) or 2,
        'criterion': params.get('criterion', 'squared_error'),
        'bootstrap': bootstrap,
        'max_features': _normalize_max_features(params.get('max_features', 'sqrt')),
        'max_samples': max_samples,
    }

    if not normalized['bootstrap']:
        normalized['max_samples'] = None

    return normalized


def serialize_random_forest_hyperparameters(model_or_params: Any) -> dict[str, Any]:
    """Serialize only the constructor-safe Random Forest parameters used by cache restore."""
    if hasattr(model_or_params, 'get_params'):
        params = model_or_params.get_params(deep=False)
    else:
        params = model_or_params

    normalized = normalize_random_forest_hyperparameters(params)
    return {key: normalized.get(key) for key in RF_CACHE_KEYS}


def build_cached_random_forest_model(hyperparameters: Mapping[str, Any] | None) -> RandomForestRegressor:
    """Rebuild an unfitted RandomForestRegressor from cached hyperparameters."""
    params = normalize_random_forest_hyperparameters(hyperparameters)
    return RandomForestRegressor(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        min_samples_split=params['min_samples_split'],
        min_samples_leaf=params['min_samples_leaf'],
        criterion=params['criterion'],
        bootstrap=params['bootstrap'],
        max_features=params['max_features'],
        max_samples=params['max_samples'],
        random_state=42,
        n_jobs=-1,
    )


def normalize_xgboost_hyperparameters(hyperparameters: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return XGBoost hyperparameters in a constructor-safe form."""
    params = dict(hyperparameters or {})

    gamma = _coerce_float(params.get('gamma'))
    reg_alpha = _coerce_float(params.get('reg_alpha'))
    reg_lambda = _coerce_float(params.get('reg_lambda'))

    return {
        'n_estimators': _coerce_int(params.get('n_estimators', 500)) or 500,
        'max_depth': _coerce_int(params.get('max_depth', 6)) or 6,
        'learning_rate': _coerce_float(params.get('learning_rate', 0.1)) or 0.1,
        'subsample': _coerce_float(params.get('subsample', 0.8)) or 0.8,
        'colsample_bytree': _coerce_float(params.get('colsample_bytree', 0.8)) or 0.8,
        'min_child_weight': _coerce_int(params.get('min_child_weight', 3)) or 3,
        'gamma': 0.0 if gamma is None else gamma,
        'reg_alpha': 0.0 if reg_alpha is None else reg_alpha,
        'reg_lambda': 0.0 if reg_lambda is None else reg_lambda,
    }


def serialize_xgboost_hyperparameters(model_or_params: Any) -> dict[str, Any]:
    """Serialize only the constructor-safe XGBoost parameters used by cache restore."""
    params = _extract_estimator_params(model_or_params)
    normalized = normalize_xgboost_hyperparameters(params)
    return {key: normalized.get(key) for key in XGB_CACHE_KEYS}


def build_cached_xgboost_model(hyperparameters: Mapping[str, Any] | None) -> Any:
    """Rebuild an unfitted XGBoost regressor from cached hyperparameters."""
    import xgboost as xgb

    params = normalize_xgboost_hyperparameters(hyperparameters)
    return xgb.XGBRegressor(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        learning_rate=params['learning_rate'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        min_child_weight=params['min_child_weight'],
        gamma=params['gamma'],
        reg_alpha=params['reg_alpha'],
        reg_lambda=params['reg_lambda'],
        random_state=42,
        n_jobs=-1,
        tree_method='hist',
    )


def normalize_ridge_hyperparameters(hyperparameters: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return Ridge hyperparameters in a constructor-safe form."""
    params = dict(hyperparameters or {})
    alpha = _coerce_float(params.get('alpha', 1.0))

    return {
        'alpha': 1.0 if alpha is None else alpha,
        'solver': _coerce_choice(params.get('solver', 'auto'), default='auto'),
    }


def serialize_ridge_hyperparameters(model_or_params: Any) -> dict[str, Any]:
    """Serialize only the constructor-safe Ridge parameters used by cache restore."""
    params = _extract_estimator_params(model_or_params)
    normalized = normalize_ridge_hyperparameters(params)
    return {key: normalized.get(key) for key in RIDGE_CACHE_KEYS}


def build_cached_ridge_model(hyperparameters: Mapping[str, Any] | None) -> Ridge:
    """Rebuild an unfitted Ridge regressor from cached hyperparameters."""
    params = normalize_ridge_hyperparameters(hyperparameters)
    return Ridge(
        alpha=params['alpha'],
        solver=params['solver'],
        random_state=42,
        max_iter=10000,
    )


def normalize_svr_hyperparameters(hyperparameters: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return SVR hyperparameters in a constructor-safe form."""
    params = dict(hyperparameters or {})
    gamma = _coerce_float_or_keyword(params.get('gamma', 0.1), keywords={'scale', 'auto'})

    return {
        'kernel': _coerce_choice(params.get('kernel', 'rbf'), default='rbf'),
        'C': _coerce_float(params.get('C', 1.0)) or 1.0,
        'gamma': 0.1 if gamma is None else gamma,
        'epsilon': _coerce_float(params.get('epsilon', 0.1)) or 0.1,
    }


def serialize_svr_hyperparameters(model_or_params: Any) -> dict[str, Any]:
    """Serialize only the wrapped SVR parameters used by cache restore."""
    params = _extract_estimator_params(model_or_params, nested_attr='regressor_')
    if 'kernel' not in params:
        params = _extract_estimator_params(model_or_params, nested_attr='regressor')
    normalized = normalize_svr_hyperparameters(params)
    return {key: normalized.get(key) for key in SVR_CACHE_KEYS}


def build_cached_svr_model(hyperparameters: Mapping[str, Any] | None) -> TransformedTargetRegressor:
    """Rebuild an unfitted wrapped SVR regressor from cached hyperparameters."""
    params = normalize_svr_hyperparameters(hyperparameters)
    svr = SVR(
        kernel=params['kernel'],
        C=params['C'],
        gamma=params['gamma'],
        epsilon=params['epsilon'],
        max_iter=10000,
    )
    return TransformedTargetRegressor(regressor=svr, transformer=StandardScaler())


def invalidate_hyperparameter_cache(ticker: str, model_type: str, reason: Any) -> int:
    """Invalidate one cached hyperparameter row and emit a consistent log line."""
    try:
        from db_interactions import invalidate_hyperparameters

        invalidated = invalidate_hyperparameters(ticker=ticker, model_type=model_type)
        print(f"[CACHE] Invalidated {model_type.upper()} cached hyperparameters after restore failure: {reason}")
        return invalidated
    except Exception as exc:
        print(f"[CACHE] Failed to invalidate {model_type.upper()} cached hyperparameters: {exc}")
        return 0


def build_lstm_cache_callbacks(hyperparameters: Mapping[str, Any] | None) -> list[Any]:
    """Recreate the main LSTM callback policy from cached hyperparameters."""
    EarlyStopping, ReduceLROnPlateau = _load_keras_callback_types()
    params = dict(hyperparameters or {})
    patience = _coerce_int(params.get('patience')) or 20
    monitor_metric = 'val_mean_absolute_error'
    callbacks: list[Any] = [
        EarlyStopping(
            monitor=monitor_metric,
            patience=patience,
            restore_best_weights=True,
            verbose=0,
            min_delta=0.0001,
        )
    ]

    lr_schedule_choice = params.get('lr_schedule', 'none')
    if lr_schedule_choice == 'reduce_on_plateau':
        callbacks.append(
            ReduceLROnPlateau(
                monitor=monitor_metric,
                factor=0.5,
                patience=max(1, patience // 2),
                verbose=0,
                min_lr=1e-7,
            )
        )
    elif lr_schedule_choice == 'exp_decay':
        from keras.callbacks import LearningRateScheduler

        initial_lr = _coerce_float(params.get('learning_rate')) or 0.001
        decay_rate = _coerce_float(params.get('decay_rate')) or 0.95

        def lr_schedule(epoch: int, lr: float) -> float:
            return initial_lr * (decay_rate ** epoch)

        callbacks.append(LearningRateScheduler(lr_schedule))

    return callbacks


def fit_cached_lstm_model(
    model: Any,
    hyperparameters: Mapping[str, Any] | None,
    x_train: Any,
    y_train: Any,
    x_val: Any,
    y_val: Any,
    epochs: int,
    verbose: int = 0,
) -> Any:
    """Fit an LSTM rebuilt from cached hyperparameters without rerunning tuning."""
    params = dict(hyperparameters or {})
    batch_size = _coerce_int(params.get('batch_size')) or 32
    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        callbacks=build_lstm_cache_callbacks(params),
        verbose=verbose,
    )
    return model


def build_tcn_cache_callbacks(hyperparameters: Mapping[str, Any] | None) -> list[Any]:
    """Recreate the main TCN callback policy from cached hyperparameters."""
    EarlyStopping, ReduceLROnPlateau = _load_keras_callback_types()
    params = dict(hyperparameters or {})
    patience = _coerce_int(params.get('tcn_patience')) or 20
    monitor_metric = 'val_mean_absolute_error'
    return [
        EarlyStopping(
            monitor=monitor_metric,
            patience=patience,
            restore_best_weights=True,
            verbose=0,
            min_delta=0.0001,
        ),
        ReduceLROnPlateau(
            monitor=monitor_metric,
            factor=0.5,
            patience=max(1, patience // 2),
            verbose=0,
            min_lr=1e-7,
        ),
    ]


def fit_cached_tcn_model(
    model: Any,
    hyperparameters: Mapping[str, Any] | None,
    x_train: Any,
    y_train: Any,
    x_val: Any,
    y_val: Any,
    epochs: int,
    verbose: int = 0,
) -> Any:
    """Fit a TCN rebuilt from cached hyperparameters without rerunning tuning."""
    params = dict(hyperparameters or {})
    batch_size = _coerce_int(params.get('batch_size')) or 32
    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        callbacks=build_tcn_cache_callbacks(params),
        verbose=verbose,
    )
    return model