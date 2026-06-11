"""
Machine Learning Model Builder for Stock Price Prediction

This module provides comprehensive machine learning infrastructure for predicting stock price changes
using ensemble methods combining LSTM, Random Forest, and XGBoost models. It includes automated
hyperparameter tuning, overfitting detection, model retraining, and performance analysis.

Key Features:
-------------
- **Multi-Model Ensemble**: Combines LSTM (deep learning), Random Forest, and XGBoost predictions
- **Automated Hyperparameter Tuning**: Uses Keras Tuner with Bayesian Optimization and Hyperband
- **Overfitting Detection**: Multi-metric detection system with automatic retraining
- **Adaptive Training**: Dynamically adjusts search space when overfitting is detected
- **Early Stopping**: Prevents wasted computation when hyperparameter search converges
- **Sequence Modeling**: LSTM with bidirectional layers and batch normalization
- **Feature Importance Analysis**: Identifies most predictive features
- **Data Health Diagnostics**: Pre-training validation of dataset quality
- **Future Price Prediction**: Day-by-day forecasting with dynamic feature recalculation
- **Performance Analysis**: Comprehensive comparison of predicted vs actual values

Models:
-------
1. **LSTM (Long Short-Term Memory)**:
   - Bidirectional architecture with multiple stacked layers
   - Batch normalization for training stability
   - L2 regularization and dropout for overfitting prevention
   - Tunable loss functions (MAE, MSE, Huber, MAPE)
   - Gradient clipping to prevent exploding gradients
   - Operates on scaled data (MinMaxScaler)

2. **Random Forest Regressor**:
   - Ensemble of decision trees with bagging
   - Feature importance ranking
   - Tunable max_depth, min_samples_split, max_features
   - Bootstrap sampling for generalization
   - Operates on unscaled data (scale-invariant)

3. **XGBoost Regressor**:
   - Gradient boosting with regularization
   - L1/L2 regularization for feature selection
   - Tunable learning rate, subsample, colsample_bytree
   - Histogram-based tree construction for speed
   - Operates on unscaled data

Hyperparameter Tuning:
---------------------
- **LSTM**: Bayesian Optimization with up to 50 trials
  - Layer count, units per layer, dropout rates
  - Learning rate, optimizer (Adam/RMSprop), clipnorm
  - Loss function, batch size, early stopping patience

- **Random Forest**: Hyperband with up to 100 trials
  - n_estimators, max_depth, min_samples_split/leaf
  - max_features, bootstrap, max_samples
  - Criterion (squared_error, absolute_error, friedman_mse)

- **XGBoost**: Bayesian Optimization with up to 60 trials
  - n_estimators, max_depth, learning_rate
  - subsample, colsample_bytree, min_child_weight
  - gamma, reg_alpha (L1), reg_lambda (L2)

Overfitting Detection:
---------------------
Multi-metric system comparing train/validation/test performance:
- **MSE Degradation**: Tracks increase from train→val→test
- **R² Degradation**: Tracks decrease from train→val→test
- **MAE Degradation**: Tracks increase from train→val→test
- **Consistency Score**: Detects metric disagreement
- **Combined Score**: Weighted average (35% MSE + 25% R² + 30% MAE + 10% consistency)

If overfitting detected (score > threshold):
1. Apply constrained search space (stricter regularization)
2. Increase hyperparameter trials
3. Retrain up to max_retrains times (default: 150)
4. Early stop if hyperparameters converge (3 consecutive identical)

Data Requirements:
-----------------
- **Training**: Minimum 15% of dataset (default: 70%)
- **Validation**: 15-25% of dataset (default: 20%)
- **Test**: 5-15% of dataset (default: 10%)
- **LSTM Sequences**: Minimum time_steps + 1 samples (default: 31)
- **Features**: Automatically selected via Random Forest feature importance

Prediction Workflow:
-------------------
1. **Historical Predictions**: Re-predict on test set using pre-scaled data
2. **Future Predictions**: Day-by-day forecasting with feature recalculation
   - Dynamic features: Returns, moving averages, RSI, MACD, volatility
   - Static features: Fundamental ratios (P/E, P/S, P/B, P/FCF)
   - Ensemble: Weighted average of LSTM, RF, XGBoost predictions

Performance Metrics:
-------------------
- **MSE** (Mean Squared Error): Lower is better
- **MAE** (Mean Absolute Error): Lower is better
- **R²** (R-squared): Higher is better (max 1.0)
- **Direction Accuracy**: % of correct up/down predictions
- **Price Error**: Absolute and percentage error in price predictions

Usage Example:
-------------
>>> # Train models with automatic overfitting detection
>>> models, history, lstm_data = train_and_validate_models(
...     stock_symbol='AAPL',
...     x_train=x_train_scaled,
...     x_val=x_val_scaled,
...     x_test=x_test_scaled,
...     y_train_scaled=y_train_scaled,
...     y_val_scaled=y_val_scaled,
...     y_test_scaled=y_test_scaled,
...     y_train_unscaled=y_train_unscaled,
...     y_val_unscaled=y_val_unscaled,
...     y_test_unscaled=y_test_unscaled,
...     time_steps=30,
...     max_retrains=150,
...     overfitting_threshold=0.15
... )
>>>
>>> # Predict future prices
>>> forecast = predict_future_price_changes(
...     ticker='AAPL',
...     scaler_x=scaler_x,
...     scaler_y=scaler_y,
...     model={'lstm': models['lstm'], 'rf': models['rf'], 'xgb': models['xgb']},
...     selected_features_list=features,
...     stock_df=stock_data,
...     prediction_days=90,
...     time_steps=30
... )

Dependencies:
------------
- pandas, numpy: Data manipulation
- yfinance: Stock data download
- scikit-learn: ML models, metrics, preprocessing
- tensorflow/keras: Deep learning (LSTM)
- keras-tuner: Hyperparameter optimization
- xgboost: Gradient boosting
- matplotlib: Visualization

Authors:
--------
Joern (joerndm)

License:
--------
See repository license

Notes:
------
- GPU acceleration recommended for LSTM training
- Tuning results cached in 'tuning_dir' for resumable training
- Historical predictions use pre-scaled test data for consistency
- Future predictions recalculate dynamic features day-by-day
- Ensemble weights optimized on validation set performance
"""
import os, shutil, time, io, sys
import datetime
import tempfile
import warnings
from typing import Any

import pandas as pd
from dateutil.relativedelta import relativedelta
import yfinance as yf
import numpy as np
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'int'):
    np.int = int
import matplotlib
import matplotlib.pyplot as plt
try:
    import xgboost as xgb
except ImportError:
    class _MissingXGBoostModule:
        class XGBRegressor:
            def __init__(self, *args, **kwargs):
                raise ImportError("XGBoost not installed. Install with: pip install xgboost")

    xgb = _MissingXGBoostModule()
from sklearn.svm import SVR
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization, Input, Conv1D, Add, LayerNormalization, GlobalAveragePooling1D, Activation
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers
import keras_tuner as kt
from keras_tuner.tuners import SklearnTuner as Sklearn

matplotlib.use('Agg')
pd.set_option('future.no_silent_downcasting', True)

import split_dataset
import dimension_reduction
import monte_carlo_sim
from gpu_runtime_utils import configure_tensorflow_gpu
from model_cache_utils import (
    build_cached_random_forest_model,
    build_cached_ridge_model,
    build_cached_svr_model,
    build_cached_xgboost_model,
    fit_cached_lstm_model,
    fit_cached_tcn_model,
    invalidate_hyperparameter_cache,
    serialize_random_forest_hyperparameters,
    serialize_ridge_hyperparameters,
    serialize_svr_hyperparameters,
    serialize_xgboost_hyperparameters,
)
from pipeline_config import get_pred_config
from prediction_cache_contract import (
    PredictionTrainingRequiredError,
    raise_cache_invalidated_error,
)
from prediction_runtime_controls import (
    build_prediction_history_cache,
    clip_scaled_features,
    combine_flat_model_predictions,
    stabilize_prediction,
    summarize_scaled_feature_drift,
)


CACHED_HP_LOAD_EXCEPTIONS = (ImportError,)
CACHED_MODEL_RESTORE_EXCEPTIONS = (
    AttributeError,
    ImportError,
    KeyError,
    RuntimeError,
    TypeError,
    ValueError,
)
HYPERPARAMETER_SAVE_EXCEPTIONS = (ImportError, KeyError, ValueError)
TUNER_CHECKPOINT_RESTORE_EXCEPTIONS = (
    OSError,
    RuntimeError,
    ValueError,
    tf.errors.NotFoundError,
)

# Force UTF-8 logging globally
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="ignore")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="ignore")


def _coerce_hparam_choice(value, allowed_values, default):
    if value in allowed_values:
        return value

    if isinstance(value, str):
        normalized_lookup = {
            str(allowed).lower(): allowed
            for allowed in allowed_values
            if isinstance(allowed, str)
        }
        normalized_value = normalized_lookup.get(value.strip().lower())
        if normalized_value is not None:
            return normalized_value

    return default


def _coerce_hparam_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_hparam_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_hparam_bool(value, default):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {'true', '1', 'yes', 'y'}:
            return True
        if normalized in {'false', '0', 'no', 'n'}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _restore_or_refit_sequence_tuner_model(
    tuner,
    best_hp,
    build_model,
    fit_model,
    x_train,
    y_train,
    x_val,
    y_val,
    time_steps,
    num_features,
    epochs,
    model_label,
):
    try:
        best_models = tuner.get_best_models(num_models=1) if hasattr(tuner, 'get_best_models') else []
    except TUNER_CHECKPOINT_RESTORE_EXCEPTIONS as restore_error:
        print(
            f"[WARN] Failed to restore best {model_label} model from tuner checkpoint, "
            f"rebuilding from best hyperparameters: {restore_error}"
        )
        best_models = []

    if best_models:
        return best_models[0]

    best_model = build_model(best_hp, input_shape=(time_steps, num_features))
    best_model.build(input_shape=(None, time_steps, num_features))
    fit_model(
        best_model,
        dict(best_hp.values),
        x_train,
        y_train,
        x_val,
        y_val,
        epochs=epochs,
        verbose=0,
    )
    print(f"[OK] Rebuilt and fit {model_label} model from best hyperparameters")
    return best_model


def _fit_svr_without_convergence_warnings(model, x_values, y_values):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model.fit(x_values, y_values)
    return model


def _run_svr_tuning_search_without_convergence_warnings(tuner, x_values, y_values):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        tuner.search(x_values, y_values)


def _alignment_start_from_sequence_predictions(target_values, sequence_predictions, split_label):
    alignment_start = len(target_values) - len(sequence_predictions)
    if alignment_start < 0:
        raise ValueError(
            f"Sequence predictions for {split_label} exceed target length: "
            f"targets={len(target_values)}, predictions={len(sequence_predictions)}"
        )
    return alignment_start

def build_random_forest_model(hp, constrain_for_overfitting=False):
    """
    Builds a RandomForestRegressor with expanded tunable hyperparameters.
    
    Parameters:
    - hp: Keras Tuner hyperparameters object
    - constrain_for_overfitting: If True, apply stricter constraints to reduce overfitting
    """
    
    max_features_choice = _coerce_hparam_choice(
        hp.Choice('max_features', ['sqrt', 'log2', '0.3', '0.5', '0.8']),
        ['sqrt', 'log2', '0.3', '0.5', '0.8'],
        'sqrt',
    )
    
    if max_features_choice in ['0.3', '0.5', '0.8']:
        max_features_value = float(max_features_choice)
    else:
        max_features_value = max_features_choice
    
    # Always register all hyperparameters to avoid KeyError when Hyperband
    # replays trials from different brackets/rounds
    bootstrap = _coerce_hparam_bool(hp.Boolean('bootstrap', default=True), True)
    max_samples_value = _coerce_hparam_float(hp.Float('max_samples', 0.5, 1.0, step=0.1, default=0.8), 0.8)
    max_depth_value = _coerce_hparam_int(hp.Int('max_depth', 3, 50, step=2, default=15), 15)
    min_samples_leaf_value = _coerce_hparam_choice(
        hp.Choice('min_samples_leaf', [1, 2, 4, 8, 16], default=2),
        [1, 2, 4, 8, 16],
        2,
    )
    min_samples_split_value = _coerce_hparam_choice(
        hp.Choice('min_samples_split', [2, 5, 10, 15, 20], default=5),
        [2, 5, 10, 15, 20],
        5,
    )
    
    if constrain_for_overfitting:
        # Override with stricter values to reduce overfitting
        bootstrap = True
        max_samples_value = max(0.6, min(max_samples_value, 0.9))
        max_depth_value = min(max_depth_value, 30)
        min_samples_leaf_value = max(min_samples_leaf_value, 2)
        min_samples_split_value = max(min_samples_split_value, 5)
    
    # If bootstrap is disabled, max_samples must be None
    if not bootstrap:
        max_samples_value = None

    model = RandomForestRegressor(
        n_estimators=hp.Int('n_estimators', 100, 1500, step=100),
        max_depth=max_depth_value,
        min_samples_split=min_samples_split_value,
        min_samples_leaf=min_samples_leaf_value,
        criterion=_coerce_hparam_choice(
            hp.Choice('criterion', ['squared_error', 'absolute_error', 'friedman_mse']),
            ['squared_error', 'absolute_error', 'friedman_mse'],
            'squared_error',
        ),
        bootstrap=bootstrap,
        max_features=max_features_value,
        max_samples=max_samples_value,
        random_state=42,
        n_jobs=-1
    )
    return model

def tune_random_forest_model(stock_symbol, x_training_dataset_df, y_training_dataset_df, x_val_dataset_df, y_val_dataset_df, max_trials=20, constrain_for_overfitting=False, use_cached_hp=True, cache_max_age_days=30, cleanup_after_tuning=True, require_cached_hp=False):
    """
    Improved Random Forest tuning with better hyperparameters and optimization.
    Uses validation set for hyperparameter selection (industry standard).
    
    Parameters:
    - stock_symbol (str): The stock ticker symbol
    - x_training_dataset_df (pd.DataFrame): Training dataset (ALREADY SCALED)
    - y_training_dataset_df (pd.Series or np.ndarray): Training labels (UNSCALED for RF)
    - x_val_dataset_df (pd.DataFrame): Validation dataset (ALREADY SCALED)
    - y_val_dataset_df (pd.Series or np.ndarray): Validation labels (UNSCALED for RF)
    - max_trials (int): Maximum number of tuning trials
    - use_cached_hp (bool): Whether to use cached hyperparameters if available
    - cache_max_age_days (int): Maximum age of cached HPs in days
    - cleanup_after_tuning (bool): Whether to delete tuning directory after training
    
    Returns:
    - best_rf_model: The tuned Random Forest model
    """
    import time as time_module
    tuning_start_time = time_module.time()
    
    # Convert to numpy arrays to avoid feature name warnings
    x_train = x_training_dataset_df.values
    y_train = y_training_dataset_df.values
    x_val = x_val_dataset_df.values
    y_val = y_val_dataset_df.values
    feature_list = list(x_training_dataset_df.columns)

    # Try to load cached hyperparameters
    if use_cached_hp:
        cached_hp = None
        try:
            import db_interactions
            cached_hp = db_interactions.load_hyperparameters(
                ticker=stock_symbol,
                model_type='rf',
                max_age_days=cache_max_age_days,
                feature_list=feature_list,
                require_same_features=True
            )
            
        except CACHED_HP_LOAD_EXCEPTIONS as e:
            if require_cached_hp:
                raise PredictionTrainingRequiredError(
                    ticker=stock_symbol,
                    required_model_types=['rf'],
                    missing_model_types=['rf'],
                    message=f"Cached RF hyperparameters are unavailable for {stock_symbol}: {e}",
                ) from e
            print(f"[CACHE] Could not load cached HP: {e}, proceeding with tuning")

        if cached_hp:
            print(f"[CACHE] Using cached RF hyperparameters for {stock_symbol}")
            try:
                best_rf_model = build_cached_random_forest_model(cached_hp)
                best_rf_model.fit(x_train, y_train)
                return best_rf_model
            except CACHED_MODEL_RESTORE_EXCEPTIONS as e:
                db_interactions.invalidate_hyperparameters(ticker=stock_symbol, model_type='rf')
                if require_cached_hp:
                    raise_cache_invalidated_error(
                        stock_symbol,
                        'rf',
                        e,
                        invalidator=invalidate_hyperparameter_cache,
                    )
                print(f"[CACHE] Could not restore cached RF HP: {e}, proceeding with tuning")

    if require_cached_hp:
        raise PredictionTrainingRequiredError(
            ticker=stock_symbol,
            required_model_types=['rf'],
            missing_model_types=['rf'],
        )

    # Define the MSE scorer
    mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

    # Setup directory structure - use temp directory to avoid OneDrive locking issues
    temp_dir = os.path.join(tempfile.gettempdir(), "rf_tuning_dir")
    project_name_val = f"RF_tuning_{stock_symbol}"
    project_path = os.path.join(temp_dir, project_name_val)

    # Also clean up any old tuning_dir entries in workspace (legacy)
    legacy_path = os.path.join("tuning_dir", project_name_val)
    if os.path.exists(legacy_path):
        try:
            shutil.rmtree(legacy_path)
            print(f"[CLEANUP] Deleted legacy tuning directory: {legacy_path}")
        except (OSError, PermissionError) as e:
            print(f"[WARN] Could not delete legacy tuning directory: {e}")

    # Determine overwrite: if stale/incomplete directory exists (no oracle.json), clean it
    overwrite_val = True
    if os.path.exists(project_path):
        oracle_path = os.path.join(project_path, "oracle.json")
        if os.path.exists(oracle_path):
            print(f"[RESUME] Found existing RF tuning at {project_path}. Continuing...")
            overwrite_val = False
        else:
            print("[WARN] Found incomplete RF tuning directory (no oracle.json). Starting fresh...")
            try:
                shutil.rmtree(project_path)
            except (OSError, PermissionError) as e:
                print(f"[WARN] Could not delete incomplete tuning directory: {e}")
    else:
        print(f"[NEW] Starting new RF tuning for {stock_symbol}")

    # Combine train and validation for fitting, but track validation score separately
    # This allows the tuner to use train+val for CV while we manually evaluate on validation
    # Note: We'll use a custom validation split to maintain separation
    from sklearn.model_selection import PredefinedSplit
    
    # Create combined dataset
    x_combined = np.vstack([x_train, x_val])
    y_combined = np.concatenate([y_train, y_val])
    
    # Create split indices: -1 for training, 0 for validation
    split_indices = np.concatenate([
        np.full(len(x_train), -1),  # Training samples get -1
        np.zeros(len(x_val))         # Validation samples get 0
    ])
    
    # Create PredefinedSplit (ensures validation set is used for scoring)
    ps = PredefinedSplit(test_fold=split_indices)
    
    # Create lambda that passes constrain_for_overfitting
    def build_rf_wrapper(hp):
        return build_random_forest_model(hp, constrain_for_overfitting=constrain_for_overfitting)

    # Create tuner with PredefinedSplit (uses validation set for scoring)
    tuner = Sklearn(
        oracle=kt.oracles.Hyperband(
            objective=kt.Objective('score', 'min'),
            max_epochs=max_trials,
            factor=3,
            seed=42
        ),
        hypermodel=build_rf_wrapper,
        scoring=mse_scorer,
        cv=ps,  # Use predefined split to ensure validation set is used
        directory=temp_dir,
        project_name=project_name_val,
        overwrite=overwrite_val
    )

    # Search for best hyperparameters using train+val with predefined split
    print(f"[SEARCH] Starting Random Forest hyperparameter tuning for {stock_symbol}...")
    tuner.search(x_combined, y_combined)

    # Get best hyperparameters
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Build and train best model on TRAINING DATA ONLY
    best_rf_model = tuner.hypermodel.build(best_hp)
    best_rf_model.fit(x_train, y_train)

    # Print best hyperparameters
    print("\n[RF] Best Random Forest hyperparameters found:")
    for param, value in best_hp.values.items():
        print(f"  - {param}: {value}")

    # Feature importance logging
    importances = best_rf_model.feature_importances_
    feature_names = x_training_dataset_df.columns

    print("\n[DATA] Top 10 Feature Importances:")
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    for _, row in feature_importance_df.head(10).iterrows():
        print(f"  - {row['feature']}: {row['importance']:.4f}")

    # Save hyperparameters to database for future use
    tuning_time = time_module.time() - tuning_start_time
    try:
        import db_interactions
        val_pred = best_rf_model.predict(x_val)
        val_mse = mean_squared_error(y_val, val_pred)
        val_r2 = r2_score(y_val, val_pred)
        val_mae = mean_absolute_error(y_val, val_pred)
        
        # Ensure feature_list contains strings (column indices may be ints)
        feature_list_str = [str(f) for f in feature_list] if feature_list else feature_list
        
        db_interactions.save_hyperparameters(
            ticker=stock_symbol,
            model_type='rf',
            hyperparameters=serialize_random_forest_hyperparameters(best_rf_model),
            num_trials=max_trials,
            best_score=val_mse,
            tuning_time_seconds=tuning_time,
            training_samples=len(x_train),
            num_features=x_train.shape[1],
            feature_list=feature_list_str,
            val_mse=val_mse,
            val_r2=val_r2,
            val_mae=val_mae,
            is_constrained=constrain_for_overfitting
        )
    except HYPERPARAMETER_SAVE_EXCEPTIONS as e:
        print(f"[WARNING] Could not save RF hyperparameters to DB: {e}")

    # Cleanup tuning directory to save disk space
    if cleanup_after_tuning:
        try:
            shutil.rmtree(project_path)
            print(f"[CLEANUP] Deleted tuning directory: {project_path}")
        except (OSError, PermissionError) as e:
            print(f"[WARNING] Could not delete tuning directory: {e}")

    return best_rf_model

def evaluate_random_forest_model(model, x_train, y_train, x_val, y_val, x_test, y_test):
    """
    Evaluate Random Forest model on all three sets.

    Returns:
    - train_metrics, val_metrics, test_metrics (dicts with mse, r2, and mae)
    """
    train_pred = model.predict(x_train)
    val_pred = model.predict(x_val)
    test_pred = model.predict(x_test)

    train_metrics = {
        'mse': mean_squared_error(y_train, train_pred),
        'r2': r2_score(y_train, train_pred),
        'mae': mean_absolute_error(y_train, train_pred)
    }

    val_metrics = {
        'mse': mean_squared_error(y_val, val_pred),
        'r2': r2_score(y_val, val_pred),
        'mae': mean_absolute_error(y_val, val_pred)
    }

    test_metrics = {
        'mse': mean_squared_error(y_test, test_pred),
        'r2': r2_score(y_test, test_pred),
        'mae': mean_absolute_error(y_test, test_pred)
    }

    return train_metrics, val_metrics, test_metrics

def build_xgboost_model(hp, constrain_for_overfitting=False):
    """
    Builds an XGBoost regressor with tunable hyperparameters.
    XGBoost often outperforms Random Forest on financial data.
    
    Parameters:
    - hp: Keras Tuner hyperparameters object
    - constrain_for_overfitting: If True, apply stricter constraints to reduce overfitting
    """
    if constrain_for_overfitting:
        # Stricter constraints when overfitting is detected
        model = xgb.XGBRegressor(
            n_estimators=hp.Int('n_estimators', 100, 800, step=50),  # Lower ceiling
            max_depth=hp.Int('max_depth', 3, 10, step=1),  # Shallower trees
            learning_rate=hp.Float('learning_rate', 0.01, 0.2, sampling='log'),
            subsample=hp.Float('subsample', 0.7, 0.9, step=0.1),  # Higher floor
            colsample_bytree=hp.Float('colsample_bytree', 0.7, 0.9, step=0.1),  # Higher floor
            min_child_weight=hp.Int('min_child_weight', 3, 15, step=1),  # Higher floor
            gamma=hp.Float('gamma', 0.1, 0.5, step=0.1),  # Higher regularization
            reg_alpha=hp.Float('reg_alpha', 0.1, 1.0, step=0.1),  # Stronger L1
            reg_lambda=hp.Float('reg_lambda', 0.1, 1.0, step=0.1),  # Stronger L2
            random_state=42,
            n_jobs=-1,
            tree_method='hist'
        )
    else:
        # Standard search space
        model = xgb.XGBRegressor(
            n_estimators=hp.Int('n_estimators', 100, 1000, step=50),
            max_depth=hp.Int('max_depth', 3, 15, step=1),
            learning_rate=hp.Float('learning_rate', 0.01, 0.3, sampling='log'),
            subsample=hp.Float('subsample', 0.6, 1.0, step=0.1),
            colsample_bytree=hp.Float('colsample_bytree', 0.6, 1.0, step=0.1),
            min_child_weight=hp.Int('min_child_weight', 1, 10, step=1),
            gamma=hp.Float('gamma', 0.0, 0.5, step=0.1),
            reg_alpha=hp.Float('reg_alpha', 0.0, 1.0, step=0.1),  # L1 regularization
            reg_lambda=hp.Float('reg_lambda', 0.0, 1.0, step=0.1),  # L2 regularization
            random_state=42,
            n_jobs=-1,
            tree_method='hist'  # Faster training
        )
    return model

def tune_xgboost_model(stock_symbol, x_training_dataset_df, y_training_dataset_df, x_val_dataset_df, y_val_dataset_df, max_trials=30, constrain_for_overfitting=False, use_cached_hp=True, cache_max_age_days=30, cleanup_after_tuning=True, require_cached_hp=False):
    """
    Tunes XGBoost model using validation set for hyperparameter selection.
    
    Parameters:
    - stock_symbol (str): The stock ticker symbol
    - x_training_dataset_df (pd.DataFrame): Training dataset (ALREADY SCALED)
    - y_training_dataset_df (pd.Series or np.ndarray): Training labels (UNSCALED for XGBoost)
    - x_val_dataset_df (pd.DataFrame): Validation dataset (ALREADY SCALED)
    - y_val_dataset_df (pd.Series or np.ndarray): Validation labels (UNSCALED for XGBoost)
    - max_trials (int): Maximum number of tuning trials
    - use_cached_hp (bool): Whether to use cached hyperparameters if available
    - cache_max_age_days (int): Maximum age of cached HPs in days
    - cleanup_after_tuning (bool): Whether to delete tuning directory after training
    
    Returns:
    - best_xgb_model: The tuned XGBoost model
    """
    import time as time_module
    tuning_start_time = time_module.time()
    
    # Convert to numpy arrays
    x_train = x_training_dataset_df.values
    y_train = y_training_dataset_df.values
    x_val = x_val_dataset_df.values
    y_val = y_val_dataset_df.values
    feature_list = list(x_training_dataset_df.columns)

    # Try to load cached hyperparameters
    if use_cached_hp:
        cached_hp = None
        try:
            import db_interactions
            cached_hp = db_interactions.load_hyperparameters(
                ticker=stock_symbol,
                model_type='xgb',
                max_age_days=cache_max_age_days,
                feature_list=feature_list,
                require_same_features=True
            )
            
        except CACHED_HP_LOAD_EXCEPTIONS as e:
            if require_cached_hp:
                raise PredictionTrainingRequiredError(
                    ticker=stock_symbol,
                    required_model_types=['xgb'],
                    missing_model_types=['xgb'],
                    message=f"Cached XGB hyperparameters are unavailable for {stock_symbol}: {e}",
                ) from e
            print(f"[CACHE] Could not load cached HP: {e}, proceeding with tuning")

        if cached_hp:
            print(f"[CACHE] Using cached XGBoost hyperparameters for {stock_symbol}")
            try:
                best_xgb_model = build_cached_xgboost_model(cached_hp)
                best_xgb_model.fit(x_train, y_train)
                return best_xgb_model
            except CACHED_MODEL_RESTORE_EXCEPTIONS as e:
                db_interactions.invalidate_hyperparameters(ticker=stock_symbol, model_type='xgb')
                if require_cached_hp:
                    raise_cache_invalidated_error(
                        stock_symbol,
                        'xgb',
                        e,
                        invalidator=invalidate_hyperparameter_cache,
                    )
                print(f"[CACHE] Could not restore cached XGB HP: {e}, proceeding with tuning")

    if require_cached_hp:
        raise PredictionTrainingRequiredError(
            ticker=stock_symbol,
            required_model_types=['xgb'],
            missing_model_types=['xgb'],
        )

    # Define the MSE scorer
    mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

    # Setup directory structure - use temp directory to avoid OneDrive locking issues
    temp_dir = os.path.join(tempfile.gettempdir(), "xgb_tuning_dir")
    project_name_val = f"XGB_tuning_{stock_symbol}"
    project_path = os.path.join(temp_dir, project_name_val)

    # Also clean up any old tuning_dir entries in workspace (legacy)
    legacy_path = os.path.join("tuning_dir", project_name_val)
    if os.path.exists(legacy_path):
        try:
            shutil.rmtree(legacy_path)
            print(f"[CLEANUP] Deleted legacy tuning directory: {legacy_path}")
        except (OSError, PermissionError) as e:
            print(f"[WARN] Could not delete legacy tuning directory: {e}")

    # Determine overwrite: if stale/incomplete directory exists (no oracle.json), clean it
    overwrite_val = True
    if os.path.exists(project_path):
        oracle_path = os.path.join(project_path, "oracle.json")
        if os.path.exists(oracle_path):
            print(f"[RESUME] Found existing XGB tuning at {project_path}. Continuing...")
            overwrite_val = False
        else:
            print("[WARN] Found incomplete XGB tuning directory (no oracle.json). Starting fresh...")
            try:
                shutil.rmtree(project_path)
            except (OSError, PermissionError) as e:
                print(f"[WARN] Could not delete incomplete tuning directory: {e}")
    else:
        print(f"[NEW] Starting new XGB tuning for {stock_symbol}")

    # Use PredefinedSplit for validation
    from sklearn.model_selection import PredefinedSplit
    
    # Create combined dataset
    x_combined = np.vstack([x_train, x_val])
    y_combined = np.concatenate([y_train, y_val])
    
    # Create split indices: -1 for training, 0 for validation
    split_indices = np.concatenate([
        np.full(len(x_train), -1),
        np.zeros(len(x_val))
    ])
    
    ps = PredefinedSplit(test_fold=split_indices)
    
    # Create lambda that passes constrain_for_overfitting
    def build_xgb_wrapper(hp):
        return build_xgboost_model(hp, constrain_for_overfitting=constrain_for_overfitting)

    # Create tuner with Bayesian Optimization
    tuner = Sklearn(
        oracle=kt.oracles.BayesianOptimization(
            objective=kt.Objective('score', 'min'),
            max_trials=max_trials,
            num_initial_points=min(5, max_trials // 3),
            seed=42
        ),
        hypermodel=build_xgb_wrapper,
        scoring=mse_scorer,
        cv=ps,
        directory=temp_dir,
        project_name=project_name_val,
        overwrite=overwrite_val
    )

    # Search for best hyperparameters
    print(f"[SEARCH] Starting XGBoost hyperparameter tuning for {stock_symbol}...")
    tuner.search(x_combined, y_combined)

    # Get best hyperparameters
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Build and train best model on TRAINING DATA ONLY
    best_xgb_model = tuner.hypermodel.build(best_hp)
    best_xgb_model.fit(x_train, y_train)

    # Print best hyperparameters
    print("\n[XGB] Best XGBoost hyperparameters found:")
    for param, value in best_hp.values.items():
        print(f"  - {param}: {value}")

    # Feature importance logging
    importances = best_xgb_model.feature_importances_
    feature_names = x_training_dataset_df.columns

    print("\n[DATA] Top 10 Feature Importances (XGBoost):")
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    for _, row in feature_importance_df.head(10).iterrows():
        print(f"  - {row['feature']}: {row['importance']:.4f}")

    # Save hyperparameters to database for future use
    tuning_time = time_module.time() - tuning_start_time
    try:
        import db_interactions
        val_pred = best_xgb_model.predict(x_val)
        val_mse = mean_squared_error(y_val, val_pred)
        val_r2 = r2_score(y_val, val_pred)
        val_mae = mean_absolute_error(y_val, val_pred)
        
        # Ensure feature_list contains strings (column indices may be ints)
        feature_list_str = [str(f) for f in feature_list] if feature_list else feature_list
        
        db_interactions.save_hyperparameters(
            ticker=stock_symbol,
            model_type='xgb',
            hyperparameters=serialize_xgboost_hyperparameters(best_xgb_model),
            num_trials=max_trials,
            best_score=val_mse,
            tuning_time_seconds=tuning_time,
            training_samples=len(x_train),
            num_features=x_train.shape[1],
            feature_list=feature_list_str,
            val_mse=val_mse,
            val_r2=val_r2,
            val_mae=val_mae,
            is_constrained=constrain_for_overfitting
        )
    except HYPERPARAMETER_SAVE_EXCEPTIONS as e:
        print(f"[WARNING] Could not save XGB hyperparameters to DB: {e}")

    # Cleanup tuning directory to save disk space
    if cleanup_after_tuning:
        try:
            shutil.rmtree(project_path)
            print(f"[CLEANUP] Deleted tuning directory: {project_path}")
        except (OSError, PermissionError) as e:
            print(f"[WARNING] Could not delete tuning directory: {e}")

    return best_xgb_model

def evaluate_xgboost_model(model, x_train, y_train, x_val, y_val, x_test, y_test):
    """
    Evaluate XGBoost model on all three sets.

    Returns:
    - train_metrics, val_metrics, test_metrics (dicts with mse, r2, and mae)
    """
    train_pred = model.predict(x_train)
    val_pred = model.predict(x_val)
    test_pred = model.predict(x_test)

    train_metrics = {
        'mse': mean_squared_error(y_train, train_pred),
        'r2': r2_score(y_train, train_pred),
        'mae': mean_absolute_error(y_train, train_pred)
    }

    val_metrics = {
        'mse': mean_squared_error(y_val, val_pred),
        'r2': r2_score(y_val, val_pred),
        'mae': mean_absolute_error(y_val, val_pred)
    }

    test_metrics = {
        'mse': mean_squared_error(y_test, test_pred),
        'r2': r2_score(y_test, test_pred),
        'mae': mean_absolute_error(y_test, test_pred)
    }

    return train_metrics, val_metrics, test_metrics


# ============================================================================
# Ridge Regression — fast linear baseline for the ensemble (#23)
# ============================================================================
def build_ridge_model(hp, constrain_for_overfitting=False):
    """
    Builds a Ridge regression model with tunable hyperparameters.

    Parameters:
    - hp: Keras Tuner hyperparameters object
    - constrain_for_overfitting: If True, prefer stronger regularization
    """
    alpha = hp.Float('alpha', 1e-3, 1e3, sampling='log', default=1.0)
    solver = hp.Choice('solver', ['auto', 'svd', 'cholesky', 'lsqr', 'sag'], default='auto')

    if constrain_for_overfitting:
        alpha = max(alpha, 1.0)  # Ensure stronger regularization

    model = Ridge(
        alpha=alpha,
        solver=solver,
        random_state=42,
        max_iter=10000
    )
    return model


def tune_ridge_model(stock_symbol, x_training_dataset_df, y_training_dataset_df,
                     x_val_dataset_df, y_val_dataset_df, max_trials=50,
                     constrain_for_overfitting=False, use_cached_hp=True,
                     cache_max_age_days=30, cleanup_after_tuning=True,
                     require_cached_hp=False):
    """
    Tune Ridge regression using Keras Tuner with validation-set scoring.
    Follows the same pattern as tune_random_forest_model.
    """
    import time as time_module
    from sklearn.model_selection import PredefinedSplit

    tuning_start_time = time_module.time()

    x_train = x_training_dataset_df.values
    y_train = y_training_dataset_df.values
    x_val = x_val_dataset_df.values
    y_val = y_val_dataset_df.values
    feature_list = list(x_training_dataset_df.columns)

    # Try cached hyperparameters
    if use_cached_hp:
        cached_hp = None
        try:
            import db_interactions
            cached_hp = db_interactions.load_hyperparameters(
                ticker=stock_symbol, model_type='ridge',
                max_age_days=cache_max_age_days, feature_list=feature_list,
                require_same_features=True
            )
        except CACHED_HP_LOAD_EXCEPTIONS as e:
            if require_cached_hp:
                raise PredictionTrainingRequiredError(
                    ticker=stock_symbol,
                    required_model_types=['ridge'],
                    missing_model_types=['ridge'],
                    message=f"Cached Ridge hyperparameters are unavailable for {stock_symbol}: {e}",
                ) from e
            print(f"[CACHE] Could not load cached Ridge HP: {e}, proceeding with tuning")

        if cached_hp:
            print(f"[CACHE] Using cached Ridge hyperparameters for {stock_symbol}")
            try:
                best_model = build_cached_ridge_model(cached_hp)
                best_model.fit(x_train, y_train)
                return best_model
            except CACHED_MODEL_RESTORE_EXCEPTIONS as e:
                db_interactions.invalidate_hyperparameters(ticker=stock_symbol, model_type='ridge')
                if require_cached_hp:
                    raise_cache_invalidated_error(
                        stock_symbol,
                        'ridge',
                        e,
                        invalidator=invalidate_hyperparameter_cache,
                    )
                print(f"[CACHE] Could not restore cached Ridge HP: {e}, proceeding with tuning")

    if require_cached_hp:
        raise PredictionTrainingRequiredError(
            ticker=stock_symbol,
            required_model_types=['ridge'],
            missing_model_types=['ridge'],
        )

    mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

    temp_dir = os.path.join(tempfile.gettempdir(), "ridge_tuning_dir")
    project_name_val = f"Ridge_tuning_{stock_symbol}"
    project_path = os.path.join(temp_dir, project_name_val)

    overwrite_val = True
    if os.path.exists(project_path):
        oracle_path = os.path.join(project_path, "oracle.json")
        if os.path.exists(oracle_path):
            overwrite_val = False
        else:
            try:
                shutil.rmtree(project_path)
            except (OSError, PermissionError):
                pass

    x_combined = np.vstack([x_train, x_val])
    y_combined = np.concatenate([y_train, y_val])
    split_indices = np.concatenate([np.full(len(x_train), -1), np.zeros(len(x_val))])
    ps = PredefinedSplit(test_fold=split_indices)

    def build_ridge_wrapper(hp):
        return build_ridge_model(hp, constrain_for_overfitting=constrain_for_overfitting)

    tuner = Sklearn(
        oracle=kt.oracles.BayesianOptimization(
            objective=kt.Objective('score', 'min'),
            max_trials=max_trials, seed=42
        ),
        hypermodel=build_ridge_wrapper,
        scoring=mse_scorer,
        cv=ps,
        directory=temp_dir,
        project_name=project_name_val,
        overwrite=overwrite_val
    )

    print(f"[SEARCH] Starting Ridge hyperparameter tuning for {stock_symbol}...")
    tuner.search(x_combined, y_combined)

    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.hypermodel.build(best_hp)
    best_model.fit(x_train, y_train)

    print("\n[RIDGE] Best Ridge hyperparameters found:")
    for param, value in best_hp.values.items():
        print(f"  - {param}: {value}")

    # Save to DB
    tuning_time = time_module.time() - tuning_start_time
    try:
        import db_interactions
        val_pred = best_model.predict(x_val)
        val_mse = mean_squared_error(y_val, val_pred)
        val_r2 = r2_score(y_val, val_pred)
        val_mae = mean_absolute_error(y_val, val_pred)
        feature_list_str = [str(f) for f in feature_list] if feature_list else feature_list
        db_interactions.save_hyperparameters(
            ticker=stock_symbol, model_type='ridge',
            hyperparameters=serialize_ridge_hyperparameters(best_model), num_trials=max_trials,
            best_score=val_mse, tuning_time_seconds=tuning_time,
            training_samples=len(x_train), num_features=x_train.shape[1],
            feature_list=feature_list_str, val_mse=val_mse,
            val_r2=val_r2, val_mae=val_mae,
            is_constrained=constrain_for_overfitting
        )
    except HYPERPARAMETER_SAVE_EXCEPTIONS as e:
        print(f"[WARNING] Could not save Ridge hyperparameters to DB: {e}")

    if cleanup_after_tuning:
        try:
            shutil.rmtree(project_path)
        except (OSError, PermissionError):
            pass

    return best_model


def evaluate_ridge_model(model, x_train, y_train, x_val, y_val, x_test, y_test):
    """Evaluate Ridge model on all three sets."""
    train_pred = model.predict(x_train)
    val_pred = model.predict(x_val)
    test_pred = model.predict(x_test)

    return (
        {'mse': mean_squared_error(y_train, train_pred),
         'r2': r2_score(y_train, train_pred),
         'mae': mean_absolute_error(y_train, train_pred)},
        {'mse': mean_squared_error(y_val, val_pred),
         'r2': r2_score(y_val, val_pred),
         'mae': mean_absolute_error(y_val, val_pred)},
        {'mse': mean_squared_error(y_test, test_pred),
         'r2': r2_score(y_test, test_pred),
         'mae': mean_absolute_error(y_test, test_pred)}
    )


# ============================================================================
# SVR — modernized with DB caching, constrain param, validation split (#23)
# ============================================================================
def build_svr_model(hp, constrain_for_overfitting=False):
    """
    Builds an SVR model with tunable hyperparameters.
    Modernized version with constrain_for_overfitting support.

    Search bounds are capped to prevent hangs:
    - poly kernel excluded: produces unbounded kernel matrix entries on tabular
      financial data (C=100, gamma=1, degree=2 → R2=-208); inappropriate for
      this domain.
    - C max 10: higher C with rbf causes O(n^2-n^3) kernel fit times
    - gamma max 0.5: for ~50 MinMaxScaled features, gamma>0.5 makes rbf kernel
      degenerate (exp(-gamma*8) ≈ 0 for gamma>=1)
    - max_iter=10000: hard per-fit cap; ConvergenceWarning is acceptable
    """
    kernel_choice = hp.Choice('kernel', ['linear', 'rbf'])
    C = hp.Float('C', 1e-3, 10.0, sampling='log', default=1.0)
    gamma = hp.Float('gamma', 1e-4, 0.5, sampling='log', default=0.1)
    epsilon = hp.Float('epsilon', 1e-4, 1.0, sampling='log', default=0.1)

    if constrain_for_overfitting:
        C = min(C, 10.0)
        epsilon = max(epsilon, 0.01)

    svr = SVR(
        kernel=kernel_choice,
        C=C,
        gamma=gamma,
        epsilon=epsilon,
        max_iter=10000
    )
    return TransformedTargetRegressor(regressor=svr, transformer=StandardScaler())


def tune_svr_model(stock_symbol, x_training_dataset_df, y_training_dataset_df,
                   x_val_dataset_df, y_val_dataset_df, max_trials=40,
                   constrain_for_overfitting=False, use_cached_hp=True,
                   cache_max_age_days=30, cleanup_after_tuning=True,
                   require_cached_hp=False):
    """
    Tune SVR using Keras Tuner with validation-set scoring.
    Modernized version following the RF/XGB pattern with DB caching.
    """
    import time as time_module
    from sklearn.model_selection import PredefinedSplit

    tuning_start_time = time_module.time()

    x_train = x_training_dataset_df.values
    y_train = y_training_dataset_df.values
    x_val = x_val_dataset_df.values
    y_val = y_val_dataset_df.values
    feature_list = list(x_training_dataset_df.columns)

    # Try cached hyperparameters
    if use_cached_hp:
        cached_hp = None
        try:
            import db_interactions
            cached_hp = db_interactions.load_hyperparameters(
                ticker=stock_symbol, model_type='svr',
                max_age_days=cache_max_age_days, feature_list=feature_list,
                require_same_features=True
            )
        except CACHED_HP_LOAD_EXCEPTIONS as e:
            if require_cached_hp:
                raise PredictionTrainingRequiredError(
                    ticker=stock_symbol,
                    required_model_types=['svr'],
                    missing_model_types=['svr'],
                    message=f"Cached SVR hyperparameters are unavailable for {stock_symbol}: {e}",
                ) from e
            print(f"[CACHE] Could not load cached SVR HP: {e}, proceeding with tuning")

        if cached_hp:
            print(f"[CACHE] Using cached SVR hyperparameters for {stock_symbol}")
            try:
                best_model = build_cached_svr_model(cached_hp)
                _fit_svr_without_convergence_warnings(best_model, x_train, y_train)
                return best_model
            except CACHED_MODEL_RESTORE_EXCEPTIONS as e:
                db_interactions.invalidate_hyperparameters(ticker=stock_symbol, model_type='svr')
                if require_cached_hp:
                    raise_cache_invalidated_error(
                        stock_symbol,
                        'svr',
                        e,
                        invalidator=invalidate_hyperparameter_cache,
                    )
                print(f"[CACHE] Could not restore cached SVR HP: {e}, proceeding with tuning")

    if require_cached_hp:
        raise PredictionTrainingRequiredError(
            ticker=stock_symbol,
            required_model_types=['svr'],
            missing_model_types=['svr'],
        )

    mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

    temp_dir = os.path.join(tempfile.gettempdir(), "svr_tuning_dir")
    project_name_val = f"SVR_tuning_{stock_symbol}"
    project_path = os.path.join(temp_dir, project_name_val)

    overwrite_val = True
    if os.path.exists(project_path):
        oracle_path = os.path.join(project_path, "oracle.json")
        if os.path.exists(oracle_path):
            overwrite_val = False
        else:
            try:
                shutil.rmtree(project_path)
            except (OSError, PermissionError):
                pass

    x_combined = np.vstack([x_train, x_val])
    y_combined = np.concatenate([y_train, y_val])
    split_indices = np.concatenate([np.full(len(x_train), -1), np.zeros(len(x_val))])
    ps = PredefinedSplit(test_fold=split_indices)

    def build_svr_wrapper(hp):
        return build_svr_model(hp, constrain_for_overfitting=constrain_for_overfitting)

    tuner = Sklearn(
        oracle=kt.oracles.BayesianOptimization(
            objective=kt.Objective('score', 'min'),
            max_trials=max_trials, seed=42
        ),
        hypermodel=build_svr_wrapper,
        scoring=mse_scorer,
        cv=ps,
        directory=temp_dir,
        project_name=project_name_val,
        overwrite=overwrite_val
    )

    print(f"[SEARCH] Starting SVR hyperparameter tuning for {stock_symbol}...")
    _run_svr_tuning_search_without_convergence_warnings(tuner, x_combined, y_combined)

    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.hypermodel.build(best_hp)
    _fit_svr_without_convergence_warnings(best_model, x_train, y_train)

    print("\n[SVR] Best SVR hyperparameters found:")
    for param, value in best_hp.values.items():
        print(f"  - {param}: {value}")

    # Save to DB
    tuning_time = time_module.time() - tuning_start_time
    try:
        import db_interactions
        val_pred = best_model.predict(x_val)
        val_mse = mean_squared_error(y_val, val_pred)
        val_r2 = r2_score(y_val, val_pred)
        val_mae = mean_absolute_error(y_val, val_pred)
        feature_list_str = [str(f) for f in feature_list] if feature_list else feature_list
        db_interactions.save_hyperparameters(
            ticker=stock_symbol, model_type='svr',
            hyperparameters=serialize_svr_hyperparameters(best_model), num_trials=max_trials,
            best_score=val_mse, tuning_time_seconds=tuning_time,
            training_samples=len(x_train), num_features=x_train.shape[1],
            feature_list=feature_list_str, val_mse=val_mse,
            val_r2=val_r2, val_mae=val_mae,
            is_constrained=constrain_for_overfitting
        )
    except HYPERPARAMETER_SAVE_EXCEPTIONS as e:
        print(f"[WARNING] Could not save SVR hyperparameters to DB: {e}")

    if cleanup_after_tuning:
        try:
            shutil.rmtree(project_path)
        except (OSError, PermissionError):
            pass

    return best_model


def evaluate_svr_model(model, x_train, y_train, x_val, y_val, x_test, y_test):
    """Evaluate SVR model on all three sets."""
    train_pred = model.predict(x_train)
    val_pred = model.predict(x_val)
    test_pred = model.predict(x_test)

    return (
        {'mse': mean_squared_error(y_train, train_pred),
         'r2': r2_score(y_train, train_pred),
         'mae': mean_absolute_error(y_train, train_pred)},
        {'mse': mean_squared_error(y_val, val_pred),
         'r2': r2_score(y_val, val_pred),
         'mae': mean_absolute_error(y_val, val_pred)},
        {'mse': mean_squared_error(y_test, test_pred),
         'r2': r2_score(y_test, test_pred),
         'mae': mean_absolute_error(y_test, test_pred)}
    )


def build_svm_model(hp):
    """
    Builds an SVR model with tunable hyperparameters.

    Parameters:
    - hp: Keras Tuner hyperparameters object

    Returns:
    - model: Configured SVR model
    """
    kernel_choice = hp.Choice('kernel', ['linear', 'rbf'])
    svr = SVR(
        kernel=kernel_choice,
        C=hp.Float('C', 1e-3, 10.0, sampling='log'),
        gamma=hp.Float('gamma', 1e-4, 0.5, sampling='log'),
        max_iter=10000
    )
    return TransformedTargetRegressor(regressor=svr, transformer=StandardScaler())

def tune_svm_model(stock_symbol, training_dataset_df, max_trials=20):
    """
    Tunes an SVM model using Keras Tuner.

    Parameters:
    - stock_symbol (str): The stock ticker symbol
    - training_dataset_df (pd.DataFrame): Training data with features and target
    - max_trials (int): Maximum number of tuning trials

    Returns:
    - best_svm_model: The tuned SVM model
    """
    x_train = training_dataset_df.drop(["prediction"], axis=1).values
    y_train = training_dataset_df["prediction"].values

    mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

    tuner = Sklearn(
        oracle=kt.oracles.BayesianOptimization(
            objective=kt.Objective('score', 'min'),
            max_trials=max_trials,
            seed=42
        ),
        hypermodel=build_svm_model,
        scoring=mse_scorer,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        directory="tuning_dir",
        project_name=f"SVM_tuning_{stock_symbol}",
        overwrite=False
    )

    tuner.search(x_train, y_train)

    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_svm_model = tuner.hypermodel.build(best_hp)
    best_svm_model.fit(x_train, y_train)

    print("[SVM] Best SVM hyperparameters found:")
    for param, value in best_hp.values.items():
        print(f"- {param}: {value}")

    return best_svm_model

def create_sequences(data, time_steps):
    """
    Converts a 2D feature array into a 3D sequence array for LSTM.
    
    Parameters:
    - data (np.array): 2D array of shape (samples, features)
    - time_steps (int): Number of time steps per sequence
    
    Returns:
    - tuple[np.array, np.array]: Sequence inputs and next-step targets
    
    Example:
    >>> data = np.array([[1,2], [3,4], [5,6], [7,8], [9,10]])
    >>> sequences, targets = create_sequences(data, time_steps=3)
    >>> sequences.shape
    (2, 3, 2)
    """
    data = np.asarray(data)
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    if len(data) <= time_steps:
        empty_sequences = np.empty((0, time_steps, data.shape[1]))
        empty_targets = np.empty((0, data.shape[1]))
        return empty_sequences, empty_targets

    Xs = []
    ys = []
    for i in range(len(data) - time_steps):
        Xs.append(data[i:(i + time_steps)])
        ys.append(data[i + time_steps])

    result = np.array(Xs)
    targets = np.array(ys)
    
    # Validate output shape
    expected_shape = (len(data) - time_steps, time_steps, data.shape[1])
    if result.shape != expected_shape:
        raise ValueError(
            f"Sequence creation failed. Expected shape {expected_shape}, got {result.shape}"
        )
    
    return result, targets

def prepare_lstm_datasets(x_train, y_train, x_val, y_val, x_test, y_test, time_steps):
    """
    Prepares datasets for LSTM training by creating sequences.
    
    Parameters:
    - x_train, y_train, x_val, y_val, x_test, y_test: Raw feature/target arrays
    - time_steps (int): Number of time steps for sequence creation
    
    Returns:
    - Dictionary containing all reshaped datasets with metadata
    
    Raises:
    - ValueError: If dataset size is insufficient for sequence creation
    """
    # Validate dataset sizes
    min_required_samples = time_steps + 1
    
    if len(x_train) < min_required_samples:
        raise ValueError(f"Training set too small: {len(x_train)} samples, need at least {min_required_samples}")
    if len(x_val) < min_required_samples:
        raise ValueError(f"Validation set too small: {len(x_val)} samples, need at least {min_required_samples}")
    if len(x_test) < min_required_samples:
        raise ValueError(f"Test set too small: {len(x_test)} samples, need at least {min_required_samples}")
    
    # Create sequences
    x_train_lstm, _ = create_sequences(x_train, time_steps)
    y_train_lstm = y_train[time_steps:].reshape(-1, 1)
    
    x_val_lstm, _ = create_sequences(x_val, time_steps)
    y_val_lstm = y_val[time_steps:].reshape(-1, 1)
    
    x_test_lstm, _ = create_sequences(x_test, time_steps)
    y_test_lstm = y_test[time_steps:].reshape(-1, 1)
    
    # Validate shapes
    assert x_train_lstm.ndim == 3, f"x_train_lstm should be 3D, got {x_train_lstm.ndim}D"
    assert x_val_lstm.ndim == 3, f"x_val_lstm should be 3D, got {x_val_lstm.ndim}D"
    assert x_test_lstm.ndim == 3, f"x_test_lstm should be 3D, got {x_test_lstm.ndim}D"
    
    return {
        'train': {'x': x_train_lstm, 'y': y_train_lstm},
        'val': {'x': x_val_lstm, 'y': y_val_lstm},
        'test': {'x': x_test_lstm, 'y': y_test_lstm},
        'metadata': {
            'time_steps': time_steps,
            'num_features': x_train_lstm.shape[2],
            'train_samples': x_train_lstm.shape[0],
            'val_samples': x_val_lstm.shape[0],
            'test_samples': x_test_lstm.shape[0]
        }
    }

def build_lstm_model(hp, input_shape):
    """
    Builds a Bidirectional LSTM model with tunable hyperparameters for time series prediction.
    
    The model consists of:
    - Multiple stacked Bidirectional LSTM layers with L2 regularization
    - BatchNormalization layers after each LSTM layer for training stability
    - Dropout layers for regularization
    - Dense layers with tunable activation functions
    - Linear output layer for regression
    
    Parameters:
    - hp (keras_tuner.HyperParameters): Keras Tuner hyperparameters object for tuning
    - input_shape (tuple): Shape of input sequences (time_steps, num_features)
    
    Returns:
    - keras.Sequential: Compiled LSTM model ready for training
    
    Hyperparameters tuned:
    - n_layers: Number of LSTM layers (1 to 5)
    - input_units: Units in first LSTM layer (32-512, step=16)
    - units_{i}: Units in subsequent LSTM layers (32-512, step=16)
    - l2_reg_{i}: L2 regularization strength (1e-6 to 1e-3, log scale)
    - dropout_{i}: Dropout rate (0.1-0.6, step=0.1)
    - dense_1/dense_2: Dense layer units
    - dense_1_activation: Activation function (relu, tanh)
    - dense_2_activation: Activation function (relu, sigmoid)
    - optimizer: Optimizer choice (adam, rmsprop)
    - learning_rate: Learning rate (1e-5 to 1e-3, log scale)
    - clipnorm: Gradient clipping norm (0.5-2.0, step=0.5)
    - loss: Loss function (MAE, MSE, Huber, MAPE)
    """
    model = Sequential()
    # First LSTM layer
    model.add(
        Bidirectional(
            LSTM(
                units=hp.Int(
                    "input_units",
                    min_value=32,
                    max_value=512,
                    step=16
                ),
                return_sequences=True,
                kernel_regularizer=regularizers.l2(
                    hp.Float(
                        "l2_reg_input",
                        1e-6, 1e-3,
                        sampling="log"
                    )
                ),
                input_shape=input_shape
            )
        )
    )
    
    # Add BatchNormalization after first LSTM layer to stabilize training
    model.add(BatchNormalization())

    max_amount_layers = 5
    for i in range(1, _coerce_hparam_int(hp.Int('n_layers', 1, max_amount_layers), 1)):
        model.add(
            Bidirectional(
                LSTM(
                    units=hp.Int(
                        f"units_{i}",
                        min_value=32,
                        max_value=512,
                        step=16
                    ),
                    return_sequences=True,
                    kernel_regularizer=regularizers.l2(
                        hp.Float(
                            f"l2_reg_{i}",
                            1e-6, 1e-3,
                            sampling="log"
                        )
                    )
                )
            )
        )
        
        # Add BatchNormalization after each LSTM layer
        model.add(BatchNormalization())

        model.add(
            Dropout(
                hp.Float(
                    f"dropout_{i}",
                    min_value=0.1,
                    max_value=0.6,
                    step=0.1
                )
            )
        )

    # Final LSTM (collapse to 2D)
    model.add(
        Bidirectional(
            LSTM(
                units=hp.Int(
                    "final_units",
                    min_value=32,
                    max_value=512,
                    step=16
                ),
                return_sequences=False,
                kernel_regularizer=regularizers.l2(
                    hp.Float(
                        "l2_reg_final",
                        1e-6, 1e-3,
                        sampling="log"
                    )
               )
            )
        )
    )
    
    # Add BatchNormalization after final LSTM layer
    model.add(BatchNormalization())

    model.add(
        Dense(
            units=hp.Int(
                "dense_1",
                min_value=16,
                max_value=128,
                step=16
            ),
            activation=_coerce_hparam_choice(
                hp.Choice(
                    "dense_1_activation",
                    ["relu", "tanh"]
                ),
                ["relu", "tanh"],
                "relu",
            )
        )
    )

    model.add(
        Dropout(
            hp.Float(
                "dropout__dense_1",
                min_value=0.1,
                max_value=0.8,
                step=0.05
            )
        )
    )

    model.add(
        Dense(
            units=hp.Int(
                "dense_2",
                min_value=4,
                max_value=96,
                step=4),
            activation=_coerce_hparam_choice(
                hp.Choice(
                    "dense_2_activation",
                    ["relu", "sigmoid"]
                ),
                ["relu", "sigmoid"],
                "relu",
            )
        )
    )

    model.add(
        Dropout(
            hp.Float(
                "dropout_dense_2",
                min_value=0.1,
                max_value=0.8,
                step=0.05
            )
        )
    )

    # Output layers
    model.add(
        Dense(
            1,
            activation="linear"
        )
    )

    # Optimizer choice with gradient clipping to prevent exploding gradients
    optimizer_choice = _coerce_hparam_choice(
        hp.Choice("optimizer", ["adam", "rmsprop"]),
        ["adam", "rmsprop"],
        "adam",
    )
    learning_rate = _coerce_hparam_float(hp.Float("learning_rate", 1e-5, 1e-3, sampling="log"), 1e-3)
    clipnorm = _coerce_hparam_float(hp.Float("clipnorm", 0.5, 2.0, step=0.5), 1.0)

    if optimizer_choice == "adam":
        optimizer = Adam(learning_rate=learning_rate, clipnorm=clipnorm)
    else:
        optimizer = RMSprop(learning_rate=learning_rate, clipnorm=clipnorm)

    # Loss choice - MAE added to reduce mode collapse risk (less sensitive to outliers than MSE)
    loss_choice = _coerce_hparam_choice(
        hp.Choice(
            "loss",
            ["mean_absolute_error", "mean_squared_error", "huber", "mean_absolute_percentage_error"]
        ),
        ["mean_absolute_error", "mean_squared_error", "huber", "mean_absolute_percentage_error"],
        "mean_squared_error",
    )

    # Optimizer with tunable learning rate
    model.compile(
        optimizer=optimizer,
        loss=loss_choice,
        metrics=["mean_absolute_error", "mean_squared_error", "accuracy"]
    )
    return model

def tune_lstm_model(stock, x_train_lstm, y_train_lstm, x_val_lstm, y_val_lstm, time_steps, num_features, 
                    max_trials=25, executions_per_trial=1, epochs=50, retries=3, delay=5,
                    use_cached_hp=True, cache_max_age_days=30, cleanup_after_tuning=True,
                    require_cached_hp=False):
    """
    Tunes LSTM model hyperparameters using Keras Tuner.
    
    Parameters:
    - stock (str): Stock ticker
    - x_train_lstm (np.array): Pre-shaped training sequences (N, time_steps, num_features)
    - y_train_lstm (np.array): Training targets (N, 1)
    - x_val_lstm (np.array): Pre-shaped validation sequences
    - y_val_lstm (np.array): Validation targets
    - time_steps (int): Number of time steps in sequences
    - num_features (int): Number of features per time step
    - max_trials (int): Maximum hyperparameter trials
    - executions_per_trial (int): Executions per trial for averaging
    - epochs (int): Training epochs
    - retries (int): Number of retry attempts on failure
    - delay (int): Seconds to wait between retries
    - use_cached_hp (bool): If True, check database for cached hyperparameters first
    - cache_max_age_days (int): Maximum age of cached hyperparameters in days
    - cleanup_after_tuning (bool): If True, delete tuning directory after extracting model
    
    Returns:
    - best_model: Tuned Keras model
    
    Raises:
    - RuntimeError: If tuning fails after all retries
    """

    # Validate input shapes
    expected_shape = (x_train_lstm.shape[0], time_steps, num_features)
    if x_train_lstm.shape != expected_shape:
        raise ValueError(f"x_train_lstm shape mismatch. Expected {expected_shape}, got {x_train_lstm.shape}")
    
    # Check for cached hyperparameters in database first
    if use_cached_hp:
        cached = None
        try:
            from db_interactions import load_hyperparameters

            cached = load_hyperparameters(stock, 'lstm', max_age_days=cache_max_age_days)
        except CACHED_HP_LOAD_EXCEPTIONS as e:
            if require_cached_hp:
                raise PredictionTrainingRequiredError(
                    ticker=stock,
                    required_model_types=['lstm'],
                    missing_model_types=['lstm'],
                    message=f"Cached LSTM hyperparameters are unavailable for {stock}: {e}",
                ) from e
            print(f"[WARN] Failed to load cached LSTM hyperparameters, will retune: {e}")

        if cached is not None:
            print(f"[CACHE] Loading cached LSTM hyperparameters for {stock}")
            try:
                # Build model from cached hyperparameters
                hp = kt.HyperParameters()
                for key, value in cached.items():
                    hp.Fixed(key, value)
                
                best_model = build_lstm_model(hp, input_shape=(time_steps, num_features))
                best_model.build(input_shape=(None, time_steps, num_features))
                fit_cached_lstm_model(
                    best_model,
                    cached,
                    x_train_lstm,
                    y_train_lstm,
                    x_val_lstm,
                    y_val_lstm,
                    epochs=epochs,
                    verbose=0,
                )
                print("[CACHE] Successfully rebuilt and fit LSTM model from cached hyperparameters")
                return best_model
            except CACHED_MODEL_RESTORE_EXCEPTIONS as e:
                if require_cached_hp:
                    raise_cache_invalidated_error(
                        stock,
                        'lstm',
                        e,
                        invalidator=invalidate_hyperparameter_cache,
                    )
                print(f"[WARN] Failed to build from cached HP, will retune: {e}")

    if require_cached_hp:
        raise PredictionTrainingRequiredError(
            ticker=stock,
            required_model_types=['lstm'],
            missing_model_types=['lstm'],
        )
    
    # Setup directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tuning_dir = os.path.join(script_dir, "tuning_dir")
    temp_dir = os.path.join(tempfile.gettempdir(), "temp_tuning_dir")
    project_name_val = f"LSTM_tuning_{stock}"
    finished_project_path = os.path.join(tuning_dir, project_name_val)
    temp_project_path = os.path.join(temp_dir, project_name_val)

    print(f"Script directory: {script_dir}")
    print(f"Finished tuning directory: {finished_project_path}")
    print(f"Temporary tuning directory: {temp_project_path}")

    # PRIORITY 1: Check for finished tuning
    best_model = load_best_model_from_finished_tuning(
        finished_project_path,
        time_steps,
        num_features
    )
    if best_model is not None:
        print(f"[OK] Loaded best model from finished tuning: {finished_project_path}")
        return best_model

    # --- PRIORITY 2: Check for partial tuning and continue ---
    # Track tuning time for DB storage
    tuning_start_time = time.time()

    overwrite_val = False
    if os.path.exists(temp_project_path):
        print(f"[RESUME] Found partial tuning at {temp_project_path}. Continuing tuning...")
        overwrite_val = False  # Continue from existing
    else:
        print("[NEW] No partial tuning found. Starting new tuning...")
        overwrite_val = True

    # Define tuner with HyperModel class (FIXED APPROACH)
    class LSTMHyperModel(kt.HyperModel):
        def __init__(self, input_shape):
            super().__init__()
            self.input_shape = input_shape

        def declare_hyperparameters(self, hp):
            pass  # Hyperparameters declared in build method

        def build(self, hp):
            return build_lstm_model(hp, self.input_shape)

        def fit(self, hp, model, *args, **kwargs):
            # Build dynamic callbacks based on hyperparameters
            patience = hp.Int("patience", min_value=10, max_value=40, step=5)
            lr_schedule_choice = hp.Choice("lr_schedule", ["none", "reduce_on_plateau", "exp_decay"])
            
            # Monitor val_mae for MAE-based losses, val_loss otherwise
            monitor_metric = "val_mean_absolute_error"

            callbacks = [
                EarlyStopping(
                    monitor=monitor_metric,
                    patience=patience,
                    restore_best_weights=True,
                    verbose=1,
                    min_delta=0.0001  # Require meaningful improvement
                )
            ]

            if lr_schedule_choice == "reduce_on_plateau":
                callbacks.append(
                    ReduceLROnPlateau(
                        monitor=monitor_metric,
                        factor=0.5,
                        patience=patience // 2,
                        verbose=1,
                        min_lr=1e-7
                    )
                )
            elif lr_schedule_choice == "exp_decay":
                from tensorflow.keras.callbacks import LearningRateScheduler
                initial_lr = hp.get("learning_rate")
                decay_rate = hp.Float("decay_rate", 0.9, 0.99, step=0.01)

                def lr_schedule(_epoch, _lr):
                    return initial_lr * (decay_rate ** _epoch)

                callbacks.append(LearningRateScheduler(lr_schedule))

            # Add callbacks to kwargs
            kwargs['callbacks'] = callbacks

            # Call the default fit method
            return model.fit(*args, **kwargs)

    # Create hypermodel instance
    hypermodel = LSTMHyperModel(input_shape=(time_steps, num_features))

    # Define tuner with BayesianOptimization (more efficient than RandomSearch)
    tuner = kt.BayesianOptimization(
        hypermodel,
        objective="val_loss",
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        num_initial_points=min(5, max_trials // 2),  # Random exploration first
        alpha=0.0001,
        beta=2.6,
        directory=temp_dir,
        project_name=project_name_val,
        overwrite=overwrite_val
    )

    # Retry loop for search
    for attempt in range(retries):
        try:
            tuner.search(
                x_train_lstm,
                y_train_lstm,
                epochs=epochs,
                batch_size=kt.HyperParameters().Choice("batch_size", [4, 8, 12, 16, 20, 24, 28, 32]),
                validation_data=(x_val_lstm, y_val_lstm),
                verbose=1
            )
            break  # Success

        except tf.errors.ResourceExhaustedError as oom_error:
            print(f"[WARN] OOM Error on attempt {attempt+1}: {oom_error}")
            print("[CLEANUP] Clearing GPU memory and skipping failed trial...")

            tf.keras.backend.clear_session()
            import gc
            gc.collect()

            if attempt < retries - 1:
                print(f"[WAIT] Waiting {delay} seconds before retry...")
                time.sleep(delay)
            else:
                raise RuntimeError("LSTM tuning failed after all retries due to OOM errors") from oom_error

        except (UnicodeDecodeError, tf.errors.FailedPreconditionError, tf.errors.InternalError) as e:
            print(f"Attempt {attempt+1} failed: {e}")
            tf.keras.backend.clear_session()
            import gc
            gc.collect()
            if os.path.exists(temp_project_path):
                shutil.rmtree(temp_project_path)
            if attempt < retries - 1:
                print(f"[WAIT] Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                raise RuntimeError(f"LSTM tuning failed after {retries} attempts: {e}") from e

    # Get best model
    best_trials = tuner.oracle.get_best_trials(num_trials=1)
    if not best_trials:
        raise RuntimeError("Keras Tuner failed to find any successful trials.")

    best_hp = best_trials[0].hyperparameters

    print("[OK] Best hyperparameters found:")
    for param, value in best_hp.values.items():
        print(f"- {param}: {value}")

    best_model = _restore_or_refit_sequence_tuner_model(
        tuner=tuner,
        best_hp=best_hp,
        build_model=build_lstm_model,
        fit_model=fit_cached_lstm_model,
        x_train=x_train_lstm,
        y_train=y_train_lstm,
        x_val=x_val_lstm,
        y_val=y_val_lstm,
        time_steps=time_steps,
        num_features=num_features,
        epochs=epochs,
        model_label='LSTM',
    )
    print("Best model architecture:")
    print(best_model.summary())

    # Save hyperparameters to database for future use
    tuning_time = time.time() - tuning_start_time
    try:
        from db_interactions import save_hyperparameters
        hp_dict = dict(best_hp.values)
        
        # Get validation loss from best trial
        val_loss = best_trials[0].metrics.get_best_value('val_loss')
        val_mae = best_trials[0].metrics.get_best_value('val_mean_absolute_error')
        
        # Calculate val_r2 on validation set
        val_pred = best_model.predict(x_val_lstm, verbose=0).flatten()
        from sklearn.metrics import r2_score as _r2_score
        val_r2_score = _r2_score(y_val_lstm.flatten(), val_pred)
        
        save_hyperparameters(
            ticker=stock,
            model_type='lstm',
            hyperparameters=hp_dict,
            num_trials=max_trials,
            best_score=val_loss if val_loss else val_mae,
            tuning_time_seconds=tuning_time,
            training_samples=len(x_train_lstm),
            num_features=num_features,
            val_mse=val_loss,
            val_r2=val_r2_score,
            val_mae=val_mae
        )
        print(f"[OK] LSTM hyperparameters saved to database for {stock}")
    except HYPERPARAMETER_SAVE_EXCEPTIONS as e:
        print(f"[WARN] Failed to save LSTM hyperparameters to database: {e}")

    # Cleanup temp dir
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            print("[CLEANUP] Cleaned up local temp directory.")
        except (OSError, PermissionError) as e:
            print(f"Warning: Could not delete temp directory {temp_dir}. Error: {e}")

    if cleanup_after_tuning:
        # Delete the temp project path since we have the model and saved HP to DB
        if os.path.exists(temp_project_path):
            try:
                shutil.rmtree(temp_project_path)
                print(f"[OK] Cleaned up LSTM tuning directory: {temp_project_path}")
            except (OSError, PermissionError) as e:
                print(f"[WARN] Could not delete LSTM tuning directory: {e}")
        # Also delete any existing final destination
        final_dest = os.path.join(script_dir, f"tuning_dir/{project_name_val}")
        if os.path.exists(final_dest):
            try:
                shutil.rmtree(final_dest)
                print(f"[OK] Cleaned up final LSTM tuning directory: {final_dest}")
            except (OSError, PermissionError) as e:
                print(f"[WARN] Could not delete final LSTM tuning directory: {e}")
    else:
        # Move tuning folder to script directory (original behavior)
        final_dest = os.path.join(script_dir, f"tuning_dir/{project_name_val}")
        print(f"Moving tuning folder to: {final_dest}")
        if os.path.exists(final_dest):
            shutil.rmtree(final_dest)
        shutil.move(temp_project_path, final_dest)

    return best_model

def evaluate_lstm_model(model, x_train, y_train, x_val, y_val, x_test, y_test):
    """
    Evaluate LSTM model on all three sets.

    Returns:
    - train_metrics, val_metrics, test_metrics (dicts with mse, r2, and mae)
    """
    train_pred = model.predict(x_train, verbose=0)
    val_pred = model.predict(x_val, verbose=0)
    test_pred = model.predict(x_test, verbose=0)

    train_metrics = {
        'mse': mean_squared_error(y_train, train_pred),
        'r2': r2_score(y_train, train_pred),
        'mae': mean_absolute_error(y_train, train_pred)
    }

    val_metrics = {
        'mse': mean_squared_error(y_val, val_pred),
        'r2': r2_score(y_val, val_pred),
        'mae': mean_absolute_error(y_val, val_pred)
    }

    test_metrics = {
        'mse': mean_squared_error(y_test, test_pred),
        'r2': r2_score(y_test, test_pred),
        'mae': mean_absolute_error(y_test, test_pred)
    }

    return train_metrics, val_metrics, test_metrics


# =============================================================================
# TCN (TEMPORAL CONVOLUTIONAL NETWORK) IMPLEMENTATION
# =============================================================================
# TCN is an alternative to LSTM that:
# - Is less prone to mode collapse
# - Can be parallelized (faster training)
# - Has better gradient flow through dilated convolutions
# - Captures long-range dependencies more effectively
# =============================================================================

def tcn_residual_block(x, dilation_rate, nb_filters, kernel_size, dropout_rate=0.1, 
                       use_layer_norm=True, l2_reg=1e-4):
    """
    Create a single TCN residual block with dilated causal convolution.
    
    The residual block consists of:
    1. Dilated causal convolution
    2. Layer normalization (optional)
    3. Activation (ReLU)
    4. Dropout
    5. Skip connection (residual)
    
    Parameters:
    - x: Input tensor
    - dilation_rate: Dilation rate for causal convolution
    - nb_filters: Number of convolutional filters
    - kernel_size: Size of convolution kernel
    - dropout_rate: Dropout rate for regularization
    - use_layer_norm: Whether to use layer normalization
    - l2_reg: L2 regularization strength
    
    Returns:
    - Output tensor after residual block
    """
    # First dilated causal convolution
    conv1 = Conv1D(
        filters=nb_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding='causal',
        kernel_regularizer=regularizers.l2(l2_reg),
        kernel_initializer='he_normal'
    )(x)
    
    if use_layer_norm:
        conv1 = LayerNormalization()(conv1)
    
    conv1 = Activation('relu')(conv1)
    conv1 = Dropout(dropout_rate)(conv1)
    
    # Second dilated causal convolution
    conv2 = Conv1D(
        filters=nb_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding='causal',
        kernel_regularizer=regularizers.l2(l2_reg),
        kernel_initializer='he_normal'
    )(conv1)
    
    if use_layer_norm:
        conv2 = LayerNormalization()(conv2)
    
    conv2 = Activation('relu')(conv2)
    conv2 = Dropout(dropout_rate)(conv2)
    
    # Skip connection (1x1 convolution if channel dimensions don't match)
    if x.shape[-1] != nb_filters:
        x = Conv1D(filters=nb_filters, kernel_size=1, padding='same')(x)
    
    # Residual connection
    return Add()([x, conv2])


def build_tcn_model(hp, input_shape):
    """
    Build a Temporal Convolutional Network (TCN) model with tunable hyperparameters.
    
    TCN Architecture:
    - Stack of residual blocks with increasing dilation rates (1, 2, 4, 8, 16, ...)
    - Causal convolutions to preserve temporal ordering
    - Residual connections for stable gradient flow
    - Layer normalization for training stability
    
    Parameters:
    - hp (keras_tuner.HyperParameters): Keras Tuner hyperparameters object
    - input_shape (tuple): Shape of input sequences (time_steps, num_features)
    
    Returns:
    - keras.Model: Compiled TCN model ready for training
    
    Hyperparameters tuned:
    - nb_filters: Number of filters per layer (32-128)
    - kernel_size: Convolution kernel size (2-8)
    - nb_stacks: Number of residual stacks (1-3)
    - dilations_per_stack: Number of dilation levels per stack (4-7)
    - dropout_rate: Dropout rate (0.05-0.3)
    - l2_reg: L2 regularization (1e-6 to 1e-3)
    - dense_units: Dense layer units before output (16-64)
    - optimizer: Optimizer choice (adam, rmsprop)
    - learning_rate: Learning rate (1e-5 to 1e-3)
    - loss: Loss function (MAE, MSE, Huber)
    """
    # Input layer
    inputs = Input(shape=input_shape)
    
    # TCN hyperparameters
    nb_filters = hp.Int('tcn_nb_filters', min_value=32, max_value=128, step=16)
    kernel_size = hp.Int('tcn_kernel_size', min_value=2, max_value=8, step=2)
    nb_stacks = hp.Int('tcn_nb_stacks', min_value=1, max_value=3, step=1)
    dilations_per_stack = hp.Int('tcn_dilations_per_stack', min_value=4, max_value=7, step=1)
    dropout_rate = hp.Float('tcn_dropout', min_value=0.05, max_value=0.3, step=0.05)
    l2_reg = hp.Float('tcn_l2_reg', min_value=1e-6, max_value=1e-3, sampling='log')
    use_layer_norm = hp.Boolean('tcn_use_layer_norm', default=True)
    
    # Initial projection to filter dimension
    x = Conv1D(nb_filters, kernel_size=1, padding='same')(inputs)
    
    # Build TCN stacks with exponentially increasing dilation rates
    # Dilation rates: [1, 2, 4, 8, 16, ...] capture different temporal scales
    for _ in range(nb_stacks):
        for i in range(dilations_per_stack):
            dilation_rate = 2 ** i  # Exponential dilation: 1, 2, 4, 8, 16, 32, 64
            x = tcn_residual_block(
                x, 
                dilation_rate=dilation_rate,
                nb_filters=nb_filters,
                kernel_size=kernel_size,
                dropout_rate=dropout_rate,
                use_layer_norm=use_layer_norm,
                l2_reg=l2_reg
            )
    
    # Use only the last time step (causal: no future information)
    # Alternative: GlobalAveragePooling1D() for aggregated representation
    use_global_pooling = hp.Boolean('tcn_use_global_pooling', default=False)
    
    if use_global_pooling:
        x = GlobalAveragePooling1D()(x)
    else:
        # Take only the last time step output
        x = x[:, -1, :]
    
    # Dense layer before output
    dense_units = hp.Int('tcn_dense_units', min_value=16, max_value=64, step=16)
    x = Dense(dense_units, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = Dropout(dropout_rate)(x)
    
    # Output layer (linear for regression)
    outputs = Dense(1, activation='linear')(x)
    
    # Build model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Optimizer configuration
    optimizer_choice = hp.Choice('tcn_optimizer', ['adam', 'rmsprop'])
    learning_rate = hp.Float('tcn_learning_rate', min_value=1e-5, max_value=1e-3, sampling='log')
    clipnorm = hp.Float('tcn_clipnorm', min_value=0.5, max_value=2.0, step=0.5)
    
    if optimizer_choice == 'adam':
        optimizer = Adam(learning_rate=learning_rate, clipnorm=clipnorm)
    else:
        optimizer = RMSprop(learning_rate=learning_rate, clipnorm=clipnorm)
    
    # Loss function - prioritize MAE and Huber (more robust to outliers)
    loss_choice = hp.Choice('tcn_loss', ['mean_absolute_error', 'huber', 'mean_squared_error'])
    
    model.compile(
        optimizer=optimizer,
        loss=loss_choice,
        metrics=['mean_absolute_error', 'mean_squared_error']
    )
    
    return model


def tune_tcn_model(stock, x_train, y_train, x_val, y_val, time_steps, num_features, 
                   max_trials=30, epochs=100, retries=3, delay=5,
                   use_cached_hp=True, cache_max_age_days=30, cleanup_after_tuning=True,
                   require_cached_hp=False):
    """
    Tune TCN model hyperparameters using Keras Tuner.
    
    Parameters:
    - stock (str): Stock ticker for naming the tuning project
    - x_train (np.array): Training sequences (N, time_steps, num_features)
    - y_train (np.array): Training targets (N, 1)
    - x_val (np.array): Validation sequences
    - y_val (np.array): Validation targets
    - time_steps (int): Number of time steps in sequences
    - num_features (int): Number of features per time step
    - max_trials (int): Maximum hyperparameter trials
    - epochs (int): Training epochs per trial
    - retries (int): Number of retry attempts on failure
    - delay (int): Seconds to wait between retries
    - use_cached_hp (bool): If True, check database for cached hyperparameters first
    - cache_max_age_days (int): Maximum age of cached hyperparameters in days
    - cleanup_after_tuning (bool): If True, delete tuning directory after extracting model
    
    Returns:
    - best_model: Tuned TCN model
    
    Raises:
    - RuntimeError: If tuning fails after all retries
    """
    # Validate input shapes
    expected_shape = (x_train.shape[0], time_steps, num_features)
    if x_train.shape != expected_shape:
        raise ValueError(f"x_train shape mismatch. Expected {expected_shape}, got {x_train.shape}")
    
    # Check for cached hyperparameters in database first
    if use_cached_hp:
        cached = None
        try:
            from db_interactions import load_hyperparameters

            cached = load_hyperparameters(stock, 'tcn', max_age_days=cache_max_age_days)
        except CACHED_HP_LOAD_EXCEPTIONS as e:
            if require_cached_hp:
                raise PredictionTrainingRequiredError(
                    ticker=stock,
                    required_model_types=['tcn'],
                    missing_model_types=['tcn'],
                    message=f"Cached TCN hyperparameters are unavailable for {stock}: {e}",
                ) from e
            print(f"[WARN] Failed to load cached TCN hyperparameters, will retune: {e}")

        if cached is not None:
            print(f"[CACHE] Loading cached TCN hyperparameters for {stock}")
            try:
                # Build model from cached hyperparameters
                hp = kt.HyperParameters()
                for key, value in cached.items():
                    # Set each hyperparameter with its cached value
                    hp.Fixed(key, value)
                
                best_model = build_tcn_model(hp, input_shape=(time_steps, num_features))
                best_model.build(input_shape=(None, time_steps, num_features))
                fit_cached_tcn_model(
                    best_model,
                    cached,
                    x_train,
                    y_train,
                    x_val,
                    y_val,
                    epochs=epochs,
                    verbose=0,
                )
                print("[CACHE] Successfully rebuilt and fit TCN model from cached hyperparameters")
                return best_model
            except CACHED_MODEL_RESTORE_EXCEPTIONS as e:
                if require_cached_hp:
                    raise_cache_invalidated_error(
                        stock,
                        'tcn',
                        e,
                        invalidator=invalidate_hyperparameter_cache,
                    )
                print(f"[WARN] Failed to build from cached HP, will retune: {e}")

    if require_cached_hp:
        raise PredictionTrainingRequiredError(
            ticker=stock,
            required_model_types=['tcn'],
            missing_model_types=['tcn'],
        )
    
    # Setup directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tuning_dir = os.path.join(script_dir, "tuning_dir")
    temp_dir = os.path.join(tempfile.gettempdir(), "temp_tuning_dir")
    project_name = f"TCN_tuning_{stock}"
    finished_project_path = os.path.join(tuning_dir, project_name)
    temp_project_path = os.path.join(temp_dir, project_name)
    
    print(f"\n[TCN] Starting TCN tuning for {stock}")
    print(f"   Finished tuning directory: {finished_project_path}")
    print(f"   Temporary tuning directory: {temp_project_path}")
    
    # PRIORITY 1: Check for finished tuning
    best_model = load_best_tcn_model(finished_project_path, time_steps, num_features)
    if best_model is not None:
        print(f"[OK] Loaded best TCN model from finished tuning: {finished_project_path}")
        return best_model
    
    # Track tuning time for DB storage
    tuning_start_time = time.time()

    # PRIORITY 2: Check for partial tuning and continue
    overwrite_val = not os.path.exists(temp_project_path)
    if not overwrite_val:
        print(f"[RESUME] Found partial TCN tuning at {temp_project_path}. Continuing...")
    else:
        print("[NEW] No partial TCN tuning found. Starting new tuning...")
    
    # Define TCN HyperModel
    class TCNHyperModel(kt.HyperModel):
        def __init__(self, input_shape):
            super().__init__()
            self.input_shape = input_shape
        
        def build(self, hp):
            return build_tcn_model(hp, self.input_shape)
        
        def fit(self, hp, model, *args, **kwargs):
            # Dynamic callbacks
            patience = hp.Int('tcn_patience', min_value=10, max_value=30, step=5)
            monitor_metric = 'val_mean_absolute_error'
            
            callbacks = [
                EarlyStopping(
                    monitor=monitor_metric,
                    patience=patience,
                    restore_best_weights=True,
                    verbose=1,
                    min_delta=0.0001
                ),
                ReduceLROnPlateau(
                    monitor=monitor_metric,
                    factor=0.5,
                    patience=patience // 2,
                    verbose=1,
                    min_lr=1e-7
                )
            ]
            
            kwargs['callbacks'] = callbacks
            return model.fit(*args, **kwargs)
    
    # Create tuner
    tuner = kt.RandomSearch(
        TCNHyperModel(input_shape=(time_steps, num_features)),
        objective=kt.Objective('val_mean_absolute_error', direction='min'),
        max_trials=max_trials,
        directory=temp_dir,
        project_name=project_name,
        overwrite=overwrite_val
    )
    
    # Run tuning with retries
    for attempt in range(retries):
        try:
            print(f"\n[TCN] Tuning attempt {attempt + 1}/{retries}")
            
            tuner.search(
                x_train, y_train,
                epochs=epochs,
                validation_data=(x_val, y_val),
                verbose=1,
                batch_size=32
            )
            break  # Success
            
        except tf.errors.ResourceExhaustedError as oom_error:
            print(f"[WARN] OOM Error on TCN attempt {attempt+1}: {oom_error}")
            tf.keras.backend.clear_session()
            import gc
            gc.collect()
            
            if attempt < retries - 1:
                print(f"[WAIT] Waiting {delay} seconds before retry...")
                time.sleep(delay)
            else:
                raise RuntimeError("TCN tuning failed due to OOM errors") from oom_error
                
        except Exception as e:
            print(f"[ERROR] TCN tuning attempt {attempt+1} failed: {e}")
            if os.path.exists(temp_project_path):
                shutil.rmtree(temp_project_path)
            if attempt < retries - 1:
                print(f"[WAIT] Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                raise RuntimeError(f"TCN tuning failed after {retries} attempts: {e}") from e
    
    # Get best model
    best_trials = tuner.oracle.get_best_trials(num_trials=1)
    if not best_trials:
        raise RuntimeError("TCN Tuner failed to find any successful trials.")
    
    best_hp = best_trials[0].hyperparameters
    
    print("\n[OK] Best TCN hyperparameters found:")
    for param, value in best_hp.values.items():
        if param.startswith('tcn_'):
            print(f"   - {param}: {value}")
    
    best_model = _restore_or_refit_sequence_tuner_model(
        tuner=tuner,
        best_hp=best_hp,
        build_model=build_tcn_model,
        fit_model=fit_cached_tcn_model,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        time_steps=time_steps,
        num_features=num_features,
        epochs=epochs,
        model_label='TCN',
    )
    
    print("\nTCN Model Summary:")
    best_model.summary()
    
    # Save hyperparameters to database for future use
    tuning_time = time.time() - tuning_start_time
    try:
        from db_interactions import save_hyperparameters
        # Extract only tcn_ parameters for storage
        hp_dict = {k: v for k, v in best_hp.values.items() if k.startswith('tcn_')}
        
        # Get validation loss from best trial
        val_loss = best_trials[0].metrics.get_best_value('val_loss')
        val_mae = best_trials[0].metrics.get_best_value('val_mean_absolute_error')
        
        # Calculate val_r2 on validation set
        val_pred = best_model.predict(x_val, verbose=0).flatten()
        from sklearn.metrics import r2_score as _r2_score
        val_r2_score = _r2_score(y_val.flatten(), val_pred)
        
        save_hyperparameters(
            ticker=stock,
            model_type='tcn',
            hyperparameters=hp_dict,
            num_trials=max_trials,
            best_score=val_loss if val_loss else val_mae,
            tuning_time_seconds=tuning_time,
            training_samples=len(x_train),
            num_features=num_features,
            val_mse=val_loss,
            val_r2=val_r2_score,
            val_mae=val_mae
        )
        print(f"[OK] TCN hyperparameters saved to database for {stock}")
    except HYPERPARAMETER_SAVE_EXCEPTIONS as e:
        print(f"[WARN] Failed to save TCN hyperparameters to database: {e}")
    
    # Cleanup temp directory and optionally the final tuning directory
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
        except (OSError, PermissionError) as e:
            print(f"[WARN] Could not delete temp directory: {e}")
    
    if cleanup_after_tuning:
        # Delete the temp project path since we have the model and saved HP to DB
        if os.path.exists(temp_project_path):
            try:
                shutil.rmtree(temp_project_path)
                print(f"[OK] Cleaned up TCN tuning directory: {temp_project_path}")
            except (OSError, PermissionError) as e:
                print(f"[WARN] Could not delete TCN tuning directory: {e}")
        # Also delete any existing final destination
        final_dest = os.path.join(script_dir, f"tuning_dir/{project_name}")
        if os.path.exists(final_dest):
            try:
                shutil.rmtree(final_dest)
                print(f"[OK] Cleaned up final TCN tuning directory: {final_dest}")
            except (OSError, PermissionError) as e:
                print(f"[WARN] Could not delete final TCN tuning directory: {e}")
    else:
        # Move tuning folder to permanent location (original behavior)
        final_dest = os.path.join(script_dir, f"tuning_dir/{project_name}")
        print(f"Moving TCN tuning folder to: {final_dest}")
        if os.path.exists(final_dest):
            shutil.rmtree(final_dest)
        shutil.move(temp_project_path, final_dest)
    
    return best_model


def load_best_tcn_model(finished_project_path, time_steps, num_features):
    """
    Load a previously tuned TCN model from disk.
    
    Parameters:
    - finished_project_path: Path to finished tuning project
    - time_steps: Number of time steps for input shape
    - num_features: Number of features per time step
    
    Returns:
    - keras.Model: Loaded TCN model, or None if not found
    """
    if not os.path.exists(finished_project_path):
        print(f"[INFO] No finished TCN tuning found at: {finished_project_path}")
        return None
    
    try:
        oracle_path = os.path.join(finished_project_path, "oracle.json")
        if not os.path.exists(oracle_path):
            print(f"[WARN] oracle.json not found in {finished_project_path}")
            return None
        
        # Load tuner to get best hyperparameters
        temp_tuner = kt.RandomSearch(
            lambda hp: build_tcn_model(hp, input_shape=(time_steps, num_features)),
            objective='val_loss',
            max_trials=1,
            directory=os.path.dirname(finished_project_path),
            project_name=os.path.basename(finished_project_path),
            overwrite=False
        )
        
        best_trials = temp_tuner.oracle.get_best_trials(num_trials=1)
        if not best_trials:
            print("[WARN] No successful TCN trials found")
            return None
        
        best_hp = best_trials[0].hyperparameters
        best_model = build_tcn_model(best_hp, input_shape=(time_steps, num_features))
        best_model.build(input_shape=(None, time_steps, num_features))
        
        print(f"[OK] Loaded TCN model from {finished_project_path}")
        return best_model
        
    except Exception as e:
        print(f"[ERROR] Failed to load TCN model: {e}")
        return None


def evaluate_tcn_model(model, x_train, y_train, x_val, y_val, x_test, y_test):
    """
    Evaluate TCN model on all three sets.
    
    Parameters:
    - model: Trained TCN model
    - x_train, y_train: Training data
    - x_val, y_val: Validation data
    - x_test, y_test: Test data
    
    Returns:
    - train_metrics, val_metrics, test_metrics (dicts with mse, r2, mae)
    """
    train_pred = model.predict(x_train, verbose=0)
    val_pred = model.predict(x_val, verbose=0)
    test_pred = model.predict(x_test, verbose=0)
    
    train_metrics = {
        'mse': mean_squared_error(y_train, train_pred),
        'r2': r2_score(y_train, train_pred),
        'mae': mean_absolute_error(y_train, train_pred)
    }
    
    val_metrics = {
        'mse': mean_squared_error(y_val, val_pred),
        'r2': r2_score(y_val, val_pred),
        'mae': mean_absolute_error(y_val, val_pred)
    }
    
    test_metrics = {
        'mse': mean_squared_error(y_test, test_pred),
        'r2': r2_score(y_test, test_pred),
        'mae': mean_absolute_error(y_test, test_pred)
    }
    
    return train_metrics, val_metrics, test_metrics


# ============================================================================
# MONTE CARLO DROPOUT AND PREDICTION AGGREGATION
# ============================================================================

def predict_with_uncertainty(model, x_input, n_iterations=50, model_type='tcn'):
    """
    Run multiple predictions using Monte Carlo Dropout to get mean + uncertainty estimate.
    
    Monte Carlo Dropout keeps dropout active during inference to sample from
    the model's learned distribution. This provides:
    - Mean prediction (consensus across all runs)
    - Standard deviation (uncertainty measure)
    - Confidence intervals (5th-95th percentile)
    
    Parameters:
    - model: Trained Keras model with dropout layers
    - x_input: Input data shaped for the model (batch_size, time_steps, features)
    - n_iterations: Number of forward passes with dropout (default: 50)
    - model_type: Type of model ('tcn', 'lstm') for logging
    
    Returns:
    - dict with keys:
        - 'mean': Average of all predictions
        - 'std': Standard deviation (uncertainty)
        - 'median': Median prediction
        - 'percentile_5': 5th percentile (lower bound of 90% CI)
        - 'percentile_95': 95th percentile (upper bound of 90% CI)
        - 'all_predictions': Array of individual predictions
        - 'confidence': Confidence level based on std (High/Medium/Low)
    """
    predictions = []
    
    for i in range(n_iterations):
        # With training=True, dropout remains active during inference
        # This samples from the model's uncertainty distribution
        try:
            pred = model(x_input, training=True)
            pred_value = pred.numpy().flatten()[0] if hasattr(pred, 'numpy') else float(pred.flatten()[0])
            predictions.append(pred_value)
        except Exception as e:
            # Fallback to regular predict if training mode fails
            if i == 0:
                print(f"[WARN] {model_type.upper()} MC Dropout failed, using regular predict: {e}")
            pred = model.predict(x_input, verbose=0)
            predictions.append(pred.flatten()[0])
    
    predictions = np.array(predictions)
    
    # Calculate statistics
    mean_pred = np.mean(predictions)
    std_pred = np.std(predictions)
    
    # Determine confidence level based on coefficient of variation
    # CV = std / |mean| (normalized uncertainty)
    cv = std_pred / (abs(mean_pred) + 1e-8)
    if cv < 0.3:
        confidence = 'High'
    elif cv < 0.7:
        confidence = 'Medium'
    else:
        confidence = 'Low'
    
    return {
        'mean': mean_pred,
        'std': std_pred,
        'median': np.median(predictions),
        'percentile_5': np.percentile(predictions, 5),
        'percentile_95': np.percentile(predictions, 95),
        'percentile_25': np.percentile(predictions, 25),
        'percentile_75': np.percentile(predictions, 75),
        'all_predictions': predictions,
        'confidence': confidence,
        'n_iterations': n_iterations
    }


def aggregate_model_predictions(predictions_dict, weights=None):
    """
    Combine predictions from multiple models using weighted averaging.
    
    Supports three weighting strategies:
    1. Equal weights (weights=None)
    2. Custom weights (weights provided as dict)
    3. Uncertainty-weighted (weights='uncertainty') - inverse variance weighting
    
    Parameters:
    - predictions_dict: Dict of {model_name: {'mean': value, 'std': uncertainty, ...}}
    - weights: None for equal, dict for custom, or 'uncertainty' for inverse-variance
    
    Returns:
    - dict with:
        - 'ensemble_mean': Weighted average prediction
        - 'ensemble_std': Combined uncertainty
        - 'model_weights': Dict of actual weights used
        - 'model_contributions': Each model's contribution to final prediction
    """
    model_names = list(predictions_dict.keys())
    means = np.array([predictions_dict[m]['mean'] for m in model_names])
    stds = np.array([predictions_dict[m].get('std', 0.01) for m in model_names])
    
    # Determine weights
    if weights is None:
        # Equal weighting
        w = np.ones(len(model_names)) / len(model_names)
    elif weights == 'uncertainty':
        # Inverse-variance weighting (lower uncertainty = higher weight)
        variances = stds ** 2 + 1e-8  # Add small epsilon to avoid division by zero
        inv_variances = 1.0 / variances
        w = inv_variances / inv_variances.sum()
    elif isinstance(weights, dict):
        # Custom weights
        w = np.array([weights.get(m, 1.0) for m in model_names])
        w = w / w.sum()  # Normalize
    else:
        raise ValueError(f"Unknown weights type: {weights}")
    
    # Weighted average prediction
    ensemble_mean = np.sum(means * w)
    
    # Combined uncertainty (assuming independence)
    # Var(weighted sum) = sum(w_i^2 * var_i)
    ensemble_variance = np.sum((w ** 2) * (stds ** 2))
    ensemble_std = np.sqrt(ensemble_variance)
    
    # Model contributions
    contributions = {m: float(means[i] * w[i]) for i, m in enumerate(model_names)}
    weight_dict = {m: float(w[i]) for i, m in enumerate(model_names)}
    
    return {
        'ensemble_mean': ensemble_mean,
        'ensemble_std': ensemble_std,
        'model_weights': weight_dict,
        'model_contributions': contributions,
        'individual_predictions': {m: predictions_dict[m]['mean'] for m in model_names}
    }


def multi_run_prediction(model, x_input, scaler_y, n_runs=30, model_type='tcn'):
    """
    Perform multiple prediction runs with Monte Carlo Dropout and aggregate results.
    
    This is the main function for uncertainty-aware predictions:
    1. Runs n_runs forward passes with dropout active
    2. Inverse transforms predictions to original scale
    3. Computes statistics and confidence intervals
    
    Parameters:
    - model: Trained Keras model (TCN or LSTM)
    - x_input: Input data shaped for model
    - scaler_y: MinMaxScaler used for target variable
    - n_runs: Number of Monte Carlo iterations (default: 30)
    - model_type: 'tcn' or 'lstm'
    
    Returns:
    - dict with prediction statistics in ORIGINAL scale (percentage returns)
    """
    # Get predictions with uncertainty (in scaled space)
    mc_results = predict_with_uncertainty(model, x_input, n_iterations=n_runs, model_type=model_type)
    
    # Inverse transform all predictions to original scale
    scaled_predictions = mc_results['all_predictions'].reshape(-1, 1)
    original_predictions = scaler_y.inverse_transform(scaled_predictions).flatten()
    
    # Recalculate statistics in original scale
    mean_pred = np.mean(original_predictions)
    std_pred = np.std(original_predictions)
    
    # Confidence level based on coefficient of variation
    cv = std_pred / (abs(mean_pred) + 1e-8)
    if cv < 0.5:
        confidence = 'High'
    elif cv < 1.0:
        confidence = 'Medium'
    else:
        confidence = 'Low'
    
    return {
        'mean': mean_pred,
        'std': std_pred,
        'median': np.median(original_predictions),
        'percentile_5': np.percentile(original_predictions, 5),
        'percentile_95': np.percentile(original_predictions, 95),
        'percentile_25': np.percentile(original_predictions, 25),
        'percentile_75': np.percentile(original_predictions, 75),
        'min': np.min(original_predictions),
        'max': np.max(original_predictions),
        'confidence': confidence,
        'n_runs': n_runs,
        'all_predictions': original_predictions
    }


def load_best_model_from_finished_tuning(finished_project_path, time_steps, num_features):
    """
    Attempts to load a previously tuned best model from the finished tuning directory.
    
    Parameters:
    - finished_project_path (str): Path to the finished tuning project directory
    - time_steps (int): Number of time steps for LSTM input shape
    - num_features (int): Number of features per time step
    
    Returns:
    - keras.Model: The loaded best model, or None if not found/loadable
    """
    if not os.path.exists(finished_project_path):
        print(f"[INFO] No finished tuning found at: {finished_project_path}")
        return None

    try:
        # Look for the best model file
        oracle_path = os.path.join(finished_project_path, "oracle.json")

        if not os.path.exists(oracle_path):
            print(f"[WARN] oracle.json not found in {finished_project_path}")
            return None

        # Load the tuner to get best hyperparameters
        temp_tuner = kt.RandomSearch(
            lambda hp: build_lstm_model(hp, input_shape=(time_steps, num_features)),
            objective="val_loss",
            max_trials=1,
            directory=os.path.dirname(finished_project_path),
            project_name=os.path.basename(finished_project_path),
            overwrite=False
        )

        # Get best hyperparameters
        best_trials = temp_tuner.oracle.get_best_trials(num_trials=1)

        if not best_trials:
            print(f"[WARN] No successful trials found in {finished_project_path}")
            return None

        best_hp = best_trials[0].hyperparameters

        # Rebuild the model with best hyperparameters
        best_model = build_lstm_model(best_hp, input_shape=(time_steps, num_features))
        best_model.build(input_shape=(None, time_steps, num_features))

        print("[OK] Successfully loaded best model from finished tuning:")
        print(f"   Path: {finished_project_path}")
        print("   Best hyperparameters:")
        for param, value in best_hp.values.items():
            print(f"     - {param}: {value}")

        return best_model

    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as e:
        print(f"[WARN] Failed to load model from {finished_project_path}")
        print(f"   Error: {e}")
        print("   Will start new tuning instead.")
        return None

def detect_overfitting(train_metrics, val_metrics, test_metrics, model_name, threshold=0.15, use_multi_metric=True):
    """
    Detect overfitting by comparing training, validation, and test metrics.
    Uses multiple metrics (MSE, R², MAE) for robust detection.
    
    Parameters:
    - train_metrics (dict): Training metrics with 'mse', 'r2', 'mae'
    - val_metrics (dict): Validation metrics with 'mse', 'r2', 'mae'
    - test_metrics (dict): Test metrics with 'mse', 'r2', 'mae'
    - model_name (str): Name of the model for logging
    - threshold (float): Maximum allowed degradation ratio (default 0.15 = 15%)
    - use_multi_metric (bool): Use multiple metrics for detection (default True)
    
    Returns:
    - is_overfitted (bool): True if overfitting detected
    - overfitting_score (float): Severity of overfitting (0 = none, >1 = severe)
    - details (dict): Diagnostic breakdown of metrics and score components
    """
    details = {
        'model_name': model_name,
        'threshold': threshold,
        'use_multi_metric': use_multi_metric,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
    }

    if use_multi_metric:
        # ===== MULTI-METRIC OVERFITTING DETECTION =====
        
        # 1. MSE Degradation (lower is better)
        train_val_mse_ratio = (val_metrics['mse'] - train_metrics['mse']) / train_metrics['mse']
        val_test_mse_ratio = (test_metrics['mse'] - val_metrics['mse']) / val_metrics['mse']
        mse_score = max(train_val_mse_ratio, val_test_mse_ratio)
        
        # 2. R² Degradation (higher is better, so invert the logic)
        train_val_r2_ratio = (train_metrics['r2'] - val_metrics['r2']) / max(abs(train_metrics['r2']), 0.01)
        val_test_r2_ratio = (val_metrics['r2'] - test_metrics['r2']) / max(abs(val_metrics['r2']), 0.01)
        r2_degradation_score = max(train_val_r2_ratio, val_test_r2_ratio)
        
        # 3. MAE Degradation (lower is better)
        train_val_mae_ratio = (val_metrics['mae'] - train_metrics['mae']) / train_metrics['mae']
        val_test_mae_ratio = (test_metrics['mae'] - val_metrics['mae']) / val_metrics['mae']
        mae_score = max(train_val_mae_ratio, val_test_mae_ratio)
        
        # 4. Consistency Score (how aligned are the metrics?)
        # If MSE increases but R² doesn't degrade proportionally, something is off
        metric_scores = [mse_score, r2_degradation_score, mae_score]
        consistency_score = np.std(metric_scores) / (np.mean(np.abs(metric_scores)) + 0.01)
        
        # 5. Combined Overfitting Score (weighted average)
        # MSE and MAE are most important, R² secondary, consistency is a tiebreaker
        overfitting_score = (
            0.40 * mse_score + 
            0.25 * r2_degradation_score + 
            0.30 * mae_score + 
            0.05 * consistency_score
        )
        details.update({
            'mse_score': mse_score,
            'r2_degradation_score': r2_degradation_score,
            'mae_score': mae_score,
            'consistency_score': consistency_score,
        })
        
        is_overfitted = overfitting_score >= threshold
        
        print(f"\n{'='*60}")
        print(f"[DETECT] MULTI-METRIC OVERFITTING DETECTION: {model_name}")
        print(f"{'='*60}")
        print("METRICS:")
        print(f"  Train:      MSE={train_metrics['mse']:.6f}  R2={train_metrics['r2']:.4f}  MAE={train_metrics['mae']:.6f}")
        print(f"  Validation: MSE={val_metrics['mse']:.6f}  R2={val_metrics['r2']:.4f}  MAE={val_metrics['mae']:.6f}")
        print(f"  Test:       MSE={test_metrics['mse']:.6f}  R2={test_metrics['r2']:.4f}  MAE={test_metrics['mae']:.6f}")
        print(f"{'-'*60}")
        print("DEGRADATION ANALYSIS:")
        print(f"  MSE:         Train->Val={train_val_mse_ratio*100:>6.2f}%  Val->Test={val_test_mse_ratio*100:>6.2f}%  Score={mse_score:.4f}")
        print(f"  R2:          Train->Val={train_val_r2_ratio*100:>6.2f}%  Val->Test={val_test_r2_ratio*100:>6.2f}%  Score={r2_degradation_score:.4f}")
        print(f"  MAE:         Train->Val={train_val_mae_ratio*100:>6.2f}%  Val->Test={val_test_mae_ratio*100:>6.2f}%  Score={mae_score:.4f}")
        print(f"  Consistency: {consistency_score:.4f}")
        print(f"{'-'*60}")
        print("FINAL ASSESSMENT:")
        print(f"  Combined overfitting score: {overfitting_score:.4f}")
        print(f"  Threshold:                  {threshold:.4f}")
        print(f"{'-'*60}")
        
        if is_overfitted:
            print(f"[WARN] OVERFITTING DETECTED! (score: {overfitting_score:.4f} > threshold: {threshold:.4f})")
        else:
            print(f"[OK] No overfitting detected (score: {overfitting_score:.4f} <= threshold: {threshold:.4f})")
    
    else:
        # ===== LEGACY SINGLE-METRIC DETECTION (MSE only) =====
        train_val_mse_ratio = (val_metrics['mse'] - train_metrics['mse']) / train_metrics['mse']
        val_test_mse_ratio = (test_metrics['mse'] - val_metrics['mse']) / val_metrics['mse']
        overfitting_score = max(train_val_mse_ratio, val_test_mse_ratio)
        details.update({
            'train_val_mse_ratio': train_val_mse_ratio,
            'val_test_mse_ratio': val_test_mse_ratio,
        })
        is_overfitted = overfitting_score >= threshold
        
        print(f"\n{'='*60}")
        print(f"[DETECT] OVERFITTING DETECTION: {model_name}")
        print(f"{'='*60}")
        print(f"Train MSE:      {train_metrics['mse']:.6f}  |  R2: {train_metrics['r2']:.4f}")
        print(f"Validation MSE: {val_metrics['mse']:.6f}  |  R2: {val_metrics['r2']:.4f}")
        print(f"Test MSE:       {test_metrics['mse']:.6f}  |  R2: {test_metrics['r2']:.4f}")
        print(f"{'-'*60}")
        print(f"Train -> Val degradation: {train_val_mse_ratio*100:.2f}%")
        print(f"Val -> Test degradation:  {val_test_mse_ratio*100:.2f}%")
        print(f"Overfitting score:       {overfitting_score:.4f}")
        print(f"Threshold:               {threshold:.4f}")
        print(f"{'-'*60}")
        
        if is_overfitted:
            print(f"[WARN] OVERFITTING DETECTED! (score: {overfitting_score:.4f} > threshold: {threshold:.4f})")
        else:
            print(f"[OK] No overfitting detected (score: {overfitting_score:.4f} <= threshold: {threshold:.4f})")

    print(f"{'='*60}\n")
    return is_overfitted, overfitting_score, details

def check_data_health(x_train, x_val, x_test, y_train, y_val, y_test, model_name, stock_symbol=None):
    """
    Diagnostic checks for data quality issues that may cause overfitting.
    
    Parameters:
    - x_train, x_val, x_test: Feature arrays
    - y_train, y_val, y_test: Target arrays
    - model_name: Name for logging
    - stock_symbol: Optional ticker symbol for context in output
    
    Returns:
    - dict with diagnostic results and warnings
    """
    diagnostics = {
        'warnings': [],
        'recommendations': [],
        'pass_diagnostic': True
    }
    
    ticker_label = f" ({stock_symbol})" if stock_symbol else ""
    print(f"\n{'='*60}")
    print(f"[DIAG] DATA HEALTH CHECK: {model_name}{ticker_label}")
    print(f"{'='*60}")
    
    # Check 1: Sample sizes
    train_size = len(x_train)
    val_size = len(x_val)
    test_size = len(x_test)
    feature_count = x_train.shape[1] if len(x_train.shape) > 1 else 1
    
    print("[DATA] Dataset Sizes:")
    print(f"   Train: {train_size} samples")
    print(f"   Val:   {val_size} samples")
    print(f"   Test:  {test_size} samples")
    print(f"   Features: {feature_count}")
    
    # Check 2: Feature-to-sample ratio
    samples_per_feature = train_size / max(feature_count, 1)
    if samples_per_feature < 10:
        diagnostics['warnings'].append(f"Very few samples per feature ({samples_per_feature:.1f})")
        diagnostics['recommendations'].append("Consider dimensionality reduction or getting more data")
        diagnostics['pass_diagnostic'] = False
        print(f"   [WARN] Warning: Only {samples_per_feature:.1f} samples per feature (recommend >10)")
    else:
        print(f"   [OK] Samples per feature: {samples_per_feature:.1f}")
    
    # Check 3: Dataset size balance
    val_ratio = val_size / train_size
    # test_ratio = test_size / train_size
    if val_ratio < 0.1 or val_ratio > 0.4:
        diagnostics['warnings'].append(f"Validation set size unusual ({val_ratio*100:.1f}% of train)")
        print(f"   [WARN] Warning: Val/Train ratio {val_ratio*100:.1f}% (recommend 15-25%)")
    else:
        print(f"   [OK] Val/Train ratio: {val_ratio*100:.1f}%")
    
    # Check 4: Target variance
    y_train_var = np.var(y_train)
    y_val_var = np.var(y_val)
    y_test_var = np.var(y_test)
    
    variance_ratio = max(y_train_var, y_val_var, y_test_var) / (min(y_train_var, y_val_var, y_test_var) + 1e-10)
    
    print("\n[VAR] Target Variance:")
    print(f"   Train: {y_train_var:.6f}")
    print(f"   Val:   {y_val_var:.6f}")
    print(f"   Test:  {y_test_var:.6f}")
    
    if variance_ratio > 10:
        diagnostics['warnings'].append(f"High variance mismatch ({variance_ratio:.1f}x)")
        diagnostics['recommendations'].append("Data splits may not be representative - consider reshuffling")
        print(f"   [WARN] Warning: {variance_ratio:.1f}x variance difference (may indicate distribution shift)")
    else:
        print(f"   [OK] Variance ratio: {variance_ratio:.1f}x")
    
    # Check 5: Check for extremely small values that might indicate scaling issues
    y_train_mean = np.mean(np.abs(y_train))
    if y_train_mean < 1e-6:
        diagnostics['warnings'].append("Target values extremely small (potential scaling issue)")
        print(f"   [WARN] Warning: Target mean {y_train_mean:.2e} very small")
    
    print(f"{'='*60}")
    
    if diagnostics['warnings']:
        print(f"\n[WARN] {len(diagnostics['warnings'])} warning(s) detected:")
        for warning in diagnostics['warnings']:
            print(f"   - {warning}")
        if diagnostics['recommendations']:
            print("\n[TIPS] Recommendations:")
            for rec in diagnostics['recommendations']:
                print(f"   - {rec}")
    else:
        print("\n[OK] All diagnostic checks passed")
    
    print(f"{'='*60}\n")
    
    return diagnostics

def are_hyperparameters_identical(hp1, hp2, tolerance=0.01):
    """
    Check if two sets of hyperparameters are essentially identical.
    
    Parameters:
    - hp1, hp2: Hyperparameter dictionaries
    - tolerance: Relative tolerance for float comparisons
    
    Returns:
    - bool: True if hyperparameters are identical
    """
    if hp1.keys() != hp2.keys():
        return False
    
    for key in hp1.keys():
        val1, val2 = hp1[key], hp2[key]
        
        # Skip tuner-specific keys
        if key.startswith('tuner/'):
            continue
        
        # Compare numeric values with tolerance
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            if abs(val1 - val2) > tolerance * max(abs(val1), abs(val2), 1e-12):
                return False
        # Compare non-numeric values exactly
        elif val1 != val2:
            return False
    
    return True

def train_and_validate_models(stock_symbol, x_train, x_val, x_test, y_train_scaled, y_val_scaled, y_test_scaled, y_train_unscaled, y_val_unscaled, y_test_unscaled, time_steps, scaler_y=None, max_retrains=10, overfitting_threshold=0.15, lstm_trials=25, lstm_executions=1, lstm_epochs=50, rf_trials=50, xgb_trials=30, rf_retrain_increment=25, xgb_retrain_increment=10, lstm_retrain_trials_increment=10, lstm_retrain_executions_increment=2, use_multi_metric_detection=True, use_tcn=True, use_sequence_model=True, tcn_trials=30, tcn_epochs=100, tcn_retrain_increment=10, cache_max_age_days=30, cache_only=False):
    """
    Train and validate models with automatic retraining if overfitting is detected.
    Includes LSTM/TCN, Random Forest, XGBoost, and ensemble predictions.
    
    Parameters:
    - stock_symbol (str): Stock ticker
    - x_train, x_val, x_test: Feature arrays
    - y_train_scaled, y_val_scaled, y_test_scaled: Scaled target arrays
    - y_train_unscaled, y_val_unscaled, y_test_unscaled: Unscaled target arrays
    - time_steps (int): Number of time steps for LSTM/TCN sequences
    - scaler_y (MinMaxScaler, optional): Scaler for y values, used to compute unscaled TCN metrics for ensemble weighting
    - max_retrains (int): Maximum retraining attempts
    - overfitting_threshold (float): Overfitting detection threshold
    - lstm_trials (int): Max trials for LSTM tuning
    - lstm_executions (int): Executions per trial for LSTM
    - lstm_epochs (int): Training epochs for LSTM
    - rf_trials (int): Max trials for Random Forest tuning
    - xgb_trials (int): Max trials for XGBoost tuning
    - rf_retrain_increment (int): Trials to add when retraining RF
    - xgb_retrain_increment (int): Trials to add when retraining XGBoost
    - lstm_retrain_trials_increment (int): Trials to add when retraining LSTM
    - lstm_retrain_executions_increment (int): Executions to add when retraining LSTM
    - use_multi_metric_detection (bool): Whether to use multi-metric overfitting detection
    - use_tcn (bool): Whether to use TCN instead of LSTM (recommended)
    - tcn_trials (int): Max trials for TCN tuning
    - tcn_epochs (int): Training epochs for TCN
    - tcn_retrain_increment (int): Trials to add when retraining TCN
    - cache_max_age_days (int): Maximum age for reusable cached hyperparameters
    - cache_only (bool): If True, require valid cached hyperparameters and never retune
    - lstm_retrain_executions_increment (int): Executions to add when retraining LSTM
    - use_multi_metric_detection (bool): Whether to use multi-metric overfitting detection
    
    Returns:
    - models (dict): Dictionary containing all trained models and ensemble weights
    - training_history: Dict with all metrics and decisions
    - lstm_datasets: Prepared LSTM/TCN datasets for later use
    """

    training_history: dict[str, Any] = {
        'lstm': [],
        'tcn': [],
        'random_forest': [],
        'xgboost': [],
        'ensemble': None,
        'final_decision': None,
        'diagnostics': {},
        'early_stopping_triggered': {},
        'cache_only': cache_only,
        'sequence_model_type': ('tcn' if use_tcn else 'lstm') if use_sequence_model else 'none'
    }

    if cache_only:
        max_retrains = 1
    
    # Run diagnostic checks before training
    print("\n" + "="*60)
    print("[DIAG] RUNNING PRE-TRAINING DIAGNOSTICS")
    print("="*60)
    
    diagnostics_rf = check_data_health(
        x_train, x_val, x_test,
        y_train_unscaled, y_val_unscaled, y_test_unscaled,
        "Random Forest / XGBoost",
        stock_symbol=stock_symbol
    )
    training_history['diagnostics']['rf_xgb'] = diagnostics_rf
    
    diagnostics_lstm = check_data_health(
        x_train, x_val, x_test,
        y_train_scaled, y_val_scaled, y_test_scaled,
        "TCN/LSTM",
        stock_symbol=stock_symbol
    )
    training_history['diagnostics']['lstm'] = diagnostics_lstm

    # Convert to DataFrames for Random Forest
    x_train_df = pd.DataFrame(x_train)
    x_val_df = pd.DataFrame(x_val)
    x_test_df = pd.DataFrame(x_test)

    y_train_unscaled_series = pd.Series(y_train_unscaled)  # UNSCALED for Random Forest
    y_val_unscaled_series = pd.Series(y_val_unscaled)        # UNSCALED for Random Forest
    y_test_unscaled_series = pd.Series(y_test_unscaled)      # UNSCALED for Random Forest

    # Prepare sequence datasets ONCE (used by both TCN and LSTM)
    lstm_datasets = None
    if use_sequence_model:
        seq_label = "TCN" if use_tcn else "LSTM"
        print("\n" + "="*60)
        print(f"[DATA] PREPARING {seq_label} SEQUENCE DATASETS")
        print("="*60)

        lstm_datasets = prepare_lstm_datasets(
            x_train, y_train_scaled,
            x_val, y_val_scaled,
            x_test, y_test_scaled,
            time_steps
        )

        print(f"[OK] {seq_label} sequence datasets prepared:")
        print(f"   - Training samples: {lstm_datasets['metadata']['train_samples']}")
        print(f"   - Validation samples: {lstm_datasets['metadata']['val_samples']}")
        print(f"   - Test samples: {lstm_datasets['metadata']['test_samples']}")
        print(f"   - Time steps: {lstm_datasets['metadata']['time_steps']}")
        print(f"   - Features: {lstm_datasets['metadata']['num_features']}")
    else:
        print("\n" + "="*60)
        print("[INFO] SEQUENCE MODEL DISABLED — Training RF + XGB + Ridge + SVR only")
        print("="*60)

    lstm_model = None
    tcn_model = None
    rf_model = None
    xgb_model = None
    lstm_overfitted = False
    tcn_overfitted = False
    rf_overfitted = False
    xgb_overfitted = False
    sequence_model = None  # Will hold either LSTM or TCN model

    # ===== SEQUENCE MODEL TRAINING (TCN or LSTM) =====
    if use_sequence_model and use_tcn:
        # ===== TCN TRAINING LOOP =====
        print("\n" + "="*60)
        print("[TCN] STARTING TCN MODEL TRAINING")
        print("="*60)
        print("[INFO] TCN (Temporal Convolutional Network) selected as sequence model")
        print("[INFO] TCN advantages: parallelizable, better gradient flow, less prone to mode collapse")

        tcn_current_trials = tcn_trials
        # Cap sequence model retrains — each retrain does a full Keras Tuner search
        # (30+ trials x 100 epochs each), so 5 retrains is already 15,000+ training runs.
        tcn_max_retrains = min(max_retrains, 5)

        for tcn_attempt in range(tcn_max_retrains):
            print(f"\n[DATA] TCN Training Attempt {tcn_attempt + 1}/{tcn_max_retrains}")

            tcn_model = tune_tcn_model(
                stock_symbol,
                lstm_datasets['train']['x'],
                lstm_datasets['train']['y'],
                lstm_datasets['val']['x'],
                lstm_datasets['val']['y'],
                lstm_datasets['metadata']['time_steps'],
                lstm_datasets['metadata']['num_features'],
                max_trials=tcn_current_trials,
                epochs=tcn_epochs,
                use_cached_hp=(tcn_attempt == 0),
                cache_max_age_days=cache_max_age_days,
                require_cached_hp=cache_only,
            )

            # Evaluate TCN
            train_metrics, val_metrics, test_metrics = evaluate_tcn_model(
                tcn_model,
                lstm_datasets['train']['x'], lstm_datasets['train']['y'],
                lstm_datasets['val']['x'], lstm_datasets['val']['y'],
                lstm_datasets['test']['x'], lstm_datasets['test']['y']
            )

            # Store history
            training_history['tcn'].append({
                'attempt': tcn_attempt + 1,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics
            })

            # Detect overfitting
            tcn_overfitted, tcn_overfitting_score, _ = detect_overfitting(
                train_metrics, val_metrics, test_metrics, "TCN", overfitting_threshold, use_multi_metric_detection
            )

            if not tcn_overfitted:
                print(f"[OK] TCN model accepted after {tcn_attempt + 1} attempt(s)")
                break
            elif tcn_overfitting_score > 1.0:
                # Score is far beyond recoverable — model fundamentally cannot generalize
                # on this data. More hyperparameter search won't fix it.
                print(f"[WARN] TCN overfitting score ({tcn_overfitting_score:.2f}) is extreme (>{1.0}) — model cannot generalize.")
                print("   Accepting current best model and continuing to next model type.")
                break
            elif tcn_attempt < tcn_max_retrains - 1:
                if tcn_attempt == 0:
                    try:
                        from db_interactions import invalidate_hyperparameters
                        invalidate_hyperparameters(ticker=stock_symbol, model_type='tcn')
                        print("[CACHE] Invalidated bad TCN cached hyperparameters")
                    except ImportError:
                        pass
                print("[WARN] Retraining TCN with adjusted hyperparameters...")
                print(f"   Increasing trials: {tcn_current_trials} -> {tcn_current_trials + tcn_retrain_increment}")
                tcn_current_trials += tcn_retrain_increment
            else:
                print("[WARN] TCN reached maximum retrain attempts. Accepting current model.")

        sequence_model = tcn_model
        sequence_model_overfitted = tcn_overfitted
        sequence_model_name = "TCN"
    elif use_sequence_model and not use_tcn:
        # ===== LSTM TRAINING LOOP =====
        print("\n" + "="*60)
        print("[LSTM] STARTING LSTM MODEL TRAINING")
        print("="*60)

        # Cap sequence model retrains — each retrain does a full Keras Tuner search
        lstm_max_retrains = min(max_retrains, 5)

        for lstm_attempt in range(lstm_max_retrains):
            print(f"\n[DATA] LSTM Training Attempt {lstm_attempt + 1}/{lstm_max_retrains}")

            lstm_model = tune_lstm_model(
                stock_symbol,
                lstm_datasets['train']['x'],
                lstm_datasets['train']['y'],
                lstm_datasets['val']['x'],
                lstm_datasets['val']['y'],
                lstm_datasets['metadata']['time_steps'],
                lstm_datasets['metadata']['num_features'],
                max_trials=lstm_trials,
                executions_per_trial=lstm_executions,
                epochs=lstm_epochs,
                use_cached_hp=(lstm_attempt == 0),
                cache_max_age_days=cache_max_age_days,
                require_cached_hp=cache_only,
            )

            # Evaluate LSTM
            train_metrics, val_metrics, test_metrics = evaluate_lstm_model(
                lstm_model,
                lstm_datasets['train']['x'], lstm_datasets['train']['y'],
                lstm_datasets['val']['x'], lstm_datasets['val']['y'],
                lstm_datasets['test']['x'], lstm_datasets['test']['y']
            )

            # Store history
            training_history['lstm'].append({
                'attempt': lstm_attempt + 1,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics
            })

            # Detect overfitting
            lstm_overfitted, lstm_overfitting_score, _ = detect_overfitting(
                train_metrics, val_metrics, test_metrics, "LSTM", overfitting_threshold, use_multi_metric_detection
            )

            if not lstm_overfitted:
                print(f"[OK] LSTM model accepted after {lstm_attempt + 1} attempt(s)")
                break
            elif lstm_overfitting_score > 1.0:
                # Score is far beyond recoverable — model fundamentally cannot generalize
                # on this data. More hyperparameter search won't fix it.
                print(f"[WARN] LSTM overfitting score ({lstm_overfitting_score:.2f}) is extreme (>{1.0}) — model cannot generalize.")
                print("   Accepting current best model and continuing to next model type.")
                break
            elif lstm_attempt < lstm_max_retrains - 1:
                if lstm_attempt == 0:
                    try:
                        from db_interactions import invalidate_hyperparameters
                        invalidate_hyperparameters(ticker=stock_symbol, model_type='lstm')
                        print("[CACHE] Invalidated bad LSTM cached hyperparameters")
                    except ImportError:
                        pass
                print("[WARN] Retraining LSTM with adjusted hyperparameters...")
                print(f"   Increasing trials: {lstm_trials} -> {lstm_trials + lstm_retrain_trials_increment}")
                print(f"   Increasing executions: {lstm_executions} -> {lstm_executions + lstm_retrain_executions_increment}")
                lstm_trials += lstm_retrain_trials_increment
                lstm_executions += lstm_retrain_executions_increment
            else:
                print("[WARN] LSTM reached maximum retrain attempts. Accepting current model.")

        sequence_model = lstm_model
        sequence_model_overfitted = lstm_overfitted
        sequence_model_name = "LSTM"
    else:
        # Sequence model disabled — 4-model ensemble only
        sequence_model = None
        sequence_model_overfitted = False
        sequence_model_name = "none"

    # ===== RANDOM FOREST TRAINING LOOP =====
    print("\n" + "="*60)
    print("[RF] STARTING RANDOM FOREST MODEL TRAINING")
    print("="*60)
    
    rf_previous_hyperparams = None
    rf_identical_count = 0
    rf_best_score = float('inf')
    rf_search_space_constrained = False

    for rf_attempt in range(max_retrains):
        print(f"\n[DATA] Random Forest Training Attempt {rf_attempt + 1}/{max_retrains}")
        
        # Constrain search space if overfitting detected in previous attempts
        if rf_attempt > 0 and rf_overfitted and not rf_search_space_constrained:
            print("\n[TUNE] APPLYING SEARCH SPACE CONSTRAINTS (overfitting detected)")
            print("   - Reducing max_depth ceiling: 50 -> 30")
            print("   - Increasing min_samples_leaf floor: 1 -> 2")
            print("   - Forcing bootstrap=True for better generalization")
            rf_search_space_constrained = True

        rf_model = tune_random_forest_model(
            stock_symbol,
            x_train_df,
            y_train_unscaled_series,
            x_val_df,
            y_val_unscaled_series,
            max_trials=rf_trials,
            constrain_for_overfitting=rf_search_space_constrained,
            use_cached_hp=(rf_attempt == 0),
            cache_max_age_days=cache_max_age_days,
            require_cached_hp=cache_only,
        )
        
        # Extract hyperparameters for comparison
        rf_current_hyperparams = rf_model.get_params()

        # Evaluate Random Forest
        train_metrics, val_metrics, test_metrics = evaluate_random_forest_model(
            rf_model, x_train_df, y_train_unscaled_series,
            x_val_df, y_val_unscaled_series,
            x_test_df, y_test_unscaled_series
        )

        # Store history
        training_history['random_forest'].append({
            'attempt': rf_attempt + 1,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics
        })

        # Detect overfitting
        rf_overfitted, rf_overfitting_score, _ = detect_overfitting(
            train_metrics, val_metrics, test_metrics, "Random Forest", overfitting_threshold, use_multi_metric_detection
        )
        
        # Track best score
        if rf_overfitting_score < rf_best_score:
            rf_best_score = rf_overfitting_score
        
        # Early stopping: Check if hyperparameters are identical
        if rf_previous_hyperparams is not None:
            if are_hyperparameters_identical(rf_current_hyperparams, rf_previous_hyperparams):
                rf_identical_count += 1
                print(f"\n[WARN] WARNING: Identical hyperparameters found ({rf_identical_count} consecutive)")
                
                if rf_identical_count >= 3:
                    print("\n[STOP] EARLY STOPPING TRIGGERED: Random Forest")
                    print("   Reason: Hyperparameter search converged to same solution 3 times")
                    print("   This indicates the model cannot improve further with current data")
                    print("   Recommendations:")
                    print("     - Collect more training data")
                    print("     - Improve feature engineering")
                    print("     - Consider simpler model architecture")
                    print("   Accepting current model as final.")
                    training_history['early_stopping_triggered']['random_forest'] = {
                        'attempt': rf_attempt + 1,
                        'reason': 'identical_hyperparameters',
                        'count': rf_identical_count
                    }
                    break
            else:
                rf_identical_count = 0  # Reset counter
        
        rf_previous_hyperparams = rf_current_hyperparams.copy()

        if not rf_overfitted:
            print(f"[OK] Random Forest model accepted after {rf_attempt + 1} attempt(s)")
            break
        elif rf_attempt < max_retrains - 1:
            if rf_attempt == 0:
                try:
                    from db_interactions import invalidate_hyperparameters
                    invalidate_hyperparameters(ticker=stock_symbol, model_type='rf')
                    print("[CACHE] Invalidated bad RF cached hyperparameters")
                except ImportError:
                    pass
            print("[WARN] Retraining Random Forest with adjusted hyperparameters...")
            if not rf_search_space_constrained:
                print(f"   Strategy: Increasing trials: {rf_trials} -> {rf_trials + rf_retrain_increment}")
                rf_trials += rf_retrain_increment
            else:
                print(f"   Strategy: Using constrained search space with {rf_trials} trials")
        else:
            print("[WARN] Random Forest reached maximum retrain attempts. Accepting current model.")

    # ===== XGBOOST TRAINING LOOP =====
    print("\n" + "="*60)
    print("[XGB] STARTING XGBOOST MODEL TRAINING")
    print("="*60)
    
    xgb_previous_hyperparams = None
    xgb_identical_count = 0
    xgb_best_score = float('inf')
    xgb_search_space_constrained = False

    for xgb_attempt in range(max_retrains):
        print(f"\n[DATA] XGBoost Training Attempt {xgb_attempt + 1}/{max_retrains}")
        
        # Constrain search space if overfitting detected in previous attempts
        if xgb_attempt > 0 and xgb_overfitted and not xgb_search_space_constrained:
            print("\n[TUNE] APPLYING SEARCH SPACE CONSTRAINTS (overfitting detected)")
            print("   - Reducing max_depth ceiling: 15 -> 10")
            print("   - Increasing min_child_weight floor: 1 -> 3")
            print("   - Strengthening regularization (alpha, lambda)")
            print("   - Narrowing subsample/colsample ranges")
            xgb_search_space_constrained = True

        xgb_model = tune_xgboost_model(
            stock_symbol,
            x_train_df,
            y_train_unscaled_series,
            x_val_df,
            y_val_unscaled_series,
            max_trials=xgb_trials,
            constrain_for_overfitting=xgb_search_space_constrained,
            use_cached_hp=(xgb_attempt == 0),
            cache_max_age_days=cache_max_age_days,
            require_cached_hp=cache_only,
        )
        
        # Extract hyperparameters for comparison
        xgb_current_hyperparams = xgb_model.get_params()

        # Evaluate XGBoost
        train_metrics, val_metrics, test_metrics = evaluate_xgboost_model(
            xgb_model, x_train_df, y_train_unscaled_series,
            x_val_df, y_val_unscaled_series,
            x_test_df, y_test_unscaled_series
        )

        # Store history
        training_history['xgboost'].append({
            'attempt': xgb_attempt + 1,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics
        })

        # Detect overfitting
        xgb_overfitted, xgb_overfitting_score, _ = detect_overfitting(
            train_metrics, val_metrics, test_metrics, "XGBoost", overfitting_threshold, use_multi_metric_detection
        )
        
        # Track best score
        if xgb_overfitting_score < xgb_best_score:
            xgb_best_score = xgb_overfitting_score
        
        # Early stopping: Check if hyperparameters are identical
        if xgb_previous_hyperparams is not None:
            if are_hyperparameters_identical(xgb_current_hyperparams, xgb_previous_hyperparams):
                xgb_identical_count += 1
                print(f"\n[WARN] WARNING: Identical hyperparameters found ({xgb_identical_count} consecutive)")
                
                if xgb_identical_count >= 3:
                    print("\n[STOP] EARLY STOPPING TRIGGERED: XGBoost")
                    print("   Reason: Hyperparameter search converged to same solution 3 times")
                    print("   This indicates the model cannot improve further with current data")
                    print("   Recommendations:")
                    print("     - Collect more training data")
                    print("     - Improve feature engineering")
                    print("     - Consider simpler model architecture")
                    print("   Accepting current model as final.")
                    training_history['early_stopping_triggered']['xgboost'] = {
                        'attempt': xgb_attempt + 1,
                        'reason': 'identical_hyperparameters',
                        'count': xgb_identical_count
                    }
                    break
            else:
                xgb_identical_count = 0  # Reset counter
        
        xgb_previous_hyperparams = xgb_current_hyperparams.copy()

        if not xgb_overfitted:
            print(f"[OK] XGBoost model accepted after {xgb_attempt + 1} attempt(s)")
            break
        elif xgb_attempt < max_retrains - 1:
            if xgb_attempt == 0:
                try:
                    from db_interactions import invalidate_hyperparameters
                    invalidate_hyperparameters(ticker=stock_symbol, model_type='xgb')
                    print("[CACHE] Invalidated bad XGBoost cached hyperparameters")
                except ImportError:
                    pass
            print("[WARN] Retraining XGBoost with adjusted hyperparameters...")
            if not xgb_search_space_constrained:
                print(f"   Strategy: Increasing trials: {xgb_trials} -> {xgb_trials + xgb_retrain_increment}")
                xgb_trials += xgb_retrain_increment
            else:
                print(f"   Strategy: Using constrained search space with {xgb_trials} trials")
        else:
            print("[WARN] XGBoost reached maximum retrain attempts. Accepting current model.")

    # ===== RIDGE REGRESSION TRAINING =====
    print("\n" + "="*60)
    print(f"[RIDGE] TRAINING RIDGE REGRESSION for {stock_symbol}")
    print("="*60)
    ridge_model = tune_ridge_model(
        stock_symbol, x_train_df, y_train_unscaled_series,
        x_val_df, y_val_unscaled_series,
        max_trials=50,
        constrain_for_overfitting=False,
        use_cached_hp=True, cache_max_age_days=cache_max_age_days,
        cleanup_after_tuning=True,
        require_cached_hp=cache_only,
    )
    ridge_train_metrics, ridge_val_metrics, ridge_test_metrics = evaluate_ridge_model(
        ridge_model, x_train_df.values, y_train_unscaled_series.values,
        x_val_df.values, y_val_unscaled_series.values,
        x_test_df.values, y_test_unscaled_series.values
    )
    training_history['ridge'] = [{
        'train_metrics': ridge_train_metrics,
        'val_metrics': ridge_val_metrics,
        'test_metrics': ridge_test_metrics
    }]
    print(f"[RIDGE] Val MSE: {ridge_val_metrics['mse']:.6f}, R2: {ridge_val_metrics['r2']:.4f}")

    # ===== SVR TRAINING =====
    print("\n" + "="*60)
    print(f"[SVR] TRAINING SVR for {stock_symbol}")
    print("="*60)
    svr_model = tune_svr_model(
        stock_symbol, x_train_df, y_train_unscaled_series,
        x_val_df, y_val_unscaled_series,
        max_trials=40,
        constrain_for_overfitting=False,
        use_cached_hp=True, cache_max_age_days=cache_max_age_days,
        cleanup_after_tuning=True,
        require_cached_hp=cache_only,
    )
    svr_train_metrics, svr_val_metrics, svr_test_metrics = evaluate_svr_model(
        svr_model, x_train_df.values, y_train_unscaled_series.values,
        x_val_df.values, y_val_unscaled_series.values,
        x_test_df.values, y_test_unscaled_series.values
    )
    training_history['svr'] = [{
        'train_metrics': svr_train_metrics,
        'val_metrics': svr_val_metrics,
        'test_metrics': svr_test_metrics
    }]
    print(f"[SVR] Val MSE: {svr_val_metrics['mse']:.6f}, R2: {svr_val_metrics['r2']:.4f}")

    # ===== ENSEMBLE EVALUATION =====
    print("\n" + "="*60)
    print("[ENSEMBLE] EVALUATING ENSEMBLE PREDICTIONS")
    print("="*60)

    # RF and XGBoost predictions (full length, already in UNSCALED space)
    rf_train_pred_full = rf_model.predict(x_train_df.values)
    rf_val_pred_full = rf_model.predict(x_val_df.values)
    rf_test_pred_full = rf_model.predict(x_test_df.values)

    xgb_train_pred_full = xgb_model.predict(x_train_df)
    xgb_val_pred_full = xgb_model.predict(x_val_df)
    xgb_test_pred_full = xgb_model.predict(x_test_df)

    # Ridge and SVR predictions (full length, already in UNSCALED space)
    ridge_train_pred_full = ridge_model.predict(x_train_df.values)
    ridge_val_pred_full = ridge_model.predict(x_val_df.values)
    ridge_test_pred_full = ridge_model.predict(x_test_df.values)

    svr_train_pred_full = svr_model.predict(x_train_df.values)
    svr_val_pred_full = svr_model.predict(x_val_df.values)
    svr_test_pred_full = svr_model.predict(x_test_df.values)

    if use_sequence_model:
        # Get predictions from sequence model (TCN or LSTM) - these are in SCALED space
        seq_train_pred_scaled = sequence_model.predict(lstm_datasets['train']['x'], verbose=0).flatten()
        seq_val_pred_scaled = sequence_model.predict(lstm_datasets['val']['x'], verbose=0).flatten()
        seq_test_pred_scaled = sequence_model.predict(lstm_datasets['test']['x'], verbose=0).flatten()

        # Inverse-transform sequence model predictions to UNSCALED space
        if scaler_y is not None:
            seq_train_pred = scaler_y.inverse_transform(seq_train_pred_scaled.reshape(-1, 1)).flatten()
            seq_val_pred = scaler_y.inverse_transform(seq_val_pred_scaled.reshape(-1, 1)).flatten()
            seq_test_pred = scaler_y.inverse_transform(seq_test_pred_scaled.reshape(-1, 1)).flatten()
        else:
            print("[WARN] scaler_y not provided — ensemble weights may be inaccurate")
            seq_train_pred = seq_train_pred_scaled
            seq_val_pred = seq_val_pred_scaled
            seq_test_pred = seq_test_pred_scaled

        train_alignment_start = _alignment_start_from_sequence_predictions(
            y_train_unscaled_series.values,
            seq_train_pred,
            'train',
        )
        val_alignment_start = _alignment_start_from_sequence_predictions(
            y_val_unscaled_series.values,
            seq_val_pred,
            'validation',
        )
        test_alignment_start = _alignment_start_from_sequence_predictions(
            y_test_unscaled_series.values,
            seq_test_pred,
            'test',
        )

        rf_train_pred = rf_train_pred_full[train_alignment_start:]
        rf_val_pred = rf_val_pred_full[val_alignment_start:]
        rf_test_pred = rf_test_pred_full[test_alignment_start:]
        xgb_train_pred = xgb_train_pred_full[train_alignment_start:]
        xgb_val_pred = xgb_val_pred_full[val_alignment_start:]
        xgb_test_pred = xgb_test_pred_full[test_alignment_start:]
        ridge_train_pred = ridge_train_pred_full[train_alignment_start:]
        ridge_val_pred = ridge_val_pred_full[val_alignment_start:]
        ridge_test_pred = ridge_test_pred_full[test_alignment_start:]
        svr_train_pred = svr_train_pred_full[train_alignment_start:]
        svr_val_pred = svr_val_pred_full[val_alignment_start:]
        svr_test_pred = svr_test_pred_full[test_alignment_start:]

        y_train_aligned = y_train_unscaled_series.iloc[train_alignment_start:].values
        y_val_aligned = y_val_unscaled_series.iloc[val_alignment_start:].values
        y_test_aligned = y_test_unscaled_series.iloc[test_alignment_start:].values

        # Calculate ensemble weights using UNSCALED validation MSE for all 5 models
        seq_model_key = 'tcn' if use_tcn else 'lstm'
        seq_val_mse = mean_squared_error(y_val_aligned, seq_val_pred)
        rf_val_mse = mean_squared_error(y_val_aligned, rf_val_pred)
        xgb_val_mse = mean_squared_error(y_val_aligned, xgb_val_pred)
        ridge_val_mse = mean_squared_error(y_val_aligned, ridge_val_pred)
        svr_val_mse = mean_squared_error(y_val_aligned, svr_val_pred)

        print("\n[DATA] Validation MSE (all in unscaled return space):")
        print(f"   - {sequence_model_name}: {seq_val_mse:.8f}")
        print(f"   - Random Forest:        {rf_val_mse:.8f}")
        print(f"   - XGBoost:              {xgb_val_mse:.8f}")
        print(f"   - Ridge:                {ridge_val_mse:.8f}")
        print(f"   - SVR:                  {svr_val_mse:.8f}")

        inv_mse_sum = ((1/seq_val_mse) + (1/rf_val_mse) + (1/xgb_val_mse) +
                       (1/ridge_val_mse) + (1/svr_val_mse))
        weights = {
            'seq': (1/seq_val_mse) / inv_mse_sum,
            'rf': (1/rf_val_mse) / inv_mse_sum,
            'xgb': (1/xgb_val_mse) / inv_mse_sum,
            'ridge': (1/ridge_val_mse) / inv_mse_sum,
            'svr': (1/svr_val_mse) / inv_mse_sum
        }
    else:
        # No sequence model — use full predictions (no alignment trimming needed)
        rf_train_pred = rf_train_pred_full
        rf_val_pred = rf_val_pred_full
        rf_test_pred = rf_test_pred_full
        xgb_train_pred = xgb_train_pred_full
        xgb_val_pred = xgb_val_pred_full
        xgb_test_pred = xgb_test_pred_full
        ridge_train_pred = ridge_train_pred_full
        ridge_val_pred = ridge_val_pred_full
        ridge_test_pred = ridge_test_pred_full
        svr_train_pred = svr_train_pred_full
        svr_val_pred = svr_val_pred_full
        svr_test_pred = svr_test_pred_full

        y_train_aligned = y_train_unscaled_series.values
        y_val_aligned = y_val_unscaled_series.values
        y_test_aligned = y_test_unscaled_series.values

        seq_model_key = 'none'
        rf_val_mse = mean_squared_error(y_val_aligned, rf_val_pred)
        xgb_val_mse = mean_squared_error(y_val_aligned, xgb_val_pred)
        ridge_val_mse = mean_squared_error(y_val_aligned, ridge_val_pred)
        svr_val_mse = mean_squared_error(y_val_aligned, svr_val_pred)

        print("\n[DATA] Validation MSE (all in unscaled return space):")
        print(f"   - Random Forest: {rf_val_mse:.8f}")
        print(f"   - XGBoost:       {xgb_val_mse:.8f}")
        print(f"   - Ridge:         {ridge_val_mse:.8f}")
        print(f"   - SVR:           {svr_val_mse:.8f}")

        inv_mse_sum = (1/rf_val_mse) + (1/xgb_val_mse) + (1/ridge_val_mse) + (1/svr_val_mse)
        weights = {
            'seq': 0.0,
            'rf': (1/rf_val_mse) / inv_mse_sum,
            'xgb': (1/xgb_val_mse) / inv_mse_sum,
            'ridge': (1/ridge_val_mse) / inv_mse_sum,
            'svr': (1/svr_val_mse) / inv_mse_sum
        }

    # CRITICAL: Zero out negligible weights to prevent extreme-output models from contaminating ensemble.
    MIN_ENSEMBLE_WEIGHT = 0.005
    zeroed_models = []
    for name, w in weights.items():
        if w < MIN_ENSEMBLE_WEIGHT:
            weights[name] = 0.0
            zeroed_models.append(name)
    if zeroed_models:
        remaining_sum = sum(weights.values())
        if remaining_sum > 0:
            for name in weights:
                weights[name] /= remaining_sum
        print(f"\n[ENSEMBLE] Zeroed out negligible weights: {zeroed_models}")
    seq_weight = weights['seq']
    rf_weight = weights['rf']
    xgb_weight = weights['xgb']
    ridge_weight = weights['ridge']
    svr_weight = weights['svr']

    print("\n[DATA] Ensemble Weights (based on validation performance):")
    if use_sequence_model:
        print(f"   - {sequence_model_name}:         {seq_weight:.6f}")
    print(f"   - Random Forest: {rf_weight:.6f}")
    print(f"   - XGBoost:      {xgb_weight:.6f}")
    print(f"   - Ridge:        {ridge_weight:.6f}")
    print(f"   - SVR:          {svr_weight:.6f}")

    # Create ensemble predictions (all predictions now in UNSCALED space)
    ensemble_train_pred = rf_weight * rf_train_pred + xgb_weight * xgb_train_pred + ridge_weight * ridge_train_pred + svr_weight * svr_train_pred
    ensemble_val_pred = rf_weight * rf_val_pred + xgb_weight * xgb_val_pred + ridge_weight * ridge_val_pred + svr_weight * svr_val_pred
    ensemble_test_pred = rf_weight * rf_test_pred + xgb_weight * xgb_test_pred + ridge_weight * ridge_test_pred + svr_weight * svr_test_pred
    if use_sequence_model:
        ensemble_train_pred += seq_weight * seq_train_pred
        ensemble_val_pred += seq_weight * seq_val_pred
        ensemble_test_pred += seq_weight * seq_test_pred

    # Evaluate ensemble (using aligned ground truth — all in UNSCALED space)
    ensemble_train_metrics = {
        'mse': mean_squared_error(y_train_aligned, ensemble_train_pred),
        'r2': r2_score(y_train_aligned, ensemble_train_pred),
        'mae': mean_absolute_error(y_train_aligned, ensemble_train_pred)
    }
    ensemble_val_metrics = {
        'mse': mean_squared_error(y_val_aligned, ensemble_val_pred),
        'r2': r2_score(y_val_aligned, ensemble_val_pred),
        'mae': mean_absolute_error(y_val_aligned, ensemble_val_pred)
    }
    ensemble_test_metrics = {
        'mse': mean_squared_error(y_test_aligned, ensemble_test_pred),
        'r2': r2_score(y_test_aligned, ensemble_test_pred),
        'mae': mean_absolute_error(y_test_aligned, ensemble_test_pred)
    }

    # Store ensemble results
    ensemble_history: dict[str, Any] = {
        'weights': {seq_model_key: seq_weight, 'rf': rf_weight, 'xgb': xgb_weight,
                    'ridge': ridge_weight, 'svr': svr_weight},
        'train_metrics': ensemble_train_metrics,
        'val_metrics': ensemble_val_metrics,
        'test_metrics': ensemble_test_metrics
    }
    training_history['ensemble'] = ensemble_history

    # Detect ensemble overfitting
    ensemble_overfitted, _ensemble_overfitting_score, _ = detect_overfitting(
        ensemble_train_metrics, ensemble_val_metrics, ensemble_test_metrics, 
        "Ensemble", overfitting_threshold
    )

    # Final summary
    print("\n" + "="*60)
    print("[SUMMARY] TRAINING SUMMARY")
    print("="*60)
    if use_sequence_model:
        print(f"{sequence_model_name} Training Attempts:          {len(training_history[seq_model_key])}")
    print(f"Random Forest Training Attempts: {len(training_history['random_forest'])}")
    print(f"XGBoost Training Attempts:       {len(training_history['xgboost'])}")
    print(f"Ridge Training Attempts:         {len(training_history['ridge'])}")
    print(f"SVR Training Attempts:           {len(training_history['svr'])}")
    print("\n[DATA] FINAL TEST SET PERFORMANCE:")
    if use_sequence_model:
        print(f"   {sequence_model_name}:         R2={training_history[seq_model_key][-1]['test_metrics']['r2']:.4f}, MSE={training_history[seq_model_key][-1]['test_metrics']['mse']:.6f}")
    print(f"   Random Forest: R2={training_history['random_forest'][-1]['test_metrics']['r2']:.4f}, MSE={training_history['random_forest'][-1]['test_metrics']['mse']:.6f}")
    print(f"   XGBoost:      R2={training_history['xgboost'][-1]['test_metrics']['r2']:.4f}, MSE={training_history['xgboost'][-1]['test_metrics']['mse']:.6f}")
    print(f"   Ridge:        R2={training_history['ridge'][-1]['test_metrics']['r2']:.4f}, MSE={training_history['ridge'][-1]['test_metrics']['mse']:.6f}")
    print(f"   SVR:          R2={training_history['svr'][-1]['test_metrics']['r2']:.4f}, MSE={training_history['svr'][-1]['test_metrics']['mse']:.6f}")
    print(f"   [BEST] ENSEMBLE:   R2={ensemble_test_metrics['r2']:.4f}, MSE={ensemble_test_metrics['mse']:.6f}")
    print("="*60 + "\n")

    final_decision: dict[str, bool] = {
        'rf_final': not rf_overfitted,
        'xgb_final': not xgb_overfitted,
        'ensemble_final': not ensemble_overfitted
    }
    if use_sequence_model:
        final_decision[f'{seq_model_key}_final'] = not sequence_model_overfitted
    training_history['final_decision'] = final_decision

    # Return models dict for ensemble use
    ensemble_weights = ensemble_history['weights']
    models: dict[str, Any] = {
        'sequence_model': sequence_model,
        'sequence_model_type': seq_model_key,  # 'tcn', 'lstm', or 'none'
        'lstm': lstm_model if (use_sequence_model and not use_tcn) else None,
        'tcn': tcn_model if (use_sequence_model and use_tcn) else None,
        'rf': rf_model,
        'xgb': xgb_model,
        'ridge': ridge_model,
        'svr': svr_model,
        'ensemble_weights': ensemble_weights,
    }

    return models, training_history, lstm_datasets

def predict_future_price_changes(ticker, scaler_x, scaler_y, model, selected_features_list, stock_df, prediction_days, time_steps, historical_prediction_dataset_df=None, use_mc_dropout=True, mc_iterations=30, ensemble_weights=None):
    """
    Predicts the future stock price changes day by day with optional Monte Carlo Dropout
    for uncertainty estimation.

    Parameters:
    - ticker (str): The stock ticker.
    - scaler_x (MinMaxScaler): The scaler for x values.
    - scaler_y (MinMaxScaler): The scaler for y values (for inverse-transforming sequence model predictions).
    - model (dict): Dictionary containing 'sequence_model' (TCN or LSTM), 'rf', 'xgb' models.
    - selected_features_list (list): The list of selected features.
    - stock_df (pandas.DataFrame): A DataFrame containing the stock data.
    - prediction_days (int): The number of days to predict.
    - time_steps (int): Number of time steps for sequence model sequences.
    - historical_prediction_dataset_df (pd.DataFrame, optional): Historical prediction data.
    - use_mc_dropout (bool): Enable Monte Carlo Dropout for uncertainty estimation (default: True)
    - mc_iterations (int): Number of MC Dropout iterations (default: 30)
    - ensemble_weights (dict, optional): Weights for ensemble models from training validation MSE.
      Expected format: {'tcn': 0.4, 'rf': 0.3, 'xgb': 0.3} or {'lstm': 0.4, 'rf': 0.3, 'xgb': 0.3}
      If None, uses equal weights (1/3 each).

    Returns:
    pandas.DataFrame: A DataFrame containing the predicted stock prices with uncertainty estimates.

    Raises:
    - ValueError: If the prediction could not be completed.
    """

    try:
        # Extract individual models from the 'model' dictionary
        # Support both new format (sequence_model) and legacy format (lstm)
        sequence_model = model.get('sequence_model')
        sequence_model_type = model.get('sequence_model_type', 'lstm')
        
        # Fallback to legacy 'lstm' key if sequence_model not provided
        if sequence_model is None:
            sequence_model = model.get('lstm') or model.get('tcn')
            if model.get('tcn') is not None:
                sequence_model_type = 'tcn'
            elif model.get('lstm') is not None:
                sequence_model_type = 'lstm'
        
        rf_model = model['rf']
        xgb_model = model.get('xgb', None)  # XGBoost model (optional for backward compatibility)
        ridge_model = model.get('ridge', None)  # Ridge model (optional)
        svr_model = model.get('svr', None)  # SVR model (optional)
        
        # Get ensemble weights from model dict or parameter
        if ensemble_weights is None:
            ensemble_weights = model.get('ensemble_weights', None)
        
        # Normalize ensemble weights and include all 5 models
        if ensemble_weights is not None:
            # Get the sequence model weight (tcn or lstm)
            seq_key = 'tcn' if sequence_model_type == 'tcn' else 'lstm'
            seq_weight = ensemble_weights.get(seq_key, 0.20)
            rf_weight = ensemble_weights.get('rf', 0.20)
            xgb_weight = ensemble_weights.get('xgb', 0.20)
            ridge_weight = ensemble_weights.get('ridge', 0.20)
            svr_weight = ensemble_weights.get('svr', 0.20)
            
            # Normalize to sum to 1.0
            weight_sum = seq_weight + rf_weight + xgb_weight + ridge_weight + svr_weight
            seq_weight /= weight_sum
            rf_weight /= weight_sum
            xgb_weight /= weight_sum
            ridge_weight /= weight_sum
            svr_weight /= weight_sum
        else:
            # Default: equal weights
            seq_weight = 1/5
            rf_weight = 1/5
            xgb_weight = 1/5
            ridge_weight = 1/5
            svr_weight = 1/5
        
        # Zero out weights for models that aren't available
        if sequence_model is None:
            seq_weight = 0.0
        if ridge_model is None:
            ridge_weight = 0.0
        if svr_model is None:
            svr_weight = 0.0
        
        # CRITICAL: Zero out negligible weights to prevent extreme-output models from contaminating ensemble.
        MIN_ENSEMBLE_WEIGHT = 0.005
        weights_dict = {'seq': seq_weight, 'rf': rf_weight, 'xgb': xgb_weight,
                        'ridge': ridge_weight, 'svr': svr_weight}
        zeroed_in_pred = []
        for name, w in weights_dict.items():
            if w < MIN_ENSEMBLE_WEIGHT:
                weights_dict[name] = 0.0
                zeroed_in_pred.append(name)
        if zeroed_in_pred:
            remaining_sum = sum(weights_dict.values())
            if remaining_sum > 0:
                for name in weights_dict:
                    weights_dict[name] /= remaining_sum
            print(f"[ENSEMBLE] Zeroed out negligible weights in prediction: {zeroed_in_pred}")
        seq_weight = weights_dict['seq']
        rf_weight = weights_dict['rf']
        xgb_weight = weights_dict['xgb']
        ridge_weight = weights_dict['ridge']
        svr_weight = weights_dict['svr']
        
        if sequence_model is not None:
            print(f"\n[MODEL] Using {sequence_model_type.upper()} as sequence model for predictions")
        else:
            print("\n[MODEL] Sequence model DISABLED — using 4-model ensemble (RF+XGB+Ridge+SVR)")
        print(f"[ENSEMBLE] Weights: {sequence_model_type.upper()}={seq_weight:.6f}, RF={rf_weight:.6f}, XGB={xgb_weight:.6f}, Ridge={ridge_weight:.6f}, SVR={svr_weight:.6f}")
        if use_mc_dropout:
            print(f"[MC DROPOUT] Enabled with {mc_iterations} iterations per prediction")

        # Define dynamic features that need recalculation
        short_term_dynamic_list = [
            # Returns
            '1M', '3M', '6M', '9M', '1Y', '2Y', '3Y', '4Y', '5Y',
            # Moving averages
            'sma_5', 'sma_20', 'sma_40', 'sma_120', 'sma_200',
            # Exponential moving averages
            'ema_5', 'ema_20', 'ema_40', 'ema_120', 'ema_200',
            # Standard deviations
            "std_Div_5", "std_Div_20", "std_Div_40", "std_Div_120", "std_Div_200",
            # Bollinger Bands
            "bollinger_Band_5_2STD", "bollinger_Band_20_2STD", "bollinger_Band_40_2STD",
            "bollinger_Band_120_2STD", "bollinger_Band_200_2STD",
            # Valuation ratios
            'p_s', 'p_e', 'p_b', 'p_fcf',
            # Momentum
            "momentum",
            # Technical indicators
            'rsi_14', 'macd', 'macd_histogram', 'macd_signal', 'atr_14',
            # Volume indicators
            'volume_sma_20', 'volume_ema_20', 'volume_ratio', 'vwap', 'obv',
            # Volatility indicators
            'volatility_5d', 'volatility_20d', 'volatility_60d'
        ]

        # Identify fundamental features (these are static and carried forward)
        fundamental_features = ['revenue', 'eps', 'book_Value_Per_Share', 'free_Cash_Flow_Per_Share',
                               'average_shares', 'operating_Cash_Flow', 'capital_Expenditure']

        features_list = selected_features_list.copy()

        # Append metadata columns
        features_list.insert(0, "date")
        features_list.insert(1, "close_Price")
        features_list.insert(2, "1D")

        stock_mod_df = stock_df.copy()
        historical_price_predictions = []
        pred_cfg = get_pred_config()

        # --- Process historical prediction dataset if provided ---
        if historical_prediction_dataset_df is not None and len(historical_prediction_dataset_df) > 0:
            pred_count = len(historical_prediction_dataset_df)
            pred_dates = stock_df["date"].iloc[-pred_count:].reset_index(drop=True)
            seed_index = max(len(stock_df) - pred_count - 1, 0)
            prev_historical_predicted_price = float(stock_df["close_Price"].iloc[seed_index])

            for run in range(len(historical_prediction_dataset_df)):
                # For historical predictions, use the SAME pre-scaled test data for all models
                # This ensures consistent scaling between LSTM, RF, and XGB
                
                # Prepare sequence model input (requires time_steps of history from scaled data)
                if sequence_model is not None:
                    start_idx = max(0, run - time_steps + 1)
                    end_idx = run + 1
                    
                    if end_idx - start_idx < time_steps:
                        padding_needed = time_steps - (end_idx - start_idx)
                        x_lstm_scaled = historical_prediction_dataset_df.iloc[0:end_idx][selected_features_list]
                        first_row = x_lstm_scaled.iloc[0:1]
                        padding = pd.concat([first_row] * padding_needed, ignore_index=True)
                        x_lstm_scaled = pd.concat([padding, x_lstm_scaled], ignore_index=True)
                    else:
                        x_lstm_scaled = historical_prediction_dataset_df.iloc[start_idx:end_idx][selected_features_list]
                    
                    x_lstm_array = x_lstm_scaled.values if hasattr(x_lstm_scaled, 'values') else np.array(x_lstm_scaled)
                    
                    if run < 6:
                        print(f"\n[DEBUG] Historical Prediction {run+1}:")
                        print("   Using PRE-SCALED test data (no re-scaling needed)")
                        print(f"   Input shape: {x_lstm_array.shape}")
                        print(f"   Input mean: {np.mean(x_lstm_array):.4f}, std: {np.std(x_lstm_array):.4f}")
                        print(f"   Input min: {np.min(x_lstm_array):.4f}, max: {np.max(x_lstm_array):.4f}")
                        print(f"   Sample features (last row): {x_lstm_array[-1, :5]}")
                    
                    x_lstm_array_np = np.asarray(x_lstm_array)
                    scaled_x_input_seq = x_lstm_array_np.reshape(
                        (1, time_steps, x_lstm_array_np.shape[1])
                    )

                # Prepare RF/XGB input (only current day features from test dataset)
                scaled_x_input_rf_df = historical_prediction_dataset_df.iloc[run:run+1][selected_features_list]

                # --- Predict with all available models ---

                # Sequence model (TCN/LSTM) prediction (scaled) -> inverse transform to original scale
                if sequence_model is not None:
                    forecast_seq_scaled = sequence_model.predict(scaled_x_input_seq, verbose=0)[0][0]
                    forecast_seq = scaler_y.inverse_transform([[forecast_seq_scaled]])[0][0]
                    
                    # DEBUG: Show sequence model prediction process
                    if run < 6:
                        print(f"   {sequence_model_type.upper()} scaled output: {forecast_seq_scaled:.6f}")
                        print(f"   {sequence_model_type.upper()} unscaled output: {forecast_seq:.6f} ({forecast_seq*100:.3f}%)")
                        print(f"   scaler_y min: {scaler_y.data_min_}, max: {scaler_y.data_max_}")
                else:
                    forecast_seq = 0.0

                # Random Forest prediction (already unscaled)
                # Convert DataFrame to numpy to avoid feature name warning
                forecast_rf = rf_model.predict(scaled_x_input_rf_df.values)[0]
                
                # Ridge prediction (already unscaled)
                if ridge_model is not None:
                    forecast_ridge = ridge_model.predict(scaled_x_input_rf_df.values)[0]
                else:
                    forecast_ridge = 0.0
                
                # SVR prediction (already unscaled)
                if svr_model is not None:
                    forecast_svr = svr_model.predict(scaled_x_input_rf_df.values)[0]
                else:
                    forecast_svr = 0.0
                
                # XGBoost prediction (already unscaled)
                if xgb_model is not None:
                    forecast_xgb = xgb_model.predict(scaled_x_input_rf_df.values)[0]
                else:
                    forecast_xgb = None

                # Historical diagnostics should use the same bounded flat-model contract as future mode.
                hist_models = {'rf': forecast_rf}
                if forecast_xgb is not None:
                    hist_models['xgb'] = forecast_xgb
                if ridge_model is not None:
                    hist_models['ridge'] = forecast_ridge
                if svr_model is not None:
                    hist_models['svr'] = forecast_svr

                historical_blend = combine_flat_model_predictions(
                    hist_models,
                    weights={'rf': rf_weight, 'xgb': xgb_weight, 'ridge': ridge_weight, 'svr': svr_weight},
                    max_daily_return=pred_cfg.max_daily_return,
                )
                hist_models = historical_blend['clipped_predictions']
                forecast_rf = hist_models['rf']
                if forecast_xgb is not None:
                    forecast_xgb = hist_models['xgb']
                if ridge_model is not None:
                    forecast_ridge = hist_models['ridge']
                if svr_model is not None:
                    forecast_svr = hist_models['svr']
                hist_weights = historical_blend['weights']
                forecast_price_change = historical_blend['ensemble_prediction']
                    
                # Get actual value for comparison
                current_date = pred_dates.iloc[run]
                actual_price_change = stock_df.loc[stock_df["date"] == current_date, "1D"].values[0] if "1D" in stock_df.columns else None
                
                model_str = ", ".join(f"{k.upper()}={v:+.6f}" for k, v in hist_models.items())
                weight_str = ", ".join(f"{k.upper()}={v:.3f}" for k, v in hist_weights.items())
                clipped_models = historical_blend['clipped_models']
                clipped_str = ", ".join(
                    f"{name.upper()} {raw:+.6f}->{clipped:+.6f}"
                    for name, (raw, clipped) in clipped_models.items()
                )

                print(f"\n[PRED] Historical Prediction Day {run+1} ({current_date.strftime('%Y-%m-%d')}):")
                if sequence_model is not None:
                    print(f"   {sequence_model_type.upper()}:      {forecast_seq:+.6f} ({forecast_seq*100:+.3f}%)")
                print(f"   RF:        {forecast_rf:+.6f} ({forecast_rf*100:+.3f}%)")
                if forecast_xgb is not None:
                    print(f"   XGB:       {forecast_xgb:+.6f} ({forecast_xgb*100:+.3f}%)")
                if ridge_model is not None:
                    print(f"   Ridge:     {forecast_ridge:+.6f} ({forecast_ridge*100:+.3f}%)")
                if svr_model is not None:
                    print(f"   SVR:       {forecast_svr:+.6f} ({forecast_svr*100:+.3f}%)")
                print(f"   Ensemble:  {forecast_price_change:+.6f} ({forecast_price_change*100:+.3f}%)")
                print(f"   Weights:   {weight_str}")
                if clipped_str:
                    print(f"   Clipped:   {clipped_str}")
                if actual_price_change is not None:
                    print(f"   Actual:    {actual_price_change:+.6f} ({actual_price_change*100:+.3f}%)")
                    err_parts = [f"RF={abs(forecast_rf-actual_price_change)*100:.3f}%"]
                    if forecast_xgb is not None:
                        err_parts.append(f"XGB={abs(forecast_xgb-actual_price_change)*100:.3f}%")
                    if sequence_model is not None:
                        err_parts.insert(0, f"{sequence_model_type.upper()}={abs(forecast_seq-actual_price_change)*100:.3f}%")
                    if ridge_model is not None:
                        err_parts.append(f"Ridge={abs(forecast_ridge-actual_price_change)*100:.3f}%")
                    if svr_model is not None:
                        err_parts.append(f"SVR={abs(forecast_svr-actual_price_change)*100:.3f}%")
                    err_parts.append(f"Ensemble={abs(forecast_price_change-actual_price_change)*100:.3f}%")
                    print(f"   Errors:    {', '.join(err_parts)}")

                if model_str and run < 6:
                    print(f"   Models:    {model_str}")

                # Update stock_mod_df with predictions
                stock_mod_df.loc[stock_mod_df["date"] == current_date, "1D"] = forecast_price_change

                # Track historical predicted prices separately so future forecasts still
                # start from the actual latest close instead of a compounded backtest path.
                prev_historical_predicted_price = prev_historical_predicted_price * (1 + forecast_price_change)
                historical_price_predictions.append(
                    {
                        "date": current_date,
                        "predicted_close_Price": prev_historical_predicted_price,
                    }
                )

        # --- Future predictions loop ---
        
        history_cache = build_prediction_history_cache(
            ticker,
            stock_mod_df,
            selected_features_list,
            history_fetcher=yf.download,
        )
        max_daily_return = pred_cfg.max_daily_return

        # Calculate historical statistics for prediction stabilization (fixes #1-2)
        historical_returns = stock_df["1D"].dropna()
        historical_mean = historical_returns.mean()
        historical_std = historical_returns.std()
        print("\n[STABILIZATION] Historical return statistics:")
        print(f"   Mean: {historical_mean*100:.4f}%")
        print(f"   Std:  {historical_std*100:.4f}%")
        print("   Using for noise injection and mean reversion constraints.")
        
        # Track recent predictions for directional balance
        recent_prediction_values = []
        
        # Track uncertainty data for each future prediction day
        uncertainty_data = []
        # Track per-model predictions for each future day
        per_model_predictions = []
        
        # --- DIAGNOSTIC: Check scaler range vs current data ---
        print("\n[DIAG] Scaler X selected-feature range check (training min/max vs current values):")
        exclude_cols_diag = ["date", "name", "date_published", "ticker", "currency", "open_Price", "high_Price", "low_Price", "close_Price", "trade_Volume", "1D", "prediction", "financial_date_used"]
        all_diag_features = [col for col in stock_mod_df.columns if col not in exclude_cols_diag]
        current_row = stock_mod_df.iloc[-1:][all_diag_features].apply(pd.to_numeric, errors='coerce')
        scaled_current = scaler_x.transform(current_row)
        drift_summary = summarize_scaled_feature_drift(
            scaled_current,
            feature_names=all_diag_features,
            selected_features=selected_features_list,
        )
        mild_out_of_range_features = [
            (feature_name, scaled_value, float(current_row[feature_name].iloc[0]))
            for feature_name, scaled_value in drift_summary['mild_out_of_range']
        ]
        severe_out_of_range_features = [
            (feature_name, scaled_value, float(current_row[feature_name].iloc[0]))
            for feature_name, scaled_value in drift_summary['severe_out_of_range']
        ]
        if severe_out_of_range_features:
            print(
                f"   [WARN] {len(severe_out_of_range_features)} selected features exceed clip band [-1,2]:"
            )
            for feat_name, scaled_val, raw_val in sorted(severe_out_of_range_features, key=lambda x: abs(x[1]), reverse=True)[:15]:
                print(f"      {feat_name}: scaled={scaled_val:.3f}  raw={raw_val:.2f}")
        elif mild_out_of_range_features:
            print(
                f"   [INFO] {len(mild_out_of_range_features)} selected features are outside training [0,1] range "
                "but still within the runtime clip band [-1,2]."
            )
            for feat_name, scaled_val, raw_val in sorted(mild_out_of_range_features, key=lambda x: abs(x[1]), reverse=True)[:15]:
                print(f"      {feat_name}: scaled={scaled_val:.3f}  raw={raw_val:.2f}")
        else:
            print(f"   [OK] All {drift_summary['feature_count']} selected features within scaler range")
        print()

        for run in range(prediction_days):
            print("stock_mod_df before prediction:\n", stock_mod_df.tail(3))
            future_df = stock_mod_df.iloc[-1].copy().to_frame().transpose()
            future_day = pd.to_datetime(stock_mod_df.iloc[-1]["date"]) + relativedelta(days=1)
            # print every column name in future_df
            # print("Columns in future_df:", future_df.columns.tolist())

            # Skip weekends
            while future_day.weekday() >= 5: # 5=Sat, 6=Sun
                future_day += datetime.timedelta(days=1)

            future_df["date"] = future_day.strftime("%Y-%m-%d")

            # --- Carry forward fundamental features with decay ---
            # As predictions move further from real data, fundamental features
            # gradually decay toward the training-set mean (feature degradation).
            for fundamental_feature in fundamental_features:
                if fundamental_feature in stock_mod_df.columns:
                    real_value = stock_df.iloc[-1][fundamental_feature]
                    decay_rate = pred_cfg.feature_decay_rates.get(fundamental_feature, 0.02)
                    decay_factor = max(0.0, 1.0 - decay_rate * (run + 1))
                    # Blend toward 0 (scaled mean) as confidence drops
                    future_df[fundamental_feature] = real_value * decay_factor

            # --- Recalculate dynamic features ---
            for feature in short_term_dynamic_list:
                if feature not in selected_features_list:
                    continue

                # print(f"[CALC] Recalculating feature: {feature}")
                def get_historical_data(period):
                    return history_cache.get(period, pd.DataFrame())

                try:
                    # Calculate returns for various periods
                    if feature == "1M":
                        if len(stock_mod_df) <= 21:
                            hist_df = get_historical_data("1y")
                            if not hist_df.empty and len(hist_df) >= 22:
                                future_df["1M"] = hist_df.iloc[-22:]["close_Price"].pct_change(21).iloc[-1]
                            else:
                                future_df["1M"] = 0
                        else:
                            future_df["1M"] = stock_mod_df.iloc[-22:]["close_Price"].pct_change(21).iloc[-1]

                    elif feature == "3M":
                        if len(stock_mod_df) <= 63:
                            hist_df = get_historical_data("1y")
                            if not hist_df.empty and len(hist_df) >= 64:
                                future_df["3M"] = hist_df.iloc[-64:]["close_Price"].pct_change(63).iloc[-1]
                            else:
                                future_df["3M"] = 0
                        else:
                            future_df["3M"] = stock_mod_df.iloc[-64:]["close_Price"].pct_change(63).iloc[-1]

                    elif feature == "6M":
                        if len(stock_mod_df) <= 126:
                            hist_df = get_historical_data("1y")
                            if not hist_df.empty and len(hist_df) >= 127:
                                future_df["6M"] = hist_df.iloc[-127:]["close_Price"].pct_change(126).iloc[-1]
                            else:
                                future_df["6M"] = 0
                        else:
                            future_df["6M"] = stock_mod_df.iloc[-127:]["close_Price"].pct_change(126).iloc[-1]

                    elif feature == "9M":
                        if len(stock_mod_df) <= 189:
                            hist_df = get_historical_data("1y")
                            if not hist_df.empty and len(hist_df) >= 190:
                                future_df["9M"] = hist_df.iloc[-190:]["close_Price"].pct_change(189).iloc[-1]
                            else:
                                future_df["9M"] = 0
                        else:
                            future_df["9M"] = stock_mod_df.iloc[-190:]["close_Price"].pct_change(189).iloc[-1]

                    elif feature == "1Y":
                        if len(stock_mod_df) <= 252:
                            hist_df = get_historical_data("2y")
                            if not hist_df.empty and len(hist_df) >= 253:
                                future_df["1Y"] = hist_df.iloc[-253:]["close_Price"].pct_change(252).iloc[-1]
                            else:
                                future_df["1Y"] = 0
                        else:
                            future_df["1Y"] = stock_mod_df.iloc[-253:]["close_Price"].pct_change(252).iloc[-1]

                    elif feature == "2Y":
                        if len(stock_mod_df) <= 504:
                            hist_df = get_historical_data("3y")
                            if not hist_df.empty and len(hist_df) >= 505:
                                future_df["2Y"] = hist_df.iloc[-505:]["close_Price"].pct_change(504).iloc[-1]
                            else:
                                future_df["2Y"] = 0
                        else:
                            future_df["2Y"] = stock_mod_df.iloc[-505:]["close_Price"].pct_change(504).iloc[-1]

                    elif feature == "3Y":
                        if len(stock_mod_df) <= 756:
                            hist_df = get_historical_data("4y")
                            if not hist_df.empty and len(hist_df) >= 757:
                                future_df["3Y"] = hist_df.iloc[-757:]["close_Price"].pct_change(756).iloc[-1]
                            else:
                                future_df["3Y"] = 0
                        else:
                            future_df["3Y"] = stock_mod_df.iloc[-757:]["close_Price"].pct_change(756).iloc[-1]

                    elif feature == "4Y":
                        if len(stock_mod_df) <= 1008:
                            hist_df = get_historical_data("5y")
                            if not hist_df.empty and len(hist_df) >= 1009:
                                future_df["4Y"] = hist_df.iloc[-1009:]["close_Price"].pct_change(1008).iloc[-1]
                            else:
                                future_df["4Y"] = 0
                        else:
                            future_df["4Y"] = stock_mod_df.iloc[-1009:]["close_Price"].pct_change(1008).iloc[-1]

                    elif feature == "5Y":
                        if len(stock_mod_df) <= 1260:
                            hist_df = get_historical_data("6y")
                            if not hist_df.empty and len(hist_df) >= 1261:
                                future_df["5Y"] = hist_df.iloc[-1261:]["close_Price"].pct_change(1260).iloc[-1]
                            else:
                                future_df["5Y"] = 0
                        else:
                            future_df["5Y"] = stock_mod_df.iloc[-1261:]["close_Price"].pct_change(1260).iloc[-1]

                    # Moving averages
                    elif feature == "sma_5":
                        future_df["sma_5"] = stock_mod_df.iloc[-5:]["close_Price"].mean()

                    elif feature == "sma_20":
                        future_df["sma_20"] = stock_mod_df.iloc[-20:]["close_Price"].mean()

                    elif feature == "sma_40":
                        future_df["sma_40"] = stock_mod_df.iloc[-40:]["close_Price"].mean()

                    elif feature == "sma_120":
                        future_df["sma_120"] = stock_mod_df.iloc[-120:]["close_Price"].mean()

                    elif feature == "sma_200":
                        future_df["sma_200"] = stock_mod_df.iloc[-200:]["close_Price"].mean()

                    elif feature == "ema_5":
                        future_df["ema_5"] = stock_mod_df.iloc[-5:]["close_Price"].ewm(span=5, adjust=False).mean().iloc[-1]

                    elif feature == "ema_20":
                        future_df["ema_20"] = stock_mod_df.iloc[-20:]["close_Price"].ewm(span=20, adjust=False).mean().iloc[-1]

                    elif feature == "ema_40":
                        future_df["ema_40"] = stock_mod_df.iloc[-40:]["close_Price"].ewm(span=40, adjust=False).mean().iloc[-1]

                    elif feature == "ema_120":
                        future_df["ema_120"] = stock_mod_df.iloc[-120:]["close_Price"].ewm(span=120, adjust=False).mean().iloc[-1]

                    elif feature == "ema_200":
                        future_df["ema_200"] = stock_mod_df.iloc[-200:]["close_Price"].ewm(span=200, adjust=False).mean().iloc[-1]

                    # Standard deviations
                    elif feature == "std_Div_5":
                        future_df["std_Div_5"] = stock_mod_df.iloc[-5:]["close_Price"].std()

                    elif feature == "std_Div_20":
                        future_df["std_Div_20"] = stock_mod_df.iloc[-20:]["close_Price"].std()

                    elif feature == "std_Div_40":
                        future_df["std_Div_40"] = stock_mod_df.iloc[-40:]["close_Price"].std()

                    elif feature == "std_Div_120":
                        future_df["std_Div_120"] = stock_mod_df.iloc[-120:]["close_Price"].std()

                    elif feature == "std_Div_200":
                        future_df["std_Div_200"] = stock_mod_df.iloc[-200:]["close_Price"].std()

                    # Bollinger Bands
                    elif feature == "bollinger_Band_5_2STD":
                        std_div_5 = stock_mod_df.iloc[-5:]["close_Price"].std()
                        sma_5 = stock_mod_df.iloc[-5:]["close_Price"].mean()
                        bollinger_Band_5_Upper = sma_5 + (std_div_5 * 2)
                        bollinger_Band_5_Lower = sma_5 - (std_div_5 * 2)
                        future_df["bollinger_Band_5_2STD"] = bollinger_Band_5_Upper - bollinger_Band_5_Lower

                    elif feature == "bollinger_Band_20_2STD":
                        std_div_20 = stock_mod_df.iloc[-20:]["close_Price"].std()
                        sma_20 = stock_mod_df.iloc[-20:]["close_Price"].mean()
                        bollinger_Band_20_Upper = sma_20 + (std_div_20 * 2)
                        bollinger_Band_20_Lower = sma_20 - (std_div_20 * 2)
                        future_df["bollinger_Band_20_2STD"] = bollinger_Band_20_Upper - bollinger_Band_20_Lower

                    elif feature == "bollinger_Band_40_2STD":
                        std_div_40 = stock_mod_df.iloc[-40:]["close_Price"].std()
                        sma_40 = stock_mod_df.iloc[-40:]["close_Price"].mean()
                        bollinger_Band_40_Upper = sma_40 + (std_div_40 * 2)
                        bollinger_Band_40_Lower = sma_40 - (std_div_40 * 2)
                        future_df["bollinger_Band_40_2STD"] = bollinger_Band_40_Upper - bollinger_Band_40_Lower

                    elif feature == "bollinger_Band_120_2STD":
                        std_div_120 = stock_mod_df.iloc[-120:]["close_Price"].std()
                        sma_120 = stock_mod_df.iloc[-120:]["close_Price"].mean()
                        bollinger_Band_120_Upper = sma_120 + (std_div_120 * 2)
                        bollinger_Band_120_Lower = sma_120 - (std_div_120 * 2)
                        future_df["bollinger_Band_120_2STD"] = bollinger_Band_120_Upper - bollinger_Band_120_Lower

                    elif feature == "bollinger_Band_200_2STD":
                        std_div_200 = stock_mod_df.iloc[-200:]["close_Price"].std()
                        sma_200 = stock_mod_df.iloc[-200:]["close_Price"].mean()
                        bollinger_Band_200_Upper = sma_200 + (std_div_200 * 2)
                        bollinger_Band_200_Lower = sma_200 - (std_div_200 * 2)
                        future_df["bollinger_Band_200_2STD"] = bollinger_Band_200_Upper - bollinger_Band_200_Lower

                    # Valuation ratios (use carried-forward fundamental data)
                    elif feature == "p_s":
                        if "revenue" in future_df.columns and "average_shares" in future_df.columns:
                            revenue = future_df["revenue"].iloc[0]
                            avg_shares = future_df["average_shares"].iloc[0]
                            if pd.notna(revenue) and pd.notna(avg_shares) and avg_shares != 0:
                                future_df["p_s"] = future_df["close_Price"].iloc[0] / (revenue / avg_shares)
                            else:
                                future_df["p_s"] = stock_mod_df.iloc[-1]["p_s"] if "p_s" in stock_mod_df.columns else 0
                        else:
                            future_df["p_s"] = stock_mod_df.iloc[-1]["p_s"] if "p_s" in stock_mod_df.columns else 0

                    elif feature == "p_e":
                        if "eps" in future_df.columns:
                            eps = future_df["eps"].iloc[0]
                            if pd.notna(eps) and eps != 0:
                                future_df["p_e"] = future_df["close_Price"].iloc[0] / eps
                            else:
                                future_df["p_e"] = stock_mod_df.iloc[-1]["p_e"] if "p_e" in stock_mod_df.columns else 0
                        else:
                            future_df["p_e"] = stock_mod_df.iloc[-1]["p_e"] if "p_e" in stock_mod_df.columns else 0

                    elif feature == "p_b":
                        if "book_Value_Per_Share" in future_df.columns:
                            bvps = future_df["book_Value_Per_Share"].iloc[0]
                            if pd.notna(bvps) and bvps != 0:
                                future_df["p_b"] = future_df["close_Price"].iloc[0] / bvps
                            else:
                                future_df["p_b"] = stock_mod_df.iloc[-1]["p_b"] if "p_b" in stock_mod_df.columns else 0
                        else:
                            future_df["p_b"] = stock_mod_df.iloc[-1]["p_b"] if "p_b" in stock_mod_df.columns else 0

                    elif feature == "p_fcf":
                        if "free_Cash_Flow_Per_Share" in future_df.columns:
                            fcfps = future_df["free_Cash_Flow_Per_Share"].iloc[0]
                            if pd.notna(fcfps) and fcfps != 0:
                                future_df["p_fcf"] = future_df["close_Price"].iloc[0] / fcfps
                            else:
                                future_df["p_fcf"] = stock_mod_df.iloc[-1]["p_fcf"] if "p_fcf" in stock_mod_df.columns else 0
                        else:
                            future_df["p_fcf"] = stock_mod_df.iloc[-1]["p_fcf"] if "p_fcf" in stock_mod_df.columns else 0

                    # Momentum (Fix #3: Added decay and bounds to prevent momentum trap)
                    elif feature == "momentum":
                        prev_momentum = stock_mod_df.iloc[-1].get("momentum", 0)
                        
                        # Apply decay factor to prevent self-reinforcing loops
                        # Momentum naturally decays towards 0 over time
                        decay_factor = 0.90  # 10% decay per day
                        decayed_momentum = prev_momentum * decay_factor
                        
                        if stock_mod_df.iloc[-1]["close_Price"] >= stock_mod_df.iloc[-2]["close_Price"]:
                            # Price went up
                            if decayed_momentum <= 0:
                                momentum = 1  # Reset to 1 on direction change
                            else:
                                momentum = decayed_momentum + 1  # Increment with decay
                        else:
                            # Price went down
                            if decayed_momentum >= 0:
                                momentum = -1  # Reset to -1 on direction change
                            else:
                                momentum = decayed_momentum - 1  # Decrement with decay
                        
                        # Cap momentum to prevent extreme values (bounds: -15 to 15)
                        momentum = np.clip(momentum, -15, 15)
                        future_df["momentum"] = momentum

                    # Technical indicators (dynamic calculation using predicted prices)
                    elif feature == "rsi_14":
                        # Calculate RSI from last 14 days including predictions
                        if len(stock_mod_df) >= 15:
                            prices = stock_mod_df.iloc[-15:]["close_Price"]
                            deltas = prices.diff()
                            gains = deltas.where(deltas > 0, 0)
                            losses = -deltas.where(deltas < 0, 0)
                            avg_gain = gains.rolling(window=14, min_periods=14).mean().iloc[-1]
                            avg_loss = losses.rolling(window=14, min_periods=14).mean().iloc[-1]
                            if avg_loss != 0:
                                rs = avg_gain / avg_loss
                                rsi = 100 - (100 / (1 + rs))
                            else:
                                rsi = 100 if avg_gain > 0 else 50
                            future_df["rsi_14"] = rsi
                        else:
                            # Not enough history, carry forward
                            future_df["rsi_14"] = stock_mod_df.iloc[-1]["rsi_14"] if "rsi_14" in stock_mod_df.columns else 50.0

                    elif feature == "macd":
                        # Calculate MACD from 12 and 26-day EMAs of predicted prices
                        if len(stock_mod_df) >= 26:
                            prices = stock_mod_df["close_Price"]
                            ema_12 = prices.ewm(span=12, adjust=False).mean().iloc[-1]
                            ema_26 = prices.ewm(span=26, adjust=False).mean().iloc[-1]
                            future_df["macd"] = ema_12 - ema_26
                        else:
                            # Not enough history, carry forward
                            future_df["macd"] = stock_mod_df.iloc[-1]["macd"] if "macd" in stock_mod_df.columns else 0.0

                    elif feature == "macd_histogram":
                        # Calculate MACD histogram (MACD - Signal)
                        if "macd" in future_df.columns and "macd_signal" in future_df.columns:
                            future_df["macd_histogram"] = future_df["macd"].iloc[0] - future_df["macd_signal"].iloc[0]
                        else:
                            # Not enough history, carry forward
                            future_df["macd_histogram"] = stock_mod_df.iloc[-1]["macd_histogram"] if "macd_histogram" in stock_mod_df.columns else 0.0

                    elif feature == "macd_signal":
                        # Calculate MACD signal line (9-day EMA of MACD)
                        if len(stock_mod_df) >= 35 and "macd" in stock_mod_df.columns:  # 26 for MACD + 9 for signal
                            macd_values = stock_mod_df.iloc[-35:]["macd"]
                            signal = macd_values.ewm(span=9, adjust=False).mean().iloc[-1]
                            future_df["macd_signal"] = signal
                        else:
                            # Not enough history, carry forward
                            future_df["macd_signal"] = stock_mod_df.iloc[-1]["macd_signal"] if "macd_signal" in stock_mod_df.columns else 0.0

                    elif feature == "atr_14":
                        # ATR requires high/low/close data, carry forward
                        future_df["atr_14"] = stock_mod_df.iloc[-1]["atr_14"] if "atr_14" in stock_mod_df.columns else 0.0

                    # Volume indicators (carry forward - volume data not available for future)
                    elif feature == "volume_sma_20":
                        # Volume SMA, carry forward last known value
                        future_df["volume_sma_20"] = stock_mod_df.iloc[-1]["volume_sma_20"] if "volume_sma_20" in stock_mod_df.columns else 0.0

                    elif feature == "volume_ema_20":
                        # Volume EMA, carry forward
                        future_df["volume_ema_20"] = stock_mod_df.iloc[-1]["volume_ema_20"] if "volume_ema_20" in stock_mod_df.columns else 0.0

                    elif feature == "volume_ratio":
                        # Volume ratio, carry forward
                        future_df["volume_ratio"] = stock_mod_df.iloc[-1]["volume_ratio"] if "volume_ratio" in stock_mod_df.columns else 1.0

                    elif feature == "vwap":
                        # VWAP, carry forward
                        future_df["vwap"] = stock_mod_df.iloc[-1]["vwap"] if "vwap" in stock_mod_df.columns else future_df["close_Price"].iloc[0]

                    elif feature == "obv":
                        # OBV (On-Balance Volume), carry forward
                        future_df["obv"] = stock_mod_df.iloc[-1]["obv"] if "obv" in stock_mod_df.columns else 0.0

                    # Volatility indicators
                    elif feature == "volatility_5d":
                        # 5-day volatility based on returns
                        returns = stock_mod_df.iloc[-5:]["close_Price"].pct_change()
                        future_df["volatility_5d"] = returns.std() if len(returns) >= 5 else 0.0

                    elif feature == "volatility_20d":
                        # 20-day volatility
                        returns = stock_mod_df.iloc[-20:]["close_Price"].pct_change()
                        future_df["volatility_20d"] = returns.std() if len(returns) >= 20 else 0.0

                    elif feature == "volatility_60d":
                        # 60-day volatility
                        returns = stock_mod_df.iloc[-60:]["close_Price"].pct_change()
                        future_df["volatility_60d"] = returns.std() if len(returns) >= 60 else 0.0

                except (ValueError, KeyError, IndexError, ZeroDivisionError) as e:
                    print(f"[WARN] Error calculating feature '{feature}': {e}")
                    # Carry forward the last known value or use 0
                    if feature in stock_mod_df.columns:
                        future_df[feature] = stock_mod_df.iloc[-1][feature]
                    else:
                        future_df[feature] = 0

            # Convert date to datetime
            future_df["date"] = pd.to_datetime(future_df["date"])
            # print("future_df:\n", future_df[["date", "close_Price"]])
            # print every column name in future_df
            # print("Columns in future_df:", future_df.columns.tolist())

            # Concat the newly calculated day to historical data
            stock_mod_df = pd.concat([stock_mod_df, future_df], axis=0).reset_index(drop=True)
            # print every column name in stock_mod_df
            # print("Columns in stock_mod_df:", stock_mod_df.columns.tolist())
            # print("stock_mod_df after prediction:\n", stock_mod_df[["date", "close_Price"]].tail(3))

            # --- Prepare Input Data for the three models ---

            # A. LSTM Input (Last time_steps sequence)
            # Need to scale ALL features first (scaler was fitted on all features)
            # then select the subset for LSTM
            
            # Get all features except metadata and raw OHLCV columns (same as during training)
            # Raw OHLCV excluded because they won't be available for future predictions
            exclude_cols = ["date", "name", "date_published", "ticker", "currency", "open_Price", "high_Price", "low_Price", "close_Price", "trade_Volume", "1D", "prediction", "financial_date_used"]
            all_features = [col for col in stock_mod_df.columns if col not in exclude_cols]
            
            # Get the last time_steps rows with ALL features for scaling
            x_seq_all_features = stock_mod_df.iloc[-time_steps:][all_features]

            # Check and handle NaN in sequence model input
            if x_seq_all_features.isnull().any().any():
                print(f"[WARN] NaN detected in {sequence_model_type.upper()} input at day {run+1}")
                nan_cols = x_seq_all_features.columns[x_seq_all_features.isnull().any()].tolist()
                print(f"   Columns with NaN: {nan_cols}")
                # printout the values of the columns with NaN before filling
                print(x_seq_all_features[nan_cols])
                x_seq_all_features = x_seq_all_features.ffill().bfill().fillna(0)

            # Scale ALL features (scaler expects all features it was fitted on)
            scaled_x_seq_all_df = scaler_x.transform(x_seq_all_features)
            
            # CRITICAL: Clamp scaled features to prevent OOD extrapolation in neural network.
            # Financial features can grow far beyond training range, producing scaled values
            # in the billions instead of [0,1]. Clamping to [-1, 2] preserves mild extrapolation
            # while preventing catastrophic neural network outputs.
            scaled_x_seq_all_df = clip_scaled_features(scaled_x_seq_all_df)
            
            # Now select only the features needed for sequence model
            scaled_x_seq_df = scaled_x_seq_all_df[selected_features_list]
            
            # Convert to numpy array after scaling
            scaled_x_seq_array = scaled_x_seq_df.values if hasattr(scaled_x_seq_df, 'values') else np.array(scaled_x_seq_df)
            scaled_x_input_seq = scaled_x_seq_array.reshape(1, time_steps, scaled_x_seq_array.shape[1])

            # B. Random Forest/XGB Input (current day's features, SCALED like training data)
            # CRITICAL: RF/XGB were trained on MinMax-scaled features.
            # Must scale through scaler_x before prediction (same as TCN path above).
            x_input_rf_all_features = stock_mod_df.iloc[-1:][all_features]
            x_input_rf_all_features = x_input_rf_all_features.apply(pd.to_numeric, errors='coerce')

            # Check and handle NaN in RF input
            if x_input_rf_all_features.isnull().any().any():
                print(f"[WARN] NaN detected in RF input at day {run+1}")
                nan_cols = x_input_rf_all_features.columns[x_input_rf_all_features.isnull().any()].tolist()
                print(f"   Columns with NaN: {nan_cols}")
                x_input_rf_all_features = x_input_rf_all_features.ffill().bfill().fillna(0)

            # Scale ALL features first, then select the model's features
            scaled_x_input_rf_all = scaler_x.transform(x_input_rf_all_features)
            scaled_x_input_rf_all = clip_scaled_features(scaled_x_input_rf_all)
            x_input_rf_df = scaled_x_input_rf_all[selected_features_list]

            # DIAGNOSTIC: Show scaled feature statistics for the first few predictions
            if run < 3:
                rf_vals = x_input_rf_df.values.flatten()
                oor_count = np.sum((rf_vals < -0.1) | (rf_vals > 1.1))
                print(f"\n[DIAG] Future Day {run+1} RF/XGB input: "
                      f"mean={np.mean(rf_vals):.3f}, min={np.min(rf_vals):.3f}, "
                      f"max={np.max(rf_vals):.3f}, out-of-range={oor_count}/{len(rf_vals)}")

            # --- Predict and Ensemble with all available models ---

            # Sequence model (TCN/LSTM) prediction with Monte Carlo Dropout for uncertainty
            if sequence_model is not None:
                if use_mc_dropout:
                    # Use MC Dropout for uncertainty estimation
                    mc_results = multi_run_prediction(
                        sequence_model, scaled_x_input_seq, scaler_y, 
                        n_runs=mc_iterations, model_type=sequence_model_type
                    )
                    forecast_seq = mc_results['mean']
                    forecast_seq_std = mc_results['std']
                    forecast_seq_ci_low = mc_results['percentile_5']
                    forecast_seq_ci_high = mc_results['percentile_95']
                    forecast_seq_confidence = mc_results['confidence']
                else:
                    # Standard single prediction (faster, no uncertainty)
                    forecast_seq_scaled = sequence_model.predict(scaled_x_input_seq, verbose=0)[0][0]
                    forecast_seq = scaler_y.inverse_transform([[forecast_seq_scaled]])[0][0]
                    forecast_seq_std = 0.0
                    forecast_seq_ci_low = forecast_seq
                    forecast_seq_ci_high = forecast_seq
                    forecast_seq_confidence = 'N/A'
            else:
                # Sequence model disabled
                forecast_seq = 0.0
                forecast_seq_std = 0.0
                forecast_seq_ci_low = 0.0
                forecast_seq_ci_high = 0.0
                forecast_seq_confidence = 'N/A'

            # Random Forest prediction (already unscaled)
            # Convert DataFrame to numpy to avoid feature name warning
            forecast_rf = rf_model.predict(x_input_rf_df.values)[0]

            # XGBoost prediction (already unscaled)
            if xgb_model is not None:
                forecast_xgb = xgb_model.predict(x_input_rf_df.values)[0]
            else:
                forecast_xgb = 0.0

            # Ridge prediction (already unscaled)
            if ridge_model is not None:
                forecast_ridge = ridge_model.predict(x_input_rf_df.values)[0]
            else:
                forecast_ridge = 0.0

            # SVR prediction (already unscaled)
            if svr_model is not None:
                forecast_svr = svr_model.predict(x_input_rf_df.values)[0]
            else:
                forecast_svr = 0.0

            if xgb_model is not None:
                # CRITICAL: Clip individual model predictions to a sane range before ensemble.
                forecast_seq_raw = forecast_seq
                forecast_rf_raw = forecast_rf
                forecast_xgb_raw = forecast_xgb
                forecast_ridge_raw = forecast_ridge if ridge_model is not None else None
                forecast_svr_raw = forecast_svr if svr_model is not None else None
                forecast_seq = np.clip(forecast_seq, -max_daily_return, max_daily_return)
                forecast_rf = np.clip(forecast_rf, -max_daily_return, max_daily_return)
                forecast_xgb = np.clip(forecast_xgb, -max_daily_return, max_daily_return)
                forecast_ridge = np.clip(forecast_ridge, -max_daily_return, max_daily_return)
                forecast_svr = np.clip(forecast_svr, -max_daily_return, max_daily_return)
                
                if run < 3:
                    clipped = []
                    if forecast_seq != forecast_seq_raw:
                        clipped.append(f"{sequence_model_type.upper()}: {forecast_seq_raw:+.4f} -> {forecast_seq:+.4f}")
                    if forecast_rf != forecast_rf_raw:
                        clipped.append(f"RF: {forecast_rf_raw:+.4f} -> {forecast_rf:+.4f}")
                    if forecast_xgb != forecast_xgb_raw:
                        clipped.append(f"XGB: {forecast_xgb_raw:+.4f} -> {forecast_xgb:+.4f}")
                    if ridge_model is not None and forecast_ridge_raw is not None and forecast_ridge != forecast_ridge_raw:
                        clipped.append(f"Ridge: {forecast_ridge_raw:+.4f} -> {forecast_ridge:+.4f}")
                    if svr_model is not None and forecast_svr_raw is not None and forecast_svr != forecast_svr_raw:
                        clipped.append(f"SVR: {forecast_svr_raw:+.4f} -> {forecast_svr:+.4f}")
                    if clipped:
                        print(f"[CLIP] Day {run+1} predictions clipped: {'; '.join(clipped)}")

                # ENSEMBLE: Validation MSE-weighted combination of all 5 models
                raw_ensemble = (
                    seq_weight * forecast_seq +
                    rf_weight * forecast_rf +
                    xgb_weight * forecast_xgb +
                    ridge_weight * forecast_ridge +
                    svr_weight * forecast_svr
                )
                ensemble_std = np.sqrt(
                    seq_weight**2 * forecast_seq_std**2 +
                    rf_weight**2 * (historical_std * 0.5)**2 +
                    xgb_weight**2 * (historical_std * 0.5)**2 +
                    ridge_weight**2 * (historical_std * 0.5)**2 +
                    svr_weight**2 * (historical_std * 0.5)**2
                )
                
                # Apply mean reversion guard rail for extreme predictions only
                forecast_price_change = stabilize_prediction(
                    raw_ensemble,
                    recent_prediction_values,
                    historical_mean,
                    historical_std,
                    pred_cfg,
                )
                if run < 5 and abs(forecast_price_change - raw_ensemble) > 1e-9:
                    print(
                        f"[STABILIZE] Day {run+1}: ensemble adjusted from "
                        f"{raw_ensemble:+.6f} to {forecast_price_change:+.6f}"
                    )

                future_date = future_df["date"].iloc[0]
                print(f"\n[FORECAST] Future Prediction Day {run+1} ({future_date}):")
                if sequence_model is not None:
                    if use_mc_dropout:
                        print(f"   {sequence_model_type.upper()}:      {forecast_seq:+.6f} ({forecast_seq*100:+.3f}%) ± {forecast_seq_std*100:.3f}% [{forecast_seq_confidence}]")
                        print(f"       90% CI: [{forecast_seq_ci_low*100:+.3f}%, {forecast_seq_ci_high*100:+.3f}%]")
                    else:
                        print(f"   {sequence_model_type.upper()}:      {forecast_seq:+.6f} ({forecast_seq*100:+.3f}%)")
                print(f"   RF:        {forecast_rf:+.6f} ({forecast_rf*100:+.3f}%)")
                print(f"   XGB:       {forecast_xgb:+.6f} ({forecast_xgb*100:+.3f}%)")
                if ridge_model is not None:
                    print(f"   Ridge:     {forecast_ridge:+.6f} ({forecast_ridge*100:+.3f}%)")
                if svr_model is not None:
                    print(f"   SVR:       {forecast_svr:+.6f} ({forecast_svr*100:+.3f}%)")
                print(f"   Ensemble:  {forecast_price_change:+.6f} ({forecast_price_change*100:+.3f}%) [w: {sequence_model_type.upper()}={seq_weight:.2f}, RF={rf_weight:.2f}, XGB={xgb_weight:.2f}, Ridge={ridge_weight:.2f}, SVR={svr_weight:.2f}]")

                # Show model agreement/disagreement
                predictions = [forecast_rf, forecast_xgb]
                if sequence_model is not None:
                    predictions.append(forecast_seq)
                if ridge_model is not None:
                    predictions.append(forecast_ridge)
                if svr_model is not None:
                    predictions.append(forecast_svr)
                std_dev = np.std(predictions)
                print(f"   Agreement: std={std_dev:.6f} ({'High' if std_dev < 0.01 else 'Medium' if std_dev < 0.02 else 'Low'} consensus)")
            else:
                # XGB unavailable — use whatever models we have
                forecast_seq = np.clip(forecast_seq, -max_daily_return, max_daily_return)
                forecast_rf = np.clip(forecast_rf, -max_daily_return, max_daily_return)
                if ridge_model is not None:
                    forecast_ridge = np.clip(forecast_ridge, -max_daily_return, max_daily_return)
                if svr_model is not None:
                    forecast_svr = np.clip(forecast_svr, -max_daily_return, max_daily_return)

                # Collect available models and their weights
                fallback_models = {'seq': (seq_weight, forecast_seq), 'rf': (rf_weight, forecast_rf)}
                if ridge_model is not None:
                    fallback_models['ridge'] = (ridge_weight, forecast_ridge)
                if svr_model is not None:
                    fallback_models['svr'] = (svr_weight, forecast_svr)

                # Renormalize weights
                fb_total = sum(w for w, _ in fallback_models.values())
                fb_weights = {k: w / fb_total for k, (w, _) in fallback_models.items()}

                raw_ensemble = sum(fb_weights[k] * pred for k, (_, pred) in fallback_models.items())
                ensemble_std = np.sqrt(
                    fb_weights['seq']**2 * forecast_seq_std**2 +
                    sum(fb_weights[k]**2 * (historical_std * 0.5)**2 for k in fb_weights if k != 'seq')
                )
                
                # Apply mean reversion guard rail for extreme predictions only
                forecast_price_change = stabilize_prediction(
                    raw_ensemble,
                    recent_prediction_values,
                    historical_mean,
                    historical_std,
                    pred_cfg,
                )
                if run < 5 and abs(forecast_price_change - raw_ensemble) > 1e-9:
                    print(
                        f"[STABILIZE] Day {run+1}: ensemble adjusted from "
                        f"{raw_ensemble:+.6f} to {forecast_price_change:+.6f}"
                    )

                future_date = future_df["date"].iloc[0]
                weight_str = ", ".join(f"{k.upper()}={fb_weights[k]:.2f}" for k in fb_weights)
                print(f"\n[FORECAST] Future Prediction Day {run+1} ({future_date}): [no-XGB fallback: {weight_str}]")
                if use_mc_dropout:
                    print(f"   {sequence_model_type.upper()}:      {forecast_seq:+.6f} ({forecast_seq*100:+.3f}%) ± {forecast_seq_std*100:.3f}% [{forecast_seq_confidence}]")
                    print(f"       90% CI: [{forecast_seq_ci_low*100:+.3f}%, {forecast_seq_ci_high*100:+.3f}%]")
                else:
                    print(f"   {sequence_model_type.upper()}:      {forecast_seq:+.6f} ({forecast_seq*100:+.3f}%)")
                print(f"   RF:        {forecast_rf:+.6f} ({forecast_rf*100:+.3f}%)")
                if ridge_model is not None:
                    print(f"   Ridge:     {forecast_ridge:+.6f} ({forecast_ridge*100:+.3f}%)")
                if svr_model is not None:
                    print(f"   SVR:       {forecast_svr:+.6f} ({forecast_svr*100:+.3f}%)")
                print(f"   Ensemble:  {forecast_price_change:+.6f} ({forecast_price_change*100:+.3f}%) [w: {weight_str}]")

                predictions = [forecast_seq, forecast_rf]
                if ridge_model is not None:
                    predictions.append(forecast_ridge)
                if svr_model is not None:
                    predictions.append(forecast_svr)
                std_dev = np.std(predictions)
                print(f"   Agreement: std={std_dev:.6f} ({'High' if std_dev < 0.01 else 'Medium' if std_dev < 0.02 else 'Low'} consensus)")

            # --- Update stock_mod_df with the Ensemble Forecast ---

            # Update the 1D column with the calculated value
            stock_mod_df.loc[len(stock_mod_df)-1, "1D"] = forecast_price_change

            # Update the close_Price column with the calculated value
            prev_price = stock_mod_df.loc[len(stock_mod_df)-2, "close_Price"]
            new_price = prev_price * (1 + forecast_price_change)
            stock_mod_df.loc[len(stock_mod_df)-1, "close_Price"] = new_price
            recent_prediction_values.append(forecast_price_change)

            # Store uncertainty data for this prediction day
            uncertainty_data.append({
                'date': stock_mod_df.loc[len(stock_mod_df)-1, "date"],
                'predicted_price': new_price,
                'prediction_std': ensemble_std if 'ensemble_std' in dir() else forecast_seq_std,
                'lower_95': new_price * (1 + forecast_seq_ci_low) if forecast_seq_ci_low != forecast_seq else None,
                'lower_68': new_price * (1 + np.percentile([forecast_seq, forecast_rf] + ([forecast_xgb] if xgb_model is not None else []), 16) - forecast_price_change) if use_mc_dropout else None,
                'upper_68': new_price * (1 + np.percentile([forecast_seq, forecast_rf] + ([forecast_xgb] if xgb_model is not None else []), 84) - forecast_price_change) if use_mc_dropout else None,
                'upper_95': new_price * (1 + forecast_seq_ci_high) if forecast_seq_ci_high != forecast_seq else None,
            })

            # Store per-model predicted prices for analysis
            per_model_entry = {
                'date': stock_mod_df.loc[len(stock_mod_df)-1, "date"],
                'price_rf': prev_price * (1 + forecast_rf),
                'price_xgb': prev_price * (1 + forecast_xgb) if xgb_model is not None else None,
                'price_ridge': prev_price * (1 + forecast_ridge) if ridge_model is not None else None,
                'price_svr': prev_price * (1 + forecast_svr) if svr_model is not None else None,
                'price_ensemble': new_price,
            }
            if sequence_model is not None:
                per_model_entry['price_seq'] = prev_price * (1 + forecast_seq)
            per_model_predictions.append(per_model_entry)
            # print("Columns in stock_mod_df:", stock_mod_df.columns.tolist())
            # print("stock_mod_df after prediction:\n", stock_mod_df[["date", "close_Price"]].tail(3))

        # --- Final cleanup ---
        columns_to_convert = stock_mod_df.columns.drop(["date", "ticker", "currency"], errors='ignore').to_list()
        for column in columns_to_convert:
            stock_mod_df[column] = pd.to_numeric(stock_mod_df[column], errors='coerce').fillna(0)

        stock_mod_df = stock_mod_df[features_list].copy()
        stock_mod_df['predicted_close_Price'] = stock_mod_df['close_Price']

        if historical_price_predictions:
            historical_price_df = pd.DataFrame(historical_price_predictions)
            historical_price_df['date'] = historical_price_df['date'].astype(str)
            stock_mod_df['date'] = stock_mod_df['date'].astype(str)
            for _, hrow in historical_price_df.iterrows():
                mask = stock_mod_df['date'] == hrow['date']
                if mask.any():
                    stock_mod_df.loc[mask, 'predicted_close_Price'] = hrow['predicted_close_Price']
        
        # Merge uncertainty data into the returned DataFrame
        if uncertainty_data:
            uncertainty_df = pd.DataFrame(uncertainty_data)
            uncertainty_df['date'] = uncertainty_df['date'].astype(str)
            stock_mod_df['date'] = stock_mod_df['date'].astype(str)
            
            # Add uncertainty columns (only for future prediction rows)
            stock_mod_df[['prediction_std', 'lower_95', 'lower_68', 'upper_68', 'upper_95']] = None
            
            # Match by date and fill in uncertainty values
            for _, urow in uncertainty_df.iterrows():
                mask = stock_mod_df['date'] == urow['date']
                if mask.any():
                    stock_mod_df.loc[mask, 'prediction_std'] = urow['prediction_std']
                    stock_mod_df.loc[mask, 'lower_95'] = urow['lower_95']
                    stock_mod_df.loc[mask, 'lower_68'] = urow['lower_68']
                    stock_mod_df.loc[mask, 'upper_68'] = urow['upper_68']
                    stock_mod_df.loc[mask, 'upper_95'] = urow['upper_95']
            
            # Also add a 'std' column alias for export_stock_prediction_extended compatibility
            stock_mod_df['std'] = stock_mod_df['prediction_std']
        
        # Merge per-model predictions into the returned DataFrame
        if per_model_predictions:
            pm_df = pd.DataFrame(per_model_predictions)
            pm_df['date'] = pm_df['date'].astype(str)
            model_price_cols = [c for c in pm_df.columns if c.startswith('price_') and c != 'price_ensemble']
            stock_mod_df[model_price_cols] = None
            for _, prow in pm_df.iterrows():
                mask = stock_mod_df['date'] == prow['date']
                if mask.any():
                    for col in model_price_cols:
                        stock_mod_df.loc[mask, col] = prow[col]
        
        return stock_mod_df

    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        raise ValueError("The prediction could not be completed. Please check the input data.") from e

def analyze_prediction_performance(stock_df, forecast_df, historical_prediction_count):
    """
    Analyzes and compares model predictions vs actual values for historical data.
    
    Parameters:
    - stock_df: Original stock data with actual values
    - forecast_df: Predictions from models
    - historical_prediction_count: Number of historical predictions made
    
    Returns:
    None (prints analysis)
    """
    print("\n" + "="*80)
    print("[ANALYSIS] PREDICTION PERFORMANCE")
    print("="*80)
    
    if historical_prediction_count > 0:
        # Get only the historical forecast period, excluding appended future rows.
        forecast_dates = pd.to_datetime(forecast_df["date"], errors='coerce')
        last_actual_date = pd.to_datetime(stock_df["date"], errors='coerce').max()
        hist_forecast = forecast_df.loc[forecast_dates <= last_actual_date].iloc[-historical_prediction_count:]
        hist_actual = stock_df.iloc[-historical_prediction_count:]
        
        # Compare predicted vs actual price changes
        if "1D" in hist_actual.columns and "1D" in hist_forecast.columns:
            predicted_changes = hist_forecast["1D"].values
            actual_changes = hist_actual["1D"].values
            
            # Calculate errors
            errors = predicted_changes - actual_changes
            abs_errors = np.abs(errors)
            pct_errors = abs_errors * 100  # Convert to percentage points
            
            print(f"\n[DATA] Historical Prediction Accuracy (Last {historical_prediction_count} days):")
            print(f"   Mean Absolute Error:    {np.mean(abs_errors):.6f} ({np.mean(pct_errors):.3f}%)")
            print(f"   Median Absolute Error:  {np.median(abs_errors):.6f} ({np.median(pct_errors):.3f}%)")
            print(f"   Std Dev of Errors:      {np.std(errors):.6f}")
            print(f"   Max Error:              {np.max(abs_errors):.6f} ({np.max(pct_errors):.3f}%)")
            print(f"   Min Error:              {np.min(abs_errors):.6f} ({np.min(pct_errors):.3f}%)")
            
            # Direction accuracy (did we predict up/down correctly?)
            predicted_direction = np.sign(predicted_changes)
            actual_direction = np.sign(actual_changes)
            direction_accuracy = np.mean(predicted_direction == actual_direction) * 100
            print(f"   Direction Accuracy:     {direction_accuracy:.2f}%")
            
            # Compare actual vs predicted prices
            predicted_price_col = "predicted_close_Price" if "predicted_close_Price" in hist_forecast.columns else "close_Price"
            if predicted_price_col in hist_forecast.columns and "close_Price" in hist_actual.columns:
                # Ensure we get 1D arrays by using .squeeze() to remove extra dimensions
                predicted_prices = hist_forecast[predicted_price_col].values
                actual_prices = hist_actual["close_Price"].values
                
                # Handle case where values might be 2D (e.g., shape (50,2) instead of (50,))
                if len(predicted_prices.shape) > 1:
                    predicted_prices = predicted_prices.flatten()
                if len(actual_prices.shape) > 1:
                    actual_prices = actual_prices.flatten()
                
                # Ensure both arrays have the same length
                min_len = min(len(predicted_prices), len(actual_prices))
                predicted_prices = predicted_prices[:min_len]
                actual_prices = actual_prices[:min_len]
                
                price_errors = predicted_prices - actual_prices
                price_pct_errors = (price_errors / actual_prices) * 100
                
                print("\n[PRICE] Price Prediction Accuracy:")
                print(f"   Mean Price Error:       {np.mean(np.abs(price_errors)):.2f} ({np.mean(np.abs(price_pct_errors)):.2f}%)")
                print(f"   Median Price Error:     {np.median(np.abs(price_errors)):.2f} ({np.median(np.abs(price_pct_errors)):.2f}%)")
                print(f"   Max Price Error:        {np.max(np.abs(price_errors)):.2f} ({np.max(np.abs(price_pct_errors)):.2f}%)")
                
                # Show day-by-day comparison
                print("\n[TABLE] Day-by-Day Comparison (Historical):")
                print(f"{'Date':<12} {'Actual Price':>12} {'Pred Price':>12} {'Error':>10} {'Actual %':>10} {'Pred %':>10}")
                print("-" * 78)
                # Use min_len to ensure we don't exceed array bounds
                for i in range(min_len):
                    date = hist_actual.iloc[i]["date"].strftime('%Y-%m-%d') if pd.notna(hist_actual.iloc[i]["date"]) else "N/A"
                    act_price = actual_prices[i]
                    pred_price = predicted_prices[i]
                    price_err = pred_price - act_price
                    act_change = actual_changes[i] * 100
                    pred_change = predicted_changes[i] * 100
                    print(f"{date:<12} {act_price:>12.2f} {pred_price:>12.2f} {price_err:>+10.2f} {act_change:>+9.2f}% {pred_change:>+9.2f}%")
    
    print("\n" + "="*80)

def calculate_predicted_profit(forecast_df, prediction_days):
    """
    Predicts the stock price using different models.

    Parameters:
    - traning_dataset (pandas.DataFrame): A DataFrame containing the traning dataset.
    - test_dataset (pandas.DataFrame): A DataFrame containing the test dataset.
    - prediction_dataset (pandas.DataFrame): A DataFrame containing the prediction dataset.
    - stock_df (pandas.DataFrame): A DataFrame containing the stock data.

    Returns:
    pandas.DataFrame: A DataFrame containing the predicted stock prices.

    Raises:
    ValueError: If the prediction could not be completed.
    """

    try:
        predicted_return_df = forecast_df.loc[len(forecast_df)-prediction_days:, "close_Price"]
        predicted_return = ((predicted_return_df.iloc[-1] / predicted_return_df.iloc[0]) - 1) * 100
        if predicted_return > 0:
            print(f"The prediction expects a profitable return on: {round(predicted_return, 2)}%, over the next {prediction_days} days.")
        elif predicted_return < 0:
            print(f"The prediction expects a loss of: {round(predicted_return, 2)}%, over the next {prediction_days} days.")
        else:
            print(f"The prediction expects no return over the next {prediction_days} days.")
        return float(predicted_return)
    except ValueError:
        print(f"The model could not predict an expected return over the next {prediction_days} days.")
        return None

def plot_graph(stock_data_df, forecast_data_df):
    """
    Plots a graph of the stock data.

    Parameters:
    - stock_data_df (pandas.DataFrame): A DataFrame containing the actual stock data.
    - forecast_data_df (pandas.DataFrame): A DataFrame containing the predicted stock data.

    Returns:
    None

    Raises:
    - FileNotFoundError: If the graph could not be saved.
    """

    print(stock_data_df["close_Price"])
    print(forecast_data_df["close_Price"])
    # Plot the graph
    plt.figure(figsize=(18, 8))
    stock_data_df = stock_data_df.copy()
    forecast_data_df = forecast_data_df.copy()
    stock_data_df["date"] = stock_data_df["date"].astype('datetime64[ns]')
    forecast_data_df["date"] = forecast_data_df["date"].astype('datetime64[ns]')

    # Only show predictions from the last actual data date onward
    last_actual_date = stock_data_df["date"].max()
    forecast_data_df = forecast_data_df.loc[forecast_data_df["date"] >= last_actual_date]

    stock_data_df = stock_data_df.set_index("date")
    forecast_data_df = forecast_data_df.set_index("date")

    # Plot actual stock price as the primary/main line
    stock_data_df["close_Price"].plot(linewidth=1.5, color='tab:blue')
    # Plot predictions only from forecast start date
    forecast_data_df["close_Price"].plot(linewidth=1.5, color='tab:orange', linestyle='--')

    legend_list = ["Stock Price", "Predicted Stock Price"]
    plt.legend(legend_list,
        loc="best"
    )
    plt.xlabel("Date")
    plt.ylabel("Closing price")
    plt.title(f"Stock Price Prediction of {stock_data_df.iloc[0]['ticker']}")
    # Change " " in stock_data_df.iloc[0]["Name"] to "_" to avoid error when saving the graph
    stock_data_df = stock_data_df.replace({"ticker": [" ", "/"]}, {"ticker": "_"}, regex=True)
    stock_name = stock_data_df.iloc[0]["ticker"]
    graph_name = f"stock_prediction_of_{stock_name}.png"
    my_path = os.path.abspath(__file__)
    path = os.path.dirname(my_path)
    # Save the graph
    try:
        plt.savefig(os.path.join(path, "generated_graphs", graph_name), bbox_inches="tight", pad_inches=0.5, transparent=False, format="png")
        plt.clf()
        plt.close("all")

    except FileNotFoundError:
        plt.close("all")

# Run the main function
def main():
    """Main function to run the stock price prediction pipeline."""
    import db_interactions

    configure_tensorflow_gpu(memory_limit_mb=7168)

    start_time = time.time()

    # Import stock symbols from DB
    stock_symbols_list = db_interactions.import_ticker_list()
    stock_symbol = stock_symbols_list[0]
    stock_symbol = "A"
    print(f"Selected stock symbol: {stock_symbol}")

    # Import stock data
    stock_data_df = db_interactions.import_stock_dataset(stock_symbol)
    stock_data_df["date"] = pd.to_datetime(stock_data_df["date"])

    # --- Targeted cleaning instead of blanket dropna ---
    stock_data_df = stock_data_df.dropna(axis=1, how="all")
    critical_cols = ["date", "ticker"]
    price_cols = [c for c in ["close_Price", "open_Price", "high_Price", "low_Price"] if c in stock_data_df.columns]
    critical_cols.extend(price_cols)
    stock_data_df = stock_data_df.dropna(subset=critical_cols)
    feature_cols = [c for c in stock_data_df.columns if c not in critical_cols]
    if feature_cols:
        stock_data_df[feature_cols] = stock_data_df[feature_cols].ffill().bfill()
    stock_data_df = stock_data_df.dropna(axis=0, how="any")
    stock_data_df = stock_data_df.dropna(axis=1, how="any")

    # Split the dataset into training, validation, test data and prediction data
    validation_size = 0.20
    test_size = 0.10
    scaler_x, scaler_y, x_train_scaled, x_val_scaled, x_test_scaled, \
        y_train_scaled, y_val_scaled, y_test_scaled, x_predictions = \
        split_dataset.dataset_train_test_split(
            stock_data_df, test_size, validation_size=validation_size
        )

    # Inverse-transform y values for Random Forest (RF is scale-invariant)
    y_train_unscaled = scaler_y.inverse_transform(
        y_train_scaled.reshape(-1, 1)
    ).flatten()
    y_val_unscaled = scaler_y.inverse_transform(
        y_val_scaled.reshape(-1, 1)
    ).flatten()
    y_test_unscaled = scaler_y.inverse_transform(
        y_test_scaled.reshape(-1, 1)
    ).flatten()

    # Convert to DataFrames for feature selection
    x_training_data = pd.DataFrame(x_train_scaled)
    x_val_data = pd.DataFrame(x_val_scaled)
    x_test_data = pd.DataFrame(x_test_scaled)
    y_training_data_df = pd.Series(y_train_unscaled)
    y_val_data_df = pd.Series(y_val_unscaled)
    y_test_data_df = pd.Series(y_test_unscaled)
    prediction_data = x_predictions

    max_features = len(x_training_data.columns)
    print(f"Max features:\n{max_features}")
    feature_amount = max_features

    x_training_dataset, x_val_dataset, x_test_dataset, x_prediction_dataset, \
        _, selected_features_list = \
        dimension_reduction.feature_selection_rf(
            feature_amount,
            x_training_data,
            x_val_data,
            x_test_data,
            y_training_data_df,
            y_val_data_df,
            y_test_data_df,
            prediction_data,
            stock_data_df
        )

    x_training_dataset_df = pd.DataFrame(
        x_training_dataset, columns=selected_features_list
    )
    y_training_data_df = y_training_data_df.reset_index(drop=True)
    x_val_dataset_df = pd.DataFrame(x_val_dataset, columns=selected_features_list)
    y_val_data_df = y_val_data_df.reset_index(drop=True)
    x_test_dataset_df = pd.DataFrame(x_test_dataset, columns=selected_features_list)
    y_test_data_df = y_test_data_df.reset_index(drop=True)
    x_prediction_dataset_df = pd.DataFrame(
        x_prediction_dataset, columns=selected_features_list
    )

    time_steps = 30

    y_train_scaled_for_lstm = pd.Series(y_train_scaled)
    y_test_scaled_for_lstm = pd.Series(y_test_scaled)
    y_val_scaled_for_lstm = pd.Series(y_val_scaled)

    # MAIN TRAINING WITH AUTOMATIC RETRAINING AND OVERFITTING DETECTION
    models, training_history, _ = train_and_validate_models(
        stock_symbol=stock_symbol,
        x_train=x_training_dataset_df.values,
        x_val=x_val_dataset_df.values,
        x_test=x_test_dataset_df.values,
        y_train_scaled=y_train_scaled_for_lstm.values,
        y_val_scaled=y_val_scaled_for_lstm.values,
        y_test_scaled=y_test_scaled_for_lstm.values,
        y_train_unscaled=y_train_unscaled,
        y_val_unscaled=y_val_unscaled,
        y_test_unscaled=y_test_unscaled,
        time_steps=time_steps,
        scaler_y=scaler_y,
        max_retrains=150,
        overfitting_threshold=0.15,
        lstm_trials=50,
        lstm_executions=10,
        lstm_epochs=500,
        lstm_retrain_trials_increment=10,
        lstm_retrain_executions_increment=2,
        rf_trials=100,
        rf_retrain_increment=25,
        xgb_trials=60,
        xgb_retrain_increment=10,
        use_multi_metric_detection=True,
        # TCN Configuration (enabled by default - replaces mode-collapsed LSTM)
        use_tcn=True,
        tcn_trials=30,
        tcn_epochs=100,
        tcn_retrain_increment=10
    )
    
    # Extract models using new generic interface
    sequence_model_type = models['sequence_model_type']  # 'tcn' or 'lstm'

    # Print training history summary
    print("\n" + "="*60)
    print("[DATA] COMPLETE TRAINING HISTORY")
    print("="*60)
    seq_model_key = 'tcn' if sequence_model_type == 'tcn' else 'lstm'
    for i, seq_history in enumerate(training_history[seq_model_key]):
        print(f"\n{sequence_model_type.upper()} Attempt {i+1}:")
        print(f"  Test MSE: {seq_history['test_metrics']['mse']:.6f}")
        print(f"  Test R2:  {seq_history['test_metrics']['r2']:.4f}")

    for i, _ in enumerate(training_history['random_forest']):
        print(f"\nRandom Forest Attempt {i+1}:")

    # Predict the future stock price changes with Monte Carlo Dropout uncertainty
    amount_of_days = time_steps * 3
    forecast_df = predict_future_price_changes(
        ticker=stock_symbol,
        scaler_x=scaler_x,
        scaler_y=scaler_y,
        model=models,  # Pass the full models dict which includes sequence_model, rf, xgb
        selected_features_list=selected_features_list,
        stock_df=stock_data_df,
        prediction_days=amount_of_days,
        time_steps=time_steps,
        historical_prediction_dataset_df=x_prediction_dataset_df,
        # Monte Carlo Dropout settings for uncertainty estimation
        use_mc_dropout=True,
        mc_iterations=30  # Number of forward passes per prediction
    )

    print("Forecast DataFrame:")
    print(forecast_df)
    print(forecast_df.columns.tolist())

    # Analyze prediction performance
    historical_pred_count = len(x_prediction_dataset_df) \
        if x_prediction_dataset_df is not None else 0
    analyze_prediction_performance(stock_data_df, forecast_df, historical_pred_count)

    plt.plot(forecast_df["close_Price"], color="green")
    plt.xlabel("Date")
    plt.ylabel("Opening price")
    legend_list = ["Predicted Stock Price"]
    plt.legend(legend_list, loc="best")
    stock_name = stock_data_df.iloc[0]["ticker"]
    graph_name = f"future_stock_prediction_of_{stock_name}.png"
    my_path = os.path.abspath(__file__)
    path = os.path.dirname(my_path)
    try:
        plt.savefig(
            os.path.join(path, "generated_graphs", graph_name),
            bbox_inches="tight", pad_inches=0.5, transparent=False, format="png"
        )
        plt.clf()
        plt.close("all")
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "The graph could not be saved. Please check the file name or path."
        ) from e

    # Calculate the predicted profit
    calculate_predicted_profit(forecast_df, amount_of_days)

    # Export the forecast to excel file
    forecast_file_name = f"forecast_{stock_symbol}.xlsx"
    my_path = os.path.abspath(__file__)
    forecast_file_path = os.path.join(
        os.path.dirname(my_path), "generated_forecasts", forecast_file_name
    )
    forecast_df.to_excel(forecast_file_path, index=False)

    # Plot the graph
    plot_graph(stock_data_df, forecast_df)

    # Run a Monte Carlo simulation
    year_amount = 10
    sim_amount = 1000
    _, _ = monte_carlo_sim.monte_carlo_analysis(
        0, stock_data_df, forecast_df, year_amount, sim_amount
    )
    forecast_df = forecast_df.rename(columns={"close_Price": stock_symbol + "_price"})

    # Calculate the execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"[TIME] Total execution time: {execution_time:.2f} seconds")


if __name__ == "__main__":
    main()
