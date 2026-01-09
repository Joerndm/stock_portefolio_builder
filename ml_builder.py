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
import time
import datetime
import math
import tempfile
import shutil

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
from sklearn import tree
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import RepeatedKFold
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers
import keras
import keras_tuner as kt
from keras_tuner.tuners import Sklearn

matplotlib.use('Agg')
pd.set_option('future.no_silent_downcasting', True)

import split_dataset
import dimension_reduction
import monte_carlo_sim

# Force UTF-8 logging globally
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="ignore")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="ignore")

def build_random_forest_model(hp, constrain_for_overfitting=False):
    """
    Builds a RandomForestRegressor with expanded tunable hyperparameters.
    
    Parameters:
    - hp: Keras Tuner hyperparameters object
    - constrain_for_overfitting: If True, apply stricter constraints to reduce overfitting
    """
    
    max_features_choice = hp.Choice('max_features', ['sqrt', 'log2', '0.3', '0.5', '0.8'])
    
    if max_features_choice in ['0.3', '0.5', '0.8']:
        max_features_value = float(max_features_choice)
    else:
        max_features_value = max_features_choice
    
    if constrain_for_overfitting:
        # Stricter constraints when overfitting is detected
        bootstrap = True  # Force bootstrap for better generalization
        max_samples_value = hp.Float('max_samples', 0.6, 0.9, step=0.1)  # Reduced range
        max_depth_value = hp.Int('max_depth', 3, 30, step=2)  # Lower ceiling
        min_samples_leaf_value = hp.Choice('min_samples_leaf', [2, 4, 8, 16])  # Higher floor
        min_samples_split_value = hp.Choice('min_samples_split', [5, 10, 15, 20])  # Higher floor
    else:
        # Standard search space
        bootstrap = hp.Boolean('bootstrap')
        if bootstrap:
            max_samples_value = hp.Float('max_samples', 0.5, 1.0, step=0.1)
        else:
            max_samples_value = None
        max_depth_value = hp.Int('max_depth', 3, 50, step=2)
        min_samples_leaf_value = hp.Choice('min_samples_leaf', [1, 2, 4, 8, 16])
        min_samples_split_value = hp.Choice('min_samples_split', [2, 5, 10, 15])

    model = RandomForestRegressor(
        n_estimators=hp.Int('n_estimators', 100, 1500, step=100),
        max_depth=max_depth_value,
        min_samples_split=min_samples_split_value,
        min_samples_leaf=min_samples_leaf_value,
        criterion=hp.Choice('criterion', ['squared_error', 'absolute_error', 'friedman_mse']),
        bootstrap=bootstrap,
        max_features=max_features_value,
        max_samples=max_samples_value,
        random_state=42,
        n_jobs=-1
    )
    return model

def tune_random_forest_model(stock_symbol, x_training_dataset_df, y_training_dataset_df, x_val_dataset_df, y_val_dataset_df, max_trials=20, constrain_for_overfitting=False):
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
    
    Returns:
    - best_rf_model: The tuned Random Forest model
    """
    
    # Convert to numpy arrays to avoid feature name warnings
    x_train = x_training_dataset_df.values
    y_train = y_training_dataset_df.values
    x_val = x_val_dataset_df.values
    y_val = y_val_dataset_df.values

    # Define the MSE scorer
    mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

    # Setup directory structure
    overwrite_val = False
    directory_val = "tuning_dir"
    project_name_val = f"RF_tuning_{stock_symbol}"
    project_path = os.path.join(directory_val, project_name_val)

    # Cleanup old tuning directory if needed
    if os.path.exists(project_path):
        if overwrite_val:
            try:
                shutil.rmtree(project_path)
                print(f"🗑️  Deleted old tuning directory: {project_path}")
            except Exception as e:
                print(f"⚠️  Warning: Could not delete old tuning directory {project_path}.Error: {e}")

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
        directory=directory_val,
        project_name=project_name_val,
        overwrite=overwrite_val
    )

    # Search for best hyperparameters using train+val with predefined split
    print(f"🔍 Starting Random Forest hyperparameter tuning for {stock_symbol}...")
    tuner.search(x_combined, y_combined)

    # Get best hyperparameters
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Build and train best model on TRAINING DATA ONLY
    best_rf_model = tuner.hypermodel.build(best_hp)
    best_rf_model.fit(x_train, y_train)

    # Print best hyperparameters
    print("\n🌳 Best Random Forest hyperparameters found:")
    for param, value in best_hp.values.items():
        print(f"  ✓ {param}: {value}")

    # Feature importance logging
    importances = best_rf_model.feature_importances_
    feature_names = x_training_dataset_df.columns

    print("\n📊 Top 10 Feature Importances:")
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    for idx, row in feature_importance_df.head(10).iterrows():
        print(f"  • {row['feature']}: {row['importance']:.4f}")

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
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("XGBoost not installed. Install with: pip install xgboost")
    
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

def tune_xgboost_model(stock_symbol, x_training_dataset_df, y_training_dataset_df, x_val_dataset_df, y_val_dataset_df, max_trials=30, constrain_for_overfitting=False):
    """
    Tunes XGBoost model using validation set for hyperparameter selection.
    
    Parameters:
    - stock_symbol (str): The stock ticker symbol
    - x_training_dataset_df (pd.DataFrame): Training dataset (ALREADY SCALED)
    - y_training_dataset_df (pd.Series or np.ndarray): Training labels (UNSCALED for XGBoost)
    - x_val_dataset_df (pd.DataFrame): Validation dataset (ALREADY SCALED)
    - y_val_dataset_df (pd.Series or np.ndarray): Validation labels (UNSCALED for XGBoost)
    - max_trials (int): Maximum number of tuning trials
    
    Returns:
    - best_xgb_model: The tuned XGBoost model
    """
    
    # Convert to numpy arrays
    x_train = x_training_dataset_df.values
    y_train = y_training_dataset_df.values
    x_val = x_val_dataset_df.values
    y_val = y_val_dataset_df.values

    # Define the MSE scorer
    mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

    # Setup directory structure
    overwrite_val = False
    directory_val = "tuning_dir"
    project_name_val = f"XGB_tuning_{stock_symbol}"
    project_path = os.path.join(directory_val, project_name_val)

    # Cleanup old tuning directory if needed
    if os.path.exists(project_path):
        if overwrite_val:
            try:
                shutil.rmtree(project_path)
                print(f"🗑️  Deleted old tuning directory: {project_path}")
            except Exception as e:
                print(f"⚠️  Warning: Could not delete old tuning directory {project_path}. Error: {e}")

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
        directory=directory_val,
        project_name=project_name_val,
        overwrite=overwrite_val
    )

    # Search for best hyperparameters
    print(f"🔍 Starting XGBoost hyperparameter tuning for {stock_symbol}...")
    tuner.search(x_combined, y_combined)

    # Get best hyperparameters
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Build and train best model on TRAINING DATA ONLY
    best_xgb_model = tuner.hypermodel.build(best_hp)
    best_xgb_model.fit(x_train, y_train)

    # Print best hyperparameters
    print("\n🚀 Best XGBoost hyperparameters found:")
    for param, value in best_hp.values.items():
        print(f"  ✓ {param}: {value}")

    # Feature importance logging
    importances = best_xgb_model.feature_importances_
    feature_names = x_training_dataset_df.columns

    print("\n📊 Top 10 Feature Importances (XGBoost):")
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    for idx, row in feature_importance_df.head(10).iterrows():
        print(f"  • {row['feature']}: {row['importance']:.4f}")

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

def build_svm_model(hp):
    """
    Builds an SVR model with tunable hyperparameters.

    Parameters:
    - hp: Keras Tuner hyperparameters object

    Returns:
    - model: Configured SVR model
    """
    kernel_choice = hp.Choice('kernel', ['linear', 'rbf', 'poly'])
    model = SVR(
        kernel=kernel_choice,
        C=hp.Float('C', 1e-3, 1e3, sampling='log'),
        gamma=hp.Float('gamma', 1e-4, 1e1, sampling='log'),
        degree=hp.Int('degree', 2, 5) if kernel_choice == 'poly' else 3
    )
    return model

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

    print("⚖️ Best SVM hyperparameters found:")
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
    - np.array: 3D array of shape (samples - time_steps + 1, time_steps, features)
    
    Raises:
    - ValueError: If data is insufficient for sequence creation
    
    Example:
    >>> data = np.array([[1,2], [3,4], [5,6], [7,8], [9,10]])
    >>> sequences = create_sequences(data, time_steps=3)
    >>> sequences.shape
    (3, 3, 2)
    """
    if len(data) < time_steps:
        raise ValueError(
            f"Insufficient data for sequence creation. "
            f"Need at least {time_steps} samples, got {len(data)}"
        )
    
    Xs = []
    for i in range(len(data) - time_steps + 1):
        Xs.append(data[i:(i + time_steps)])
    
    result = np.array(Xs)
    
    # Validate output shape
    expected_shape = (len(data) - time_steps + 1, time_steps, data.shape[1])
    if result.shape != expected_shape:
        raise ValueError(
            f"Sequence creation failed. Expected shape {expected_shape}, got {result.shape}"
        )
    
    return result

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
    x_train_lstm = create_sequences(x_train, time_steps)
    y_train_lstm = y_train[time_steps-1:].reshape(-1, 1)
    
    x_val_lstm = create_sequences(x_val, time_steps)
    y_val_lstm = y_val[time_steps-1:].reshape(-1, 1)
    
    x_test_lstm = create_sequences(x_test, time_steps)
    y_test_lstm = y_test[time_steps-1:].reshape(-1, 1)
    
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
    for i in range(1, hp.Int('n_layers', 1, max_amount_layers)):
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
                    f"final_units",
                    min_value=32,
                    max_value=512,
                    step=16
                ),
                return_sequences=False,
                kernel_regularizer=regularizers.l2(
                    hp.Float(
                        f"l2_reg_final",
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
            activation=hp.Choice(
                "dense_1_activation",
                ["relu", "tanh"]
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
            activation=hp.Choice(
                "dense_2_activation",
                ["relu", "sigmoid"]
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
    optimizer_choice = hp.Choice("optimizer", ["adam", "rmsprop"])
    learning_rate = hp.Float("learning_rate", 1e-5, 1e-3, sampling="log")
    clipnorm = hp.Float("clipnorm", 0.5, 2.0, step=0.5)

    if optimizer_choice == "adam":
        optimizer = Adam(learning_rate=learning_rate, clipnorm=clipnorm)
    else:
        optimizer = RMSprop(learning_rate=learning_rate, clipnorm=clipnorm)

    # Loss choice - MAE added to reduce mode collapse risk (less sensitive to outliers than MSE)
    loss_choice = hp.Choice(
        "loss",
        ["mean_absolute_error", "mean_squared_error", "huber", "mean_absolute_percentage_error"]
    )

    # Optimizer with tunable learning rate
    model.compile(
        optimizer=optimizer,
        loss=loss_choice,
        metrics=["mean_absolute_error", "mean_squared_error", "accuracy"]
    )
    return model

def tune_lstm_model(stock, x_train_lstm, y_train_lstm, x_val_lstm, y_val_lstm, time_steps, num_features, max_trials=25, executions_per_trial=1, epochs=50, retries=3, delay=5):
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
    
    Returns:
    - best_model: Tuned Keras model
    
    Raises:
    - RuntimeError: If tuning fails after all retries
    """

    # Validate input shapes
    expected_shape = (x_train_lstm.shape[0], time_steps, num_features)
    if x_train_lstm.shape != expected_shape:
        raise ValueError(f"x_train_lstm shape mismatch. Expected {expected_shape}, got {x_train_lstm.shape}")
    
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
        print(f"✅ Loaded best model from finished tuning: {finished_project_path}")
        return best_model

    # --- PRIORITY 2: Check for partial tuning and continue ---
    overwrite_val = False
    if os.path.exists(temp_project_path):
        print(f"⏸️  Found partial tuning at {temp_project_path}. Continuing tuning...")
        overwrite_val = False  # Continue from existing
    else:
        print("🆕 No partial tuning found. Starting new tuning...")
        overwrite_val = True

    # Define tuner with HyperModel class (FIXED APPROACH)
    class LSTMHyperModel(kt.HyperModel):
        def __init__(self, input_shape):
            self.input_shape = input_shape

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
                from keras.callbacks import LearningRateScheduler
                initial_lr = hp.get("learning_rate")
                decay_rate = hp.Float("decay_rate", 0.9, 0.99, step=0.01)

                def lr_schedule(epoch, lr):
                    return initial_lr * (decay_rate ** epoch)

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
            print(f"⚠️ OOM Error on attempt {attempt+1}: {oom_error}")
            print("🧹 Clearing GPU memory and skipping failed trial...")

            tf.keras.backend.clear_session()
            import gc
            gc.collect()

            if attempt < retries - 1:
                print(f"⏳ Waiting {delay} seconds before retry...")
                time.sleep(delay)
            else:
                raise RuntimeError("LSTM tuning failed after all retries due to OOM errors")

        except (UnicodeDecodeError, tf.errors.FailedPreconditionError, tf.errors.InternalError) as e:
            print(f"Attempt {attempt+1} failed: {e}")
            if os.path.exists(temp_project_path):
                shutil.rmtree(temp_project_path)
            if attempt < retries - 1:
                print(f"⏳ Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                raise RuntimeError(f"LSTM tuning failed after {retries} attempts: {e}")

    # Get best model
    best_trials = tuner.oracle.get_best_trials(num_trials=1)
    if not best_trials:
        raise RuntimeError("Keras Tuner failed to find any successful trials.")

    best_hp = best_trials[0].hyperparameters

    print("✅ Best hyperparameters found:")
    for param, value in best_hp.values.items():
        print(f"- {param}: {value}")

    best_model = build_lstm_model(
        best_hp,
        input_shape=(time_steps, num_features)
    )

    best_model.build(input_shape=(None, time_steps, num_features))
    print("Best model architecture:")
    print(best_model.summary())

    # Move tuning folder to script directory
    final_dest = os.path.join(script_dir, f"tuning_dir/{project_name_val}")
    print(f"Moving tuning folder to: {final_dest}")
    if os.path.exists(final_dest):
        shutil.rmtree(final_dest)

    shutil.move(temp_project_path, final_dest)

    # Cleanup temp dir
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            print("🧹 Cleaned up local temp directory.")
        except Exception as e:
            print(f"Warning: Could not delete temp directory {temp_dir}. Error: {e}")

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
        print(f"ℹ️  No finished tuning found at: {finished_project_path}")
        return None

    try:
        # Look for the best model file
        oracle_path = os.path.join(finished_project_path, "oracle.json")

        if not os.path.exists(oracle_path):
            print(f"⚠️  oracle.json not found in {finished_project_path}")
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
            print(f"⚠️  No successful trials found in {finished_project_path}")
            return None

        best_hp = best_trials[0].hyperparameters

        # Rebuild the model with best hyperparameters
        best_model = build_lstm_model(best_hp, input_shape=(time_steps, num_features))
        best_model.build(input_shape=(None, time_steps, num_features))

        print(f"✅ Successfully loaded best model from finished tuning:")
        print(f"   Path: {finished_project_path}")
        print(f"   Best hyperparameters:")
        for param, value in best_hp.values.items():
            print(f"     • {param}: {value}")

        return best_model

    except Exception as e:
        print(f"⚠️  Failed to load model from {finished_project_path}")
        print(f"   Error: {e}")
        print(f"   Will start new tuning instead.")
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
    """

    if use_multi_metric:
        # ===== MULTI-METRIC OVERFITTING DETECTION =====
        
        # 1. MSE Degradation (lower is better)
        train_val_mse_ratio = (val_metrics['mse'] - train_metrics['mse']) / train_metrics['mse']
        val_test_mse_ratio = (test_metrics['mse'] - val_metrics['mse']) / val_metrics['mse']
        mse_score = max(train_val_mse_ratio, val_test_mse_ratio)
        
        # 2. R² Degradation (higher is better, so invert the logic)
        train_val_r2_ratio = (train_metrics['r2'] - val_metrics['r2']) / max(abs(train_metrics['r2']), 0.01)
        val_test_r2_ratio = (val_metrics['r2'] - test_metrics['r2']) / max(abs(val_metrics['r2']), 0.01)
        r2_score = max(train_val_r2_ratio, val_test_r2_ratio)
        
        # 3. MAE Degradation (lower is better)
        train_val_mae_ratio = (val_metrics['mae'] - train_metrics['mae']) / train_metrics['mae']
        val_test_mae_ratio = (test_metrics['mae'] - val_metrics['mae']) / val_metrics['mae']
        mae_score = max(train_val_mae_ratio, val_test_mae_ratio)
        
        # 4. Consistency Score (how aligned are the metrics?)
        # If MSE increases but R² doesn't degrade proportionally, something is off
        metric_scores = [mse_score, r2_score, mae_score]
        consistency_score = np.std(metric_scores) / (np.mean(np.abs(metric_scores)) + 0.01)
        
        # 5. Combined Overfitting Score (weighted average)
        # MSE and MAE are most important, R² secondary, consistency is a tiebreaker
        overfitting_score = (
            0.35 * mse_score + 
            0.25 * r2_score + 
            0.30 * mae_score + 
            0.10 * consistency_score
        )
        
        is_overfitted = overfitting_score > threshold
        
        print(f"\n{'='*60}")
        print(f"🔍 MULTI-METRIC OVERFITTING DETECTION: {model_name}")
        print(f"{'='*60}")
        print(f"METRICS:")
        print(f"  Train:      MSE={train_metrics['mse']:.6f}  R²={train_metrics['r2']:.4f}  MAE={train_metrics['mae']:.6f}")
        print(f"  Validation: MSE={val_metrics['mse']:.6f}  R²={val_metrics['r2']:.4f}  MAE={val_metrics['mae']:.6f}")
        print(f"  Test:       MSE={test_metrics['mse']:.6f}  R²={test_metrics['r2']:.4f}  MAE={test_metrics['mae']:.6f}")
        print(f"{'-'*60}")
        print(f"DEGRADATION ANALYSIS:")
        print(f"  MSE:         Train→Val={train_val_mse_ratio*100:>6.2f}%  Val→Test={val_test_mse_ratio*100:>6.2f}%  Score={mse_score:.4f}")
        print(f"  R²:          Train→Val={train_val_r2_ratio*100:>6.2f}%  Val→Test={val_test_r2_ratio*100:>6.2f}%  Score={r2_score:.4f}")
        print(f"  MAE:         Train→Val={train_val_mae_ratio*100:>6.2f}%  Val→Test={val_test_mae_ratio*100:>6.2f}%  Score={mae_score:.4f}")
        print(f"  Consistency: {consistency_score:.4f}")
        print(f"{'-'*60}")
        print(f"FINAL ASSESSMENT:")
        print(f"  Combined overfitting score: {overfitting_score:.4f}")
        print(f"  Threshold:                  {threshold:.4f}")
        print(f"{'-'*60}")
        
        if is_overfitted:
            print(f"⚠️  OVERFITTING DETECTED! (score: {overfitting_score:.4f} > threshold: {threshold:.4f})")
        else:
            print(f"✅ No overfitting detected (score: {overfitting_score:.4f} ≤ threshold: {threshold:.4f})")
    
    else:
        # ===== LEGACY SINGLE-METRIC DETECTION (MSE only) =====
        train_val_mse_ratio = (val_metrics['mse'] - train_metrics['mse']) / train_metrics['mse']
        val_test_mse_ratio = (test_metrics['mse'] - val_metrics['mse']) / val_metrics['mse']
        overfitting_score = max(train_val_mse_ratio, val_test_mse_ratio)
        is_overfitted = overfitting_score > threshold
        
        print(f"\n{'='*60}")
        print(f"🔍 OVERFITTING DETECTION: {model_name}")
        print(f"{'='*60}")
        print(f"Train MSE:      {train_metrics['mse']:.6f}  |  R²: {train_metrics['r2']:.4f}")
        print(f"Validation MSE: {val_metrics['mse']:.6f}  |  R²: {val_metrics['r2']:.4f}")
        print(f"Test MSE:       {test_metrics['mse']:.6f}  |  R²: {test_metrics['r2']:.4f}")
        print(f"{'-'*60}")
        print(f"Train → Val degradation: {train_val_mse_ratio*100:.2f}%")
        print(f"Val → Test degradation:  {val_test_mse_ratio*100:.2f}%")
        print(f"Overfitting score:       {overfitting_score:.4f}")
        print(f"Threshold:               {threshold:.4f}")
        print(f"{'-'*60}")
        
        if is_overfitted:
            print(f"⚠️  OVERFITTING DETECTED! (score: {overfitting_score:.4f} > threshold: {threshold:.4f})")
        else:
            print(f"✅ No overfitting detected (score: {overfitting_score:.4f} ≤ threshold: {threshold:.4f})")

    print(f"{'='*60}\n")
    return is_overfitted, overfitting_score

def check_data_health(x_train, x_val, x_test, y_train, y_val, y_test, model_name):
    """
    Diagnostic checks for data quality issues that may cause overfitting.
    
    Parameters:
    - x_train, x_val, x_test: Feature arrays
    - y_train, y_val, y_test: Target arrays
    - model_name: Name for logging
    
    Returns:
    - dict with diagnostic results and warnings
    """
    diagnostics = {
        'warnings': [],
        'recommendations': [],
        'pass_diagnostic': True
    }
    
    print(f"\n{'='*60}")
    print(f"🔬 DATA HEALTH CHECK: {model_name}")
    print(f"{'='*60}")
    
    # Check 1: Sample sizes
    train_size = len(x_train)
    val_size = len(x_val)
    test_size = len(x_test)
    feature_count = x_train.shape[1] if len(x_train.shape) > 1 else 1
    
    print(f"📊 Dataset Sizes:")
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
        print(f"   ⚠️  Warning: Only {samples_per_feature:.1f} samples per feature (recommend >10)")
    else:
        print(f"   ✅ Samples per feature: {samples_per_feature:.1f}")
    
    # Check 3: Dataset size balance
    val_ratio = val_size / train_size
    test_ratio = test_size / train_size
    if val_ratio < 0.1 or val_ratio > 0.4:
        diagnostics['warnings'].append(f"Validation set size unusual ({val_ratio*100:.1f}% of train)")
        print(f"   ⚠️  Warning: Val/Train ratio {val_ratio*100:.1f}% (recommend 15-25%)")
    else:
        print(f"   ✅ Val/Train ratio: {val_ratio*100:.1f}%")
    
    # Check 4: Target variance
    y_train_var = np.var(y_train)
    y_val_var = np.var(y_val)
    y_test_var = np.var(y_test)
    
    variance_ratio = max(y_train_var, y_val_var, y_test_var) / (min(y_train_var, y_val_var, y_test_var) + 1e-10)
    
    print(f"\n📈 Target Variance:")
    print(f"   Train: {y_train_var:.6f}")
    print(f"   Val:   {y_val_var:.6f}")
    print(f"   Test:  {y_test_var:.6f}")
    
    if variance_ratio > 10:
        diagnostics['warnings'].append(f"High variance mismatch ({variance_ratio:.1f}x)")
        diagnostics['recommendations'].append("Data splits may not be representative - consider reshuffling")
        print(f"   ⚠️  Warning: {variance_ratio:.1f}x variance difference (may indicate distribution shift)")
    else:
        print(f"   ✅ Variance ratio: {variance_ratio:.1f}x")
    
    # Check 5: Check for extremely small values that might indicate scaling issues
    y_train_mean = np.mean(np.abs(y_train))
    if y_train_mean < 1e-6:
        diagnostics['warnings'].append("Target values extremely small (potential scaling issue)")
        print(f"   ⚠️  Warning: Target mean {y_train_mean:.2e} very small")
    
    print(f"{'='*60}")
    
    if diagnostics['warnings']:
        print(f"\n⚠️  {len(diagnostics['warnings'])} warning(s) detected:")
        for warning in diagnostics['warnings']:
            print(f"   • {warning}")
        if diagnostics['recommendations']:
            print(f"\n💡 Recommendations:")
            for rec in diagnostics['recommendations']:
                print(f"   • {rec}")
    else:
        print(f"\n✅ All diagnostic checks passed")
    
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
            if abs(val1 - val2) > tolerance * max(abs(val1), abs(val2), 1):
                return False
        # Compare non-numeric values exactly
        elif val1 != val2:
            return False
    
    return True

def train_and_validate_models(stock_symbol, x_train, x_val, x_test, y_train_scaled, y_val_scaled, y_test_scaled, y_train_unscaled, y_val_unscaled, y_test_unscaled, time_steps, max_retrains=10, overfitting_threshold=0.15, lstm_trials=25, lstm_executions=1, lstm_epochs=50, rf_trials=50, xgb_trials=30, rf_retrain_increment=25, xgb_retrain_increment=10, lstm_retrain_trials_increment=10, lstm_retrain_executions_increment=2, use_multi_metric_detection=True):
    """
    Train and validate models with automatic retraining if overfitting is detected.
    Includes LSTM, Random Forest, XGBoost, and ensemble predictions.
    
    Parameters:
    - stock_symbol (str): Stock ticker
    - x_train, x_val, x_test: Feature arrays
    - y_train_scaled, y_val_scaled, y_test_scaled: Scaled target arrays
    - y_train_unscaled, y_val_unscaled, y_test_unscaled: Unscaled target arrays
    - time_steps (int): Number of time steps for LSTM sequences
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
    
    Returns:
    - models (dict): Dictionary containing all trained models and ensemble weights
    - training_history: Dict with all metrics and decisions
    - lstm_datasets: Prepared LSTM datasets for later use
    """

    training_history = {
        'lstm': [],
        'random_forest': [],
        'xgboost': [],
        'ensemble': None,
        'final_decision': None,
        'diagnostics': {},
        'early_stopping_triggered': {}
    }
    
    # Run diagnostic checks before training
    print("\n" + "="*60)
    print("🔬 RUNNING PRE-TRAINING DIAGNOSTICS")
    print("="*60)
    
    diagnostics_rf = check_data_health(
        x_train, x_val, x_test,
        y_train_unscaled, y_val_unscaled, y_test_unscaled,
        "Random Forest / XGBoost"
    )
    training_history['diagnostics']['rf_xgb'] = diagnostics_rf
    
    diagnostics_lstm = check_data_health(
        x_train, x_val, x_test,
        y_train_scaled, y_val_scaled, y_test_scaled,
        "LSTM"
    )
    training_history['diagnostics']['lstm'] = diagnostics_lstm

    # Convert to DataFrames for Random Forest
    x_train_df = pd.DataFrame(x_train)
    x_val_df = pd.DataFrame(x_val)
    x_test_df = pd.DataFrame(x_test)

    y_train_unscaled_series = pd.Series(y_train_unscaled)  # UNSCALED for Random Forest
    y_val_unscaled_series = pd.Series(y_val_unscaled)        # UNSCALED for Random Forest
    y_test_unscaled_series = pd.Series(y_test_unscaled)      # UNSCALED for Random Forest

    # Prepare LSTM datasets ONCE
    print("\n" + "="*60)
    print("📊 PREPARING LSTM DATASETS")
    print("="*60)

    lstm_datasets = prepare_lstm_datasets(
        x_train, y_train_scaled,
        x_val, y_val_scaled,
        x_test, y_test_scaled,
        time_steps
    )

    print(f"✅ LSTM datasets prepared:")
    print(f"   - Training samples: {lstm_datasets['metadata']['train_samples']}")
    print(f"   - Validation samples: {lstm_datasets['metadata']['val_samples']}")
    print(f"   - Test samples: {lstm_datasets['metadata']['test_samples']}")
    print(f"   - Time steps: {lstm_datasets['metadata']['time_steps']}")
    print(f"   - Features: {lstm_datasets['metadata']['num_features']}")

    lstm_model = None
    rf_model = None
    xgb_model = None
    lstm_overfitted = False
    rf_overfitted = False
    xgb_overfitted = False

    # ===== LSTM TRAINING LOOP =====
    print("\n" + "="*60)
    print("🚀 STARTING LSTM MODEL TRAINING")
    print("="*60)

    for lstm_attempt in range(max_retrains):
        print(f"\n📊 LSTM Training Attempt {lstm_attempt + 1}/{max_retrains}")

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
            epochs=lstm_epochs
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
        lstm_overfitted, lstm_overfitting_score = detect_overfitting(
            train_metrics, val_metrics, test_metrics, "LSTM", overfitting_threshold, use_multi_metric_detection
        )

        if not lstm_overfitted:
            print(f"✅ LSTM model accepted after {lstm_attempt + 1} attempt(s)")
            break
        elif lstm_attempt < max_retrains - 1:
            print(f"⚠️  Retraining LSTM with adjusted hyperparameters...")
            print(f"   Increasing trials: {lstm_trials} → {lstm_trials + lstm_retrain_trials_increment}")
            print(f"   Increasing executions: {lstm_executions} → {lstm_executions + lstm_retrain_executions_increment}")
            lstm_trials += lstm_retrain_trials_increment
            lstm_executions += lstm_retrain_executions_increment
        else:
            print(f"⚠️  LSTM reached maximum retrain attempts. Accepting current model.")

    # ===== RANDOM FOREST TRAINING LOOP =====
    print("\n" + "="*60)
    print("🌳 STARTING RANDOM FOREST MODEL TRAINING")
    print("="*60)
    
    rf_previous_hyperparams = None
    rf_identical_count = 0
    rf_best_score = float('inf')
    rf_search_space_constrained = False

    for rf_attempt in range(max_retrains):
        print(f"\n📊 Random Forest Training Attempt {rf_attempt + 1}/{max_retrains}")
        
        # Constrain search space if overfitting detected in previous attempts
        if rf_attempt > 0 and rf_overfitted and not rf_search_space_constrained:
            print(f"\n🔧 APPLYING SEARCH SPACE CONSTRAINTS (overfitting detected)")
            print(f"   • Reducing max_depth ceiling: 50 → 30")
            print(f"   • Increasing min_samples_leaf floor: 1 → 2")
            print(f"   • Forcing bootstrap=True for better generalization")
            rf_search_space_constrained = True

        rf_model = tune_random_forest_model(
            stock_symbol,
            x_train_df,
            y_train_unscaled_series,
            x_val_df,
            y_val_unscaled_series,
            max_trials=rf_trials,
            constrain_for_overfitting=rf_search_space_constrained
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
        rf_overfitted, rf_overfitting_score = detect_overfitting(
            train_metrics, val_metrics, test_metrics, "Random Forest", overfitting_threshold, use_multi_metric_detection
        )
        
        # Track best score
        if rf_overfitting_score < rf_best_score:
            rf_best_score = rf_overfitting_score
        
        # Early stopping: Check if hyperparameters are identical
        if rf_previous_hyperparams is not None:
            if are_hyperparameters_identical(rf_current_hyperparams, rf_previous_hyperparams):
                rf_identical_count += 1
                print(f"\n⚠️  WARNING: Identical hyperparameters found ({rf_identical_count} consecutive)")
                
                if rf_identical_count >= 3:
                    print(f"\n🛑 EARLY STOPPING TRIGGERED: Random Forest")
                    print(f"   Reason: Hyperparameter search converged to same solution 3 times")
                    print(f"   This indicates the model cannot improve further with current data")
                    print(f"   Recommendations:")
                    print(f"     • Collect more training data")
                    print(f"     • Improve feature engineering")
                    print(f"     • Consider simpler model architecture")
                    print(f"   Accepting current model as final.")
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
            print(f"✅ Random Forest model accepted after {rf_attempt + 1} attempt(s)")
            break
        elif rf_attempt < max_retrains - 1:
            print(f"⚠️  Retraining Random Forest with adjusted hyperparameters...")
            if not rf_search_space_constrained:
                print(f"   Strategy: Increasing trials: {rf_trials} → {rf_trials + rf_retrain_increment}")
                rf_trials += rf_retrain_increment
            else:
                print(f"   Strategy: Using constrained search space with {rf_trials} trials")
        else:
            print(f"⚠️  Random Forest reached maximum retrain attempts. Accepting current model.")

    # ===== XGBOOST TRAINING LOOP =====
    print("\n" + "="*60)
    print("🚀 STARTING XGBOOST MODEL TRAINING")
    print("="*60)
    
    xgb_previous_hyperparams = None
    xgb_identical_count = 0
    xgb_best_score = float('inf')
    xgb_search_space_constrained = False

    for xgb_attempt in range(max_retrains):
        print(f"\n📊 XGBoost Training Attempt {xgb_attempt + 1}/{max_retrains}")
        
        # Constrain search space if overfitting detected in previous attempts
        if xgb_attempt > 0 and xgb_overfitted and not xgb_search_space_constrained:
            print(f"\n🔧 APPLYING SEARCH SPACE CONSTRAINTS (overfitting detected)")
            print(f"   • Reducing max_depth ceiling: 15 → 10")
            print(f"   • Increasing min_child_weight floor: 1 → 3")
            print(f"   • Strengthening regularization (alpha, lambda)")
            print(f"   • Narrowing subsample/colsample ranges")
            xgb_search_space_constrained = True

        xgb_model = tune_xgboost_model(
            stock_symbol,
            x_train_df,
            y_train_unscaled_series,
            x_val_df,
            y_val_unscaled_series,
            max_trials=xgb_trials,
            constrain_for_overfitting=xgb_search_space_constrained
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
        xgb_overfitted, xgb_overfitting_score = detect_overfitting(
            train_metrics, val_metrics, test_metrics, "XGBoost", overfitting_threshold, use_multi_metric_detection
        )
        
        # Track best score
        if xgb_overfitting_score < xgb_best_score:
            xgb_best_score = xgb_overfitting_score
        
        # Early stopping: Check if hyperparameters are identical
        if xgb_previous_hyperparams is not None:
            if are_hyperparameters_identical(xgb_current_hyperparams, xgb_previous_hyperparams):
                xgb_identical_count += 1
                print(f"\n⚠️  WARNING: Identical hyperparameters found ({xgb_identical_count} consecutive)")
                
                if xgb_identical_count >= 3:
                    print(f"\n🛑 EARLY STOPPING TRIGGERED: XGBoost")
                    print(f"   Reason: Hyperparameter search converged to same solution 3 times")
                    print(f"   This indicates the model cannot improve further with current data")
                    print(f"   Recommendations:")
                    print(f"     • Collect more training data")
                    print(f"     • Improve feature engineering")
                    print(f"     • Consider simpler model architecture")
                    print(f"   Accepting current model as final.")
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
            print(f"✅ XGBoost model accepted after {xgb_attempt + 1} attempt(s)")
            break
        elif xgb_attempt < max_retrains - 1:
            print(f"⚠️  Retraining XGBoost with adjusted hyperparameters...")
            if not xgb_search_space_constrained:
                print(f"   Strategy: Increasing trials: {xgb_trials} → {xgb_trials + xgb_retrain_increment}")
                xgb_trials += xgb_retrain_increment
            else:
                print(f"   Strategy: Using constrained search space with {xgb_trials} trials")
        else:
            print(f"⚠️  XGBoost reached maximum retrain attempts. Accepting current model.")

    # ===== ENSEMBLE EVALUATION =====
    print("\n" + "="*60)
    print("🎯 EVALUATING ENSEMBLE PREDICTIONS")
    print("="*60)

    # Get predictions from all models
    lstm_train_pred = lstm_model.predict(lstm_datasets['train']['x'], verbose=0).flatten()
    lstm_val_pred = lstm_model.predict(lstm_datasets['val']['x'], verbose=0).flatten()
    lstm_test_pred = lstm_model.predict(lstm_datasets['test']['x'], verbose=0).flatten()

    # RF and XGBoost predictions (full length, need to align with LSTM)
    # Convert to numpy to avoid feature name warnings
    rf_train_pred_full = rf_model.predict(x_train_df.values)
    rf_val_pred_full = rf_model.predict(x_val_df.values)
    rf_test_pred_full = rf_model.predict(x_test_df.values)

    xgb_train_pred_full = xgb_model.predict(x_train_df)
    xgb_val_pred_full = xgb_model.predict(x_val_df)
    xgb_test_pred_full = xgb_model.predict(x_test_df)

    # Align RF/XGBoost predictions with LSTM (trim first time_steps-1 samples)
    rf_train_pred = rf_train_pred_full[time_steps-1:]
    rf_val_pred = rf_val_pred_full[time_steps-1:]
    rf_test_pred = rf_test_pred_full[time_steps-1:]

    xgb_train_pred = xgb_train_pred_full[time_steps-1:]
    xgb_val_pred = xgb_val_pred_full[time_steps-1:]
    xgb_test_pred = xgb_test_pred_full[time_steps-1:]

    # Align ground truth values with LSTM sequences
    y_train_aligned = y_train_unscaled_series.iloc[time_steps-1:].values
    y_val_aligned = y_val_unscaled_series.iloc[time_steps-1:].values
    y_test_aligned = y_test_unscaled_series.iloc[time_steps-1:].values

    # Simple weighted ensemble (weights optimized on validation set)
    # Calculate weights based on inverse validation MSE
    lstm_val_mse = training_history['lstm'][-1]['val_metrics']['mse']
    rf_val_mse = training_history['random_forest'][-1]['val_metrics']['mse']
    xgb_val_mse = training_history['xgboost'][-1]['val_metrics']['mse']

    # Inverse MSE weights (lower MSE = higher weight)
    inv_mse_sum = (1/lstm_val_mse) + (1/rf_val_mse) + (1/xgb_val_mse)
    lstm_weight = (1/lstm_val_mse) / inv_mse_sum
    rf_weight = (1/rf_val_mse) / inv_mse_sum
    xgb_weight = (1/xgb_val_mse) / inv_mse_sum

    print(f"📊 Ensemble Weights (based on validation performance):")
    print(f"   - LSTM:         {lstm_weight:.3f}")
    print(f"   - Random Forest: {rf_weight:.3f}")
    print(f"   - XGBoost:      {xgb_weight:.3f}")

    # Create ensemble predictions
    ensemble_train_pred = (lstm_weight * lstm_train_pred + 
                          rf_weight * rf_train_pred + 
                          xgb_weight * xgb_train_pred)
    ensemble_val_pred = (lstm_weight * lstm_val_pred + 
                        rf_weight * rf_val_pred + 
                        xgb_weight * xgb_val_pred)
    ensemble_test_pred = (lstm_weight * lstm_test_pred + 
                         rf_weight * rf_test_pred + 
                         xgb_weight * xgb_test_pred)

    # Evaluate ensemble (using aligned ground truth)
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
    training_history['ensemble'] = {
        'weights': {'lstm': lstm_weight, 'rf': rf_weight, 'xgb': xgb_weight},
        'train_metrics': ensemble_train_metrics,
        'val_metrics': ensemble_val_metrics,
        'test_metrics': ensemble_test_metrics
    }

    # Detect ensemble overfitting
    ensemble_overfitted, ensemble_overfitting_score = detect_overfitting(
        ensemble_train_metrics, ensemble_val_metrics, ensemble_test_metrics, 
        "Ensemble", overfitting_threshold
    )

    # Final summary
    print("\n" + "="*60)
    print("📋 TRAINING SUMMARY")
    print("="*60)
    print(f"LSTM Training Attempts:          {len(training_history['lstm'])}")
    print(f"Random Forest Training Attempts: {len(training_history['random_forest'])}")
    print(f"XGBoost Training Attempts:       {len(training_history['xgboost'])}")
    print("\n📊 FINAL TEST SET PERFORMANCE:")
    print(f"   LSTM:         R²={training_history['lstm'][-1]['test_metrics']['r2']:.4f}, MSE={training_history['lstm'][-1]['test_metrics']['mse']:.6f}")
    print(f"   Random Forest: R²={training_history['random_forest'][-1]['test_metrics']['r2']:.4f}, MSE={training_history['random_forest'][-1]['test_metrics']['mse']:.6f}")
    print(f"   XGBoost:      R²={training_history['xgboost'][-1]['test_metrics']['r2']:.4f}, MSE={training_history['xgboost'][-1]['test_metrics']['mse']:.6f}")
    print(f"   🎯 ENSEMBLE:   R²={ensemble_test_metrics['r2']:.4f}, MSE={ensemble_test_metrics['mse']:.6f}")
    print("="*60 + "\n")

    training_history['final_decision'] = {
        'lstm_final': not lstm_overfitted,
        'rf_final': not rf_overfitted,
        'xgb_final': not xgb_overfitted,
        'ensemble_final': not ensemble_overfitted
    }

    # Return models dict for ensemble use
    models = {
        'lstm': lstm_model,
        'rf': rf_model,
        'xgb': xgb_model,
        'ensemble_weights': training_history['ensemble']['weights']
    }

    return models, training_history, lstm_datasets

def predict_future_price_changes(ticker, scaler_x, scaler_y, model, selected_features_list, stock_df, prediction_days, time_steps, historical_prediction_dataset_df=None):
    """
    Predicts the future stock price changes day by day.

    Parameters:
    - ticker (str): The stock ticker.
    - scaler_x (MinMaxScaler): The scaler for x values.
    - scaler_y (MinMaxScaler): The scaler for y values (for inverse-transforming LSTM predictions).
    - model (dict): Dictionary containing 'lstm' and 'rf' models.
    - selected_features_list (list): The list of selected features.
    - stock_df (pandas.DataFrame): A DataFrame containing the stock data.
    - prediction_days (int): The number of days to predict.
    - time_steps (int): Number of time steps for LSTM sequences.
    - historical_prediction_dataset_df (pd.DataFrame, optional): Historical prediction data.

    Returns:
    pandas.DataFrame: A DataFrame containing the predicted stock prices.

    Raises:
    - ValueError: If the prediction could not be completed.
    """

    try:
        # Extract individual models from the 'model' dictionary
        lstm_model = model['lstm']
        rf_model = model['rf']
        xgb_model = model.get('xgb', None)  # XGBoost model (optional for backward compatibility)

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
            'RSI_14', 'macd', 'macd_histogram', 'macd_signal', 'ATR_14',
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

        # --- Process historical prediction dataset if provided ---
        if historical_prediction_dataset_df is not None and len(historical_prediction_dataset_df) > 0:
            pred_count = len(historical_prediction_dataset_df)
            pred_dates = stock_df["date"].iloc[-pred_count:].copy().to_frame()

            for run in range(len(historical_prediction_dataset_df)):
                # For historical predictions, use the SAME pre-scaled test data for all models
                # This ensures consistent scaling between LSTM, RF, and XGB
                
                # Prepare LSTM input (requires time_steps of history from scaled data)
                # We need to get the sequence ending at the current prediction point
                start_idx = max(0, run - time_steps + 1)
                end_idx = run + 1
                
                # Extract time_steps worth of scaled historical features
                if end_idx - start_idx < time_steps:
                    # Not enough history, pad with first available data
                    padding_needed = time_steps - (end_idx - start_idx)
                    x_lstm_scaled = historical_prediction_dataset_df.iloc[0:end_idx][selected_features_list]
                    # Pad with first row repeated
                    first_row = x_lstm_scaled.iloc[0:1]
                    padding = pd.concat([first_row] * padding_needed, ignore_index=True)
                    x_lstm_scaled = pd.concat([padding, x_lstm_scaled], ignore_index=True)
                else:
                    x_lstm_scaled = historical_prediction_dataset_df.iloc[start_idx:end_idx][selected_features_list]
                
                # Prepare RF/XGB input (only current day features from test dataset)
                scaled_x_input_rf_df = historical_prediction_dataset_df.iloc[run:run+1][selected_features_list]
                    
                # Convert to NumPy array for LSTM
                x_lstm_array = x_lstm_scaled.values if hasattr(x_lstm_scaled, 'values') else np.array(x_lstm_scaled)
                
                # DEBUG: Print input statistics for first few predictions
                if run < 6:
                    print(f"\n🔍 DEBUG - Historical Prediction {run+1}:")
                    print(f"   Using PRE-SCALED test data (no re-scaling needed)")
                    print(f"   Input shape: {x_lstm_array.shape}")
                    print(f"   Input mean: {np.mean(x_lstm_array):.4f}, std: {np.std(x_lstm_array):.4f}")
                    print(f"   Input min: {np.min(x_lstm_array):.4f}, max: {np.max(x_lstm_array):.4f}")
                    print(f"   Sample features (last row): {x_lstm_array[-1, :5]}")
                
                # Reshape for LSTM (no scaling needed - already scaled!)
                scaled_x_input_lstm = x_lstm_array.reshape(1, time_steps, x_lstm_array.shape[1])

                # # Check for NaN in RF input
                # if scaled_x_input_rf_df.isnull().any().any():
                #     print(f"⚠️ Warning: NaN detected in RF input at historical step {run}")
                #     scaled_x_input_rf_df = scaled_x_input_rf_df.ffill().bfill().fillna(0)

                # --- Predict with all three models ---

                # LSTM prediction (scaled) -> inverse transform to original scale
                forecast_lstm_scaled = lstm_model.predict(scaled_x_input_lstm, verbose=0)[0][0]
                forecast_lstm = scaler_y.inverse_transform([[forecast_lstm_scaled]])[0][0]
                
                # DEBUG: Show LSTM prediction process
                if run < 6:
                    print(f"   LSTM scaled output: {forecast_lstm_scaled:.6f}")
                    print(f"   LSTM unscaled output: {forecast_lstm:.6f} ({forecast_lstm*100:.3f}%)")
                    print(f"   scaler_y min: {scaler_y.data_min_}, max: {scaler_y.data_max_}")

                # Random Forest prediction (already unscaled)
                # Convert DataFrame to numpy to avoid feature name warning
                forecast_rf = rf_model.predict(scaled_x_input_rf_df.values)[0]
                
                # XGBoost prediction (already unscaled)
                if xgb_model is not None:
                    forecast_xgb = xgb_model.predict(scaled_x_input_rf_df.values)[0]
                    
                    # ENSEMBLE: RF + XGB only (LSTM disabled due to mode collapse at -11%)
                    # RF: 0.2-1.1% error, XGB: 0-1.8% error, LSTM: 9-11% error
                    forecast_price_change = (forecast_rf + forecast_xgb) / 2
                    
                    # Get actual value for comparison
                    current_date = pred_dates["date"].iloc[run]
                    actual_price_change = stock_df.loc[stock_df["date"] == current_date, "1D"].values[0] if "1D" in stock_df.columns else None
                    
                    print(f"\n📊 Historical Prediction Day {run+1} ({current_date.strftime('%Y-%m-%d')}):")
                    print(f"   LSTM:      {forecast_lstm:+.6f} ({forecast_lstm*100:+.3f}%)")
                    print(f"   RF:        {forecast_rf:+.6f} ({forecast_rf*100:+.3f}%)")
                    print(f"   XGB:       {forecast_xgb:+.6f} ({forecast_xgb*100:+.3f}%)")
                    print(f"   Ensemble:  {forecast_price_change:+.6f} ({forecast_price_change*100:+.3f}%)")
                    if actual_price_change is not None:
                        print(f"   Actual:    {actual_price_change:+.6f} ({actual_price_change*100:+.3f}%)")
                        print(f"   Errors:    LSTM={abs(forecast_lstm-actual_price_change)*100:.3f}%, RF={abs(forecast_rf-actual_price_change)*100:.3f}%, XGB={abs(forecast_xgb-actual_price_change)*100:.3f}%, Ensemble={abs(forecast_price_change-actual_price_change)*100:.3f}%")
                else:
                    # Only RF available (LSTM disabled, XGB unavailable)
                    forecast_price_change = forecast_rf
                    
                    current_date = pred_dates["date"].iloc[run]
                    actual_price_change = stock_df.loc[stock_df["date"] == current_date, "1D"].values[0] if "1D" in stock_df.columns else None
                    
                    print(f"\n📊 Historical Prediction Day {run+1} ({current_date.strftime('%Y-%m-%d')}):")
                    print(f"   LSTM:      {forecast_lstm:+.6f} ({forecast_lstm*100:+.3f}%)")
                    print(f"   RF:        {forecast_rf:+.6f} ({forecast_rf*100:+.3f}%)")
                    print(f"   Ensemble:  {forecast_price_change:+.6f} ({forecast_price_change*100:+.3f}%)")
                    if actual_price_change is not None:
                        print(f"   Actual:    {actual_price_change:+.6f} ({actual_price_change*100:+.3f}%)")
                        print(f"   Errors:    LSTM={abs(forecast_lstm-actual_price_change)*100:.3f}%, RF={abs(forecast_rf-actual_price_change)*100:.3f}%, Ensemble={abs(forecast_price_change-actual_price_change)*100:.3f}%")

                # Update stock_mod_df with predictions
                stock_mod_df.loc[stock_mod_df["date"] == current_date, "1D"] = forecast_price_change

                # Calculate new price based on price change
                prev_price = stock_mod_df["close_Price"].iloc[-(pred_count+1)]
                stock_mod_df.loc[stock_mod_df["date"] == current_date, "close_Price"] = prev_price * (1 + forecast_price_change)

                pred_count -= 1

        # --- Future predictions loop ---
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

            # --- Carry forward fundamental features (they don't change daily) ---
            for fundamental_feature in fundamental_features:
                if fundamental_feature in stock_mod_df.columns:
                    future_df[fundamental_feature] = stock_df.iloc[-1][fundamental_feature]

            # --- Recalculate dynamic features ---
            for feature in short_term_dynamic_list:
                if feature not in selected_features_list:
                    continue

                # print(f"🔄 Recalculating feature: {feature}")
                # Helper function to get historical data if needed
                def get_historical_data(period):
                    try:
                        hist_df = pd.DataFrame(yf.download(ticker, period=period, progress=False, auto_adjust=False))
                        if not hist_df.empty:
                            # print("hist_df\n", hist_df)
                            hist_df = hist_df["Close"].reset_index()
                            hist_df = hist_df.loc[hist_df["Date"] < stock_mod_df.iloc[0]["date"]]
                            # print("hist_df\n", hist_df)
                            hist_df = hist_df.rename(columns={ticker: "close_Price", "Date": "date"})
                            # print("hist_df\n", hist_df)
                            hist_df = pd.concat([hist_df, stock_mod_df[["date", "close_Price"]]], axis=0)
                        return hist_df
                    except Exception as e:
                        print(f"⚠️ Warning: Could not download historical data: {e}")
                        return pd.DataFrame()

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

                    # Momentum
                    elif feature == "momentum":
                        if stock_mod_df.iloc[-1]["close_Price"] >= stock_mod_df.iloc[-2]["close_Price"]:
                            momentum = 1 if stock_mod_df.iloc[-1].get("momentum", 0) <= 0 else stock_mod_df.iloc[-1]["momentum"] + 1
                        else:
                            momentum = -1 if stock_mod_df.iloc[-1].get("momentum", 0) >= 0 else stock_mod_df.iloc[-1]["momentum"] - 1
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

                    elif feature == "ATR_14":
                        # ATR requires high/low/close data, carry forward
                        future_df["ATR_14"] = stock_mod_df.iloc[-1]["ATR_14"] if "ATR_14" in stock_mod_df.columns else 0.0

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

                except Exception as e:
                    print(f"⚠️ Warning: Error calculating feature '{feature}': {e}")
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
            exclude_cols = ["date", "ticker", "currency", "open_Price", "high_Price", "low_Price", "close_Price", "trade_Volume", "1D"]
            all_features = [col for col in stock_mod_df.columns if col not in exclude_cols]
            
            # Get the last time_steps rows with ALL features for scaling
            x_lstm_all_features = stock_mod_df.iloc[-time_steps:][all_features]

            # Check and handle NaN in LSTM input
            if x_lstm_all_features.isnull().any().any():
                print(f"⚠️ Warning: NaN detected in LSTM input at day {run+1}")
                nan_cols = x_lstm_all_features.columns[x_lstm_all_features.isnull().any()].tolist()
                print(f"   Columns with NaN: {nan_cols}")
                # printout the values of the columns with NaN before filling
                print(x_lstm_all_features[nan_cols])
                x_lstm_all_features = x_lstm_all_features.ffill().bfill().fillna(0)

            # Scale ALL features (scaler expects all features it was fitted on)
            scaled_x_lstm_all_df = scaler_x.transform(x_lstm_all_features)
            
            # Now select only the features needed for LSTM
            scaled_x_lstm_df = scaled_x_lstm_all_df[selected_features_list]
            
            # Convert to numpy array after scaling
            scaled_x_lstm_array = scaled_x_lstm_df.values if hasattr(scaled_x_lstm_df, 'values') else np.array(scaled_x_lstm_df)
            scaled_x_input_lstm = scaled_x_lstm_array.reshape(1, time_steps, scaled_x_lstm_array.shape[1])

            # B. Random Forest Input (Only the current day's features)
            x_input_rf_df = stock_mod_df.iloc[-1:][selected_features_list]

            # Convert all columns to numeric (fix dtype issue for XGBoost/RF)
            # XGBoost requires int/float/bool/category dtypes, not object
            x_input_rf_df = x_input_rf_df.apply(pd.to_numeric, errors='coerce')

            # Check and handle NaN in RF input
            if x_input_rf_df.isnull().any().any():
                print(f"⚠️ Warning: NaN detected in RF input at day {run+1}")
                nan_cols = x_input_rf_df.columns[x_input_rf_df.isnull().any()].tolist()
                print(f"   Columns with NaN: {nan_cols}")
                x_input_rf_df = x_input_rf_df.ffill().bfill().fillna(0)

            # --- Predict and Ensemble with all three models ---

            # LSTM prediction (scaled) -> inverse transform to original scale
            forecast_lstm_scaled = lstm_model.predict(scaled_x_input_lstm, verbose=0)[0][0]
            forecast_lstm = scaler_y.inverse_transform([[forecast_lstm_scaled]])[0][0]

            # Random Forest prediction (already unscaled)
            # Convert DataFrame to numpy to avoid feature name warning
            forecast_rf = rf_model.predict(x_input_rf_df.values)[0]

            # XGBoost prediction (already unscaled)
            if xgb_model is not None:
                forecast_xgb = xgb_model.predict(x_input_rf_df.values)[0]

                # ENSEMBLE: RF + XGB only (LSTM disabled due to mode collapse at -11%)
                # RF: 0.2-1.1% error, XGB: 0-1.8% error, LSTM: 9-11% error
                forecast_price_change = (forecast_rf + forecast_xgb) / 2

                future_date = future_df["date"].iloc[0]
                print(f"\n🔮 Future Prediction Day {run+1} ({future_date}):")
                print(f"   LSTM:      {forecast_lstm:+.6f} ({forecast_lstm*100:+.3f}%)")
                print(f"   RF:        {forecast_rf:+.6f} ({forecast_rf*100:+.3f}%)")
                print(f"   XGB:       {forecast_xgb:+.6f} ({forecast_xgb*100:+.3f}%)")
                print(f"   Ensemble:  {forecast_price_change:+.6f} ({forecast_price_change*100:+.3f}%)")

                # Show model agreement/disagreement
                predictions = [forecast_lstm, forecast_rf, forecast_xgb]
                std_dev = np.std(predictions)
                print(f"   Agreement: σ={std_dev:.6f} ({'High' if std_dev < 0.01 else 'Medium' if std_dev < 0.02 else 'Low'} consensus)")
            else:
                # Only RF available (LSTM disabled, XGB unavailable)
                forecast_price_change = forecast_rf

                future_date = future_df["date"].iloc[0]
                print(f"\n🔮 Future Prediction Day {run+1} ({future_date}):")
                print(f"   LSTM:      {forecast_lstm:+.6f} ({forecast_lstm*100:+.3f}%)")
                print(f"   RF:        {forecast_rf:+.6f} ({forecast_rf*100:+.3f}%)")
                print(f"   Ensemble:  {forecast_price_change:+.6f} ({forecast_price_change*100:+.3f}%)")

                predictions = [forecast_lstm, forecast_rf]
                std_dev = np.std(predictions)
                print(f"   Agreement: σ={std_dev:.6f} ({'High' if std_dev < 0.01 else 'Medium' if std_dev < 0.02 else 'Low'} consensus)")

            # --- Update stock_mod_df with the Ensemble Forecast ---

            # Update the 1D column with the calculated value
            stock_mod_df.loc[len(stock_mod_df)-1, "1D"] = forecast_price_change

            # Update the close_Price column with the calculated value
            prev_price = stock_mod_df.loc[len(stock_mod_df)-2, "close_Price"]
            stock_mod_df.loc[len(stock_mod_df)-1, "close_Price"] = prev_price * (1 + forecast_price_change)
            # print("Columns in stock_mod_df:", stock_mod_df.columns.tolist())
            # print("stock_mod_df after prediction:\n", stock_mod_df[["date", "close_Price"]].tail(3))

        # --- Final cleanup ---
        columns_to_convert = stock_mod_df.columns.drop(["date", "ticker", "currency"], errors='ignore').to_list()
        for column in columns_to_convert:
            stock_mod_df[column] = pd.to_numeric(stock_mod_df[column], errors='coerce').fillna(0)

        stock_mod_df = stock_mod_df[features_list]
        # print("Columns in stock_mod_df:", stock_mod_df.columns.tolist())
        return stock_mod_df

    except Exception as e:
        print(f"❌ Prediction failed: {e}")
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
    print("📈 PREDICTION PERFORMANCE ANALYSIS")
    print("="*80)
    
    if historical_prediction_count > 0:
        # Get historical prediction period
        hist_forecast = forecast_df.iloc[-historical_prediction_count:]
        hist_actual = stock_df.iloc[-historical_prediction_count:]
        
        # Compare predicted vs actual price changes
        if "1D" in hist_actual.columns and "1D" in hist_forecast.columns:
            predicted_changes = hist_forecast["1D"].values
            actual_changes = hist_actual["1D"].values
            
            # Calculate errors
            errors = predicted_changes - actual_changes
            abs_errors = np.abs(errors)
            pct_errors = abs_errors * 100  # Convert to percentage points
            
            print(f"\n📊 Historical Prediction Accuracy (Last {historical_prediction_count} days):")
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
            if "close_Price" in hist_forecast.columns and "close_Price" in hist_actual.columns:
                # Ensure we get 1D arrays by using .squeeze() to remove extra dimensions
                predicted_prices = hist_forecast["close_Price"].values
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
                
                print(f"\n💰 Price Prediction Accuracy:")
                print(f"   Mean Price Error:       {np.mean(np.abs(price_errors)):.2f} ({np.mean(np.abs(price_pct_errors)):.2f}%)")
                print(f"   Median Price Error:     {np.median(np.abs(price_errors)):.2f} ({np.median(np.abs(price_pct_errors)):.2f}%)")
                print(f"   Max Price Error:        {np.max(np.abs(price_errors)):.2f} ({np.max(np.abs(price_pct_errors)):.2f}%)")
                
                # Show day-by-day comparison
                print(f"\n📅 Day-by-Day Comparison (Historical):")
                print(f"{'Date':<12} {'Actual Price':>12} {'Pred Price':>12} {'Error':>10} {'Actual Δ%':>10} {'Pred Δ%':>10}")
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
    except ValueError:
        print(f"The model could not predict an expected return over the next {prediction_days} days.")

def plot_graph(stock_data_df, forecast_data_df):
    """
    Plots a graph of the stock data.

    Parameters:
    - stock_data_df (pandas.DataFrame): A DataFrame containing the stock data.

    Returns:
    None

    Raises:
    - FileNotFoundError: If the graph could not be saved.
    """

    print(stock_data_df["close_Price"])
    print(forecast_data_df["close_Price"])
    # Plot the graph
    plt.figure(figsize=(18, 8))
    stock_data_df["date"] = stock_data_df["date"].astype('datetime64[ns]')
    forecast_data_df["date"] = forecast_data_df["date"].astype('datetime64[ns]')
    # forecast_data_df = forecast_data_df.loc[forecast_data_df["date"] >= stock_data_df.iloc[-1]["date"]]
    stock_data_df = stock_data_df.set_index("date")
    forecast_data_df = forecast_data_df.set_index("date")
    stock_data_df["close_Price"].plot()
    forecast_data_df["close_Price"].plot()
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
    graph_name = str(f"stock_prediction_of_{stock_name}.png")
    my_path = os.path.abspath(__file__)
    path = os.path.dirname(my_path)
    # Save the graph
    try:
        plt.savefig(os.path.join(path, "generated_graphs", graph_name), bbox_inches="tight", pad_inches=0.5, transparent=False, format="png")
        plt.clf()
        plt.close("all")

    except FileNotFoundError as e:
        plt.close("all")

# Run the main function
if __name__ == "__main__":
    import db_interactions

    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_virtual_device_configuration(
            gpu,
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7168)]  # Limit to 7GB
        )

    start_time = time.time()

    # Import stock symbols from DB
    stock_symbols_list = db_interactions.import_ticker_list()
    print(stock_symbols_list)
    stock_symbol = stock_symbols_list[0]
    stock_symbol = "DEMANT.CO"
    print(stock_symbol)

    # Import stock data
    stock_data_df = db_interactions.import_stock_dataset(stock_symbol)
    # Change the date column to datetime 64
    stock_data_df["date"] = pd.to_datetime(stock_data_df["date"])
    # Drop the columns that are empty
    stock_data_df = stock_data_df.dropna(axis=0, how="any")
    stock_data_df = stock_data_df.dropna(axis=1, how="any")
    print("Stock DataFrame describe:")
    print(stock_data_df.describe())

    # Split the dataset into training, validation, test data and prediction data
    validation_size = 0.20
    test_size = 0.10
    scaler_x, scaler_y, x_train_scaled, x_val_scaled, x_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled, x_Predictions = split_dataset.dataset_train_test_split(stock_data_df, test_size, validation_size=validation_size)
    # print("x_train_scaled.info()")
    # print(x_train_scaled.info())
    # print("x_val_scaled.info()")
    # print(x_val_scaled.info())
    # print("x_test_scaled.info()")
    # print(x_test_scaled.info())
    # print("x_Predictions.info()")
    # print(x_Predictions.info())

    # Inverse-transform y values for Random Forest (RF is scale-invariant, needs unscaled y)
    y_train_unscaled = scaler_y.inverse_transform(y_train_scaled.reshape(-1, 1)).flatten()
    y_val_unscaled = scaler_y.inverse_transform(y_val_scaled.reshape(-1, 1)).flatten()
    y_test_unscaled = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()

    # Convert to DataFrames for feature selection
    x_training_data = pd.DataFrame(x_train_scaled)
    x_val_data = pd.DataFrame(x_val_scaled)
    x_test_data = pd.DataFrame(x_test_scaled)
    y_training_data_df = pd.Series(y_train_unscaled)  # UNSCALED for Random Forest
    y_val_data_df = pd.Series(y_val_unscaled)         # UNSCALED for Random Forest
    y_test_data_df = pd.Series(y_test_unscaled)       # UNSCALED for Random Forest
    prediction_data = x_Predictions

    max_features = len(x_training_data.columns)
    print(f"Max features:\n{max_features}")
    feature_amount = max_features
    # Use RandomForest feature importance (11.8% better Test MAE than SelectKBest)
    # Test results: RF Test MAE 0.009387 vs SelectKBest 0.010640, Test R² 0.668 vs 0.535
    x_training_dataset, x_val_dataset, x_test_dataset, x_prediction_dataset, selected_features_model, selected_features_list = dimension_reduction.feature_selection_rf(
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

    # # DEBUG: Check feature counts
    # print(f"Features after selection: {len(selected_features_list)}")
    x_training_dataset_df = pd.DataFrame(x_training_dataset, columns=selected_features_list)
    y_training_data_df = y_training_data_df.reset_index(drop=True)
    # Convert back to DataFrames after feature selection
    x_val_dataset_df = pd.DataFrame(x_val_dataset, columns=selected_features_list)
    y_val_data_df = y_val_data_df.reset_index(drop=True)

    x_test_dataset_df = pd.DataFrame(x_test_dataset, columns=selected_features_list)
    y_test_data_df = y_test_data_df.reset_index(drop=True)

    x_prediction_dataset_df = pd.DataFrame(x_prediction_dataset, columns=selected_features_list)

    TIME_STEPS = 30 # Set TIME_STEPS for LSTM

    y_train_scaled_for_lstm = pd.Series(y_train_scaled)
    y_test_scaled_for_lstm = pd.Series(y_test_scaled)
    y_val_scaled_for_lstm = pd.Series(y_val_scaled)

    # MAIN TRAINING WITH AUTOMATIC RETRAINING AND OVERFITTING DETECTION
    models, training_history, lstm_datasets = train_and_validate_models(
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
        time_steps=TIME_STEPS,
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
        use_multi_metric_detection=True
    )
    lstm_model = models['lstm']
    rf_model = models['rf']
    xgb_model = models['xgb']
    ensemble_weights = models['ensemble_weights']

    # Print training history summary
    print("\n" + "="*60)
    print("📊 COMPLETE TRAINING HISTORY")
    print("="*60)
    for i, lstm_history in enumerate(training_history['lstm']):
        print(f"\nLSTM Attempt {i+1}:")
        print(f"  Test MSE: {lstm_history['test_metrics']['mse']:.6f}")
        print(f"  Test R²:  {lstm_history['test_metrics']['r2']:.4f}")

    for i, rf_history in enumerate(training_history['random_forest']):
        print(f"\nRandom Forest Attempt {i+1}:")

    # Predict the future stock price changes
    amount_of_days = TIME_STEPS * 3
    forecast_df = predict_future_price_changes(
        ticker=stock_symbol,
        scaler_x=scaler_x,
        scaler_y=scaler_y,
        model={'lstm': lstm_model, 'rf': rf_model, 'xgb': xgb_model},
        selected_features_list=selected_features_list,
        stock_df=stock_data_df,
        prediction_days=amount_of_days,
        time_steps=TIME_STEPS,  # Add this parameter
        historical_prediction_dataset_df=x_prediction_dataset_df
    )

    print("Forecast DataFrame:")
    print(forecast_df)
    print(forecast_df.columns.tolist())
    
    # Analyze prediction performance
    historical_pred_count = len(x_prediction_dataset_df) if x_prediction_dataset_df is not None else 0
    analyze_prediction_performance(stock_data_df, forecast_df, historical_pred_count)
    
    plt.plot(forecast_df["close_Price"], color="green")
    plt.xlabel("Date")
    plt.ylabel("Opening price")
    legend_list = ["Predicted Stock Price"]
    plt.legend(legend_list,
        loc="best"
    )
    stock_name = stock_data_df.iloc[0]["ticker"]
    graph_name = str(f"future_stock_prediction_of_{stock_name}.png")
    my_path = os.path.abspath(__file__)
    path = os.path.dirname(my_path)
    # Save the graph
    try:
        plt.savefig(os.path.join(path, "generated_graphs", graph_name), bbox_inches="tight", pad_inches=0.5, transparent=False, format="png")
        plt.clf()
        plt.close("all")

    except FileNotFoundError as e:
        raise FileNotFoundError("The graph could not be saved. Please check the file name or path.") from e

    # Calculate the predicted profit
    calculate_predicted_profit(forecast_df, amount_of_days)

    # export the forecast to the excel file
    import openpyxl
    forecast_file_name = f"forecast_{stock_symbol}.xlsx"
    my_path = os.path.abspath(__file__)
    forecast_file_path = os.path.join(os.path.dirname(my_path), "generated_forecasts", forecast_file_name)
    forecast_df.to_excel(forecast_file_path, index=False)
    # Plot the graph
    plot_graph(stock_data_df, forecast_df)

    # Run a Monte Carlo simulation
    year_amount = 10
    sim_amount = 1000
    monte_carlo_day_df, monte_carlo_year_df = monte_carlo_sim.monte_carlo_analysis(0, stock_data_df, forecast_df, year_amount, sim_amount)
    forecast_df = forecast_df.rename(columns={"close_Price": stock_symbol + "_price"})

    # Calculate the execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"⏱️ Total execution time: {execution_time:.2f} seconds")
