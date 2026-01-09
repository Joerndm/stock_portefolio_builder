"""
FEATURE SELECTION COMPARISON TEST
==================================

Compares SelectKBest (linear correlation) vs RandomForest feature importance
to determine which yields better model performance.

Tests both methods with the same data and models, then reports which is better.

Usage:
    python test_reports/compare_feature_selection.py
"""

import sys
import os
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import db_interactions
import split_dataset
import dimension_reduction

# Colors for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'


def train_and_evaluate_rf(x_train, y_train, x_val, y_val, x_test, y_test, method_name):
    """Train Random Forest and return validation + test metrics."""
    print(f"\n{BLUE}Training Random Forest with {method_name} features...{RESET}")
    
    # Train RF with reasonable hyperparameters
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    
    start_time = time.time()
    rf_model.fit(x_train, y_train)
    train_time = time.time() - start_time
    
    # Make predictions
    train_pred = rf_model.predict(x_train)
    val_pred = rf_model.predict(x_val)
    test_pred = rf_model.predict(x_test)
    
    # Calculate metrics
    metrics = {
        'train': {
            'mse': mean_squared_error(y_train, train_pred),
            'mae': mean_absolute_error(y_train, train_pred),
            'r2': r2_score(y_train, train_pred)
        },
        'val': {
            'mse': mean_squared_error(y_val, val_pred),
            'mae': mean_absolute_error(y_val, val_pred),
            'r2': r2_score(y_val, val_pred)
        },
        'test': {
            'mse': mean_squared_error(y_test, test_pred),
            'mae': mean_absolute_error(y_test, test_pred),
            'r2': r2_score(y_test, test_pred)
        },
        'train_time': train_time
    }
    
    return metrics, rf_model


def compare_feature_selection_methods(stock_symbol="DEMANT.CO", n_features=50):
    """
    Compare SelectKBest vs RandomForest feature selection.
    
    Parameters:
    - stock_symbol: Stock to test on
    - n_features: Number of features to select
    """
    
    print("\n" + "="*80)
    print(f"{BOLD}FEATURE SELECTION COMPARISON TEST{RESET}")
    print("="*80)
    print(f"Stock: {stock_symbol}")
    print(f"Features to select: {n_features}")
    print("="*80)
    
    # Load data
    print(f"\n{BLUE}📊 Loading stock data...{RESET}")
    stock_data_df = db_interactions.import_stock_dataset(stock_symbol)
    stock_data_df = stock_data_df.dropna(axis=0, how="any")
    stock_data_df = stock_data_df.dropna(axis=1, how="any")
    print(f"   Dataset shape: {stock_data_df.shape}")
    
    # Split data
    print(f"\n{BLUE}🔪 Splitting data (65% train, 15% val, 20% test)...{RESET}")
    scaler_x, scaler_y, x_train_scaled, x_val_scaled, x_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled, x_predictions = split_dataset.dataset_train_test_split(
        stock_data_df, test_size=0.20, validation_size=0.15
    )
    
    # Inverse transform y for Random Forest (RF is scale-invariant)
    y_train_unscaled = scaler_y.inverse_transform(y_train_scaled.reshape(-1, 1)).flatten()
    y_val_unscaled = scaler_y.inverse_transform(y_val_scaled.reshape(-1, 1)).flatten()
    y_test_unscaled = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()
    
    # Convert to DataFrames
    x_train_df = pd.DataFrame(x_train_scaled)
    x_val_df = pd.DataFrame(x_val_scaled)
    x_test_df = pd.DataFrame(x_test_scaled)
    
    y_train_series = pd.Series(y_train_unscaled)
    y_val_series = pd.Series(y_val_unscaled)
    y_test_series = pd.Series(y_test_unscaled)
    
    print(f"   Train: {x_train_df.shape}, Val: {x_val_df.shape}, Test: {x_test_df.shape}")
    
    # ========================================================================
    # METHOD 1: SelectKBest (Linear Correlation)
    # ========================================================================
    print("\n" + "="*80)
    print(f"{BOLD}METHOD 1: SelectKBest (Linear Correlation){RESET}")
    print("="*80)
    
    x_train_kbest, x_val_kbest, x_test_kbest, x_pred_kbest, selector_kbest, features_kbest = dimension_reduction.feature_selection(
        n_features, x_train_df, x_val_df, x_test_df,
        y_train_series, y_val_series, y_test_series,
        x_predictions, stock_data_df
    )
    
    metrics_kbest, model_kbest = train_and_evaluate_rf(
        x_train_kbest, y_train_series,
        x_val_kbest, y_val_series,
        x_test_kbest, y_test_series,
        "SelectKBest"
    )
    
    # ========================================================================
    # METHOD 2: RandomForest Feature Importance
    # ========================================================================
    print("\n" + "="*80)
    print(f"{BOLD}METHOD 2: RandomForest Feature Importance{RESET}")
    print("="*80)
    
    x_train_rf, x_val_rf, x_test_rf, x_pred_rf, selector_rf, features_rf = dimension_reduction.feature_selection_rf(
        n_features, x_train_df, x_val_df, x_test_df,
        y_train_series, y_val_series, y_test_series,
        x_predictions, stock_data_df
    )
    
    metrics_rf, model_rf = train_and_evaluate_rf(
        x_train_rf, y_train_series,
        x_val_rf, y_val_series,
        x_test_rf, y_test_series,
        "RF Importance"
    )
    
    # ========================================================================
    # COMPARISON RESULTS
    # ========================================================================
    print("\n" + "="*80)
    print(f"{BOLD}📊 COMPARISON RESULTS{RESET}")
    print("="*80)
    
    # Print metrics table
    print(f"\n{BOLD}Performance Metrics:{RESET}")
    print(f"{'Metric':<15} {'SelectKBest':<20} {'RF Importance':<20} {'Winner':<15}")
    print("-" * 70)
    
    # Validation metrics (most important for hyperparameter selection)
    val_mse_winner = "RF Importance" if metrics_rf['val']['mse'] < metrics_kbest['val']['mse'] else "SelectKBest"
    val_mae_winner = "RF Importance" if metrics_rf['val']['mae'] < metrics_kbest['val']['mae'] else "SelectKBest"
    val_r2_winner = "RF Importance" if metrics_rf['val']['r2'] > metrics_kbest['val']['r2'] else "SelectKBest"
    
    print(f"{'Val MSE':<15} {metrics_kbest['val']['mse']:<20.6f} {metrics_rf['val']['mse']:<20.6f} {val_mse_winner:<15}")
    print(f"{'Val MAE':<15} {metrics_kbest['val']['mae']:<20.6f} {metrics_rf['val']['mae']:<20.6f} {val_mae_winner:<15}")
    print(f"{'Val R²':<15} {metrics_kbest['val']['r2']:<20.6f} {metrics_rf['val']['r2']:<20.6f} {val_r2_winner:<15}")
    
    # Test metrics (generalization performance)
    test_mse_winner = "RF Importance" if metrics_rf['test']['mse'] < metrics_kbest['test']['mse'] else "SelectKBest"
    test_mae_winner = "RF Importance" if metrics_rf['test']['mae'] < metrics_kbest['test']['mae'] else "SelectKBest"
    test_r2_winner = "RF Importance" if metrics_rf['test']['r2'] > metrics_kbest['test']['r2'] else "SelectKBest"
    
    print(f"{'Test MSE':<15} {metrics_kbest['test']['mse']:<20.6f} {metrics_rf['test']['mse']:<20.6f} {test_mse_winner:<15}")
    print(f"{'Test MAE':<15} {metrics_kbest['test']['mae']:<20.6f} {metrics_rf['test']['mae']:<20.6f} {test_mae_winner:<15}")
    print(f"{'Test R²':<15} {metrics_kbest['test']['r2']:<20.6f} {metrics_rf['test']['r2']:<20.6f} {test_r2_winner:<15}")
    
    # Training time
    time_winner = "RF Importance" if metrics_rf['train_time'] < metrics_kbest['train_time'] else "SelectKBest"
    print(f"{'Train Time (s)':<15} {metrics_kbest['train_time']:<20.2f} {metrics_rf['train_time']:<20.2f} {time_winner:<15}")
    
    # Overfitting check
    print(f"\n{BOLD}Overfitting Analysis:{RESET}")
    kbest_overfit = metrics_kbest['train']['r2'] - metrics_kbest['val']['r2']
    rf_overfit = metrics_rf['train']['r2'] - metrics_rf['val']['r2']
    print(f"{'SelectKBest':<20} Train R²: {metrics_kbest['train']['r2']:.4f}, Val R²: {metrics_kbest['val']['r2']:.4f}, Gap: {kbest_overfit:.4f}")
    print(f"{'RF Importance':<20} Train R²: {metrics_rf['train']['r2']:.4f}, Val R²: {metrics_rf['val']['r2']:.4f}, Gap: {rf_overfit:.4f}")
    
    # Calculate win count
    wins_kbest = sum([
        metrics_kbest['val']['mse'] < metrics_rf['val']['mse'],
        metrics_kbest['val']['mae'] < metrics_rf['val']['mae'],
        metrics_kbest['val']['r2'] > metrics_rf['val']['r2'],
        metrics_kbest['test']['mse'] < metrics_rf['test']['mse'],
        metrics_kbest['test']['mae'] < metrics_rf['test']['mae'],
        metrics_kbest['test']['r2'] > metrics_rf['test']['r2']
    ])
    
    wins_rf = 6 - wins_kbest
    
    # Feature overlap analysis
    common_features = set(features_kbest) & set(features_rf)
    print(f"\n{BOLD}Feature Overlap:{RESET}")
    print(f"   Common features: {len(common_features)}/{n_features} ({len(common_features)/n_features*100:.1f}%)")
    print(f"   SelectKBest unique: {len(set(features_kbest) - set(features_rf))}")
    print(f"   RF Importance unique: {len(set(features_rf) - set(features_kbest))}")
    
    if len(common_features) < n_features * 0.5:
        print(f"   ⚠️  Low overlap - methods identify different feature sets!")
    
    # Final verdict
    print("\n" + "="*80)
    print(f"{BOLD}🏆 FINAL VERDICT{RESET}")
    print("="*80)
    
    if wins_rf > wins_kbest:
        print(f"{GREEN}✅ WINNER: RandomForest Feature Importance{RESET}")
        print(f"   Won {wins_rf}/6 metrics")
        print(f"\n   {BOLD}Recommendation:{RESET} Use feature_selection_rf() for better performance")
        print(f"   - Captures non-linear relationships")
        print(f"   - Better suited for tree-based models (RF, XGB)")
        print(f"   - Test MAE: {metrics_rf['test']['mae']:.6f} vs {metrics_kbest['test']['mae']:.6f}")
    elif wins_kbest > wins_rf:
        print(f"{GREEN}✅ WINNER: SelectKBest (Linear Correlation){RESET}")
        print(f"   Won {wins_kbest}/6 metrics")
        print(f"\n   {BOLD}Recommendation:{RESET} Keep current feature_selection() method")
        print(f"   - Simpler and faster")
        print(f"   - Linear correlation sufficient for this dataset")
        print(f"   - Test MAE: {metrics_kbest['test']['mae']:.6f} vs {metrics_rf['test']['mae']:.6f}")
    else:
        print(f"{YELLOW}⚖️  TIE: Both methods perform similarly{RESET}")
        print(f"   Each won {wins_kbest}/6 metrics")
        print(f"\n   {BOLD}Recommendation:{RESET} Either method is fine")
        print(f"   - RF Importance: Better for non-linear relationships")
        print(f"   - SelectKBest: Faster and simpler")
    
    print("="*80 + "\n")
    
    return {
        'kbest': metrics_kbest,
        'rf': metrics_rf,
        'winner': 'RF Importance' if wins_rf > wins_kbest else 'SelectKBest' if wins_kbest > wins_rf else 'Tie'
    }


if __name__ == '__main__':
    # Test with default stock and feature count
    try:
        results = compare_feature_selection_methods(stock_symbol="DEMANT.CO", n_features=50)
    except Exception as e:
        print(f"{RED}❌ Error during comparison: {e}{RESET}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
