"""
Model Trainer Module — Phase 1 of the portfolio pipeline.

Trains and retrains ML prediction models (TCN/LSTM, Random Forest, XGBoost)
for each stock ticker independently.

Execution strategy:
    1. Query database for tickers that have NO trained models → train those first
    2. Query database for tickers with STALE models (>max_age_days) → retrain those
    3. Skip tickers whose models are all fresh

This module can be run independently of price_predictor.py and portfolio_builder.py.
It uses the database as the single source of truth for model freshness.

Usage:
    # Train all models that need it (untrained first, then stale)
    python model_trainer.py

    # Or import and call programmatically
    from model_trainer import run_model_training
    run_model_training(max_model_age_days=30)

GPU Configuration:
    Automatically detects and configures TensorFlow GPU with 7GB memory limit.
"""

import os
import sys
import time
import traceback
from typing import List, Optional, Dict

# Suppress TF warnings (level 2 = hide warnings + info, only show errors)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import tensorflow as tf

import fetch_secrets
import db_connectors
import db_interactions
import split_dataset
import dimension_reduction
import ml_builder
from blacklist_manager import get_blacklist_manager


def configure_gpu():
    """Configure TensorFlow GPU settings for optimal performance."""
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("[GPU] No GPU detected, using CPU.")
        return False
    try:
        for gpu in gpus:
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7168)]
            )
        print(f"[GPU] Configured {len(gpus)} GPU(s) with 7GB memory limit.")
        return True
    except RuntimeError as e:
        print(f"[GPU] Configuration failed ({e}), falling back to CPU.")
        try:
            tf.config.set_visible_devices([], 'GPU')
        except Exception:
            pass
        return False


def get_stock_symbols(excluded_tickers: List[str] = None) -> List[str]:
    """
    Import all stock symbols from database, applying exclusions.
    
    Args:
        excluded_tickers: Ticker symbols to exclude
        
    Returns:
        List of ticker symbol strings
    """
    stock_symbols_list = db_interactions.import_ticker_list()
    stock_symbols_df = pd.DataFrame(stock_symbols_list, columns=["Symbol"])

    if excluded_tickers:
        stock_symbols_df = stock_symbols_df[
            ~stock_symbols_df["Symbol"].isin(excluded_tickers)
        ]

    return stock_symbols_df["Symbol"].tolist()


def validate_data_availability(stock_symbol: str) -> Dict:
    """
    Check if a ticker has data in all required DB tables before training.
    
    Returns:
        dict with 'valid' (bool), 'missing_tables' (list), 'message' (str)
    """
    checks = {
        'stock_info_data': db_interactions.does_stock_exists_stock_info_data,
        'stock_price_data': db_interactions.does_stock_exists_stock_price_data,
        'stock_income_stmt_data': db_interactions.does_stock_exists_stock_income_stmt_data,
        'stock_balancesheet_data': db_interactions.does_stock_exists_stock_balancesheet_data,
        'stock_cash_flow_data': db_interactions.does_stock_exists_stock_cash_flow_data,
        'stock_ratio_data': db_interactions.does_stock_exists_stock_ratio_data,
    }
    
    missing = []
    for table_name, check_fn in checks.items():
        try:
            if not check_fn(stock_symbol):
                missing.append(table_name)
        except Exception:
            missing.append(table_name)
    
    if missing:
        return {
            'valid': False,
            'missing_tables': missing,
            'message': f"Missing data in: {', '.join(missing)}"
        }
    return {'valid': True, 'missing_tables': [], 'message': 'OK'}


def train_single_stock(
    stock_symbol: str,
    time_steps: int = 30,
    use_tcn: bool = True
) -> Dict:
    """
    Train all ML models for a single stock ticker.
    
    This runs the full pipeline: data fetch → split → feature selection → model training.
    The models' hyperparameters are saved to the database automatically by ml_builder.
    
    Args:
        stock_symbol: Stock ticker symbol
        time_steps: Number of time steps for sequence models
        use_tcn: Whether to use TCN (True) or LSTM (False) as sequence model
        
    Returns:
        dict with 'success', 'error_message', 'execution_time', 'skipped' keys
    """
    start_time = time.time()

    try:
        # Pre-validate data availability before expensive operations
        availability = validate_data_availability(stock_symbol)
        if not availability['valid']:
            execution_time = time.time() - start_time
            return {
                'success': False,
                'skipped': True,
                'error_message': f"Skipped (missing data): {availability['message']}",
                'execution_time': execution_time
            }

        # Import stock data
        stock_data_df = db_interactions.import_stock_dataset(stock_symbol)
        stock_data_df["date"] = pd.to_datetime(stock_data_df["date"])
        rows_before = len(stock_data_df)

        # --- Targeted cleaning instead of blanket dropna ---
        # 1. Drop columns that are entirely NaN (uninformative)
        stock_data_df = stock_data_df.dropna(axis=1, how="all")
        # 2. Identify columns critical for training (price, target, key features)
        #    Only drop rows where these critical columns have NaN.
        always_required = ['date', 'ticker', 'close_Price', 'open_Price',
                           'high_Price', 'low_Price']
        critical_cols = [c for c in always_required if c in stock_data_df.columns]
        stock_data_df = stock_data_df.dropna(subset=critical_cols)
        # 3. Forward-fill then back-fill remaining NaN in feature columns
        #    (technical indicators / ratios may have leading NaN from lookback)
        feature_cols = [c for c in stock_data_df.columns
                        if c not in ('date', 'ticker')]
        stock_data_df[feature_cols] = stock_data_df[feature_cols].ffill().bfill()
        # 4. Drop any rows still containing NaN after fill
        stock_data_df = stock_data_df.dropna(axis=0, how="any")
        # 5. Drop columns that became all-NaN after row removal
        stock_data_df = stock_data_df.dropna(axis=1, how="any")

        rows_after = len(stock_data_df)
        if rows_before != rows_after:
            print(f"   [{stock_symbol}] Cleaning: {rows_before} → {rows_after} rows "
                  f"({rows_before - rows_after} dropped)")

        # Minimum rows: 1 full trading year (252 days) ensures enough data for
        # train/val/test split with time_steps sequences.
        min_rows = max(252, time_steps + 50)
        if len(stock_data_df) < min_rows:
            return {
                'success': False,
                'skipped': True,
                'error_message': f"Insufficient data: {len(stock_data_df)} rows after cleaning (need >= {min_rows} for time_steps={time_steps})",
                'execution_time': time.time() - start_time
            }

        # Split the dataset
        validation_size = 0.20
        test_size = 0.10
        scaler_x, scaler_y, x_train_scaled, x_val_scaled, x_test_scaled, \
            y_train_scaled, y_val_scaled, y_test_scaled, x_predictions = \
            split_dataset.dataset_train_test_split(
                stock_data_df, test_size, validation_size=validation_size
            )

        # Inverse-transform y values for Random Forest / XGBoost
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
        feature_amount = max_features

        # Feature selection
        x_training_dataset, x_val_dataset, x_test_dataset, x_prediction_dataset, \
            selected_features_model, selected_features_list = \
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

        # Prepare DataFrames for ML
        x_training_dataset_df = pd.DataFrame(
            x_training_dataset, columns=selected_features_list
        )
        y_training_data_df = y_training_data_df.reset_index(drop=True)
        x_val_dataset_df = pd.DataFrame(x_val_dataset, columns=selected_features_list)
        y_val_data_df = y_val_data_df.reset_index(drop=True)
        x_test_dataset_df = pd.DataFrame(x_test_dataset, columns=selected_features_list)
        y_test_data_df = y_test_data_df.reset_index(drop=True)

        y_train_scaled_for_lstm = pd.Series(y_train_scaled)
        y_test_scaled_for_lstm = pd.Series(y_test_scaled)
        y_val_scaled_for_lstm = pd.Series(y_val_scaled)

        # Train ML models (hyperparameters are saved to DB automatically)
        models, training_history, lstm_datasets = ml_builder.train_and_validate_models(
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
            use_tcn=use_tcn,
            tcn_trials=30,
            tcn_epochs=100,
            tcn_retrain_increment=10
        )

        execution_time = time.time() - start_time
        return {
            'success': True,
            'skipped': False,
            'error_message': None,
            'execution_time': execution_time
        }

    except Exception as e:
        execution_time = time.time() - start_time
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"[ERROR] Failed training {stock_symbol}: {error_msg}")
        print(traceback.format_exc())
        return {
            'success': False,
            'skipped': False,
            'error_message': error_msg,
            'execution_time': execution_time
        }


def run_model_training(
    max_model_age_days: int = 30,
    excluded_tickers: Optional[List[str]] = None,
    time_steps: int = 30,
    use_tcn: bool = True,
    max_stocks: Optional[int] = None
):
    """
    Main entry point: train models for stocks that need it.
    
    Strategy:
        1. Identify untrained tickers → train those first
        2. Identify tickers with stale (>max_model_age_days) models → retrain
        3. Skip tickers that are fully fresh
    
    Args:
        max_model_age_days: Models older than this are retrained (default: 30)
        excluded_tickers: Tickers to skip entirely
        time_steps: Time steps for sequence models
        use_tcn: Use TCN (True) or LSTM (False)
        max_stocks: Maximum number of stocks to process in this run (None = all)
        
    Returns:
        dict with training summary
    """
    overall_start = time.time()

    print("\n" + "=" * 70)
    print("MODEL TRAINER — Phase 1")
    print("=" * 70)
    print(f"Max model age: {max_model_age_days} days")
    print(f"Sequence model: {'TCN' if use_tcn else 'LSTM'}")
    print("=" * 70 + "\n")

    # Configure GPU
    has_gpu = configure_gpu()
    print(f"[GPU] {'GPU acceleration enabled' if has_gpu else 'Running on CPU'}\n")

    # Load blacklisted tickers
    blacklisted = get_blacklist_manager().get_blacklist()
    all_excluded = list(set((excluded_tickers or []) + blacklisted))

    # Query DB for model freshness
    training_needs = db_interactions.get_tickers_needing_training(
        max_age_days=max_model_age_days,
        required_model_types=['rf', 'xgb', 'tcn' if use_tcn else 'lstm']
    )

    untrained = [t for t in training_needs['untrained'] if t not in all_excluded]
    stale = [t for t in training_needs['stale'] if t not in all_excluded]
    fresh = training_needs['fresh']

    print(f"[STATUS] Untrained tickers:  {len(untrained)}")
    print(f"[STATUS] Stale tickers:      {len(stale)}")
    print(f"[STATUS] Fresh tickers:      {len(fresh)}")
    print(f"[STATUS] Excluded tickers:   {len(all_excluded)}\n")

    # Build work queue: untrained first, then stale
    work_queue = untrained + stale
    if max_stocks is not None:
        work_queue = work_queue[:max_stocks]

    if not work_queue:
        print("[INFO] All models are up to date. Nothing to train.")
        return {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'untrained_remaining': 0,
            'stale_remaining': 0,
            'execution_time': time.time() - overall_start
        }

    total = len(work_queue)
    print(f"[INFO] Processing {total} tickers ({len(untrained)} untrained + "
          f"{min(len(stale), total - len(untrained))} stale)\n")

    # Process each ticker
    results = {}
    successful = 0
    failed = 0
    skipped = 0

    for i, ticker in enumerate(work_queue):
        category = "UNTRAINED" if ticker in untrained else "STALE"
        print(f"\n{'=' * 60}")
        print(f"[{i+1}/{total}] Training {ticker} [{category}]")
        print(f"{'=' * 60}")

        result = train_single_stock(
            stock_symbol=ticker,
            time_steps=time_steps,
            use_tcn=use_tcn
        )
        results[ticker] = result

        if result['success']:
            successful += 1
            print(f"[OK] {ticker} trained in {result['execution_time']:.1f}s")
        elif result.get('skipped', False):
            skipped += 1
            print(f"[SKIP] {ticker}: {result['error_message']}")
        else:
            failed += 1
            print(f"[FAIL] {ticker}: {result['error_message']}")

    # Summary
    overall_time = time.time() - overall_start
    print("\n" + "=" * 70)
    print("MODEL TRAINING SUMMARY")
    print("=" * 70)
    print(f"Total in queue:    {total}")
    print(f"Successful:        {successful}")
    print(f"Skipped (no data): {skipped}")
    print(f"Failed:            {failed}")
    print(f"Total time:        {overall_time:.1f}s")
    if successful > 0:
        print(f"Avg time/trained:  {overall_time / successful:.1f}s")

    if skipped > 0:
        print("\nSkipped tickers (missing data in DB):")
        for ticker, result in results.items():
            if result.get('skipped', False):
                print(f"  - {ticker}: {result['error_message']}")

    if failed > 0:
        print("\nFailed tickers:")
        for ticker, result in results.items():
            if not result['success'] and not result.get('skipped', False):
                print(f"  - {ticker}: {result['error_message']}")

    print("=" * 70)

    return {
        'total_processed': total,
        'successful': successful,
        'failed': failed,
        'untrained_remaining': max(0, len(untrained) - successful),
        'stale_remaining': len(stale) - sum(
            1 for t in stale if results.get(t, {}).get('success', False)
        ),
        'execution_time': overall_time,
        'results': results
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train ML models for stock tickers")
    parser.add_argument("--max-age", type=int, default=30,
                        help="Max model age in days before retraining (default: 30)")
    parser.add_argument("--max-stocks", type=int, default=None,
                        help="Maximum number of stocks to process in this run")
    parser.add_argument("--use-lstm", action="store_true",
                        help="Use LSTM instead of TCN as sequence model")
    parser.add_argument("--time-steps", type=int, default=30,
                        help="Time steps for sequence models (default: 30)")

    args = parser.parse_args()

    summary = run_model_training(
        max_model_age_days=args.max_age,
        time_steps=args.time_steps,
        use_tcn=not args.use_lstm,
        max_stocks=args.max_stocks
    )
