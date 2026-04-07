"""
Price Predictor Module — Phase 2 of the portfolio pipeline.

Generates future stock price predictions and Monte Carlo simulations
for tickers that have trained models but no recent predictions.

Execution strategy:
    1. Query database for tickers with valid models but missing/stale predictions
    2. For each ticker: rebuild models from cached hyperparameters, generate forecasts
    3. Export predictions (with confidence intervals) and MC results to database
    4. Skip tickers that already have fresh predictions

This module can be run independently of model_trainer.py and portfolio_builder.py.
It uses the database as the single source of truth for model and prediction freshness.

Usage:
    # Predict all stocks that need predictions
    python price_predictor.py

    # Or import and call programmatically
    from price_predictor import run_predictions
    run_predictions(max_prediction_age_days=1, investment_years=7)
"""

import os
import sys
import io
import json
import time
import datetime
import traceback
from typing import List, Optional, Dict

# Suppress TF warnings
os.environ['TF_PTXAS_UNAVAILABLE'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

import fetch_secrets
import db_connectors
import db_interactions
import split_dataset
import dimension_reduction
import ml_builder
import monte_carlo_sim
from blacklist_manager import get_blacklist_manager


# ---------------------------------------------------------------------------
# Logging utilities — capture all console output to a file for analysis
# ---------------------------------------------------------------------------
PREDICTION_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prediction_logs")


class TeeLogger:
    """Duplicate stdout/stderr to a log file while still printing to console."""

    def __init__(self, log_path: str):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self._log_file = open(log_path, "w", encoding="utf-8")
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self._stdout_tee = self._Tee(self._original_stdout, self._log_file)
        self._stderr_tee = self._Tee(self._original_stderr, self._log_file)

    class _Tee:
        def __init__(self, console_stream, file_stream):
            self.console = console_stream
            self.file = file_stream

        def write(self, data):
            self.console.write(data)
            try:
                self.file.write(data)
            except (ValueError, OSError):
                pass  # file already closed

        def flush(self):
            self.console.flush()
            try:
                self.file.flush()
            except (ValueError, OSError):
                pass

        # required so other libraries treat this as a real file-like object
        def fileno(self):
            return self.console.fileno()

        def isatty(self):
            return False

    def __enter__(self):
        sys.stdout = self._stdout_tee
        sys.stderr = self._stderr_tee
        return self

    def __exit__(self, *exc):
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
        self._log_file.close()
        return False


def _save_run_summary(summary: dict, ticker_details: dict, log_dir: str):
    """Save a structured JSON summary alongside the raw log."""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(log_dir, f"prediction_summary_{ts}.json")

    # Build a JSON-safe version
    out = {
        "run_timestamp": datetime.datetime.now().isoformat(),
        "total_processed": summary.get("total_processed", 0),
        "successful": summary.get("successful", 0),
        "failed": summary.get("failed", 0),
        "execution_time_seconds": round(summary.get("execution_time", 0), 2),
        "tickers": {},
    }

    for ticker, res in ticker_details.items():
        entry = {
            "success": res["success"],
            "execution_time_seconds": round(res.get("execution_time", 0), 2),
            "error_message": res.get("error_message"),
        }
        if res.get("forecast_df") is not None:
            fdf = res["forecast_df"]
            entry["forecast_rows"] = len(fdf)
            if "close_Price" in fdf.columns:
                entry["first_price"] = round(float(fdf["close_Price"].iloc[0]), 4)
                entry["last_price"] = round(float(fdf["close_Price"].iloc[-1]), 4)
                entry["min_price"] = round(float(fdf["close_Price"].min()), 4)
                entry["max_price"] = round(float(fdf["close_Price"].max()), 4)
        out["tickers"][ticker] = entry

    os.makedirs(log_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[LOG] Saved JSON summary → {path}")


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


def save_prediction_graph(stock_data_df: pd.DataFrame, forecast_df: pd.DataFrame):
    """Save the prediction graph to generated_graphs folder."""
    plt.figure(figsize=(12, 6))
    plt.plot(forecast_df["close_Price"], color="green")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend(["Predicted Stock Price"], loc="best")

    stock_name = stock_data_df.iloc[0]["ticker"]
    graph_name = f"future_stock_prediction_of_{stock_name}.png"
    my_path = os.path.abspath(__file__)
    path = os.path.dirname(my_path)

    try:
        plt.savefig(
            os.path.join(path, "generated_graphs", graph_name),
            bbox_inches="tight", pad_inches=0.5, transparent=False, format="png"
        )
    except FileNotFoundError:
        print(f"[WARNING] Could not save graph for {stock_name}")
    finally:
        plt.clf()
        plt.close("all")


def predict_single_stock(
    stock_symbol: str,
    investment_years: int = 7,
    time_steps: int = 30
) -> Dict:
    """
    Generate predictions for a single stock using its cached model hyperparameters.
    
    This re-builds models from cached hyperparameters (fast — no tuning),
    generates forecasts, runs Monte Carlo, and exports everything to the database.
    
    Args:
        stock_symbol: Stock ticker symbol
        investment_years: Investment horizon for Monte Carlo simulation
        time_steps: Number of time steps for sequence models
        
    Returns:
        dict with 'success', 'forecast_df', 'mc_day_df', 'mc_year_df',
              'error_message', 'execution_time'
    """
    start_time = time.time()

    try:
        # Import stock data
        stock_data_df = db_interactions.import_stock_dataset(stock_symbol)
        stock_data_df["date"] = pd.to_datetime(stock_data_df["date"])

        # --- Targeted cleaning (matches model_trainer.py) ---
        # 1. Drop columns that are entirely NaN
        stock_data_df = stock_data_df.dropna(axis=1, how="all")
        # 2. Only drop rows where critical columns have NaN
        always_required = ['date', 'ticker', 'close_Price', 'open_Price',
                           'high_Price', 'low_Price']
        critical_cols = [c for c in always_required if c in stock_data_df.columns]
        stock_data_df = stock_data_df.dropna(subset=critical_cols)
        # 3. Forward-fill then back-fill remaining NaN in feature columns
        feature_cols = [c for c in stock_data_df.columns
                        if c not in ('date', 'ticker')]
        stock_data_df[feature_cols] = stock_data_df[feature_cols].ffill().bfill()
        # 4. Drop any rows still containing NaN after fill
        stock_data_df = stock_data_df.dropna(axis=0, how="any")
        # 5. Drop columns that became all-NaN after row removal
        stock_data_df = stock_data_df.dropna(axis=1, how="any")

        if len(stock_data_df) < 100:
            raise ValueError(f"Insufficient data: only {len(stock_data_df)} rows available")

        # Split the dataset
        validation_size = 0.20
        test_size = 0.10
        scaler_x, scaler_y, x_train_scaled, x_val_scaled, x_test_scaled, \
            y_train_scaled, y_val_scaled, y_test_scaled, x_predictions = \
            split_dataset.dataset_train_test_split(
                stock_data_df, test_size, validation_size=validation_size
            )

        # Inverse-transform y values for RF/XGB
        y_train_unscaled = scaler_y.inverse_transform(
            y_train_scaled.reshape(-1, 1)
        ).flatten()
        y_val_unscaled = scaler_y.inverse_transform(
            y_val_scaled.reshape(-1, 1)
        ).flatten()
        y_test_unscaled = scaler_y.inverse_transform(
            y_test_scaled.reshape(-1, 1)
        ).flatten()

        # Feature selection
        x_training_data = pd.DataFrame(x_train_scaled)
        x_val_data = pd.DataFrame(x_val_scaled)
        x_test_data = pd.DataFrame(x_test_scaled)
        y_training_data_df = pd.Series(y_train_unscaled)
        y_val_data_df = pd.Series(y_val_unscaled)
        y_test_data_df = pd.Series(y_test_unscaled)
        prediction_data = x_predictions

        max_features = len(x_training_data.columns)
        feature_amount = max_features

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
        x_prediction_dataset_df = pd.DataFrame(
            x_prediction_dataset, columns=selected_features_list
        )

        y_train_scaled_for_lstm = pd.Series(y_train_scaled)
        y_test_scaled_for_lstm = pd.Series(y_test_scaled)
        y_val_scaled_for_lstm = pd.Series(y_val_scaled)

        # Rebuild models from cached hyperparameters
        # train_and_validate_models will use cached HPs when available (no tuning overhead)
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
            use_tcn=True,
            tcn_trials=30,
            tcn_epochs=100,
            tcn_retrain_increment=10
        )

        # Generate predictions
        amount_of_days = time_steps * 3
        forecast_df = ml_builder.predict_future_price_changes(
            ticker=stock_symbol,
            scaler_x=scaler_x,
            scaler_y=scaler_y,
            model=models,
            selected_features_list=selected_features_list,
            stock_df=stock_data_df,
            prediction_days=amount_of_days,
            time_steps=time_steps,
            historical_prediction_dataset_df=x_prediction_dataset_df,
            use_mc_dropout=True,
            mc_iterations=30
        )

        # Analyze prediction performance
        historical_pred_count = len(x_prediction_dataset_df) \
            if x_prediction_dataset_df is not None else 0
        ml_builder.analyze_prediction_performance(stock_data_df, forecast_df, historical_pred_count)

        # Save prediction graph
        save_prediction_graph(stock_data_df, forecast_df)

        # Calculate predicted profit
        ml_builder.calculate_predicted_profit(forecast_df, amount_of_days)

        # Plot detailed graph
        ml_builder.plot_graph(stock_data_df, forecast_df)

        # Run Monte Carlo simulation
        sim_amount = 1000
        monte_carlo_day_df, monte_carlo_year_df = monte_carlo_sim.monte_carlo_analysis(
            0, stock_data_df, forecast_df, investment_years, sim_amount
        )

        # Get current price for database export
        current_price = float(stock_data_df['close_Price'].iloc[-1])

        # Export predictions to database
        prediction_date = datetime.date.today()

        try:
            db_interactions.export_stock_prediction_extended(
                ticker=stock_symbol,
                prediction_date=prediction_date,
                forecast_df=forecast_df,
                current_price=current_price,
                model_type="ensemble",
                mc_dropout_used=True,
                mc_iterations=30
            )
            print(f"[DB] Exported predictions for {stock_symbol}")
        except Exception as db_error:
            print(f"[WARNING] Could not export predictions to DB: {db_error}")

        try:
            db_interactions.export_monte_carlo_results(
                ticker=stock_symbol,
                simulation_date=prediction_date,
                monte_carlo_year_df=monte_carlo_year_df,
                num_simulations=sim_amount,
                starting_price=current_price
            )
            print(f"[DB] Exported Monte Carlo results for {stock_symbol}")
        except Exception as db_error:
            print(f"[WARNING] Could not export Monte Carlo to DB: {db_error}")

        execution_time = time.time() - start_time

        return {
            'success': True,
            'forecast_df': forecast_df,
            'mc_day_df': monte_carlo_day_df,
            'mc_year_df': monte_carlo_year_df,
            'error_message': None,
            'execution_time': execution_time
        }

    except Exception as e:
        execution_time = time.time() - start_time
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"[ERROR] Failed predicting {stock_symbol}: {error_msg}")
        print(traceback.format_exc())
        return {
            'success': False,
            'forecast_df': None,
            'mc_day_df': None,
            'mc_year_df': None,
            'error_message': error_msg,
            'execution_time': execution_time
        }


def run_predictions(
    max_prediction_age_days: int = 1,
    investment_years: int = 7,
    excluded_tickers: Optional[List[str]] = None,
    time_steps: int = 30,
    max_stocks: Optional[int] = None
):
    """
    Main entry point: generate predictions for stocks that need them.
    
    Strategy:
        1. Find tickers with valid models but no recent predictions → predict those first
        2. Skip tickers that already have fresh predictions
    
    Args:
        max_prediction_age_days: Predictions older than this are regenerated (default: 1)
        investment_years: Investment horizon for Monte Carlo simulations
        excluded_tickers: Tickers to skip
        time_steps: Time steps for sequence models
        max_stocks: Maximum number of stocks to predict in this run (None = all)
        
    Returns:
        dict with prediction summary
    """
    overall_start = time.time()

    print("\n" + "=" * 70)
    print("PRICE PREDICTOR — Phase 2")
    print("=" * 70)
    print(f"Max prediction age: {max_prediction_age_days} day(s)")
    print(f"Investment horizon: {investment_years} years")
    print("=" * 70 + "\n")

    # Configure GPU
    has_gpu = configure_gpu()
    print(f"[GPU] {'GPU acceleration enabled' if has_gpu else 'Running on CPU'}\n")

    # Load blacklist
    blacklisted = get_blacklist_manager().get_blacklist()
    all_excluded = list(set((excluded_tickers or []) + blacklisted))

    # Query DB for prediction freshness
    prediction_needs = db_interactions.get_tickers_needing_prediction(
        max_prediction_age_days=max_prediction_age_days
    )

    needs_prediction = [t for t in prediction_needs['needs_prediction'] if t not in all_excluded]
    recently_predicted = prediction_needs['recently_predicted']

    print(f"[STATUS] Tickers needing prediction:  {len(needs_prediction)}")
    print(f"[STATUS] Recently predicted:           {len(recently_predicted)}")
    print(f"[STATUS] Excluded:                     {len(all_excluded)}\n")

    # Limit work queue
    work_queue = needs_prediction
    if max_stocks is not None:
        work_queue = work_queue[:max_stocks]

    if not work_queue:
        print("[INFO] All predictions are up to date. Nothing to predict.")
        return {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'execution_time': time.time() - overall_start
        }

    total = len(work_queue)
    print(f"[INFO] Predicting {total} tickers\n")

    # Process each ticker
    results = {}
    successful = 0
    failed = 0

    for i, ticker in enumerate(work_queue):
        print(f"\n{'=' * 60}")
        print(f"[{i+1}/{total}] Predicting {ticker}")
        print(f"{'=' * 60}")

        result = predict_single_stock(
            stock_symbol=ticker,
            investment_years=investment_years,
            time_steps=time_steps
        )
        results[ticker] = result

        if result['success']:
            successful += 1
            print(f"[OK] {ticker} predicted in {result['execution_time']:.1f}s")
        else:
            failed += 1
            print(f"[FAIL] {ticker}: {result['error_message']}")

    # Summary
    overall_time = time.time() - overall_start
    print("\n" + "=" * 70)
    print("PREDICTION SUMMARY")
    print("=" * 70)
    print(f"Total processed:   {total}")
    print(f"Successful:        {successful}")
    print(f"Failed:            {failed}")
    print(f"Total time:        {overall_time:.1f}s")
    if total > 0:
        print(f"Avg time/stock:    {overall_time / total:.1f}s")

    if failed > 0:
        print("\nFailed tickers:")
        for ticker, result in results.items():
            if not result['success']:
                print(f"  - {ticker}: {result['error_message']}")

    print("=" * 70)

    # Save structured JSON summary for post-run analysis
    _save_run_summary(
        {"total_processed": total, "successful": successful,
         "failed": failed, "execution_time": overall_time},
        results,
        PREDICTION_LOG_DIR,
    )

    return {
        'total_processed': total,
        'successful': successful,
        'failed': failed,
        'execution_time': overall_time,
        'results': results
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate stock price predictions")
    parser.add_argument("--max-age", type=int, default=1,
                        help="Max prediction age in days before re-predicting (default: 1)")
    parser.add_argument("--years", type=int, default=7,
                        help="Investment horizon for Monte Carlo (default: 7)")
    parser.add_argument("--max-stocks", type=int, default=None,
                        help="Maximum number of stocks to predict in this run")
    parser.add_argument("--time-steps", type=int, default=30,
                        help="Time steps for sequence models (default: 30)")

    args = parser.parse_args()

    # Capture all console output to a timestamped log file
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(PREDICTION_LOG_DIR, f"prediction_run_{ts}.log")
    print(f"[LOG] Full output will be saved to: {log_path}")

    with TeeLogger(log_path):
        summary = run_predictions(
            max_prediction_age_days=args.max_age,
            investment_years=args.years,
            time_steps=args.time_steps,
            max_stocks=args.max_stocks
        )

    print(f"[LOG] Raw log saved → {log_path}")
