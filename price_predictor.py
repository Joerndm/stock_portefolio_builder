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
import json
import time
import datetime
import logging
import traceback
from typing import List, Optional, Dict

# Suppress TF warnings
os.environ['TF_PTXAS_UNAVAILABLE'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import db_interactions
import ml_builder
import monte_carlo_sim
from gpu_runtime_utils import configure_tensorflow_gpu
from blacklist_manager import get_blacklist_manager
from model_pipeline_preprocessing import prepare_modeling_data
from pipeline_config import get_gpu_config, get_data_config, get_ml_config, get_pred_config
from prediction_cache_contract import PredictionCacheContractError, require_prediction_cache

logger = logging.getLogger(__name__)


DB_EXPORT_RECOVERY_EXCEPTIONS = (KeyError, ValueError)


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
        "status_counts": summary.get("status_counts", {}),
        "execution_time_seconds": round(summary.get("execution_time", 0), 2),
        "tickers": {},
    }

    for ticker, res in ticker_details.items():
        entry = {
            "success": res["success"],
            "execution_time_seconds": round(res.get("execution_time", 0), 2),
            "error_message": res.get("error_message"),
            "cache_status": res.get("cache_status"),
            "required_model_types": res.get("required_model_types"),
            "available_model_types": res.get("available_model_types"),
            "missing_model_types": res.get("missing_model_types"),
            "failing_model_type": res.get("failing_model_type"),
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
    gpu_cfg = get_gpu_config()
    return configure_tensorflow_gpu(gpu_cfg.memory_limit_mb, logger=logger)


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
    time_steps: int = None
) -> Dict:
    """
    Generate predictions for a single stock using its cached model hyperparameters.
    
    This re-builds models from cached hyperparameters (fast — no tuning),
    generates forecasts, runs Monte Carlo, and exports everything to the database.
    
    Args:
        stock_symbol: Stock ticker symbol
        investment_years: Investment horizon for Monte Carlo simulation
        time_steps: Number of time steps for sequence models (None = use config)
        
    Returns:
        dict with 'success', 'forecast_df', 'mc_day_df', 'mc_year_df',
              'error_message', 'execution_time'
    """
    data_cfg = get_data_config()
    ml_cfg = get_ml_config()
    pred_cfg = get_pred_config()
    if time_steps is None:
        time_steps = data_cfg.time_steps
    start_time = time.time()

    try:
        cache_state = require_prediction_cache(
            stock_symbol,
            required_model_types=ml_cfg.required_model_types,
            max_age_days=pred_cfg.max_model_age_days,
        )

        # Import stock data
        stock_data_df = db_interactions.import_stock_dataset(stock_symbol)
        prepared_data = prepare_modeling_data(
            stock_data_df,
            time_steps=time_steps,
            validation_size=data_cfg.validation_size,
            test_size=data_cfg.test_size,
            min_rows_floor=data_cfg.min_rows_floor,
        )
        stock_data_df = prepared_data.stock_data_df

        # Rebuild models from cached hyperparameters
        # train_and_validate_models will use cached HPs when available (no tuning overhead)
        models, _, _ = ml_builder.train_and_validate_models(
            stock_symbol=stock_symbol,
            x_train=prepared_data.x_training_dataset_df.values,
            x_val=prepared_data.x_val_dataset_df.values,
            x_test=prepared_data.x_test_dataset_df.values,
            y_train_scaled=prepared_data.y_train_scaled,
            y_val_scaled=prepared_data.y_val_scaled,
            y_test_scaled=prepared_data.y_test_scaled,
            y_train_unscaled=prepared_data.y_train_unscaled,
            y_val_unscaled=prepared_data.y_val_unscaled,
            y_test_unscaled=prepared_data.y_test_unscaled,
            time_steps=time_steps,
            scaler_y=prepared_data.scaler_y,
            max_retrains=ml_cfg.max_retrains,
            overfitting_threshold=ml_cfg.overfitting_threshold,
            lstm_trials=ml_cfg.lstm_trials,
            lstm_executions=ml_cfg.lstm_executions,
            lstm_epochs=ml_cfg.lstm_epochs,
            lstm_retrain_trials_increment=ml_cfg.lstm_retrain_trials_increment,
            lstm_retrain_executions_increment=ml_cfg.lstm_retrain_executions_increment,
            rf_trials=ml_cfg.rf_trials,
            rf_retrain_increment=ml_cfg.rf_retrain_increment,
            xgb_trials=ml_cfg.xgb_trials,
            xgb_retrain_increment=ml_cfg.xgb_retrain_increment,
            use_multi_metric_detection=ml_cfg.use_multi_metric_detection,
            use_tcn=ml_cfg.use_tcn,
            use_sequence_model=ml_cfg.use_sequence_model,
            tcn_trials=ml_cfg.tcn_trials,
            tcn_epochs=ml_cfg.tcn_epochs,
            tcn_retrain_increment=ml_cfg.tcn_retrain_increment,
            cache_max_age_days=pred_cfg.max_model_age_days,
            cache_only=True,
        )

        # Generate predictions
        amount_of_days = time_steps * pred_cfg.prediction_days_multiplier
        forecast_df = ml_builder.predict_future_price_changes(
            ticker=stock_symbol,
            scaler_x=prepared_data.scaler_x,
            scaler_y=prepared_data.scaler_y,
            model=models,
            selected_features_list=prepared_data.selected_features_list,
            stock_df=stock_data_df,
            prediction_days=amount_of_days,
            time_steps=time_steps,
            historical_prediction_dataset_df=prepared_data.x_prediction_dataset_df,
            use_mc_dropout=pred_cfg.use_mc_dropout,
            mc_iterations=pred_cfg.mc_iterations,
        )

        # Analyze prediction performance
        historical_pred_count = len(prepared_data.x_prediction_dataset_df) \
            if prepared_data.x_prediction_dataset_df is not None else 0
        ml_builder.analyze_prediction_performance(stock_data_df, forecast_df, historical_pred_count)

        # Save prediction graph
        save_prediction_graph(stock_data_df, forecast_df)

        # Calculate predicted profit
        ml_builder.calculate_predicted_profit(forecast_df, amount_of_days)

        # Plot detailed graph
        ml_builder.plot_graph(stock_data_df, forecast_df)

        # Run Monte Carlo simulation
        monte_carlo_day_df, monte_carlo_year_df = monte_carlo_sim.monte_carlo_analysis(
            0, stock_data_df, forecast_df, investment_years, pred_cfg.sim_amount
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
                mc_dropout_used=pred_cfg.use_mc_dropout,
                mc_iterations=pred_cfg.mc_iterations,
            )
            # Export per-model predictions if available
            model_price_cols = {
                'rf': 'price_rf', 'xgb': 'price_xgb',
                'ridge': 'price_ridge', 'svr': 'price_svr',
                'seq': 'price_seq'
            }
            for model_name, col_name in model_price_cols.items():
                if col_name in forecast_df.columns and forecast_df[col_name].notna().any():
                    model_forecast = forecast_df.copy()
                    model_forecast['close_Price'] = model_forecast[col_name].combine_first(
                        model_forecast['close_Price']
                    )
                    try:
                        db_interactions.export_stock_prediction_extended(
                            ticker=stock_symbol,
                            prediction_date=prediction_date,
                            forecast_df=model_forecast,
                            current_price=current_price,
                            model_type=model_name,
                            mc_dropout_used=False,
                            mc_iterations=0
                        )
                    except DB_EXPORT_RECOVERY_EXCEPTIONS:
                        pass  # Non-critical, skip silently
            print(f"[DB] Exported predictions for {stock_symbol}")
        except DB_EXPORT_RECOVERY_EXCEPTIONS as db_error:
            print(f"[WARNING] Could not export predictions to DB: {db_error}")

        try:
            db_interactions.export_monte_carlo_results(
                ticker=stock_symbol,
                simulation_date=prediction_date,
                monte_carlo_year_df=monte_carlo_year_df,
                num_simulations=pred_cfg.sim_amount,
                starting_price=current_price
            )
            print(f"[DB] Exported Monte Carlo results for {stock_symbol}")
        except DB_EXPORT_RECOVERY_EXCEPTIONS as db_error:
            print(f"[WARNING] Could not export Monte Carlo to DB: {db_error}")

        execution_time = time.time() - start_time

        return {
            'success': True,
            'forecast_df': forecast_df,
            'mc_day_df': monte_carlo_day_df,
            'mc_year_df': monte_carlo_year_df,
            'error_message': None,
            'execution_time': execution_time,
            **cache_state.to_result(),
        }

    except PredictionCacheContractError as e:
        execution_time = time.time() - start_time
        print(f"[CACHE] {e}")
        return {
            'success': False,
            'forecast_df': None,
            'mc_day_df': None,
            'mc_year_df': None,
            'error_message': str(e),
            'execution_time': execution_time,
            **e.to_result(),
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
            'execution_time': execution_time,
            'cache_status': 'error',
        }


def run_predictions(
    max_prediction_age_days: int = 1,
    investment_years: int = 7,
    excluded_tickers: Optional[List[str]] = None,
    time_steps: int = None,
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
        time_steps: Time steps for sequence models (None = use config)
        max_stocks: Maximum number of stocks to predict in this run (None = all)
        
    Returns:
        dict with prediction summary
    """
    data_cfg = get_data_config()
    if time_steps is None:
        time_steps = data_cfg.time_steps

    overall_start = time.time()

    logger.info("")
    logger.info("=" * 70)
    logger.info("PRICE PREDICTOR — Phase 2")
    logger.info("=" * 70)
    logger.info("Max prediction age: %d day(s)", max_prediction_age_days)
    logger.info("Investment horizon: %d years", investment_years)
    logger.info("=" * 70)

    # Configure GPU
    has_gpu = configure_gpu()
    logger.info("[GPU] %s", 'GPU acceleration enabled' if has_gpu else 'Running on CPU')

    # Load blacklist
    blacklisted = get_blacklist_manager().get_blacklist()
    all_excluded = list(set((excluded_tickers or []) + blacklisted))

    # Query DB for prediction freshness
    prediction_needs = db_interactions.get_tickers_needing_prediction(
        max_prediction_age_days=max_prediction_age_days
    )

    needs_prediction = [t for t in prediction_needs['needs_prediction'] if t not in all_excluded]
    recently_predicted = prediction_needs['recently_predicted']

    logger.info("[STATUS] Tickers needing prediction:  %d", len(needs_prediction))
    logger.info("[STATUS] Recently predicted:           %d", len(recently_predicted))
    logger.info("[STATUS] Excluded:                     %d", len(all_excluded))

    # Limit work queue
    work_queue = needs_prediction
    if max_stocks is not None:
        work_queue = work_queue[:max_stocks]

    if not work_queue:
        logger.info("[INFO] All predictions are up to date. Nothing to predict.")
        return {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'execution_time': time.time() - overall_start,
            'status_counts': {},
        }

    total = len(work_queue)
    logger.info("[INFO] Predicting %d tickers", total)

    # Process each ticker
    results = {}
    successful = 0
    failed = 0
    status_counts: Dict[str, int] = {}

    for i, ticker in enumerate(work_queue):
        logger.info("")
        logger.info("=" * 60)
        logger.info("[%d/%d] Predicting %s", i + 1, total, ticker)
        logger.info("=" * 60)

        result = predict_single_stock(
            stock_symbol=ticker,
            investment_years=investment_years,
            time_steps=time_steps
        )
        results[ticker] = result
        status = result.get('cache_status') or ('cache_hit' if result.get('success') else 'error')
        status_counts[status] = status_counts.get(status, 0) + 1

        if result['success']:
            successful += 1
            logger.info("[OK] %s predicted in %.1fs", ticker, result['execution_time'])
        else:
            failed += 1
            logger.error("[FAIL] %s: %s", ticker, result['error_message'])

    # Summary
    overall_time = time.time() - overall_start
    logger.info("")
    logger.info("=" * 70)
    logger.info("PREDICTION SUMMARY")
    logger.info("=" * 70)
    logger.info("Total processed:   %d", total)
    logger.info("Successful:        %d", successful)
    logger.info("Failed:            %d", failed)
    logger.info("Total time:        %.1fs", overall_time)
    if total > 0:
        logger.info("Avg time/stock:    %.1fs", overall_time / total)
    if status_counts:
        logger.info("Cache statuses:")
        for status, count in sorted(status_counts.items()):
            logger.info("  - %s: %d", status, count)

    if failed > 0:
        logger.warning("Failed tickers:")
        for ticker, result in results.items():
            if not result['success']:
                logger.warning("  - %s: %s", ticker, result['error_message'])

    logger.info("=" * 70)

    # Save structured JSON summary for post-run analysis
    _save_run_summary(
        {"total_processed": total, "successful": successful,
         "failed": failed, "execution_time": overall_time,
         "status_counts": status_counts},
        results,
        PREDICTION_LOG_DIR,
    )

    return {
        'total_processed': total,
        'successful': successful,
        'failed': failed,
        'execution_time': overall_time,
        'status_counts': status_counts,
        'results': results
    }


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    parser = argparse.ArgumentParser(description="Generate stock price predictions")
    parser.add_argument("--max-age", type=int, default=30,
                        help="Max prediction age in days before re-predicting (default: 30)")
    parser.add_argument("--years", type=int, default=7,
                        help="Investment horizon for Monte Carlo (default: 7)")
    parser.add_argument("--max-stocks", type=int, default=None,
                        help="Maximum number of stocks to predict in this run")
    parser.add_argument("--time-steps", type=int, default=None,
                        help="Time steps for sequence models (default: from config)")

    args = parser.parse_args()

    # Capture all console output to a timestamped log file
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_path = os.path.join(PREDICTION_LOG_DIR, f"prediction_run_{run_timestamp}.log")
    logger.info("[LOG] Full output will be saved to: %s", run_log_path)

    with TeeLogger(run_log_path):
        run_summary = run_predictions(
            max_prediction_age_days=args.max_age,
            investment_years=args.years,
            time_steps=args.time_steps,
            max_stocks=args.max_stocks
        )

    logger.info("[LOG] Raw log saved → %s", run_log_path)
