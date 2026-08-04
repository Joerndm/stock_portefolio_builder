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
import time
import logging
import traceback
import json
from datetime import datetime, timedelta
from typing import List, Optional, Dict

# Suppress TF warnings (level 2 = hide warnings + info, only show errors)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd

import db_interactions
from cache_contract_admin import refresh_cache_contracts
from gpu_runtime_utils import configure_tensorflow_gpu
from blacklist_manager import get_blacklist_manager
from ml_runtime_loader import create_ml_builder_proxy
from model_pipeline_preprocessing import InsufficientDataError, prepare_modeling_data
from pipeline_config import get_gpu_config, get_data_config, get_ml_config

logger = logging.getLogger(__name__)

ml_builder = create_ml_builder_proxy()

# ---------------------------------------------------------------------------
# Insufficient-data skip cache
#
# Tickers that skip with InsufficientDataError stay "untrained" in the DB and
# would otherwise be re-picked at the front of every work queue, clogging it
# (and making scheduler cycles spin on the same skips forever). Unlike the
# blacklist, this must NOT be permanent: data-poor tickers accumulate rows
# daily and become trainable once they reach the minimum history. So skips
# are cached with a timestamp and retried after SKIP_RETRY_DAYS.
# ---------------------------------------------------------------------------
SKIP_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "insufficient_data_skips.json"
)
SKIP_RETRY_DAYS = 30


def _load_skip_cache() -> Dict[str, str]:
    """Load {ticker: iso_timestamp_of_last_skip}. Missing/corrupt file = empty."""
    try:
        with open(SKIP_CACHE_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _save_skip_cache(cache: Dict[str, str]) -> None:
    try:
        with open(SKIP_CACHE_PATH, "w", encoding="utf-8") as fh:
            json.dump(cache, fh, indent=2, sort_keys=True)
    except OSError as e:
        logger.warning("[SKIP-CACHE] Could not save %s: %s", SKIP_CACHE_PATH, e)


def _active_skips(cache: Dict[str, str], retry_days: int = SKIP_RETRY_DAYS) -> List[str]:
    """Tickers whose last insufficient-data skip is younger than retry_days."""
    cutoff = datetime.now() - timedelta(days=retry_days)
    active = []
    for ticker, stamp in cache.items():
        try:
            if datetime.fromisoformat(stamp) > cutoff:
                active.append(ticker)
        except (TypeError, ValueError):
            continue  # unparsable entry -> treat as expired, retry the ticker
    return active


def configure_gpu():
    """Configure TensorFlow GPU settings for optimal performance."""
    gpu_cfg = get_gpu_config()
    return configure_tensorflow_gpu(gpu_cfg.memory_limit_mb, logger=logger)


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
    time_steps: int = None,
    use_tcn: bool = None,
    use_sequence_model: bool = None
) -> Dict:
    """
    Train all ML models for a single stock ticker.
    
    This runs the full pipeline: data fetch → split → feature selection → model training.
    The models' hyperparameters are saved to the database automatically by ml_builder.
    
    Args:
        stock_symbol: Stock ticker symbol
        time_steps: Number of time steps for sequence models (None = use config)
        use_tcn: Whether to use TCN (True) or LSTM (False) (None = use config)
        use_sequence_model: Whether to train sequence models at all (None = use config)
        
    Returns:
        dict with 'success', 'error_message', 'execution_time', 'skipped' keys
    """
    data_cfg = get_data_config()
    ml_cfg = get_ml_config()
    if time_steps is None:
        time_steps = data_cfg.time_steps
    if use_tcn is None:
        use_tcn = ml_cfg.use_tcn
    if use_sequence_model is None:
        use_sequence_model = ml_cfg.use_sequence_model
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
        prepared_data = prepare_modeling_data(
            stock_data_df,
            time_steps=time_steps,
            validation_size=data_cfg.validation_size,
            test_size=data_cfg.test_size,
            min_rows_floor=data_cfg.min_rows_floor,
        )

        if prepared_data.dropped_rows > 0:
            logger.info("   [%s] Cleaning: %d → %d rows (%d dropped)",
                        stock_symbol,
                        prepared_data.rows_before_cleaning,
                        prepared_data.rows_after_cleaning,
                        prepared_data.dropped_rows)

        # Train ML models (hyperparameters are saved to DB automatically)
        _, _, _ = ml_builder.train_and_validate_models(
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
            use_tcn=use_tcn,
            use_sequence_model=use_sequence_model,
            tcn_trials=ml_cfg.tcn_trials,
            tcn_epochs=ml_cfg.tcn_epochs,
            tcn_retrain_increment=ml_cfg.tcn_retrain_increment
        )

        execution_time = time.time() - start_time
        return {
            'success': True,
            'skipped': False,
            'error_message': None,
            'execution_time': execution_time
        }

    except InsufficientDataError as e:
        return {
            'success': False,
            'skipped': True,
            'error_message': str(e),
            'execution_time': time.time() - start_time
        }

    except Exception as e:
        execution_time = time.time() - start_time
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error("[ERROR] Failed training %s: %s", stock_symbol, error_msg)
        logger.debug(traceback.format_exc())
        return {
            'success': False,
            'skipped': False,
            'error_message': error_msg,
            'execution_time': execution_time
        }


def run_model_training(
    max_model_age_days: int = 30,
    excluded_tickers: Optional[List[str]] = None,
    time_steps: int = None,
    use_tcn: bool = None,
    use_sequence_model: bool = None,
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
        time_steps: Time steps for sequence models (None = use config)
        use_tcn: Use TCN (True) or LSTM (False) (None = use config)
        max_stocks: Maximum number of stocks to process in this run (None = all)
        
    Returns:
        dict with training summary
    """
    ml_cfg = get_ml_config()
    data_cfg = get_data_config()
    if time_steps is None:
        time_steps = data_cfg.time_steps
    if use_tcn is None:
        use_tcn = ml_cfg.use_tcn
    if use_sequence_model is None:
        use_sequence_model = ml_cfg.use_sequence_model

    overall_start = time.time()

    logger.info("")
    logger.info("=" * 70)
    logger.info("MODEL TRAINER — Phase 1")
    logger.info("=" * 70)
    logger.info("Max model age: %d days", max_model_age_days)
    if use_sequence_model:
        logger.info("Sequence model: %s", 'TCN' if use_tcn else 'LSTM')
    else:
        logger.info("Sequence model: DISABLED (4-model ensemble: RF+XGB+Ridge+SVR)")
    logger.info("=" * 70)

    # Configure GPU
    has_gpu = configure_gpu()
    logger.info("[GPU] %s", 'GPU acceleration enabled' if has_gpu else 'Running on CPU')

    # Load blacklisted tickers
    blacklisted = get_blacklist_manager().get_blacklist()
    all_excluded = list(set((excluded_tickers or []) + blacklisted))

    # Query DB for model freshness
    training_needs = db_interactions.get_tickers_needing_training(
        max_age_days=max_model_age_days,
        required_model_types=ml_cfg.required_model_types
    )

    untrained = [t for t in training_needs['untrained'] if t not in all_excluded]
    stale = [t for t in training_needs['stale'] if t not in all_excluded]
    fresh = training_needs['fresh']

    # Exclude tickers that recently skipped on insufficient data; they are
    # retried automatically once their skip entry is older than SKIP_RETRY_DAYS.
    skip_cache = _load_skip_cache()
    recently_skipped = set(_active_skips(skip_cache))
    if recently_skipped:
        untrained = [t for t in untrained if t not in recently_skipped]
        stale = [t for t in stale if t not in recently_skipped]

    logger.info("[STATUS] Untrained tickers:  %d", len(untrained))
    logger.info("[STATUS] Stale tickers:      %d", len(stale))
    logger.info("[STATUS] Fresh tickers:      %d", len(fresh))
    logger.info("[STATUS] Excluded tickers:   %d", len(all_excluded))
    logger.info("[STATUS] Data-skip deferred: %d (retry after %dd)",
                len(recently_skipped), SKIP_RETRY_DAYS)

    # Build work queue: untrained first, then stale
    work_queue = untrained + stale
    if max_stocks is not None:
        work_queue = work_queue[:max_stocks]

    if not work_queue:
        logger.info("[INFO] All models are up to date. Nothing to train.")
        return {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'untrained_remaining': 0,
            'stale_remaining': 0,
            'execution_time': time.time() - overall_start
        }

    total = len(work_queue)
    n_untrained_in_queue = sum(1 for t in work_queue if t in untrained)
    logger.info("[INFO] Processing %d tickers (%d untrained + %d stale)",
                total, n_untrained_in_queue, total - n_untrained_in_queue)

    # Process each ticker
    results = {}
    successful = 0
    failed = 0
    skipped = 0

    for i, ticker in enumerate(work_queue):
        category = "UNTRAINED" if ticker in untrained else "STALE"
        logger.info("")
        logger.info("=" * 60)
        logger.info("[%d/%d] Training %s [%s]", i + 1, total, ticker, category)
        logger.info("=" * 60)

        result = train_single_stock(
            stock_symbol=ticker,
            time_steps=time_steps,
            use_tcn=use_tcn,
            use_sequence_model=use_sequence_model
        )
        results[ticker] = result

        if result['success']:
            successful += 1
            logger.info("[OK] %s trained in %.1fs", ticker, result['execution_time'])
            if ticker in skip_cache:  # trained successfully -> clear old skip
                skip_cache.pop(ticker, None)
                _save_skip_cache(skip_cache)
        elif result.get('skipped', False):
            skipped += 1
            logger.info("[SKIP] %s: %s", ticker, result['error_message'])
            skip_cache[ticker] = datetime.now().isoformat(timespec="seconds")
            _save_skip_cache(skip_cache)
        else:
            failed += 1
            logger.error("[FAIL] %s: %s", ticker, result['error_message'])

    # Summary
    overall_time = time.time() - overall_start
    logger.info("")
    logger.info("=" * 70)
    logger.info("MODEL TRAINING SUMMARY")
    logger.info("=" * 70)
    logger.info("Total in queue:    %d", total)
    logger.info("Successful:        %d", successful)
    logger.info("Skipped (no data): %d", skipped)
    logger.info("Failed:            %d", failed)
    logger.info("Total time:        %.1fs", overall_time)
    if successful > 0:
        logger.info("Avg time/trained:  %.1fs", overall_time / successful)

    if skipped > 0:
        logger.info("Skipped tickers (missing data in DB):")
        for ticker, result in results.items():
            if result.get('skipped', False):
                logger.info("  - %s: %s", ticker, result['error_message'])

    if failed > 0:
        logger.warning("Failed tickers:")
        for ticker, result in results.items():
            if not result['success'] and not result.get('skipped', False):
                logger.warning("  - %s: %s", ticker, result['error_message'])

    logger.info("=" * 70)

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


def main(argv: Optional[List[str]] = None) -> Dict:
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    parser = argparse.ArgumentParser(description="Train ML models for stock tickers")
    parser.add_argument("--max-age", type=int, default=30,
                        help="Max model age in days before retraining (default: 30)")
    parser.add_argument("--max-stocks", type=int, default=None,
                        help="Maximum number of stocks to process in this run")
    parser.add_argument("--use-lstm", action="store_true",
                        help="Use LSTM instead of TCN as sequence model")
    parser.add_argument("--time-steps", type=int, default=None,
                        help="Time steps for sequence models (default: from config)")
    parser.add_argument(
        "--refresh-cache-contract",
        nargs="+",
        metavar="TICKER",
        help="Refresh cached RF/XGB/Ridge/SVR rows for the given tickers against the current feature contract",
    )
    parser.add_argument(
        "--refresh-model-types",
        nargs="+",
        choices=['rf', 'xgb', 'ridge', 'svr'],
        default=None,
        help="Optional subset of flat model cache rows to refresh",
    )
    parser.add_argument(
        "--validate-prediction",
        action="store_true",
        help="After refreshing cache contracts, run predict_single_stock for each ticker",
    )

    args = parser.parse_args(argv)

    if args.refresh_cache_contract:
        return refresh_cache_contracts(
            tickers=args.refresh_cache_contract,
            model_types=args.refresh_model_types,
            time_steps=args.time_steps,
            validate_prediction=args.validate_prediction,
        )

    return run_model_training(
        max_model_age_days=args.max_age,
        time_steps=args.time_steps,
        use_tcn=not args.use_lstm if args.use_lstm else None,
        max_stocks=args.max_stocks
    )


if __name__ == "__main__":
    main()