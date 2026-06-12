"""
Comparison test: stock_analyzer vs model_trainer training pipeline.

Runs the IDENTICAL data loading + splitting + feature selection steps from
both modules side-by-side for a single ticker, and compares every intermediate
result. Does NOT actually train models (that takes hours) — it stops right
before train_and_validate_models and verifies all inputs would be identical.

Usage:
    python test_compare_training.py
    python test_compare_training.py --ticker DEMANT.CO
"""

import os
import sys
import io
import time
import contextlib
import traceback

os.environ['TF_PTXAS_UNAVAILABLE'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress TF noise during test

import numpy as np
import pandas as pd

import db_interactions
import split_dataset
import dimension_reduction
from model_trainer import validate_data_availability as _validate_data_availability


# ============================================================================
# Helper: capture stdout
# ============================================================================
@contextlib.contextmanager
def capture_output():
    """Capture stdout/stderr into strings."""
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = buf_out = io.StringIO()
    sys.stderr = buf_err = io.StringIO()
    try:
        yield buf_out, buf_err
    finally:
        sys.stdout, sys.stderr = old_out, old_err


# ============================================================================
# stock_analyzer path  (verbatim copy of process_single_stock up to ML call)
# ============================================================================
def stock_analyzer_pipeline(stock_symbol: str, time_steps: int = 30):
    """
    Exact reproduction of stock_analyzer.process_single_stock,
    stopping just before train_and_validate_models.
    Returns dict of every intermediate variable.
    """
    result = {}

    # --- Data import (identical to stock_analyzer.py lines 147-154) ---
    stock_data_df = db_interactions.import_stock_dataset(stock_symbol)
    stock_data_df["date"] = pd.to_datetime(stock_data_df["date"])
    result['raw_rows'] = len(stock_data_df)
    result['raw_columns'] = list(stock_data_df.columns)

    stock_data_df = stock_data_df.dropna(axis=0, how="any")
    stock_data_df = stock_data_df.dropna(axis=1, how="any")
    result['clean_rows'] = len(stock_data_df)
    result['clean_columns'] = list(stock_data_df.columns)

    if len(stock_data_df) < 100:
        result['error'] = f"Insufficient data: {len(stock_data_df)} rows"
        return result

    # --- Split (identical to stock_analyzer.py lines 157-163) ---
    validation_size = 0.20
    test_size = 0.10
    scaler_x, scaler_y, x_train_scaled, x_val_scaled, x_test_scaled, \
        y_train_scaled, y_val_scaled, y_test_scaled, x_predictions = \
        split_dataset.dataset_train_test_split(
            stock_data_df, test_size, validation_size=validation_size
        )

    result['x_train_shape'] = x_train_scaled.shape
    result['x_val_shape'] = x_val_scaled.shape
    result['x_test_shape'] = x_test_scaled.shape
    result['y_train_shape'] = y_train_scaled.shape
    result['y_val_shape'] = y_val_scaled.shape
    result['y_test_shape'] = y_test_scaled.shape
    result['x_predictions_shape'] = x_predictions.shape

    # --- Inverse transform (identical) ---
    y_train_unscaled = scaler_y.inverse_transform(
        y_train_scaled.reshape(-1, 1)).flatten()
    y_val_unscaled = scaler_y.inverse_transform(
        y_val_scaled.reshape(-1, 1)).flatten()
    y_test_unscaled = scaler_y.inverse_transform(
        y_test_scaled.reshape(-1, 1)).flatten()

    # --- DataFrames for feature selection (identical) ---
    x_training_data = pd.DataFrame(x_train_scaled)
    x_val_data = pd.DataFrame(x_val_scaled)
    x_test_data = pd.DataFrame(x_test_scaled)
    y_training_data_df = pd.Series(y_train_unscaled)
    y_val_data_df = pd.Series(y_val_unscaled)
    y_test_data_df = pd.Series(y_test_unscaled)
    prediction_data = x_predictions

    max_features = len(x_training_data.columns)
    feature_amount = max_features
    result['feature_amount'] = feature_amount

    # --- Feature selection (identical) ---
    x_training_dataset, x_val_dataset, x_test_dataset, x_prediction_dataset, \
        selected_features_model, selected_features_list = \
        dimension_reduction.feature_selection_rf(
            feature_amount, x_training_data, x_val_data, x_test_data,
            y_training_data_df, y_val_data_df, y_test_data_df,
            prediction_data, stock_data_df
        )

    result['selected_features'] = selected_features_list
    result['x_train_after_fs_shape'] = x_training_dataset.shape
    result['x_val_after_fs_shape'] = x_val_dataset.shape
    result['x_test_after_fs_shape'] = x_test_dataset.shape

    # --- Prepare for ML (identical) ---
    x_training_dataset_df = pd.DataFrame(x_training_dataset, columns=selected_features_list)
    y_training_data_df = y_training_data_df.reset_index(drop=True)
    x_val_dataset_df = pd.DataFrame(x_val_dataset, columns=selected_features_list)
    y_val_data_df = y_val_data_df.reset_index(drop=True)
    x_test_dataset_df = pd.DataFrame(x_test_dataset, columns=selected_features_list)
    y_test_data_df = y_test_data_df.reset_index(drop=True)
    # stock_analyzer also builds x_prediction_dataset_df (model_trainer does NOT)
    x_prediction_dataset_df = pd.DataFrame(x_prediction_dataset, columns=selected_features_list)

    y_train_scaled_for_lstm = pd.Series(y_train_scaled)
    y_test_scaled_for_lstm = pd.Series(y_test_scaled)
    y_val_scaled_for_lstm = pd.Series(y_val_scaled)

    # --- What would be passed to train_and_validate_models ---
    result['ml_x_train'] = x_training_dataset_df.values
    result['ml_x_val'] = x_val_dataset_df.values
    result['ml_x_test'] = x_test_dataset_df.values
    result['ml_y_train_scaled'] = y_train_scaled_for_lstm.values
    result['ml_y_val_scaled'] = y_val_scaled_for_lstm.values
    result['ml_y_test_scaled'] = y_test_scaled_for_lstm.values
    result['ml_y_train_unscaled'] = y_train_unscaled
    result['ml_y_val_unscaled'] = y_val_unscaled
    result['ml_y_test_unscaled'] = y_test_unscaled

    # stock_analyzer extras not in model_trainer
    result['has_x_prediction_dataset_df'] = True
    result['x_prediction_dataset_df_shape'] = x_prediction_dataset_df.shape
    result['scaler_x'] = scaler_x
    result['scaler_y'] = scaler_y

    return result


# ============================================================================
# model_trainer path  (verbatim copy of train_single_stock up to ML call)
# ============================================================================
def model_trainer_pipeline(stock_symbol: str, time_steps: int = 30):
    """
    Exact reproduction of model_trainer.train_single_stock,
    stopping just before train_and_validate_models.
    Returns dict of every intermediate variable.
    """
    result = {}

    # --- validate_data_availability (model_trainer ONLY) ---
    # NOTE: imported at module level to avoid import inside capture_output
    #       (ml_builder does sys.stdout wrapping on import that breaks StringIO)
    availability = _validate_data_availability(stock_symbol)
    result['availability'] = availability
    if not availability['valid']:
        result['error'] = f"validate_data_availability failed: {availability['message']}"
        return result

    # --- Data import (identical to model_trainer.py lines 161-166) ---
    stock_data_df = db_interactions.import_stock_dataset(stock_symbol)
    stock_data_df["date"] = pd.to_datetime(stock_data_df["date"])
    result['raw_rows'] = len(stock_data_df)
    result['raw_columns'] = list(stock_data_df.columns)

    stock_data_df = stock_data_df.dropna(axis=0, how="any")
    stock_data_df = stock_data_df.dropna(axis=1, how="any")
    result['clean_rows'] = len(stock_data_df)
    result['clean_columns'] = list(stock_data_df.columns)

    if len(stock_data_df) < 100:
        result['error'] = f"Insufficient data: {len(stock_data_df)} rows (skipped)"
        return result

    # --- Split (identical) ---
    validation_size = 0.20
    test_size = 0.10
    scaler_x, scaler_y, x_train_scaled, x_val_scaled, x_test_scaled, \
        y_train_scaled, y_val_scaled, y_test_scaled, x_predictions = \
        split_dataset.dataset_train_test_split(
            stock_data_df, test_size, validation_size=validation_size
        )

    result['x_train_shape'] = x_train_scaled.shape
    result['x_val_shape'] = x_val_scaled.shape
    result['x_test_shape'] = x_test_scaled.shape
    result['y_train_shape'] = y_train_scaled.shape
    result['y_val_shape'] = y_val_scaled.shape
    result['y_test_shape'] = y_test_scaled.shape
    result['x_predictions_shape'] = x_predictions.shape

    # --- Inverse transform (identical) ---
    y_train_unscaled = scaler_y.inverse_transform(
        y_train_scaled.reshape(-1, 1)).flatten()
    y_val_unscaled = scaler_y.inverse_transform(
        y_val_scaled.reshape(-1, 1)).flatten()
    y_test_unscaled = scaler_y.inverse_transform(
        y_test_scaled.reshape(-1, 1)).flatten()

    # --- DataFrames for feature selection (identical) ---
    x_training_data = pd.DataFrame(x_train_scaled)
    x_val_data = pd.DataFrame(x_val_scaled)
    x_test_data = pd.DataFrame(x_test_scaled)
    y_training_data_df = pd.Series(y_train_unscaled)
    y_val_data_df = pd.Series(y_val_unscaled)
    y_test_data_df = pd.Series(y_test_unscaled)
    prediction_data = x_predictions

    max_features = len(x_training_data.columns)
    feature_amount = max_features
    result['feature_amount'] = feature_amount

    # --- Feature selection (identical) ---
    x_training_dataset, x_val_dataset, x_test_dataset, x_prediction_dataset, \
        selected_features_model, selected_features_list = \
        dimension_reduction.feature_selection_rf(
            feature_amount, x_training_data, x_val_data, x_test_data,
            y_training_data_df, y_val_data_df, y_test_data_df,
            prediction_data, stock_data_df
        )

    result['selected_features'] = selected_features_list
    result['x_train_after_fs_shape'] = x_training_dataset.shape
    result['x_val_after_fs_shape'] = x_val_dataset.shape
    result['x_test_after_fs_shape'] = x_test_dataset.shape

    # --- Prepare for ML (identical) ---
    x_training_dataset_df = pd.DataFrame(x_training_dataset, columns=selected_features_list)
    y_training_data_df = y_training_data_df.reset_index(drop=True)
    x_val_dataset_df = pd.DataFrame(x_val_dataset, columns=selected_features_list)
    y_val_data_df = y_val_data_df.reset_index(drop=True)
    x_test_dataset_df = pd.DataFrame(x_test_dataset, columns=selected_features_list)
    y_test_data_df = y_test_data_df.reset_index(drop=True)
    # model_trainer does NOT build x_prediction_dataset_df

    y_train_scaled_for_lstm = pd.Series(y_train_scaled)
    y_test_scaled_for_lstm = pd.Series(y_test_scaled)
    y_val_scaled_for_lstm = pd.Series(y_val_scaled)

    # --- What would be passed to train_and_validate_models ---
    result['ml_x_train'] = x_training_dataset_df.values
    result['ml_x_val'] = x_val_dataset_df.values
    result['ml_x_test'] = x_test_dataset_df.values
    result['ml_y_train_scaled'] = y_train_scaled_for_lstm.values
    result['ml_y_val_scaled'] = y_val_scaled_for_lstm.values
    result['ml_y_test_scaled'] = y_test_scaled_for_lstm.values
    result['ml_y_train_unscaled'] = y_train_unscaled
    result['ml_y_val_unscaled'] = y_val_unscaled
    result['ml_y_test_unscaled'] = y_test_unscaled

    # model_trainer does NOT have these
    result['has_x_prediction_dataset_df'] = False
    result['scaler_x'] = scaler_x
    result['scaler_y'] = scaler_y

    return result


# ============================================================================
# Comparison logic
# ============================================================================
def compare_results(sa_result, mt_result, ticker):
    """Compare every field between stock_analyzer and model_trainer pipelines."""
    print("\n" + "=" * 80)
    print(f"  COMPARISON REPORT: {ticker}")
    print("=" * 80)

    passed = 0
    failed = 0
    warnings = 0

    def check(label, sa_val, mt_val, is_array=False):
        nonlocal passed, failed
        if is_array:
            if isinstance(sa_val, np.ndarray) and isinstance(mt_val, np.ndarray):
                if sa_val.shape != mt_val.shape:
                    print(f"  [FAIL] {label}: SHAPE MISMATCH  sa={sa_val.shape}  mt={mt_val.shape}")
                    failed += 1
                    return
                if np.allclose(sa_val, mt_val, atol=1e-10, equal_nan=True):
                    print(f"  [OK]   {label}: shapes {sa_val.shape} — values match")
                    passed += 1
                else:
                    max_diff = np.max(np.abs(sa_val - mt_val))
                    print(f"  [FAIL] {label}: shapes match {sa_val.shape} but values differ (max diff: {max_diff:.2e})")
                    failed += 1
            else:
                print(f"  [FAIL] {label}: types differ sa={type(sa_val)} mt={type(mt_val)}")
                failed += 1
        else:
            if sa_val == mt_val:
                print(f"  [OK]   {label}: {sa_val}")
                passed += 1
            else:
                print(f"  [FAIL] {label}: sa={sa_val}  mt={mt_val}")
                failed += 1

    # Check for early errors
    if 'error' in sa_result and 'error' in mt_result:
        print(f"\n  Both pipelines errored:")
        print(f"    stock_analyzer: {sa_result['error']}")
        print(f"    model_trainer:  {mt_result['error']}")
        return passed, failed, warnings

    if 'error' in sa_result:
        print(f"\n  [FAIL] stock_analyzer errored but model_trainer did not:")
        print(f"    {sa_result['error']}")
        return 0, 1, 0

    if 'error' in mt_result:
        print(f"\n  [FAIL] model_trainer errored but stock_analyzer did not:")
        print(f"    {mt_result['error']}")
        return 0, 1, 0

    # --- Data loading ---
    print(f"\n  --- Data Loading ---")
    check("Raw rows", sa_result['raw_rows'], mt_result['raw_rows'])
    check("Clean rows", sa_result['clean_rows'], mt_result['clean_rows'])
    check("Clean columns", sa_result['clean_columns'], mt_result['clean_columns'])

    # --- Dataset split shapes ---
    print(f"\n  --- Dataset Split ---")
    check("x_train shape", sa_result['x_train_shape'], mt_result['x_train_shape'])
    check("x_val shape", sa_result['x_val_shape'], mt_result['x_val_shape'])
    check("x_test shape", sa_result['x_test_shape'], mt_result['x_test_shape'])
    check("y_train shape", sa_result['y_train_shape'], mt_result['y_train_shape'])
    check("y_val shape", sa_result['y_val_shape'], mt_result['y_val_shape'])
    check("y_test shape", sa_result['y_test_shape'], mt_result['y_test_shape'])
    check("x_predictions shape", sa_result['x_predictions_shape'], mt_result['x_predictions_shape'])

    # --- Feature selection ---
    print(f"\n  --- Feature Selection ---")
    check("Feature amount (input)", sa_result['feature_amount'], mt_result['feature_amount'])
    check("Selected features", sa_result['selected_features'], mt_result['selected_features'])
    check("x_train after FS shape", sa_result['x_train_after_fs_shape'], mt_result['x_train_after_fs_shape'])
    check("x_val after FS shape", sa_result['x_val_after_fs_shape'], mt_result['x_val_after_fs_shape'])
    check("x_test after FS shape", sa_result['x_test_after_fs_shape'], mt_result['x_test_after_fs_shape'])

    # --- ML input arrays (the critical comparison) ---
    print(f"\n  --- ML Input Arrays (train_and_validate_models args) ---")
    check("ml_x_train", sa_result['ml_x_train'], mt_result['ml_x_train'], is_array=True)
    check("ml_x_val", sa_result['ml_x_val'], mt_result['ml_x_val'], is_array=True)
    check("ml_x_test", sa_result['ml_x_test'], mt_result['ml_x_test'], is_array=True)
    check("ml_y_train_scaled", sa_result['ml_y_train_scaled'], mt_result['ml_y_train_scaled'], is_array=True)
    check("ml_y_val_scaled", sa_result['ml_y_val_scaled'], mt_result['ml_y_val_scaled'], is_array=True)
    check("ml_y_test_scaled", sa_result['ml_y_test_scaled'], mt_result['ml_y_test_scaled'], is_array=True)
    check("ml_y_train_unscaled", sa_result['ml_y_train_unscaled'], mt_result['ml_y_train_unscaled'], is_array=True)
    check("ml_y_val_unscaled", sa_result['ml_y_val_unscaled'], mt_result['ml_y_val_unscaled'], is_array=True)
    check("ml_y_test_unscaled", sa_result['ml_y_test_unscaled'], mt_result['ml_y_test_unscaled'], is_array=True)

    # --- Differences unique to each pipeline ---
    print(f"\n  --- Pipeline Differences ---")
    if mt_result.get('availability'):
        avail = mt_result['availability']
        print(f"  [INFO] model_trainer validate_data_availability: valid={avail['valid']}, "
              f"missing={avail['missing_tables']}")
    else:
        print(f"  [INFO] model_trainer validate_data_availability: not run (error before)")

    sa_has_pred = sa_result.get('has_x_prediction_dataset_df', False)
    mt_has_pred = mt_result.get('has_x_prediction_dataset_df', False)
    if sa_has_pred and not mt_has_pred:
        warnings += 1
        print(f"  [WARN] stock_analyzer builds x_prediction_dataset_df "
              f"{sa_result.get('x_prediction_dataset_df_shape', '?')}, "
              f"model_trainer does NOT (needed for prediction phase)")

    # Summary
    print(f"\n  {'=' * 40}")
    total = passed + failed
    print(f"  RESULTS: {passed}/{total} checks passed, {failed} failed, {warnings} warning(s)")
    if failed == 0:
        print(f"  VERDICT: The training data pipelines are IDENTICAL")
    else:
        print(f"  VERDICT: DIFFERENCES FOUND — model_trainer diverges from stock_analyzer")
    print(f"  {'=' * 40}\n")

    return passed, failed, warnings


# ============================================================================
# Main
# ============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Compare stock_analyzer vs model_trainer training pipeline"
    )
    parser.add_argument("--ticker", type=str, default="ABBN.SW",
                        help="Stock ticker to use as test case (default: A)")
    parser.add_argument("--time-steps", type=int, default=30,
                        help="Time steps for sequence models (default: 30)")
    args = parser.parse_args()

    ticker = args.ticker
    time_steps = args.time_steps

    print("=" * 80)
    print(f"  TRAINING PIPELINE COMPARISON TEST")
    print(f"  Ticker: {ticker}    Time steps: {time_steps}")
    print("=" * 80)

    # --- Run stock_analyzer pipeline ---
    print(f"\n>>> Running stock_analyzer pipeline for {ticker}...")
    sa_start = time.time()
    try:
        with capture_output() as (sa_out, sa_err):
            sa_result = stock_analyzer_pipeline(ticker, time_steps)
        sa_stdout = sa_out.getvalue()
        sa_time = time.time() - sa_start
        if 'error' in sa_result:
            print(f"  stock_analyzer returned error: {sa_result['error']}")
        else:
            print(f"  stock_analyzer: OK ({sa_result['clean_rows']} clean rows, "
                  f"{sa_result.get('x_train_shape', '?')} train shape) [{sa_time:.1f}s]")
    except Exception as e:
        print(f"  stock_analyzer CRASHED: {e}")
        traceback.print_exc()
        sa_result = {'error': f"CRASH: {e}"}
        sa_stdout = ""
        sa_time = time.time() - sa_start

    # --- Run model_trainer pipeline ---
    print(f"\n>>> Running model_trainer pipeline for {ticker}...")
    mt_start = time.time()
    try:
        with capture_output() as (mt_out, mt_err):
            mt_result = model_trainer_pipeline(ticker, time_steps)
        mt_stdout = mt_out.getvalue()
        mt_time = time.time() - mt_start
        if 'error' in mt_result:
            print(f"  model_trainer returned error: {mt_result['error']}")
        else:
            print(f"  model_trainer: OK ({mt_result['clean_rows']} clean rows, "
                  f"{mt_result.get('x_train_shape', '?')} train shape) [{mt_time:.1f}s]")
    except Exception as e:
        print(f"  model_trainer CRASHED: {e}")
        traceback.print_exc()
        mt_result = {'error': f"CRASH: {e}"}
        mt_stdout = ""
        mt_time = time.time() - mt_start

    # --- Compare ---
    passed, failed, warnings = compare_results(sa_result, mt_result, ticker)

    # --- Show captured output if there were failures ---
    if failed > 0:
        print("\n" + "=" * 80)
        print("  CAPTURED STDOUT (stock_analyzer)")
        print("=" * 80)
        print(sa_stdout[:3000] if sa_stdout else "(empty)")

        print("\n" + "=" * 80)
        print("  CAPTURED STDOUT (model_trainer)")
        print("=" * 80)
        print(mt_stdout[:3000] if mt_stdout else "(empty)")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
