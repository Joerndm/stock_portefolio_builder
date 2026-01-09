"""
Quick test to demonstrate the model comparison analysis functionality.
Uses reduced training parameters for faster execution.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import tensorflow as tf

# Import required modules
import db_interactions
import split_dataset
import dimension_reduction
from ml_builder import train_and_validate_models, predict_future_price_changes, analyze_prediction_performance

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == "__main__":
    # Configure GPU
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    print("="*80)
    print("MODEL COMPARISON TEST - Reduced Training for Quick Analysis")
    print("="*80)
    
    start_time = time.time()

    # Import stock data
    stock_symbol = "DEMANT.CO"
    print(f"\n📊 Loading stock data for {stock_symbol}...")
    
    stock_data_df = db_interactions.import_stock_dataset(stock_symbol)
    stock_data_df["date"] = pd.to_datetime(stock_data_df["date"])
    stock_data_df = stock_data_df.dropna(axis=0, how="any")
    stock_data_df = stock_data_df.dropna(axis=1, how="any")
    
    print(f"   Data points: {len(stock_data_df)}")
    print(f"   Features: {len(stock_data_df.columns)}")

    # Split dataset
    test_size = 0.20
    validation_size = 0.15
    print(f"\n📦 Splitting dataset: Train={1-test_size-validation_size:.0%}, Val={validation_size:.0%}, Test={test_size:.0%}")
    
    scaler_x, scaler_y, x_train_scaled, x_val_scaled, x_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled, x_Predictions = split_dataset.dataset_train_test_split(
        stock_data_df, test_size, validation_size=validation_size
    )

    # Inverse-transform y values for RF/XGB
    y_train_unscaled = scaler_y.inverse_transform(y_train_scaled.reshape(-1, 1)).flatten()
    y_val_unscaled = scaler_y.inverse_transform(y_val_scaled.reshape(-1, 1)).flatten()
    y_test_unscaled = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()

    # Feature selection
    x_training_data = pd.DataFrame(x_train_scaled)
    x_val_data = pd.DataFrame(x_val_scaled)
    x_test_data = pd.DataFrame(x_test_scaled)
    y_training_data_df = pd.Series(y_train_unscaled)
    y_val_data_df = pd.Series(y_val_unscaled)
    y_test_data_df = pd.Series(y_test_unscaled)
    prediction_data = x_Predictions

    max_features = len(x_training_data.columns)
    feature_amount = max_features
    
    print(f"\n🔍 Feature selection: Using {feature_amount} features")
    
    x_training_dataset, x_val_dataset, x_test_dataset, x_prediction_dataset, selected_features_model, selected_features_list = dimension_reduction.feature_selection(
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

    x_training_dataset_df = pd.DataFrame(x_training_dataset, columns=selected_features_list)
    x_val_dataset_df = pd.DataFrame(x_val_dataset, columns=selected_features_list)
    x_test_dataset_df = pd.DataFrame(x_test_dataset, columns=selected_features_list)
    x_prediction_dataset_df = pd.DataFrame(x_prediction_dataset, columns=selected_features_list)

    TIME_STEPS = 30
    y_train_scaled_for_lstm = pd.Series(y_train_scaled)
    y_val_scaled_for_lstm = pd.Series(y_val_scaled)
    y_test_scaled_for_lstm = pd.Series(y_test_scaled)

    # REDUCED TRAINING PARAMETERS FOR QUICK TEST
    print(f"\n🚀 Starting model training (REDUCED PARAMETERS for quick test)...")
    print("   LSTM: 5 trials, 2 executions, 50 epochs")
    print("   RF: 10 trials")
    print("   XGB: 10 trials")
    
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
        max_retrains=3,  # Reduced from 150
        overfitting_threshold=0.20,
        lstm_trials=5,  # Reduced from 50
        lstm_executions=2,  # Reduced from 10
        lstm_epochs=50,  # Reduced from 500
        lstm_retrain_trials_increment=2,
        lstm_retrain_executions_increment=1,
        rf_trials=10,  # Reduced from 100
        rf_retrain_increment=5,
        xgb_trials=10,  # Reduced from 60
        xgb_retrain_increment=5,
        use_multi_metric_detection=True
    )

    lstm_model = models['lstm']
    rf_model = models['rf']
    xgb_model = models['xgb']

    print(f"\n✅ Training complete!")
    print(f"   LSTM attempts: {len(training_history['lstm'])}")
    print(f"   RF attempts: {len(training_history['random_forest'])}")
    print(f"   XGB attempts: {len(training_history['xgboost'])}")

    # Make predictions
    amount_of_days = 10  # Reduced from TIME_STEPS * 3
    print(f"\n🔮 Making predictions for {amount_of_days} days...")
    print(f"   Historical predictions: {len(x_prediction_dataset_df) if x_prediction_dataset_df is not None else 0}")
    print(f"   Future predictions: {amount_of_days}")
    
    forecast_df = predict_future_price_changes(
        ticker=stock_symbol,
        scaler_x=scaler_x,
        scaler_y=scaler_y,
        model={'lstm': lstm_model, 'rf': rf_model, 'xgb': xgb_model},
        selected_features_list=selected_features_list,
        stock_df=stock_data_df,
        prediction_days=amount_of_days,
        time_steps=TIME_STEPS,
        historical_prediction_dataset_df=x_prediction_dataset_df
    )

    # Analyze prediction performance
    print("\n" + "="*80)
    historical_pred_count = len(x_prediction_dataset_df) if x_prediction_dataset_df is not None else 0
    analyze_prediction_performance(stock_data_df, forecast_df, historical_pred_count)

    # Calculate execution time
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"\n⏱️ Total execution time: {execution_time:.2f} seconds ({execution_time/60:.1f} minutes)")
    print("="*80)
    print("✅ Test complete! Analysis above shows:")
    print("   1. Individual model predictions (LSTM, RF, XGB) for each day")
    print("   2. Ensemble predictions combining all 3 models")
    print("   3. Comparison with actual historical values")
    print("   4. Prediction accuracy metrics")
    print("   5. Day-by-day breakdown of errors")
    print("="*80)
