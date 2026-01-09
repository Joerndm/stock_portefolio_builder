"""
Module to analyze stock data and predict future stock prices.

This module provides functionality for:
- Fetching and preprocessing stock data from database
- Splitting datasets into training, validation, and test sets
- Performing feature selection using Random Forest importance
- Training multiple ML models (LSTM, Random Forest, XGBoost) with automatic retraining
- Detecting and preventing overfitting through multi-metric validation
- Predicting future stock price changes using ensemble methods
- Running Monte Carlo simulations for risk analysis
- Optimizing portfolio allocation using efficient frontier analysis
- Generating forecasts and visualization graphs

The main execution flow:
1. Import stock symbols from database
2. Fetch and clean historical stock data
3. Split data and perform feature engineering
4. Train and validate multiple ML models with overfitting detection
5. Generate future price predictions using ensemble approach
6. Perform Monte Carlo simulations for uncertainty quantification
7. Optimize portfolio using efficient frontier methodology
8. Export results to Excel and generate visualization graphs

GPU Configuration:
- Automatically detects and configures TensorFlow GPU devices
- Limits GPU memory to 7GB to prevent out-of-memory errors
- Enables memory growth for efficient GPU utilization

Dependencies:
- fetch_secrets: Secret management for API keys and credentials
- db_connectors: Database connection utilities
- db_interactions: Database CRUD operations
- stock_data_fetch: Stock data retrieval functionality
- split_dataset: Data splitting and scaling utilities
- dimension_reduction: Feature selection and dimensionality reduction
- ml_builder: Machine learning model construction and training
- monte_carlo_sim: Monte Carlo simulation for risk analysis
- efficient_frontier: Portfolio optimization using Modern Portfolio Theory
"""
import os
import time
import pandas as pd

import fetch_secrets
import db_connectors
import db_interactions
import stock_data_fetch
import split_dataset
import dimension_reduction
import ml_builder
import monte_carlo_sim
import efficient_frontier

if __name__ == "__main__":
    import db_interactions

    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_virtual_device_configuration(
            gpu,
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7168)]  # Limit to 7GB
        )
    stock_symbols_list = db_interactions.import_ticker_list()
    stock_symbols_df = pd.DataFrame(stock_symbols_list, columns=["Symbol"])
    stock_symbols_df = stock_symbols_df.drop(stock_symbols_df[stock_symbols_df["Symbol"].isin(["DANSKE.CO", "JYSK.CO", "NDA-DK.CO", "TRYG.CO", "ORSTED.CO"])].index, axis=0)
    # stock_symbols_df = stock_symbols_df.reset_index(drop=True)
    print("stock_symbols_df")
    print(stock_symbols_df)
    # Fetch stock data for the imported stock symbols
    pf_prices = pd.DataFrame()
    for index, row in stock_symbols_df.iterrows():
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

    pf_prices = pf_prices.dropna()
    portefolio_df = efficient_frontier.efficient_frontier_sim(pf_prices)
    print(portefolio_df)
