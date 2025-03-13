"""Module to analyze stock data and predict future stock prices."""
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
        stock = row["Symbol"]
        print(f"Analyzing stock number: {index+1}")
        print(f"Stock ticker: {stock}")
        stock_data_df = db_interactions.import_stock_dataset(stock)
        # Change the date column to datetime 64
        stock_data_df["date"] = pd.to_datetime(stock_data_df["date"])
        # Drop the columns that are empty
        stock_data_df = stock_data_df.dropna(axis=0, how="any")
        stock_data_df = stock_data_df.dropna(axis=1, how="any")
        # Split the dataset into traning, test data and prediction data
        test_size = 0.20
        scaler, x_training_data, x_test_data, y_training_data_df, y_test_data_df, prediction_data = split_dataset.dataset_train_test_split(stock_data_df, test_size, 1)
        # Feature selection
        feature_amount = 30
        x_training_dataset, x_test_dataset, x_prediction_dataset, selected_features_model, selected_features_list = dimension_reduction.feature_selection(feature_amount, x_training_data, x_test_data, y_training_data_df, y_test_data_df, prediction_data, stock_data_df)
        # Combine the reduced dataset with the stock price
        x_training_dataset_df = pd.DataFrame(x_training_dataset, columns=selected_features_list)
        y_training_data_df = y_training_data_df.reset_index(drop=True)
        traning_dataset_df = x_training_dataset_df.join(y_training_data_df)
        x_test_dataset_df = pd.DataFrame(x_test_dataset, columns=selected_features_list)
        y_test_data_df = y_test_data_df.reset_index(drop=True)
        test_dataset_df = x_test_dataset_df.join(y_test_data_df)
        x_prediction_dataset_df = pd.DataFrame(x_prediction_dataset, columns=selected_features_list)
        # Predict the stock price
        lstm_model = ml_builder.lstm_model(traning_dataset_df, test_dataset_df)
        amount_of_days = 10
        forecast_df = ml_builder.predict_future_price_changes(stock, scaler, lstm_model, selected_features_list, stock_data_df, amount_of_days)
        # Calculate the predicted profit
        ml_builder.calculate_predicted_profit(forecast_df, amount_of_days)
        # Plot the graph
        ml_builder.plot_graph(stock_data_df, forecast_df)
        # Run a Monte Carlo simulation
        year_amount = 10
        sim_amount = 1000
        monte_carlo_day_df, monte_carlo_year_df = monte_carlo_sim.monte_carlo_analysis(0, stock_data_df, forecast_df, year_amount, sim_amount)
        forecast_df = forecast_df.rename(columns={"open_Price": stock + "_price"})
        pf_prices = pd.concat([pf_prices, forecast_df.set_index("date")[stock + "_price"]], axis=1)
        # Calculate the execution time
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds to build dataset and ML models.")

    pf_prices = pf_prices.dropna()
    portefolio_df = efficient_frontier.efficient_frontier_sim(pf_prices)
    print(portefolio_df)
