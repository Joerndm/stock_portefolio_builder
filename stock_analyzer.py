"""Module to analyze stock data and predict future stock prices."""
import os
import time
import pandas as pd

import stock_data_fetch
import import_stock_data
import split_dataset
import dimension_reduction
import ml_builder
import monte_carlo_sim
import efficient_frontier

# Import stock symbols from a CSV file
def import_symbols(csv_file):
    """
    Imports stock symbols from a CSV file and returns a pandas DataFrame.

    The CSV file should have a column named 'Symbol' containing the stock symbols.

    Parameters:
    - csv_file (str): The path to the CSV file.

    Returns:
    pandas.DataFrame: A DataFrame containing the imported stock symbols.

    Raises:
    - FileNotFoundError: If the specified CSV file does not exist.
    - KeyError: If the CSV file does not have a column named 'Symbol'.
    """

    try:
        # Read the CSV file from current position into a DataFrame
        my_path = os.path.abspath(__file__)
        path = os.path.dirname(my_path)
        import_location = os.path.join(path, csv_file)
        print(import_location)
        df = pd.read_csv(import_location)

        # Check if the 'Symbol' column exists in the DataFrame
        if 'Symbol' not in df.columns:
            raise KeyError("CSV file does not have a column named 'Symbol'.")

        # Return the DataFrame with stock symbols
        return df[['Symbol']]

    except FileNotFoundError as e:
        raise FileNotFoundError(f"The specified CSV file '{csv_file}' does not exist.") from e

if __name__ == "__main__":
    # Import stock symbols from a CSV file
    stock_symbols_df = import_symbols('index_symbol_list_multiple_stocks.csv')
    stock_symbols_list = stock_symbols_df['Symbol'].tolist()
    print(stock_symbols_df)
    # Fetch stock data for the imported stock symbols
    pf_prices = pd.DataFrame()
    for index, row in stock_symbols_df.iterrows():
        start_time = time.time()
        stock = row["Symbol"]
        print(f"Analyzing stock number: {index+1}")
        print(f"Stock ticker: {stock}")
        # Fetch stock data for the imported stock symbols
        stock_price_data_df = stock_data_fetch.fetch_stock_price_data(stock)
        stock_price_data_df = stock_data_fetch.calculate_period_returns(stock_price_data_df)
        stock_price_data_df = stock_data_fetch.calculate_moving_averages(stock_price_data_df)
        stock_price_data_df = stock_data_fetch.calculate_standard_diviation_value(stock_price_data_df)
        stock_price_data_df = stock_data_fetch.calculate_bollinger_bands(stock_price_data_df)
        # Fetch stock data for the imported stock symbols
        full_stock_financial_data_df = stock_data_fetch.fetch_stock_financial_data(stock)
        # Combine stock data with stock financial data
        combined_stock_data_df = stock_data_fetch.combine_stock_data(stock_price_data_df, full_stock_financial_data_df)
        # Calculate ratios
        combined_stock_data_df = stock_data_fetch.calculate_ratios(combined_stock_data_df)
        combined_stock_data_df = stock_data_fetch.calculate_momentum(combined_stock_data_df)
        combined_stock_data_df = stock_data_fetch.drop_nan_values(combined_stock_data_df)
        # Create a dictionary of dataframes to export to Excel
        dataframes = {
            "Combined Stock Data": combined_stock_data_df
        }
        # Export the dataframes to an Excel file
        stock_data_fetch.export_to_excel(dataframes, 'stock_data_single.xlsx')
        # Import the stock data from an Excel file
        dataframes = stock_data_fetch.import_excel("stock_data_single.xlsx")
        for key, value in dataframes.items():
            dataframe = value


        # Export the stock data to a CSV file
        stock_data_fetch.convert_excel_to_csv(dataframe, "stock_data_single")
        stock_data_df = import_stock_data.import_as_df_from_csv('stock_data_single.csv')
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
        iterations = 7500
        nn_model = ml_builder.neural_network_model(traning_dataset_df, test_dataset_df, feature_amount*16, feature_amount*12, feature_amount*20, feature_amount*16, iterations, 1)
        amount_of_days = 252
        forecast_df = ml_builder.predict_future_price_changes(stock, scaler, nn_model, selected_features_list, stock_data_df, amount_of_days)
        print("Forecasted stock prices: ")
        print(forecast_df)
        ml_builder.calculate_predicted_profit(forecast_df, amount_of_days)
        # Plot the graph
        ml_builder.plot_graph(stock_data_df, forecast_df)
        # Run a Monte Carlo simulation
        year_amount = 20
        sim_amount = 1500
        monte_carlo_day_df, monte_carlo_year_df = monte_carlo_sim.monte_carlo_analysis(0, stock_data_df, forecast_df, year_amount, sim_amount)
        pf_prices = pd.concat([pf_prices, forecast_df.set_index("Date")["Price"]], axis=1)
        pf_prices = pf_prices.rename(columns={"Price": stock})
        # Calculate the execution time
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds to build dataset and ML models.")


    pf_prices = pf_prices.dropna()
    portefolio_df = efficient_frontier.efficient_frontier_sim(pf_prices)
    print(portefolio_df)
