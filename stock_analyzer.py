import pandas as pd

import stock_data_fetch
import import_csv_file
import split_dataset
import dimension_reduction
import ml_builder

# Import stock symbols from a CSV file
def import_stock_symbols(csv_file):
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
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file)

        # Check if the 'Symbol' column exists in the DataFrame
        if 'Symbol' not in df.columns:
            raise KeyError("CSV file does not have a column named 'Symbol'.")

        # Return the DataFrame with stock symbols
        return df[['Symbol']]

    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file '{csv_file}' does not exist.")


if __name__ == "__main__":
    import time
    
    start_time = time.time()
    # Import stock symbols from a CSV file
    stock_symbols_df = import_stock_symbols('index_symbol_list_multiple_stocks.csv')
    stock_symbols_list = stock_symbols_df['Symbol'].tolist()
    print(stock_symbols_df)

    # Fetch stock data for the imported stock symbols
    for index, row in stock_symbols_df.iterrows():
        stock = row["Symbol"]
        print(stock)
        stock_data_df = stock_data_fetch.fetch_stock_price_data(stock)
        # print(stock_data_df)
        # Fetch stock data for the imported stock symbols
        full_stock_financial_data_df = stock_data_fetch.fetch_stock_financial_data(stock)
        # print(full_stock_financial_data_df)
        # Combine stock data with stock financial data
        combined_stock_data_df = stock_data_fetch.combine_stock_data(stock_data_df, full_stock_financial_data_df)
        # print(combined_stock_data_df)
        # Calculate ratios
        combined_stock_data_df = stock_data_fetch.calculate_ratios(combined_stock_data_df)
        # print(combined_stock_data_df)
        # Create a dictionary of dataframes to export to Excel
        dataframes = {
            # "Stock Data": stock_data_df,
            # "Full Stock Financial Data": full_stock_financial_data_df,
            "Combined Stock Data": combined_stock_data_df
        }
        # Export the dataframes to an Excel file
        stock_data_fetch.export_to_excel(dataframes, 'stock_data_single_v2.xlsx')
        # Import the stock data from an Excel file
        dataframes = stock_data_fetch.import_excel("stock_data_single_v2.xlsx")
        for key, value in dataframes.items():
            dataframe = value


        # Export the stock data to a CSV file
        stock_data_fetch.convert_excel_to_csv(dataframe, "stock_data_single_v2")
        stock_data_df = import_csv_file.import_as_df('stock_data_single_v2.csv')
        # Split the dataset into traning, test data and prediction data
        x_training_data, x_test_data, y_training_data, y_test_data, prediction_data = split_dataset.dataset_train_test_split(stock_data_df, 0.20, 1)
        # Reduce the dataset dimensions with PCA
        x_training_dataset, x_test_dataset, x_prediction_dataset = dimension_reduction.feature_selection(12, x_training_data, x_test_data, y_training_data, y_test_data, prediction_data, stock_data_df)
        # Combine the reduced dataset with the stock price
        x_training_dataset_df = pd.DataFrame(x_training_dataset)
        y_training_data_df = pd.DataFrame(y_training_data, columns=["Price"])
        traning_dataset_df = x_training_dataset_df.join(y_training_data_df)
        x_test_dataset_df = pd.DataFrame(x_test_dataset)
        y_test_data_df = pd.DataFrame(y_test_data, columns=["Price"])
        test_dataset_df = x_test_dataset_df.join(y_test_data_df)
        x_prediction_dataset_df = pd.DataFrame(x_prediction_dataset)
        # Predict the stock price
        forecast_df = ml_builder.predict_price(traning_dataset_df, test_dataset_df, x_prediction_dataset_df, stock_data_df)
        # Plot the graph
        ml_builder.plot_graph(stock_data_df, forecast_df)
        # Calculate the execution time
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds to build dataset and ML models.")
