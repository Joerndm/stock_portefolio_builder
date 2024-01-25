import stock_data_fetch
import ml_builder

import pandas as pd


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
    # Import stock symbols from a CSV file
    stock_symbols_df = import_stock_symbols('index_symbol_list_multiple_stocks.csv')
    stock_symbols_list = stock_symbols_df['Symbol'].tolist()
    print(stock_symbols_df)
    # stock_symbols_df['Symbol'].tolist()
    for index, row in stock_symbols_df.iterrows():
        stock = row["Symbol"]
        print(stock)
        # Fetch stock data for the imported stock symbols
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
        # Import the stock data from a CSV file
        stock_data_df = ml_builder.import_stock_data('stock_data_single_v2.csv')
        # Predict the stock price
        forecast_df = ml_builder.predict_price(stock_data_df)
        # Plot the graph
        ml_builder.plot_graph(stock_data_df, forecast_df)