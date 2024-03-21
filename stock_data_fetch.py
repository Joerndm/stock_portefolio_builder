"""This module contains functions for fetching stock data using yfinance."""
import os
import time
import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import yfinance as yf

# Import stock symbols from a CSV file
def import_tickers_from_csv(csv_file):
    """
    Imports stock symbols from a CSV file and returns a pandas DataFrame.

    The CSV file should have a column named 'Symbol' containing the stock symbols.

    Parameters:
    - csv_file (str): The path to the CSV file.

    Returns:
    pandas.DataFrame: A DataFrame containing the imported stock symbols.

    Raises:
    - ValueError: If the csv_file parameter is empty.
    - FileNotFoundError: If the CSV file does not exist.
    - KeyError: If the CSV file does not have a column named 'Symbol'.
    """
    # Check if the csv_file parameter is empty
    if csv_file == "":
        raise ValueError("The csv_file parameter cannot be empty.")

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
        raise FileNotFoundError(f"CSV file '{csv_file}' does not exist.") from e
# Fetch company information for given ticker using yfinance
def fetch_stock_standard_data(stock_symbol):
    """
    Fetches company information for the given stock symbols and returns a pandas DataFrame.
    
    The DataFrame will contain the stock symbol, company name, and industry.
    
    Parameters:
    - stock_symbols (list): A list of stock symbols.
    
    Returns:
    pandas.DataFrame: A DataFrame containing the fetched company information.
    
    Raises:
    - ValueError: If the stock_symbols parameter is empty.
    - KeyError: If any of the stock symbols are invalid or not found.
    - KeyError: If the stock_info cannot be transformed to a pandas DataFrame.
    """
    # Check if the stock_symbols parameter is empty
    if len(stock_symbol) == "":
        raise ValueError("The stock_symbols parameter cannot be empty.")

    try:
        # Fetch the stock data for the symbol
        symbol = stock_symbol
        stock_info = yf.Ticker(symbol).info
        stock_info = {
            "ticker": stock_info["symbol"],
            "company_Name": stock_info["longName"],
            "industry": stock_info["industry"]
        }

    except KeyError as e:
        raise KeyError(f"Stock symbol '{symbol}' is invalid or not found.") from e

    try:
        # Create a DataFrame with the stock data
        stock_info_df = pd.DataFrame(
            stock_info,
            index=[0]
        )
        return stock_info_df

    except KeyError as e:
        raise KeyError("Could not transform stock_info to a pandas dataframe") from e
# Import stock data using yfinance and a list of stock symbols
def fetch_stock_price_data(stock_ticker="", start_date=(datetime.datetime.now() - relativedelta(years=15))):
    """
    Fetches stock data using yfinance for the given stock symbols and returns a pandas DataFrame.

    The DataFrame will contain the date, stock name, stock symbol, opening price, currency, and trade_Volume.

    Parameters:
    - stock_tickers (list): A list of stock symbols.

    Returns:
    pandas.DataFrame: A DataFrame containing the fetched stock data.

    Raises:
    - ValueError: If the stock_tickers parameter is empty.
    - ValueError: If the start_date parameter is empty.
    - KeyError: If any of the stock symbols are invalid or not found.
    - KeyError: If the stock_price_data cannot be transformed to a pandas DataFrame.
    - KeyError: If the stock_info cannot be transformed to a pandas DataFrame.
    - KeyError: If the stock_price_data_df cannot be joined with stock_info_df.
    """
    # Check if the stock_tickers parameter is empty
    if len(stock_ticker) == "":
        raise ValueError("The stock_tickers parameter cannot be empty.")
    # Check if the start_date parameter is empty
    if start_date == "":
        raise ValueError("The start_date parameter cannot be empty.")

    try:
        ticker = stock_ticker
        # Fetch the stock data for the ticker
        stock_price_data = yf.download(
            ticker, start=start_date
        )

    except KeyError as e:
        raise KeyError(f"Stock ticker '{stock_ticker}' is invalid or not found.") from e

    try:
        # Create a DataFrame with the stock data
        stock_price_data_df = pd.DataFrame(
            stock_price_data
        )
        # Reset the index of the DataFrame
        stock_price_data_df = stock_price_data_df.reset_index()
        stock_price_data_df = stock_price_data_df[["Date", "Open", "Volume"]]
        # Rename the columns
        stock_price_data_df = stock_price_data_df.rename(
            columns={
                "Date" : "date", "Open" : "open_Price", "Volume" : "trade_Volume"
        })

    except KeyError as e:
        raise KeyError("Could not transform stock_price_data to a pandas dataframe") from e

    try:
        # Fetch the stock data for the ticker
        stock_info = yf.Ticker(ticker).info
        stock_info = {
            "ticker": stock_info["symbol"],
            "currency": stock_info["currency"]
        }

    except KeyError as e:
        raise KeyError(f"Stock ticker '{stock_ticker}' is invalid or not found.") from e

    try:
        # Create a DataFrame with the stock data
        stock_info_df = pd.DataFrame(
            stock_info,
            index=[0]
        )

    except KeyError as e:
        raise KeyError("Could not transform stock_info to a pandas dataframe") from e

    try:
        # Create a temporary DataFrame with the stock data joined with the stock_price_data_df and stock_info_df
        stock_price_data_df = stock_price_data_df.join(
            stock_info_df,
            how="cross"
        )
        return stock_price_data_df

    except KeyError as e:
        raise KeyError("Could not join stock_price_data_df with stock_info_df") from e
# Calculate the period returns for the given stock data
def calculate_period_returns(stock_price_data_df):
    """
    Calculates the period returns for the given stock data and returns a pandas DataFrame.

    The DataFrame will contain the date, stock name, stock symbol, and the period returns.

    Parameters:
    - stock_price_data_df (pandas.DataFrame): A DataFrame containing the stock data.

    Returns:
    pandas.DataFrame: A DataFrame containing the period returns.

    Raises:
    - ValueError: If the stock_price_data_df parameter is empty.
    - ValueError: If the stock_price_data_df parameter contains null values.
    - KeyError: If the periodic returns cannot be calculated from the specified DataFrame.
    - KeyError: If the rows cannot be shifted by 1.
    """
    # Check if the stock_price_data_df parameter is empty
    if stock_price_data_df.empty:
        raise ValueError("The stock_price_data_df parameter cannot be empty.")

    if stock_price_data_df["open_Price"].isnull().values.any():
        raise ValueError("""The stock_price_data_df parameter "open_Price" cannot contain null values.""")

    try:
        # Create a new columns in stock_price_data_df called 1D, 1M, 3M, 6M, 9M, 1Y, 2Y, 3Y, 4Y, and 5Y
        stock_price_data_df["1D"] = stock_price_data_df["open_Price"].pct_change(1)
        stock_price_data_df["1M"] = stock_price_data_df["open_Price"].pct_change(21)
        stock_price_data_df["3M"] = stock_price_data_df["open_Price"].pct_change(63)
        stock_price_data_df["6M"] = stock_price_data_df["open_Price"].pct_change(126)
        stock_price_data_df["9M"] = stock_price_data_df["open_Price"].pct_change(189)
        stock_price_data_df["1Y"] = stock_price_data_df["open_Price"].pct_change(252)
        stock_price_data_df["2Y"] = stock_price_data_df["open_Price"].pct_change(504)
        stock_price_data_df["3Y"] = stock_price_data_df["open_Price"].pct_change(756)
        stock_price_data_df["4Y"] = stock_price_data_df["open_Price"].pct_change(1008)
        stock_price_data_df["5Y"] = stock_price_data_df["open_Price"].pct_change(1260)

    except KeyError as e:
        raise KeyError("COuld not calculate periodic returns from spicified dataframe.") from e

    try:
        # Shift the rows by 1
        stock_price_data_df[["1M", "3M", "6M", "9M", "1Y", "2Y", "3Y", "4Y", "5Y"]] = stock_price_data_df[["1M", "3M", "6M", "9M", "1Y", "2Y", "3Y", "4Y", "5Y"]].shift(periods=1)
        # Return the stock_price_data_df DataFrame
        return stock_price_data_df

    except KeyError as e:
        raise KeyError("Could not shift the rows by 1.") from e
# Calculate the moving averages for the given stock data
def calculate_moving_averages(stock_price_data_df):
    """
    Calculates the moving averages for the given stock data and returns a pandas DataFrame.

    The DataFrame will contain the date, stock name, stock symbol, price, and the moving averages.

    Parameters:
    - stock_price_data_df (pandas.DataFrame): A DataFrame containing the stock data.

    Returns:
    pandas.DataFrame: A DataFrame containing the moving averages.

    Raises:
    - ValueError: If the stock_price_data_df parameter is empty.
    - ValueError: If the stock_price_data_df parameter contains null values.
    - KeyError: If columns cannot be created in the specified DataFrame.
    - KeyError: If the moving averages cannot be calculated from the specified DataFrame.
    - KeyError: If the rows cannot be shifted by 1.
    """
    # Check if the stock_price_data_df parameter is empty
    if stock_price_data_df.empty:
        raise ValueError("The stock_price_data_df parameter cannot be empty.")

    if stock_price_data_df["open_Price"].isnull().values.any():
        raise ValueError("""The stock_price_data_df parameter "open_Price" cannot contain null values.""")

    try:
        # Create a new columns in stock_price_data_df called sma_40, sma_120, ema_40, and ema_120
        stock_price_data_df["sma_40"] = 0.0
        stock_price_data_df["sma_120"] = 0.0
        stock_price_data_df["ema_40"] = 0.0
        stock_price_data_df["ema_120"] = 0.0

    except KeyError as e:
        raise KeyError("Could not create new columns in the spicified dataframe.") from e

    try:
        # Loop through each row in stock_price_data_df
        for index, row in stock_price_data_df.iterrows():
            # Calculate sma_40 for every row
            if index == 0:
                sma_40 = stock_price_data_df.iloc[index]["open_Price"]
            elif index < 40:
                sma_40 = stock_price_data_df.iloc[0:index+1]["open_Price"].mean()
            elif index >= 40:
                sma_40 = stock_price_data_df.iloc[index-39:index+1]["open_Price"].mean()

            # Update the sma_40 column with the calculated value
            stock_price_data_df.loc[index, "sma_40"] = sma_40
            # Calculate sma_120 for every row
            if index == 0:
                sma_120 = stock_price_data_df.iloc[index]["open_Price"]
            elif index < 120:
                sma_120 = stock_price_data_df.iloc[0:index+1]["open_Price"].mean()
            elif index >= 120:
                sma_120 = stock_price_data_df.iloc[index-119:index+1]["open_Price"].mean()

            # Update the sma_120 column with the calculated value
            stock_price_data_df.loc[index, "sma_120"] = sma_120
            # Calculate ema_40 for every row
            if index == 0:
                ema_40 = stock_price_data_df.iloc[index]["open_Price"]
            elif index < 40:
                ema_40 = stock_price_data_df.iloc[0:index+1]["open_Price"].ewm(span=40).mean()
                if ema_40.empty:
                    ema_40 = 0.0
                else:
                    ema_40 = ema_40.values[-1]
            elif index >= 40:
                ema_40 = stock_price_data_df.iloc[index-39:index+1]["open_Price"].ewm(span=40).mean()
                ema_40 = ema_40.values[-1]

            # Update the ema_40 column with the calculated value
            stock_price_data_df.loc[index, "ema_40"] = ema_40
            # Calculate ema_120 for every row
            if index == 0:
                ema_120 = stock_price_data_df.iloc[index]["open_Price"]
            elif index < 120:
                ema_120 = stock_price_data_df.iloc[0:index+1]["open_Price"].ewm(span=120).mean()
                if ema_120.empty:
                    ema_120 = 0.0
                else:
                    ema_120 = ema_120.values[-1]
            elif index >= 120:
                ema_120 = stock_price_data_df.iloc[index-119:index+1]["open_Price"].ewm(span=120).mean()
                ema_120 = ema_120.values[-1]

            # Update the ema_120 column with the calculated value
            stock_price_data_df.loc[index, "ema_120"] = ema_120
            # Create print statement per 250 index processed
            if index % 250 == 0:
                print(f"Processed {index} rows, out of {len(stock_price_data_df)} rows.")

    except KeyError as e:
        raise KeyError("Could not calculate moving averages from specified DataFrame.") from e

    try:
        stock_price_data_df[["sma_40", "sma_120", "ema_40", "ema_120"]] = stock_price_data_df[["sma_40", "sma_120", "ema_40", "ema_120"]].shift(1)
        print("Moving averages calculated successfully.")
        # Return the stock_price_data_df DataFrame
        return stock_price_data_df

    except KeyError as e:
        raise KeyError("Could not shift the rows by 1.") from e
# Calculate the standard deviation of the stock price
def calculate_standard_diviation_value(stock_price_data_df):
    """
    Calculates the standard deviation of the stock price and returns a pandas DataFrame.

    The DataFrame will contain the date, stock name, stock symbol, price, and the standard deviation of the stock price.

    Parameters:
    - stock_price_data_df (pandas.DataFrame): A DataFrame containing the stock data.

    Returns:
    pandas.DataFrame: A DataFrame containing the standard deviation of the stock price.

    Raises:
    - ValueError: If the stock_price_data_df parameter is empty.
    - ValueError: If the stock_price_data_df parameter "open_Price" contains null values.
    - KeyError: If new columns cannot be created in the specified DataFrame.
    - KeyError: If the standard deviation of the stock price cannot be calculated from the specified DataFrame.
    - KeyError: If the rows cannot be shifted by 1.
    """
    # Checking if the combined_stock_price_data_df parameter is empty
    if stock_price_data_df.empty:
        raise ValueError("No stock data provided.")

    if stock_price_data_df["open_Price"].isnull().values.any():
        raise ValueError("""The stock_price_data_df parameter "open_Price" cannot contain null values.""")

    try:
        # Calculate the standard deviation of the stock price
        # Create a new columns in stock_price_data_df called std_Div_40, std_Div_120
        stock_price_data_df["std_Div_40"] = 0.0
        stock_price_data_df["std_Div_120"] = 0.0

    except KeyError as e:
        raise KeyError("Could not create new columns in the spicified dataframe") from e

    try:
        # Loop through each row in stock_price_data_df
        for index, row in stock_price_data_df.iterrows():
            # Calculate std_Div_40 for every row
            if index == 0:
                Std_Div_40 = 0.0
            elif index < 40:
                Std_Div_40 = stock_price_data_df.iloc[0:index+1]["open_Price"].std()
            elif index >= 40:
                Std_Div_40 = stock_price_data_df.iloc[index-39:index+1]["open_Price"].std()

            # Update the std_Div_40 column with the calculated value
            stock_price_data_df.loc[index, "std_Div_40"] = Std_Div_40
            # Calculate std_Div_120 for every row
            if index == 0:
                Std_Div_120 = 0.0
            elif index < 120:
                Std_Div_120 = stock_price_data_df.iloc[0:index+1]["open_Price"].std()
            elif index >= 120:
                Std_Div_120 = stock_price_data_df.iloc[index-119:index+1]["open_Price"].std()

            # Update the std_Div_120 column with the calculated value
            stock_price_data_df.loc[index, "std_Div_120"] = Std_Div_120
            # Create print statement per 250 index processed
            if index % 250 == 0:
                print(f"Processed {index} rows, out of {len(stock_price_data_df)} rows.")

    except KeyError as e:
        raise KeyError("Could not calculate standard deviation of the stock price.") from e

    try:
        print("Standard deviation of the stock price calculated successfully.")
        # Return the stock_price_data_df DataFrame
        stock_price_data_df[["std_Div_40", "std_Div_120"]] = stock_price_data_df[["std_Div_40", "std_Div_120"]].shift(1)
        return stock_price_data_df

    except KeyError as e:
        raise KeyError("Could not shift the rows by 1.") from e
# Calculate the stock price momentum
def calculate_bollinger_bands(stock_price_data_df):
    """
    Calculates the Bollinger Bands for the given stock data and returns a pandas DataFrame.

    The DataFrame will contain the date, stock name, stock symbol, price, and the Bollinger Bands.

    Parameters:
    - stock_price_data_df (pandas.DataFrame): A DataFrame containing the stock data.

    Returns:
    pandas.DataFrame: A DataFrame containing the Bollinger Bands.

    Raises:
    - ValueError: If the stock_price_data_df parameter is empty.
    - ValueError: If the stock_price_data_df parameter "sma_40" contains null values.
    - ValueError: If the stock_price_data_df parameter "sma_120" contains null values.
    - ValueError: If the stock_price_data_df parameter "std_Div_40" contains null values.
    - ValueError: If the stock_price_data_df parameter "std_Div_120" contains null values.
    - KeyError: If new columns cannot be created in the specified DataFrame.
    - KeyError: If the Bollinger Bands cannot be calculated from the specified DataFrame.
    - KeyError: If the rows cannot be shifted by 1.
    """
    # Checking if the stock_price_data_df parameter is empty
    if stock_price_data_df.empty:
        raise ValueError("No stock data provided.")

    if stock_price_data_df.iloc[40:]["sma_40"].isnull().values.any():
        raise ValueError("""The stock_price_data_df parameter "sma_40" cannot contain null values.""")

    if stock_price_data_df.iloc[120:]["sma_120"].isnull().values.any():
        raise ValueError("""The stock_price_data_df parameter "sma_120" cannot contain null values.""")

    if stock_price_data_df.iloc[40:]["std_Div_40"].isnull().values.any():
        raise ValueError("""The stock_price_data_df parameter "std_Div_40" cannot contain null values.""")

    if stock_price_data_df.iloc[120:]["std_Div_120"].isnull().values.any():
        raise ValueError("""The stock_price_data_df parameter "std_Div_120" cannot contain null values.""")

    try:
        # Calculate the Bollinger Bands for the given stock data
        # Create a new columns in stock_price_data_df called Bollinger_Bands_40, Bollinger_Bands_120
        stock_price_data_df["bollinger_Band_40_2STD"] = 0.0
        stock_price_data_df["bollinger_Band_120_2STD"] = 0.0

    except KeyError as e:
        raise KeyError("Could not create new columns in the spicified dataframe") from e

    try:
        # Loop through each row in stock_price_data_df
        for index, row in stock_price_data_df.iterrows():
            # Calculate Bollinger_Bands_40 for every row
            Bollinger_Band_40_Upper = stock_price_data_df.iloc[index]["sma_40"] + (stock_price_data_df.iloc[index]["std_Div_40"] * 2)
            Bollinger_Band_40_Lower = stock_price_data_df.iloc[index]["sma_40"] - (stock_price_data_df.iloc[index]["std_Div_40"] * 2)
            # Update the Bollinger_Bands_40 column with the calculated value
            stock_price_data_df.loc[index, "bollinger_Band_40_2STD"] = Bollinger_Band_40_Upper - Bollinger_Band_40_Lower
            Bollinger_Band_120_Upper = stock_price_data_df.iloc[index]["sma_120"] + (stock_price_data_df.iloc[index]["std_Div_120"] * 2)
            Bollinger_Band_120_Lower = stock_price_data_df.iloc[index]["sma_120"] - (stock_price_data_df.iloc[index]["std_Div_120"] * 2)
            # Update the Bollinger_Bands_120 column with the calculated value
            stock_price_data_df.loc[index, "bollinger_Band_120_2STD"] = Bollinger_Band_120_Upper - Bollinger_Band_120_Lower
            # Create print statement per 250 index processed
            if index % 250 == 0:
                print(f"Processed {index} rows, out of {len(stock_price_data_df)} rows.")

        print("Bollinger Bands calculated successfully.")

    except KeyError as e:
        raise KeyError("Could not calculate Bollinger Bands from specified DataFrame.") from e

    try:
        # Return the stock_price_data_df DataFrame
        stock_price_data_df[["bollinger_Band_40_2STD", "bollinger_Band_120_2STD"]] = stock_price_data_df[["bollinger_Band_40_2STD", "bollinger_Band_120_2STD"]].shift(1)
        return stock_price_data_df

    except KeyError as e:
        raise KeyError("Could not shift the rows by 1.") from e
# Calculate the stock price momentum
def calculate_momentum(stock_price_data_df):
    """
    Calculates the momentum for the given stock data and returns a pandas DataFrame.

    The DataFrame will contain the date, stock name, stock symbol, price, and the momentum.

    Parameters:
    - stock_price_data_df (pandas.DataFrame): A DataFrame containing the stock data.

    Returns:
    pandas.DataFrame: A DataFrame containing the momentum.

    Raises:
    - ValueError: If the stock_price_data_df parameter is empty.
    - ValueError: If the stock_price_data_df parameter "open_Price" contains null values.
    - KeyError: If new columns cannot be created in the specified DataFrame.
    - KeyError: If the momentum cannot be calculated from the specified DataFrame.
    - KeyError: If the rows cannot be shifted by 1.
    """
    # Checking if the stock_price_data_df parameter is empty
    if stock_price_data_df.empty:
        raise ValueError("No stock data provided.")

    if stock_price_data_df.iloc[40:]["open_Price"].isnull().values.any():
        raise ValueError("""The stock_price_data_df parameter "open_Price" cannot contain null values.""")

    # Calculate the momentum for the given stock data
    try:
        # Create a new columns in stock_price_data_df called momentum
        stock_price_data_df["momentum"] = 0.0

    except KeyError as e:
        raise KeyError("Could not create new columns in the spicified dataframe") from e

    try:
        for index, row in stock_price_data_df.iterrows():
            # Calculate std_Div_40 for every row
            if index == 0:
                momentum = 0.0
            elif stock_price_data_df.iloc[index]["open_Price"] >= stock_price_data_df.iloc[index-1]["open_Price"]:
                if stock_price_data_df.loc[index-1, "momentum"] <= 0:
                    momentum = 1
                    # Update the momentum column with the calculated value
                    stock_price_data_df.loc[index, "momentum"] = momentum
                elif stock_price_data_df.loc[index-1, "momentum"] > 0:
                    momentum = stock_price_data_df.loc[index-1, "momentum"] + 1
                    # Update the momentum column with the calculated value
                    stock_price_data_df.loc[index, "momentum"] = momentum
            elif stock_price_data_df.iloc[index]["open_Price"] < stock_price_data_df.iloc[index-1]["open_Price"]:
                if stock_price_data_df.loc[index-1, "momentum"] >= 0:
                    momentum = -1
                    # Update the momentum column with the calculated value
                    stock_price_data_df.loc[index, "momentum"] = momentum
                elif stock_price_data_df.loc[index-1, "momentum"] < 0:
                    momentum = stock_price_data_df.loc[index-1, "momentum"] - 1
                    # Update the momentum column with the calculated value
                    stock_price_data_df.loc[index, "momentum"] = momentum

            # Create print statement per 250 index processed
            if index % 250 == 0:
                print(f"Processed {index} rows, out of {len(stock_price_data_df)} rows.")

        print("Momentum calculated successfully.")

    except KeyError as e:
        raise KeyError("Could not calculate momentum from specified DataFrame.") from e

    try:
        stock_price_data_df["momentum"] = stock_price_data_df["momentum"].shift(1)
        # Return the stock_price_data_df DataFrame
        return stock_price_data_df

    except KeyError as e:
        raise KeyError("Could not shift the rows by 1.") from e
# Import financial stock data using yfinance and a list of stock symbols
def fetch_stock_financial_data(stock_symbol):
    """
    Fetches financial stock data using yfinance and a list of stock symbols.

    The function fetches financial stock data using yfinance and a list of stock symbols and returns a DataFrame with the fetched data.

    Parameters:
    - stock_symbols (list): A list of stock symbols.

    Returns:
    - stock_data_df (pd.DataFrame): A DataFrame with the fetched data.

    Raises:
    - ValueError: If the stock_symbols parameter is empty.
    - KeyError: If any of the stock symbols in the list is invalid.
    """
    # Checking if the stock_symbols parameter is empty
    if not stock_symbol:
        raise ValueError("No stock symbols provided.")

    try:
        symbol = stock_symbol
        # Checking if the stock symbol is valid
        if not yf.Ticker(symbol).info:
            raise KeyError(symbol)

        # Fetching the financial stock data using yfinance
        stock_data = yf.Ticker(symbol)
        income_stmt = stock_data.income_stmt
        income_stmt_df = pd.DataFrame(income_stmt)
        # Checking if the input DataFrame is empty
        if income_stmt_df.empty:
            raise ValueError("Input DataFrame is empty.")

        # Rotate the income_stmt_df dataframe
        income_stmt_df = income_stmt_df.transpose()
        income_stmt_df["Revenue growth"] = 0.0
        income_stmt_df["Gross Profit growth"] = 0.0
        income_stmt_df["Gross Margin"] = 0.0
        income_stmt_df["Gross Margin growth"] = 0.0
        income_stmt_df["Operating Earnings"] = 0.0
        income_stmt_df["Operating Margin"] = 0.0
        income_stmt_df["Operating Margin growth"] = 0.0
        income_stmt_df["Net Income growth"] = 0.0
        income_stmt_df["Net Income Margin"] = 0.0
        income_stmt_df["Net Income Margin growth"] = 0.0
        income_stmt_df["EPS"] = 0.0
        income_stmt_df["EPS growth"] = 0.0
        # Invert the rows in income_stmt_df dataframe
        income_stmt_df = income_stmt_df.iloc[::-1]
        # Reset the index of the income_stmt_df dataframe
        income_stmt_df = income_stmt_df.reset_index()
        # Rename the index column to Date
        income_stmt_df = income_stmt_df.rename(columns={"index": "Date"})
        # Fill the NaN values in the income_stmt_df dataframe with values from the previous row
        income_stmt_df = income_stmt_df.ffill()
        # Use stock_data to chack if bank is part of the registered industry for the stock
        stock_info = stock_data.info
        industry = stock_info["industry"]
        # Check if bank is part of the registered industry for the stock
        if "banks" in industry.lower():
            # Check is "Gross Profit" is in the income_stmt_df dataframe
            if "Gross Profit" not in income_stmt_df.columns:
                if "Operating Income" not in income_stmt_df.columns:
                    for index, row in income_stmt_df.iterrows():
                        income_stmt_df.loc[index, "Net Income Margin"] = income_stmt_df.loc[index, "Net Income Common Stockholders"] / income_stmt_df.loc[index, "Total Revenue"]
                        income_stmt_df.loc[index, "EPS"] = income_stmt_df.loc[index, "Net Income Common Stockholders"] / income_stmt_df.loc[index, "Diluted Average Shares"]
                        if index == 0:
                            income_stmt_df.loc[index, "Revenue growth"] = 0.0
                            income_stmt_df.loc[index, "Net Income growth"] = 0.0
                            income_stmt_df.loc[index, "Net Income Margin growth"] = 0.0
                            income_stmt_df.loc[index, "EPS growth"] = 0.0
                        else:
                            income_stmt_df.loc[index, "Revenue growth"] = (income_stmt_df.iloc[index]["Total Revenue"] / income_stmt_df.iloc[index-1]["Total Revenue"])-1
                            income_stmt_df.loc[index, "Net Income growth"] = (income_stmt_df.iloc[index]["Net Income Common Stockholders"] / income_stmt_df.iloc[index-1]["Net Income Common Stockholders"])-1
                            income_stmt_df.loc[index, "Net Income Margin growth"] = (income_stmt_df.iloc[index]["Net Income Margin"] / income_stmt_df.iloc[index-1]["Net Income Margin"])-1
                            income_stmt_df.loc[index, "EPS growth"] = (income_stmt_df.iloc[index]["EPS"] / income_stmt_df.iloc[index-1]["EPS"])-1
                else:
                    for index, row in income_stmt_df.iterrows():
                        income_stmt_df.loc[index, "Operating Margin"] = income_stmt_df.loc[index, "Operating Income"] / income_stmt_df.loc[index, "Total Revenue"]
                        income_stmt_df.loc[index, "Net Income Margin"] = income_stmt_df.loc[index, "Net Income Common Stockholders"] / income_stmt_df.loc[index, "Total Revenue"]
                        income_stmt_df.loc[index, "EPS"] = income_stmt_df.loc[index, "Net Income Common Stockholders"] / income_stmt_df.loc[index, "Diluted Average Shares"]
                        if index == 0:
                            income_stmt_df.loc[index, "Revenue growth"] = 0.0
                            income_stmt_df.loc[index, "Operating Earnings growth"] = 0.0
                            income_stmt_df.loc[index, "Operating Margin growth"] = 0.0
                            income_stmt_df.loc[index, "Net Income growth"] = 0.0
                            income_stmt_df.loc[index, "Net Income Margin growth"] = 0.0
                            income_stmt_df.loc[index, "EPS growth"] = 0.0
                        else:
                            income_stmt_df.loc[index, "Revenue growth"] = (income_stmt_df.iloc[index]["Total Revenue"] / income_stmt_df.iloc[index-1]["Total Revenue"])-1
                            income_stmt_df.loc[index, "Operating Earnings growth"] = (income_stmt_df.iloc[index]["Operating Income"] / income_stmt_df.iloc[index-1]["Operating Income"])-1
                            income_stmt_df.loc[index, "Operating Margin growth"] = (income_stmt_df.iloc[index]["Operating Margin"] / income_stmt_df.iloc[index-1]["Operating Margin"])-1
                            income_stmt_df.loc[index, "Net Income growth"] = (income_stmt_df.iloc[index]["Net Income Common Stockholders"] / income_stmt_df.iloc[index-1]["Net Income Common Stockholders"])-1
                            income_stmt_df.loc[index, "Net Income Margin growth"] = (income_stmt_df.iloc[index]["Net Income Margin"] / income_stmt_df.iloc[index-1]["Net Income Margin"])-1
                            income_stmt_df.loc[index, "EPS growth"] = (income_stmt_df.iloc[index]["EPS"] / income_stmt_df.iloc[index-1]["EPS"])-1
            else:
                for index, row in income_stmt_df.iterrows():
                    income_stmt_df.loc[index, "Gross Margin"] = income_stmt_df.loc[index, "Gross Profit"] / income_stmt_df.loc[index, "Total Revenue"]
                    income_stmt_df.loc[index, "Operating Margin"] = income_stmt_df.loc[index, "Operating Income"] / income_stmt_df.loc[index, "Total Revenue"]
                    income_stmt_df.loc[index, "Net Income Margin"] = income_stmt_df.loc[index, "Net Income Common Stockholders"] / income_stmt_df.loc[index, "Total Revenue"]
                    income_stmt_df.loc[index, "EPS"] = income_stmt_df.loc[index, "Net Income Common Stockholders"] / income_stmt_df.loc[index, "Diluted Average Shares"]
                    if index == 0:
                        income_stmt_df.loc[index, "Revenue growth"] = 0.0
                        income_stmt_df.loc[index, "Gross Profit growth"] = 0.0
                        income_stmt_df.loc[index, "Gross Margin growth"] = 0.0
                        income_stmt_df.loc[index, "Operating Earnings growth"] = 0.0
                        income_stmt_df.loc[index, "Operating Margin growth"] = 0.0
                        income_stmt_df.loc[index, "Net Income growth"] = 0.0
                        income_stmt_df.loc[index, "Net Income Margin growth"] = 0.0
                    else:
                        income_stmt_df.loc[index, "Revenue growth"] = (income_stmt_df.iloc[index]["Total Revenue"] / income_stmt_df.iloc[index-1]["Total Revenue"])-1
                        income_stmt_df.loc[index, "Gross Profit growth"] = (income_stmt_df.iloc[index]["Gross Profit"] / income_stmt_df.iloc[index-1]["Gross Profit"])-1
                        income_stmt_df.loc[index, "Gross Margin growth"] = (income_stmt_df.iloc[index]["Gross Margin"] / income_stmt_df.iloc[index-1]["Gross Margin"])-1
                        income_stmt_df.loc[index, "Operating Earnings growth"] = (income_stmt_df.iloc[index]["Operating Income"] / income_stmt_df.iloc[index-1]["Operating Income"])-1
                        income_stmt_df.loc[index, "Operating Margin growth"] = (income_stmt_df.iloc[index]["Operating Margin"] / income_stmt_df.iloc[index-1]["Operating Margin"])-1
                        income_stmt_df.loc[index, "Net Income growth"] = (income_stmt_df.iloc[index]["Net Income Common Stockholders"] / income_stmt_df.iloc[index-1]["Net Income Common Stockholders"])-1
                        income_stmt_df.loc[index, "Net Income Margin growth"] = (income_stmt_df.iloc[index]["Net Income Margin"] / income_stmt_df.iloc[index-1]["Net Income Margin"])-1
        elif "insurance" in industry.lower():
            # Check is "Gross Profit" is in the income_stmt_df dataframe
            if "Gross Profit" not in income_stmt_df.columns:
                if "Operating Income" not in income_stmt_df.columns:
                    for index, row in income_stmt_df.iterrows():
                        income_stmt_df.loc[index, "Net Income Margin"] = income_stmt_df.loc[index, "Net Income Common Stockholders"] / income_stmt_df.loc[index, "Total Revenue"]
                        income_stmt_df.loc[index, "EPS"] = income_stmt_df.loc[index, "Net Income Common Stockholders"] / income_stmt_df.loc[index, "Diluted Average Shares"]
                        if index == 0:
                            income_stmt_df.loc[index, "Revenue growth"] = 0.0
                            income_stmt_df.loc[index, "Net Income growth"] = 0.0
                            income_stmt_df.loc[index, "Net Income Margin growth"] = 0.0
                            income_stmt_df.loc[index, "EPS growth"] = 0.0
                        else:
                            income_stmt_df.loc[index, "Revenue growth"] = (income_stmt_df.iloc[index]["Total Revenue"] / income_stmt_df.iloc[index-1]["Total Revenue"])-1
                            income_stmt_df.loc[index, "Net Income growth"] = (income_stmt_df.iloc[index]["Net Income Common Stockholders"] / income_stmt_df.iloc[index-1]["Net Income Common Stockholders"])-1
                            income_stmt_df.loc[index, "Net Income Margin growth"] = (income_stmt_df.iloc[index]["Net Income Margin"] / income_stmt_df.iloc[index-1]["Net Income Margin"])-1
                            income_stmt_df.loc[index, "EPS growth"] = (income_stmt_df.iloc[index]["EPS"] / income_stmt_df.iloc[index-1]["EPS"])-1
                else:
                    for index, row in income_stmt_df.iterrows():
                        income_stmt_df.loc[index, "Operating Margin"] = income_stmt_df.loc[index, "Operating Income"] / income_stmt_df.loc[index, "Total Revenue"]
                        income_stmt_df.loc[index, "Net Income Margin"] = income_stmt_df.loc[index, "Net Income Common Stockholders"] / income_stmt_df.loc[index, "Total Revenue"]
                        income_stmt_df.loc[index, "EPS"] = income_stmt_df.loc[index, "Net Income Common Stockholders"] / income_stmt_df.loc[index, "Diluted Average Shares"]
                        if index == 0:
                            income_stmt_df.loc[index, "Revenue growth"] = 0.0
                            income_stmt_df.loc[index, "Operating Earnings growth"] = 0.0
                            income_stmt_df.loc[index, "Operating Margin growth"] = 0.0
                            income_stmt_df.loc[index, "Net Income growth"] = 0.0
                            income_stmt_df.loc[index, "Net Income Margin growth"] = 0.0
                            income_stmt_df.loc[index, "EPS growth"] = 0.0
                        else:
                            income_stmt_df.loc[index, "Revenue growth"] = (income_stmt_df.iloc[index]["Total Revenue"] / income_stmt_df.iloc[index-1]["Total Revenue"])-1
                            income_stmt_df.loc[index, "Operating Earnings growth"] = (income_stmt_df.iloc[index]["Operating Income"] / income_stmt_df.iloc[index-1]["Operating Income"])-1
                            income_stmt_df.loc[index, "Operating Margin growth"] = (income_stmt_df.iloc[index]["Operating Margin"] / income_stmt_df.iloc[index-1]["Operating Margin"])-1
                            income_stmt_df.loc[index, "Net Income growth"] = (income_stmt_df.iloc[index]["Net Income Common Stockholders"] / income_stmt_df.iloc[index-1]["Net Income Common Stockholders"])-1
                            income_stmt_df.loc[index, "Net Income Margin growth"] = (income_stmt_df.iloc[index]["Net Income Margin"] / income_stmt_df.iloc[index-1]["Net Income Margin"])-1
                            income_stmt_df.loc[index, "EPS growth"] = (income_stmt_df.iloc[index]["EPS"] / income_stmt_df.iloc[index-1]["EPS"])-1
            else:
                for index, row in income_stmt_df.iterrows():
                    income_stmt_df.loc[index, "Gross Margin"] = income_stmt_df.loc[index, "Gross Profit"] / income_stmt_df.loc[index, "Total Revenue"]
                    income_stmt_df.loc[index, "Operating Margin"] = income_stmt_df.loc[index, "Operating Income"] / income_stmt_df.loc[index, "Total Revenue"]
                    income_stmt_df.loc[index, "Net Income Margin"] = income_stmt_df.loc[index, "Net Income Common Stockholders"] / income_stmt_df.loc[index, "Total Revenue"]
                    income_stmt_df.loc[index, "EPS"] = income_stmt_df.loc[index, "Net Income Common Stockholders"] / income_stmt_df.loc[index, "Diluted Average Shares"]
                    if index == 0:
                        income_stmt_df.loc[index, "Revenue growth"] = 0.0
                        income_stmt_df.loc[index, "Gross Profit growth"] = 0.0
                        income_stmt_df.loc[index, "Gross Margin growth"] = 0.0
                        income_stmt_df.loc[index, "Operating Earnings growth"] = 0.0
                        income_stmt_df.loc[index, "Operating Margin growth"] = 0.0
                        income_stmt_df.loc[index, "Net Income growth"] = 0.0
                        income_stmt_df.loc[index, "Net Income Margin growth"] = 0.0
                        income_stmt_df.loc[index, "EPS growth"] = 0.0
                    else:
                        income_stmt_df.loc[index, "Revenue growth"] = (income_stmt_df.iloc[index]["Total Revenue"] / income_stmt_df.iloc[index-1]["Total Revenue"])-1
                        income_stmt_df.loc[index, "Gross Profit growth"] = (income_stmt_df.iloc[index]["Gross Profit"] / income_stmt_df.iloc[index-1]["Gross Profit"])-1
                        income_stmt_df.loc[index, "Gross Margin growth"] = (income_stmt_df.iloc[index]["Gross Margin"] / income_stmt_df.iloc[index-1]["Gross Margin"])-1
                        income_stmt_df.loc[index, "Operating Earnings growth"] = (income_stmt_df.iloc[index]["Operating Income"] / income_stmt_df.iloc[index-1]["Operating Income"])-1
                        income_stmt_df.loc[index, "Operating Margin growth"] = (income_stmt_df.iloc[index]["Operating Margin"] / income_stmt_df.iloc[index-1]["Operating Margin"])-1
                        income_stmt_df.loc[index, "Net Income growth"] = (income_stmt_df.iloc[index]["Net Income Common Stockholders"] / income_stmt_df.iloc[index-1]["Net Income Common Stockholders"])-1
                        income_stmt_df.loc[index, "Net Income Margin growth"] = (income_stmt_df.iloc[index]["Net Income Margin"] / income_stmt_df.iloc[index-1]["Net Income Margin"])-1
                        income_stmt_df.loc[index, "EPS growth"] = (income_stmt_df.iloc[index]["EPS"] / income_stmt_df.iloc[index-1]["EPS"])-1
        elif "biotechnology" in industry.lower():
            # Check is "Gross Profit" is in the income_stmt_df dataframe
            if "Gross Profit" not in income_stmt_df.columns:
                for index, row in income_stmt_df.iterrows():
                    income_stmt_df.loc[index, "Operating Margin"] = income_stmt_df.loc[index, "Operating Income"] / income_stmt_df.loc[index, "Total Revenue"]
                    income_stmt_df.loc[index, "Net Income Margin"] = income_stmt_df.loc[index, "Net Income Common Stockholders"] / income_stmt_df.loc[index, "Total Revenue"]
                    income_stmt_df.loc[index, "EPS"] = income_stmt_df.loc[index, "Net Income Common Stockholders"] / income_stmt_df.loc[index, "Diluted Average Shares"]
                    if index == 0:
                        income_stmt_df.loc[index, "Revenue growth"] = 0.0
                        income_stmt_df.loc[index, "Operating Earnings growth"] = 0.0
                        income_stmt_df.loc[index, "Operating Margin growth"] = 0.0
                        income_stmt_df.loc[index, "Net Income growth"] = 0.0
                        income_stmt_df.loc[index, "Net Income Margin growth"] = 0.0
                        income_stmt_df.loc[index, "EPS growth"] = 0.0
                    else:
                        income_stmt_df.loc[index, "Revenue growth"] = (income_stmt_df.iloc[index]["Total Revenue"] / income_stmt_df.iloc[index-1]["Total Revenue"])-1
                        income_stmt_df.loc[index, "Operating Earnings growth"] = (income_stmt_df.iloc[index]["Operating Income"] / income_stmt_df.iloc[index-1]["Operating Income"])-1
                        income_stmt_df.loc[index, "Operating Margin growth"] = (income_stmt_df.iloc[index]["Operating Margin"] / income_stmt_df.iloc[index-1]["Operating Margin"])-1
                        income_stmt_df.loc[index, "Net Income growth"] = (income_stmt_df.iloc[index]["Net Income Common Stockholders"] / income_stmt_df.iloc[index-1]["Net Income Common Stockholders"])-1
                        income_stmt_df.loc[index, "Net Income Margin growth"] = (income_stmt_df.iloc[index]["Net Income Margin"] / income_stmt_df.iloc[index-1]["Net Income Margin"])-1
                        income_stmt_df.loc[index, "EPS growth"] = (income_stmt_df.iloc[index]["EPS"] / income_stmt_df.iloc[index-1]["EPS"])-1
            else:
                for index, row in income_stmt_df.iterrows():
                    income_stmt_df.loc[index, "Gross Margin"] = income_stmt_df.loc[index, "Gross Profit"] / income_stmt_df.loc[index, "Total Revenue"]
                    income_stmt_df.loc[index, "Operating Margin"] = income_stmt_df.loc[index, "Operating Income"] / income_stmt_df.loc[index, "Total Revenue"]
                    income_stmt_df.loc[index, "Net Income Margin"] = income_stmt_df.loc[index, "Net Income Common Stockholders"] / income_stmt_df.loc[index, "Total Revenue"]
                    income_stmt_df.loc[index, "EPS"] = income_stmt_df.loc[index, "Net Income Common Stockholders"] / income_stmt_df.loc[index, "Diluted Average Shares"]
                    if index == 0:
                        income_stmt_df.loc[index, "Revenue growth"] = 0.0
                        income_stmt_df.loc[index, "Gross Profit growth"] = 0.0
                        income_stmt_df.loc[index, "Gross Margin growth"] = 0.0
                        income_stmt_df.loc[index, "Operating Earnings growth"] = 0.0
                        income_stmt_df.loc[index, "Operating Margin growth"] = 0.0
                        income_stmt_df.loc[index, "Net Income growth"] = 0.0
                        income_stmt_df.loc[index, "Net Income Margin growth"] = 0.0
                        income_stmt_df.loc[index, "EPS growth"] = 0.0
                    else:
                        income_stmt_df.loc[index, "Revenue growth"] = (income_stmt_df.iloc[index]["Total Revenue"] / income_stmt_df.iloc[index-1]["Total Revenue"])-1
                        income_stmt_df.loc[index, "Gross Profit growth"] = (income_stmt_df.iloc[index]["Gross Profit"] / income_stmt_df.iloc[index-1]["Gross Profit"])-1
                        income_stmt_df.loc[index, "Gross Margin growth"] = (income_stmt_df.iloc[index]["Gross Margin"] / income_stmt_df.iloc[index-1]["Gross Margin"])-1
                        income_stmt_df.loc[index, "Operating Earnings growth"] = (income_stmt_df.iloc[index]["Operating Income"] / income_stmt_df.iloc[index-1]["Operating Income"])-1
                        income_stmt_df.loc[index, "Operating Margin growth"] = (income_stmt_df.iloc[index]["Operating Margin"] / income_stmt_df.iloc[index-1]["Operating Margin"])-1
                        income_stmt_df.loc[index, "Net Income growth"] = (income_stmt_df.iloc[index]["Net Income Common Stockholders"] / income_stmt_df.iloc[index-1]["Net Income Common Stockholders"])-1
                        income_stmt_df.loc[index, "Net Income Margin growth"] = (income_stmt_df.iloc[index]["Net Income Margin"] / income_stmt_df.iloc[index-1]["Net Income Margin"])-1
                        income_stmt_df.loc[index, "EPS growth"] = (income_stmt_df.iloc[index]["EPS"] / income_stmt_df.iloc[index-1]["EPS"])-1
        else:
            for index, row in income_stmt_df.iterrows():
                income_stmt_df.loc[index, "Gross Margin"] = income_stmt_df.loc[index, "Gross Profit"] / income_stmt_df.loc[index, "Total Revenue"]
                income_stmt_df.loc[index, "Operating Margin"] = income_stmt_df.loc[index, "Operating Income"] / income_stmt_df.loc[index, "Total Revenue"]
                income_stmt_df.loc[index, "Net Income Margin"] = income_stmt_df.loc[index, "Net Income Common Stockholders"] / income_stmt_df.loc[index, "Total Revenue"]
                income_stmt_df.loc[index, "EPS"] = income_stmt_df.loc[index, "Net Income Common Stockholders"] / income_stmt_df.loc[index, "Diluted Average Shares"]
                if index == 0:
                    income_stmt_df.loc[index, "Revenue growth"] = 0.0
                    income_stmt_df.loc[index, "Gross Profit growth"] = 0.0
                    income_stmt_df.loc[index, "Gross Margin growth"] = 0.0
                    income_stmt_df.loc[index, "Operating Earnings growth"] = 0.0
                    income_stmt_df.loc[index, "Operating Margin growth"] = 0.0
                    income_stmt_df.loc[index, "Net Income growth"] = 0.0
                    income_stmt_df.loc[index, "Net Income Margin growth"] = 0.0
                    income_stmt_df.loc[index, "EPS growth"] = 0.0
                else:
                    income_stmt_df.loc[index, "Revenue growth"] = (income_stmt_df.iloc[index]["Total Revenue"] / income_stmt_df.iloc[index-1]["Total Revenue"])-1
                    income_stmt_df.loc[index, "Gross Profit growth"] = (income_stmt_df.iloc[index]["Gross Profit"] / income_stmt_df.iloc[index-1]["Gross Profit"])-1
                    income_stmt_df.loc[index, "Gross Margin growth"] = (income_stmt_df.iloc[index]["Gross Margin"] / income_stmt_df.iloc[index-1]["Gross Margin"])-1
                    income_stmt_df.loc[index, "Operating Earnings growth"] = (income_stmt_df.iloc[index]["Operating Income"] / income_stmt_df.iloc[index-1]["Operating Income"])-1
                    income_stmt_df.loc[index, "Operating Margin growth"] = (income_stmt_df.iloc[index]["Operating Margin"] / income_stmt_df.iloc[index-1]["Operating Margin"])-1
                    income_stmt_df.loc[index, "Net Income growth"] = (income_stmt_df.iloc[index]["Net Income Common Stockholders"] / income_stmt_df.iloc[index-1]["Net Income Common Stockholders"])-1
                    income_stmt_df.loc[index, "Net Income Margin growth"] = (income_stmt_df.iloc[index]["Net Income Margin"] / income_stmt_df.iloc[index-1]["Net Income Margin"])-1
                    income_stmt_df.loc[index, "EPS growth"] = (income_stmt_df.iloc[index]["EPS"] / income_stmt_df.iloc[index-1]["EPS"])-1

        balancesheet = stock_data.balancesheet
        balancesheet_df = pd.DataFrame(balancesheet)
        # Checking if the input DataFrame is empty
        if balancesheet_df.empty:
            raise ValueError("Input DataFrame is empty.")

        # Rotate the balancesheet_df dataframe
        balancesheet_df = balancesheet_df.transpose()
        balancesheet_df["Total Assets growth"] = 0.0
        balancesheet_df["Current Assets growth"] = 0.0
        balancesheet_df["Cash and Cash Equivalents growth"] = 0.0
        balancesheet_df["Total Liabilities growth"] = 0.0
        balancesheet_df["Total Equity growth"] = 0.0
        balancesheet_df["Current Liabilities growth"] = 0.0
        balancesheet_df["Book Value"] = 0.0
        balancesheet_df["Book Value growth"] = 0.0
        balancesheet_df["Book Value per share"] = 0.0
        balancesheet_df["Book Value per share growth"] = 0.0
        balancesheet_df["Return on Assets"] = 0.0
        balancesheet_df["Return on Assets growth"] = 0.0
        balancesheet_df["Return on Equity"] = 0.0
        balancesheet_df["Return on Equity growth"] = 0.0
        balancesheet_df["Return on Invested Capital"] = 0.0
        balancesheet_df["Return on Invested Capital growth"] = 0.0
        balancesheet_df["Current Ratio"] = 0.0
        balancesheet_df["Current Ratio growth"] = 0.0
        balancesheet_df["Quick Ratio"] = 0.0
        balancesheet_df["Quick Ratio growth"] = 0.0
        balancesheet_df["Debt to Equity"] = 0.0
        balancesheet_df["Debt to Equity growth"] = 0.0
        # Invert the rows in balancesheet_df dataframe
        balancesheet_df = balancesheet_df.iloc[::-1]
        # Reset the index of the balancesheet_df dataframe
        balancesheet_df = balancesheet_df.reset_index()
        # Rename the index column to Date
        balancesheet_df = balancesheet_df.rename(columns={"index": "Date"})
        if "banks" in industry.lower():
            if "Current Assets" not in balancesheet_df.columns:
                balancesheet_df["Current Assets"] = 0.0
                balancesheet_df = balancesheet_df.rename(columns={"Derivative Product Liabilities": "Current Liabilities"})
                for index, row in balancesheet_df.iterrows():
                    if "Trading Securities" not in balancesheet_df.columns:
                        balancesheet_df.loc[index, "Current Assets"] = balancesheet_df.loc[index, "Cash And Cash Equivalents"] + balancesheet_df.loc[index, "Receivables"] + balancesheet_df.loc[index, "Financial Assets Designatedas Fair Value Through Profitor Loss Total"]
                    else:
                        balancesheet_df.loc[index, "Current Assets"] = balancesheet_df.loc[index, "Cash And Cash Equivalents"] + balancesheet_df.loc[index, "Receivables"] + balancesheet_df.loc[index, "Trading Securities"] + balancesheet_df.loc[index, "Financial Assets Designatedas Fair Value Through Profitor Loss Total"]
                    balancesheet_df.loc[index, "Book Value"] = balancesheet_df.loc[index, "Total Equity Gross Minority Interest"]
                    balancesheet_df.loc[index, "Book Value per share"] = balancesheet_df.loc[index, "Total Equity Gross Minority Interest"] / income_stmt_df.loc[index, "Diluted Average Shares"]
                    balancesheet_df.loc[index, "Return on Assets"] = income_stmt_df.loc[index, "Net Income Common Stockholders"] / balancesheet_df.loc[index, "Total Assets"]
                    balancesheet_df.loc[index, "Return on Equity"] = income_stmt_df.loc[index, "Net Income Common Stockholders"] / balancesheet_df.loc[index, "Total Equity Gross Minority Interest"]
                    balancesheet_df.loc[index, "Return on Invested Capital"] = income_stmt_df.loc[index, "Net Income Common Stockholders"] / (balancesheet_df.loc[index, "Total Equity Gross Minority Interest"] + balancesheet_df.loc[index, "Total Liabilities Net Minority Interest"])
                    balancesheet_df.loc[index, "Current Ratio"] = (balancesheet_df.loc[index, "Current Assets"]) / balancesheet_df.loc[index, "Current Liabilities"]
                    balancesheet_df.loc[index, "Quick Ratio"] = balancesheet_df.loc[index, "Cash And Cash Equivalents"] / balancesheet_df.loc[index, "Current Liabilities"]
                    balancesheet_df.loc[index, "Debt to Equity"] = balancesheet_df.loc[index, "Total Liabilities Net Minority Interest"] / balancesheet_df.loc[index, "Total Equity Gross Minority Interest"]
                    if index == 0:
                        balancesheet_df.loc[index, "Total Assets growth"] = 0.0
                        balancesheet_df.loc[index, "Current Assets growth"] = 0.0
                        balancesheet_df.loc[index, "Cash and Cash Equivalents growth"] = 0.0
                        balancesheet_df.loc[index, "Total Liabilities growth"] = 0.0
                        balancesheet_df.loc[index, "Total Equity growth"] = 0.0
                        balancesheet_df.loc[index, "Current Liabilities growth"] = 0.0
                        balancesheet_df.loc[index, "Book Value growth"] = 0.0
                        balancesheet_df.loc[index, "Book Value per share growth"] = 0.0
                        balancesheet_df.loc[index, "Return on Assets growth"] = 0.0
                        balancesheet_df.loc[index, "Return on Equity growth"] = 0.0
                        balancesheet_df.loc[index, "Return on Invested Capital growth"] = 0.0
                        balancesheet_df.loc[index, "Current Ratio growth"] = 0.0
                        balancesheet_df.loc[index, "Quick Ratio growth"] = 0.0
                        balancesheet_df.loc[index, "Debt to Equity growth"] = 0.0
                    else:
                        balancesheet_df.loc[index, "Total Assets growth"] = (balancesheet_df.iloc[index]["Total Assets"] / balancesheet_df.iloc[index-1]["Total Assets"])-1
                        balancesheet_df.loc[index, "Current Assets growth"] = (balancesheet_df.iloc[index]["Current Assets"] / balancesheet_df.iloc[index-1]["Current Assets"])-1
                        balancesheet_df.loc[index, "Cash and Cash Equivalents growth"] = (balancesheet_df.iloc[index]["Cash And Cash Equivalents"] / balancesheet_df.iloc[index-1]["Cash And Cash Equivalents"])-1
                        balancesheet_df.loc[index, "Total Liabilities growth"] = (balancesheet_df.iloc[index]["Total Liabilities Net Minority Interest"] / balancesheet_df.iloc[index-1]["Total Liabilities Net Minority Interest"])-1
                        balancesheet_df.loc[index, "Total Equity growth"] = (balancesheet_df.iloc[index]["Total Equity Gross Minority Interest"] / balancesheet_df.iloc[index-1]["Total Equity Gross Minority Interest"])-1
                        balancesheet_df.loc[index, "Current Liabilities growth"] = (balancesheet_df.iloc[index]["Current Liabilities"] / balancesheet_df.iloc[index-1]["Current Liabilities"])-1
                        balancesheet_df.loc[index, "Book Value growth"] = (balancesheet_df.iloc[index]["Book Value"] / balancesheet_df.iloc[index-1]["Book Value"])-1
                        balancesheet_df.loc[index, "Book Value per share growth"] = (balancesheet_df.iloc[index]["Book Value per share"] / balancesheet_df.iloc[index-1]["Book Value per share"])-1
                        balancesheet_df.loc[index, "Return on Assets growth"] = (balancesheet_df.iloc[index]["Return on Assets"] / balancesheet_df.iloc[index-1]["Return on Assets"])-1
                        balancesheet_df.loc[index, "Return on Equity growth"] = (balancesheet_df.iloc[index]["Return on Equity"] / balancesheet_df.iloc[index-1]["Return on Equity"])-1
                        balancesheet_df.loc[index, "Return on Invested Capital growth"] = (balancesheet_df.iloc[index]["Return on Invested Capital"] / balancesheet_df.iloc[index-1]["Return on Invested Capital"])-1
                        balancesheet_df.loc[index, "Current Ratio growth"] = (balancesheet_df.iloc[index]["Current Ratio"] / balancesheet_df.iloc[index-1]["Current Ratio"])-1
                        balancesheet_df.loc[index, "Quick Ratio growth"] = (balancesheet_df.iloc[index]["Quick Ratio"] / balancesheet_df.iloc[index-1]["Quick Ratio"])-1
                        balancesheet_df.loc[index, "Debt to Equity growth"] = (balancesheet_df.iloc[index]["Debt to Equity"] / balancesheet_df.iloc[index-1]["Debt to Equity"])-1
            else:
                for index, row in balancesheet_df.iterrows():
                    balancesheet_df.loc[index, "Book Value"] = balancesheet_df.loc[index, "Total Equity Gross Minority Interest"]
                    balancesheet_df.loc[index, "Book Value per share"] = balancesheet_df.loc[index, "Total Equity Gross Minority Interest"] / income_stmt_df.loc[index, "Diluted Average Shares"]
                    balancesheet_df.loc[index, "Return on Assets"] = income_stmt_df.loc[index, "Net Income Common Stockholders"] / balancesheet_df.loc[index, "Total Assets"]
                    balancesheet_df.loc[index, "Return on Equity"] = income_stmt_df.loc[index, "Net Income Common Stockholders"] / balancesheet_df.loc[index, "Total Equity Gross Minority Interest"]
                    balancesheet_df.loc[index, "Current Ratio"] = balancesheet_df.loc[index, "Current Assets"] / balancesheet_df.loc[index, "Current Liabilities"]
                    balancesheet_df.loc[index, "Quick Ratio"] = balancesheet_df.loc[index, "Cash And Cash Equivalents"] / balancesheet_df.loc[index, "Current Liabilities"]
                    balancesheet_df.loc[index, "Debt to Equity"] = balancesheet_df.loc[index, "Total Liabilities Net Minority Interest"] / balancesheet_df.loc[index, "Total Equity Gross Minority Interest"]
                    if index == 0:
                        balancesheet_df.loc[index, "Total Assets growth"] = 0.0
                        balancesheet_df.loc[index, "Current Assets growth"] = 0.0
                        balancesheet_df.loc[index, "Cash and Cash Equivalents growth"] = 0.0
                        balancesheet_df.loc[index, "Total Liabilities growth"] = 0.0
                        balancesheet_df.loc[index, "Total Equity growth"] = 0.0
                        balancesheet_df.loc[index, "Current Liabilities growth"] = 0.0
                        balancesheet_df.loc[index, "Book Value growth"] = 0.0
                        balancesheet_df.loc[index, "Book Value per share growth"] = 0.0
                        balancesheet_df.loc[index, "Return on Assets growth"] = 0.0
                        balancesheet_df.loc[index, "Return on Equity growth"] = 0.0
                        balancesheet_df.loc[index, "Current Ratio growth"] = 0.0
                        balancesheet_df.loc[index, "Quick Ratio growth"] = 0.0
                        balancesheet_df.loc[index, "Debt to Equity growth"] = 0.0
                    else:
                        balancesheet_df.loc[index, "Total Assets growth"] = (balancesheet_df.iloc[index]["Total Assets"] / balancesheet_df.iloc[index-1]["Total Assets"])-1
                        balancesheet_df.loc[index, "Current Assets growth"] = (balancesheet_df.iloc[index]["Current Assets"] / balancesheet_df.iloc[index-1]["Current Assets"])-1
                        balancesheet_df.loc[index, "Cash and Cash Equivalents growth"] = (balancesheet_df.iloc[index]["Cash And Cash Equivalents"] / balancesheet_df.iloc[index-1]["Cash And Cash Equivalents"])-1
                        balancesheet_df.loc[index, "Total Liabilities growth"] = (balancesheet_df.iloc[index]["Total Liabilities Net Minority Interest"] / balancesheet_df.iloc[index-1]["Total Liabilities Net Minority Interest"])-1
                        balancesheet_df.loc[index, "Total Equity growth"] = (balancesheet_df.iloc[index]["Total Equity Gross Minority Interest"] / balancesheet_df.iloc[index-1]["Total Equity Gross Minority Interest"])-1
                        balancesheet_df.loc[index, "Current Liabilities growth"] = (balancesheet_df.iloc[index]["Current Liabilities"] / balancesheet_df.iloc[index-1]["Current Liabilities"])-1
                        balancesheet_df.loc[index, "Book Value growth"] = (balancesheet_df.iloc[index]["Book Value"] / balancesheet_df.iloc[index-1]["Book Value"])-1
                        balancesheet_df.loc[index, "Book Value per share growth"] = (balancesheet_df.iloc[index]["Book Value per share"] / balancesheet_df.iloc[index-1]["Book Value per share"])-1
                        balancesheet_df.loc[index, "Return on Assets growth"] = (balancesheet_df.iloc[index]["Return on Assets"] / balancesheet_df.iloc[index-1]["Return on Assets"])-1
                        balancesheet_df.loc[index, "Return on Equity growth"] = (balancesheet_df.iloc[index]["Return on Equity"] / balancesheet_df.iloc[index-1]["Return on Equity"])-1
                        balancesheet_df.loc[index, "Current Ratio growth"] = (balancesheet_df.iloc[index]["Current Ratio"] / balancesheet_df.iloc[index-1]["Current Ratio"])-1
                        balancesheet_df.loc[index, "Quick Ratio growth"] = (balancesheet_df.iloc[index]["Quick Ratio"] / balancesheet_df.iloc[index-1]["Quick Ratio"])-1
                        balancesheet_df.loc[index, "Debt to Equity growth"] = (balancesheet_df.iloc[index]["Debt to Equity"] / balancesheet_df.iloc[index-1]["Debt to Equity"])-1
        elif "insurance" in industry.lower():
            if "Current Assets" not in balancesheet_df.columns:
                balancesheet_df["Current Assets"] = 0.0
                balancesheet_df = balancesheet_df.rename(columns={"Derivative Product Liabilities": "Current Liabilities"})
                for index, row in balancesheet_df.iterrows():
                    if "Receivables" not in balancesheet_df.columns:
                        balancesheet_df.loc[index, "Current Assets"] = balancesheet_df.loc[index, "Cash And Cash Equivalents"] + balancesheet_df.loc[index, "Trading Securities"] + balancesheet_df.loc[index, "Financial Assets Designatedas Fair Value Through Profitor Loss Total"]
                    else:
                        balancesheet_df.loc[index, "Current Assets"] = balancesheet_df.loc[index, "Cash And Cash Equivalents"] + balancesheet_df.loc[index, "Receivables"] + balancesheet_df.loc[index, "Trading Securities"] + balancesheet_df.loc[index, "Financial Assets Designatedas Fair Value Through Profitor Loss Total"]

                    balancesheet_df.loc[index, "Book Value"] = balancesheet_df.loc[index, "Total Equity Gross Minority Interest"]
                    balancesheet_df.loc[index, "Book Value per share"] = balancesheet_df.loc[index, "Total Equity Gross Minority Interest"] / income_stmt_df.loc[index, "Diluted Average Shares"]
                    balancesheet_df.loc[index, "Return on Assets"] = income_stmt_df.loc[index, "Net Income Common Stockholders"] / balancesheet_df.loc[index, "Total Assets"]
                    balancesheet_df.loc[index, "Return on Equity"] = income_stmt_df.loc[index, "Net Income Common Stockholders"] / balancesheet_df.loc[index, "Total Equity Gross Minority Interest"]
                    balancesheet_df.loc[index, "Return on Invested Capital"] = income_stmt_df.loc[index, "Net Income Common Stockholders"] / (balancesheet_df.loc[index, "Total Equity Gross Minority Interest"] + balancesheet_df.loc[index, "Total Liabilities Net Minority Interest"])
                    balancesheet_df.loc[index, "Current Ratio"] = (balancesheet_df.loc[index, "Current Assets"]) / balancesheet_df.loc[index, "Current Liabilities"]
                    balancesheet_df.loc[index, "Quick Ratio"] = balancesheet_df.loc[index, "Cash And Cash Equivalents"] / balancesheet_df.loc[index, "Current Liabilities"]
                    balancesheet_df.loc[index, "Debt to Equity"] = balancesheet_df.loc[index, "Total Liabilities Net Minority Interest"] / balancesheet_df.loc[index, "Total Equity Gross Minority Interest"]
                    if index == 0:
                        balancesheet_df.loc[index, "Total Assets growth"] = 0.0
                        balancesheet_df.loc[index, "Current Assets growth"] = 0.0
                        balancesheet_df.loc[index, "Cash and Cash Equivalents growth"] = 0.0
                        balancesheet_df.loc[index, "Total Liabilities growth"] = 0.0
                        balancesheet_df.loc[index, "Total Equity growth"] = 0.0
                        balancesheet_df.loc[index, "Current Liabilities growth"] = 0.0
                        balancesheet_df.loc[index, "Book Value growth"] = 0.0
                        balancesheet_df.loc[index, "Book Value per share growth"] = 0.0
                        balancesheet_df.loc[index, "Return on Assets growth"] = 0.0
                        balancesheet_df.loc[index, "Return on Equity growth"] = 0.0
                        balancesheet_df.loc[index, "Return on Invested Capital growth"] = 0.0
                        balancesheet_df.loc[index, "Current Ratio growth"] = 0.0
                        balancesheet_df.loc[index, "Quick Ratio growth"] = 0.0
                        balancesheet_df.loc[index, "Debt to Equity growth"] = 0.0
                    else:
                        balancesheet_df.loc[index, "Total Assets growth"] = (balancesheet_df.iloc[index]["Total Assets"] / balancesheet_df.iloc[index-1]["Total Assets"])-1
                        balancesheet_df.loc[index, "Current Assets growth"] = (balancesheet_df.iloc[index]["Current Assets"] / balancesheet_df.iloc[index-1]["Current Assets"])-1
                        balancesheet_df.loc[index, "Cash and Cash Equivalents growth"] = (balancesheet_df.iloc[index]["Cash And Cash Equivalents"] / balancesheet_df.iloc[index-1]["Cash And Cash Equivalents"])-1
                        balancesheet_df.loc[index, "Total Liabilities growth"] = (balancesheet_df.iloc[index]["Total Liabilities Net Minority Interest"] / balancesheet_df.iloc[index-1]["Total Liabilities Net Minority Interest"])-1
                        balancesheet_df.loc[index, "Total Equity growth"] = (balancesheet_df.iloc[index]["Total Equity Gross Minority Interest"] / balancesheet_df.iloc[index-1]["Total Equity Gross Minority Interest"])-1
                        balancesheet_df.loc[index, "Current Liabilities growth"] = (balancesheet_df.iloc[index]["Current Liabilities"] / balancesheet_df.iloc[index-1]["Current Liabilities"])-1
                        balancesheet_df.loc[index, "Book Value growth"] = (balancesheet_df.iloc[index]["Book Value"] / balancesheet_df.iloc[index-1]["Book Value"])-1
                        balancesheet_df.loc[index, "Book Value per share growth"] = (balancesheet_df.iloc[index]["Book Value per share"] / balancesheet_df.iloc[index-1]["Book Value per share"])-1
                        balancesheet_df.loc[index, "Return on Assets growth"] = (balancesheet_df.iloc[index]["Return on Assets"] / balancesheet_df.iloc[index-1]["Return on Assets"])-1
                        balancesheet_df.loc[index, "Return on Equity growth"] = (balancesheet_df.iloc[index]["Return on Equity"] / balancesheet_df.iloc[index-1]["Return on Equity"])-1
                        balancesheet_df.loc[index, "Return on Invested Capital growth"] = (balancesheet_df.iloc[index]["Return on Invested Capital"] / balancesheet_df.iloc[index-1]["Return on Invested Capital"])-1
                        balancesheet_df.loc[index, "Current Ratio growth"] = (balancesheet_df.iloc[index]["Current Ratio"] / balancesheet_df.iloc[index-1]["Current Ratio"])-1
                        balancesheet_df.loc[index, "Quick Ratio growth"] = (balancesheet_df.iloc[index]["Quick Ratio"] / balancesheet_df.iloc[index-1]["Quick Ratio"])-1
                        balancesheet_df.loc[index, "Debt to Equity growth"] = (balancesheet_df.iloc[index]["Debt to Equity"] / balancesheet_df.iloc[index-1]["Debt to Equity"])-1
            else:
                for index, row in balancesheet_df.iterrows():
                    balancesheet_df.loc[index, "Book Value"] = balancesheet_df.loc[index, "Total Equity Gross Minority Interest"]
                    balancesheet_df.loc[index, "Book Value per share"] = balancesheet_df.loc[index, "Total Equity Gross Minority Interest"] / income_stmt_df.loc[index, "Diluted Average Shares"]
                    balancesheet_df.loc[index, "Return on Assets"] = income_stmt_df.loc[index, "Net Income Common Stockholders"] / balancesheet_df.loc[index, "Total Assets"]
                    balancesheet_df.loc[index, "Return on Equity"] = income_stmt_df.loc[index, "Net Income Common Stockholders"] / balancesheet_df.loc[index, "Total Equity Gross Minority Interest"]
                    balancesheet_df.loc[index, "Return on Invested Capital"] = income_stmt_df.loc[index, "Net Income Common Stockholders"] / (balancesheet_df.loc[index, "Total Equity Gross Minority Interest"] + balancesheet_df.loc[index, "Total Liabilities Net Minority Interest"])
                    balancesheet_df.loc[index, "Current Ratio"] = balancesheet_df.loc[index, "Current Assets"] / balancesheet_df.loc[index, "Current Liabilities"]
                    balancesheet_df.loc[index, "Quick Ratio"] = balancesheet_df.loc[index, "Cash Cash Equivalents And Short Term Investments"] / balancesheet_df.loc[index, "Current Liabilities"]
                    balancesheet_df.loc[index, "Debt to Equity"] = balancesheet_df.loc[index, "Total Liabilities Net Minority Interest"] / balancesheet_df.loc[index, "Total Equity Gross Minority Interest"]
                    if index == 0:
                        balancesheet_df.loc[index, "Total Assets growth"] = 0.0
                        balancesheet_df.loc[index, "Current Assets growth"] = 0.0
                        balancesheet_df.loc[index, "Cash and Cash Equivalents growth"] = 0.0
                        balancesheet_df.loc[index, "Total Liabilities growth"] = 0.0
                        balancesheet_df.loc[index, "Total Equity growth"] = 0.0
                        balancesheet_df.loc[index, "Current Liabilities growth"] = 0.0
                        balancesheet_df.loc[index, "Book Value growth"] = 0.0
                        balancesheet_df.loc[index, "Book Value per share growth"] = 0.0
                        balancesheet_df.loc[index, "Return on Assets growth"] = 0.0
                        balancesheet_df.loc[index, "Return on Equity growth"] = 0.0
                        balancesheet_df.loc[index, "Return on Invested Capital growth"] = 0.0
                        balancesheet_df.loc[index, "Current Ratio growth"] = 0.0
                        balancesheet_df.loc[index, "Quick Ratio growth"] = 0.0
                        balancesheet_df.loc[index, "Debt to Equity growth"] = 0.0
                    else:
                        balancesheet_df.loc[index, "Total Assets growth"] = (balancesheet_df.iloc[index]["Total Assets"] / balancesheet_df.iloc[index-1]["Total Assets"])-1
                        balancesheet_df.loc[index, "Current Assets growth"] = (balancesheet_df.iloc[index]["Current Assets"] / balancesheet_df.iloc[index-1]["Current Assets"])-1
                        balancesheet_df.loc[index, "Cash and Cash Equivalents growth"] = (balancesheet_df.iloc[index]["Cash Cash Equivalents And Short Term Investments"] / balancesheet_df.iloc[index-1]["Cash Cash Equivalents And Short Term Investments"])-1
                        balancesheet_df.loc[index, "Total Liabilities growth"] = (balancesheet_df.iloc[index]["Total Liabilities Net Minority Interest"] / balancesheet_df.iloc[index-1]["Total Liabilities Net Minority Interest"])-1
                        balancesheet_df.loc[index, "Total Equity growth"] = (balancesheet_df.iloc[index]["Total Equity Gross Minority Interest"] / balancesheet_df.iloc[index-1]["Total Equity Gross Minority Interest"])-1
                        balancesheet_df.loc[index, "Current Liabilities growth"] = (balancesheet_df.iloc[index]["Current Liabilities"] / balancesheet_df.iloc[index-1]["Current Liabilities"])-1
                        balancesheet_df.loc[index, "Book Value growth"] = (balancesheet_df.iloc[index]["Book Value"] / balancesheet_df.iloc[index-1]["Book Value"])-1
                        balancesheet_df.loc[index, "Book Value per share growth"] = (balancesheet_df.iloc[index]["Book Value per share"] / balancesheet_df.iloc[index-1]["Book Value per share"])-1
                        balancesheet_df.loc[index, "Return on Assets growth"] = (balancesheet_df.iloc[index]["Return on Assets"] / balancesheet_df.iloc[index-1]["Return on Assets"])-1
                        balancesheet_df.loc[index, "Return on Equity growth"] = (balancesheet_df.iloc[index]["Return on Equity"] / balancesheet_df.iloc[index-1]["Return on Equity"])-1
                        balancesheet_df.loc[index, "Return on Invested Capital growth"] = (balancesheet_df.iloc[index]["Return on Invested Capital"] / balancesheet_df.iloc[index-1]["Return on Invested Capital"])-1
                        balancesheet_df.loc[index, "Current Ratio growth"] = (balancesheet_df.iloc[index]["Current Ratio"] / balancesheet_df.iloc[index-1]["Current Ratio"])-1
                        balancesheet_df.loc[index, "Quick Ratio growth"] = (balancesheet_df.iloc[index]["Quick Ratio"] / balancesheet_df.iloc[index-1]["Quick Ratio"])-1
                        balancesheet_df.loc[index, "Debt to Equity growth"] = (balancesheet_df.iloc[index]["Debt to Equity"] / balancesheet_df.iloc[index-1]["Debt to Equity"])-1
        else:
            for index, row in balancesheet_df.iterrows():
                balancesheet_df.loc[index, "Book Value"] = balancesheet_df.loc[index, "Total Equity Gross Minority Interest"]
                balancesheet_df.loc[index, "Book Value per share"] = balancesheet_df.loc[index, "Total Equity Gross Minority Interest"] / income_stmt_df.loc[index, "Diluted Average Shares"]
                balancesheet_df.loc[index, "Return on Assets"] = income_stmt_df.loc[index, "Net Income Common Stockholders"] / balancesheet_df.loc[index, "Total Assets"]
                balancesheet_df.loc[index, "Return on Equity"] = income_stmt_df.loc[index, "Net Income Common Stockholders"] / balancesheet_df.loc[index, "Total Equity Gross Minority Interest"]
                balancesheet_df.loc[index, "Return on Invested Capital"] = income_stmt_df.loc[index, "Net Income Common Stockholders"] / (balancesheet_df.loc[index, "Total Equity Gross Minority Interest"] + balancesheet_df.loc[index, "Total Liabilities Net Minority Interest"])
                balancesheet_df.loc[index, "Current Ratio"] = balancesheet_df.loc[index, "Current Assets"] / balancesheet_df.loc[index, "Current Liabilities"]
                balancesheet_df.loc[index, "Quick Ratio"] = balancesheet_df.loc[index, "Cash Cash Equivalents And Short Term Investments"] / balancesheet_df.loc[index, "Current Liabilities"]
                balancesheet_df.loc[index, "Debt to Equity"] = balancesheet_df.loc[index, "Total Liabilities Net Minority Interest"] / balancesheet_df.loc[index, "Total Equity Gross Minority Interest"]
                if index == 0:
                    balancesheet_df.loc[index, "Total Assets growth"] = 0.0
                    balancesheet_df.loc[index, "Current Assets growth"] = 0.0
                    balancesheet_df.loc[index, "Cash and Cash Equivalents growth"] = 0.0
                    balancesheet_df.loc[index, "Total Equity growth"] = 0.0
                    balancesheet_df.loc[index, "Total Liabilities growth"] = 0.0
                    balancesheet_df.loc[index, "Current Liabilities growth"] = 0.0
                    balancesheet_df.loc[index, "Book Value growth"] = 0.0
                    balancesheet_df.loc[index, "Book Value per share growth"] = 0.0
                    balancesheet_df.loc[index, "Return on Assets growth"] = 0.0
                    balancesheet_df.loc[index, "Return on Equity growth"] = 0.0
                    balancesheet_df.loc[index, "Return on Invested Capital growth"] = 0.0
                    balancesheet_df.loc[index, "Current Ratio growth"] = 0.0
                    balancesheet_df.loc[index, "Quick Ratio growth"] = 0.0
                    balancesheet_df.loc[index, "Debt to Equity growth"] = 0.0
                else:
                    balancesheet_df.loc[index, "Total Assets growth"] = (balancesheet_df.iloc[index]["Total Assets"] / balancesheet_df.iloc[index-1]["Total Assets"])-1
                    balancesheet_df.loc[index, "Current Assets growth"] = (balancesheet_df.iloc[index]["Current Assets"] / balancesheet_df.iloc[index-1]["Current Assets"])-1
                    balancesheet_df.loc[index, "Cash and Cash Equivalents growth"] = (balancesheet_df.iloc[index]["Cash Cash Equivalents And Short Term Investments"] / balancesheet_df.iloc[index-1]["Cash Cash Equivalents And Short Term Investments"])-1
                    balancesheet_df.loc[index, "Total Liabilities growth"] = (balancesheet_df.iloc[index]["Total Liabilities Net Minority Interest"] / balancesheet_df.iloc[index-1]["Total Liabilities Net Minority Interest"])-1
                    balancesheet_df.loc[index, "Total Equity growth"] = (balancesheet_df.iloc[index]["Total Equity Gross Minority Interest"] / balancesheet_df.iloc[index-1]["Total Equity Gross Minority Interest"])-1
                    balancesheet_df.loc[index, "Current Liabilities growth"] = (balancesheet_df.iloc[index]["Current Liabilities"] / balancesheet_df.iloc[index-1]["Current Liabilities"])-1
                    balancesheet_df.loc[index, "Book Value growth"] = (balancesheet_df.iloc[index]["Book Value"] / balancesheet_df.iloc[index-1]["Book Value"])-1
                    balancesheet_df.loc[index, "Book Value per share growth"] = (balancesheet_df.iloc[index]["Book Value per share"] / balancesheet_df.iloc[index-1]["Book Value per share"])-1
                    balancesheet_df.loc[index, "Return on Assets growth"] = (balancesheet_df.iloc[index]["Return on Assets"] / balancesheet_df.iloc[index-1]["Return on Assets"])-1
                    balancesheet_df.loc[index, "Return on Equity growth"] = (balancesheet_df.iloc[index]["Return on Equity"] / balancesheet_df.iloc[index-1]["Return on Equity"])-1
                    balancesheet_df.loc[index, "Return on Invested Capital growth"] = (balancesheet_df.iloc[index]["Return on Invested Capital"] / balancesheet_df.iloc[index-1]["Return on Invested Capital"])-1
                    balancesheet_df.loc[index, "Current Ratio growth"] = (balancesheet_df.iloc[index]["Current Ratio"] / balancesheet_df.iloc[index-1]["Current Ratio"])-1
                    balancesheet_df.loc[index, "Quick Ratio growth"] = (balancesheet_df.iloc[index]["Quick Ratio"] / balancesheet_df.iloc[index-1]["Quick Ratio"])-1
                    balancesheet_df.loc[index, "Debt to Equity growth"] = (balancesheet_df.iloc[index]["Debt to Equity"] / balancesheet_df.iloc[index-1]["Debt to Equity"])-1

        cashflow = stock_data.cashflow
        cashflow_df = pd.DataFrame(cashflow)
        # Checking if the input DataFrame is empty
        if cashflow_df.empty:
            raise ValueError("Input DataFrame is empty.")

        # Rotate the cashflow_df dataframe
        cashflow_df = cashflow_df.transpose()
        cashflow_df["Free Cash Flow growth"] = 0.0
        cashflow_df["Free Cash Flow per share"] = 0.0
        cashflow_df["Free Cash Flow per share growth"] = 0.0
        # Invert the rows in cashflow_df dataframe
        cashflow_df = cashflow_df.iloc[::-1]
        # Reset the index of the cashflow_df dataframe
        cashflow_df = cashflow_df.reset_index()
        # Rename the index column to Date
        cashflow_df = cashflow_df.rename(columns={"index": "Date"})
        for index, row in cashflow_df.iterrows():
            cashflow_df.loc[index, "Free Cash Flow per share"] = (cashflow_df.loc[index, "Free Cash Flow"] / income_stmt_df.loc[index, "Diluted Average Shares"])
            if index == 0:
                cashflow_df.loc[index, "Free Cash Flow growth"] = 0.0
                cashflow_df.loc[index, "Free Cash Flow per share growth"] = 0.0
            else:
                cashflow_df.loc[index, "Free Cash Flow growth"] = (cashflow_df.iloc[index]["Free Cash Flow"] / cashflow_df.iloc[index-1]["Free Cash Flow"])-1
                cashflow_df.loc[index, "Free Cash Flow per share growth"] = (cashflow_df.iloc[index]["Free Cash Flow per share"] / cashflow_df.iloc[index-1]["Free Cash Flow per share"])-1

        # Join income_stmt_df, balancesheet_df and cashflow_df dataframes on the Date column
        full_stock_financial_data_df = pd.merge(income_stmt_df, balancesheet_df, on="Date")
        full_stock_financial_data_df = pd.merge(full_stock_financial_data_df, cashflow_df, on="Date")
        # Drop row 0 from full_stock_financial_data_df
        full_stock_financial_data_df["Ticker"] = symbol
        full_stock_financial_data_df = full_stock_financial_data_df.drop([0])
        full_stock_financial_data_df = full_stock_financial_data_df.reset_index(drop=True)
        if "banks" in industry.lower():
            if "Gross Profit" not in income_stmt_df.columns:
                if "Operating Income" not in income_stmt_df.columns:
                    full_stock_financial_data_df = full_stock_financial_data_df.rename(columns={
                        "Diluted Average Shares": "Amount of stocks", "Total Revenue": "Revenue",
                        "Cash And Cash Equivalents": "Cash and Cash Equivalents",
                        "Total Liabilities Net Minority Interest": "Total Liabilities",
                        "Total Equity Gross Minority Interest": "Total Equity"
                    })
                    full_stock_financial_data_df = full_stock_financial_data_df[[
                        "Ticker", "Date", "Amount of stocks", "Revenue", "Revenue growth", "Net Income", "Net Income growth",
                        "Net Income Margin", "Net Income Margin growth", "EPS", "EPS growth", "Total Assets", "Total Assets growth",
                        "Current Assets", "Current Assets growth", "Cash and Cash Equivalents", "Cash and Cash Equivalents growth",
                        "Total Liabilities", "Total Liabilities growth", "Total Equity", "Total Equity growth", "Current Liabilities",
                        "Current Liabilities growth", "Book Value", "Book Value growth", "Book Value per share",
                        "Book Value per share growth", "Return on Assets", "Return on Assets growth", "Return on Equity",
                        "Return on Equity growth", "Return on Invested Capital", "Return on Invested Capital growth", 
                        "Current Ratio", "Current Ratio growth", "Quick Ratio", "Quick Ratio growth", "Debt to Equity",
                        "Debt to Equity growth","Free Cash Flow", "Free Cash Flow growth", "Free Cash Flow per share",
                        "Free Cash Flow per share growth"
                    ]]
                else:
                    full_stock_financial_data_df = full_stock_financial_data_df.rename(columns={
                        "Diluted Average Shares": "Amount of stocks", "Total Revenue": "Revenue",
                        "Cash And Cash Equivalents": "Cash and Cash Equivalents",
                        "Total Liabilities Net Minority Interest": "Total Liabilities",
                        "Total Equity Gross Minority Interest": "Total Equity"
                    })
                    full_stock_financial_data_df = full_stock_financial_data_df[[
                        "Ticker", "Date", "Amount of stocks", "Revenue", "Revenue growth", "Operating Income", "Operating Earnings growth",
                        "Operating Margin", "Operating Margin growth", "Net Income", "Net Income growth", "Net Income Margin",
                        "Net Income Margin growth", "EPS", "EPS growth", "Total Assets", "Total Assets growth", "Current Assets",
                        "Current Assets growth", "Cash and Cash Equivalents", "Cash and Cash Equivalents growth", "Total Liabilities",
                        "Total Liabilities growth", "Total Equity", "Total Equity growth", "Current Liabilities", "Current Liabilities growth",
                        "Book Value", "Book Value growth", "Book Value per share", "Book Value per share growth", "Return on Assets",
                        "Return on Assets growth", "Return on Equity", "Return on Equity growth", "Return on Invested Capital",
                        "Return on Invested Capital growth", "Current Ratio", "Current Ratio growth", "Quick Ratio", "Quick Ratio growth",
                        "Debt to Equity", "Debt to Equity growth", "Free Cash Flow", "Free Cash Flow growth", "Free Cash Flow per share",
                        "Free Cash Flow per share growth"
                    ]]
        elif "insurance" in industry.lower():
            if "Gross Profit" not in income_stmt_df.columns:
                if "Operating Income" not in income_stmt_df.columns:
                    full_stock_financial_data_df = full_stock_financial_data_df.rename(columns={
                        "Diluted Average Shares": "Amount of stocks", "Total Revenue": "Revenue",
                        "Cash And Cash Equivalents": "Cash and Cash Equivalents",
                        "Total Liabilities Net Minority Interest": "Total Liabilities",
                        "Total Equity Gross Minority Interest": "Total Equity"
                    })
                    full_stock_financial_data_df = full_stock_financial_data_df[[
                        "Ticker", "Date", "Amount of stocks", "Revenue", "Revenue growth", "Net Income", "Net Income growth", "Net Income Margin",
                        "Net Income Margin growth", "EPS", "EPS growth", "Total Assets", "Total Assets growth", "Current Assets", "Current Assets growth",
                        "Cash and Cash Equivalents", "Cash and Cash Equivalents growth", "Total Liabilities", "Total Liabilities growth", "Total Equity",
                        "Total Equity growth", "Current Liabilities", "Current Liabilities growth", "Book Value", "Book Value growth", "Book Value per share",
                        "Book Value per share growth", "Return on Assets", "Return on Assets growth", "Return on Equity", "Return on Equity growth",
                        "Return on Invested Capital", "Return on Invested Capital growth", "Current Ratio", "Current Ratio growth", "Quick Ratio",
                        "Quick Ratio growth", "Debt to Equity", "Debt to Equity growth", "Free Cash Flow", "Free Cash Flow growth", "Free Cash Flow per share",
                        "Free Cash Flow per share growth"
                    ]]
                else:
                    full_stock_financial_data_df = full_stock_financial_data_df.rename(columns={
                        "Diluted Average Shares": "Amount of stocks", "Total Revenue": "Revenue",
                        "Cash And Cash Equivalents": "Cash and Cash Equivalents",
                        "Total Liabilities Net Minority Interest": "Total Liabilities",
                        "Total Equity Gross Minority Interest": "Total Equity"
                    })
                    full_stock_financial_data_df = full_stock_financial_data_df[[
                        "Ticker" ,"Date", "Amount of stocks", "Revenue", "Revenue growth", "Operating Income", "Operating Earnings growth", "Operating Margin",
                        "Operating Margin growth", "Net Income", "Net Income growth", "Net Income Margin", "Net Income Margin growth", "EPS", "EPS growth",
                        "Total Assets", "Total Assets growth", "Current Assets", "Current Assets growth", "Cash and Cash Equivalents",
                        "Cash and Cash Equivalents growth", "Total Liabilities", "Total Liabilities growth", "Total Equity", "Total Equity growth",
                        "Current Liabilities", "Current Liabilities growth", "Book Value", "Book Value growth", "Book Value per share", "Book Value per share growth",
                        "Return on Assets", "Return on Assets growth", "Return on Equity", "Return on Equity growth", "Return on Invested Capital",
                        "Return on Invested Capital growth", "Current Ratio", "Current Ratio growth", "Quick Ratio", "Quick Ratio growth", "Debt to Equity",
                        "Debt to Equity growth", "Free Cash Flow", "Free Cash Flow growth", "Free Cash Flow per share", "Free Cash Flow per share growth"
                    ]]
            else:
                full_stock_financial_data_df = full_stock_financial_data_df.rename(columns={
                    "Diluted Average Shares": "Amount of stocks", "Total Revenue": "Revenue",
                    "Cash Cash Equivalents And Short Term Investments": "Cash and Cash Equivalents",
                    "Total Liabilities Net Minority Interest": "Total Liabilities",
                    "Total Equity Gross Minority Interest": "Total Equity"
                })
                full_stock_financial_data_df = full_stock_financial_data_df[[
                    "Ticker", "Date", "Amount of stocks", "Revenue", "Revenue growth", "Gross Profit", "Gross Profit growth",
                    "Gross Margin", "Gross Margin growth", "Operating Earnings", "Operating Earnings growth", "Operating Margin",
                    "Operating Margin growth", "Net Income", "Net Income growth", "Net Income Margin", "Net Income Margin growth",
                    "EPS", "EPS growth", "Total Assets", "Total Assets growth", "Current Assets", "Current Assets growth",
                    "Cash and Cash Equivalents", "Cash and Cash Equivalents growth", "Total Liabilities",
                    "Total Liabilities growth", "Total Equity", "Total Equity growth", "Current Liabilities",
                    "Current Liabilities growth", "Book Value", "Book Value growth", "Book Value per share",
                    "Book Value per share growth", "Return on Assets", "Return on Assets growth", "Return on Equity",
                    "Return on Equity growth", "Return on Invested Capital", "Return on Invested Capital growth", "Current Ratio",
                    "Current Ratio growth", "Quick Ratio", "Quick Ratio growth", "Debt to Equity", "Debt to Equity growth",
                    "Free Cash Flow", "Free Cash Flow growth", "Free Cash Flow per share", "Free Cash Flow per share growth"
                ]]
        elif "biotechnology" in industry.lower():
            if "Gross Profit" not in income_stmt_df.columns:
                full_stock_financial_data_df = full_stock_financial_data_df.rename(columns={
                    "Diluted Average Shares": "Amount of stocks", "Total Revenue": "Revenue",
                    "Cash Cash Equivalents And Short Term Investments": "Cash and Cash Equivalents",
                    "Total Liabilities Net Minority Interest": "Total Liabilities",
                    "Total Equity Gross Minority Interest": "Total Equity"
                })
                full_stock_financial_data_df = full_stock_financial_data_df[[
                    "Ticker" ,"Date", "Amount of stocks", "Revenue", "Revenue growth", "Operating Earnings", "Operating Earnings growth",
                    "Operating Margin", "Operating Margin growth", "Net Income", "Net Income growth", "Net Income Margin",
                    "Net Income Margin growth", "EPS", "EPS growth", "Total Assets", "Total Assets growth", "Current Assets",
                    "Current Assets growth", "Cash and Cash Equivalents", "Cash and Cash Equivalents growth", "Total Liabilities",
                    "Total Liabilities growth", "Total Equity", "Total Equity growth", "Current Liabilities",
                    "Current Liabilities growth", "Book Value", "Book Value growth", "Book Value per share",
                    "Book Value per share growth", "Return on Assets", "Return on Assets growth", "Return on Equity",
                    "Return on Equity growth", "Return on Invested Capital", "Return on Invested Capital growth", "Current Ratio",
                    "Current Ratio growth", "Quick Ratio", "Quick Ratio growth", "Debt to Equity", "Debt to Equity growth",
                    "Free Cash Flow", "Free Cash Flow growth", "Free Cash Flow per share", "Free Cash Flow per share growth"
                ]]
            else:
                full_stock_financial_data_df = full_stock_financial_data_df.rename(columns={
                    "Diluted Average Shares": "Amount of stocks", "Total Revenue": "Revenue",
                    "Cash Cash Equivalents And Short Term Investments": "Cash and Cash Equivalents",
                    "Total Liabilities Net Minority Interest": "Total Liabilities",
                    "Total Equity Gross Minority Interest": "Total Equity"
                })
                full_stock_financial_data_df = full_stock_financial_data_df[[
                    "Ticker" ,"Date", "Amount of stocks", "Revenue", "Revenue growth", "Gross Profit", "Gross Profit growth",
                    "Gross Margin", "Gross Margin growth", "Operating Earnings", "Operating Earnings growth", "Operating Margin",
                    "Operating Margin growth", "Net Income", "Net Income growth", "Net Income Margin", "Net Income Margin growth",
                    "EPS", "EPS growth", "Total Assets", "Total Assets growth", "Current Assets", "Current Assets growth",
                    "Cash and Cash Equivalents", "Cash and Cash Equivalents growth", "Total Liabilities",
                    "Total Liabilities growth", "Total Equity", "Total Equity growth", "Current Liabilities",
                    "Current Liabilities growth", "Book Value", "Book Value growth", "Book Value per share",
                    "Book Value per share growth", "Return on Assets", "Return on Assets growth", "Return on Equity",
                    "Return on Equity growth", "Return on Invested Capital", "Return on Invested Capital growth", "Current Ratio",
                    "Current Ratio growth", "Quick Ratio", "Quick Ratio growth", "Debt to Equity", "Debt to Equity growth",
                    "Free Cash Flow", "Free Cash Flow growth", "Free Cash Flow per share", "Free Cash Flow per share growth"
                ]]
        else:
            full_stock_financial_data_df = full_stock_financial_data_df.rename(columns={
                "Diluted Average Shares": "Amount of stocks", "Total Revenue": "Revenue",
                "Cash Cash Equivalents And Short Term Investments": "Cash and Cash Equivalents",
                "Total Liabilities Net Minority Interest": "Total Liabilities",
                "Total Equity Gross Minority Interest": "Total Equity"
            })
            full_stock_financial_data_df = full_stock_financial_data_df[[
                "Ticker", "Date", "Amount of stocks", "Revenue", "Revenue growth", "Gross Profit", "Gross Profit growth",
                "Gross Margin", "Gross Margin growth", "Operating Earnings", "Operating Earnings growth", "Operating Margin",
                "Operating Margin growth", "Net Income", "Net Income growth", "Net Income Margin", "Net Income Margin growth",
                "EPS", "EPS growth", "Total Assets", "Total Assets growth", "Current Assets", "Current Assets growth",
                "Cash and Cash Equivalents", "Cash and Cash Equivalents growth", "Total Liabilities",
                "Total Liabilities growth", "Total Equity", "Total Equity growth", "Current Liabilities",
                "Current Liabilities growth", "Book Value", "Book Value growth", "Book Value per share",
                "Book Value per share growth", "Return on Assets", "Return on Assets growth", "Return on Equity",
                "Return on Equity growth", "Return on Invested Capital", "Return on Invested Capital growth", "Current Ratio",
                "Current Ratio growth", "Quick Ratio", "Quick Ratio growth", "Debt to Equity", "Debt to Equity growth",
                "Free Cash Flow", "Free Cash Flow growth", "Free Cash Flow per share", "Free Cash Flow per share growth"
            ]]

        full_stock_financial_data_df = full_stock_financial_data_df.rename(columns={"Date": "date",
            "Ticker": "ticker", "Amount of stocks": "average_shares", "Revenue": "revenue", "Revenue growth": "revenue_Growth",
            "Gross Profit": "gross_Profit", "Gross Profit growth": "gross_Profit_Growth", "Gross Margin": "gross_Margin",
            "Gross Margin growth": "gross_Margin_Growth", "Operating Earnings": "operating_Earning",
            "Operating Earnings growth": "operating_Earning_Growth", "Operating Margin": "operating_Earning_Margin",
            "Operating Margin growth": "operating_Earning_Margin_Growth", "Net Income": "net_Income", "Net Income growth": "net_Income_Growth",
            "Net Income Margin": "net_Income_Margin", "Net Income Margin growth": "net_Income_Margin_Growth", "EPS": "eps",
            "EPS growth": "eps_Growth", "Total Assets": "total_Assets", "Total Assets growth": "total_Assets_Growth",
            "Current Assets": "current_Assets", "Current Assets growth": "current_Assets_Growth", "Cash and Cash Equivalents": "cash_And_Cash_Equivalents",
            "Cash and Cash Equivalents growth": "cash_And_Cash_Equivalents_Growth", "Total Equity": "equity", "Total Equity growth": "equity_Growth",
            "Total Liabilities": "liabilities", "Total Liabilities growth": "liabilities_Growth", "Current Liabilities": "current_Liabilities",
            "Current Liabilities growth": "current_Liabilities_Growth", "Book Value": "book_Value", "Book Value growth": "book_Value_Growth",
            "Book Value per share": "book_Value_Per_Share", "Book Value per share growth": "book_Value_Per_Share_Growth",
            "Return on Assets": "return_On_Assets", "Return on Assets growth": "return_On_Assets_Growth", "Return on Equity": "return_On_Equity",
            "Return on Equity growth": "return_On_Equity_Growth", "Current Ratio": "current_Ratio", "Current Ratio growth": "current_Ratio_Growth",
            "Quick Ratio": "quick_Ratio", "Quick Ratio growth": "quick_Ratio_Growth", "Debt to Equity": "debt_To_Equity",
            "Debt to Equity growth": "debt_To_Equity_Growth", "Free Cash Flow": "free_Cash_Flow", "Free Cash Flow growth": "free_Cash_Flow_Growth",
            "Free Cash Flow per share": "free_Cash_Flow_Per_Share", "Free Cash Flow per share growth": "free_Cash_Flow_Per_Share_Growth"
            })
        return full_stock_financial_data_df

    except KeyError as e:
        raise KeyError(f"Stock symbol '{symbol}' is invalid or not found.") from e
# Create a function the combines dataframe from fetch_stock_price_data with full_stock_financial_data_df from fetch_stock_financial_data
def combine_stock_data(stock_price_data_df, full_stock_financial_data_df):
    """
    Combines stock data with financial stock data.

    The function combines stock data with financial stock data and returns a DataFrame with the combined data.

    Parameters:
    - stock_price_data_df (pd.DataFrame): A DataFrame with stock data.
    - full_stock_financial_data_df (pd.DataFrame): A DataFrame with financial stock data.

    Returns:
    - combined_stock_data_df (pd.DataFrame): A DataFrame with the combined data.

    Raises:
    - ValueError: If the stock_price_data_df parameter is empty or if the full_stock_financial_data_df parameter is empty.
    - ValueError: Failed to combine stock data and financial stock data.
    """
    # Checking if the stock_price_data_df parameter is empty
    if stock_price_data_df.empty:
        raise ValueError("No stock data provided.")

    # Checking if the full_stock_financial_data_df parameter is empty
    if full_stock_financial_data_df.empty:
        raise ValueError("No financial stock data provided.")

    try:
        # Create a list of column names to copy from full_stock_financial_data_df to stock_price_data_df
        column_names = full_stock_financial_data_df.columns[2:]
        # Create a copy of stock_price_data_df
        combined_stock_data_df = stock_price_data_df.copy()
        # Add columns from full_stock_financial_data_df to stock_price_data_df
        for year in range(len(full_stock_financial_data_df["Date"])):
            combined_stock_data_df.loc[combined_stock_data_df["Date"] >= full_stock_financial_data_df.iloc[year]["Date"], column_names] = full_stock_financial_data_df.iloc[year].values[2:]

        print("Stock data and financial stock data combined successfully.")
        return combined_stock_data_df

    except ValueError as e:
        raise ValueError(f"Error combining stock data: {e}") from e
# Create a function that calculates P/S, P/E, P/B and P/FCF ratios
def calculate_ratios(combined_stock_data_df):
    """
    Calculates P/S, P/E, P/B and P/FCF ratios.

    The function calculates the P/S, P/E, P/B and P/FCF ratios and adds them to the DataFrame.

    Parameters:
    - combined_stock_data_df (pd.DataFrame): A DataFrame with combined stock data.

    Returns:
    - combined_stock_data_df (pd.DataFrame): A DataFrame with the ratios added.

    Raises:
    - ValueError: If the combined_stock_data_df parameter is empty.
    """
    # Checking if the combined_stock_data_df parameter is empty
    if combined_stock_data_df.empty:
        raise ValueError("No combined stock data provided.")

    # Calculate the P/S ratio
    combined_stock_data_df["P/S"] = combined_stock_data_df["open_Price"] / (combined_stock_data_df["Revenue"] / combined_stock_data_df["Amount of stocks"])
    # Calculate the P/E ratio
    combined_stock_data_df["P/E"] = combined_stock_data_df["open_Price"] / combined_stock_data_df["EPS"]
    # Calculate the P/B ratio
    combined_stock_data_df["P/B"] = combined_stock_data_df["open_Price"] / combined_stock_data_df["Book Value per share"]
    # Calculate the P/FCF ratio
    combined_stock_data_df["P/FCF"] = combined_stock_data_df["open_Price"] / combined_stock_data_df["Free Cash Flow per share growth"]
    print("Ratios have been calculated successfully, and added to the dataframe.")
    combined_stock_data_df[["P/S", "P/E", "P/B", "P/FCF"]] = combined_stock_data_df[["P/S", "P/E", "P/B", "P/FCF"]].shift(1)
    return combined_stock_data_df
# Drop rows with NaN values in combined_stock_data_df
def drop_nan_values(combined_stock_data_df):
    """
    Drops rows with NaN values in the given DataFrame.

    Parameters:
    - combined_stock_data_df (pd.DataFrame): A DataFrame with combined stock data.

    Returns:
    - combined_stock_data_df (pd.DataFrame): A DataFrame with the NaN values dropped.

    Raises:
    - ValueError: If the combined_stock_data_df parameter is empty.
    """
    # Checking if the combined_stock_data_df parameter is empty
    if combined_stock_data_df.empty:
        raise ValueError("No combined stock data provided.")

    # Drop rows with NaN values in combined_stock_data_df
    combined_stock_data_df = combined_stock_data_df.dropna()
    combined_stock_data_df = combined_stock_data_df.reset_index(drop=True)
    return combined_stock_data_df
# Create a function that exports the dataframes to an Excel file
def export_to_excel(dataframes, excel_file):
    """
    Exports the given dataframes to an Excel file with separate sheets.

    Each dataframe will be exported to a separate sheet in the Excel file.

    Parameters:
    - dataframes (dict): A dictionary of dataframes, where the keys are the sheet names and the values are the dataframes.
    - excel_file (str): The path to the Excel file to export.

    Raises:
    - ValueError: If the dataframes parameter is empty or if the excel_file parameter is empty.
    """
    try:
        # Check if the dataframes parameter is empty
        if not dataframes:
            raise ValueError("No dataframes provided.")

        # Check if the excel_file parameter is empty
        if not excel_file:
            raise ValueError("No Excel file path provided.")

        # Create a Pandas Excel writer object
        my_path = os.path.abspath(__file__)
        path = os.path.dirname(my_path)
        export_location = os.path.join(path, excel_file)
        writer = pd.ExcelWriter(export_location, engine='xlsxwriter')

        # Export each dataframe to a separate sheet in the Excel file
        for sheet_name, dataframe in dataframes.items():
            dataframe.to_excel(writer, sheet_name=sheet_name, index=False)

        # Save the Excel file
        writer.close()
        print("Dataframes have been successfully exported as xlsx file.")


    except ValueError as e:
        raise ValueError(f"Error exporting to Excel: {e}") from e
# Import stock symbols from a xlsx file
def import_excel(excel_file):
    """
    Imports the data from the given Excel file.

    Parameters:
    - excel_file (str): The path to the Excel file to import.

    Returns:
    - A dictionary of dataframes, where the keys are the sheet names and the values are the dataframes.

    Raises:
    - ValueError: If the excel_file parameter is empty.
    """
    try:
        # Check if the excel_file parameter is empty
        if not excel_file:
            raise ValueError("No Excel file path provided.")

        # Create a dictionary of dataframes to return
        dataframes = {}

        # Create a Pandas Excel file reader object
        my_path = os.path.abspath(__file__)
        path = os.path.dirname(my_path)
        import_location = os.path.join(path, excel_file)
        reader = pd.ExcelFile(import_location)

        # Import each sheet in the Excel file into a separate dataframe
        for sheet_name in reader.sheet_names:
            dataframes[sheet_name] = pd.read_excel(reader, sheet_name=sheet_name)

        # Return the dictionary of dataframes
        return dataframes

    except ValueError as e:
        raise ValueError(f"Error importing from Excel: {e}") from e
# Convert the Excel file to a CSV file
def convert_excel_to_csv(dataframe, file_name):
    """
    Converts the given Excel file to a CSV file.

    Parameters:
    - dataframe (dataframe): The dataframe to convert.
    - file_name (str): The name of the CSV file to create.

    Raises:
    - ValueError: If the dataframe parameter is empty or if the file_name parameter is empty.
    """
    try:
        # Check if the dataframe parameter is empty
        if dataframe.empty:
            raise ValueError("No dataframe provided.")

        # Check if the file_name parameter is empty
        if not file_name:
            raise ValueError("No file name provided.")

        # Convert the Excel file to a CSV file
        my_path = os.path.abspath(__file__)
        path = os.path.dirname(my_path)
        export_location = os.path.join(path, f"{file_name}.csv")
        dataframe.to_csv(export_location, sep=',', encoding='utf-8', index=False)

    except ValueError as e:
        raise ValueError(f"Error converting to CSV: {e}") from e
# Run the main function
if __name__ == "__main__":
    start_time = time.time()
    # stock_tickers_df = import_tickers_from_csv("index_symbol_list_single_stock.csv")
    # stock_tickers_list = stock_tickers_df["Symbol"].tolist()
    # ticker = stock_tickers_list[0]
    # print(ticker)
    # stock_info_data_df = fetch_stock_standard_data(ticker)
    import db_interactions
    ticker_list = db_interactions.import_ticker_list()
    for ticker in ticker_list:
        print(ticker)
        if db_interactions.does_stock_exists_stock_price_data(ticker) is False:
            print("Stock does not exist")
            # Fetch stock data for the imported stock symbols
            stock_price_data_df = fetch_stock_price_data(ticker)
            stock_price_data_df = calculate_period_returns(stock_price_data_df)
            stock_price_data_df = calculate_moving_averages(stock_price_data_df)
            stock_price_data_df = calculate_standard_diviation_value(stock_price_data_df)
            stock_price_data_df = calculate_bollinger_bands(stock_price_data_df)
            stock_price_data_df = calculate_momentum(stock_price_data_df)
            stock_price_data_df = stock_price_data_df.dropna()
            db_interactions.export_to_stock_price_data(stock_price_data_df)
        elif db_interactions.does_stock_exists_stock_price_data(ticker) is True:
            print("Stock exists")
            stock_price_data_df = db_interactions.import_from_stock_price_data()
            date = stock_price_data_df.iloc[0]["date"]
            if str(date) == datetime.datetime.now().strftime("%Y-%m-%d"):
                print(f"Today's price data has already been fetched for {ticker}")
            else:
                new_date = date + relativedelta(days=1)
                if new_date.weekday() == 5:
                    new_date = new_date + datetime.timedelta(days=1)
                elif new_date.weekday() == 6:
                    new_date = new_date + datetime.timedelta(days=2)

                stock_price_data_df = db_interactions.import_from_stock_price_data(252*5+1)
                stock_price_data_df["date"] = pd.to_datetime(stock_price_data_df["date"])
                print(f"New date is: {new_date}")
                new_stock_price_data_df = fetch_stock_price_data(ticker, new_date)
                stock_price_data_df = pd.concat([stock_price_data_df, new_stock_price_data_df], axis=0, ignore_index=True)
                stock_price_data_df = calculate_period_returns(stock_price_data_df)
                stock_price_data_df = calculate_moving_averages(stock_price_data_df)
                stock_price_data_df = calculate_standard_diviation_value(stock_price_data_df)
                stock_price_data_df = calculate_bollinger_bands(stock_price_data_df)
                stock_price_data_df = calculate_momentum(stock_price_data_df)
                stock_price_data_df = stock_price_data_df.loc[stock_price_data_df["date"] >= new_stock_price_data_df.loc[0, "date"]]
                db_interactions.export_to_stock_price_data(stock_price_data_df)

        if db_interactions.does_stock_exists_stock_income_stmt_data(ticker) is False:
            print("Stock does not exist")
            # Fetch stock financial data for the imported stock symbols
            full_stock_financial_data_df = fetch_stock_financial_data(ticker)
            # Export the stock financial data to the database
            db_interactions.export_to_stock_financial_data(full_stock_financial_data_df)
        elif db_interactions.does_stock_exists_stock_income_stmt_data(ticker) is True:
            print("Financial stock data exists")
            # Import the latest stock financial data from the database for the spicific stock ticker
            full_stock_financial_data_df = db_interactions.import_stock_financial_data(stock_ticker=ticker)
            db_date = full_stock_financial_data_df.iloc[0]["date"]
            full_stock_financial_data_df = fetch_stock_financial_data(ticker)
            source_date = full_stock_financial_data_df["date"].dt.date.iloc[-1]
            if db_date == source_date:
                print(f"The lastest financial data has already been fetched for {ticker}")
            else:
                # Fetch stock financial data for the imported stock symbols
                full_stock_financial_data_df = fetch_stock_financial_data(ticker)
                full_stock_financial_data_df = full_stock_financial_data_df.loc[full_stock_financial_data_df["date"].dt.date > db_date]
                # Export the stock financial data to the database
                db_interactions.export_to_stock_financial_data(full_stock_financial_data_df)


    # # Combine stock data with stock financial data
    # combined_stock_data_df = combine_stock_data(stock_price_data_df, full_stock_financial_data_df)
    # # print(combined_stock_data_df)
    # # Calculate ratios
    # combined_stock_data_df = calculate_ratios(combined_stock_data_df)
    # combined_stock_data_df = drop_nan_values(combined_stock_data_df)
    # # Create a dictionary of dataframes to export to Excel
    # dataframes = {
    #     "Stock Price Data": stock_price_data_df,
    #     "Full Stock Financial Data": full_stock_financial_data_df,
    #     "Combined Stock Data": combined_stock_data_df
    # }
    # # Export the dataframes to an Excel file
    # export_to_excel(dataframes, "stock_data_single.xlsx")
    # # Import the stock data from an Excel file
    # dataframes = import_excel("stock_data_single.xlsx")
    # for key, value in dataframes.items():
    #     dataframe = value

    # # Export the stock data to a CSV file
    # convert_excel_to_csv(dataframe, "stock_data_single")

    # Calculate the execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds to build dataset.")
