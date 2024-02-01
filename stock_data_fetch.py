import pandas as pd
import yfinance as yf
import time
import datetime
from dateutil.relativedelta import relativedelta

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
    
# Import stock data using yfinance and a list of stock symbols
def fetch_stock_price_data(stock_symbol):
    """
    Fetches stock data using yfinance for the given stock symbols and returns a pandas DataFrame.

    The DataFrame will contain the date, stock name, stock symbol, opening price, currency, and trade volume.

    Parameters:
    - stock_symbols (list): A list of stock symbols.

    Returns:
    pandas.DataFrame: A DataFrame containing the fetched stock data.

    Raises:
    - ValueError: If the stock_symbols parameter is empty.
    - KeyError: If any of the stock symbols are invalid or not found.
    """

    # Check if the stock_symbols parameter is empty
    if len(stock_symbol) == "":
        raise ValueError("The stock_symbols parameter cannot be empty.")

    # Create a DataFrame to store the fetched stock data
    stock_data_df = pd.DataFrame(columns=[
            "Date", "Name", "Ticker", "Price", "Currency", "Trade volume"
        ])
    try:
        symbol = stock_symbol
        # Fetch the stock data for the symbol
        stock_price_data = yf.download(
            symbol, period="8y"
        )
        # Reset the index of the DataFrame
        stock_price_data_df = pd.DataFrame(
            stock_price_data
        )
        # Reset the index of the DataFrame
        stock_price_data_df = stock_price_data_df.reset_index()
        stock_price_data_df = stock_price_data_df[[
            "Date", "Open", "Volume"
        ]]
        # Rename the columns
        stock_price_data_df = stock_price_data_df.rename(
            columns={
                "Open": "Price", "Volume": "Trade volume"
        })
        # Fetch the stock data for the symbol
        stock_info = yf.Ticker(symbol).info
        stock_info = {
            "Name": stock_info["longName"],
            "Ticker": stock_info["symbol"],
            "Currency": stock_info["currency"]
        }
        # Create a DataFrame with the stock data
        stock_info_df = pd.DataFrame(
            stock_info,
            index=[0]
        )
        # Create a temporary DataFrame with the stock data joined with the stock_price_data_df and stock_info_df
        temp_stock_data_df = stock_price_data_df.join(
            stock_info_df,
            how="cross"
        )
        # Create a new columns in temp_stock_data_df called 1M, 3M, 6M, 9M, 1Y, 2Y, 3Y, 4Y, and 5Y
        temp_stock_data_df["1M"] = 0.0
        temp_stock_data_df["3M"] = 0.0
        temp_stock_data_df["6M"] = 0.0
        temp_stock_data_df["9M"] = 0.0
        temp_stock_data_df["1Y"] = 0.0
        temp_stock_data_df["2Y"] = 0.0
        temp_stock_data_df["3Y"] = 0.0
        temp_stock_data_df["4Y"] = 0.0
        temp_stock_data_df["5Y"] = 0.0
        # Loop through each row in temp_stock_data_df
        for index, row in temp_stock_data_df.iterrows():
            # Calculate the date 1 month ago
            date = temp_stock_data_df["Date"].loc[index]
            date_1_month_ago = date - relativedelta(months=1)
            if date_1_month_ago.weekday() == 5:
                date_1_month_ago = date_1_month_ago - datetime.timedelta(days=1)


            if date_1_month_ago.weekday() == 6:
                date_1_month_ago = date_1_month_ago - datetime.timedelta(days=2)


            date_1_month_ago = date_1_month_ago.strftime("%Y-%m-%d")
            date_1_month_ago = temp_stock_data_df["Date"].loc[temp_stock_data_df["Date"] <= date_1_month_ago]
            if date_1_month_ago.empty:
                date_1_month_ago = None
            else:
                date_1_month_ago = date_1_month_ago.values[-1]
                # Format the date with dtype datetime64[ns]
                date_1_month_ago = pd.to_datetime(date_1_month_ago)


            # Calculate the 1 month change
            if date_1_month_ago != None:
                one_month_change = (((temp_stock_data_df.loc[index, "Price"] / temp_stock_data_df.loc[temp_stock_data_df["Date"] == date_1_month_ago, "Price"])-1))
                one_month_change = one_month_change.values[-1]
            else:
                one_month_change = (((temp_stock_data_df.loc[index, "Price"] / temp_stock_data_df.loc[0, "Price"])-1))


            # Update the 1M Change column with the calculated value
            temp_stock_data_df.loc[index, "1M"] = one_month_change
            # Calculate the date 3 months ago
            date = temp_stock_data_df["Date"].loc[index]
            date_3_months_ago = date - relativedelta(months=3)
            if date_3_months_ago.weekday() == 5:
                date_3_months_ago = date_3_months_ago - datetime.timedelta(days=1)


            if date_3_months_ago.weekday() == 6:
                date_3_months_ago = date_3_months_ago - datetime.timedelta(days=2)


            date_3_months_ago = date_3_months_ago.strftime("%Y-%m-%d")
            date_3_months_ago = temp_stock_data_df["Date"].loc[temp_stock_data_df["Date"] <= date_3_months_ago]
            if date_3_months_ago.empty:
                date_3_months_ago = None
            else:
                date_3_months_ago = date_3_months_ago.values[-1]
                # Format the date with dtype datetime64[ns]
                date_3_months_ago = pd.to_datetime(date_3_months_ago)


            # Calculate the 3 month change
            if date_3_months_ago != None:
                three_month_change = (((temp_stock_data_df["Price"].loc[index]/temp_stock_data_df["Price"].loc[temp_stock_data_df["Date"] == date_3_months_ago])-1))
                three_month_change = three_month_change.values[-1]
            else:
                three_month_change = (((temp_stock_data_df["Price"].loc[index]/temp_stock_data_df["Price"].iloc[0])-1))


            # Update the 3M Change column with the calculated value
            temp_stock_data_df.loc[index, "3M"] = three_month_change
            # Calculate the date 6 months ago
            date = temp_stock_data_df["Date"].loc[index]
            date_6_months_ago = date - relativedelta(months=6)
            if date_6_months_ago.weekday() == 5:
                date_6_months_ago = date_6_months_ago - datetime.timedelta(days=1)


            if date_6_months_ago.weekday() == 6:
                date_6_months_ago = date_6_months_ago - datetime.timedelta(days=2)


            date_6_months_ago = date_6_months_ago.strftime("%Y-%m-%d")
            date_6_months_ago = temp_stock_data_df["Date"].loc[temp_stock_data_df["Date"] <= date_6_months_ago]
            if date_6_months_ago.empty:
                date_6_months_ago = None
            else:
                date_6_months_ago = date_6_months_ago.values[-1]
                # Format the date with dtype datetime64[ns]
                date_6_months_ago = pd.to_datetime(date_6_months_ago)


            # Calculate the 6 month change
            if date_6_months_ago != None:
                six_month_change = (((temp_stock_data_df["Price"].loc[index]/temp_stock_data_df["Price"].loc[temp_stock_data_df["Date"] == date_6_months_ago])-1))
                six_month_change = six_month_change.values[-1]
            else:
                six_month_change = (((temp_stock_data_df["Price"].loc[index]/temp_stock_data_df["Price"].iloc[0])-1))


            # Update the 6M Change column with the calculated value
            temp_stock_data_df.loc[index, "6M"] = six_month_change
            # Calculate the date 9 months ago
            date = temp_stock_data_df["Date"].loc[index]
            date_9_months_ago = date - relativedelta(months=9)
            if date_9_months_ago.weekday() == 5:
                date_9_months_ago = date_9_months_ago - datetime.timedelta(days=1)


            if date_9_months_ago.weekday() == 6:
                date_9_months_ago = date_9_months_ago - datetime.timedelta(days=2)


            date_9_months_ago = date_9_months_ago.strftime("%Y-%m-%d")
            date_9_months_ago = temp_stock_data_df["Date"].loc[temp_stock_data_df["Date"] <= date_9_months_ago]
            if date_9_months_ago.empty:
                date_9_months_ago = None
            else:
                date_9_months_ago = date_9_months_ago.values[-1]
                # Format the date with dtype datetime64[ns]
                date_9_months_ago = pd.to_datetime(date_9_months_ago)

            
            # Calculate the 9 month change
            if date_9_months_ago != None:
                nine_month_change = (((temp_stock_data_df["Price"].loc[index]/temp_stock_data_df["Price"].loc[temp_stock_data_df["Date"] == date_9_months_ago])-1))
                nine_month_change = nine_month_change.values[-1]
            else:
                nine_month_change = (((temp_stock_data_df["Price"].loc[index]/temp_stock_data_df["Price"].iloc[0])-1))


            # Update the 9M Change column with the calculated value
            temp_stock_data_df.loc[index, "9M"] = nine_month_change
            # Calculate the date 1 year ago
            date = temp_stock_data_df["Date"].loc[index]
            date_1_year_ago = date - relativedelta(years=1)
            if date_1_year_ago.weekday() == 5:
                date_1_year_ago = date_1_year_ago - datetime.timedelta(days=1)


            if date_1_year_ago.weekday() == 6:
                date_1_year_ago = date_1_year_ago - datetime.timedelta(days=2)


            date_1_year_ago = date_1_year_ago.strftime("%Y-%m-%d")
            date_1_year_ago = temp_stock_data_df["Date"].loc[temp_stock_data_df["Date"] <= date_1_year_ago]
            if date_1_year_ago.empty:
                date_1_year_ago = None
            else:
                date_1_year_ago = date_1_year_ago.values[-1]
                # Format the date with dtype datetime64[ns]
                date_1_year_ago = pd.to_datetime(date_1_year_ago)


            # Calculate the 1 year change
            if date_1_year_ago != None:
                one_year_change = (((temp_stock_data_df["Price"].loc[index]/temp_stock_data_df["Price"].loc[temp_stock_data_df["Date"] == date_1_year_ago])-1))
                one_year_change = one_year_change.values[-1]
            else:
                one_year_change = (((temp_stock_data_df["Price"].loc[index]/temp_stock_data_df["Price"].iloc[0])-1))


            # Update the 1Y Change column with the calculated value
            temp_stock_data_df.loc[index, "1Y"] = one_year_change
            # Calculate the date 2 years ago
            date = temp_stock_data_df["Date"].loc[index]
            date_2_years_ago = date - relativedelta(years=2)
            if date_2_years_ago.weekday() == 5:
                date_2_years_ago = date_2_years_ago - datetime.timedelta(days=1)


            if date_2_years_ago.weekday() == 6:
                date_2_years_ago = date_2_years_ago - datetime.timedelta(days=2)


            date_2_years_ago = date_2_years_ago.strftime("%Y-%m-%d")
            date_2_years_ago = temp_stock_data_df["Date"].loc[temp_stock_data_df["Date"] <= date_2_years_ago]
            if date_2_years_ago.empty:
                date_2_years_ago = None
            else:
                date_2_years_ago = date_2_years_ago.values[-1]
                # Format the date with dtype datetime64[ns]
                date_2_years_ago = pd.to_datetime(date_2_years_ago)


            # Calculate the 2 year change
            if date_2_years_ago != None:
                two_year_change = (((temp_stock_data_df["Price"].loc[index]/temp_stock_data_df["Price"].loc[temp_stock_data_df["Date"] == date_2_years_ago])-1))
                two_year_change = two_year_change.values[-1]
            else:
                two_year_change = (((temp_stock_data_df["Price"].loc[index]/temp_stock_data_df["Price"].iloc[0])-1))


            # Update the 2Y Change column with the calculated value
            temp_stock_data_df.loc[index, "2Y"] = two_year_change
            # Calculate the date 3 years ago
            date = temp_stock_data_df["Date"].loc[index]
            date_3_years_ago = date - relativedelta(years=3)
            if date_3_years_ago.weekday() == 5:
                date_3_years_ago = date_3_years_ago - datetime.timedelta(days=1)


            if date_3_years_ago.weekday() == 6:
                date_3_years_ago = date_3_years_ago - datetime.timedelta(days=2)


            date_3_years_ago = date_3_years_ago.strftime("%Y-%m-%d")
            date_3_years_ago = temp_stock_data_df["Date"].loc[temp_stock_data_df["Date"] <= date_3_years_ago]
            if date_3_years_ago.empty:
                date_3_years_ago = None
            else:
                date_3_years_ago = date_3_years_ago.values[-1]
                # Format the date with dtype datetime64[ns]
                date_3_years_ago = pd.to_datetime(date_3_years_ago)


            # Calculate the 3 year change
            if date_3_years_ago != None:
                three_year_change = (((temp_stock_data_df["Price"].loc[index]/temp_stock_data_df["Price"].loc[temp_stock_data_df["Date"] == date_3_years_ago])-1))
                three_year_change = three_year_change.values[-1]
            else:
                three_year_change = (((temp_stock_data_df["Price"].loc[index]/temp_stock_data_df["Price"].iloc[0])-1))


            # Update the 3Y Change column with the calculated value
            temp_stock_data_df.loc[index, "3Y"] = three_year_change
            # Calculate the date 4 years ago
            date = temp_stock_data_df["Date"].loc[index]
            date_4_years_ago = date - relativedelta(years=4)
            if date_4_years_ago.weekday() == 5:
                date_4_years_ago = date_4_years_ago - datetime.timedelta(days=1)


            if date_4_years_ago.weekday() == 6:
                date_4_years_ago = date_4_years_ago - datetime.timedelta(days=2)


            date_4_years_ago = date_4_years_ago.strftime("%Y-%m-%d")
            date_4_years_ago = temp_stock_data_df["Date"].loc[temp_stock_data_df["Date"] <= date_4_years_ago]
            if date_4_years_ago.empty:
                date_4_years_ago = None
            else:
                date_4_years_ago = date_4_years_ago.values[-1]
                # Format the date with dtype datetime64[ns]
                date_4_years_ago = pd.to_datetime(date_4_years_ago)


            # Calculate the 4 year change
            if date_4_years_ago != None:
                four_year_change = (((temp_stock_data_df["Price"].loc[index]/temp_stock_data_df["Price"].loc[temp_stock_data_df["Date"] == date_4_years_ago])-1))
                four_year_change = four_year_change.values[-1]
            else:
                four_year_change = (((temp_stock_data_df["Price"].loc[index]/temp_stock_data_df["Price"].iloc[0])-1))


            # Update the 4Y Change column with the calculated value
            temp_stock_data_df.loc[index, "4Y"] = four_year_change
            # Calculate the date 5 years ago
            date = temp_stock_data_df["Date"].loc[index]
            date_5_years_ago = date - relativedelta(years=5)
            if date_5_years_ago.weekday() == 5:
                date_5_years_ago = date_5_years_ago - datetime.timedelta(days=1)


            if date_5_years_ago.weekday() == 6:
                date_5_years_ago = date_5_years_ago - datetime.timedelta(days=2)


            date_5_years_ago = date_5_years_ago.strftime("%Y-%m-%d")
            date_5_years_ago = temp_stock_data_df["Date"].loc[temp_stock_data_df["Date"] <= date_5_years_ago]
            if date_5_years_ago.empty:
                date_5_years_ago = None
            else:
                date_5_years_ago = date_5_years_ago.values[-1]
                # Format the date with dtype datetime64[ns]
                date_5_years_ago = pd.to_datetime(date_5_years_ago)


            # Calculate the 5 year change
            if date_5_years_ago != None:
                five_year_change = (((temp_stock_data_df["Price"].loc[index]/temp_stock_data_df["Price"].loc[temp_stock_data_df["Date"] == date_5_years_ago])-1))
                five_year_change = five_year_change.values[-1]
            else:
                five_year_change = (((temp_stock_data_df["Price"].loc[index]/temp_stock_data_df["Price"].iloc[0])-1))


            # Update the 5Y Change column with the calculated value
            temp_stock_data_df.loc[index, "5Y"] = five_year_change
            # Create print statement per 100 index processed
            if index % 250 == 0:
                print(f"Processed {index} rows, out of {len(temp_stock_data_df)} rows.")


        # Create a new columns in temp_stock_data_df called SMA_40, SMA_120, EMA_40, and EMA_120
        temp_stock_data_df["SMA_40"] = 0.0
        temp_stock_data_df["SMA_120"] = 0.0
        temp_stock_data_df["EMA_40"] = 0.0
        temp_stock_data_df["EMA_120"] = 0.0
        # Loop through each row in temp_stock_data_df
        for index, row in temp_stock_data_df.iterrows():
            # Calculate SMA_40 for every row
            if index >= 40:
                sma_40 = temp_stock_data_df.iloc[index-39:index+1]["Price"].mean()
            else:
                sma_40 = temp_stock_data_df.iloc[0:index]["Price"].mean()


            # Update the SMA_40 column with the calculated value
            temp_stock_data_df.loc[index, "SMA_40"] = sma_40
            # Calculate SMA_120 for every row
            if index >= 120:
                sma_120 = temp_stock_data_df.iloc[index-119:index+1]["Price"].mean()
            else:
                sma_120 = temp_stock_data_df.iloc[0:index+1]["Price"].mean()


            # Update the SMA_120 column with the calculated value
            temp_stock_data_df.loc[index, "SMA_120"] = sma_120
            # Calculate EMA_40 for every row
            if index >= 40:
                ema_40 = temp_stock_data_df.iloc[index-39:index+1]["Price"].ewm(span=40).mean()
                ema_40 = ema_40.values[-1]
            else:
                ema_40 = temp_stock_data_df.iloc[0:index+1]["Price"].ewm(span=40).mean()
                if ema_40.empty:
                    ema_40 = 0.0
                else:
                    ema_40 = ema_40.values[-1]


            # Update the EMA_40 column with the calculated value
            temp_stock_data_df.loc[index, "EMA_40"] = ema_40
            # Calculate EMA_120 for every row
            if index >= 120:
                ema_120 = temp_stock_data_df.iloc[index-119:index+1]["Price"].ewm(span=120).mean()
                ema_120 = ema_120.values[-1]
            else:
                ema_120 = temp_stock_data_df.iloc[0:index+1]["Price"].ewm(span=120).mean()
                if ema_120.empty:
                    ema_120 = 0.0
                else:
                    ema_120 = ema_120.values[-1]


            # Update the EMA_120 column with the calculated value
            temp_stock_data_df.loc[index, "EMA_120"] = ema_120
            # Create print statement per 100 index processed
            if index % 250 == 0:
                print(f"Processed {index} rows, out of {len(temp_stock_data_df)} rows.")


    except KeyError:
        raise KeyError(f"Stock symbol '{stock_symbol}' is invalid or not found.")

    # Return the DataFrame with stock data
    return temp_stock_data_df

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

    # Creating an empty DataFrame
    stock_data_df = pd.DataFrame()
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
        # Use stock_data to chack if bank is part of the registered industry for the stock
        stock_info = stock_data.info
        industry = stock_info["industry"]
        # Check if bank is part of the registered industry for the stock
        if "banks" in industry.lower():
            # Check is "Gross Profit" is in the income_stmt_df dataframe
            if "Gross Profit" not in income_stmt_df.columns:
                if "Operating Income" not in income_stmt_df.columns:
                    for index, row in income_stmt_df.iterrows():
                        income_stmt_df.loc[index, "Net Income Margin"] = (income_stmt_df.loc[index, "Net Income Common Stockholders"] / income_stmt_df.loc[index, "Total Revenue"])
                        income_stmt_df.loc[index, "EPS"] = (income_stmt_df.loc[index, "Net Income Common Stockholders"] / income_stmt_df.loc[index, "Diluted Average Shares"])
                        if index == 0:
                            income_stmt_df.loc[index, "Revenue growth"] = 0.0
                            income_stmt_df.loc[index, "Net Income growth"] = 0.0
                            income_stmt_df.loc[index, "Net Income Margin growth"] = 0.0
                            income_stmt_df.loc[index, "EPS growth"] = 0.0
                        else:
                            income_stmt_df.loc[index, "Revenue growth"] = ((income_stmt_df.iloc[index]["Total Revenue"] / income_stmt_df.iloc[index-1]["Total Revenue"])-1)
                            income_stmt_df.loc[index, "Net Income growth"] = ((income_stmt_df.iloc[index]["Net Income Common Stockholders"] / income_stmt_df.iloc[index-1]["Net Income Common Stockholders"])-1)
                            income_stmt_df.loc[index, "Net Income Margin growth"] = ((income_stmt_df.iloc[index]["Net Income Margin"] / income_stmt_df.iloc[index-1]["Net Income Margin"])-1)
                            income_stmt_df.loc[index, "EPS growth"] = ((income_stmt_df.iloc[index]["EPS"] / income_stmt_df.iloc[index-1]["EPS"])-1)
                else:
                    for index, row in income_stmt_df.iterrows():
                        income_stmt_df.loc[index, "Operating Margin"] = (income_stmt_df.loc[index, "Operating Income"] / income_stmt_df.loc[index, "Total Revenue"])
                        income_stmt_df.loc[index, "Net Income Margin"] = (income_stmt_df.loc[index, "Net Income Common Stockholders"] / income_stmt_df.loc[index, "Total Revenue"])
                        income_stmt_df.loc[index, "EPS"] = (income_stmt_df.loc[index, "Net Income Common Stockholders"] / income_stmt_df.loc[index, "Diluted Average Shares"])
                        if index == 0:
                            income_stmt_df.loc[index, "Revenue growth"] = 0.0
                            income_stmt_df.loc[index, "Operating Earnings growth"] = 0.0
                            income_stmt_df.loc[index, "Operating Margin growth"] = 0.0
                            income_stmt_df.loc[index, "Net Income growth"] = 0.0
                            income_stmt_df.loc[index, "Net Income Margin growth"] = 0.0
                            income_stmt_df.loc[index, "EPS growth"] = 0.0
                        else:
                            income_stmt_df.loc[index, "Revenue growth"] = ((income_stmt_df.iloc[index]["Total Revenue"] / income_stmt_df.iloc[index-1]["Total Revenue"])-1)
                            income_stmt_df.loc[index, "Operating Earnings growth"] = ((income_stmt_df.iloc[index]["Operating Income"] / income_stmt_df.iloc[index-1]["Operating Income"])-1)
                            income_stmt_df.loc[index, "Operating Margin growth"] = ((income_stmt_df.iloc[index]["Operating Margin"] / income_stmt_df.iloc[index-1]["Operating Margin"])-1)
                            income_stmt_df.loc[index, "Net Income growth"] = ((income_stmt_df.iloc[index]["Net Income Common Stockholders"] / income_stmt_df.iloc[index-1]["Net Income Common Stockholders"])-1)
                            income_stmt_df.loc[index, "Net Income Margin growth"] = ((income_stmt_df.iloc[index]["Net Income Margin"] / income_stmt_df.iloc[index-1]["Net Income Margin"])-1)
                            income_stmt_df.loc[index, "EPS growth"] = ((income_stmt_df.iloc[index]["EPS"] / income_stmt_df.iloc[index-1]["EPS"])-1)
            else:
                for index, row in income_stmt_df.iterrows():
                    income_stmt_df.loc[index, "Gross Margin"] = (income_stmt_df.loc[index, "Gross Profit"] / income_stmt_df.loc[index, "Total Revenue"])
                    income_stmt_df.loc[index, "Operating Margin"] = (income_stmt_df.loc[index, "Operating Income"] / income_stmt_df.loc[index, "Total Revenue"])
                    income_stmt_df.loc[index, "Net Income Margin"] = (income_stmt_df.loc[index, "Net Income Common Stockholders"] / income_stmt_df.loc[index, "Total Revenue"])
                    income_stmt_df.loc[index, "EPS"] = (income_stmt_df.loc[index, "Net Income Common Stockholders"] / income_stmt_df.loc[index, "Diluted Average Shares"])
                    if index == 0:
                        income_stmt_df.loc[index, "Revenue growth"] = 0.0
                        income_stmt_df.loc[index, "Gross Profit growth"] = 0.0
                        income_stmt_df.loc[index, "Gross Margin growth"] = 0.0
                        income_stmt_df.loc[index, "Operating Earnings growth"] = 0.0
                        income_stmt_df.loc[index, "Operating Margin growth"] = 0.0
                        income_stmt_df.loc[index, "Net Income growth"] = 0.0
                        income_stmt_df.loc[index, "Net Income Margin growth"] = 0.0
                    else:
                        income_stmt_df.loc[index, "Revenue growth"] = ((income_stmt_df.iloc[index]["Total Revenue"] / income_stmt_df.iloc[index-1]["Total Revenue"])-1)
                        income_stmt_df.loc[index, "Gross Profit growth"] = ((income_stmt_df.iloc[index]["Gross Profit"] / income_stmt_df.iloc[index-1]["Gross Profit"])-1)
                        income_stmt_df.loc[index, "Gross Margin growth"] = ((income_stmt_df.iloc[index]["Gross Margin"] / income_stmt_df.iloc[index-1]["Gross Margin"])-1)
                        income_stmt_df.loc[index, "Operating Earnings growth"] = ((income_stmt_df.iloc[index]["Operating Income"] / income_stmt_df.iloc[index-1]["Operating Income"])-1)
                        income_stmt_df.loc[index, "Operating Margin growth"] = ((income_stmt_df.iloc[index]["Operating Margin"] / income_stmt_df.iloc[index-1]["Operating Margin"])-1)
                        income_stmt_df.loc
        elif "insurance" in industry.lower():
            # Check is "Gross Profit" is in the income_stmt_df dataframe
            if "Gross Profit" not in income_stmt_df.columns:
                if "Operating Income" not in income_stmt_df.columns:
                    for index, row in income_stmt_df.iterrows():
                        income_stmt_df.loc[index, "Net Income Margin"] = (income_stmt_df.loc[index, "Net Income Common Stockholders"] / income_stmt_df.loc[index, "Total Revenue"])
                        income_stmt_df.loc[index, "EPS"] = (income_stmt_df.loc[index, "Net Income Common Stockholders"] / income_stmt_df.loc[index, "Diluted Average Shares"])
                        if index == 0:
                            income_stmt_df.loc[index, "Revenue growth"] = 0.0
                            income_stmt_df.loc[index, "Net Income growth"] = 0.0
                            income_stmt_df.loc[index, "Net Income Margin growth"] = 0.0
                            income_stmt_df.loc[index, "EPS growth"] = 0.0
                        else:
                            income_stmt_df.loc[index, "Revenue growth"] = ((income_stmt_df.iloc[index]["Total Revenue"] / income_stmt_df.iloc[index-1]["Total Revenue"])-1)
                            income_stmt_df.loc[index, "Net Income growth"] = ((income_stmt_df.iloc[index]["Net Income Common Stockholders"] / income_stmt_df.iloc[index-1]["Net Income Common Stockholders"])-1)
                            income_stmt_df.loc[index, "Net Income Margin growth"] = ((income_stmt_df.iloc[index]["Net Income Margin"] / income_stmt_df.iloc[index-1]["Net Income Margin"])-1)
                            income_stmt_df.loc[index, "EPS growth"] = ((income_stmt_df.iloc[index]["EPS"] / income_stmt_df.iloc[index-1]["EPS"])-1)
                else:
                    for index, row in income_stmt_df.iterrows():
                        income_stmt_df.loc[index, "Operating Margin"] = (income_stmt_df.loc[index, "Operating Income"] / income_stmt_df.loc[index, "Total Revenue"])
                        income_stmt_df.loc[index, "Net Income Margin"] = (income_stmt_df.loc[index, "Net Income Common Stockholders"] / income_stmt_df.loc[index, "Total Revenue"])
                        income_stmt_df.loc[index, "EPS"] = (income_stmt_df.loc[index, "Net Income Common Stockholders"] / income_stmt_df.loc[index, "Diluted Average Shares"])
                        if index == 0:
                            income_stmt_df.loc[index, "Revenue growth"] = 0.0
                            income_stmt_df.loc[index, "Operating Earnings growth"] = 0.0
                            income_stmt_df.loc[index, "Operating Margin growth"] = 0.0
                            income_stmt_df.loc[index, "Net Income growth"] = 0.0
                            income_stmt_df.loc[index, "Net Income Margin growth"] = 0.0
                            income_stmt_df.loc[index, "EPS growth"] = 0.0
                        else:
                            income_stmt_df.loc[index, "Revenue growth"] = ((income_stmt_df.iloc[index]["Total Revenue"] / income_stmt_df.iloc[index-1]["Total Revenue"])-1)
                            income_stmt_df.loc[index, "Operating Earnings growth"] = ((income_stmt_df.iloc[index]["Operating Income"] / income_stmt_df.iloc[index-1]["Operating Income"])-1)
                            income_stmt_df.loc[index, "Operating Margin growth"] = ((income_stmt_df.iloc[index]["Operating Margin"] / income_stmt_df.iloc[index-1]["Operating Margin"])-1)
                            income_stmt_df.loc[index, "Net Income growth"] = ((income_stmt_df.iloc[index]["Net Income Common Stockholders"] / income_stmt_df.iloc[index-1]["Net Income Common Stockholders"])-1)
                            income_stmt_df.loc[index, "Net Income Margin growth"] = ((income_stmt_df.iloc[index]["Net Income Margin"] / income_stmt_df.iloc[index-1]["Net Income Margin"])-1)
                            income_stmt_df.loc[index, "EPS growth"] = ((income_stmt_df.iloc[index]["EPS"] / income_stmt_df.iloc[index-1]["EPS"])-1)
            else:
                for index, row in income_stmt_df.iterrows():
                    income_stmt_df.loc[index, "Gross Margin"] = (income_stmt_df.loc[index, "Gross Profit"] / income_stmt_df.loc[index, "Total Revenue"])
                    income_stmt_df.loc[index, "Operating Margin"] = (income_stmt_df.loc[index, "Operating Income"] / income_stmt_df.loc[index, "Total Revenue"])
                    income_stmt_df.loc[index, "Net Income Margin"] = (income_stmt_df.loc[index, "Net Income Common Stockholders"] / income_stmt_df.loc[index, "Total Revenue"])
                    income_stmt_df.loc[index, "EPS"] = (income_stmt_df.loc[index, "Net Income Common Stockholders"] / income_stmt_df.loc[index, "Diluted Average Shares"])
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
                        income_stmt_df.loc[index, "Revenue growth"] = ((income_stmt_df.iloc[index]["Total Revenue"] / income_stmt_df.iloc[index-1]["Total Revenue"])-1)
                        income_stmt_df.loc[index, "Gross Profit growth"] = ((income_stmt_df.iloc[index]["Gross Profit"] / income_stmt_df.iloc[index-1]["Gross Profit"])-1)
                        income_stmt_df.loc[index, "Gross Margin growth"] = ((income_stmt_df.iloc[index]["Gross Margin"] / income_stmt_df.iloc[index-1]["Gross Margin"])-1)
                        income_stmt_df.loc[index, "Operating Earnings growth"] = ((income_stmt_df.iloc[index]["Operating Income"] / income_stmt_df.iloc[index-1]["Operating Income"])-1)
                        income_stmt_df.loc[index, "Operating Margin growth"] = ((income_stmt_df.iloc[index]["Operating Margin"] / income_stmt_df.iloc[index-1]["Operating Margin"])-1)
                        income_stmt_df.loc[index, "Net Income growth"] = ((income_stmt_df.iloc[index]["Net Income Common Stockholders"] / income_stmt_df.iloc[index-1]["Net Income Common Stockholders"])-1)
                        income_stmt_df.loc[index, "Net Income Margin growth"] = ((income_stmt_df.iloc[index]["Net Income Margin"] / income_stmt_df.iloc[index-1]["Net Income Margin"])-1)
                        income_stmt_df.loc[index, "EPS growth"] = ((income_stmt_df.iloc[index]["EPS"] / income_stmt_df.iloc[index-1]["EPS"])-1)
        elif "biotechnology" in industry.lower():
            # Check is "Gross Profit" is in the income_stmt_df dataframe
            if "Gross Profit" not in income_stmt_df.columns:
                for index, row in income_stmt_df.iterrows():
                    income_stmt_df.loc[index, "Operating Margin"] = (income_stmt_df.loc[index, "Operating Income"] / income_stmt_df.loc[index, "Total Revenue"])
                    income_stmt_df.loc[index, "Net Income Margin"] = (income_stmt_df.loc[index, "Net Income Common Stockholders"] / income_stmt_df.loc[index, "Total Revenue"])
                    income_stmt_df.loc[index, "EPS"] = (income_stmt_df.loc[index, "Net Income Common Stockholders"] / income_stmt_df.loc[index, "Diluted Average Shares"])
                    if index == 0:
                        income_stmt_df.loc[index, "Revenue growth"] = 0.0
                        income_stmt_df.loc[index, "Operating Earnings growth"] = 0.0
                        income_stmt_df.loc[index, "Operating Margin growth"] = 0.0
                        income_stmt_df.loc[index, "Net Income growth"] = 0.0
                        income_stmt_df.loc[index, "Net Income Margin growth"] = 0.0
                        income_stmt_df.loc[index, "EPS growth"] = 0.0
                    else:
                        income_stmt_df.loc[index, "Revenue growth"] = ((income_stmt_df.iloc[index]["Total Revenue"] / income_stmt_df.iloc[index-1]["Total Revenue"])-1)
                        income_stmt_df.loc[index, "Operating Earnings growth"] = ((income_stmt_df.iloc[index]["Operating Income"] / income_stmt_df.iloc[index-1]["Operating Income"])-1)
                        income_stmt_df.loc[index, "Operating Margin growth"] = ((income_stmt_df.iloc[index]["Operating Margin"] / income_stmt_df.iloc[index-1]["Operating Margin"])-1)
                        income_stmt_df.loc[index, "Net Income growth"] = ((income_stmt_df.iloc[index]["Net Income Common Stockholders"] / income_stmt_df.iloc[index-1]["Net Income Common Stockholders"])-1)
                        income_stmt_df.loc[index, "Net Income Margin growth"] = ((income_stmt_df.iloc[index]["Net Income Margin"] / income_stmt_df.iloc[index-1]["Net Income Margin"])-1)
                        income_stmt_df.loc[index, "EPS growth"] = ((income_stmt_df.iloc[index]["EPS"] / income_stmt_df.iloc[index-1]["EPS"])-1)
            else:
                for index, row in income_stmt_df.iterrows():
                    income_stmt_df.loc[index, "Gross Margin"] = (income_stmt_df.loc[index, "Gross Profit"] / income_stmt_df.loc[index, "Total Revenue"])
                    income_stmt_df.loc[index, "Operating Margin"] = (income_stmt_df.loc[index, "Operating Income"] / income_stmt_df.loc[index, "Total Revenue"])
                    income_stmt_df.loc[index, "Net Income Margin"] = (income_stmt_df.loc[index, "Net Income Common Stockholders"] / income_stmt_df.loc[index, "Total Revenue"])
                    income_stmt_df.loc[index, "EPS"] = (income_stmt_df.loc[index, "Net Income Common Stockholders"] / income_stmt_df.loc[index, "Diluted Average Shares"])
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
                        income_stmt_df.loc[index, "Revenue growth"] = ((income_stmt_df.iloc[index]["Total Revenue"] / income_stmt_df.iloc[index-1]["Total Revenue"])-1)
                        income_stmt_df.loc[index, "Gross Profit growth"] = ((income_stmt_df.iloc[index]["Gross Profit"] / income_stmt_df.iloc[index-1]["Gross Profit"])-1)
                        income_stmt_df.loc[index, "Gross Margin growth"] = ((income_stmt_df.iloc[index]["Gross Margin"] / income_stmt_df.iloc[index-1]["Gross Margin"])-1)
                        income_stmt_df.loc[index, "Operating Earnings growth"] = ((income_stmt_df.iloc[index]["Operating Income"] / income_stmt_df.iloc[index-1]["Operating Income"])-1)
                        income_stmt_df.loc[index, "Operating Margin growth"] = ((income_stmt_df.iloc[index]["Operating Margin"] / income_stmt_df.iloc[index-1]["Operating Margin"])-1)
                        income_stmt_df.loc[index, "Net Income growth"] = ((income_stmt_df.iloc[index]["Net Income Common Stockholders"] / income_stmt_df.iloc[index-1]["Net Income Common Stockholders"])-1)
                        income_stmt_df.loc[index, "Net Income Margin growth"] = ((income_stmt_df.iloc[index]["Net Income Margin"] / income_stmt_df.iloc[index-1]["Net Income Margin"])-1)
                        income_stmt_df.loc[index, "EPS growth"] = ((income_stmt_df.iloc[index]["EPS"] / income_stmt_df.iloc[index-1]["EPS"])-1)
        else:
            for index, row in income_stmt_df.iterrows():
                income_stmt_df.loc[index, "Gross Margin"] = (income_stmt_df.loc[index, "Gross Profit"] / income_stmt_df.loc[index, "Total Revenue"])
                income_stmt_df.loc[index, "Operating Margin"] = (income_stmt_df.loc[index, "Operating Income"] / income_stmt_df.loc[index, "Total Revenue"])
                income_stmt_df.loc[index, "Net Income Margin"] = (income_stmt_df.loc[index, "Net Income Common Stockholders"] / income_stmt_df.loc[index, "Total Revenue"])
                income_stmt_df.loc[index, "EPS"] = (income_stmt_df.loc[index, "Net Income Common Stockholders"] / income_stmt_df.loc[index, "Diluted Average Shares"])
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
                    income_stmt_df.loc[index, "Revenue growth"] = ((income_stmt_df.iloc[index]["Total Revenue"] / income_stmt_df.iloc[index-1]["Total Revenue"])-1)
                    income_stmt_df.loc[index, "Gross Profit growth"] = ((income_stmt_df.iloc[index]["Gross Profit"] / income_stmt_df.iloc[index-1]["Gross Profit"])-1)
                    income_stmt_df.loc[index, "Gross Margin growth"] = ((income_stmt_df.iloc[index]["Gross Margin"] / income_stmt_df.iloc[index-1]["Gross Margin"])-1)
                    income_stmt_df.loc[index, "Operating Earnings growth"] = ((income_stmt_df.iloc[index]["Operating Income"] / income_stmt_df.iloc[index-1]["Operating Income"])-1)
                    income_stmt_df.loc[index, "Operating Margin growth"] = ((income_stmt_df.iloc[index]["Operating Margin"] / income_stmt_df.iloc[index-1]["Operating Margin"])-1)
                    income_stmt_df.loc[index, "Net Income growth"] = ((income_stmt_df.iloc[index]["Net Income Common Stockholders"] / income_stmt_df.iloc[index-1]["Net Income Common Stockholders"])-1)
                    income_stmt_df.loc[index, "Net Income Margin growth"] = ((income_stmt_df.iloc[index]["Net Income Margin"] / income_stmt_df.iloc[index-1]["Net Income Margin"])-1)
                    income_stmt_df.loc[index, "EPS growth"] = ((income_stmt_df.iloc[index]["EPS"] / income_stmt_df.iloc[index-1]["EPS"])-1)


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
                    balancesheet_df.loc[index, "Book Value"] = (balancesheet_df.loc[index, "Total Equity Gross Minority Interest"])
                    balancesheet_df.loc[index, "Book Value per share"] = (balancesheet_df.loc[index, "Total Equity Gross Minority Interest"] / income_stmt_df.loc[index, "Diluted Average Shares"])
                    balancesheet_df.loc[index, "Return on Assets"] = (income_stmt_df.loc[index, "Net Income Common Stockholders"] / balancesheet_df.loc[index, "Total Assets"])
                    balancesheet_df.loc[index, "Return on Equity"] = (income_stmt_df.loc[index, "Net Income Common Stockholders"] / balancesheet_df.loc[index, "Total Equity Gross Minority Interest"])
                    balancesheet_df.loc[index, "Current Ratio"] = ((balancesheet_df.loc[index, "Current Assets"]) / balancesheet_df.loc[index, "Current Liabilities"])
                    balancesheet_df.loc[index, "Quick Ratio"] = (balancesheet_df.loc[index, "Cash And Cash Equivalents"] / balancesheet_df.loc[index, "Current Liabilities"])
                    balancesheet_df.loc[index, "Debt to Equity"] = (balancesheet_df.loc[index, "Total Liabilities Net Minority Interest"] / balancesheet_df.loc[index, "Total Equity Gross Minority Interest"])
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
                        balancesheet_df.loc[index, "Total Assets growth"] = ((balancesheet_df.iloc[index]["Total Assets"] / balancesheet_df.iloc[index-1]["Total Assets"])-1)
                        balancesheet_df.loc[index, "Current Assets growth"] = ((balancesheet_df.iloc[index]["Current Assets"] / balancesheet_df.iloc[index-1]["Current Assets"])-1)
                        balancesheet_df.loc[index, "Cash and Cash Equivalents growth"] = ((balancesheet_df.iloc[index]["Cash And Cash Equivalents"] / balancesheet_df.iloc[index-1]["Cash And Cash Equivalents"])-1)
                        balancesheet_df.loc[index, "Total Liabilities growth"] = ((balancesheet_df.iloc[index]["Total Liabilities Net Minority Interest"] / balancesheet_df.iloc[index-1]["Total Liabilities Net Minority Interest"])-1)
                        balancesheet_df.loc[index, "Total Equity growth"] = ((balancesheet_df.iloc[index]["Total Equity Gross Minority Interest"] / balancesheet_df.iloc[index-1]["Total Equity Gross Minority Interest"])-1)
                        balancesheet_df.loc[index, "Current Liabilities growth"] = ((balancesheet_df.iloc[index]["Current Liabilities"] / balancesheet_df.iloc[index-1]["Current Liabilities"])-1)
                        balancesheet_df.loc[index, "Book Value growth"] = ((balancesheet_df.iloc[index]["Book Value"] / balancesheet_df.iloc[index-1]["Book Value"])-1)
                        balancesheet_df.loc[index, "Book Value per share growth"] = ((balancesheet_df.iloc[index]["Book Value per share"] / balancesheet_df.iloc[index-1]["Book Value per share"])-1)
                        balancesheet_df.loc[index, "Return on Assets growth"] = ((balancesheet_df.iloc[index]["Return on Assets"] / balancesheet_df.iloc[index-1]["Return on Assets"])-1)
                        balancesheet_df.loc[index, "Return on Equity growth"] = ((balancesheet_df.iloc[index]["Return on Equity"] / balancesheet_df.iloc[index-1]["Return on Equity"])-1)
                        balancesheet_df.loc[index, "Current Ratio growth"] = ((balancesheet_df.iloc[index]["Current Ratio"] / balancesheet_df.iloc[index-1]["Current Ratio"])-1)
                        balancesheet_df.loc[index, "Quick Ratio growth"] = ((balancesheet_df.iloc[index]["Quick Ratio"] / balancesheet_df.iloc[index-1]["Quick Ratio"])-1)
                        balancesheet_df.loc[index, "Debt to Equity growth"] = ((balancesheet_df.iloc[index]["Debt to Equity"] / balancesheet_df.iloc[index-1]["Debt to Equity"])-1)                
            else:
                for index, row in balancesheet_df.iterrows():
                    balancesheet_df.loc[index, "Book Value"] = (balancesheet_df.loc[index, "Total Equity Gross Minority Interest"])
                    balancesheet_df.loc[index, "Book Value per share"] = (balancesheet_df.loc[index, "Total Equity Gross Minority Interest"] / income_stmt_df.loc[index, "Diluted Average Shares"])
                    balancesheet_df.loc[index, "Return on Assets"] = (income_stmt_df.loc[index, "Net Income Common Stockholders"] / balancesheet_df.loc[index, "Total Assets"])
                    balancesheet_df.loc[index, "Return on Equity"] = (income_stmt_df.loc[index, "Net Income Common Stockholders"] / balancesheet_df.loc[index, "Total Equity Gross Minority Interest"])
                    balancesheet_df.loc[index, "Current Ratio"] = (balancesheet_df.loc[index, "Current Assets"] / balancesheet_df.loc[index, "Current Liabilities"])
                    balancesheet_df.loc[index, "Quick Ratio"] = (balancesheet_df.loc[index, "Cash And Cash Equivalents"] / balancesheet_df.loc[index, "Current Liabilities"])
                    balancesheet_df.loc[index, "Debt to Equity"] = (balancesheet_df.loc[index, "Total Liabilities Net Minority Interest"] / balancesheet_df.loc[index, "Total Equity Gross Minority Interest"])
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
                        balancesheet_df.loc[index, "Total Assets growth"] = ((balancesheet_df.iloc[index]["Total Assets"] / balancesheet_df.iloc[index-1]["Total Assets"])-1)
                        balancesheet_df.loc[index, "Current Assets growth"] = ((balancesheet_df.iloc[index]["Current Assets"] / balancesheet_df.iloc[index-1]["Current Assets"])-1)
                        balancesheet_df.loc[index, "Cash and Cash Equivalents growth"] = ((balancesheet_df.iloc[index]["Cash And Cash Equivalents"] / balancesheet_df.iloc[index-1]["Cash And Cash Equivalents"])-1)
                        balancesheet_df.loc[index, "Total Liabilities growth"] = ((balancesheet_df.iloc[index]["Total Liabilities Net Minority Interest"] / balancesheet_df.iloc[index-1]["Total Liabilities Net Minority Interest"])-1)
                        balancesheet_df.loc[index, "Total Equity growth"] = ((balancesheet_df.iloc[index]["Total Equity Gross Minority Interest"] / balancesheet_df.iloc[index-1]["Total Equity Gross Minority Interest"])-1)
                        balancesheet_df.loc[index, "Current Liabilities growth"] = ((balancesheet_df.iloc[index]["Current Liabilities"] / balancesheet_df.iloc[index-1]["Current Liabilities"])-1)
                        balancesheet_df.loc[index, "Book Value growth"] = ((balancesheet_df.iloc[index]["Book Value"] / balancesheet_df.iloc[index-1]["Book Value"])-1)
                        balancesheet_df.loc[index, "Book Value per share growth"] = ((balancesheet_df.iloc[index]["Book Value per share"] / balancesheet_df.iloc[index-1]["Book Value per share"])-1)
                        balancesheet_df.loc[index, "Return on Assets growth"] = ((balancesheet_df.iloc[index]["Return on Assets"] / balancesheet_df.iloc[index-1]["Return on Assets"])-1)
                        balancesheet_df.loc[index, "Return on Equity growth"] = ((balancesheet_df.iloc[index]["Return on Equity"] / balancesheet_df.iloc[index-1]["Return on Equity"])-1)
                        balancesheet_df.loc[index, "Current Ratio growth"] = ((balancesheet_df.iloc[index]["Current Ratio"] / balancesheet_df.iloc[index-1]["Current Ratio"])-1)
                        balancesheet_df.loc[index, "Quick Ratio growth"] = ((balancesheet_df.iloc[index]["Quick Ratio"] / balancesheet_df.iloc[index-1]["Quick Ratio"])-1)
                        balancesheet_df.loc[index, "Debt to Equity growth"] = ((balancesheet_df.iloc[index]["Debt to Equity"] / balancesheet_df.iloc[index-1]["Debt to Equity"])-1)
        elif "insurance" in industry.lower():
            if "Current Assets" not in balancesheet_df.columns:
                balancesheet_df["Current Assets"] = 0.0
                balancesheet_df = balancesheet_df.rename(columns={"Derivative Product Liabilities": "Current Liabilities"})
                for index, row in balancesheet_df.iterrows():
                    if "Receivables" not in balancesheet_df.columns:
                        balancesheet_df.loc[index, "Current Assets"] = balancesheet_df.loc[index, "Cash And Cash Equivalents"] + balancesheet_df.loc[index, "Trading Securities"] + balancesheet_df.loc[index, "Financial Assets Designatedas Fair Value Through Profitor Loss Total"]
                    else:
                        balancesheet_df.loc[index, "Current Assets"] = balancesheet_df.loc[index, "Cash And Cash Equivalents"] + balancesheet_df.loc[index, "Receivables"] + balancesheet_df.loc[index, "Trading Securities"] + balancesheet_df.loc[index, "Financial Assets Designatedas Fair Value Through Profitor Loss Total"]
                    balancesheet_df.loc[index, "Book Value"] = (balancesheet_df.loc[index, "Total Equity Gross Minority Interest"])
                    balancesheet_df.loc[index, "Book Value per share"] = (balancesheet_df.loc[index, "Total Equity Gross Minority Interest"] / income_stmt_df.loc[index, "Diluted Average Shares"])
                    balancesheet_df.loc[index, "Return on Assets"] = (income_stmt_df.loc[index, "Net Income Common Stockholders"] / balancesheet_df.loc[index, "Total Assets"])
                    balancesheet_df.loc[index, "Return on Equity"] = (income_stmt_df.loc[index, "Net Income Common Stockholders"] / balancesheet_df.loc[index, "Total Equity Gross Minority Interest"])
                    balancesheet_df.loc[index, "Current Ratio"] = ((balancesheet_df.loc[index, "Current Assets"]) / balancesheet_df.loc[index, "Current Liabilities"])
                    balancesheet_df.loc[index, "Quick Ratio"] = (balancesheet_df.loc[index, "Cash And Cash Equivalents"] / balancesheet_df.loc[index, "Current Liabilities"])
                    balancesheet_df.loc[index, "Debt to Equity"] = (balancesheet_df.loc[index, "Total Liabilities Net Minority Interest"] / balancesheet_df.loc[index, "Total Equity Gross Minority Interest"])
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
                        balancesheet_df.loc[index, "Total Assets growth"] = ((balancesheet_df.iloc[index]["Total Assets"] / balancesheet_df.iloc[index-1]["Total Assets"])-1)
                        balancesheet_df.loc[index, "Current Assets growth"] = ((balancesheet_df.iloc[index]["Current Assets"] / balancesheet_df.iloc[index-1]["Current Assets"])-1)
                        balancesheet_df.loc[index, "Cash and Cash Equivalents growth"] = ((balancesheet_df.iloc[index]["Cash And Cash Equivalents"] / balancesheet_df.iloc[index-1]["Cash And Cash Equivalents"])-1)
                        balancesheet_df.loc[index, "Total Liabilities growth"] = ((balancesheet_df.iloc[index]["Total Liabilities Net Minority Interest"] / balancesheet_df.iloc[index-1]["Total Liabilities Net Minority Interest"])-1)
                        balancesheet_df.loc[index, "Total Equity growth"] = ((balancesheet_df.iloc[index]["Total Equity Gross Minority Interest"] / balancesheet_df.iloc[index-1]["Total Equity Gross Minority Interest"])-1)
                        balancesheet_df.loc[index, "Current Liabilities growth"] = ((balancesheet_df.iloc[index]["Current Liabilities"] / balancesheet_df.iloc[index-1]["Current Liabilities"])-1)
                        balancesheet_df.loc[index, "Book Value growth"] = ((balancesheet_df.iloc[index]["Book Value"] / balancesheet_df.iloc[index-1]["Book Value"])-1)
                        balancesheet_df.loc[index, "Book Value per share growth"] = ((balancesheet_df.iloc[index]["Book Value per share"] / balancesheet_df.iloc[index-1]["Book Value per share"])-1)
                        balancesheet_df.loc[index, "Return on Assets growth"] = ((balancesheet_df.iloc[index]["Return on Assets"] / balancesheet_df.iloc[index-1]["Return on Assets"])-1)
                        balancesheet_df.loc[index, "Return on Equity growth"] = ((balancesheet_df.iloc[index]["Return on Equity"] / balancesheet_df.iloc[index-1]["Return on Equity"])-1)
                        balancesheet_df.loc[index, "Current Ratio growth"] = ((balancesheet_df.iloc[index]["Current Ratio"] / balancesheet_df.iloc[index-1]["Current Ratio"])-1)
                        balancesheet_df.loc[index, "Quick Ratio growth"] = ((balancesheet_df.iloc[index]["Quick Ratio"] / balancesheet_df.iloc[index-1]["Quick Ratio"])-1)
                        balancesheet_df.loc[index, "Debt to Equity growth"] = ((balancesheet_df.iloc[index]["Debt to Equity"] / balancesheet_df.iloc[index-1]["Debt to Equity"])-1)
            else:
                for index, row in balancesheet_df.iterrows():
                    balancesheet_df.loc[index, "Book Value"] = (balancesheet_df.loc[index, "Total Equity Gross Minority Interest"])
                    balancesheet_df.loc[index, "Book Value per share"] = (balancesheet_df.loc[index, "Total Equity Gross Minority Interest"] / income_stmt_df.loc[index, "Diluted Average Shares"])
                    balancesheet_df.loc[index, "Return on Assets"] = (income_stmt_df.loc[index, "Net Income Common Stockholders"] / balancesheet_df.loc[index, "Total Assets"])
                    balancesheet_df.loc[index, "Return on Equity"] = (income_stmt_df.loc[index, "Net Income Common Stockholders"] / balancesheet_df.loc[index, "Total Equity Gross Minority Interest"])
                    balancesheet_df.loc[index, "Current Ratio"] = (balancesheet_df.loc[index, "Current Assets"] / balancesheet_df.loc[index, "Current Liabilities"])
                    balancesheet_df.loc[index, "Quick Ratio"] = (balancesheet_df.loc[index, "Cash Cash Equivalents And Short Term Investments"] / balancesheet_df.loc[index, "Current Liabilities"])
                    balancesheet_df.loc[index, "Debt to Equity"] = (balancesheet_df.loc[index, "Total Liabilities Net Minority Interest"] / balancesheet_df.loc[index, "Total Equity Gross Minority Interest"])
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
                        balancesheet_df.loc[index, "Total Assets growth"] = ((balancesheet_df.iloc[index]["Total Assets"] / balancesheet_df.iloc[index-1]["Total Assets"])-1)
                        balancesheet_df.loc[index, "Current Assets growth"] = ((balancesheet_df.iloc[index]["Current Assets"] / balancesheet_df.iloc[index-1]["Current Assets"])-1)
                        balancesheet_df.loc[index, "Cash and Cash Equivalents growth"] = ((balancesheet_df.iloc[index]["Cash Cash Equivalents And Short Term Investments"] / balancesheet_df.iloc[index-1]["Cash Cash Equivalents And Short Term Investments"])-1)
                        balancesheet_df.loc[index, "Total Liabilities growth"] = ((balancesheet_df.iloc[index]["Total Liabilities Net Minority Interest"] / balancesheet_df.iloc[index-1]["Total Liabilities Net Minority Interest"])-1)
                        balancesheet_df.loc[index, "Total Equity growth"] = ((balancesheet_df.iloc[index]["Total Equity Gross Minority Interest"] / balancesheet_df.iloc[index-1]["Total Equity Gross Minority Interest"])-1)
                        balancesheet_df.loc[index, "Current Liabilities growth"] = ((balancesheet_df.iloc[index]["Current Liabilities"] / balancesheet_df.iloc[index-1]["Current Liabilities"])-1)
                        balancesheet_df.loc[index, "Book Value growth"] = ((balancesheet_df.iloc[index]["Book Value"] / balancesheet_df.iloc[index-1]["Book Value"])-1)
                        balancesheet_df.loc[index, "Book Value per share growth"] = ((balancesheet_df.iloc[index]["Book Value per share"] / balancesheet_df.iloc[index-1]["Book Value per share"])-1)
                        balancesheet_df.loc[index, "Return on Assets growth"] = ((balancesheet_df.iloc[index]["Return on Assets"] / balancesheet_df.iloc[index-1]["Return on Assets"])-1)
                        balancesheet_df.loc[index, "Return on Equity growth"] = ((balancesheet_df.iloc[index]["Return on Equity"] / balancesheet_df.iloc[index-1]["Return on Equity"])-1)
                        balancesheet_df.loc[index, "Current Ratio growth"] = ((balancesheet_df.iloc[index]["Current Ratio"] / balancesheet_df.iloc[index-1]["Current Ratio"])-1)
                        balancesheet_df.loc[index, "Quick Ratio growth"] = ((balancesheet_df.iloc[index]["Quick Ratio"] / balancesheet_df.iloc[index-1]["Quick Ratio"])-1)
                        balancesheet_df.loc[index, "Debt to Equity growth"] = ((balancesheet_df.iloc[index]["Debt to Equity"] / balancesheet_df.iloc[index-1]["Debt to Equity"])-1)
        else:
            for index, row in balancesheet_df.iterrows():
                balancesheet_df.loc[index, "Book Value"] = (balancesheet_df.loc[index, "Total Equity Gross Minority Interest"])
                balancesheet_df.loc[index, "Book Value per share"] = (balancesheet_df.loc[index, "Total Equity Gross Minority Interest"] / income_stmt_df.loc[index, "Diluted Average Shares"])
                balancesheet_df.loc[index, "Return on Assets"] = (income_stmt_df.loc[index, "Net Income Common Stockholders"] / balancesheet_df.loc[index, "Total Assets"])
                balancesheet_df.loc[index, "Return on Equity"] = (income_stmt_df.loc[index, "Net Income Common Stockholders"] / balancesheet_df.loc[index, "Total Equity Gross Minority Interest"])
                balancesheet_df.loc[index, "Current Ratio"] = (balancesheet_df.loc[index, "Current Assets"] / balancesheet_df.loc[index, "Current Liabilities"])
                balancesheet_df.loc[index, "Quick Ratio"] = (balancesheet_df.loc[index, "Cash Cash Equivalents And Short Term Investments"] / balancesheet_df.loc[index, "Current Liabilities"])
                balancesheet_df.loc[index, "Debt to Equity"] = (balancesheet_df.loc[index, "Total Liabilities Net Minority Interest"] / balancesheet_df.loc[index, "Total Equity Gross Minority Interest"])
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
                    balancesheet_df.loc[index, "Current Ratio growth"] = 0.0
                    balancesheet_df.loc[index, "Quick Ratio growth"] = 0.0
                    balancesheet_df.loc[index, "Debt to Equity growth"] = 0.0
                else:
                    balancesheet_df.loc[index, "Total Assets growth"] = ((balancesheet_df.iloc[index]["Total Assets"] / balancesheet_df.iloc[index-1]["Total Assets"])-1)
                    balancesheet_df.loc[index, "Current Assets growth"] = ((balancesheet_df.iloc[index]["Current Assets"] / balancesheet_df.iloc[index-1]["Current Assets"])-1)
                    balancesheet_df.loc[index, "Cash and Cash Equivalents growth"] = ((balancesheet_df.iloc[index]["Cash Cash Equivalents And Short Term Investments"] / balancesheet_df.iloc[index-1]["Cash Cash Equivalents And Short Term Investments"])-1)
                    balancesheet_df.loc[index, "Total Liabilities growth"] = ((balancesheet_df.iloc[index]["Total Liabilities Net Minority Interest"] / balancesheet_df.iloc[index-1]["Total Liabilities Net Minority Interest"])-1)
                    balancesheet_df.loc[index, "Total Equity growth"] = ((balancesheet_df.iloc[index]["Total Equity Gross Minority Interest"] / balancesheet_df.iloc[index-1]["Total Equity Gross Minority Interest"])-1)
                    balancesheet_df.loc[index, "Current Liabilities growth"] = ((balancesheet_df.iloc[index]["Current Liabilities"] / balancesheet_df.iloc[index-1]["Current Liabilities"])-1)
                    balancesheet_df.loc[index, "Book Value growth"] = ((balancesheet_df.iloc[index]["Book Value"] / balancesheet_df.iloc[index-1]["Book Value"])-1)
                    balancesheet_df.loc[index, "Book Value per share growth"] = ((balancesheet_df.iloc[index]["Book Value per share"] / balancesheet_df.iloc[index-1]["Book Value per share"])-1)
                    balancesheet_df.loc[index, "Return on Assets growth"] = ((balancesheet_df.iloc[index]["Return on Assets"] / balancesheet_df.iloc[index-1]["Return on Assets"])-1)
                    balancesheet_df.loc[index, "Return on Equity growth"] = ((balancesheet_df.iloc[index]["Return on Equity"] / balancesheet_df.iloc[index-1]["Return on Equity"])-1)
                    balancesheet_df.loc[index, "Current Ratio growth"] = ((balancesheet_df.iloc[index]["Current Ratio"] / balancesheet_df.iloc[index-1]["Current Ratio"])-1)
                    balancesheet_df.loc[index, "Quick Ratio growth"] = ((balancesheet_df.iloc[index]["Quick Ratio"] / balancesheet_df.iloc[index-1]["Quick Ratio"])-1)
                    balancesheet_df.loc[index, "Debt to Equity growth"] = ((balancesheet_df.iloc[index]["Debt to Equity"] / balancesheet_df.iloc[index-1]["Debt to Equity"])-1)


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
                cashflow_df.loc[index, "Free Cash Flow growth"] = ((cashflow_df.iloc[index]["Free Cash Flow"] / cashflow_df.iloc[index-1]["Free Cash Flow"])-1)
                cashflow_df.loc[index, "Free Cash Flow per share growth"] = ((cashflow_df.iloc[index]["Free Cash Flow per share"] / cashflow_df.iloc[index-1]["Free Cash Flow per share"])-1)

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
                        "Ticker" ,"Date", "Amount of stocks", "Revenue", "Revenue growth", "Net Income", "Net Income growth", "Net Income Margin", "Net Income Margin growth",
                        "EPS", "EPS growth", "Total Assets", "Total Assets growth", "Current Assets", "Current Assets growth",
                        "Cash and Cash Equivalents", "Cash and Cash Equivalents growth", "Total Liabilities",
                        "Total Liabilities growth", "Total Equity", "Total Equity growth", "Current Liabilities",
                        "Current Liabilities growth", "Book Value", "Book Value growth", "Book Value per share",
                        "Book Value per share growth", "Return on Assets", "Return on Assets growth", "Return on Equity",
                        "Return on Equity growth", "Current Ratio", "Current Ratio growth", "Quick Ratio",
                        "Quick Ratio growth", "Debt to Equity", "Debt to Equity growth",
                        "Free Cash Flow", "Free Cash Flow growth", "Free Cash Flow per share", "Free Cash Flow per share growth"
                    ]]
                else:
                    full_stock_financial_data_df = full_stock_financial_data_df.rename(columns={
                        "Diluted Average Shares": "Amount of stocks", "Total Revenue": "Revenue",
                        "Cash And Cash Equivalents": "Cash and Cash Equivalents",
                        "Total Liabilities Net Minority Interest": "Total Liabilities",
                        "Total Equity Gross Minority Interest": "Total Equity"
                    })
                    full_stock_financial_data_df = full_stock_financial_data_df[[
                        "Ticker" ,"Date", "Amount of stocks", "Revenue", "Revenue growth", "Operating Income", "Operating Earnings growth", "Operating Margin", "Operating Margin growth",
                        "Net Income", "Net Income growth", "Net Income Margin", "Net Income Margin growth", "EPS", "EPS growth", "Total Assets",
                        "Total Assets growth", "Current Assets", "Current Assets growth", "Cash and Cash Equivalents",
                        "Cash and Cash Equivalents growth", "Total Liabilities", "Total Liabilities growth", "Total Equity",
                        "Total Equity growth", "Current Liabilities", "Current Liabilities growth", "Book Value", "Book Value growth",
                        "Book Value per share", "Book Value per share growth", "Return on Assets", "Return on Assets growth",
                        "Return on Equity", "Return on Equity growth", "Current Ratio", "Current Ratio growth", "Quick Ratio",
                        "Quick Ratio growth", "Debt to Equity", "Debt to Equity growth",
                        "Free Cash Flow", "Free Cash Flow growth", "Free Cash Flow per share", "Free Cash Flow per share growth"
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
                        "Ticker" ,"Date", "Amount of stocks", "Revenue", "Revenue growth", "Net Income", "Net Income growth", "Net Income Margin", "Net Income Margin growth",
                        "EPS", "EPS growth", "Total Assets", "Total Assets growth", "Current Assets", "Current Assets growth",
                        "Cash and Cash Equivalents", "Cash and Cash Equivalents growth", "Total Liabilities",
                        "Total Liabilities growth", "Total Equity", "Total Equity growth", "Current Liabilities",
                        "Current Liabilities growth", "Book Value", "Book Value growth", "Book Value per share",
                        "Book Value per share growth", "Return on Assets", "Return on Assets growth", "Return on Equity",
                        "Return on Equity growth", "Current Ratio", "Current Ratio growth", "Quick Ratio",
                        "Quick Ratio growth", "Debt to Equity", "Debt to Equity growth",
                        "Free Cash Flow", "Free Cash Flow growth", "Free Cash Flow per share", "Free Cash Flow per share growth"
                    ]]
                else:
                    full_stock_financial_data_df = full_stock_financial_data_df.rename(columns={
                        "Diluted Average Shares": "Amount of stocks", "Total Revenue": "Revenue",
                        "Cash And Cash Equivalents": "Cash and Cash Equivalents",
                        "Total Liabilities Net Minority Interest": "Total Liabilities",
                        "Total Equity Gross Minority Interest": "Total Equity"
                    })
                    full_stock_financial_data_df = full_stock_financial_data_df[[
                        "Ticker" ,"Date", "Amount of stocks", "Revenue", "Revenue growth", "Operating Income", "Operating Earnings growth", "Operating Margin", "Operating Margin growth",
                        "Net Income", "Net Income growth", "Net Income Margin", "Net Income Margin growth", "EPS", "EPS growth", "Total Assets",
                        "Total Assets growth", "Current Assets", "Current Assets growth", "Cash and Cash Equivalents",
                        "Cash and Cash Equivalents growth", "Total Liabilities", "Total Liabilities growth", "Total Equity",
                        "Total Equity growth", "Current Liabilities", "Current Liabilities growth", "Book Value", "Book Value growth",
                        "Book Value per share", "Book Value per share growth", "Return on Assets", "Return on Assets growth",
                        "Return on Equity", "Return on Equity growth", "Current Ratio", "Current Ratio growth", "Quick Ratio",
                        "Quick Ratio growth", "Debt to Equity", "Debt to Equity growth",
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
                    "Return on Equity growth", "Current Ratio", "Current Ratio growth", "Quick Ratio",
                    "Quick Ratio growth", "Debt to Equity", "Debt to Equity growth",
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
                    "Return on Equity growth", "Current Ratio", "Current Ratio growth", "Quick Ratio",
                    "Quick Ratio growth", "Debt to Equity", "Debt to Equity growth",
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
                    "Return on Equity growth", "Current Ratio", "Current Ratio growth", "Quick Ratio",
                    "Quick Ratio growth", "Debt to Equity", "Debt to Equity growth",
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
                "Return on Equity growth", "Current Ratio", "Current Ratio growth", "Quick Ratio",
                "Quick Ratio growth", "Debt to Equity", "Debt to Equity growth",
                "Free Cash Flow", "Free Cash Flow growth", "Free Cash Flow per share", "Free Cash Flow per share growth"
            ]]


    except KeyError:
        raise KeyError(f"Stock symbol '{symbol}' is invalid or not found.")
    
    
    return full_stock_financial_data_df     
#Create a function the combines dataframe from fetch_stock_price_data with full_stock_financial_data_df from fetch_stock_financial_data
def combine_stock_data(stock_data_df, full_stock_financial_data_df):
    """
    Combines stock data with financial stock data.

    The function combines stock data with financial stock data and returns a DataFrame with the combined data.

    Parameters:
    - stock_data_df (pd.DataFrame): A DataFrame with stock data.
    - full_stock_financial_data_df (pd.DataFrame): A DataFrame with financial stock data.

    Returns:
    - combined_stock_data_df (pd.DataFrame): A DataFrame with the combined data.

    Raises:
    - ValueError: If the stock_data_df parameter is empty or if the full_stock_financial_data_df parameter is empty.
    """
    
    # Checking if the stock_data_df parameter is empty
    if stock_data_df.empty:
        raise ValueError("No stock data provided.")
    

    # Checking if the full_stock_financial_data_df parameter is empty
    if full_stock_financial_data_df.empty:
        raise ValueError("No financial stock data provided.")


    # Create a list of column names to copy from full_stock_financial_data_df to stock_data_df
    column_names = full_stock_financial_data_df.columns[2:]
    # Create a copy of stock_data_df
    combined_stock_data_df = stock_data_df.copy()
    # Add columns from full_stock_financial_data_df to stock_data_df
    for year in range(len(full_stock_financial_data_df["Date"])):
        combined_stock_data_df.loc[combined_stock_data_df["Date"] >= full_stock_financial_data_df.iloc[year]["Date"], column_names] = full_stock_financial_data_df.iloc[year].values[2:]


    # Drop rows with NaN values in stock_data_df
    combined_stock_data_df = combined_stock_data_df.dropna()
    combined_stock_data_df = combined_stock_data_df.reset_index(drop=True)
    print("Stock data and financial stock data combined successfully.")
    return combined_stock_data_df

# Create a function that calculates P/S, P/E, P/B and P/FCF ratios
def calculate_ratios(combined_stock_data_df):
    # Calculate the P/S ratio
    combined_stock_data_df["P/S"] = combined_stock_data_df["Price"] / (combined_stock_data_df["Revenue"] / combined_stock_data_df["Amount of stocks"])
    # Calculate the P/E ratio
    combined_stock_data_df["P/E"] = combined_stock_data_df["Price"] / combined_stock_data_df["EPS"]
    # Calculate the P/B ratio
    combined_stock_data_df["P/B"] = combined_stock_data_df["Price"] / combined_stock_data_df["Book Value per share"]
    # Calculate the P/FCF ratio
    combined_stock_data_df["P/FCF"] = combined_stock_data_df["Price"] / combined_stock_data_df["Free Cash Flow per share growth"]
    print("Ratios have been calculated successfully, and added to the dataframe.")
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
        writer = pd.ExcelWriter(excel_file, engine="xlsxwriter")

        # Export each dataframe to a separate sheet in the Excel file
        for sheet_name, dataframe in dataframes.items():
            dataframe.to_excel(writer, sheet_name=sheet_name, index=False)

        # Save the Excel file
        writer.close()
        print("Dataframes have been successfully exported as xlsx file.")

    except ValueError as e:
        raise ValueError(f"Error exporting to Excel: {e}")

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
        reader = pd.ExcelFile(excel_file)

        # Import each sheet in the Excel file into a separate dataframe
        for sheet_name in reader.sheet_names:
            dataframes[sheet_name] = pd.read_excel(reader, sheet_name=sheet_name)

        # Return the dictionary of dataframes
        return dataframes

    except ValueError as e:
        raise ValueError(f"Error importing from Excel: {e}")
    
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
        dataframe.to_csv(f"{file_name}.csv", sep=',', encoding='utf-8', index=False)
        

    except ValueError as e:
        raise ValueError(f"Error converting to CSV: {e}")


if __name__ == "__main__":
    start_time = time.time()
    # Import stock symbols from a CSV file
    stock_symbols_df = import_stock_symbols('index_symbol_list_single_stock.csv')
    stock_symbols_list = stock_symbols_df['Symbol'].tolist()
    stock_symbol = stock_symbols_list[0]
    print(stock_symbol)
    # Fetch stock data for the imported stock symbols
    stock_data_df = fetch_stock_price_data(stock_symbol)
    # print(stock_data_df)
    # Fetch stock data for the imported stock symbols
    full_stock_financial_data_df = fetch_stock_financial_data(stock_symbol)
    # print(full_stock_financial_data_df)
    # Combine stock data with stock financial data
    combined_stock_data_df = combine_stock_data(stock_data_df, full_stock_financial_data_df)
    # print(combined_stock_data_df)
    # Calculate ratios
    combined_stock_data_df = calculate_ratios(combined_stock_data_df)
    # print(combined_stock_data_df)
    # Create a dictionary of dataframes to export to Excel
    dataframes = {
        # "Stock Data": stock_data_df,
        # "Full Stock Financial Data": full_stock_financial_data_df,
        "Combined Stock Data": combined_stock_data_df
    }
    # Export the dataframes to an Excel file
    export_to_excel(dataframes, 'stock_data_single_v2.xlsx')
    # Import the stock data from an Excel file
    dataframes = import_excel("stock_data_single_v2.xlsx")
    for key, value in dataframes.items():
        dataframe = value


    # Export the stock data to a CSV file
    convert_excel_to_csv(dataframe, "stock_data_single_v2")
    # Calculate the execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds to build dataset.")
