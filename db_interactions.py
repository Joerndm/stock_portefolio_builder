"""
Database Interaction Module

This module provides functions for interacting with a MySQL database containing stock market data.
It handles importing and exporting data across multiple tables including stock information, price data,
financial statements, ratios, and predictions.

Tables:
    - stock_info_data: Basic stock information and tickers
    - stock_price_data: Historical price and technical indicator data
    - stock_income_stmt_data: Income statement financial data
    - stock_balancesheet_data: Balance sheet financial data
    - stock_cash_flow_data: Cash flow statement data
    - stock_ratio_data: Financial ratios and metrics (TTM preferred, annual fallback)
    - stock_prediction_data: Stock prediction data

Functions:
    - import_ticker_list: Retrieves all stock tickers from the database
    - does_stock_exists_*: Checks if a stock exists in specific tables
    - import_stock_*: Imports data from various tables
    - export_stock_*: Exports data to various tables
    - import_stock_dataset: Combines and imports complete stock dataset

Note:
    TTM ratio calculations happen at data fetch time via calculate_ratios_ttm_with_fallback()
    in ttm_financial_calculator.py. The single stock_ratio_data table stores the result.

Dependencies:
    - pandas: For data manipulation and SQL operations
    - fetch_secrets: For retrieving database credentials
    - db_connectors: For establishing database connections

Author: Stock Portfolio Builder
"""
import pandas as pd

import fetch_secrets
import db_connectors

def import_ticker_list():
    """
    This function imports the ticker list from the stock_info_data table in the database.

    Args:
    None
    
    Returns:
    ticker_list: list

    Raises:
    - KeyError: If the secrets cannot be fetched.
    - KeyError: If the connection to the database cannot be established.
    - KeyError: If the tickers cannot be fetched from stock_info_data_df in the database.
    """
    try:
        # Fetch the secrets from the secret_import function
        db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()

    except Exception as e:
        raise KeyError(f"Could not fetch the secrets. Error: {e}") from e

    try:
        db_con = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)

    except Exception as e:
        raise KeyError(f"Could not establish connection to the database. Error: {e}") from e

    try:
        quary = "SELECT ticker FROM stock_info_data"
        stock_info_data_df = pd.read_sql(sql=quary, con=db_con)
        ticker_list = stock_info_data_df["ticker"].tolist()
        return ticker_list

    except Exception as e:
        raise KeyError(f"Could not fetch the tickers from stock_info_data_df in the database. Error: {e}") from e

def does_stock_exists_stock_info_data(stock_ticker=""):
    """
    This function controls if the specified stock exists in the stock_info_data table in the database.

    Args:
    stock_ticker: str

    Returns:
    True: bool
    False: bool

    Raises:
    - ValueError: If the stock_ticker parameter is empty.
    - KeyError: If the secrets cannot be fetched.
    - KeyError: If the connection to the database cannot be established.
    - KeyError: If the control if specified ticker is in stock_info_data in the database cannot be completed.
    """
    # Check if the stock_info_data parameter is empty
    if stock_ticker == "":
        raise ValueError("The stock_symbol parameter cannot be empty.")

    try:
        # Fetch the secrets from the secret_import function
        db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()

    except Exception as e:
        raise KeyError(f"Could not fetch the secrets. Error: {e}") from e

    try:
        db_con = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)

    except Exception as e:
        raise KeyError(f"Could not establish connection to the database. Error: {e}") from e

    try:
        # Fetch the stock_info_data_df from the database
        quary = f"""SELECT * FROM stock_info_data
            WHERE ticker = "{stock_ticker}"
            """
        stock_info_data_df = pd.read_sql(sql=quary, con=db_con)
        # Check if the stock_info_data_df is empty
        if stock_info_data_df.empty:
            response = False
        else:
            response = True

        return response

    except Exception as e:
        raise KeyError(f"Could not control if specified ticker is in stock_info_data in the database. Error: {e}") from e

def does_stock_exists_stock_price_data(stock_ticker=""):
    """
    This function controls if the specified stock exists in the stock_price_data table in the database.

    Args:
    stock_ticker: str

    Returns:
    True: bool
    False: bool

    Raises:
    - ValueError: If the stock_ticker parameter is empty.
    - KeyError: If the secrets cannot be fetched.
    - KeyError: If the connection to the database cannot be established.
    - KeyError: If the control if specified ticker is in stock_price_data in the database cannot be completed.
    """
    # Check if the stock_price_data_df parameter is empty
    if stock_ticker == "":
        raise ValueError("The stock_symbol parameter cannot be empty.")

    try:
        # Fetch the secrets from the secret_import function
        db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()

    except Exception as e:
        raise KeyError(f"Could not fetch the secrets. Error: {e}") from e

    try:
        db_con = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)

    except Exception as e:
        raise KeyError(f"Could not establish connection to the database. Error: {e}") from e

    try:
        # Fetch the stock_price_data_df from the database
        quary = f"""SELECT * FROM stock_price_data
            WHERE ticker = "{stock_ticker}"
            """
        stock_price_data_df = pd.read_sql(sql=quary, con=db_con)
        # Check if the stock_price_data_df is empty
        if stock_price_data_df.empty:
            response = False
        else:
            response = True

        return response

    except Exception as e:
        raise KeyError(f"Could not control if specified ticker is in stock_price_data in the database. Error: {e}") from e

def does_stock_exists_stock_income_stmt_data(stock_ticker=""):
    """
    This function controls if the specified stock exists in the stock_income_stmt_data table in the database.

    Args:
    stock_ticker: str

    Returns:
    True: bool
    False: bool

    Raises:
    - ValueError: If the stock_ticker parameter is empty.
    - KeyError: If the secrets cannot be fetched.
    - KeyError: If the connection to the database cannot be established.
    - KeyError: If the control if specified ticker is in stock_income_stmt_data in the database cannot be completed.
    """
    # Check if the stock_price_data_df parameter is empty
    if stock_ticker == "":
        raise ValueError("The stock_symbol parameter cannot be empty.")

    try:
        # Fetch the secrets from the secret_import function
        db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()

    except Exception as e:
        raise KeyError(f"Could not fetch the secrets. Error: {e}") from e

    try:
        db_con = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)

    except Exception as e:
        raise KeyError(f"Could not establish connection to the database. Error: {e}") from e

    try:
        # Fetch the stock_price_data_df from the database
        quary = f"""SELECT * FROM stock_income_stmt_data
            WHERE ticker = "{stock_ticker}"
            """
        stock_income_stmt_data_df = pd.read_sql(sql=quary, con=db_con)
        # Check if the stock_income_stmt_data_df is empty
        if stock_income_stmt_data_df.empty:
            response = False
        else:
            response = True

        return response

    except Exception as e:
        raise KeyError(f"Could not control if specified ticker is in stock_info_data in the database. Error: {e}") from e

def does_stock_exists_stock_balancesheet_data(stock_ticker=""):
    """
    This function controls if the specified stock exists in the stock_balancesheet_data table in the database.

    Args:
    stock_ticker: str

    Returns:
    True: bool
    False: bool

    Raises:
    - ValueError: If the stock_ticker parameter is empty.
    - KeyError: If the secrets cannot be fetched.
    - KeyError: If the connection to the database cannot be established.
    - KeyError: If the control if specified ticker is in stock_balancesheet_data in the database cannot be completed.
    """
    # Check if the stock_balancesheet_data_df parameter is empty
    if stock_ticker == "":
        raise ValueError("The stock_symbol parameter cannot be empty.")

    try:
        # Fetch the secrets from the secret_import function
        db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()

    except Exception as e:
        raise KeyError(f"Could not fetch the secrets. Error: {e}") from e

    try:
        db_con = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)

    except Exception as e:
        raise KeyError(f"Could not establish connection to the database. Error: {e}") from e

    try:
        # Fetch the stock_balancesheet_data_df from the database
        quary = f"""SELECT * FROM stock_balancesheet_data
            WHERE ticker = "{stock_ticker}"
            """
        stock_balancesheet_data_df = pd.read_sql(sql=quary, con=db_con)
        # Check if the stock_balancesheet_data_df is empty
        if stock_balancesheet_data_df.empty:
            response = False
        else:
            response = True

        return response

    except Exception as e:
        raise KeyError(f"Could not control if specified ticker is in stock_balancesheet_data in the database. Error: {e}") from e

def does_stock_exists_stock_cash_flow_data(stock_ticker=""):
    """
    This function controls if the specified stock exists in the stock_cash_flow_data table in the database.

    Args:
    stock_ticker: str

    Returns:
    True: bool
    False: bool

    Raises:
    - ValueError: If the stock_ticker parameter is empty.
    - KeyError: If the secrets cannot be fetched.
    - KeyError: If the connection to the database cannot be established.
    - KeyError: If the control if specified ticker is in stock_cash_flow_data in the database cannot be completed.
    """
    # Check if the stock_cash_flow_data_df parameter is empty
    if stock_ticker == "":
        raise ValueError("The stock_symbol parameter cannot be empty.")

    try:
        # Fetch the secrets from the secret_import function
        db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()

    except Exception as e:
        raise KeyError(f"Could not fetch the secrets. Error: {e}") from e

    try:
        db_con = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)

    except Exception as e:
        raise KeyError(f"Could not establish connection to the database. Error: {e}") from e

    try:
        # Fetch the stock_cash_flow_data_df from the database
        quary = f"""SELECT * FROM stock_cash_flow_data
            WHERE ticker = "{stock_ticker}"
            """
        stock_cash_flow_data_df = pd.read_sql(sql=quary, con=db_con)
        # Check if the stock_cash_flow_data_df is empty
        if stock_cash_flow_data_df.empty:
            response = False
        else:
            response = True

        return response

    except Exception as e:
        raise KeyError(f"Could not control if specified ticker is in stock_cash_flow_data in the database. Error: {e}") from e

def does_stock_exists_stock_ratio_data(stock_ticker=""):
    """
    This function controls if the specified stock exists in the stock_ratio_data table in the database.

    Args:
    stock_ticker: str

    Returns:
    True: bool
    False: bool

    Raises:
    - ValueError: If the stock_ticker parameter is empty.
    - KeyError: If the secrets cannot be fetched.
    - KeyError: If the connection to the database cannot be established.
    - KeyError: If the control if specified ticker is in stock_ratio_data in the database cannot be completed.
    """
    # Check if the stock_cash_flow_data_df parameter is empty
    if stock_ticker == "":
        raise ValueError("The stock_symbol parameter cannot be empty.")

    try:
        # Fetch the secrets from the secret_import function
        db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()

    except Exception as e:
        raise KeyError(f"Could not fetch the secrets. Error: {e}") from e

    try:
        db_con = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)

    except Exception as e:
        raise KeyError(f"Could not establish connection to the database. Error: {e}") from e

    try:
        # Fetch the stock_cash_flow_data_df from the database
        quary = f"""SELECT * FROM stock_ratio_data
            WHERE ticker = "{stock_ticker}"
            """
        stock_cash_flow_data_df = pd.read_sql(sql=quary, con=db_con)
        # Check if the stock_cash_flow_data_df is empty
        if stock_cash_flow_data_df.empty:
            response = False
        else:
            response = True

        return response

    except Exception as e:
        raise KeyError(f"Could not control if specified ticker is in stock_cash_flow_data in the database. Error: {e}") from e

def does_stock_exists_stock_prediction_data(stock_ticker=""):
    """
    This function controls if the specified stock exists in the stock_prediction_data table in the database.

    Args:
    stock_ticker: str

    Returns:
    True: bool
    False: bool

    Raises:
    - ValueError: If the stock_ticker parameter is empty.
    - KeyError: If the secrets cannot be fetched.
    - KeyError: If the connection to the database cannot be established.
    - KeyError: If the control if specified ticker is in stock_prediction_data in the database cannot be completed.
    """
    # Check if the stock_prediction_data_df parameter is empty
    if stock_ticker == "":
        raise ValueError("The stock_symbol parameter cannot be empty.")

    try:
        # Fetch the secrets from the secret_import function
        db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()

    except Exception as e:
        raise KeyError(f"Could not fetch the secrets. Error: {e}") from e

    try:
        db_con = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)

    except Exception as e:
        raise KeyError(f"Could not establish connection to the database. Error: {e}") from e

    try:
        # Fetch the stock_prediction_data_df from the database
        quary = f"""SELECT * FROM stock_prediction_data
            WHERE ticker = "{stock_ticker}"
            """
        stock_prediction_data_df = pd.read_sql(sql=quary, con=db_con)
        # Check if the stock_cash_flow_data_df is empty
        if stock_prediction_data_df.empty:
            response = False
        else:
            response = True

        return response

    except Exception as e:
        raise KeyError(f"Could not control if specified ticker is in stock_prediction_data in the database. Error: {e}") from e

def import_stock_info_data(stock_ticker=""):
    """
    This function imports the stock_info_data from the stock_info_data table in the database.
    
    Args:
    None
    
    Returns:
    stock_info_data_df: pandas DataFrame
    
    Raises:
    - KeyError: If the secrets cannot be fetched.
    - KeyError: If the connection to the database cannot be established.
    - KeyError: If the stock_info_data cannot be fetched from stock_info_data in the database to stock_info_data_df.
    """
    try:
        # Fetch the secrets from the secret_import function
        db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()

    except Exception as e:
        raise KeyError(f"Could not fetch the secrets. Error: {e}") from e

    try:
        db_con = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)

    except Exception as e:
        raise KeyError(f"Could not establish connection to the database. Error: {e}") from e

    try:
        if stock_ticker == "":
            quary = """SELECT *
                FROM stock_info_data
                """
            stock_info_data_df = pd.read_sql(sql=quary, con=db_con)
            return stock_info_data_df

    except Exception as e:
        raise KeyError(f"Could not import from stock_info_data in the database to stock_info_data_df . Error: {e}") from e

    try:
        if stock_ticker != "":
            if does_stock_exists_stock_info_data is False:
                raise ValueError("The stock does not exist in the stock_info_data table.")
            elif does_stock_exists_stock_info_data is True:
                quary = f"""SELECT *
                    FROM stock_info_data
                    WHERE ticker = "{stock_ticker}"
                    """
                stock_info_data_df = pd.read_sql(sql=quary, con=db_con)
                return stock_info_data_df

    except Exception as e:
        raise KeyError(f"Could not import from stock_info_data in the database to stock_info_data_df with a specific ticker. Error: {e}") from e

def export_stock_info_data(stock_info_data_df=""):
    """
    This function exports the stock_info_data from the stock_info_data_df to the stock_info_data table in the database.
    
    Args:
    stock_price_data_df: pandas DataFrame
    
    Returns:
    None
    
    Raises:
    - ValueError: If the stock_info_data_df parameter is empty.
    - ValueError: If the stock_info_data_df parameter contains NaN values.
    - ValueError: If the stock_info_data_df parameter exceeds more than 3 columns.
    - KeyError: If the secrets cannot be fetched.
    - KeyError: If the connection to the database cannot be established.
    - KeyError: If the stock_info_data_df cannot be exported to stock_info_data in the database.
    """
    if stock_info_data_df.empty:
        raise ValueError("The stock_info_data_df parameter cannot be empty.")

    if stock_info_data_df["ticker"].isnull().values.any():
        raise ValueError("""The stock_info_data_df parameter "ticker" cannot contain NaN values.""")

    if len(stock_info_data_df.columns) > 3:
        raise ValueError("The stock_info_data_df parameter must not exceed more than 3 columns.")

    try:
        # Fetch the secrets from the secret_import function
        db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()

    except Exception as e:
        raise KeyError(f"Could not fetch the secrets. Error: {e}") from e

    try:
        db_con = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)

    except Exception as e:
        raise KeyError(f"Could not establish connection to the database. Error: {e}") from e

    try:
        stock_info_data_df.to_sql(name="stock_info_data", con=db_con, index=False, if_exists="append")

    except Exception as e:
        raise KeyError(f"Could not export from stock_info_data_df to stock_info_data in the database. Error: {e}") from e

def import_stock_price_data(amount = 1, stock_ticker=""):
    """
    This function imports the stock_price_data from the stock_price_data table in the database.
    
    Args:
    amount: int
    stock_ticker: str

    Returns:
    stock_price_data_df: pandas DataFrame

    Raises:
    - ValueError: If the amount parameter is less than 1.
    - KeyError: If the secrets cannot be fetched.
    - KeyError: If the connection to the database cannot be established.
    - KeyError: If the stock_price_data_df cannot be imported from stock_price_data in the database to stock_price_data_df.
    - ValueError: If the stock does not exist in the stock_price_data table.
    - KeyError: If the stock_price_data_df cannot be imported from stock_price_data in the database to stock_price_data_df with a specific ticker.
    """
    if amount < 1:
        raise ValueError("The amount parameter cannot be less than 1.")

    try:
        # Fetch the secrets from the secret_import function
        db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()

    except Exception as e:
        raise KeyError(f"Could not fetch the secrets. Error: {e}") from e

    try:
        db_con = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)

    except Exception as e:
        raise KeyError(f"Could not establish connection to the database. Error: {e}") from e

    try:
        if stock_ticker == "":
            quary = f"""SELECT * FROM
            (SELECT * FROM stock_price_data
            ORDER BY date DESC LIMIT {amount}) AS temp
            ORDER BY date ASC
            """
            stock_price_data_df = pd.read_sql(sql=quary, con=db_con)
            return stock_price_data_df

    except Exception as e:
        raise KeyError(f"Could not import from stock_price_data in the database to stock_price_data_df . Error: {e}") from e

    try:
        if stock_ticker != "":
            if does_stock_exists_stock_price_data(stock_ticker) is False:
                raise ValueError("The stock does not exist in the stock_price_data table.")
            elif does_stock_exists_stock_price_data(stock_ticker) is True:
                quary = f"""SELECT * FROM
                (SELECT * FROM stock_price_data
                WHERE ticker = "{stock_ticker}"
                ORDER BY date DESC LIMIT {amount}) AS temp
                ORDER BY date ASC
                """
                stock_price_data_df = pd.read_sql(sql=quary, con=db_con)
                return stock_price_data_df

    except Exception as e:
        raise KeyError(f"Could not import from stock_price_data in the database to stock_price_data_df with a specific ticker. Error: {e}") from e

def export_stock_price_data(stock_price_data_df=""):
    """
    This function exports the stock_price_data from the stock_price_data_df to the stock_price_data table in the database.

    Args:
    stock_price_data_df: pandas DataFrame

    Returns:
    None

    Raises:
    - ValueError: If the stock_price_data_df is empty.
    - ValueError: If the stock_price_data_df parameter does not contain the column "date".
    - ValueError: If the stock_price_data_df parameter does not contain the column "ticker".
    - ValueError: If the stock_price_data_df parameter "date" contains NaN values.
    - ValueError: If the stock_price_data_df parameter "ticker" contains NaN values.
    - KeyError: If the secrets cannot be fetched.
    - KeyError: If the connection to the database cannot be established.
    - KeyError: If column cannot be dropped from stock_price_data_df.
    - KeyError: If the stock_price_data_df cannot be exported to stock_price_data in the database.
    """
    if stock_price_data_df.empty:
        raise ValueError("The stock_price_data_df cannot be empty.")

    if 'date' not in stock_price_data_df.columns:
        raise ValueError("The stock_price_data_df parameter must contain the column 'date'.")

    if 'ticker' not in stock_price_data_df.columns:
        raise ValueError("The stock_price_data_df parameter must contain the column 'ticker'.")

    if stock_price_data_df["date"].isnull().values.any():
        raise ValueError("""The stock_price_data_df parameter "date" cannot contain NaN values.""")

    if stock_price_data_df["ticker"].isnull().values.any():
        raise ValueError("""The stock_price_data_df parameter "ticker" cannot contain NaN values.""")

    try:
        # Fetch the secrets from the secret_import function
        db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()

    except Exception as e:
        raise KeyError(f"Could not fetch the secrets. Error: {e}") from e

    try:
        db_con = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)

    except Exception as e:
        raise KeyError(f"Could not establish connection to the database. Error: {e}") from e

    try:
        # Step 1: Rename columns to match database schema (case-sensitive) BEFORE removing duplicates
        # This ensures we keep the right version of each column
        column_rename_map = {
            'RSI_14': 'rsi_14',
            'ATR_14': 'atr_14',
            'ATRr_14': 'atr_14',  # Handle both variations
            'ATRl_14': 'atr_14'   # Handle all ATR variations
        }
        stock_price_data_df = stock_price_data_df.rename(columns=column_rename_map)
        
        # Step 2: Remove duplicate columns (keep first occurrence) AFTER renaming
        stock_price_data_df = stock_price_data_df.loc[:, ~stock_price_data_df.columns.duplicated()]
        
        # Step 3: Define all expected database columns (from ddl.sql)
        db_columns = [
            'date', 'ticker', 'currency', 'trade_Volume', 
            'open_Price', 'high_Price', 'low_Price', 'close_Price',
            '1D', '1M', '3M', '6M', '9M', '1Y', '2Y', '3Y', '4Y', '5Y',
            'sma_5', 'sma_20', 'sma_40', 'sma_120', 'sma_200',
            'ema_5', 'ema_20', 'ema_40', 'ema_120', 'ema_200',
            'std_Div_5', 'std_Div_20', 'std_Div_40', 'std_Div_120', 'std_Div_200',
            'bollinger_Band_5_2STD', 'bollinger_Band_20_2STD', 'bollinger_Band_40_2STD',
            'bollinger_Band_120_2STD', 'bollinger_Band_200_2STD',
            'momentum', 'rsi_14', 'atr_14', 'macd', 'macd_signal', 'macd_histogram',
            'volume_sma_20', 'volume_ema_20', 'volume_ratio', 'vwap', 'obv',
            'volatility_5d', 'volatility_20d', 'volatility_60d'
        ]
        
        # Step 4: Add missing columns with None/NULL values
        for col in db_columns:
            if col not in stock_price_data_df.columns:
                stock_price_data_df[col] = None
        
        # Step 5: Select only database columns in correct order
        stock_price_data_df = stock_price_data_df[db_columns]
        
    except Exception as e:
        raise KeyError(f"Could not prepare stock_price_data_df columns for database export. Error: {e}") from e

    try:
        # Use UPSERT (INSERT ... ON DUPLICATE KEY UPDATE) to handle existing records
        from sqlalchemy import text
        
        # For better performance with many rows, use batch insert with duplicate handling
        ticker = stock_price_data_df['ticker'].iloc[0] if len(stock_price_data_df) > 0 else ''
        min_date = stock_price_data_df['date'].min()
        max_date = stock_price_data_df['date'].max()
        
        # Delete existing records in the date range for this ticker
        with db_con.begin() as connection:
            delete_query = text("""
                DELETE FROM stock_price_data 
                WHERE ticker = :ticker 
                AND date >= :min_date 
                AND date <= :max_date
            """)
            connection.execute(delete_query, {
                'ticker': ticker,
                'min_date': min_date,
                'max_date': max_date
            })
        
        # Now insert the new data
        stock_price_data_df.to_sql(name="stock_price_data", con=db_con, index=False, if_exists="append")
        print(f"✓ Exported {len(stock_price_data_df)} price records for {ticker}")

    except Exception as e:
        raise KeyError(f"Could not export from stock_price_data_df to stock_price_data in the database. Error: {e}") from e

def import_stock_financial_data(amount = 1, stock_ticker=""):
    """
    This function imports the stock_financial_data from the stock_income_stmt_data, stock_balancesheet_data and stock_cash_flow_data tables in the database.

    Args:
    amount: int
    stock_ticker: str

    Returns:
    stock_financial_data_df: pandas DataFrame

    Raises:
    - ValueError: If the amount parameter is less than 1.
    - KeyError: If the secrets cannot be fetched.
    - KeyError: If the connection to the database cannot be established.
    - KeyError: If the stock_income_stmt_data_df cannot be created from stock_price_data_df.
    - KeyError: If the stock_balancesheet_data_df cannot be created from stock_price_data_df.
    - KeyError: If the stock_cash_flow_data_df cannot be created from stock_price_data_df.
    - KeyError: If the stock_financial_data_df cannot be imported from stock_income_stmt_data, stock_balancesheet_data and stock_cash_flow_data in the database.
    - ValueError: If the stock does not exist in the stock_income_stmt_data table.
    - KeyError: If the stock_financial_data_df cannot be imported from stock_income_stmt_data, stock_balancesheet_data and stock_cash_flow_data in the database with a specific ticker.
    """
    try:
        # Fetch the secrets from the secret_import function
        db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()

    except Exception as e:
        raise KeyError(f"Could not fetch the secrets. Error: {e}") from e

    try:
        db_con = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)

    except Exception as e:
        raise KeyError(f"Could not establish connection to the database. Error: {e}") from e

    if amount < 1:
        raise ValueError("The amount parameter cannot be less than 1.")

    try:
        if stock_ticker == "":
            income_stmt_quary = f"""SELECT * FROM
                (SELECT * FROM stock_income_stmt_data
                ORDER BY financial_Statement_Date DESC LIMIT {amount}) AS temp
                ORDER BY financial_Statement_Date ASC
                """
            balancesheet_quary = f"""SELECT * FROM
                (SELECT * FROM stock_balancesheet_data
                ORDER BY financial_Statement_Date DESC LIMIT {amount}) AS temp
                ORDER BY financial_Statement_Date ASC
                """
            cash_flow_quary = f"""SELECT * FROM
                (SELECT * FROM stock_cash_flow_data
                ORDER BY financial_Statement_Date DESC LIMIT {amount}) AS temp
                ORDER BY financial_Statement_Date ASC
                """
            stock_income_stmt_data_df = pd.read_sql(sql=income_stmt_quary, con=db_con)
            stock_balancesheet_data_df = pd.read_sql(sql=balancesheet_quary, con=db_con)
            stock_cash_flow_data_df = pd.read_sql(sql=cash_flow_quary, con=db_con)
            stock_financial_data_df = pd.merge(stock_income_stmt_data_df, stock_balancesheet_data_df, on=["financial_Statement_Date", "ticker"])
            stock_financial_data_df = pd.merge(stock_financial_data_df, stock_cash_flow_data_df, on=["financial_Statement_Date", "ticker"])
            stock_financial_data_df = stock_financial_data_df.rename(columns={"financial_Statement_Date": "date"})
            return stock_financial_data_df

    except Exception as e:
        raise KeyError(f"Could not import from stock_income_stmt_data, stock_balancesheet_data and stock_cash_flow_data in the database to stock_financial_data_df. Error: {e}") from e

    try:
        if stock_ticker != "":
            if does_stock_exists_stock_income_stmt_data(stock_ticker) is False:
                raise ValueError("The stock does not exist in the stock_income_stmt_data table.")
            elif does_stock_exists_stock_income_stmt_data(stock_ticker) is True:
                income_stmt_quary = f"""SELECT * FROM
                    (SELECT * FROM stock_income_stmt_data
                    WHERE ticker = "{stock_ticker}"
                    ORDER BY financial_Statement_Date DESC LIMIT {amount}
                    ) AS temp
                    ORDER BY financial_Statement_Date ASC
                    """
                balancesheet_quary = f"""SELECT * FROM
                    (SELECT * FROM stock_balancesheet_data
                    WHERE ticker = "{stock_ticker}"
                    ORDER BY financial_Statement_Date DESC LIMIT {amount}
                    ) AS temp
                    ORDER BY financial_Statement_Date ASC
                    """
                cash_flow_quary = f"""SELECT * FROM
                    (SELECT * FROM stock_cash_flow_data
                    WHERE ticker = "{stock_ticker}"
                    ORDER BY financial_Statement_Date DESC LIMIT {amount}
                    ) AS temp
                    ORDER BY financial_Statement_Date ASC
                    """
                stock_income_stmt_data_df = pd.read_sql(sql=income_stmt_quary, con=db_con)
                stock_balancesheet_data_df = pd.read_sql(sql=balancesheet_quary, con=db_con)
                stock_cash_flow_data_df = pd.read_sql(sql=cash_flow_quary, con=db_con)
                stock_financial_data_df = pd.merge(stock_income_stmt_data_df, stock_balancesheet_data_df, on=["financial_Statement_Date", "ticker"])
                stock_financial_data_df = pd.merge(stock_financial_data_df, stock_cash_flow_data_df, on=["financial_Statement_Date", "ticker"])
                stock_financial_data_df = stock_financial_data_df.rename(columns={"financial_Statement_Date": "date"})
                return stock_financial_data_df

    except Exception as e:
        raise KeyError(f"Could not import from stock_income_stmt_data, stock_balancesheet_data and stock_cash_flow_data in the database to stock_financial_data_df with a specific ticker. Error: {e}") from e

def export_stock_financial_data(stock_financial_data_df=""):
    """
    This function exports the stock_financial_data from the stock_financial_data_df to the stock_financial_data table in the database.

    Args:
    stock_financial_data_df: pandas DataFrame

    Returns:
    None

    Raises:
    - ValueError: If the stock_financial_data_df parameter is empty.
    - ValueError: If the stock_financial_data_df parameter "date" contains NaN values.
    - ValueError: If the stock_financial_data_df parameter "ticker" contains NaN values.
    - KeyError: If the secrets cannot be fetched.
    - KeyError: If the connection to the database cannot be established.
    - KeyError: If the stock_income_stmt_data_df cannot be created from stock_price_data_df.
    - KeyError: If the stock_balancesheet_data_df cannot be created from stock_price_data_df.
    - KeyError: If the stock_cash_flow_data_df cannot be created from stock_price_data_df.
    - KeyError: If the stock_financial_data_df cannot be exported to stock_income_stmt_data, stock_balancesheet_data and stock_cash_flow_data in the database.
    """
    if stock_financial_data_df.empty:
        raise ValueError("The stock_financial_data_df parameter cannot be empty.")

    if stock_financial_data_df["date"].isnull().values.any():
        raise ValueError("""The stock_financial_data_df parameter "date" cannot contain NaN values.""")

    if stock_financial_data_df["ticker"].isnull().values.any():
        raise ValueError("""The stock_financial_data_df parameter "ticker" cannot contain NaN values.""")

    try:
        # Fetch the secrets from the secret_import function
        db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()

    except Exception as e:
        raise KeyError(f"Could not fetch the secrets. Error: {e}") from e

    try:
        db_con = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)

    except Exception as e:
        raise KeyError(f"Could not establish connection to the database. Error: {e}") from e

    try:
        income_stmt_columns = ["date_published", "revenue", "revenue_Growth", "gross_Profit",
            "gross_Profit_Growth", "gross_Margin", "gross_Margin_Growth", "operating_Earning",
            "operating_Earning_Growth", "operating_Earning_Margin", "operating_Earning_Margin_Growth",
            "net_Income", "net_Income_Growth", "net_Income_Margin", "net_Income_Margin_Growth", "eps",
            "eps_Growth", "average_shares"
            ]
        for column in income_stmt_columns.copy():
            if column not in stock_financial_data_df.columns:
                income_stmt_columns.remove(column)

        income_stmt_columns.insert(0, 'date')
        income_stmt_columns.insert(2, 'ticker')
        stock_income_stmt_data_df = stock_financial_data_df[income_stmt_columns]
        stock_income_stmt_data_df = stock_income_stmt_data_df.rename(columns={"date": "financial_Statement_Date"})

    except Exception as e:
        raise KeyError(f"Could not create stock_income_stmt_data_df from stock_price_data_df. Error: {e}") from e

    try:
        balancesheet_columns = ["date_published", "total_Assets", "total_Assets_Growth",
            "current_Assets", "current_Assets_Growth", "cash_And_Cash_Equivalents",
            "cash_And_Cash_Equivalents_Growth", "equity", "equity_Growth", "liabilities",
            "liabilities_Growth", "current_Liabilities", "current_Liabilities_Growth",
            "book_Value", "book_Value_Growth", "book_Value_Per_Share", "book_Value_Per_Share_Growth",
            "return_On_Assets", "return_On_Assets_Growth", "return_On_Equity", "return_On_Equity_Growth",
            "current_Ratio", "current_Ratio_Growth", "quick_Ratio", "quick_Ratio_Growth", "debt_To_Equity",
            "debt_To_Equity_Growth"
            ]
        for column in balancesheet_columns.copy():
            if column not in stock_financial_data_df.columns:
                balancesheet_columns.remove(column)

        balancesheet_columns.insert(0, 'date')
        balancesheet_columns.insert(2, 'ticker')
        stock_balancesheet_data_df = stock_financial_data_df[balancesheet_columns]
        stock_balancesheet_data_df = stock_balancesheet_data_df.rename(columns={"date": "financial_Statement_Date"})

    except Exception as e:
        raise KeyError(f"Could not create stock_balancesheet_data_df from stock_price_data_df. Error: {e}") from e

    try:
        cash_flow_columns = ["date_published", "free_Cash_Flow", "free_Cash_Flow_Growth",
            "free_Cash_Flow_Per_Share", "free_Cash_Flow_Per_Share_Growth"
            ]
        for column in cash_flow_columns.copy():
            if column not in stock_financial_data_df.columns:
                cash_flow_columns.remove(column)

        cash_flow_columns.insert(0, 'date')
        cash_flow_columns.insert(2, 'ticker')
        stock_cash_flow_data_df = stock_financial_data_df[cash_flow_columns]
        stock_cash_flow_data_df = stock_cash_flow_data_df.rename(columns={"date": "financial_Statement_Date"})

    except Exception as e:
        raise KeyError(f"Could not create stock_cash_flow_data_df from stock_price_data_df. Error: {e}") from e

    try:
        from sqlalchemy import text
        
        ticker = stock_income_stmt_data_df['ticker'].iloc[0] if len(stock_income_stmt_data_df) > 0 else ''
        
        # Delete existing financial data for this ticker, then insert new data
        with db_con.begin() as connection:
            # Delete from income statement table
            connection.execute(text("""
                DELETE FROM stock_income_stmt_data WHERE ticker = :ticker
            """), {'ticker': ticker})
            
            # Delete from balance sheet table
            connection.execute(text("""
                DELETE FROM stock_balancesheet_data WHERE ticker = :ticker
            """), {'ticker': ticker})
            
            # Delete from cash flow table
            connection.execute(text("""
                DELETE FROM stock_cash_flow_data WHERE ticker = :ticker
            """), {'ticker': ticker})
        
        # Now insert the new data
        stock_income_stmt_data_df.to_sql(name="stock_income_stmt_data", con=db_con, index=False, if_exists="append")
        stock_balancesheet_data_df.to_sql(name="stock_balancesheet_data", con=db_con, index=False, if_exists="append")
        stock_cash_flow_data_df.to_sql(name="stock_cash_flow_data", con=db_con, index=False, if_exists="append")
        print(f"✓ Exported financial data for {ticker}")

    except Exception as e:
        raise KeyError(f"Could not export from stock_financial_data_df to stock_income_stmt_data, stock_balancesheet_data and stock_cash_flow_data in the database. Error: {e}") from e

def import_stock_ratio_data(amount = 1, stock_ticker=""):
    """
    This function imports the stock_ratio_data from the stock_ratio_data table in the database.
    
    Args:
    amount: int
    Stock_ticker: str

    Returns:
    stock_ratio_data_df: pandas DataFrame

    Raises:
    - ValueError: If the amount parameter is less than 1.
    - KeyError: If the secrets cannot be fetched.
    - KeyError: If the connection to the database cannot be established.
    - KeyError: If the stock_ratio_data cannot be fetched from stock_ratio_data_df in the database to stock_ratio_data_df.    
    - ValueError: If the stock does not exist in the stock_ratio_data table.
    - KeyError: If the stock_ratio_data_df cannot be imported from stock_ratio_data in the database to stock_ratio_data_df with a specific ticker.
    """

    if amount < 1:
        raise ValueError("The amount parameter cannot be less than 1.")

    try:
        # Fetch the secrets from the secret_import function
        db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()

    except Exception as e:
        raise KeyError(f"Could not fetch the secrets. Error: {e}") from e

    try:
        db_con = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)

    except Exception as e:
        raise KeyError(f"Could not establish connection to the database. Error: {e}") from e

    try:
        if stock_ticker == "":
            quary = f"""SELECT * FROM
            (SELECT * FROM stock_ratio_data
            ORDER BY date DESC LIMIT {amount}) AS temp
            ORDER BY date ASC
            """
            stock_ratio_data_df = pd.read_sql(sql=quary, con=db_con)
            return stock_ratio_data_df

    except Exception as e:
        raise KeyError(f"Could not import from stock_ratio_data in the database to stock_ratio_data_df. Error: {e}") from e

    try:
        if stock_ticker != "":
            if does_stock_exists_stock_ratio_data(stock_ticker) is False:
                raise ValueError("The stock does not exist in the stock_ratio_data table.")
            elif does_stock_exists_stock_ratio_data(stock_ticker) is True:
                quary = f"""SELECT * FROM
                (SELECT * FROM stock_ratio_data
                WHERE ticker = "{stock_ticker}"
                ORDER BY date DESC LIMIT {amount}) AS temp
                ORDER BY date ASC
                """
                stock_ratio_data_df = pd.read_sql(sql=quary, con=db_con)
                return stock_ratio_data_df

    except Exception as e:
        raise KeyError(f"Could not import from stock_ratio_data in the database to stock_ratio_data_df with a specific ticker. Error: {e}") from e

def export_stock_ratio_data(stock_ratio_data_df=""):
    """
    This function exports the stock_ratio_data from the stock_ratio_data_df to the stock_ratio_data table in the database.

    Args:
    stock_ratio_data_df: pandas DataFrame

    Returns:
    None

    Raises:
    - ValueError: If the stock_ratio_data_df parameter is empty.
    - ValueError: If the stock_ratio_data_df parameter does not contain the column "date".
    - ValueError: If the stock_ratio_data_df parameter does not contain the column "ticker".
    - ValueError: If the stock_ratio_data_df parameter "date" contains NaN values.
    - ValueError: If the stock_ratio_data_df parameter "ticker" contains NaN values.
    - KeyError: If the secrets cannot be fetched.
    - KeyError: If the connection to the database cannot be established.
    - KeyError: If the stock_ratio_data_df cannot be exported to stock_ratio_data in the database.
    """
    if stock_ratio_data_df.empty:
        raise ValueError("The stock_ratio_data_df parameter cannot be empty.")

    if 'date' not in stock_ratio_data_df.columns:
        raise ValueError("The stock_ratio_data_df parameter must contain the column 'date'.")

    if 'ticker' not in stock_ratio_data_df.columns:
        raise ValueError("The stock_ratio_data_df parameter must contain the column 'ticker'.")

    if stock_ratio_data_df["date"].isnull().values.any():
        raise ValueError("""The stock_ratio_data_df parameter "date" cannot contain NaN values.""")

    if stock_ratio_data_df["ticker"].isnull().values.any():
        raise ValueError("""The stock_ratio_data_df parameter "ticker" cannot contain NaN values.""")

    try:
        # Fetch the secrets from the secret_import function
        db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()

    except Exception as e:
        raise KeyError(f"Could not fetch the secrets. Error: {e}") from e

    try:
        db_con = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)

    except Exception as e:
        raise KeyError(f"Could not establish connection to the database. Error: {e}") from e

    try:
        # Remove duplicates within the DataFrame itself (keep last occurrence)
        stock_ratio_data_df = stock_ratio_data_df.drop_duplicates(subset=['date', 'ticker'], keep='last')
        
        # Check which dates already exist in the database and only insert new ones
        if len(stock_ratio_data_df) > 0:
            ticker = stock_ratio_data_df['ticker'].iloc[0]
            
            # Convert all dates to datetime.date for consistent comparison
            stock_ratio_data_df['date'] = pd.to_datetime(stock_ratio_data_df['date']).dt.date
            
            # Get existing dates for this ticker in the date range
            from sqlalchemy import text
            min_date = stock_ratio_data_df['date'].min()
            max_date = stock_ratio_data_df['date'].max()
            
            existing_query = text("""
                SELECT date FROM stock_ratio_data 
                WHERE ticker = :ticker 
                AND date >= :min_date 
                AND date <= :max_date
            """)
            
            existing_dates = pd.read_sql(existing_query, db_con, params={
                'ticker': ticker,
                'min_date': str(min_date),  # Convert to string for SQL
                'max_date': str(max_date)
            })
            
            if not existing_dates.empty:
                # Convert to set for fast lookup
                existing_date_set = set(pd.to_datetime(existing_dates['date']).dt.date)
                # Filter to only new records
                stock_ratio_data_df = stock_ratio_data_df[~stock_ratio_data_df['date'].isin(existing_date_set)]
                
                if stock_ratio_data_df.empty:
                    print(f"   ↳ No new ratio records to insert (all {len(existing_date_set)} dates already exist)")
                    return
        
    except Exception as e:
        print(f"⚠️  WARNING: Could not check existing records: {e}")
        # Continue anyway - duplicates will be caught by database constraints if any

    try:
        stock_ratio_data_df.to_sql(name="stock_ratio_data", con=db_con, index=False, if_exists="append")

    except Exception as e:
        raise KeyError(f"Could not export from stock_ratio_data_df to stock_price_data in the database. Error: {e}") from e


def delete_stock_ratio_data_from_date(stock_ticker: str, from_date: str) -> int:
    """
    Delete ratio data for a ticker from a specific date onwards.
    
    This is used when new financial data is available and ratios need to be recalculated.
    
    Args:
        stock_ticker: Stock ticker symbol
        from_date: Date string (YYYY-MM-DD) from which to delete ratios (inclusive)
    
    Returns:
        Number of rows deleted
    
    Raises:
        - ValueError: If stock_ticker or from_date is empty
        - KeyError: If database connection fails
    """
    if not stock_ticker:
        raise ValueError("stock_ticker cannot be empty")
    if not from_date:
        raise ValueError("from_date cannot be empty")
    
    try:
        db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()
    except Exception as e:
        raise KeyError(f"Could not fetch the secrets. Error: {e}") from e
    
    try:
        db_con = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)
    except Exception as e:
        raise KeyError(f"Could not establish connection to the database. Error: {e}") from e
    
    try:
        from sqlalchemy import text
        
        # First count how many rows will be deleted
        count_query = text("""
            SELECT COUNT(*) as count FROM stock_ratio_data 
            WHERE ticker = :ticker AND date >= :from_date
        """)
        result = pd.read_sql(count_query, db_con, params={
            'ticker': stock_ticker,
            'from_date': from_date
        })
        rows_to_delete = result['count'].iloc[0]
        
        if rows_to_delete > 0:
            # Delete the rows
            delete_query = text("""
                DELETE FROM stock_ratio_data 
                WHERE ticker = :ticker AND date >= :from_date
            """)
            with db_con.connect() as conn:
                conn.execute(delete_query, {'ticker': stock_ticker, 'from_date': from_date})
                conn.commit()
        
        return rows_to_delete
        
    except Exception as e:
        raise KeyError(f"Could not delete ratio data for {stock_ticker} from {from_date}. Error: {e}") from e


def get_newest_financial_date(stock_ticker: str, include_quarterly: bool = True) -> tuple:
    """
    Get the newest financial statement date for a ticker.
    
    Checks both annual and quarterly financial data to find the most recent
    fiscal period end date.
    
    Args:
        stock_ticker: Stock ticker symbol
        include_quarterly: Whether to include quarterly data in the check
    
    Returns:
        Tuple of (newest_date, source) where source is 'annual' or 'quarterly'
        Returns (None, None) if no financial data exists
    
    Raises:
        - KeyError: If database connection fails
    """
    try:
        db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()
    except Exception as e:
        raise KeyError(f"Could not fetch the secrets. Error: {e}") from e
    
    try:
        db_con = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)
    except Exception as e:
        raise KeyError(f"Could not establish connection to the database. Error: {e}") from e
    
    newest_date = None
    source = None
    
    try:
        # Check annual data
        annual_query = f"""
            SELECT MAX(financial_Statement_Date) as newest_date 
            FROM stock_income_stmt_data 
            WHERE ticker = '{stock_ticker}'
        """
        annual_result = pd.read_sql(sql=annual_query, con=db_con)
        annual_date = annual_result['newest_date'].iloc[0]
        
        if annual_date is not None:
            newest_date = pd.to_datetime(annual_date).date()
            source = 'annual'
        
        # Check quarterly data if requested
        if include_quarterly:
            quarterly_query = f"""
                SELECT MAX(fiscal_quarter_end) as newest_date 
                FROM stock_income_stmt_quarterly 
                WHERE ticker = '{stock_ticker}'
            """
            try:
                quarterly_result = pd.read_sql(sql=quarterly_query, con=db_con)
                quarterly_date = quarterly_result['newest_date'].iloc[0]
                
                if quarterly_date is not None:
                    quarterly_date = pd.to_datetime(quarterly_date).date()
                    if newest_date is None or quarterly_date > newest_date:
                        newest_date = quarterly_date
                        source = 'quarterly'
            except Exception:
                # Quarterly table might not exist
                pass
        
        return newest_date, source
        
    except Exception as e:
        print(f"Warning: Could not get newest financial date for {stock_ticker}: {e}")
        return None, None


def get_last_ratio_financial_date(stock_ticker: str) -> tuple:
    """
    Get the financial_date_used from the most recent ratio record for a ticker.
    
    This helps determine if new financial data is available that requires
    ratio recalculation.
    
    Args:
        stock_ticker: Stock ticker symbol
    
    Returns:
        Tuple of (last_ratio_date, financial_date_used)
        Returns (None, None) if no ratio data exists
    
    Raises:
        - KeyError: If database connection fails
    """
    try:
        db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()
    except Exception as e:
        raise KeyError(f"Could not fetch the secrets. Error: {e}") from e
    
    try:
        db_con = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)
    except Exception as e:
        raise KeyError(f"Could not establish connection to the database. Error: {e}") from e
    
    try:
        query = f"""
            SELECT date, financial_date_used 
            FROM stock_ratio_data 
            WHERE ticker = '{stock_ticker}' 
            ORDER BY date DESC 
            LIMIT 1
        """
        result = pd.read_sql(sql=query, con=db_con)
        
        if result.empty:
            return None, None
        
        last_ratio_date = pd.to_datetime(result['date'].iloc[0]).date()
        financial_date_used = result['financial_date_used'].iloc[0]
        
        if financial_date_used is not None:
            financial_date_used = pd.to_datetime(financial_date_used).date()
        
        return last_ratio_date, financial_date_used
        
    except Exception as e:
        print(f"Warning: Could not get last ratio financial date for {stock_ticker}: {e}")
        return None, None


def get_all_financial_dates(stock_ticker: str, include_quarterly: bool = True) -> pd.DataFrame:
    """
    Get all financial statement dates for a ticker, sorted oldest to newest.
    
    This is used for initial population to calculate ratios from the earliest
    financial statement onwards.
    
    Args:
        stock_ticker: Stock ticker symbol
        include_quarterly: Whether to include quarterly data
    
    Returns:
        DataFrame with columns ['date', 'source'] sorted by date ascending
    
    Raises:
        - KeyError: If database connection fails
    """
    try:
        db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()
    except Exception as e:
        raise KeyError(f"Could not fetch the secrets. Error: {e}") from e
    
    try:
        db_con = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)
    except Exception as e:
        raise KeyError(f"Could not establish connection to the database. Error: {e}") from e
    
    all_dates = []
    
    try:
        # Get annual dates
        annual_query = f"""
            SELECT DISTINCT financial_Statement_Date as date, 'annual' as source
            FROM stock_income_stmt_data 
            WHERE ticker = '{stock_ticker}'
        """
        annual_result = pd.read_sql(sql=annual_query, con=db_con)
        if not annual_result.empty:
            all_dates.append(annual_result)
        
        # Get quarterly dates if requested
        if include_quarterly:
            quarterly_query = f"""
                SELECT DISTINCT fiscal_quarter_end as date, 'quarterly' as source
                FROM stock_income_stmt_quarterly 
                WHERE ticker = '{stock_ticker}'
            """
            try:
                quarterly_result = pd.read_sql(sql=quarterly_query, con=db_con)
                if not quarterly_result.empty:
                    all_dates.append(quarterly_result)
            except Exception:
                # Quarterly table might not exist
                pass
        
        if not all_dates:
            return pd.DataFrame(columns=['date', 'source'])
        
        result = pd.concat(all_dates, ignore_index=True)
        result['date'] = pd.to_datetime(result['date'])
        result = result.drop_duplicates(subset=['date']).sort_values('date')
        
        return result.reset_index(drop=True)
        
    except Exception as e:
        print(f"Warning: Could not get financial dates for {stock_ticker}: {e}")
        return pd.DataFrame(columns=['date', 'source'])


# Define valid columns for quarterly tables based on database schema
QUARTERLY_INCOME_COLUMNS = [
    'fiscal_quarter_end', 'fiscal_year', 'fiscal_quarter', 'date_published', 'ticker',
    'revenue_q', 'gross_profit_q', 'operating_income_q', 'net_income_q',
    'eps_basic_q', 'eps_diluted_q', 'shares_diluted', 'ebitda_q',
    'revenue_ttm', 'gross_profit_ttm', 'operating_income_ttm', 'net_income_ttm',
    'eps_basic_ttm', 'eps_diluted_ttm', 'ebitda_ttm',
    'gross_margin_ttm', 'operating_margin_ttm', 'net_margin_ttm', 'ebitda_margin_ttm',
    'revenue_growth_yoy', 'gross_profit_growth_yoy', 'operating_income_growth_yoy',
    'net_income_growth_yoy', 'eps_growth_yoy', 'revenue_growth_qoq', 'net_income_growth_qoq'
]

QUARTERLY_BALANCESHEET_COLUMNS = [
    'fiscal_quarter_end', 'fiscal_year', 'fiscal_quarter', 'date_published', 'ticker',
    'total_assets', 'current_assets', 'cash_and_equivalents', 'cash_and_investments',
    'accounts_receivable', 'inventory', 'goodwill', 'intangible_assets',
    'total_liabilities', 'current_liabilities', 'accounts_payable',
    'total_debt', 'short_term_debt', 'long_term_debt',
    'total_equity', 'retained_earnings',
    'current_ratio', 'quick_ratio', 'cash_ratio', 'debt_to_equity', 'debt_to_assets',
    'book_value_per_share', 'tangible_book_per_share',
    'roa_ttm', 'roe_ttm', 'roic_ttm',
    'assets_growth_yoy', 'equity_growth_yoy', 'book_value_growth_yoy'
]

QUARTERLY_CASHFLOW_COLUMNS = [
    'fiscal_quarter_end', 'fiscal_year', 'fiscal_quarter', 'date_published', 'ticker',
    'operating_cash_flow_q', 'capex_q', 'free_cash_flow_q',
    'investing_cash_flow_q', 'financing_cash_flow_q',
    'dividends_paid_q', 'share_repurchases_q',
    'operating_cash_flow_ttm', 'capex_ttm', 'free_cash_flow_ttm',
    'dividends_paid_ttm', 'share_repurchases_ttm',
    'fcf_per_share_ttm', 'ocf_per_share_ttm',
    'fcf_margin_ttm', 'capex_to_revenue_ttm', 'fcf_conversion_ttm',
    'fcf_growth_yoy', 'ocf_growth_yoy'
]


def _filter_to_valid_columns(df, valid_columns):
    """Filter DataFrame to only include columns that exist in the database schema."""
    existing_valid = [col for col in df.columns if col in valid_columns]
    return df[existing_valid].copy()


# ============================================
# QUARTERLY FETCH METADATA FUNCTIONS
# ============================================
# Functions to track when quarterly data was last fetched
# Used to implement smart caching and reduce unnecessary API calls

def get_quarterly_fetch_metadata(ticker: str) -> dict:
    """
    Get quarterly fetch metadata for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary with keys: last_fetch_date, last_quarter_end, quarters_count
        Returns None values if no metadata exists
    """
    try:
        db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()
        db_con = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)
        
        query = f"""
            SELECT last_fetch_date, last_quarter_end, quarters_count
            FROM quarterly_fetch_metadata 
            WHERE ticker = '{ticker}'
        """
        df = pd.read_sql(sql=query, con=db_con)
        
        if df.empty:
            return {
                'last_fetch_date': None,
                'last_quarter_end': None,
                'quarters_count': 0
            }
        
        row = df.iloc[0]
        return {
            'last_fetch_date': pd.to_datetime(row['last_fetch_date']).date() if pd.notna(row['last_fetch_date']) else None,
            'last_quarter_end': pd.to_datetime(row['last_quarter_end']).date() if pd.notna(row['last_quarter_end']) else None,
            'quarters_count': int(row['quarters_count']) if pd.notna(row['quarters_count']) else 0
        }
        
    except Exception as e:
        # Table might not exist yet - return empty metadata
        print(f"   ↳ Note: Could not get quarterly fetch metadata: {e}")
        return {
            'last_fetch_date': None,
            'last_quarter_end': None,
            'quarters_count': 0
        }


def update_quarterly_fetch_metadata(ticker: str, last_quarter_end=None, quarters_count: int = 0) -> bool:
    """
    Update quarterly fetch metadata after a successful fetch.
    
    Args:
        ticker: Stock ticker symbol
        last_quarter_end: Most recent fiscal quarter end date (datetime.date or None)
        quarters_count: Number of quarterly records in database
        
    Returns:
        True if update was successful
    """
    from datetime import date
    
    try:
        db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()
        db_con = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)
        
        from sqlalchemy import text
        
        today = date.today()
        last_quarter_str = f"'{last_quarter_end}'" if last_quarter_end else "NULL"
        
        with db_con.begin() as connection:
            # Use INSERT ... ON DUPLICATE KEY UPDATE for upsert
            connection.execute(text(f"""
                INSERT INTO quarterly_fetch_metadata 
                    (ticker, last_fetch_date, last_quarter_end, quarters_count)
                VALUES 
                    ('{ticker}', '{today}', {last_quarter_str}, {quarters_count})
                ON DUPLICATE KEY UPDATE
                    last_fetch_date = '{today}',
                    last_quarter_end = {last_quarter_str},
                    quarters_count = {quarters_count}
            """))
        
        return True
        
    except Exception as e:
        print(f"   ↳ Warning: Could not update quarterly fetch metadata: {e}")
        return False


def should_fetch_quarterly_data(ticker: str, max_days_since_fetch: int = 30, 
                                 max_days_since_quarter: int = 100) -> tuple:
    """
    Determine if quarterly data should be fetched from yfinance API.
    
    Implements smart caching logic:
    - If never fetched before: fetch
    - If last fetch was > max_days_since_fetch days ago: fetch  
    - If most recent quarter is > max_days_since_quarter days old: fetch
    - Otherwise: skip fetch and use existing database data
    
    Args:
        ticker: Stock ticker symbol
        max_days_since_fetch: Maximum days since last API fetch (default: 30)
        max_days_since_quarter: Maximum days since most recent quarter end (default: 100)
            100 days allows for ~45 day reporting delay + buffer
        
    Returns:
        Tuple of (should_fetch: bool, reason: str)
    """
    from datetime import date
    
    metadata = get_quarterly_fetch_metadata(ticker)
    today = date.today()
    
    # Never fetched - must fetch
    if metadata['last_fetch_date'] is None:
        return True, "never fetched before"
    
    days_since_fetch = (today - metadata['last_fetch_date']).days
    
    # Check if it's been too long since last fetch
    if days_since_fetch >= max_days_since_fetch:
        return True, f"last fetch was {days_since_fetch} days ago (threshold: {max_days_since_fetch})"
    
    # Check if we have recent quarter data
    if metadata['last_quarter_end'] is None:
        return True, "no quarter data in database"
    
    days_since_quarter = (today - metadata['last_quarter_end']).days
    
    # If most recent quarter is too old, a new quarter might be available
    if days_since_quarter >= max_days_since_quarter:
        return True, f"most recent quarter is {days_since_quarter} days old (threshold: {max_days_since_quarter})"
    
    # Check if we have enough quarters for TTM (at least 4)
    if metadata['quarters_count'] < 4:
        return True, f"only {metadata['quarters_count']} quarters in database (need 4 for TTM)"
    
    # All checks passed - no need to fetch
    return False, f"data is fresh (fetched {days_since_fetch} days ago, latest quarter: {metadata['last_quarter_end']})"


def get_existing_quarterly_dates(ticker: str, table_type: str = 'income') -> list:
    """
    Get list of fiscal quarter end dates already in database for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        table_type: 'income', 'balancesheet', or 'cashflow'
        
    Returns:
        List of datetime.date objects for existing quarters
    """
    table_map = {
        'income': 'stock_income_stmt_quarterly',
        'balancesheet': 'stock_balancesheet_quarterly',
        'cashflow': 'stock_cashflow_quarterly'
    }
    table_name = table_map.get(table_type, 'stock_income_stmt_quarterly')
    
    try:
        db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()
        db_con = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)
        
        query = f"""
            SELECT DISTINCT fiscal_quarter_end 
            FROM {table_name} 
            WHERE ticker = '{ticker}'
            ORDER BY fiscal_quarter_end
        """
        df = pd.read_sql(sql=query, con=db_con)
        
        if df.empty:
            return []
        
        # Convert to list of dates
        dates = pd.to_datetime(df['fiscal_quarter_end']).dt.date.tolist()
        return dates
        
    except Exception as e:
        print(f"Warning: Could not check existing quarterly dates: {e}")
        return []


def import_quarterly_income_data(ticker: str) -> pd.DataFrame:
    """
    Import quarterly income statement data from database for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        DataFrame with quarterly income data, sorted by fiscal_quarter_end
    """
    try:
        db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()
        db_con = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)
        
        query = f"""
            SELECT * FROM stock_income_stmt_quarterly 
            WHERE ticker = '{ticker}'
            ORDER BY fiscal_quarter_end DESC
        """
        df = pd.read_sql(sql=query, con=db_con)
        return df
        
    except Exception as e:
        print(f"Warning: Could not import quarterly income data: {e}")
        return pd.DataFrame()


def import_quarterly_balancesheet_data(ticker: str) -> pd.DataFrame:
    """
    Import quarterly balance sheet data from database for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        DataFrame with quarterly balance sheet data, sorted by fiscal_quarter_end
    """
    try:
        db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()
        db_con = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)
        
        query = f"""
            SELECT * FROM stock_balancesheet_quarterly 
            WHERE ticker = '{ticker}'
            ORDER BY fiscal_quarter_end DESC
        """
        df = pd.read_sql(sql=query, con=db_con)
        return df
        
    except Exception as e:
        print(f"Warning: Could not import quarterly balance sheet data: {e}")
        return pd.DataFrame()


def import_quarterly_cashflow_data(ticker: str) -> pd.DataFrame:
    """
    Import quarterly cash flow data from database for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        DataFrame with quarterly cash flow data, sorted by fiscal_quarter_end
    """
    try:
        db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()
        db_con = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)
        
        query = f"""
            SELECT * FROM stock_cashflow_quarterly 
            WHERE ticker = '{ticker}'
            ORDER BY fiscal_quarter_end DESC
        """
        df = pd.read_sql(sql=query, con=db_con)
        return df
        
    except Exception as e:
        print(f"Warning: Could not import quarterly cash flow data: {e}")
        return pd.DataFrame()


def count_quarterly_reports(ticker: str) -> int:
    """
    Count how many quarterly income reports exist in database for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Number of quarterly reports available
    """
    try:
        db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()
        db_con = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)
        
        query = f"""
            SELECT COUNT(*) as cnt FROM stock_income_stmt_quarterly 
            WHERE ticker = '{ticker}'
        """
        df = pd.read_sql(sql=query, con=db_con)
        return int(df['cnt'].iloc[0])
        
    except Exception as e:
        print(f"Warning: Could not count quarterly reports: {e}")
        return 0


def has_sufficient_quarterly_data(ticker: str, min_quarters: int = 4) -> bool:
    """
    Check if ticker has enough quarterly reports for TTM calculation.
    
    Args:
        ticker: Stock ticker symbol
        min_quarters: Minimum quarters needed (default 4 for TTM)
        
    Returns:
        True if sufficient data exists
    """
    return count_quarterly_reports(ticker) >= min_quarters


def export_quarterly_income_stmt(quarterly_income_df):
    """
    Export quarterly income statement data to the stock_income_stmt_quarterly table.
    
    Args:
        quarterly_income_df: DataFrame with quarterly income statement data
        
    Raises:
        ValueError: If required columns missing
        KeyError: If database operations fail
    """
    if quarterly_income_df.empty:
        raise ValueError("Income statement DataFrame cannot be empty")
    
    required_cols = ['fiscal_quarter_end', 'ticker']
    for col in required_cols:
        if col not in quarterly_income_df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    try:
        db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()
        db_con = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)
        
        # Delete existing records for this ticker before inserting
        from sqlalchemy import text
        ticker = quarterly_income_df['ticker'].iloc[0]
        
        with db_con.begin() as connection:
            connection.execute(text("""
                DELETE FROM stock_income_stmt_quarterly WHERE ticker = :ticker
            """), {'ticker': ticker})
        
        # Filter to only valid columns that exist in database schema
        filtered_df = _filter_to_valid_columns(quarterly_income_df, QUARTERLY_INCOME_COLUMNS)
        
        if filtered_df.empty or len(filtered_df.columns) < 2:
            print(f"   ↳ No valid quarterly income columns to export")
            return
        
        filtered_df.to_sql(
            name="stock_income_stmt_quarterly",
            con=db_con,
            index=False,
            if_exists="append"
        )
        print(f"   ✓ Exported {len(filtered_df)} quarterly income statement records")
        
    except Exception as e:
        raise KeyError(f"Could not export quarterly income statement: {e}") from e


def export_quarterly_balancesheet(quarterly_bs_df):
    """
    Export quarterly balance sheet data to the stock_balancesheet_quarterly table.
    
    Args:
        quarterly_bs_df: DataFrame with quarterly balance sheet data
        
    Raises:
        ValueError: If required columns missing
        KeyError: If database operations fail
    """
    if quarterly_bs_df.empty:
        raise ValueError("Balance sheet DataFrame cannot be empty")
    
    required_cols = ['fiscal_quarter_end', 'ticker']
    for col in required_cols:
        if col not in quarterly_bs_df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    try:
        db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()
        db_con = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)
        
        # Delete existing records for this ticker before inserting
        from sqlalchemy import text
        ticker = quarterly_bs_df['ticker'].iloc[0]
        
        with db_con.begin() as connection:
            connection.execute(text("""
                DELETE FROM stock_balancesheet_quarterly WHERE ticker = :ticker
            """), {'ticker': ticker})
        
        # Filter to only valid columns that exist in database schema
        filtered_df = _filter_to_valid_columns(quarterly_bs_df, QUARTERLY_BALANCESHEET_COLUMNS)
        
        if filtered_df.empty or len(filtered_df.columns) < 2:
            print(f"   ↳ No valid quarterly balance sheet columns to export")
            return
        
        filtered_df.to_sql(
            name="stock_balancesheet_quarterly",
            con=db_con,
            index=False,
            if_exists="append"
        )
        print(f"   ✓ Exported {len(filtered_df)} quarterly balance sheet records")
        
    except Exception as e:
        raise KeyError(f"Could not export quarterly balance sheet: {e}") from e


def export_quarterly_cashflow(quarterly_cf_df):
    """
    Export quarterly cash flow data to the stock_cashflow_quarterly table.
    
    Args:
        quarterly_cf_df: DataFrame with quarterly cash flow data
        
    Raises:
        ValueError: If required columns missing
        KeyError: If database operations fail
    """
    if quarterly_cf_df.empty:
        raise ValueError("Cash flow DataFrame cannot be empty")
    
    required_cols = ['fiscal_quarter_end', 'ticker']
    for col in required_cols:
        if col not in quarterly_cf_df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    try:
        db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()
        db_con = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)
        
        # Delete existing records for this ticker before inserting
        from sqlalchemy import text
        ticker = quarterly_cf_df['ticker'].iloc[0]
        
        with db_con.begin() as connection:
            connection.execute(text("""
                DELETE FROM stock_cashflow_quarterly WHERE ticker = :ticker
            """), {'ticker': ticker})
        
        # Filter to only valid columns that exist in database schema
        filtered_df = _filter_to_valid_columns(quarterly_cf_df, QUARTERLY_CASHFLOW_COLUMNS)
        
        if filtered_df.empty or len(filtered_df.columns) < 2:
            print(f"   ↳ No valid quarterly cash flow columns to export")
            return
        
        filtered_df.to_sql(
            name="stock_cashflow_quarterly",
            con=db_con,
            index=False,
            if_exists="append"
        )
        print(f"   ✓ Exported {len(filtered_df)} quarterly cash flow records")
        
    except Exception as e:
        raise KeyError(f"Could not export quarterly cash flow: {e}") from e


def import_stock_dataset(stock_ticker=""):
    """
    This function imports a complete stock dataset by combining data from multiple tables in the database.
    It merges stock price data, financial statements (income, balance sheet, cash flow), and ratio data,
    along with VIX data for market volatility reference.

    Args:
    stock_ticker: str - The stock ticker symbol to fetch the complete dataset for

    Returns:
    combined_stock_data_df: pandas DataFrame - A comprehensive dataset containing price data with 
                                               forward-filled financial statement data and ratio data

    Raises:
    - KeyError: If the secrets cannot be fetched.
    - KeyError: If the connection to the database cannot be established.
    - ValueError: If stock_ticker is empty or if the stock does not exist in any of the required tables
                  (stock_info_data, stock_price_data, stock_income_stmt_data, stock_balancesheet_data, 
                   stock_cash_flow_data, stock_ratio_data).
    - KeyError: If the dataset cannot be imported from the database tables.
    """
    try:
        # Fetch the secrets from the secret_import function
        db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()

    except Exception as e:
        raise KeyError(f"Could not fetch the secrets. Error: {e}") from e

    try:
        db_con = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)

    except Exception as e:
        raise KeyError(f"Could not establish connection to the database. Error: {e}") from e

    try:
        if stock_ticker == "":
            print("Please input a stock ticker to fetch dataset.")

    except Exception as e:
        raise KeyError(f"Could not import from stock_ratio_data in the database to stock_ratio_data_df. Error: {e}") from e

    try:
        if stock_ticker != "":
            if does_stock_exists_stock_info_data(stock_ticker) is False:
                raise ValueError("The stock does not exist in the stock_info_data table.")

            elif does_stock_exists_stock_price_data(stock_ticker) is False:
                raise ValueError("The stock does not exist in the stock_price_data table.")

            elif does_stock_exists_stock_income_stmt_data(stock_ticker) is False:
                raise ValueError("The stock does not exist in the stock_income_stmt_data table.")

            elif does_stock_exists_stock_balancesheet_data(stock_ticker) is False:
                raise ValueError("The stock does not exist in the stock_balancesheet_data table.")

            elif does_stock_exists_stock_cash_flow_data is False:
                raise ValueError("The stock does not exist in the stock_cash_flow_data table.")

            elif does_stock_exists_stock_ratio_data(stock_ticker) is False:
                raise ValueError("The stock does not exist in the stock_ratio_data table.")

            else:
                price_quary = f"""SELECT * FROM
                    (SELECT * FROM stock_price_data
                    WHERE ticker = "{stock_ticker}"
                    ORDER BY date DESC) AS temp
                    ORDER BY date ASC
                    """
                stock_price_data_df = pd.read_sql(sql=price_quary, con=db_con)

                vix_price_quary = f"""SELECT * FROM
                    (SELECT * FROM stock_price_data
                    WHERE ticker = "{"^VIX"}"
                    ORDER BY date DESC) AS temp
                    ORDER BY date ASC
                    """
                vix_price_data_df = pd.read_sql(sql=vix_price_quary, con=db_con)
                vix_price_data_df = vix_price_data_df.rename(columns={'open_Price': 'VIX_open_Price'})
                stock_price_data_df = stock_price_data_df.merge(vix_price_data_df[['date', 'VIX_open_Price']], on='date', how='left')
                stock_price_data_df['VIX_open_Price'] = stock_price_data_df['VIX_open_Price'].ffill()
                income_stmt_quary = f"""SELECT * FROM
                    (SELECT * FROM stock_income_stmt_data
                    WHERE ticker = "{stock_ticker}"
                    ORDER BY financial_Statement_Date DESC
                    ) AS temp
                    ORDER BY financial_Statement_Date ASC
                    """
                balancesheet_quary = f"""SELECT * FROM
                    (SELECT * FROM stock_balancesheet_data
                    WHERE ticker = "{stock_ticker}"
                    ORDER BY financial_Statement_Date DESC
                    ) AS temp
                    ORDER BY financial_Statement_Date ASC
                    """
                cash_flow_quary = f"""SELECT * FROM
                    (SELECT * FROM stock_cash_flow_data
                    WHERE ticker = "{stock_ticker}"
                    ORDER BY financial_Statement_Date DESC
                    ) AS temp
                    ORDER BY financial_Statement_Date ASC
                    """
                stock_income_stmt_data_df = pd.read_sql(sql=income_stmt_quary, con=db_con)
                stock_income_stmt_data_df = stock_income_stmt_data_df.drop(columns=stock_income_stmt_data_df.columns[1])
                stock_balancesheet_data_df = pd.read_sql(sql=balancesheet_quary, con=db_con)
                stock_balancesheet_data_df = stock_balancesheet_data_df.drop(columns=stock_balancesheet_data_df.columns[1])
                stock_cash_flow_data_df = pd.read_sql(sql=cash_flow_quary, con=db_con)
                stock_cash_flow_data_df = stock_cash_flow_data_df.drop(columns=stock_cash_flow_data_df.columns[1])
                stock_financial_data_df = pd.merge(stock_income_stmt_data_df, stock_balancesheet_data_df, on=["financial_Statement_Date", "ticker"])
                stock_financial_data_df = pd.merge(stock_financial_data_df, stock_cash_flow_data_df, on=["financial_Statement_Date", "ticker"])
                stock_financial_data_df = stock_financial_data_df.rename(columns={"financial_Statement_Date": "date"})
                ratio_quary = f"""SELECT * FROM
                    (SELECT * FROM stock_ratio_data
                    WHERE ticker = "{stock_ticker}"
                    ORDER BY date DESC) AS temp
                    ORDER BY date ASC
                    """
                stock_ratio_data_df = pd.read_sql(sql=ratio_quary, con=db_con)
                # Create a list of column names to copy from stock_financial_data_df to stock_price_data_df
                column_names = stock_financial_data_df.columns[2:]
                # Create a copy of stock_price_data_df
                combined_stock_data_df = stock_price_data_df.copy()
                # Add columns from stock_financial_data_df to stock_price_data_df
                for year in range(len(stock_financial_data_df["date"])):
                    combined_stock_data_df.loc[combined_stock_data_df["date"] >= stock_financial_data_df.iloc[year]["date"], column_names] = stock_financial_data_df.iloc[year].values[2:]

                combined_stock_data_df = pd.merge(combined_stock_data_df, stock_ratio_data_df, on=["date", "ticker"])
                return combined_stock_data_df

    except Exception as e:
        raise KeyError(f"Could not import from stock_ratio_data in the database to stock_ratio_data_df with a specific ticker. Error: {e}") from e


# Run the main function
if __name__ == "__main__":
    TICKER = "NOVO-B"
    if does_stock_exists_stock_price_data(TICKER) is False:
        print("Stock does not exist")
    elif does_stock_exists_stock_price_data(TICKER) is True:
        print("Stock exists")
