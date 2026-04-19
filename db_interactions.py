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
import os
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
        quary = "SELECT ticker FROM stock_info_data WHERE industry != 'Index'"
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
        from sqlalchemy import text
        quary = text("SELECT * FROM stock_info_data WHERE ticker = :ticker")
        stock_info_data_df = pd.read_sql(sql=quary, con=db_con, params={"ticker": stock_ticker})
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
        from sqlalchemy import text
        quary = text("SELECT * FROM stock_price_data WHERE ticker = :ticker")
        stock_price_data_df = pd.read_sql(sql=quary, con=db_con, params={"ticker": stock_ticker})
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
        from sqlalchemy import text
        quary = text("SELECT * FROM stock_income_stmt_data WHERE ticker = :ticker")
        stock_income_stmt_data_df = pd.read_sql(sql=quary, con=db_con, params={"ticker": stock_ticker})
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
        from sqlalchemy import text
        quary = text("SELECT * FROM stock_balancesheet_data WHERE ticker = :ticker")
        stock_balancesheet_data_df = pd.read_sql(sql=quary, con=db_con, params={"ticker": stock_ticker})
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
        from sqlalchemy import text
        quary = text("SELECT * FROM stock_cash_flow_data WHERE ticker = :ticker")
        stock_cash_flow_data_df = pd.read_sql(sql=quary, con=db_con, params={"ticker": stock_ticker})
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
        # Fetch the stock_ratio_data_df from the database
        from sqlalchemy import text
        quary = text("SELECT * FROM stock_ratio_data WHERE ticker = :ticker")
        stock_cash_flow_data_df = pd.read_sql(sql=quary, con=db_con, params={"ticker": stock_ticker})
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
        from sqlalchemy import text
        quary = text("SELECT * FROM stock_prediction_data WHERE ticker = :ticker")
        stock_prediction_data_df = pd.read_sql(sql=quary, con=db_con, params={"ticker": stock_ticker})
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
                from sqlalchemy import text
                quary = text("SELECT * FROM stock_info_data WHERE ticker = :ticker")
                stock_info_data_df = pd.read_sql(sql=quary, con=db_con, params={"ticker": stock_ticker})
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
            from sqlalchemy import text
            quary = text("SELECT * FROM (SELECT * FROM stock_price_data ORDER BY date DESC LIMIT :amount) AS temp ORDER BY date ASC")
            stock_price_data_df = pd.read_sql(sql=quary, con=db_con, params={"amount": int(amount)})
            return stock_price_data_df

    except Exception as e:
        raise KeyError(f"Could not import from stock_price_data in the database to stock_price_data_df . Error: {e}") from e

    try:
        if stock_ticker != "":
            if does_stock_exists_stock_price_data(stock_ticker) is False:
                raise ValueError("The stock does not exist in the stock_price_data table.")
            elif does_stock_exists_stock_price_data(stock_ticker) is True:
                from sqlalchemy import text
                quary = text("SELECT * FROM (SELECT * FROM stock_price_data WHERE ticker = :ticker ORDER BY date DESC LIMIT :amount) AS temp ORDER BY date ASC")
                stock_price_data_df = pd.read_sql(sql=quary, con=db_con, params={"ticker": stock_ticker, "amount": int(amount)})
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
            from sqlalchemy import text
            _params = {"amount": int(amount)}
            income_stmt_quary = text("SELECT * FROM (SELECT * FROM stock_income_stmt_data ORDER BY financial_Statement_Date DESC LIMIT :amount) AS temp ORDER BY financial_Statement_Date ASC")
            balancesheet_quary = text("SELECT * FROM (SELECT * FROM stock_balancesheet_data ORDER BY financial_Statement_Date DESC LIMIT :amount) AS temp ORDER BY financial_Statement_Date ASC")
            cash_flow_quary = text("SELECT * FROM (SELECT * FROM stock_cash_flow_data ORDER BY financial_Statement_Date DESC LIMIT :amount) AS temp ORDER BY financial_Statement_Date ASC")
            stock_income_stmt_data_df = pd.read_sql(sql=income_stmt_quary, con=db_con, params=_params)
            stock_balancesheet_data_df = pd.read_sql(sql=balancesheet_quary, con=db_con, params=_params)
            stock_cash_flow_data_df = pd.read_sql(sql=cash_flow_quary, con=db_con, params=_params)
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
                from sqlalchemy import text
                _params = {"ticker": stock_ticker, "amount": int(amount)}
                income_stmt_quary = text("SELECT * FROM (SELECT * FROM stock_income_stmt_data WHERE ticker = :ticker ORDER BY financial_Statement_Date DESC LIMIT :amount) AS temp ORDER BY financial_Statement_Date ASC")
                balancesheet_quary = text("SELECT * FROM (SELECT * FROM stock_balancesheet_data WHERE ticker = :ticker ORDER BY financial_Statement_Date DESC LIMIT :amount) AS temp ORDER BY financial_Statement_Date ASC")
                cash_flow_quary = text("SELECT * FROM (SELECT * FROM stock_cash_flow_data WHERE ticker = :ticker ORDER BY financial_Statement_Date DESC LIMIT :amount) AS temp ORDER BY financial_Statement_Date ASC")
                stock_income_stmt_data_df = pd.read_sql(sql=income_stmt_quary, con=db_con, params=_params)
                stock_balancesheet_data_df = pd.read_sql(sql=balancesheet_quary, con=db_con, params=_params)
                stock_cash_flow_data_df = pd.read_sql(sql=cash_flow_quary, con=db_con, params=_params)
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
            from sqlalchemy import text
            quary = text("SELECT * FROM (SELECT * FROM stock_ratio_data ORDER BY date DESC LIMIT :amount) AS temp ORDER BY date ASC")
            stock_ratio_data_df = pd.read_sql(sql=quary, con=db_con, params={"amount": int(amount)})
            return stock_ratio_data_df

    except Exception as e:
        raise KeyError(f"Could not import from stock_ratio_data in the database to stock_ratio_data_df. Error: {e}") from e

    try:
        if stock_ticker != "":
            if does_stock_exists_stock_ratio_data(stock_ticker) is False:
                raise ValueError("The stock does not exist in the stock_ratio_data table.")
            elif does_stock_exists_stock_ratio_data(stock_ticker) is True:
                from sqlalchemy import text
                quary = text("SELECT * FROM (SELECT * FROM stock_ratio_data WHERE ticker = :ticker ORDER BY date DESC LIMIT :amount) AS temp ORDER BY date ASC")
                stock_ratio_data_df = pd.read_sql(sql=quary, con=db_con, params={"ticker": stock_ticker, "amount": int(amount)})
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
        from sqlalchemy import text
        annual_query = text("SELECT MAX(financial_Statement_Date) as newest_date FROM stock_income_stmt_data WHERE ticker = :ticker")
        annual_result = pd.read_sql(sql=annual_query, con=db_con, params={"ticker": stock_ticker})
        annual_date = annual_result['newest_date'].iloc[0]
        
        if annual_date is not None:
            newest_date = pd.to_datetime(annual_date).date()
            source = 'annual'
        
        # Check quarterly data if requested
        if include_quarterly:
            quarterly_query = text("SELECT MAX(fiscal_quarter_end) as newest_date FROM stock_income_stmt_quarterly WHERE ticker = :ticker")
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
        from sqlalchemy import text
        query = text("SELECT date, financial_date_used FROM stock_ratio_data WHERE ticker = :ticker ORDER BY date DESC LIMIT 1")
        result = pd.read_sql(sql=query, con=db_con, params={"ticker": stock_ticker})
        
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
        from sqlalchemy import text
        annual_query = text("SELECT DISTINCT financial_Statement_Date as date, 'annual' as source FROM stock_income_stmt_data WHERE ticker = :ticker")
        annual_result = pd.read_sql(sql=annual_query, con=db_con, params={"ticker": stock_ticker})
        if not annual_result.empty:
            all_dates.append(annual_result)
        
        # Get quarterly dates if requested
        if include_quarterly:
            quarterly_query = text("SELECT DISTINCT fiscal_quarter_end as date, 'quarterly' as source FROM stock_income_stmt_quarterly WHERE ticker = :ticker")
            try:
                quarterly_result = pd.read_sql(sql=quarterly_query, con=db_con, params={"ticker": stock_ticker})
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
        
        from sqlalchemy import text
        query = text("SELECT last_fetch_date, last_quarter_end, quarters_count FROM quarterly_fetch_metadata WHERE ticker = :ticker")
        df = pd.read_sql(sql=query, con=db_con, params={"ticker": ticker})
        
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
        
        with db_con.begin() as connection:
            # Use INSERT ... ON DUPLICATE KEY UPDATE for upsert
            connection.execute(text("""
                INSERT INTO quarterly_fetch_metadata 
                    (ticker, last_fetch_date, last_quarter_end, quarters_count)
                VALUES 
                    (:ticker, :today, :last_quarter_end, :quarters_count)
                ON DUPLICATE KEY UPDATE
                    last_fetch_date = :today,
                    last_quarter_end = :last_quarter_end,
                    quarters_count = :quarters_count
            """), {
                "ticker": ticker,
                "today": today,
                "last_quarter_end": last_quarter_end,
                "quarters_count": quarters_count,
            })
        
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
        
        from sqlalchemy import text
        query = text(f"SELECT DISTINCT fiscal_quarter_end FROM {table_name} WHERE ticker = :ticker ORDER BY fiscal_quarter_end")
        df = pd.read_sql(sql=query, con=db_con, params={"ticker": ticker})
        
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
        
        from sqlalchemy import text
        query = text("SELECT * FROM stock_income_stmt_quarterly WHERE ticker = :ticker ORDER BY fiscal_quarter_end DESC")
        df = pd.read_sql(sql=query, con=db_con, params={"ticker": ticker})
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
        
        from sqlalchemy import text
        query = text("SELECT * FROM stock_balancesheet_quarterly WHERE ticker = :ticker ORDER BY fiscal_quarter_end DESC")
        df = pd.read_sql(sql=query, con=db_con, params={"ticker": ticker})
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
        
        from sqlalchemy import text
        query = text("SELECT * FROM stock_cashflow_quarterly WHERE ticker = :ticker ORDER BY fiscal_quarter_end DESC")
        df = pd.read_sql(sql=query, con=db_con, params={"ticker": ticker})
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
        
        from sqlalchemy import text
        query = text("SELECT COUNT(*) as cnt FROM stock_income_stmt_quarterly WHERE ticker = :ticker")
        df = pd.read_sql(sql=query, con=db_con, params={"ticker": ticker})
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

            elif does_stock_exists_stock_cash_flow_data(stock_ticker) is False:
                raise ValueError("The stock does not exist in the stock_cash_flow_data table.")

            elif does_stock_exists_stock_ratio_data(stock_ticker) is False:
                raise ValueError("The stock does not exist in the stock_ratio_data table.")

            else:
                from sqlalchemy import text
                _ticker_params = {"ticker": stock_ticker}
                price_quary = text("SELECT * FROM (SELECT * FROM stock_price_data WHERE ticker = :ticker ORDER BY date DESC) AS temp ORDER BY date ASC")
                stock_price_data_df = pd.read_sql(sql=price_quary, con=db_con, params=_ticker_params)

                vix_price_quary = text("SELECT * FROM (SELECT * FROM stock_price_data WHERE ticker = :ticker ORDER BY date DESC) AS temp ORDER BY date ASC")
                vix_price_data_df = pd.read_sql(sql=vix_price_quary, con=db_con, params={"ticker": "^VIX"})
                vix_price_data_df = vix_price_data_df.rename(columns={'open_Price': 'VIX_open_Price'})
                stock_price_data_df = stock_price_data_df.merge(vix_price_data_df[['date', 'VIX_open_Price']], on='date', how='left')
                stock_price_data_df['VIX_open_Price'] = stock_price_data_df['VIX_open_Price'].ffill()
                income_stmt_quary = text("SELECT * FROM (SELECT * FROM stock_income_stmt_data WHERE ticker = :ticker ORDER BY financial_Statement_Date DESC) AS temp ORDER BY financial_Statement_Date ASC")
                balancesheet_quary = text("SELECT * FROM (SELECT * FROM stock_balancesheet_data WHERE ticker = :ticker ORDER BY financial_Statement_Date DESC) AS temp ORDER BY financial_Statement_Date ASC")
                cash_flow_quary = text("SELECT * FROM (SELECT * FROM stock_cash_flow_data WHERE ticker = :ticker ORDER BY financial_Statement_Date DESC) AS temp ORDER BY financial_Statement_Date ASC")
                stock_income_stmt_data_df = pd.read_sql(sql=income_stmt_quary, con=db_con, params=_ticker_params)
                stock_income_stmt_data_df = stock_income_stmt_data_df.drop(columns=stock_income_stmt_data_df.columns[1])
                stock_balancesheet_data_df = pd.read_sql(sql=balancesheet_quary, con=db_con, params=_ticker_params)
                stock_balancesheet_data_df = stock_balancesheet_data_df.drop(columns=stock_balancesheet_data_df.columns[1])
                stock_cash_flow_data_df = pd.read_sql(sql=cash_flow_quary, con=db_con, params=_ticker_params)
                stock_cash_flow_data_df = stock_cash_flow_data_df.drop(columns=stock_cash_flow_data_df.columns[1])
                stock_financial_data_df = pd.merge(stock_income_stmt_data_df, stock_balancesheet_data_df, on=["financial_Statement_Date", "ticker"])
                stock_financial_data_df = pd.merge(stock_financial_data_df, stock_cash_flow_data_df, on=["financial_Statement_Date", "ticker"])
                stock_financial_data_df = stock_financial_data_df.rename(columns={"financial_Statement_Date": "date"})
                ratio_quary = text("SELECT * FROM (SELECT * FROM stock_ratio_data WHERE ticker = :ticker ORDER BY date DESC) AS temp ORDER BY date ASC")
                stock_ratio_data_df = pd.read_sql(sql=ratio_quary, con=db_con, params=_ticker_params)

                # Normalize date columns to datetime64 to prevent type-mismatch
                # issues (date vs datetime64) when merging/comparing.
                stock_price_data_df["date"] = pd.to_datetime(stock_price_data_df["date"])
                stock_financial_data_df["date"] = pd.to_datetime(stock_financial_data_df["date"])
                stock_ratio_data_df["date"] = pd.to_datetime(stock_ratio_data_df["date"])

                # Create a list of column names to copy from stock_financial_data_df to stock_price_data_df
                column_names = stock_financial_data_df.columns[2:]
                # Create a copy of stock_price_data_df
                combined_stock_data_df = stock_price_data_df.copy()
                # Forward-fill: rows after each financial statement date get that statement's values
                for year in range(len(stock_financial_data_df["date"])):
                    combined_stock_data_df.loc[combined_stock_data_df["date"] >= stock_financial_data_df.iloc[year]["date"], column_names] = stock_financial_data_df.iloc[year].values[2:]

                # Backward-fill: rows before the earliest financial statement get the
                # earliest available values. Without this, stocks whose ratio data
                # starts years before the first financial report lose most rows to
                # dropna because all financial columns are NaN.
                combined_stock_data_df[column_names] = combined_stock_data_df[column_names].bfill()

                # LEFT JOIN keeps all price rows; ratio columns will be NaN for
                # dates not covered by ratio data (e.g. before first calculation).
                combined_stock_data_df = pd.merge(
                    combined_stock_data_df, stock_ratio_data_df,
                    on=["date", "ticker"], how="left"
                )
                return combined_stock_data_df

    except Exception as e:
        raise KeyError(f"Could not import from stock_ratio_data in the database to stock_ratio_data_df with a specific ticker. Error: {e}") from e


# ============================================================================
# PREDICTION AND MONTE CARLO EXPORT FUNCTIONS
# ============================================================================

def export_stock_prediction_extended(
    ticker,
    prediction_date,
    forecast_df,
    current_price,
    model_type="ensemble",
    model_version=None,
    mc_dropout_used=True,
    mc_iterations=30
):
    """
    Export extended stock prediction results to the database.
    
    This function stores ML prediction results with confidence intervals
    for multiple prediction horizons (30, 60, 90, 252 days).
    
    Args:
        ticker (str): Stock ticker symbol
        prediction_date (datetime.date): Date when prediction was made
        forecast_df (pd.DataFrame): Forecast DataFrame with 'close_Price' column
        current_price (float): Current stock price at prediction time
        model_type (str): Type of model used (e.g., 'ensemble', 'tcn', 'lstm')
        model_version (str, optional): Model version identifier
        mc_dropout_used (bool): Whether Monte Carlo Dropout was used
        mc_iterations (int): Number of MC iterations if used
    
    Returns:
        int: Number of rows inserted
    
    Raises:
        ValueError: If required parameters are invalid
        KeyError: If database connection fails
    """
    import datetime
    from sqlalchemy import text
    
    if ticker == "" or ticker is None:
        raise ValueError("Ticker cannot be empty")
    
    if forecast_df is None or forecast_df.empty:
        raise ValueError("Forecast DataFrame cannot be empty")
    
    try:
        db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()
    except Exception as e:
        raise KeyError(f"Could not fetch the secrets. Error: {e}") from e
    
    try:
        db_con = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)
    except Exception as e:
        raise KeyError(f"Could not establish connection to the database. Error: {e}") from e
    
    try:
        # Determine price column name
        price_col = None
        for col in ['close_Price', 'predicted_price', 'price']:
            if col in forecast_df.columns:
                price_col = col
                break
        
        if price_col is None:
            raise ValueError("Forecast DataFrame must contain a price column")
        
        # Prediction horizons to store (in trading days)
        horizons = [30, 60, 90, 252]  # ~1 month, 2 months, 3 months, 1 year
        
        records = []
        for horizon in horizons:
            if horizon < len(forecast_df):
                raw_price = forecast_df[price_col].iloc[horizon]
                if raw_price is None or (isinstance(raw_price, float) and pd.isna(raw_price)):
                    continue  # Skip horizons with missing predictions
                predicted_price = float(raw_price)
                
                # Calculate predicted return
                if current_price > 0:
                    predicted_return = (predicted_price - current_price) / current_price
                else:
                    predicted_return = None
                
                # Calculate target date
                target_date = prediction_date + datetime.timedelta(days=int(horizon * 365 / 252))
                
                # Get confidence intervals if available
                conf_lower_5 = None
                conf_lower_16 = None
                conf_upper_84 = None
                conf_upper_95 = None
                pred_std = None
                
                # Check for uncertainty columns (guard against None/NaN)
                if 'lower_95' in forecast_df.columns and forecast_df['lower_95'].iloc[horizon] is not None:
                    val = forecast_df['lower_95'].iloc[horizon]
                    conf_lower_5 = float(val) if not pd.isna(val) else None
                if 'lower_68' in forecast_df.columns and forecast_df['lower_68'].iloc[horizon] is not None:
                    val = forecast_df['lower_68'].iloc[horizon]
                    conf_lower_16 = float(val) if not pd.isna(val) else None
                if 'upper_68' in forecast_df.columns and forecast_df['upper_68'].iloc[horizon] is not None:
                    val = forecast_df['upper_68'].iloc[horizon]
                    conf_upper_84 = float(val) if not pd.isna(val) else None
                if 'upper_95' in forecast_df.columns and forecast_df['upper_95'].iloc[horizon] is not None:
                    val = forecast_df['upper_95'].iloc[horizon]
                    conf_upper_95 = float(val) if not pd.isna(val) else None
                if 'std' in forecast_df.columns and forecast_df['std'].iloc[horizon] is not None:
                    val = forecast_df['std'].iloc[horizon]
                    pred_std = float(val) if not pd.isna(val) else None
                
                records.append({
                    'prediction_date': prediction_date,
                    'ticker': ticker,
                    'prediction_horizon_days': horizon,
                    'target_date': target_date,
                    'predicted_price': predicted_price,
                    'current_price': current_price,
                    'predicted_return': predicted_return,
                    'confidence_lower_5': conf_lower_5,
                    'confidence_lower_16': conf_lower_16,
                    'confidence_upper_84': conf_upper_84,
                    'confidence_upper_95': conf_upper_95,
                    'model_type': model_type,
                    'model_version': model_version,
                    'prediction_std': pred_std,
                    'mc_dropout_used': mc_dropout_used,
                    'mc_iterations': mc_iterations if mc_dropout_used else None
                })
        
        if records:
            prediction_df = pd.DataFrame(records)
            
            # Delete existing predictions for same date/ticker to allow updates
            delete_query = text("""
                DELETE FROM stock_prediction_extended 
                WHERE prediction_date = :pred_date AND ticker = :ticker
            """)
            with db_con.begin() as conn:
                conn.execute(delete_query, {'pred_date': prediction_date, 'ticker': ticker})
            
            # Insert new predictions
            prediction_df.to_sql(
                'stock_prediction_extended',
                db_con,
                if_exists='append',
                index=False
            )
            
            return len(records)
        
        return 0
    
    except Exception as e:
        raise KeyError(f"Could not export prediction data to database. Error: {e}") from e


def export_monte_carlo_results(
    ticker,
    simulation_date,
    monte_carlo_year_df,
    num_simulations=1000,
    mu_used=None,
    sigma_used=None,
    starting_price=None
):
    """
    Export Monte Carlo simulation results to the database.
    
    This function stores yearly Monte Carlo simulation percentiles and metrics
    for risk analysis and portfolio construction.
    
    Args:
        ticker (str): Stock ticker symbol
        simulation_date (datetime.date): Date when simulation was run
        monte_carlo_year_df (pd.DataFrame): DataFrame indexed by year with percentile columns
        num_simulations (int): Number of simulation paths run
        mu_used (float, optional): Drift parameter used in GBM
        sigma_used (float, optional): Volatility parameter used in GBM
        starting_price (float, optional): Starting price for simulation
    
    Returns:
        int: Number of rows inserted
    
    Raises:
        ValueError: If required parameters are invalid
        KeyError: If database connection fails
    """
    from sqlalchemy import text
    
    if ticker == "" or ticker is None:
        raise ValueError("Ticker cannot be empty")
    
    if monte_carlo_year_df is None or monte_carlo_year_df.empty:
        raise ValueError("Monte Carlo DataFrame cannot be empty")
    
    try:
        db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()
    except Exception as e:
        raise KeyError(f"Could not fetch the secrets. Error: {e}") from e
    
    try:
        db_con = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)
    except Exception as e:
        raise KeyError(f"Could not establish connection to the database. Error: {e}") from e
    
    try:
        records = []
        
        # Iterate through years (index of the DataFrame)
        for year in monte_carlo_year_df.index:
            if year == 0:  # Skip year 0 (starting point)
                continue
            
            row = monte_carlo_year_df.loc[year]
            
            # Extract percentiles from standard column names
            record = {
                'simulation_date': simulation_date,
                'ticker': ticker,
                'simulation_year': int(year),
                'num_simulations': num_simulations,
                'percentile_5': row.get('5th Percentile'),
                'percentile_16': row.get('16th Percentile'),
                'mean_price': row.get('Mean'),
                'percentile_84': row.get('84th Percentile'),
                'percentile_95': row.get('95th Percentile'),
                'mu_used': mu_used,
                'sigma_used': sigma_used,
                'starting_price': starting_price
            }
            
            # Calculate return metrics if starting price is available
            if starting_price and starting_price > 0 and year > 0:
                if record['percentile_5']:
                    record['return_percentile_5'] = (record['percentile_5'] / starting_price) ** (1/year) - 1
                if record['mean_price']:
                    record['return_mean'] = (record['mean_price'] / starting_price) ** (1/year) - 1
                if record['percentile_95']:
                    record['return_percentile_95'] = (record['percentile_95'] / starting_price) ** (1/year) - 1
            
            records.append(record)
        
        if records:
            mc_df = pd.DataFrame(records)
            
            # Delete existing results for same date/ticker to allow updates
            delete_query = text("""
                DELETE FROM monte_carlo_results 
                WHERE simulation_date = :sim_date AND ticker = :ticker
            """)
            with db_con.begin() as conn:
                conn.execute(delete_query, {'sim_date': simulation_date, 'ticker': ticker})
            
            # Insert new results
            mc_df.to_sql(
                'monte_carlo_results',
                db_con,
                if_exists='append',
                index=False
            )
            
            return len(records)
        
        return 0
    
    except Exception as e:
        raise KeyError(f"Could not export Monte Carlo results to database. Error: {e}") from e


def create_portfolio_run(
    risk_level,
    investment_years,
    portfolio_size,
    industries_filter=None,
    countries_filter=None,
    excluded_tickers=None,
    run_name=None
):
    """
    Create a new portfolio optimization run record in the database.
    
    This function creates a tracking record for a portfolio optimization run
    and returns the run_id for associating stock predictions and holdings.
    
    Args:
        risk_level (str): Risk level ('low', 'medium', 'high')
        investment_years (int): Investment horizon in years
        portfolio_size (int): Target number of stocks in portfolio
        industries_filter (list, optional): List of industries to include
        countries_filter (list, optional): List of countries to include
        excluded_tickers (list, optional): List of tickers to exclude
        run_name (str, optional): Optional name for this run
    
    Returns:
        int: The created run_id
    
    Raises:
        KeyError: If database connection fails
    """
    import json
    from sqlalchemy import text
    
    try:
        db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()
    except Exception as e:
        raise KeyError(f"Could not fetch the secrets. Error: {e}") from e
    
    try:
        db_con = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)
    except Exception as e:
        raise KeyError(f"Could not establish connection to the database. Error: {e}") from e
    
    try:
        insert_query = text("""
            INSERT INTO portfolio_runs 
            (run_name, risk_level, investment_years, portfolio_size, 
             industries_filter, countries_filter, excluded_tickers, status)
            VALUES 
            (:run_name, :risk_level, :investment_years, :portfolio_size,
             :industries_filter, :countries_filter, :excluded_tickers, 'running')
        """)
        
        with db_con.begin() as conn:
            result = conn.execute(insert_query, {
                'run_name': run_name,
                'risk_level': risk_level,
                'investment_years': investment_years,
                'portfolio_size': portfolio_size,
                'industries_filter': json.dumps(industries_filter) if industries_filter else None,
                'countries_filter': json.dumps(countries_filter) if countries_filter else None,
                'excluded_tickers': json.dumps(excluded_tickers) if excluded_tickers else None
            })
            
            # Get the auto-generated run_id
            run_id = result.lastrowid
        
        return run_id
    
    except Exception as e:
        raise KeyError(f"Could not create portfolio run record. Error: {e}") from e


def update_portfolio_run(
    run_id,
    total_stocks_analyzed=None,
    successful_predictions=None,
    failed_predictions=None,
    expected_return=None,
    expected_volatility=None,
    sharpe_ratio=None,
    mc_return_p5=None,
    mc_return_mean=None,
    mc_return_p95=None,
    status=None,
    error_message=None,
    execution_time_seconds=None
):
    """
    Update an existing portfolio run record with results.
    
    Args:
        run_id (int): The run_id to update
        total_stocks_analyzed (int, optional): Number of stocks analyzed
        successful_predictions (int, optional): Number of successful predictions
        failed_predictions (int, optional): Number of failed predictions
        expected_return (float, optional): Portfolio expected return
        expected_volatility (float, optional): Portfolio volatility
        sharpe_ratio (float, optional): Portfolio Sharpe ratio
        mc_return_p5 (float, optional): Monte Carlo 5th percentile return
        mc_return_mean (float, optional): Monte Carlo mean return
        mc_return_p95 (float, optional): Monte Carlo 95th percentile return
        status (str, optional): Run status ('running', 'completed', 'failed')
        error_message (str, optional): Error message if failed
        execution_time_seconds (float, optional): Total execution time
    
    Returns:
        bool: True if update successful
    
    Raises:
        KeyError: If database connection fails
    """
    from sqlalchemy import text
    
    try:
        db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()
    except Exception as e:
        raise KeyError(f"Could not fetch the secrets. Error: {e}") from e
    
    try:
        db_con = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)
    except Exception as e:
        raise KeyError(f"Could not establish connection to the database. Error: {e}") from e
    
    try:
        # Build dynamic update query
        updates = []
        params = {'run_id': run_id}
        
        if total_stocks_analyzed is not None:
            updates.append("total_stocks_analyzed = :total_stocks_analyzed")
            params['total_stocks_analyzed'] = total_stocks_analyzed
        if successful_predictions is not None:
            updates.append("successful_predictions = :successful_predictions")
            params['successful_predictions'] = successful_predictions
        if failed_predictions is not None:
            updates.append("failed_predictions = :failed_predictions")
            params['failed_predictions'] = failed_predictions
        if expected_return is not None:
            updates.append("expected_return = :expected_return")
            params['expected_return'] = expected_return
        if expected_volatility is not None:
            updates.append("expected_volatility = :expected_volatility")
            params['expected_volatility'] = expected_volatility
        if sharpe_ratio is not None:
            updates.append("sharpe_ratio = :sharpe_ratio")
            params['sharpe_ratio'] = sharpe_ratio
        if mc_return_p5 is not None:
            updates.append("mc_return_p5 = :mc_return_p5")
            params['mc_return_p5'] = mc_return_p5
        if mc_return_mean is not None:
            updates.append("mc_return_mean = :mc_return_mean")
            params['mc_return_mean'] = mc_return_mean
        if mc_return_p95 is not None:
            updates.append("mc_return_p95 = :mc_return_p95")
            params['mc_return_p95'] = mc_return_p95
        if status is not None:
            updates.append("status = :status")
            params['status'] = status
        if error_message is not None:
            updates.append("error_message = :error_message")
            params['error_message'] = error_message
        if execution_time_seconds is not None:
            updates.append("execution_time_seconds = :execution_time_seconds")
            params['execution_time_seconds'] = execution_time_seconds
        
        if updates:
            update_query = text(f"""
                UPDATE portfolio_runs 
                SET {', '.join(updates)}
                WHERE run_id = :run_id
            """)
            
            with db_con.begin() as conn:
                conn.execute(update_query, params)
        
        return True
    
    except Exception as e:
        raise KeyError(f"Could not update portfolio run record. Error: {e}") from e


def export_portfolio_holdings(run_id, holdings_df):
    """
    Export portfolio holdings for a specific run to the database.
    
    Args:
        run_id (int): The portfolio run ID
        holdings_df (pd.DataFrame): DataFrame with columns:
            - ticker: Stock ticker symbol
            - weight: Portfolio weight (0-1)
            - rank: Rank in portfolio (optional)
            - expected_return: Expected return (optional)
            - volatility: Volatility (optional)
            - industry: Industry classification (optional)
    
    Returns:
        int: Number of holdings inserted
    
    Raises:
        ValueError: If required columns are missing
        KeyError: If database connection fails
    """
    from sqlalchemy import text
    
    if holdings_df is None or holdings_df.empty:
        raise ValueError("Holdings DataFrame cannot be empty")
    
    if 'ticker' not in holdings_df.columns or 'weight' not in holdings_df.columns:
        raise ValueError("Holdings DataFrame must contain 'ticker' and 'weight' columns")
    
    try:
        db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()
    except Exception as e:
        raise KeyError(f"Could not fetch the secrets. Error: {e}") from e
    
    try:
        db_con = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)
    except Exception as e:
        raise KeyError(f"Could not establish connection to the database. Error: {e}") from e
    
    try:
        # Add run_id to DataFrame
        holdings_df = holdings_df.copy()
        holdings_df['run_id'] = run_id
        
        # Ensure columns match database schema
        db_columns = [
            'run_id', 'ticker', 'weight', 'rank', 'expected_return', 
            'volatility', 'sharpe_ratio', 'correlation_to_portfolio',
            'marginal_contribution_to_risk', 'industry', 'country'
        ]
        
        # Only keep columns that exist in both DataFrame and database
        columns_to_export = [col for col in db_columns if col in holdings_df.columns]
        holdings_to_export = holdings_df[columns_to_export]
        
        # Insert holdings
        holdings_to_export.to_sql(
            'portfolio_holdings',
            db_con,
            if_exists='append',
            index=False
        )
        
        return len(holdings_to_export)
    
    except Exception as e:
        raise KeyError(f"Could not export portfolio holdings. Error: {e}") from e


def import_monte_carlo_results(ticker=None, simulation_date=None, years=None):
    """
    Import Monte Carlo simulation results from the database.
    
    Args:
        ticker (str, optional): Filter by ticker symbol
        simulation_date (date, optional): Filter by simulation date
        years (list, optional): List of years to retrieve
    
    Returns:
        pd.DataFrame: Monte Carlo results
    
    Raises:
        KeyError: If database connection fails
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
        query = "SELECT * FROM monte_carlo_results WHERE 1=1"
        params = {}
        
        if ticker:
            query += " AND ticker = %(ticker)s"
            params['ticker'] = ticker
        
        if simulation_date:
            query += " AND simulation_date = %(sim_date)s"
            params['sim_date'] = simulation_date
        
        if years:
            query += f" AND simulation_year IN ({','.join(map(str, years))})"
        
        query += " ORDER BY ticker, simulation_year"
        
        return pd.read_sql(query, db_con, params=params)
    
    except Exception as e:
        raise KeyError(f"Could not import Monte Carlo results. Error: {e}") from e


def import_stock_predictions_extended(ticker=None, prediction_date=None):
    """
    Import extended stock predictions from the database.
    
    Args:
        ticker (str, optional): Filter by ticker symbol
        prediction_date (date, optional): Filter by prediction date
    
    Returns:
        pd.DataFrame: Extended prediction results
    
    Raises:
        KeyError: If database connection fails
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
        query = "SELECT * FROM stock_prediction_extended WHERE 1=1"
        params = {}
        
        if ticker:
            query += " AND ticker = %(ticker)s"
            params['ticker'] = ticker
        
        if prediction_date:
            query += " AND prediction_date = %(pred_date)s"
            params['pred_date'] = prediction_date
        
        query += " ORDER BY ticker, prediction_horizon_days"
        
        return pd.read_sql(query, db_con, params=params)
    
    except Exception as e:
        raise KeyError(f"Could not import stock predictions. Error: {e}") from e


# ============================================================================
# MODEL / PREDICTION FRESHNESS QUERY FUNCTIONS
# ============================================================================

def get_model_status_for_all_tickers(max_age_days=30):
    """
    Get model training status for all tickers in the database.
    
    Returns a DataFrame with columns:
        ticker, model_type, tuning_date, age_days, is_fresh
    
    Args:
        max_age_days: Models older than this are considered stale
        
    Returns:
        pd.DataFrame with model status per ticker/model_type,
        or empty DataFrame on error
    """
    try:
        db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()
        db_con = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)
    except Exception as e:
        print(f"[WARNING] Could not connect to database: {e}")
        return pd.DataFrame()
    
    try:
        query = """
            SELECT 
                ticker,
                model_type,
                tuning_date,
                DATEDIFF(NOW(), tuning_date) AS age_days,
                CASE WHEN DATEDIFF(NOW(), tuning_date) <= %(max_age)s AND is_valid = TRUE 
                     THEN TRUE ELSE FALSE END AS is_fresh
            FROM model_hyperparameters
            WHERE is_valid = TRUE
            ORDER BY ticker, model_type
        """
        return pd.read_sql(query, db_con, params={'max_age': max_age_days})
    except Exception as e:
        print(f"[WARNING] Could not query model status: {e}")
        return pd.DataFrame()


def get_tickers_needing_training(max_age_days=30, required_model_types=None):
    """
    Get tickers that need model training — either no models exist or models are stale.
    
    Args:
        max_age_days: Models older than this are considered stale
        required_model_types: List of required model types (default: ['rf', 'xgb', 'tcn'])
        
    Returns:
        dict with keys:
            'untrained': list of tickers with no models at all
            'stale': list of tickers with at least one stale model
            'fresh': list of tickers with all models fresh
    """
    if required_model_types is None:
        required_model_types = ['rf', 'xgb', 'tcn']
    
    try:
        db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()
        db_con = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)
    except Exception as e:
        print(f"[WARNING] Could not connect to database: {e}")
        return {'untrained': [], 'stale': [], 'fresh': []}
    
    try:
        # Get all non-index tickers (matches import_ticker_list filter)
        all_tickers_df = pd.read_sql("SELECT ticker FROM stock_info_data WHERE industry != 'Index'", db_con)
        all_tickers = set(all_tickers_df['ticker'].tolist())
        
        # Get model status
        model_status = get_model_status_for_all_tickers(max_age_days)
        
        if model_status.empty:
            return {
                'untrained': list(all_tickers),
                'stale': [],
                'fresh': []
            }
        
        # Group by ticker
        tickers_with_models = set(model_status['ticker'].unique())
        untrained = list(all_tickers - tickers_with_models)
        
        stale = []
        fresh = []
        
        for ticker in tickers_with_models:
            ticker_models = model_status[model_status['ticker'] == ticker]
            
            # Check if all required models exist and are fresh
            has_all_required = all(
                mt in ticker_models['model_type'].values 
                for mt in required_model_types
            )
            all_fresh = ticker_models['is_fresh'].all()
            
            if has_all_required and all_fresh:
                fresh.append(ticker)
            else:
                stale.append(ticker)
        
        return {
            'untrained': sorted(untrained),
            'stale': sorted(stale),
            'fresh': sorted(fresh)
        }
    except Exception as e:
        print(f"[WARNING] Could not determine training needs: {e}")
        return {'untrained': [], 'stale': [], 'fresh': []}


def get_tickers_needing_prediction(max_prediction_age_days=1):
    """
    Get tickers that have trained models but no recent predictions.
    
    Args:
        max_prediction_age_days: Predictions older than this need refreshing
        
    Returns:
        dict with keys:
            'needs_prediction': list of tickers needing new predictions
            'recently_predicted': list of tickers with fresh predictions
    """
    try:
        db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()
        db_con = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)
    except Exception as e:
        print(f"[WARNING] Could not connect to database: {e}")
        return {'needs_prediction': [], 'recently_predicted': []}
    
    try:
        # Tickers with valid models
        models_query = """
            SELECT DISTINCT ticker 
            FROM model_hyperparameters 
            WHERE is_valid = TRUE
        """
        model_tickers_df = pd.read_sql(models_query, db_con)
        model_tickers = set(model_tickers_df['ticker'].tolist()) if not model_tickers_df.empty else set()
        
        # Tickers with recent predictions
        predictions_query = """
            SELECT DISTINCT ticker
            FROM stock_prediction_extended
            WHERE DATEDIFF(NOW(), prediction_date) <= %(max_age)s
        """
        predicted_df = pd.read_sql(predictions_query, db_con, params={'max_age': max_prediction_age_days})
        recently_predicted = set(predicted_df['ticker'].tolist()) if not predicted_df.empty else set()
        
        # Tickers that have models but no recent predictions
        needs_prediction = model_tickers - recently_predicted
        
        return {
            'needs_prediction': sorted(list(needs_prediction)),
            'recently_predicted': sorted(list(recently_predicted))
        }
    except Exception as e:
        print(f"[WARNING] Could not determine prediction needs: {e}")
        return {'needs_prediction': [], 'recently_predicted': []}


def get_tickers_with_predictions(max_age_days=7):
    """
    Get tickers that have recent ML predictions available for portfolio construction.
    
    Args:
        max_age_days: Maximum age of predictions to consider
        
    Returns:
        list of ticker symbols with recent predictions
    """
    try:
        db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()
        db_con = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)
    except Exception as e:
        print(f"[WARNING] Could not connect to database: {e}")
        return []
    
    try:
        query = """
            SELECT DISTINCT ticker
            FROM stock_prediction_extended
            WHERE DATEDIFF(NOW(), prediction_date) <= %(max_age)s
            ORDER BY ticker
        """
        result = pd.read_sql(query, db_con, params={'max_age': max_age_days})
        return result['ticker'].tolist() if not result.empty else []
    except Exception as e:
        print(f"[WARNING] Could not query prediction tickers: {e}")
        return []


def get_prediction_summary(ticker, max_age_days=7):
    """
    Get the latest prediction summary for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        max_age_days: Maximum age of predictions to consider
        
    Returns:
        dict with prediction info or None if no recent predictions
    """
    try:
        db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()
        db_con = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)
    except Exception as e:
        print(f"[WARNING] Could not connect to database: {e}")
        return None
    
    try:
        query = """
            SELECT 
                prediction_date,
                prediction_horizon_days,
                predicted_price,
                current_price,
                predicted_return,
                confidence_lower_5,
                confidence_upper_95,
                model_type,
                prediction_std
            FROM stock_prediction_extended
            WHERE ticker = %(ticker)s
            AND DATEDIFF(NOW(), prediction_date) <= %(max_age)s
            ORDER BY prediction_horizon_days
        """
        result = pd.read_sql(query, db_con, params={'ticker': ticker, 'max_age': max_age_days})
        
        if result.empty:
            return None
        
        return {
            'prediction_date': result['prediction_date'].iloc[0],
            'current_price': result['current_price'].iloc[0],
            'horizons': result.to_dict('records')
        }
    except Exception as e:
        print(f"[WARNING] Could not query prediction summary: {e}")
        return None


# ============================================================================
# HYPERPARAMETER STORAGE FUNCTIONS
# ============================================================================

def save_hyperparameters(
    ticker,
    model_type,
    hyperparameters,
    num_trials=None,
    best_score=None,
    tuning_time_seconds=None,
    training_samples=None,
    num_features=None,
    feature_list=None,
    val_mse=None,
    val_r2=None,
    val_mae=None,
    is_constrained=False
):
    """
    Save best hyperparameters to database for reuse.
    
    This allows skipping tuning if valid hyperparameters already exist,
    significantly reducing tuning_dir storage usage.
    
    Args:
        ticker (str): Stock ticker symbol
        model_type (str): Model type ('rf', 'xgb', 'lstm', 'tcn')
        hyperparameters (dict): Best hyperparameters as dictionary
        num_trials (int, optional): Number of trials in tuning session
        best_score (float, optional): Best validation score achieved
        tuning_time_seconds (float, optional): Time taken for tuning
        training_samples (int, optional): Number of training samples
        num_features (int, optional): Number of features
        feature_list (list, optional): List of feature names (for hash)
        val_mse (float, optional): Validation MSE
        val_r2 (float, optional): Validation R2
        val_mae (float, optional): Validation MAE
        is_constrained (bool): Whether overfitting constraints were applied
    
    Returns:
        bool: True if saved successfully
    
    Raises:
        ValueError: If required parameters are invalid
        KeyError: If database connection fails
    """
    import json
    import hashlib
    from sqlalchemy import text
    
    if not ticker or not model_type:
        raise ValueError("Ticker and model_type are required")
    
    # Normalize model_type to lowercase to match DB enum
    model_type = model_type.lower()
    
    if model_type not in ('rf', 'xgb', 'lstm', 'tcn'):
        raise ValueError(f"Invalid model_type: {model_type}. Must be one of: rf, xgb, lstm, tcn")
    
    if not hyperparameters:
        raise ValueError("Hyperparameters dictionary cannot be empty")
    
    try:
        db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()
    except Exception as e:
        raise KeyError(f"Could not fetch the secrets. Error: {e}") from e
    
    try:
        db_con = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)
    except Exception as e:
        raise KeyError(f"Could not establish connection to the database. Error: {e}") from e
    
    try:
        # Create feature hash if feature list provided
        feature_hash = None
        if feature_list:
            feature_str = ','.join(sorted(str(f) for f in feature_list))
            feature_hash = hashlib.sha256(feature_str.encode()).hexdigest()[:64]
        
        # Convert hyperparameters to JSON string
        hp_json = json.dumps(hyperparameters, default=str)
        
        # Use REPLACE to upsert (MySQL specific)
        upsert_query = text("""
            REPLACE INTO model_hyperparameters 
            (ticker, model_type, hyperparameters, num_trials, best_score, 
             tuning_time_seconds, training_samples, num_features, feature_hash,
             val_mse, val_r2, val_mae, is_constrained, is_valid, tuning_date)
            VALUES 
            (:ticker, :model_type, :hyperparameters, :num_trials, :best_score,
             :tuning_time_seconds, :training_samples, :num_features, :feature_hash,
             :val_mse, :val_r2, :val_mae, :is_constrained, TRUE, CURRENT_TIMESTAMP)
        """)
        
        with db_con.begin() as conn:
            conn.execute(upsert_query, {
                'ticker': ticker,
                'model_type': model_type,
                'hyperparameters': hp_json,
                'num_trials': num_trials,
                'best_score': best_score,
                'tuning_time_seconds': tuning_time_seconds,
                'training_samples': training_samples,
                'num_features': num_features,
                'feature_hash': feature_hash,
                'val_mse': val_mse,
                'val_r2': val_r2,
                'val_mae': val_mae,
                'is_constrained': is_constrained
            })
        
        print(f"[DB] Saved {model_type.upper()} hyperparameters for {ticker}")
        return True
    
    except Exception as e:
        print(f"[WARNING] Could not save hyperparameters: {e}")
        return False


def load_hyperparameters(
    ticker,
    model_type,
    max_age_days=30,
    feature_list=None,
    require_same_features=False
):
    """
    Load cached hyperparameters from database.
    
    Args:
        ticker (str): Stock ticker symbol
        model_type (str): Model type ('rf', 'xgb', 'lstm', 'tcn')
        max_age_days (int): Maximum age of cached HPs in days (None = no limit)
        feature_list (list, optional): Current feature list for validation
        require_same_features (bool): If True, only return if feature hash matches
    
    Returns:
        dict or None: Hyperparameters dict if valid cache exists, None otherwise
    
    Raises:
        KeyError: If database connection fails
    """
    import json
    import hashlib
    from datetime import datetime, timedelta
    
    if not ticker or not model_type:
        return None
    
    # Normalize model_type to lowercase to match DB enum
    model_type = model_type.lower()
    
    try:
        db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()
    except Exception as e:
        print(f"[WARNING] Could not fetch secrets: {e}")
        return None
    
    try:
        db_con = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)
    except Exception as e:
        print(f"[WARNING] Could not connect to database: {e}")
        return None
    
    try:
        query = """
            SELECT hyperparameters, tuning_date, feature_hash, val_mse, val_r2
            FROM model_hyperparameters 
            WHERE ticker = %(ticker)s 
            AND model_type = %(model_type)s 
            AND is_valid = TRUE
        """
        params = {'ticker': ticker, 'model_type': model_type}
        
        result = pd.read_sql(query, db_con, params=params)
        
        if result.empty:
            return None
        
        row = result.iloc[0]
        tuning_date = row['tuning_date']
        
        # Check age
        if max_age_days is not None:
            if isinstance(tuning_date, str):
                tuning_date = datetime.fromisoformat(tuning_date)
            age = datetime.now() - tuning_date
            if age.days > max_age_days:
                print(f"[CACHE] {model_type.upper()} HPs for {ticker} expired ({age.days} days old)")
                return None
        
        # Check feature hash if required
        if require_same_features and feature_list:
            feature_str = ','.join(sorted(str(f) for f in feature_list))
            current_hash = hashlib.sha256(feature_str.encode()).hexdigest()[:64]
            if row['feature_hash'] != current_hash:
                print(f"[CACHE] {model_type.upper()} HPs for {ticker} have different features")
                return None
        
        # Parse and return hyperparameters
        hp_json = row['hyperparameters']
        hyperparameters = json.loads(hp_json) if isinstance(hp_json, str) else hp_json
        
        mse_str = f"{row['val_mse']:.4f}" if row['val_mse'] is not None else "N/A"
        r2_str = f"{row['val_r2']:.4f}" if row['val_r2'] is not None else "N/A"
        print(f"[CACHE] Loaded {model_type.upper()} hyperparameters for {ticker} "
              f"(MSE: {mse_str}, R2: {r2_str})")
        
        return hyperparameters
    
    except Exception as e:
        print(f"[WARNING] Could not load hyperparameters: {e}")
        return None


def invalidate_hyperparameters(ticker=None, model_type=None):
    """
    Invalidate cached hyperparameters to force re-tuning.
    
    Args:
        ticker (str, optional): Specific ticker to invalidate (None = all)
        model_type (str, optional): Specific model type (None = all types)
    
    Returns:
        int: Number of records invalidated
    """
    from sqlalchemy import text
    
    try:
        db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()
        db_con = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)
    except Exception as e:
        print(f"[WARNING] Could not connect to database: {e}")
        return 0
    
    try:
        query = "UPDATE model_hyperparameters SET is_valid = FALSE WHERE 1=1"
        params = {}
        
        if ticker:
            query += " AND ticker = :ticker"
            params['ticker'] = ticker
        
        if model_type:
            query += " AND model_type = :model_type"
            params['model_type'] = model_type
        
        with db_con.begin() as conn:
            result = conn.execute(text(query), params)
            count = result.rowcount
        
        print(f"[DB] Invalidated {count} hyperparameter cache entries")
        return count
    
    except Exception as e:
        print(f"[WARNING] Could not invalidate hyperparameters: {e}")
        return 0


def cleanup_tuning_directory(tuning_dir="tuning_dir", keep_tickers=None, dry_run=False):
    """
    Clean up tuning directory to free disk space.
    
    This function removes tuning directories for stocks that have
    valid hyperparameters cached in the database.
    
    Args:
        tuning_dir (str): Path to tuning directory
        keep_tickers (list, optional): List of tickers to keep (others deleted)
        dry_run (bool): If True, only report what would be deleted
    
    Returns:
        dict: Summary of cleanup operation
    """
    import shutil
    
    if not os.path.exists(tuning_dir):
        return {"status": "tuning_dir not found", "deleted": 0, "freed_mb": 0}
    
    summary = {
        "status": "completed",
        "deleted": 0,
        "kept": 0,
        "freed_mb": 0,
        "deleted_dirs": [],
        "kept_dirs": []
    }
    
    try:
        for item in os.listdir(tuning_dir):
            item_path = os.path.join(tuning_dir, item)
            
            if not os.path.isdir(item_path):
                continue
            
            # Parse ticker from directory name (e.g., "RF_tuning_AAPL" -> "AAPL")
            parts = item.split('_')
            if len(parts) >= 3:
                ticker = '_'.join(parts[2:])  # Handle tickers with underscores
            else:
                continue
            
            # Check if we should keep this ticker
            should_keep = keep_tickers and ticker in keep_tickers
            
            if should_keep:
                summary["kept"] += 1
                summary["kept_dirs"].append(item)
                continue
            
            # Calculate size
            size_bytes = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, dirnames, filenames in os.walk(item_path)
                for filename in filenames
            )
            size_mb = size_bytes / (1024 * 1024)
            
            if dry_run:
                print(f"[DRY RUN] Would delete: {item} ({size_mb:.1f} MB)")
                summary["freed_mb"] += size_mb
                summary["deleted"] += 1
                summary["deleted_dirs"].append(item)
            else:
                try:
                    shutil.rmtree(item_path)
                    print(f"[CLEANUP] Deleted: {item} ({size_mb:.1f} MB)")
                    summary["freed_mb"] += size_mb
                    summary["deleted"] += 1
                    summary["deleted_dirs"].append(item)
                except (OSError, PermissionError) as e:
                    print(f"[WARNING] Could not delete {item}: {e}")
        
        return summary
    
    except Exception as e:
        summary["status"] = f"error: {e}"
        return summary


def diagnose_stock_pipeline(stock_ticker):
    """Diagnose data availability for a ticker across all pipeline tables.
    
    Returns a dict with row counts and date ranges for each table,
    plus the final merged row count, to identify data gaps.
    """
    from sqlalchemy import text
    db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()
    engine = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)

    tables = {
        "stock_price_data": "date",
        "stock_income_stmt_data": "financial_Statement_Date",
        "stock_balancesheet_data": "financial_Statement_Date",
        "stock_cash_flow_data": "financial_Statement_Date",
        "stock_ratio_data": "date",
    }
    report = {"ticker": stock_ticker, "tables": {}}

    with engine.connect() as conn:
        for table, date_col in tables.items():
            row = conn.execute(
                text(f"SELECT COUNT(*) AS cnt, MIN({date_col}) AS min_dt, MAX({date_col}) AS max_dt "
                     f"FROM {table} WHERE ticker = :ticker"),
                {"ticker": stock_ticker},
            ).mappings().first()
            report["tables"][table] = {
                "rows": int(row["cnt"]),
                "min_date": str(row["min_dt"]) if row["min_dt"] else None,
                "max_date": str(row["max_dt"]) if row["max_dt"] else None,
            }

    # Check merged row count using the real pipeline function
    try:
        merged_df = import_stock_dataset(stock_ticker)
        report["merged_rows"] = len(merged_df) if merged_df is not None else 0
    except Exception as e:
        report["merged_rows"] = 0
        report["merge_error"] = str(e)

    return report


# ─── Beta Data Functions ─────────────────────────────────────────────

def export_stock_beta_data(beta_df):
    """
    Export stock beta data to the stock_beta_data table.
    
    Uses UPSERT logic: deletes existing records for the same ticker/index/date range,
    then inserts new data.
    
    Args:
        beta_df: DataFrame with columns: date, ticker, index_code, index_symbol,
                 beta_60d, beta_120d, beta_252d, correlation_252d, r_squared_252d
    """
    if beta_df.empty:
        return
    
    required_cols = ['date', 'ticker', 'index_code']
    missing = [c for c in required_cols if c not in beta_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()
    engine = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)
    
    # Only keep valid DB columns
    db_columns = [
        'date', 'ticker', 'index_code', 'index_symbol',
        'beta_60d', 'beta_120d', 'beta_252d',
        'correlation_252d', 'r_squared_252d'
    ]
    export_df = beta_df[[c for c in db_columns if c in beta_df.columns]].copy()
    
    # Drop rows with no beta values at all
    beta_cols = [c for c in ['beta_60d', 'beta_120d', 'beta_252d'] if c in export_df.columns]
    if beta_cols:
        export_df = export_df.dropna(subset=beta_cols, how='all')
    
    if export_df.empty:
        return
    
    try:
        from sqlalchemy import text
        
        ticker = export_df['ticker'].iloc[0]
        index_code = export_df['index_code'].iloc[0]
        min_date = export_df['date'].min()
        max_date = export_df['date'].max()
        
        with engine.begin() as connection:
            connection.execute(text("""
                DELETE FROM stock_beta_data
                WHERE ticker = :ticker
                  AND index_code = :index_code
                  AND date >= :min_date
                  AND date <= :max_date
            """), {
                'ticker': ticker,
                'index_code': index_code,
                'min_date': min_date,
                'max_date': max_date,
            })
        
        export_df.to_sql(name="stock_beta_data", con=engine, index=False, if_exists="append")
        print(f"✓ Exported {len(export_df)} beta records for {ticker} vs {index_code}")
        
    except Exception as e:
        print(f"❌ Error exporting beta data: {e}")
    finally:
        engine.dispose()


def import_stock_beta_data(ticker: str, index_code: str = None, amount: int = 1) -> pd.DataFrame:
    """
    Import beta data for a stock, optionally filtered by index.
    
    Args:
        ticker: Stock ticker symbol
        index_code: Optional index code filter (e.g., 'SP500', 'C25')
        amount: Number of most recent records to return per index
        
    Returns:
        DataFrame with beta data
    """
    db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()
    engine = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)
    
    try:
        if index_code:
            query = """
                SELECT * FROM stock_beta_data
                WHERE ticker = %s AND index_code = %s
                ORDER BY date DESC
                LIMIT %s
            """
            df = pd.read_sql(query, engine, params=(ticker, index_code, amount))
        else:
            # Get latest record for each index
            query = """
                SELECT b.* FROM stock_beta_data b
                INNER JOIN (
                    SELECT ticker, index_code, MAX(date) AS max_date
                    FROM stock_beta_data
                    WHERE ticker = %s
                    GROUP BY ticker, index_code
                ) latest ON b.ticker = latest.ticker
                    AND b.index_code = latest.index_code
                    AND b.date = latest.max_date
                ORDER BY b.index_code
            """
            df = pd.read_sql(query, engine, params=(ticker,))
        return df
    except Exception as e:
        print(f"Error importing beta data for {ticker}: {e}")
        return pd.DataFrame()
    finally:
        engine.dispose()


def get_available_beta_indices(ticker: str = None) -> list:
    """
    Get list of index codes that have beta data.
    
    Args:
        ticker: Optional filter — only return indices for this ticker
        
    Returns:
        List of index_code strings
    """
    db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()
    engine = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)
    
    try:
        if ticker:
            query = "SELECT DISTINCT index_code FROM stock_beta_data WHERE ticker = %s ORDER BY index_code"
            df = pd.read_sql(query, engine, params=(ticker,))
        else:
            query = "SELECT DISTINCT index_code FROM stock_beta_data ORDER BY index_code"
            df = pd.read_sql(query, engine)
        return df['index_code'].tolist() if not df.empty else []
    except Exception as e:
        print(f"Error getting beta indices: {e}")
        return []
    finally:
        engine.dispose()


# Run the main function
if __name__ == "__main__":
    TICKER = "PLTR"
    report = diagnose_stock_pipeline(TICKER)
    print(f"Data availability report for {TICKER}:")
    for table, info in report["tables"].items():
        print(f"  {table}: {info['rows']} rows, date range: {info['min_date']} to {info['max_date']}")

    if does_stock_exists_stock_price_data(TICKER) is False:
        print("Stock does not exist")
    elif does_stock_exists_stock_price_data(TICKER) is True:
        print("Stock exists")
