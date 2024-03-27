"""This module contains functions for interacting with the database."""
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
    - ValueError: If the stock_price_data_df parameter is empty.
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
        raise ValueError("The stock_price_data_df parameter cannot be empty.")

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
        column_list = ['currency', 'trade_Volume',
            'open_Price', '1D', '1M', '3M', '6M', '9M', '1Y', '2Y', '3Y', '4Y', '5Y',
            'sma_40', 'sma_120', 'ema_40', 'ema_120', 'std_Div_40', 'std_Div_120',
            'bollinger_Band_40_2STD', 'bollinger_Band_120_2STD', 'momentum'
            ]

        for column in column_list:
            if column not in stock_price_data_df.columns:
                column_list.pop(column)

        column_list.insert(0, 'date')
        column_list.insert(1, 'ticker')

    except Exception as e:
        raise KeyError(f"Could not drop columns from stock_price_data_df. Error: {e}") from e

    try:
        stock_price_data_df = stock_price_data_df[column_list]
        stock_price_data_df.to_sql(name="stock_price_data", con=db_con, index=False, if_exists="append")

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
        stock_income_stmt_data_df.to_sql(name="stock_income_stmt_data", con=db_con, index=False, if_exists="append")
        stock_balancesheet_data_df.to_sql(name="stock_balancesheet_data", con=db_con, index=False, if_exists="append")
        stock_cash_flow_data_df.to_sql(name="stock_cash_flow_data", con=db_con, index=False, if_exists="append")

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
                ORDER BY date DESC LIMIT {amount}) AS temp
                WHERE ticker = "{stock_ticker}"
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
        stock_ratio_data_df.to_sql(name="stock_ratio_data", con=db_con, index=False, if_exists="append")

    except Exception as e:
        raise KeyError(f"Could not export from stock_ratio_data_df to stock_price_data in the database. Error: {e}") from e

# Run the main function
if __name__ == "__main__":
    TICKER = "NOVO-B"
    if does_stock_exists_stock_price_data(TICKER) is False:
        print("Stock does not exist")
    elif does_stock_exists_stock_price_data(TICKER) is True:
        print("Stock exists")
