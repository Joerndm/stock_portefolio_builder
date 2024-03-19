""""""
import pandas as pd

import fetch_secrets
import db_connectors

def import_ticker_list():
    """
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
        stock_info_data_df = pd.read_sql(sql="SELECT ticker FROM stock_info_data", con=db_con)
        ticker_list = stock_info_data_df["ticker"].tolist()
        return ticker_list
    
    except Exception as e:
        raise KeyError(f"Could not fetch the tickers from stock_info_data_df in the database. Error: {e}") from e

def control_if_stock_exists(stock_symbol=""):
    """
    This function checks if a stock exists in the database.

    Parameters:
    stock_symbol (str): The stock symbol to check if exists in the database.

    Returns:
    bool: True if the stock exists in the database, False if the stock does not exist in the database.
    """
    # Check if the stock_price_data_df parameter is empty
    if stock_symbol == "":
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
            WHERE ticker = "{stock_symbol}"
            """
        stock_price_data_df = pd.read_sql(sql=quary, con=db_con)
        # Check if the stock_info_data_df is empty
        if stock_price_data_df.empty:
            return False
        else:
            return True

    except Exception as e:
        raise KeyError(f". Error: {e}") from e

def export_to_stock_price_data(stock_price_data_df=""):
    """
    """
    if stock_price_data_df.empty:
        raise ValueError("The stock_price_data_df parameter cannot be empty.")

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
        stock_price_data_df.to_sql(name="stock_price_data", con=db_con, index=False, if_exists="append")

    except Exception as e:
        raise KeyError(f"Could not export from stock_price_data_df to stock_price_data in the database. Error: {e}") from e

def import_from_stock_price_data(amount = 1):
    """
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
        quary = f"""SELECT * FROM
        (SELECT * FROM stock_price_data ORDER BY date DESC LIMIT {amount}) AS temp
        ORDER BY date ASC
        """
        stock_price_data_df = pd.read_sql(sql=quary, con=db_con)
        return stock_price_data_df

    except Exception as e:
        raise KeyError(f"Could not import from stock_price_data in the database to stock_price_data_df . Error: {e}") from e

# Run the main function
if __name__ == "__main__":
    symbol = "NOVO-B"
    if control_if_stock_exists(symbol) == False:
        print("Stock does not exist")
    elif control_if_stock_exists(symbol) == True:
        print("Stock exists")
