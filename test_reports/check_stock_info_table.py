"""Check what tickers are in the stock_info_data table."""
import fetch_secrets
import db_connectors
import pandas as pd

db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()
db_con = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)

query = "SELECT ticker FROM stock_info_data ORDER BY ticker"
tickers = pd.read_sql(query, db_con)
print(f"Tickers in stock_info_data table: {len(tickers)}")
print(tickers)

db_con.close()
