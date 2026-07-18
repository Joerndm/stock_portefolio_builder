"""Quick check of feature completeness for first 3 tickers."""
import fetch_secrets
import db_connectors
import pandas as pd

db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()
db_con = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)

query = '''SELECT ticker, 
    COUNT(*) as total_rows,
    COUNT(sma_5) as sma_5_count,
    COUNT(sma_20) as sma_20_count,
    COUNT(sma_200) as sma_200_count,
    COUNT(std_Div_40) as std_40_count,
    COUNT(bollinger_Band_40_2STD) as bb_40_count
FROM stock_price_data 
WHERE ticker IN ('^VIX', 'AMBU-B.CO', 'DEMANT.CO')
GROUP BY ticker'''

result = pd.read_sql(query, db_con)
print(result.to_string())

print("\n" + "="*60)
print("Checking feature coverage percentages...")
print("="*60)
for _, row in result.iterrows():
    ticker = row['ticker']
    total = row['total_rows']
    sma5_pct = (row['sma_5_count'] / total * 100) if total > 0 else 0
    sma200_pct = (row['sma_200_count'] / total * 100) if total > 0 else 0
    print(f"\n{ticker}:")
    print(f"  Total rows: {total}")
    print(f"  sma_5:  {row['sma_5_count']:4d} ({sma5_pct:.1f}%)")
    print(f"  sma_200: {row['sma_200_count']:4d} ({sma200_pct:.1f}%)")
    print(f"  std_Div_40: {row['std_40_count']:4d}")
    print(f"  bollinger_Band_40_2STD: {row['bb_40_count']:4d}")
