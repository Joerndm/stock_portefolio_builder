"""Check financial data for CARL-B.CO."""
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import db_connectors, fetch_secrets

h,u,p,n = fetch_secrets.secret_import()
con = db_connectors.pandas_mysql_connector(h,u,p,n)

# Check income statement data
query = """
SELECT financial_Statement_Date, ticker, revenue, eps, average_shares 
FROM stock_income_stmt_data 
WHERE ticker='CARL-B.CO' 
ORDER BY financial_Statement_Date DESC 
LIMIT 5
"""
income_df = pd.read_sql(query, con)
print("=== DB INCOME STATEMENT DATA ===")
print(income_df.to_string())

# Get yfinance annual data
import yfinance as yf
ticker = yf.Ticker('CARL-B.CO')
income = ticker.income_stmt
print("\n=== YFINANCE ANNUAL INCOME STATEMENT ===")
if not income.empty:
    print("Columns (dates):", income.columns.tolist())
    if 'Total Revenue' in income.index:
        print("Total Revenue:", income.loc['Total Revenue'].values)
    if 'Basic Average Shares' in income.index:
        print("Basic Average Shares:", income.loc['Basic Average Shares'].values)
    if 'Diluted Average Shares' in income.index:
        print("Diluted Average Shares:", income.loc['Diluted Average Shares'].values)

# Check current sharesOutstanding
info = ticker.info
print(f"\n=== CURRENT SHARES INFO ===")
print(f"sharesOutstanding: {info.get('sharesOutstanding'):,}")
print(f"impliedSharesOutstanding: {info.get('impliedSharesOutstanding'):,}")

# Calculate ratio
db_shares = income_df['average_shares'].iloc[0]
yf_shares = info.get('sharesOutstanding')
print(f"\n=== SHARES COMPARISON ===")
print(f"DB average_shares: {db_shares:,.0f}")
print(f"YF sharesOutstanding: {yf_shares:,}")
print(f"Ratio DB/YF: {db_shares/yf_shares:.4f}")

# What does the TTM calculator use for annual data?
print("\n=== TTM CALCULATOR - ANNUAL DATA FETCH ===")
from ttm_financial_calculator import TTMFinancialCalculator
calc = TTMFinancialCalculator()
annual = calc.fetch_annual_financial_data('CARL-B.CO')
if annual.get('income') is not None and not annual['income'].empty:
    print("Annual income data from yfinance:")
    print(annual['income'].head())
