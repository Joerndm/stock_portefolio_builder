"""Comprehensive script to populate ALL database tables including stock_ratio_data."""
import fetch_secrets
import db_connectors
import db_interactions
import stock_data_fetch as sdf
import pandas as pd
import datetime

db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()
db_con = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)

# Get list of all tickers
all_tickers = [
    '^VIX', 'AMBU-B.CO', 'BAVA.CO', 'CARL-B.CO', 'COLO-B.CO', 'DANSKE.CO', 'DEMANT.CO',
    'DSV.CO', 'FLS.CO', 'GMAB.CO', 'GN.CO', 'ISS.CO', 'JYSK.CO', 'MAERSK-A.CO', 'MAERSK-B.CO',
    'NDA-DK.CO', 'NKT.CO', 'NOVO-B.CO', 'ORSTED.CO', 'PNDORA.CO', 'RBREW.CO', 'ROCK-B.CO',
    'TRYG.CO', 'VWS.CO', 'ZEAL.CO'
]

print("="*80)
print("COMPREHENSIVE DATABASE POPULATION")
print("="*80)

# Track statistics
stats = {
    'financial_data_fetched': 0,
    'financial_data_exists': 0,
    'ratio_data_calculated': 0,
    'ratio_data_exists': 0,
    'errors': []
}

for i, ticker in enumerate(all_tickers, 1):
    print(f"\n[{i}/{len(all_tickers)}] Processing {ticker}...")
    print("-"*80)
    
    try:
        # ====================================================================
        # Step 1: Financial Data (Income Statement, Balance Sheet, Cash Flow)
        # ====================================================================
        print(f"  Step 1: Checking financial data...")
        
        if not db_interactions.does_stock_exists_stock_income_stmt_data(ticker):
            print(f"    Fetching financial data for {ticker}...")
            full_stock_financial_data_df = sdf.fetch_stock_financial_data(ticker)
            db_interactions.export_stock_financial_data(full_stock_financial_data_df)
            print(f"    Financial data exported to database")
            stats['financial_data_fetched'] += 1
        else:
            print(f"    Financial data already exists")
            stats['financial_data_exists'] += 1
        
        # ====================================================================
        # Step 2: Stock Ratio Data (P/S, P/E, P/B, P/FCF)
        # ====================================================================
        print(f"  Step 2: Checking ratio data...")
        
        if not db_interactions.does_stock_exists_stock_ratio_data(ticker):
            print(f"    Calculating ratio data for {ticker}...")
            
            # Import financial data
            TABEL_NAME = "stock_income_stmt_data"
            query = f"""SELECT COUNT(financial_Statement_Date)
                        FROM {TABEL_NAME}
                        WHERE ticker = '{ticker}'
                        """
            entry_amount = pd.read_sql(sql=query, con=db_con)
            
            if entry_amount.iloc[0, 0] > 0:
                # Import financial and price data
                full_stock_financial_data_df = db_interactions.import_stock_financial_data(
                    amount=entry_amount.iloc[0, 0], 
                    stock_ticker=ticker
                )
                full_stock_financial_data_df = full_stock_financial_data_df.dropna(axis=1)
                date = full_stock_financial_data_df.iloc[0]["date"]
                
                TABEL_NAME = "stock_price_data"
                query = f"""SELECT *
                            FROM {TABEL_NAME}
                            WHERE ticker = '{ticker}' AND date >= '{date}'
                            """
                stock_price_data_df = pd.read_sql(sql=query, con=db_con)
                
                # Calculate ratios
                combined_stock_data_df = sdf.combine_stock_data(stock_price_data_df, full_stock_financial_data_df)
                combined_stock_data_df = sdf.calculate_ratios(combined_stock_data_df)
                stock_ratio_data_df = combined_stock_data_df[['date', 'ticker', 'P/S', 'P/E', 'P/B', 'P/FCF']]
                stock_ratio_data_df = stock_ratio_data_df.rename(columns={
                    "P/S": "p_s", "P/E": "p_e", "P/B": "p_b", "P/FCF": "p_fcf"
                })
                stock_ratio_data_df = sdf.drop_nan_values(stock_ratio_data_df)
                
                # Export to database
                db_interactions.export_stock_ratio_data(stock_ratio_data_df)
                print(f"    Ratio data exported: {len(stock_ratio_data_df)} rows")
                stats['ratio_data_calculated'] += 1
            else:
                print(f"    No financial data available, skipping ratio calculation")
        else:
            print(f"    Ratio data already exists")
            stats['ratio_data_exists'] += 1
            
        print(f"  Completed {ticker}")
        
    except Exception as e:
        error_msg = f"ERROR processing {ticker}: {str(e)}"
        print(f"  {error_msg}")
        stats['errors'].append(error_msg)
        continue

# ====================================================================
# Final Report
# ====================================================================
print("\n" + "="*80)
print("FINAL REPORT")
print("="*80)
print(f"\nFinancial Data:")
print(f"  Fetched: {stats['financial_data_fetched']}")
print(f"  Already existed: {stats['financial_data_exists']}")

print(f"\nRatio Data:")
print(f"  Calculated: {stats['ratio_data_calculated']}")
print(f"  Already existed: {stats['ratio_data_exists']}")

if stats['errors']:
    print(f"\nErrors: {len(stats['errors'])}")
    for error in stats['errors']:
        print(f"  - {error}")

# Verify final state
print("\n" + "="*80)
print("DATABASE VERIFICATION")
print("="*80)

tables = ['stock_info_data', 'stock_price_data', 'stock_income_stmt_data', 
          'stock_cash_flow_data', 'stock_ratio_data']

for table in tables:
    query = f"SELECT COUNT(DISTINCT ticker) as ticker_count, COUNT(*) as total_rows FROM {table}"
    try:
        result = pd.read_sql(query, db_con)
        tickers = result['ticker_count'].iloc[0]
        rows = result['total_rows'].iloc[0]
        print(f"{table:30s}: {tickers:2d} tickers, {rows:5d} rows")
    except Exception as e:
        print(f"{table:30s}: ERROR - {str(e)}")

print("\n" + "="*80)
print("POPULATION COMPLETE")
print("="*80)
