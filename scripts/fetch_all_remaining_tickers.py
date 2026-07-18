"""Fetch all remaining tickers that don't have data yet."""
import fetch_secrets
import db_connectors
import db_interactions
import stock_data_fetch as sdf
import yfinance as yf
import pandas as pd

db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()
db_con = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)

# Get list of all tickers that should be processed
all_tickers = [
    '^VIX', 'AMBU-B.CO', 'BAVA.CO', 'CARL-B.CO', 'COLO-B.CO', 'DANSKE.CO', 'DEMANT.CO',
    'DSV.CO', 'FLS.CO', 'GMAB.CO', 'GN.CO', 'ISS.CO', 'JYSK.CO', 'MAERSK-A.CO', 'MAERSK-B.CO',
    'NDA-DK.CO', 'NKT.CO', 'NOVO-B.CO', 'ORSTED.CO', 'PNDORA.CO', 'RBREW.CO', 'ROCK-B.CO',
    'TRYG.CO', 'VWS.CO', 'ZEAL.CO'
]

# Check which ones already have data
print("="*60)
print("Checking which tickers already have data...")
print("="*60)
tickers_to_process = []
for ticker in all_tickers:
    has_data = db_interactions.does_stock_exists_stock_price_data(ticker)
    if not has_data:
        tickers_to_process.append(ticker)
        print(f"❌ {ticker}: No data")
    else:
        print(f"✅ {ticker}: Has data")

print(f"\n{"="*60}")
print(f"Found {len(tickers_to_process)} tickers to process")
print(f"{"="*60}\n")

# Process each ticker
for i, ticker in enumerate(tickers_to_process, 1):
    print(f"\n[{i}/{len(tickers_to_process)}] Processing {ticker}...")
    print("-"*60)
    
    try:
        stock_info = yf.Ticker(ticker).info
        
        # Fetch and process data
        stock_price_data_df = sdf.fetch_stock_price_data(ticker)
        print(f"  Fetched {len(stock_price_data_df)} rows")
        
        stock_price_data_df = sdf.calculate_period_returns(stock_price_data_df)
        stock_price_data_df = sdf.add_technical_indicators(stock_price_data_df)
        stock_price_data_df = sdf.add_volume_indicators(stock_price_data_df)
        
        # Skip volatility indicators for indices
        if stock_info.get("typeDisp") != "Index":
            stock_price_data_df = sdf.add_volatility_indicators(stock_price_data_df)
        
        stock_price_data_df = sdf.calculate_moving_averages(stock_price_data_df)
        stock_price_data_df = sdf.calculate_standard_diviation_value(stock_price_data_df)
        stock_price_data_df = sdf.calculate_bollinger_bands(stock_price_data_df)
        stock_price_data_df = sdf.calculate_momentum(stock_price_data_df)
        
        # Drop rows with NaN only in critical columns
        critical_cols = ['date', 'ticker', 'close_Price', 'open_Price', 'high_Price', 'low_Price']
        stock_price_data_df = stock_price_data_df.dropna(subset=critical_cols)
        
        if not stock_price_data_df.empty:
            db_interactions.export_stock_price_data(stock_price_data_df)
            print(f"  ✅ Successfully exported {len(stock_price_data_df)} rows to database")
        else:
            print(f"  ⚠️  WARNING: DataFrame is empty after dropna(), skipping export")
            
    except Exception as e:
        print(f"  ❌ ERROR processing {ticker}: {str(e)}")
        continue

print(f"\n{"="*60}")
print("BATCH PROCESSING COMPLETE")
print(f"{"="*60}")

# Final summary
print("\nChecking final status...")
for ticker in all_tickers:
    has_data = db_interactions.does_stock_exists_stock_price_data(ticker)
    status = "✅" if has_data else "❌"
    print(f"{status} {ticker}")
