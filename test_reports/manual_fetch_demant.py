"""Manual trigger for DEMANT.CO to test the full pipeline."""
import fetch_secrets
import db_connectors
import db_interactions
import stock_data_fetch as sdf
import yfinance as yf

db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()
db_con = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)

ticker = "DEMANT.CO"
print(f"Testing full pipeline for {ticker}")

# Check if stock price data exists
has_price_data = db_interactions.does_stock_exists_stock_price_data(ticker)
print(f"does_stock_exists_stock_price_data({ticker}): {has_price_data}")

if not has_price_data:
    print(f"\nFetching stock data for {ticker}...")
    stock_info = yf.Ticker(ticker).info
    
    # Fetch and process data
    stock_price_data_df = sdf.fetch_stock_price_data(ticker)
    print(f"Fetched {len(stock_price_data_df)} rows")
    
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
    print(f"After dropna(): {len(stock_price_data_df)} rows remaining")
    
    if not stock_price_data_df.empty:
        print("\nExporting to database...")
        db_interactions.export_stock_price_data(stock_price_data_df)
        print("✅ Successfully exported to database!")
        
        # Verify
        print("\nVerifying data in database...")
        import pandas as pd
        query = f"SELECT COUNT(*) as cnt FROM stock_price_data WHERE ticker = '{ticker}'"
        result = pd.read_sql(query, db_con)
        print(f"Rows in database for {ticker}: {result['cnt'].iloc[0]}")
    else:
        print("⚠️ WARNING: DataFrame is empty after dropna(), skipping export")
else:
    print(f"✅ Stock price data already exists for {ticker}")
