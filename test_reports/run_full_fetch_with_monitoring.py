"""
Monitor feature calculations during full data fetch.
This script wraps the stock_data_fetch.py execution and monitors feature completeness.
"""

import subprocess
import sys
import mysql.connector
import pandas as pd
from fetch_secrets import secret_import

print("="*80)
print("FULL DATA FETCH WITH FEATURE MONITORING")
print("="*80)

# Connect to database
try:
    db_host, db_user, db_password, db_name = secret_import()
    db_con = mysql.connector.connect(
        host=db_host,
        user=db_user,
        password=db_password,
        database=db_name
    )
    print("\nDatabase connection: OK")
except Exception as e:
    print(f"\nDatabase connection failed: {e}")
    sys.exit(1)

# Get list of tickers from database
try:
    query = "SELECT ticker FROM stock_info_data ORDER BY ticker"
    tickers_df = pd.read_sql(query, db_con)
    tickers = tickers_df['ticker'].tolist()
    print(f"\nFound {len(tickers)} tickers in database:")
    for ticker in tickers:
        print(f"  - {ticker}")
except Exception as e:
    print(f"Failed to get ticker list: {e}")
    db_con.close()
    sys.exit(1)

print("\n" + "="*80)
print("STARTING DATA FETCH")
print("="*80)

# Run stock_data_fetch.py
print("\nExecuting stock_data_fetch.py in conda environment fetch_stock_data_py_3_12...")
print("(This will take several minutes depending on data volume)\n")

try:
    # Run in the correct conda environment
    result = subprocess.run(
        ["conda", "run", "-n", "fetch_stock_data_py_3_12", "python", "stock_data_fetch.py"],
        capture_output=True,
        text=True,
        cwd="C:\\Users\\joern\\OneDrive\\Dokumenter\\Privat\\Aktie_database\\code\\stock_portefolio_builder"
    )
    
    # Print the output
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    if result.returncode != 0:
        print(f"\n⚠️  WARNING: Script exited with code {result.returncode}")
except Exception as e:
    print(f"Error running script: {e}")
    db_con.close()
    sys.exit(1)

print("\n" + "="*80)
print("POST-FETCH FEATURE VALIDATION")
print("="*80)

# Check feature completeness for each ticker
all_features = [
    'sma_5', 'sma_20', 'sma_40', 'sma_120', 'sma_200',
    'ema_5', 'ema_20', 'ema_40', 'ema_120', 'ema_200',
    'std_Div_5', 'std_Div_20', 'std_Div_40', 'std_Div_120', 'std_Div_200',
    'bollinger_Band_5_2STD', 'bollinger_Band_20_2STD', 'bollinger_Band_40_2STD',
    'bollinger_Band_120_2STD', 'bollinger_Band_200_2STD'
]

print("\nChecking feature completeness for each ticker:\n")

summary_data = []

for ticker in tickers:
    try:
        # Get row count
        query = f"SELECT COUNT(*) as row_count FROM stock_price_data WHERE ticker = '{ticker}'"
        row_count = pd.read_sql(query, db_con).iloc[0]['row_count']
        
        if row_count == 0:
            print(f"{ticker:12s} - No data found")
            summary_data.append({
                'ticker': ticker,
                'rows': 0,
                'features_ok': 0,
                'status': '❌ NO DATA'
            })
            continue
        
        # Check feature presence and non-null counts
        query = f"SELECT * FROM stock_price_data WHERE ticker = '{ticker}' LIMIT 1"
        sample = pd.read_sql(query, db_con)
        
        features_present = []
        features_missing = []
        
        for feature in all_features:
            if feature in sample.columns:
                # Count non-null values for this feature
                query = f"SELECT COUNT({feature}) as non_null_count FROM stock_price_data WHERE ticker = '{ticker}' AND {feature} IS NOT NULL"
                non_null_count = pd.read_sql(query, db_con).iloc[0]['non_null_count']
                pct = (non_null_count / row_count) * 100 if row_count > 0 else 0
                
                if non_null_count > 0:
                    features_present.append(f"{feature}:{pct:.0f}%")
                else:
                    features_missing.append(feature)
            else:
                features_missing.append(feature)
        
        # Determine status
        if len(features_missing) == 0:
            status = "✅ ALL OK"
        elif len(features_present) >= 15:
            status = f"⚠️  {len(features_present)}/20"
        else:
            status = f"❌ {len(features_present)}/20"
        
        print(f"{ticker:12s} - {row_count:4d} rows - {status}")
        
        # Show missing features if any
        if features_missing and len(features_missing) <= 5:
            print(f"             Missing: {', '.join(features_missing)}")
        
        summary_data.append({
            'ticker': ticker,
            'rows': row_count,
            'features_ok': len(features_present),
            'status': status
        })
        
    except Exception as e:
        print(f"{ticker:12s} - Error: {e}")
        summary_data.append({
            'ticker': ticker,
            'rows': 0,
            'features_ok': 0,
            'status': f'❌ ERROR: {str(e)[:30]}'
        })

# Summary statistics
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

summary_df = pd.DataFrame(summary_data)

total_tickers = len(summary_df)
complete_tickers = len(summary_df[summary_df['features_ok'] == 20])
partial_tickers = len(summary_df[(summary_df['features_ok'] > 0) & (summary_df['features_ok'] < 20)])
failed_tickers = len(summary_df[summary_df['features_ok'] == 0])

print(f"\nTotal tickers: {total_tickers}")
print(f"  ✅ Complete (20/20 features): {complete_tickers}")
print(f"  ⚠️  Partial (<20 features):     {partial_tickers}")
print(f"  ❌ Failed (0 features):        {failed_tickers}")

# Sample validation - check actual values for one ticker
print("\n" + "="*80)
print("SAMPLE VALUE VALIDATION (DEMANT.CO)")
print("="*80)

try:
    query = """
        SELECT date, close_Price, sma_5, sma_40, sma_200, 
               std_Div_5, std_Div_40, bollinger_Band_40_2STD
        FROM stock_price_data 
        WHERE ticker = 'DEMANT.CO' 
        ORDER BY date DESC 
        LIMIT 5
    """
    sample_df = pd.read_sql(query, db_con)
    print(sample_df.to_string(index=False))
    
    # Validate that values are reasonable
    print("\nValue checks:")
    if not sample_df.empty:
        latest = sample_df.iloc[0]
        
        # Check SMA values are close to price
        if pd.notna(latest['sma_5']) and pd.notna(latest['close_Price']):
            sma5_diff_pct = abs(latest['sma_5'] - latest['close_Price']) / latest['close_Price'] * 100
            if sma5_diff_pct < 10:
                print(f"  ✅ sma_5 within 10% of price ({sma5_diff_pct:.1f}%)")
            else:
                print(f"  ⚠️  sma_5 differs by {sma5_diff_pct:.1f}% from price")
        
        # Check std_Div is positive
        if pd.notna(latest['std_Div_40']) and latest['std_Div_40'] > 0:
            print(f"  ✅ std_Div_40 is positive ({latest['std_Div_40']:.2f})")
        
        # Check Bollinger Band = 4 * std_Div
        if pd.notna(latest['std_Div_40']) and pd.notna(latest['bollinger_Band_40_2STD']):
            expected_bb = 4.0 * latest['std_Div_40']
            if abs(latest['bollinger_Band_40_2STD'] - expected_bb) < 0.01:
                print(f"  ✅ bollinger_Band_40_2STD = 4 * std_Div_40")
            else:
                print(f"  ⚠️  bollinger_Band_40_2STD calculation issue")
    
except Exception as e:
    print(f"Could not validate sample: {e}")

db_con.close()

print("\n" + "="*80)
print("MONITORING COMPLETE")
print("="*80)

if complete_tickers == total_tickers:
    print("\n🎉 SUCCESS: All tickers have complete feature sets!")
elif complete_tickers + partial_tickers == total_tickers:
    print(f"\n⚠️  PARTIAL SUCCESS: {complete_tickers}/{total_tickers} complete, {partial_tickers} partial")
else:
    print(f"\n❌ ISSUES FOUND: {failed_tickers} tickers failed")

sys.exit(0 if failed_tickers == 0 else 1)
