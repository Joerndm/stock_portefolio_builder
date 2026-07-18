"""Check feature completeness in the database after data fetch."""

import mysql.connector
import pandas as pd
from fetch_secrets import secret_import

print("="*80)
print("FEATURE COMPLETENESS CHECK")
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
    print("\n✅ Database connection established")
except Exception as e:
    print(f"\n❌ Database connection failed: {e}")
    exit(1)

# Get list of tickers
try:
    query = "SELECT ticker FROM stock_info_data ORDER BY ticker"
    tickers_df = pd.read_sql(query, db_con)
    tickers = tickers_df['ticker'].tolist()
    print(f"✅ Found {len(tickers)} tickers")
except Exception as e:
    print(f"❌ Failed to get ticker list: {e}")
    db_con.close()
    exit(1)

# Define expected features
new_features = [
    'sma_5', 'sma_20', 'sma_40', 'sma_120', 'sma_200',
    'ema_5', 'ema_20', 'ema_40', 'ema_120', 'ema_200',
    'std_Div_5', 'std_Div_20', 'std_Div_40', 'std_Div_120', 'std_Div_200',
    'bollinger_Band_5_2STD', 'bollinger_Band_20_2STD', 'bollinger_Band_40_2STD',
    'bollinger_Band_120_2STD', 'bollinger_Band_200_2STD'
]

print("\n" + "="*80)
print("TICKER-BY-TICKER ANALYSIS")
print("="*80 + "\n")

results = []

for ticker in tickers:
    try:
        # Get row count
        query = f"SELECT COUNT(*) as cnt FROM stock_price_data WHERE ticker = '{ticker}'"
        row_count = pd.read_sql(query, db_con).iloc[0]['cnt']
        
        if row_count == 0:
            print(f"{ticker:12s} │ No data")
            results.append({'ticker': ticker, 'rows': 0, 'features': 0, 'status': '❌ NO DATA'})
            continue
        
        # Count non-null values for each feature
        feature_counts = {}
        for feature in new_features:
            query = f"SELECT COUNT({feature}) as cnt FROM stock_price_data WHERE ticker = '{ticker}' AND {feature} IS NOT NULL"
            count = pd.read_sql(query, db_con).iloc[0]['cnt']
            feature_counts[feature] = count
        
        # Determine how many features have data
        features_with_data = sum(1 for c in feature_counts.values() if c > 0)
        
        # Calculate average coverage for features with data
        if features_with_data > 0:
            avg_coverage = sum(c for c in feature_counts.values() if c > 0) / features_with_data / row_count * 100
        else:
            avg_coverage = 0
        
        # Determine status
        if features_with_data == 20:
            status = f"✅ {features_with_data}/20 ({avg_coverage:.0f}%)"
        elif features_with_data >= 15:
            status = f"⚠️  {features_with_data}/20 ({avg_coverage:.0f}%)"
        else:
            status = f"❌ {features_with_data}/20 ({avg_coverage:.0f}%)"
        
        print(f"{ticker:12s} │ {row_count:4d} rows │ {status}")
        
        # Show specific feature coverage for last ticker (as sample)
        if ticker == tickers[-1]:
            print(f"\n  Sample coverage for {ticker}:")
            for feature in ['sma_5', 'sma_40', 'sma_200', 'std_Div_40', 'bollinger_Band_40_2STD']:
                count = feature_counts.get(feature, 0)
                pct = (count / row_count * 100) if row_count > 0 else 0
                print(f"    {feature:25s}: {count:3d}/{row_count} ({pct:5.1f}%)")
        
        results.append({
            'ticker': ticker,
            'rows': row_count,
            'features': features_with_data,
            'status': status
        })
        
    except Exception as e:
        print(f"{ticker:12s} │ Error: {str(e)[:50]}")
        results.append({'ticker': ticker, 'rows': 0, 'features': 0, 'status': f'❌ ERROR'})

# Summary
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80 + "\n")

results_df = pd.DataFrame(results)

complete = len(results_df[results_df['features'] == 20])
partial = len(results_df[(results_df['features'] > 0) & (results_df['features'] < 20)])
failed = len(results_df[results_df['features'] == 0])

print(f"Total tickers:     {len(results_df)}")
print(f"  ✅ Complete:      {complete} (all 20 features)")
print(f"  ⚠️  Partial:       {partial} (some features)")
print(f"  ❌ No data:       {failed} (no features)")

# Sample data validation
print("\n" + "="*80)
print("SAMPLE DATA VALIDATION")
print("="*80 + "\n")

try:
    query = """
        SELECT date, ticker, close_Price, sma_5, sma_40, sma_200,
               std_Div_5, std_Div_40, bollinger_Band_40_2STD
        FROM stock_price_data
        WHERE ticker = 'DEMANT.CO'
        ORDER BY date DESC
        LIMIT 5
    """
    sample = pd.read_sql(query, db_con)
    
    if not sample.empty:
        print("Latest 5 rows for DEMANT.CO:")
        print(sample.to_string(index=False))
        
        # Validate calculations
        print("\n✓ Validation checks:")
        latest = sample.iloc[0]
        
        # Check Bollinger Band = 4 * std_Div
        if pd.notna(latest['std_Div_40']) and pd.notna(latest['bollinger_Band_40_2STD']):
            expected = 4.0 * latest['std_Div_40']
            actual = latest['bollinger_Band_40_2STD']
            if abs(actual - expected) < 0.01:
                print(f"  ✅ bollinger_Band_40_2STD = 4 * std_Div_40")
                print(f"     ({actual:.2f} = 4 * {latest['std_Div_40']:.2f})")
            else:
                print(f"  ❌ bollinger_Band_40_2STD calculation error")
                print(f"     Expected: {expected:.2f}, Got: {actual:.2f}")
        
        # Check SMA values are reasonable
        if pd.notna(latest['sma_5']) and pd.notna(latest['close_Price']):
            diff_pct = abs(latest['sma_5'] - latest['close_Price']) / latest['close_Price'] * 100
            if diff_pct < 10:
                print(f"  ✅ sma_5 within 10% of close_Price ({diff_pct:.1f}%)")
            else:
                print(f"  ⚠️  sma_5 differs by {diff_pct:.1f}% from close_Price")
        
        # Check std_Div is positive
        if pd.notna(latest['std_Div_40']) and latest['std_Div_40'] > 0:
            print(f"  ✅ std_Div_40 is positive ({latest['std_Div_40']:.2f})")
        
    else:
        print("❌ No data found for DEMANT.CO")

except Exception as e:
    print(f"❌ Sample validation failed: {e}")

db_con.close()

print("\n" + "="*80)
print("CHECK COMPLETE")
print("="*80)

if failed == 0 and partial == 0:
    print("\n🎉 SUCCESS: All tickers have complete feature sets!")
    exit(0)
elif failed == 0:
    print(f"\n⚠️  PARTIAL: {complete} complete, {partial} need attention")
    exit(0)
else:
    print(f"\n❌ ISSUES: {failed} tickers have no data")
    exit(1)
