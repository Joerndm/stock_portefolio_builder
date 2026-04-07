"""
Prediction Accuracy Analysis Tool

Compares predicted prices against actual prices for all tickers with predictions,
computes accuracy metrics, identifies systematic biases, and generates a report.

Usage:
    python analyze_prediction_accuracy.py
    python analyze_prediction_accuracy.py --ticker AAPL
    python analyze_prediction_accuracy.py --output report.txt
"""
import argparse
import os
import numpy as np
import pandas as pd
from datetime import datetime
from sqlalchemy import text as sa_text
import matplotlib.pyplot as plt

import fetch_secrets
import db_connectors


def get_predictions(engine, ticker=None):
    """Get all predictions from stock_prediction_extended."""
    q = """
        SELECT prediction_date, ticker, prediction_horizon_days,
               target_date, predicted_price, current_price,
               predicted_return, confidence_lower_5, confidence_upper_95,
               model_type, prediction_std
        FROM stock_prediction_extended
    """
    params = {}
    if ticker:
        q += " WHERE ticker = :ticker"
        params['ticker'] = ticker
    q += " ORDER BY ticker, prediction_date, prediction_horizon_days"
    return pd.read_sql(sa_text(q), engine, params=params)


def get_actual_prices(engine, tickers, date_start, date_end):
    """Get actual closing prices for the given tickers and date range."""
    placeholders = ', '.join([f':t{i}' for i in range(len(tickers))])
    q = sa_text(f"""
        SELECT ticker, date, close_Price
        FROM stock_price_data
        WHERE ticker IN ({placeholders})
          AND date BETWEEN :start AND :end
        ORDER BY ticker, date
    """)
    params = {f't{i}': t for i, t in enumerate(tickers)}
    params['start'] = date_start
    params['end'] = date_end
    return pd.read_sql(q, engine, params=params)


def analyze_accuracy(predictions_df, actuals_df):
    """
    Compare predictions against actual prices.
    Returns a DataFrame with accuracy metrics per ticker per horizon.
    """
    # Merge predictions with actual prices at target_date
    actuals_df['date'] = pd.to_datetime(actuals_df['date'])
    predictions_df['target_date'] = pd.to_datetime(predictions_df['target_date'])
    predictions_df['prediction_date'] = pd.to_datetime(predictions_df['prediction_date'])

    merged = predictions_df.merge(
        actuals_df.rename(columns={'date': 'target_date', 'close_Price': 'actual_price'}),
        on=['ticker', 'target_date'],
        how='left'
    )

    # Only analyze predictions where we now have actual data
    merged = merged.dropna(subset=['actual_price'])

    if len(merged) == 0:
        return None, None, None

    # Calculate metrics
    merged['price_error'] = merged['predicted_price'] - merged['actual_price']
    merged['price_error_pct'] = merged['price_error'] / merged['actual_price'] * 100
    merged['abs_error_pct'] = merged['price_error_pct'].abs()
    merged['actual_return'] = (merged['actual_price'] - merged['current_price']) / merged['current_price']
    merged['return_error'] = merged['predicted_return'] - merged['actual_return']
    merged['direction_correct'] = (
        (merged['predicted_return'] > 0) == (merged['actual_return'] > 0)
    ).astype(int)
    merged['in_confidence'] = (
        (merged['actual_price'] >= merged['confidence_lower_5']) &
        (merged['actual_price'] <= merged['confidence_upper_95'])
    ).astype(int)

    # Aggregate metrics per horizon
    horizon_metrics = merged.groupby('prediction_horizon_days').agg(
        n_predictions=('ticker', 'count'),
        n_tickers=('ticker', 'nunique'),
        mean_error_pct=('price_error_pct', 'mean'),
        median_error_pct=('price_error_pct', 'median'),
        mae_pct=('abs_error_pct', 'mean'),
        rmse_pct=('abs_error_pct', lambda x: np.sqrt((x ** 2).mean())),
        direction_accuracy=('direction_correct', 'mean'),
        confidence_coverage=('in_confidence', 'mean'),
        mean_predicted_return=('predicted_return', 'mean'),
        mean_actual_return=('actual_return', 'mean'),
        std_actual_return=('actual_return', 'std'),
    ).round(4)

    # Per-ticker metrics
    ticker_metrics = merged.groupby(['ticker', 'prediction_horizon_days']).agg(
        n_predictions=('prediction_date', 'count'),
        mean_error_pct=('price_error_pct', 'mean'),
        mae_pct=('abs_error_pct', 'mean'),
        direction_accuracy=('direction_correct', 'mean'),
        confidence_coverage=('in_confidence', 'mean'),
        mean_predicted_return=('predicted_return', 'mean'),
        mean_actual_return=('actual_return', 'mean'),
    ).round(4)

    return merged, horizon_metrics, ticker_metrics


def identify_biases(merged_df):
    """Identify systematic prediction biases."""
    biases = []

    if merged_df is None or len(merged_df) == 0:
        return biases

    # 1. Overall direction bias
    mean_error = merged_df['price_error_pct'].mean()
    if mean_error > 2:
        biases.append(f"SYSTEMATIC UPWARD BIAS: predictions are {mean_error:.1f}% too high on average")
    elif mean_error < -2:
        biases.append(f"SYSTEMATIC DOWNWARD BIAS: predictions are {abs(mean_error):.1f}% too low on average")

    # 2. Error growth with horizon
    for horizon in sorted(merged_df['prediction_horizon_days'].unique()):
        h_data = merged_df[merged_df['prediction_horizon_days'] == horizon]
        mae = h_data['abs_error_pct'].mean()
        biases.append(f"  {horizon}D horizon: MAE={mae:.1f}%, direction={h_data['direction_correct'].mean():.1%}")

    # 3. Confidence interval calibration
    in_ci = merged_df['in_confidence'].mean()
    if in_ci < 0.80:
        biases.append(f"UNDERCONFIDENT INTERVALS: only {in_ci:.1%} of actuals within 90% CI (expected ~90%)")
    elif in_ci > 0.98:
        biases.append(f"OVERWIDE INTERVALS: {in_ci:.1%} of actuals within 90% CI (intervals too wide)")

    # 4. Sector/market bias
    mean_pred_return = merged_df['predicted_return'].mean()
    mean_actual_return = merged_df['actual_return'].mean()
    if abs(mean_pred_return - mean_actual_return) > 0.05:
        biases.append(
            f"RETURN LEVEL MISMATCH: predicted avg return={mean_pred_return:.2%}, "
            f"actual avg return={mean_actual_return:.2%}"
        )

    return biases


def generate_report(horizon_metrics, ticker_metrics, biases, merged_df, output_file=None):
    """Generate and print/save the accuracy report."""
    lines = []
    lines.append("=" * 80)
    lines.append("PREDICTION ACCURACY ANALYSIS REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)

    if horizon_metrics is None:
        lines.append("\nNo predictions with matching actual prices found.")
        lines.append("Predictions may target future dates where no actuals exist yet.")
        report = "\n".join(lines)
        print(report)
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
        return

    lines.append("\n--- OVERALL METRICS BY PREDICTION HORIZON ---")
    lines.append(horizon_metrics.to_string())

    lines.append("\n\n--- SYSTEMATIC BIASES ---")
    for bias in biases:
        lines.append(f"  {bias}")

    # Top/bottom performers
    if ticker_metrics is not None and len(ticker_metrics) > 0:
        lines.append("\n\n--- TOP 10 MOST ACCURATE TICKERS (lowest MAE%) ---")
        best = ticker_metrics.sort_values('mae_pct').head(10)
        lines.append(best.to_string())

        lines.append("\n\n--- TOP 10 LEAST ACCURATE TICKERS (highest MAE%) ---")
        worst = ticker_metrics.sort_values('mae_pct', ascending=False).head(10)
        lines.append(worst.to_string())

        # Directional accuracy leaders
        lines.append("\n\n--- TOP 10 BEST DIRECTIONAL ACCURACY ---")
        best_dir = ticker_metrics.sort_values('direction_accuracy', ascending=False).head(10)
        lines.append(best_dir.to_string())

    # Known prediction pipeline issues
    lines.append("\n\n" + "=" * 80)
    lines.append("PREDICTION PIPELINE ACCURACY ANALYSIS")
    lines.append("=" * 80)
    lines.append("""
KEY FINDINGS FROM CODE REVIEW:

1. DATA LEAKAGE (FIXED): The train/test split used random shuffling instead of 
   chronological ordering. This caused the model to train on "future" data and 
   test on "past" data, severely inflating training metrics. 
   -> FIX APPLIED: Now uses time-based split preserving chronological order.

2. FEATURE SCALER LEAKAGE (FIXED): MinMaxScaler for X features was fit on ALL 
   data (train+val+test combined), leaking distribution information.
   -> FIX APPLIED: Now fits scaler on training data only.

3. RECURSIVE ERROR ACCUMULATION: Each future day's prediction feeds into the 
   next day's features (close_Price, SMAs, RSI, MACD, etc.). Small errors 
   compound exponentially over the 90-day forecast horizon.
   -> RECOMMENDATION: Use direct multi-step prediction or limit recursive horizon.

4. FEATURE RECALCULATION FROM PREDICTIONS: Technical indicators (SMA, EMA, RSI, 
   MACD, Bollinger Bands) are recomputed from predicted (not actual) prices during 
   forecasting. This amplifies bias as indicators drift from reality.
   -> RECOMMENDATION: Weight recent features more heavily; consider freezing 
   some indicators at their last known values.

5. ARTIFICIAL NOISE INJECTION: Post-prediction heuristics (add_prediction_uncertainty, 
   apply_mean_reversion, apply_directional_balance) add random noise and corrections.
   -> RECOMMENDATION: Remove or reduce these; let the model speak for itself.

6. VOLUME INDICATORS STALE: Volume-based features (SMA, EMA, ratio, VWAP, OBV) 
   are simply carried forward during future prediction as constants.
   -> RECOMMENDATION: Either exclude volume features from future prediction or 
   model volume separately.

7. ENSEMBLE INCONSISTENCY: Historical predictions use simple (RF+XGB)/2, while 
   future predictions use inverse-MSE weighted (TCN+RF+XGB).
   -> RECOMMENDATION: Use consistent ensemble method throughout.

EXPECTED IMPACT OF FIXES:
- The time-based split and scaler fix will likely REDUCE reported training metrics 
  (since the model can no longer "cheat" on test data), but IMPROVE real-world 
  prediction accuracy since the model learns proper temporal patterns.
- Models will need retraining with the new split to see the benefit.
""")

    report = "\n".join(lines)
    print(report)
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {output_file}")


def plot_accuracy_summary(merged_df, output_dir):
    """Generate accuracy visualization plots."""
    if merged_df is None or len(merged_df) == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Error distribution by horizon
    horizons = sorted(merged_df['prediction_horizon_days'].unique())
    data = [merged_df[merged_df['prediction_horizon_days'] == h]['price_error_pct'].values for h in horizons]
    axes[0, 0].boxplot(data, labels=[f'{h}D' for h in horizons])
    axes[0, 0].set_title('Price Error (%) by Prediction Horizon')
    axes[0, 0].set_ylabel('Error (%)')
    axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)

    # 2. Direction accuracy by horizon
    dir_acc = merged_df.groupby('prediction_horizon_days')['direction_correct'].mean()
    axes[0, 1].bar([f'{h}D' for h in dir_acc.index], dir_acc.values, color='steelblue')
    axes[0, 1].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random (50%)')
    axes[0, 1].set_title('Directional Accuracy by Horizon')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()

    # 3. Predicted vs Actual returns scatter
    axes[1, 0].scatter(
        merged_df['actual_return'] * 100,
        merged_df['predicted_return'] * 100,
        alpha=0.3, s=10
    )
    lim = max(abs(merged_df['actual_return'].max()), abs(merged_df['predicted_return'].max())) * 100
    axes[1, 0].plot([-lim, lim], [-lim, lim], 'r--', alpha=0.5)
    axes[1, 0].set_xlabel('Actual Return (%)')
    axes[1, 0].set_ylabel('Predicted Return (%)')
    axes[1, 0].set_title('Predicted vs Actual Returns')

    # 4. MAE by ticker (top 20 worst)
    ticker_mae = merged_df.groupby('ticker')['abs_error_pct'].mean().sort_values(ascending=False).head(20)
    axes[1, 1].barh(ticker_mae.index, ticker_mae.values, color='coral')
    axes[1, 1].set_xlabel('Mean Absolute Error (%)')
    axes[1, 1].set_title('Top 20 Least Accurate Tickers')
    axes[1, 1].invert_yaxis()

    plt.tight_layout()
    graph_path = os.path.join(output_dir, "prediction_accuracy_analysis.png")
    plt.savefig(graph_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[GRAPH] Saved: {graph_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze prediction accuracy')
    parser.add_argument('--ticker', help='Analyze specific ticker only')
    parser.add_argument('--output', default='prediction_accuracy_report.txt', help='Output report file')
    args = parser.parse_args()

    db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()
    engine = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    graphs_dir = os.path.join(script_dir, "generated_graphs")

    # Get predictions
    print("Loading predictions...")
    predictions = get_predictions(engine, args.ticker)

    if len(predictions) == 0:
        print("No predictions found in database.")
        print("Run the prediction pipeline first: python price_predictor.py")
        return

    print(f"Found {len(predictions)} predictions for {predictions['ticker'].nunique()} tickers")

    # Get actual prices
    tickers = predictions['ticker'].unique().tolist()
    date_start = predictions['target_date'].min()
    date_end = predictions['target_date'].max()
    print(f"Loading actual prices for {date_start} to {date_end}...")
    actuals = get_actual_prices(engine, tickers, str(date_start), str(date_end))
    print(f"Found {len(actuals)} actual price records")

    # Analyze
    result = analyze_accuracy(predictions, actuals)
    merged_df, horizon_metrics, ticker_metrics = result
    if merged_df is None:
        print("\nNo predictions have matching actual prices yet.")
        print("This means prediction target dates are still in the future.")
        print("Run this analysis again after those dates have passed.")

        # Still generate the pipeline analysis report
        generate_report(None, None, [], None, args.output)
        return
    biases = identify_biases(merged_df)

    # Generate report
    generate_report(horizon_metrics, ticker_metrics, biases, merged_df, args.output)

    # Generate plots
    plot_accuracy_summary(merged_df, graphs_dir)


if __name__ == '__main__':
    main()
