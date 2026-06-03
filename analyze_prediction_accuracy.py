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
    Handles both ensemble and individual model predictions.
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

    # Filter to ensemble-only for backward-compatible horizon/ticker metrics
    ensemble_merged = merged[merged['model_type'] == 'ensemble']
    if len(ensemble_merged) == 0:
        # Fallback: treat all as ensemble if no model_type distinction
        ensemble_merged = merged

    # Aggregate metrics per horizon (ensemble only)
    horizon_metrics = ensemble_merged.groupby('prediction_horizon_days').agg(
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

    # Per-ticker metrics (ensemble only)
    ticker_metrics = ensemble_merged.groupby(['ticker', 'prediction_horizon_days']).agg(
        n_predictions=('prediction_date', 'count'),
        mean_error_pct=('price_error_pct', 'mean'),
        mae_pct=('abs_error_pct', 'mean'),
        direction_accuracy=('direction_correct', 'mean'),
        confidence_coverage=('in_confidence', 'mean'),
        mean_predicted_return=('predicted_return', 'mean'),
        mean_actual_return=('actual_return', 'mean'),
    ).round(4)

    return merged, horizon_metrics, ticker_metrics


def analyze_model_comparison(merged_df):
    """
    Compare individual model performance against each other and the ensemble.
    Returns a DataFrame with per-model metrics by horizon.
    """
    if merged_df is None or len(merged_df) == 0:
        return None

    model_types = merged_df['model_type'].unique()
    if len(model_types) <= 1:
        return None

    # Per-model metrics by horizon
    model_metrics = merged_df.groupby(['model_type', 'prediction_horizon_days']).agg(
        n_predictions=('ticker', 'count'),
        n_tickers=('ticker', 'nunique'),
        mean_error_pct=('price_error_pct', 'mean'),
        mae_pct=('abs_error_pct', 'mean'),
        rmse_pct=('abs_error_pct', lambda x: np.sqrt((x ** 2).mean())),
        direction_accuracy=('direction_correct', 'mean'),
        mean_predicted_return=('predicted_return', 'mean'),
        mean_actual_return=('actual_return', 'mean'),
    ).round(4)

    return model_metrics


def analyze_ensemble_combinations(merged_df):
    """
    Compute hypothetical ensemble combinations from individual model predictions.
    Compares RF+XGB, RF+XGB+Ridge, RF+XGB+Ridge+SVR, and all models (incl. seq).
    Returns a list of (combo_name, horizon, mae_pct, direction_accuracy, mean_error_pct).
    """
    if merged_df is None or len(merged_df) == 0:
        return None

    model_types = set(merged_df['model_type'].unique())
    individual_models = model_types - {'ensemble'}
    if len(individual_models) < 2:
        return None

    # Pivot: for each (prediction_date, ticker, horizon), get each model's predicted_price
    pivot_cols = ['prediction_date', 'ticker', 'prediction_horizon_days']
    pivoted = merged_df.pivot_table(
        index=pivot_cols,
        columns='model_type',
        values=['predicted_price', 'actual_price', 'current_price'],
        aggfunc='first'
    )

    if pivoted.empty:
        return None

    # Flatten column names
    prices = {}
    for mt in model_types:
        col = ('predicted_price', mt)
        if col in pivoted.columns:
            prices[mt] = pivoted[col]

    anchor = 'ensemble' if 'ensemble' in model_types else sorted(model_types)[0]
    actual = pivoted[('actual_price', anchor)]
    current = pivoted[('current_price', anchor)]
    horizon = pivoted.index.get_level_values('prediction_horizon_days')

    # Define ensemble combinations to test
    combo_defs = [
        ('RF only', ['rf']),
        ('XGB only', ['xgb']),
        ('Ridge only', ['ridge']),
        ('SVR only', ['svr']),
        ('RF+XGB', ['rf', 'xgb']),
        ('RF+XGB+Ridge', ['rf', 'xgb', 'ridge']),
        ('RF+XGB+SVR', ['rf', 'xgb', 'svr']),
        ('RF+XGB+Ridge+SVR', ['rf', 'xgb', 'ridge', 'svr']),
        ('All (incl. seq)', list(individual_models)),
    ]
    # Only add seq-including combos if seq model exists
    if 'seq' in individual_models:
        combo_defs.insert(-1, ('Seq+RF+XGB', ['seq', 'rf', 'xgb']))
        combo_defs.insert(-1, ('Seq+RF+XGB+Ridge+SVR', ['seq', 'rf', 'xgb', 'ridge', 'svr']))

    results = []
    for combo_name, models in combo_defs:
        # Check all required models have data
        available = [m for m in models if m in prices]
        if len(available) < len(models):
            continue

        # Equal-weight ensemble of these models
        combo_price = sum(prices[m] for m in available) / len(available)
        valid = combo_price.notna() & actual.notna() & (actual > 0)

        if valid.sum() == 0:
            continue

        combo_p = combo_price[valid]
        actual_p = actual[valid]
        current_p = current[valid]
        h = horizon[valid]

        error_pct = (combo_p - actual_p) / actual_p * 100
        abs_error_pct = error_pct.abs()
        pred_return = (combo_p - current_p) / current_p
        actual_return = (actual_p - current_p) / current_p
        direction_correct = ((pred_return > 0) == (actual_return > 0)).astype(int)

        # Per-horizon results
        temp_df = pd.DataFrame({
            'horizon': h,
            'abs_error_pct': abs_error_pct,
            'error_pct': error_pct,
            'direction_correct': direction_correct,
        })
        for hz, group in temp_df.groupby('horizon'):
            results.append({
                'ensemble_combo': combo_name,
                'horizon_days': hz,
                'n_predictions': len(group),
                'mae_pct': round(group['abs_error_pct'].mean(), 4),
                'mean_error_pct': round(group['error_pct'].mean(), 4),
                'direction_accuracy': round(group['direction_correct'].mean(), 4),
            })

    if not results:
        return None

    return pd.DataFrame(results)


def identify_biases(merged_df):
    """Identify systematic prediction biases (ensemble predictions only)."""
    biases = []

    if merged_df is None or len(merged_df) == 0:
        return biases

    # Use only ensemble predictions — CI columns and bias stats are only
    # meaningful for the ensemble row, not individual model rows.
    if 'model_type' in merged_df.columns:
        df = merged_df[merged_df['model_type'] == 'ensemble']
        if len(df) == 0:
            df = merged_df  # fallback: no ensemble rows, use all
    else:
        df = merged_df

    # 1. Overall direction bias
    mean_error = df['price_error_pct'].mean()
    if mean_error > 2:
        biases.append(f"SYSTEMATIC UPWARD BIAS: predictions are {mean_error:.1f}% too high on average")
    elif mean_error < -2:
        biases.append(f"SYSTEMATIC DOWNWARD BIAS: predictions are {abs(mean_error):.1f}% too low on average")

    # 2. Error growth with horizon
    for horizon in sorted(df['prediction_horizon_days'].unique()):
        h_data = df[df['prediction_horizon_days'] == horizon]
        mae = h_data['abs_error_pct'].mean()
        biases.append(f"  {horizon}D horizon: MAE={mae:.1f}%, direction={h_data['direction_correct'].mean():.1%}")

    # 3. Confidence interval calibration
    in_ci = df['in_confidence'].mean()
    if in_ci < 0.80:
        biases.append(f"UNDERCONFIDENT INTERVALS: only {in_ci:.1%} of actuals within 90% CI (expected ~90%)")
    elif in_ci > 0.98:
        biases.append(f"OVERWIDE INTERVALS: {in_ci:.1%} of actuals within 90% CI (intervals too wide)")

    # 4. Sector/market bias
    mean_pred_return = df['predicted_return'].mean()
    mean_actual_return = df['actual_return'].mean()
    if abs(mean_pred_return - mean_actual_return) > 0.05:
        biases.append(
            f"RETURN LEVEL MISMATCH: predicted avg return={mean_pred_return:.2%}, "
            f"actual avg return={mean_actual_return:.2%}"
        )

    return biases


def generate_report(horizon_metrics, ticker_metrics, biases, merged_df, output_file=None,
                     model_comparison=None, ensemble_combos=None):
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

    # Individual model comparison
    if model_comparison is not None and len(model_comparison) > 0:
        lines.append("\n\n" + "=" * 80)
        lines.append("INDIVIDUAL MODEL COMPARISON")
        lines.append("=" * 80)
        lines.append("\nMetrics by model type and prediction horizon:")
        lines.append(model_comparison.to_string())

        # Rank models by MAE for each horizon
        lines.append("\n\nModel Ranking by MAE% (lower is better):")
        for hz in model_comparison.index.get_level_values('prediction_horizon_days').unique():
            hz_data = model_comparison.xs(hz, level='prediction_horizon_days')
            ranked = hz_data.sort_values('mae_pct')
            lines.append(f"\n  {hz}D Horizon:")
            for rank, (model, row) in enumerate(ranked.iterrows(), 1):
                lines.append(f"    #{rank} {model:>10s}: MAE={row['mae_pct']:.2f}%  "
                           f"Dir={row['direction_accuracy']:.1%}  "
                           f"Bias={row['mean_error_pct']:+.2f}%")

    # Ensemble combination comparison
    if ensemble_combos is not None and len(ensemble_combos) > 0:
        lines.append("\n\n" + "=" * 80)
        lines.append("ENSEMBLE COMBINATION COMPARISON")
        lines.append("=" * 80)
        lines.append("\nEqual-weight ensemble combinations (hypothetical):")

        for hz in sorted(ensemble_combos['horizon_days'].unique()):
            hz_data = ensemble_combos[ensemble_combos['horizon_days'] == hz].sort_values('mae_pct')
            lines.append(f"\n  {hz}D Horizon:")
            lines.append(f"  {'Combination':<30s} {'MAE%':>8s} {'Dir.Acc':>8s} {'Bias%':>8s} {'N':>5s}")
            lines.append(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*5}")
            for _, row in hz_data.iterrows():
                lines.append(f"  {row['ensemble_combo']:<30s} {row['mae_pct']:>8.2f} "
                           f"{row['direction_accuracy']:>8.1%} {row['mean_error_pct']:>+8.2f} "
                           f"{row['n_predictions']:>5d}")

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


def plot_model_comparison(merged_df, output_dir):
    """Generate model comparison visualization plots."""
    if merged_df is None or len(merged_df) == 0:
        return

    model_types = sorted(merged_df['model_type'].unique())
    if len(model_types) <= 1:
        return

    horizons = sorted(merged_df['prediction_horizon_days'].unique())
    fig, axes = plt.subplots(1, min(len(horizons), 4), figsize=(6 * min(len(horizons), 4), 6),
                             squeeze=False)

    colors = plt.cm.Set2(np.linspace(0, 1, len(model_types)))

    for i, hz in enumerate(horizons[:4]):
        ax = axes[0, i]
        hz_data = merged_df[merged_df['prediction_horizon_days'] == hz]
        model_mae = hz_data.groupby('model_type')['abs_error_pct'].mean().sort_values()

        bars = ax.bar(range(len(model_mae)), model_mae.values,
                      color=[colors[model_types.index(m)] for m in model_mae.index])
        ax.set_xticks(range(len(model_mae)))
        ax.set_xticklabels(model_mae.index, rotation=45, ha='right')
        ax.set_ylabel('MAE (%)')
        ax.set_title(f'{hz}D Horizon: Model MAE%')

        # Add value labels
        for bar, val in zip(bars, model_mae.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    graph_path = os.path.join(output_dir, "model_comparison.png")
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

    # Model comparison analysis
    print("Analyzing individual model performance...")
    model_comparison = analyze_model_comparison(merged_df)
    ensemble_combos = analyze_ensemble_combinations(merged_df)

    if model_comparison is not None:
        n_models = model_comparison.index.get_level_values('model_type').nunique()
        print(f"Compared {n_models} model types")
    else:
        print("No per-model predictions found (only ensemble). "
              "Re-run predictions to store per-model data.")

    if ensemble_combos is not None:
        print(f"Evaluated {ensemble_combos['ensemble_combo'].nunique()} ensemble combinations")

    # Generate report
    generate_report(horizon_metrics, ticker_metrics, biases, merged_df, args.output,
                    model_comparison=model_comparison, ensemble_combos=ensemble_combos)

    # Generate plots
    plot_accuracy_summary(merged_df, graphs_dir)
    if model_comparison is not None:
        plot_model_comparison(merged_df, graphs_dir)


if __name__ == '__main__':
    main()