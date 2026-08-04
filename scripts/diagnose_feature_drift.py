"""
Feature drift diagnostic — verifies the non-stationarity hypothesis.

For each given ticker, replicates the splitter's exact chronology (same
fractions, same exclusions, same train-only MinMax scaling) and reports,
per feature, how far outside the training range [0, 1] the validation and
test windows drift. Linear/kernel models (Ridge, SVR) extrapolate
catastrophically on out-of-range features; tree models merely saturate.

If price-level features (SMA/EMA/Bollinger and friends) dominate the top
of this report with large out-of-range fractions, the fix is making them
stationary (e.g. close/sma - 1) rather than anything about the scaler.

Usage (inside the ml or app container, DB required):
    python scripts/diagnose_feature_drift.py BPE.MI BR
    python scripts/diagnose_feature_drift.py BPSO.MI --top 25
"""
import argparse
import os
import sys

# Bootstrap: allow running from scripts/ by making the repo root importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import db_interactions

EXCLUDE_COLS = ["open_Price", "high_Price", "low_Price", "close_Price",
                "trade_Volume", "1D", "prediction"]
DROP_COLS = ["date", "name", "date_published", "ticker", "currency",
             "financial_date_used"]
TEST_SIZE = 0.10
VALIDATION_SIZE = 0.20
FORECAST_FRACTION = 0.05


def analyze_ticker(ticker: str, top: int) -> None:
    print("=" * 72)
    print(f"FEATURE DRIFT ANALYSIS: {ticker}")
    print("=" * 72)

    df = db_interactions.import_stock_dataset(ticker)
    if df is None or len(df) == 0:
        print(f"  No dataset returned for {ticker} — skipping.")
        return

    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)

    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df = df.select_dtypes(include=[np.number]).dropna()

    x = df.drop(columns=[c for c in EXCLUDE_COLS if c in df.columns])

    n_total = len(x)
    forecast_out = int(np.ceil(FORECAST_FRACTION * n_total))
    x = x.iloc[:-forecast_out] if forecast_out else x

    n = len(x)
    train_end = int(n * (1 - TEST_SIZE - VALIDATION_SIZE))
    val_end = int(n * (1 - TEST_SIZE))
    x_train, x_val, x_test = x.iloc[:train_end], x.iloc[train_end:val_end], x.iloc[val_end:]
    print(f"  Rows: {n} total | train {len(x_train)} | val {len(x_val)} | test {len(x_test)}")

    scaler = MinMaxScaler().fit(x_train)
    val_scaled = pd.DataFrame(scaler.transform(x_val), columns=x.columns)
    test_scaled = pd.DataFrame(scaler.transform(x_test), columns=x.columns)

    rows = []
    for col in x.columns:
        v, t = val_scaled[col], test_scaled[col]
        rows.append({
            "feature": col,
            "val_out_%": 100.0 * ((v < 0) | (v > 1)).mean(),
            "test_out_%": 100.0 * ((t < 0) | (t > 1)).mean(),
            "test_min": t.min(),
            "test_max": t.max(),
        })

    report = pd.DataFrame(rows).sort_values("test_out_%", ascending=False)

    print(f"\n  Top {top} features by fraction of TEST rows outside [0, 1]:")
    print("  (test_min/test_max show how far the frozen train-fitted scaler")
    print("   maps test values — e.g. 2.0 means double the training max)\n")
    with pd.option_context("display.float_format", lambda v: f"{v:8.2f}"):
        print(report.head(top).to_string(index=False))

    n_drifting = (report["test_out_%"] > 10).sum()
    worst = report.iloc[0]
    print(f"\n  SUMMARY: {n_drifting}/{len(report)} features have >10% of test rows "
          f"out of range.")
    print(f"  Worst: {worst['feature']} ({worst['test_out_%']:.0f}% out, "
          f"scaled range [{worst['test_min']:.2f}, {worst['test_max']:.2f}])")
    if n_drifting > 0:
        print("  -> Non-stationarity confirmed for the features listed above;")
        print("     candidates for close-relative or return-based transforms.")
    else:
        print("  -> Features look stationary; the ridge/SVR failures need "
              "another explanation.")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tickers", nargs="+", help="Ticker symbols to analyze")
    parser.add_argument("--top", type=int, default=15,
                        help="How many features to list (default 15)")
    args = parser.parse_args()

    for ticker in args.tickers:
        try:
            analyze_ticker(ticker, args.top)
        except Exception as exc:  # diagnostic tool: report and continue
            print(f"  ERROR analyzing {ticker}: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())