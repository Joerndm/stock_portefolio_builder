"""
Fundamental feature drift classifier.

Distinguishes TWO different behaviours in the fundamental (financial-statement)
features, which the plain drift diagnostic lumps together:

  1. EXPECTED TTM stepping — a trailing-twelve-month value is piecewise
     constant (holds for a quarter, then steps). This is by design. If the
     stepped values still oscillate within the range the TRAINING period
     already covered, the feature is fine: a scaler generalises to it.

  2. REAL DRIFT — the value trends over years (e.g. book value grows as the
     company expands), so the TEST period sits in a value range the training
     period never saw. A train-fitted scaler then maps test rows outside
     [0, 1] permanently. Linear/kernel models extrapolate badly on these.

The discriminator: overlap between the training value-range and the test
value-range. High overlap => expected stepping (leave it). Near-zero overlap
with test values consistently above/below training => real drift (candidate
for a growth/ratio transform).

For each fundamental feature this reports:
  train_min/max, test_min/max  — the raw value ranges
  range_overlap_%              — how much of the test range lies within the
                                 training range (100 = fully covered, 0 =
                                 disjoint)
  n_unique_test                — distinct values in test (low = few TTM steps,
                                 confirms the piecewise-constant shape)
  verdict                      — EXPECTED_STEPPING | REAL_DRIFT | AMBIGUOUS

Usage (inside the ml/app container, DB required):
    python scripts/classify_fundamental_drift.py BPE.MI
    python scripts/classify_fundamental_drift.py BPE.MI BR BPSO.MI
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import db_interactions  # noqa: E402

TEST_SIZE = 0.10
VALIDATION_SIZE = 0.20
FORECAST_FRACTION = 0.05

DROP_COLS = ["date", "name", "date_published", "ticker", "currency",
             "financial_date_used"]
EXCLUDE_COLS = ["open_Price", "high_Price", "low_Price", "close_Price",
                "trade_Volume", "1D", "prediction"]

# Price/technical families already handled by relativize_price_level_features;
# a feature is treated as "fundamental" if it is NOT one of these.
TECH_PREFIXES = ("sma_", "ema_", "std_Div_", "bollinger_Band_", "vwap",
                 "rsi", "macd", "momentum", "volatility", "volume_", "obv",
                 "atr", "beta")
PERIOD_RETURN_COLS = {"1M", "3M", "6M", "9M", "1Y", "2Y", "3Y", "4Y", "5Y"}


def _is_fundamental(col: str) -> bool:
    low = col.lower()
    if col in PERIOD_RETURN_COLS:
        return False
    return not any(low.startswith(p) for p in TECH_PREFIXES)


def _range_overlap_pct(train_lo, train_hi, test_lo, test_hi):
    """Percentage of the test value-range that lies within the train range."""
    if test_hi == test_lo:
        # Single constant test value: fully in range or fully out
        return 100.0 if train_lo <= test_lo <= train_hi else 0.0
    inter_lo = max(train_lo, test_lo)
    inter_hi = min(train_hi, test_hi)
    inter = max(inter_hi - inter_lo, 0.0)
    return 100.0 * inter / (test_hi - test_lo)


def analyze(ticker: str) -> None:
    print("=" * 72)
    print(f"FUNDAMENTAL DRIFT CLASSIFICATION: {ticker}")
    print("=" * 72)

    df = db_interactions.import_stock_dataset(ticker)
    if df is None or len(df) == 0:
        print(f"  No dataset for {ticker} — skipping.\n")
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
    x_train, x_test = x.iloc[:train_end], x.iloc[val_end:]
    print(f"  Rows: {n} | train {len(x_train)} | test {len(x_test)}")

    fundamentals = [c for c in x.columns if _is_fundamental(c)]
    print(f"  Fundamental features: {len(fundamentals)} of {len(x.columns)} total\n")

    rows = []
    for col in fundamentals:
        tr, te = x_train[col], x_test[col]
        tr_lo, tr_hi = tr.min(), tr.max()
        te_lo, te_hi = te.min(), te.max()
        overlap = _range_overlap_pct(tr_lo, tr_hi, te_lo, te_hi)
        n_unique_test = te.nunique()

        # Direction: is test above, below, or straddling the train range?
        if te_lo > tr_hi:
            direction = "above"
        elif te_hi < tr_lo:
            direction = "below"
        else:
            direction = "within/straddle"

        if overlap >= 50:
            verdict = "EXPECTED_STEPPING"
        elif overlap < 10 and direction in ("above", "below"):
            verdict = "REAL_DRIFT"
        else:
            verdict = "AMBIGUOUS"

        rows.append({
            "feature": col,
            "train_min": tr_lo, "train_max": tr_hi,
            "test_min": te_lo, "test_max": te_hi,
            "overlap_%": round(overlap, 1),
            "n_uniq_test": n_unique_test,
            "dir": direction,
            "verdict": verdict,
        })

    report = pd.DataFrame(rows).sort_values(["verdict", "overlap_%"])

    with pd.option_context("display.max_rows", None,
                           "display.width", 200,
                           "display.float_format", lambda v: f"{v:10.3f}"):
        print(report.to_string(index=False))

    counts = report["verdict"].value_counts().to_dict()
    print(f"\n  VERDICT COUNTS: {counts}")
    real = report[report["verdict"] == "REAL_DRIFT"]["feature"].tolist()
    if real:
        print(f"  REAL DRIFT features ({len(real)}): {', '.join(real)}")
        print("  -> These trend across years; test range is disjoint from train.")
        print("     Candidates for growth-rate or market-relative transforms.")
    else:
        print("  -> No clear real drift; remaining out-of-range behaviour is")
        print("     expected TTM stepping within the training range.")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tickers", nargs="+")
    args = parser.parse_args()
    for t in args.tickers:
        try:
            analyze(t)
        except Exception as exc:
            print(f"  ERROR analyzing {t}: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
