"""
Re-fetch and OVERWRITE a single ticker's price data through the corrected,
stationary-feature pipeline.

Unlike manual_fetch_demant.py / fetch_all_remaining_tickers.py (which SKIP
tickers that already have data), this script deliberately re-fetches and
overwrites, so it can be used to migrate an existing ticker onto the new
close-relative feature representation.

Overwrite is safe because db_interactions.export_stock_price_data already
performs delete-then-insert for the ticker's date range (it deletes existing
rows in [min_date, max_date] before inserting), so no manual delete is needed.

Use this to validate step 1 on ONE ticker before committing to a fleet-wide
re-fetch:

    python scripts/refetch_one_ticker.py BPE.MI
    python scripts/diagnose_feature_drift.py BPE.MI   # confirm features now in-range

IMPORTANT: take a database snapshot before running (see the mysqldump command
in the migration notes). This overwrites the ticker's stored price features.
"""
import argparse
import os
import sys

# Repo root importable regardless of launch directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd  # noqa: E402
import yfinance as yf  # noqa: E402

import db_interactions  # noqa: E402
import stock_data_fetch as sdf  # noqa: E402
from technical_indicators import relativize_price_level_features  # noqa: E402


def refetch_ticker(ticker: str, is_index: bool = False) -> bool:
    """Fetch `ticker` fresh, compute stationary features, overwrite in DB.
    Returns True on successful export, False otherwise."""
    print("=" * 60)
    print(f"Re-fetching (overwrite) {ticker}")
    print("=" * 60)

    df = sdf.fetch_stock_price_data(ticker)
    if df is None or df.empty:
        print(f"  No data fetched for {ticker} — aborting.")
        return False
    print(f"  Fetched {len(df)} rows")

    # yfinance frames can contain rows with null close_Price (halts, gaps,
    # delisted spans). The indicator functions validate against nulls up
    # front (calculate_period_returns raises on any null close_Price), so
    # clean those rows before the compute chain — this mirrors what the
    # production pipeline achieves via its critical-column dropna.
    before = len(df)
    df = df[df["close_Price"].notna()].reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped} rows with null close_Price")
    if df.empty:
        print(f"  No rows with valid close_Price for {ticker} — aborting.")
        return False

    # Same indicator chain as the production orchestrator, in the same order.
    df = sdf.calculate_period_returns(df)
    df = sdf.add_technical_indicators(df)
    df = sdf.add_volume_indicators(df)
    if not is_index:
        df = sdf.add_volatility_indicators(df)
    df = sdf.calculate_moving_averages(df)
    df = sdf.calculate_standard_diviation_value(df)
    df = sdf.calculate_bollinger_bands(df)
    df = sdf.calculate_momentum(df)

    # The corrected step: make price-level features stationary. MUST come
    # after every indicator shift above (single source of truth in
    # technical_indicators.py).
    df = relativize_price_level_features(df)

    critical_cols = ['date', 'ticker', 'close_Price', 'open_Price', 'high_Price', 'low_Price']
    df = df.dropna(subset=critical_cols)
    print(f"  After dropna(): {len(df)} rows")

    if df.empty:
        print("  DataFrame empty after dropna — nothing to export.")
        return False

    db_interactions.export_stock_price_data(df)  # delete-then-insert overwrite
    print(f"  ✓ Overwrote {len(df)} rows for {ticker} with stationary features")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tickers", nargs="+", help="Ticker(s) to re-fetch and overwrite")
    parser.add_argument("--index", action="store_true",
                        help="Treat as index ticker (skip volatility indicators)")
    args = parser.parse_args()

    ok = 0
    for ticker in args.tickers:
        try:
            if refetch_ticker(ticker, is_index=args.index):
                ok += 1
        except Exception as exc:
            print(f"  ERROR re-fetching {ticker}: {exc}")
    print(f"\nDone: {ok}/{len(args.tickers)} tickers overwritten successfully.")
    return 0 if ok == len(args.tickers) else 1


if __name__ == "__main__":
    sys.exit(main())