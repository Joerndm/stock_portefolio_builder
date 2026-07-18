"""
Portfolio Construction Integration Test

This script tests the end-to-end portfolio construction pipeline using stocks
and models that already exist in the database (post-training/tuning).

It bypasses the expensive ML training step and instead:
1. Loads stored predictions and Monte Carlo results from the database
2. Loads historical price data for stocks that have predictions
3. Reconstructs StockPredictionResult objects from DB data
4. Generates synthetic daily forecast series from stored prediction horizons
5. Runs the portfolio construction pipeline (efficient frontier optimization)
6. Validates portfolio outputs against investor profile constraints
7. Optionally stores the results in the database as a portfolio run

This allows rapid validation that the post-training portfolio construction
pipeline works correctly without needing to retrain any models.

If no stored predictions exist yet, the script falls back to using recent
historical price data to build synthetic forecasts for portfolio construction
testing. This still exercises the full portfolio construction pipeline
(price collection, efficient frontier, validation, DB export) end-to-end.

Usage:
    python test_portfolio_construction.py
    python test_portfolio_construction.py --profile conservative
    python test_portfolio_construction.py --profile aggressive --max-stocks 15
    python test_portfolio_construction.py --save-to-db
"""
import os
import sys
import time
import datetime
import argparse
import traceback
from typing import Dict, List, Optional, Tuple

# Suppress TF/GPU warnings for faster import
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_PTXAS_UNAVAILABLE'] = '1'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for tests
import matplotlib.pyplot as plt

import db_interactions
import monte_carlo_sim
import efficient_frontier
from portfolio_config import (
    InvestorProfile,
    RiskLevel,
    StockPredictionResult,
    get_default_profile,
    DEFAULT_RISK_FREE_RATE,
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def collect_portfolio_prices(
    results: Dict[str, StockPredictionResult],
    prediction_days: int = 90
) -> pd.DataFrame:
    """
    Collect forecasted prices from all successful predictions for portfolio
    optimization. Mirrors stock_analyzer.collect_portfolio_prices without
    requiring a TensorFlow import.
    
    Args:
        results: Dictionary of StockPredictionResult objects keyed by ticker.
        prediction_days: Number of days of forecasts to include.
        
    Returns:
        DataFrame with tickers as columns and day indices as rows.
    """
    price_data = {}

    for ticker, result in results.items():
        if result.success and result.forecast_df is not None:
            try:
                forecast = result.forecast_df.copy()
                price_col = 'close_Price' if 'close_Price' in forecast.columns else None
                if price_col is None:
                    continue
                prices = forecast[price_col].iloc[:prediction_days].values
                if len(prices) > 0:
                    price_data[ticker] = prices
            except Exception as e:
                print(f"  [WARNING] Could not extract prices for {ticker}: {e}")

    if not price_data:
        return pd.DataFrame()

    max_len = max(len(p) for p in price_data.values())
    for ticker in price_data:
        if len(price_data[ticker]) < max_len:
            last_val = price_data[ticker][-1]
            price_data[ticker] = list(price_data[ticker]) + [last_val] * (max_len - len(price_data[ticker]))

    return pd.DataFrame(price_data)


def load_tickers_with_predictions() -> List[str]:
    """
    Query the database for tickers that have stored prediction data.
    
    Returns:
        List of ticker symbols that have predictions in stock_prediction_extended.
    """
    predictions_df = db_interactions.import_stock_predictions_extended()
    if predictions_df is None or predictions_df.empty:
        return []

    return sorted(predictions_df['ticker'].unique().tolist())


def load_tickers_with_monte_carlo() -> List[str]:
    """
    Query the database for tickers that have stored Monte Carlo results.
    
    Returns:
        List of ticker symbols that have Monte Carlo results.
    """
    mc_df = db_interactions.import_monte_carlo_results()
    if mc_df is None or mc_df.empty:
        return []

    return sorted(mc_df['ticker'].unique().tolist())


def interpolate_daily_forecast(
    current_price: float,
    prediction_rows: pd.DataFrame,
    total_days: int = 90
) -> pd.DataFrame:
    """
    Build a synthetic daily forecast from the stored prediction horizons.
    
    The database stores predictions at horizons 30, 60, 90, 252 days.
    This function linearly interpolates between those anchor points to
    produce a day-by-day price series suitable for portfolio optimization.
    
    Args:
        current_price: The current (starting) stock price.
        prediction_rows: DataFrame from import_stock_predictions_extended for one ticker.
        total_days: Number of forecast days to generate.
        
    Returns:
        DataFrame with columns ['close_Price'] indexed by day number.
    """
    # Build anchor points: day 0 = current price, then each horizon
    anchors = {0: current_price}

    for _, row in prediction_rows.iterrows():
        horizon = int(row['prediction_horizon_days'])
        price = float(row['predicted_price'])
        if horizon <= total_days and not np.isnan(price):
            anchors[horizon] = price

    if len(anchors) < 2:
        # Fallback: flat forecast at current price
        return pd.DataFrame(
            {'close_Price': [current_price] * total_days},
            index=range(total_days)
        )

    # Sort anchor days
    sorted_days = sorted(anchors.keys())
    anchor_days = np.array(sorted_days)
    anchor_prices = np.array([anchors[d] for d in sorted_days])

    # Interpolate to daily
    all_days = np.arange(total_days)
    daily_prices = np.interp(all_days, anchor_days, anchor_prices)

    return pd.DataFrame({'close_Price': daily_prices}, index=all_days)


def reconstruct_monte_carlo_year_df(mc_rows: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct a monte_carlo_year_df from stored DB results.
    
    The database stores columns like percentile_5, mean_price, percentile_95, etc.
    We reconstruct the format that monte_carlo_analysis() returns:
    Index = year, columns = ['5th Percentile', '16th Percentile', 'Mean',
                              '84th Percentile', '95th Percentile']
    
    Args:
        mc_rows: DataFrame from import_monte_carlo_results for one ticker.
        
    Returns:
        DataFrame matching the monte_carlo_analysis output format.
    """
    result_rows = []
    years = []

    for _, row in mc_rows.iterrows():
        year = int(row['simulation_year'])
        years.append(year)
        result_rows.append({
            '5th Percentile': row.get('percentile_5', np.nan),
            '16th Percentile': row.get('percentile_16', np.nan),
            'Mean': row.get('mean_price', np.nan),
            '84th Percentile': row.get('percentile_84', np.nan),
            '95th Percentile': row.get('percentile_95', np.nan),
        })

    if not result_rows:
        return None

    mc_year_df = pd.DataFrame(result_rows, index=years)
    mc_year_df.index.name = None
    return mc_year_df


def reconstruct_prediction_results(
    tickers: List[str],
    predictions_df: pd.DataFrame,
    mc_df: pd.DataFrame,
    forecast_days: int = 90
) -> Dict[str, StockPredictionResult]:
    """
    Reconstruct StockPredictionResult objects from stored DB data.
    
    Args:
        tickers: List of ticker symbols to reconstruct.
        predictions_df: Full predictions DataFrame from DB.
        mc_df: Full Monte Carlo DataFrame from DB.
        forecast_days: Number of forecast days for interpolation.
        
    Returns:
        Dictionary of ticker -> StockPredictionResult.
    """
    results = {}

    for ticker in tickers:
        try:
            # Get predictions for this ticker (most recent prediction date)
            ticker_preds = predictions_df[predictions_df['ticker'] == ticker].copy()
            if ticker_preds.empty:
                print(f"  [SKIP] {ticker}: No prediction data")
                continue

            # Use most recent prediction date
            latest_date = ticker_preds['prediction_date'].max()
            ticker_preds = ticker_preds[ticker_preds['prediction_date'] == latest_date]

            # Get current price from stored predictions
            current_price = ticker_preds['current_price'].iloc[0]
            if pd.isna(current_price) or current_price <= 0:
                print(f"  [SKIP] {ticker}: Invalid current price ({current_price})")
                continue

            # Build daily forecast
            forecast_df = interpolate_daily_forecast(
                current_price=current_price,
                prediction_rows=ticker_preds,
                total_days=forecast_days
            )

            # Get Monte Carlo yearly data
            ticker_mc = mc_df[mc_df['ticker'] == ticker].copy() if not mc_df.empty else pd.DataFrame()
            monte_carlo_year_df = None
            if not ticker_mc.empty:
                latest_mc_date = ticker_mc['simulation_date'].max()
                ticker_mc = ticker_mc[ticker_mc['simulation_date'] == latest_mc_date]
                monte_carlo_year_df = reconstruct_monte_carlo_year_df(ticker_mc)

            results[ticker] = StockPredictionResult(
                symbol=ticker,
                success=True,
                forecast_df=forecast_df,
                monte_carlo_day_df=None,  # Not stored in DB
                monte_carlo_year_df=monte_carlo_year_df,
                execution_time=0.0
            )

            print(f"  [OK] {ticker}: price={current_price:.2f}, "
                  f"forecast_days={len(forecast_df)}, "
                  f"MC_years={len(monte_carlo_year_df) if monte_carlo_year_df is not None else 0}")

        except Exception as e:
            print(f"  [ERROR] {ticker}: {e}")
            results[ticker] = StockPredictionResult(
                symbol=ticker,
                success=False,
                error_message=str(e),
                execution_time=0.0
            )

    return results


def build_forecasts_from_historical_data(
    tickers: List[str],
    forecast_days: int = 90
) -> Dict[str, StockPredictionResult]:
    """
    Fallback: Build synthetic forecasts from historical price data.
    
    When no stored predictions exist in the database, this function loads
    the most recent historical price data for each stock and uses it as a
    proxy forecast. This allows testing the portfolio construction pipeline
    (efficient frontier, validation, DB export) end-to-end.
    
    The "forecast" is constructed from the last `forecast_days` of actual
    historical close prices, which provides realistic price dynamics for
    testing the portfolio optimization math.
    
    Args:
        tickers: List of ticker symbols to process.
        forecast_days: Number of days to include in the synthetic forecast.
        
    Returns:
        Dictionary of ticker -> StockPredictionResult.
    """
    results = {}

    for ticker in tickers:
        try:
            stock_data_df = db_interactions.import_stock_dataset(ticker)
            stock_data_df["date"] = pd.to_datetime(stock_data_df["date"])
            stock_data_df = stock_data_df.dropna(axis=0, how="any")
            stock_data_df = stock_data_df.dropna(axis=1, how="any")

            if len(stock_data_df) < forecast_days + 30:
                print(f"  [SKIP] {ticker}: Insufficient data ({len(stock_data_df)} rows)")
                continue

            # Use the last N days of actual prices as a proxy forecast
            recent_data = stock_data_df.tail(forecast_days).reset_index(drop=True)
            forecast_df = pd.DataFrame({
                'close_Price': recent_data['close_Price'].values
            })

            current_price = float(stock_data_df['close_Price'].iloc[-1])

            results[ticker] = StockPredictionResult(
                symbol=ticker,
                success=True,
                forecast_df=forecast_df,
                monte_carlo_day_df=None,
                monte_carlo_year_df=None,
                execution_time=0.0
            )

            print(f"  [OK] {ticker}: price={current_price:.2f}, "
                  f"days={len(forecast_df)}")

        except Exception as e:
            print(f"  [ERROR] {ticker}: {e}")

    return results


def run_monte_carlo_on_reconstructed(
    ticker: str,
    result: StockPredictionResult,
    investor_profile: InvestorProfile,
    sim_amount: int = 500
) -> StockPredictionResult:
    """
    Run a fresh Monte Carlo simulation on a reconstructed forecast.
    
    This is useful when we want to validate the MC pipeline works 
    end-to-end, using the stored forecast as input.
    
    Args:
        ticker: Stock ticker symbol.
        result: StockPredictionResult with forecast_df populated.
        investor_profile: Investor profile for years parameter.
        sim_amount: Number of MC simulations (lower than production for speed).
        
    Returns:
        Updated StockPredictionResult with fresh MC results.
    """
    if not result.success or result.forecast_df is None:
        return result

    try:
        # Load historical data for the stock (needed for starting price)
        stock_data_df = db_interactions.import_stock_dataset(ticker)
        stock_data_df["date"] = pd.to_datetime(stock_data_df["date"])
        stock_data_df = stock_data_df.dropna(axis=0, how="any")
        stock_data_df = stock_data_df.dropna(axis=1, how="any")

        mc_day_df, mc_year_df = monte_carlo_sim.monte_carlo_analysis(
            seed_number=42,
            stock_data_df=stock_data_df,
            forecast_df=result.forecast_df,
            years=investor_profile.investment_years,
            sim_amount=sim_amount
        )

        return StockPredictionResult(
            symbol=ticker,
            success=True,
            forecast_df=result.forecast_df,
            monte_carlo_day_df=mc_day_df,
            monte_carlo_year_df=mc_year_df,
            execution_time=result.execution_time
        )

    except Exception as e:
        print(f"  [MC-ERROR] {ticker}: {e}")
        return result


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_portfolio_output(
    portfolio_df: pd.DataFrame,
    investor_profile: InvestorProfile,
    num_stocks: int
) -> Dict[str, any]:
    """
    Validate the efficient frontier output against investor profile constraints.
    
    Checks:
    - Portfolio weights sum to ~1.0
    - Return and Volatility columns exist and are reasonable
    - At least some portfolios exist within the volatility cap
    - Sharpe ratios are calculated correctly
    
    Args:
        portfolio_df: Output from efficient_frontier_sim.
        investor_profile: Investor configuration.
        num_stocks: Number of stocks in the portfolio.
        
    Returns:
        Dictionary with validation results.
    """
    results = {
        'passed': True,
        'checks': {},
        'best_portfolio': None,
        'summary': {}
    }

    # Check 1: DataFrame is not empty
    is_not_empty = len(portfolio_df) > 0
    results['checks']['not_empty'] = {
        'passed': is_not_empty,
        'detail': f'{len(portfolio_df)} portfolios generated'
    }
    if not is_not_empty:
        results['passed'] = False
        return results

    # Check 2: Required columns exist
    has_return = 'Return' in portfolio_df.columns
    has_vol = 'Volatility' in portfolio_df.columns
    results['checks']['required_columns'] = {
        'passed': has_return and has_vol,
        'detail': f"Return col: {has_return}, Volatility col: {has_vol}"
    }
    if not (has_return and has_vol):
        results['passed'] = False
        return results

    # Check 3: Portfolio weights sum to approximately 1.0
    stock_cols = [c for c in portfolio_df.columns
                  if c not in ('Portefolio number', 'Return', 'Volatility', 'Sharpe')]
    if stock_cols:
        weight_sums = portfolio_df[stock_cols].sum(axis=1)
        weights_valid = ((weight_sums - 1.0).abs() < 0.01).all()
        results['checks']['weights_sum_to_1'] = {
            'passed': bool(weights_valid),
            'detail': f"Weight sum range: [{weight_sums.min():.4f}, {weight_sums.max():.4f}]"
        }
        if not weights_valid:
            results['passed'] = False
    else:
        results['checks']['weights_sum_to_1'] = {
            'passed': False,
            'detail': "No stock weight columns found"
        }
        results['passed'] = False

    # Check 4: Returns and volatilities are in reasonable ranges
    min_ret = portfolio_df['Return'].min()
    max_ret = portfolio_df['Return'].max()
    min_vol = portfolio_df['Volatility'].min()
    max_vol = portfolio_df['Volatility'].max()

    returns_reasonable = max_ret > min_ret  # There should be variety
    vol_reasonable = max_vol > min_vol and min_vol >= 0

    results['checks']['return_range'] = {
        'passed': bool(returns_reasonable),
        'detail': f"Return range: [{min_ret:.4f}, {max_ret:.4f}]"
    }
    results['checks']['volatility_range'] = {
        'passed': bool(vol_reasonable),
        'detail': f"Volatility range: [{min_vol:.4f}, {max_vol:.4f}]"
    }
    if not returns_reasonable or not vol_reasonable:
        results['passed'] = False

    # Check 5: Sharpe ratio calculation
    risk_free_rate = DEFAULT_RISK_FREE_RATE
    portfolio_df_copy = portfolio_df.copy()
    portfolio_df_copy['Sharpe'] = (
        (portfolio_df_copy['Return'] - risk_free_rate) / portfolio_df_copy['Volatility']
    )

    best_sharpe_idx = portfolio_df_copy['Sharpe'].idxmax()
    best_portfolio = portfolio_df_copy.loc[best_sharpe_idx]
    best_sharpe = best_portfolio['Sharpe']

    results['checks']['sharpe_ratio'] = {
        'passed': not np.isnan(best_sharpe) and not np.isinf(best_sharpe),
        'detail': f"Best Sharpe ratio: {best_sharpe:.4f}"
    }

    # Check 6: Portfolios within volatility cap
    vol_cap = investor_profile.get_volatility_cap()
    within_cap = portfolio_df[portfolio_df['Volatility'] <= vol_cap]
    has_within_cap = len(within_cap) > 0

    results['checks']['within_volatility_cap'] = {
        'passed': bool(has_within_cap),
        'detail': (f"{len(within_cap)}/{len(portfolio_df)} portfolios "
                   f"within {vol_cap:.0%} volatility cap")
    }

    # Build best portfolio info
    if has_within_cap:
        within_cap_copy = within_cap.copy()
        within_cap_copy['Sharpe'] = (
            (within_cap_copy['Return'] - risk_free_rate) / within_cap_copy['Volatility']
        )
        best_within_idx = within_cap_copy['Sharpe'].idxmax()
        best_within = within_cap_copy.loc[best_within_idx]

        results['best_portfolio'] = {
            'return': float(best_within['Return']),
            'volatility': float(best_within['Volatility']),
            'sharpe': float(best_within['Sharpe']),
            'weights': {col: float(best_within[col])
                        for col in stock_cols if float(best_within[col]) > 0.001}
        }
    else:
        # Use overall best Sharpe
        results['best_portfolio'] = {
            'return': float(best_portfolio['Return']),
            'volatility': float(best_portfolio['Volatility']),
            'sharpe': float(best_sharpe),
            'weights': {col: float(best_portfolio[col])
                        for col in stock_cols if float(best_portfolio[col]) > 0.001}
        }

    # Summary statistics
    results['summary'] = {
        'total_portfolios': len(portfolio_df),
        'portfolios_within_vol_cap': len(within_cap) if has_within_cap else 0,
        'return_range': (float(min_ret), float(max_ret)),
        'volatility_range': (float(min_vol), float(max_vol)),
        'best_sharpe': float(best_sharpe),
        'num_stocks': num_stocks,
        'risk_level': investor_profile.risk_level.value,
        'vol_cap': vol_cap,
    }

    return results


def validate_monte_carlo_results(
    results: Dict[str, StockPredictionResult],
    investor_profile: InvestorProfile
) -> Dict[str, any]:
    """
    Validate Monte Carlo results across all stocks.
    
    Args:
        results: Dictionary of StockPredictionResult objects.
        investor_profile: Investor configuration.
        
    Returns:
        Validation summary dictionary.
    """
    mc_summary = {
        'stocks_with_mc': 0,
        'stocks_without_mc': 0,
        'year_coverage': {},
        'price_ranges': {},
    }

    for ticker, result in results.items():
        if not result.success:
            continue

        if result.monte_carlo_year_df is not None and not result.monte_carlo_year_df.empty:
            mc_summary['stocks_with_mc'] += 1
            years = list(result.monte_carlo_year_df.index)
            mc_summary['year_coverage'][ticker] = years

            # Price range from mean trajectory
            mean_prices = result.monte_carlo_year_df['Mean'].values
            mc_summary['price_ranges'][ticker] = {
                'start': float(mean_prices[0]) if len(mean_prices) > 0 else None,
                'end': float(mean_prices[-1]) if len(mean_prices) > 0 else None,
            }
        else:
            mc_summary['stocks_without_mc'] += 1

    return mc_summary


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_portfolio_construction_test(
    profile_name: str = "balanced",
    max_stocks: Optional[int] = None,
    save_to_db: bool = False,
    run_fresh_mc: bool = False,
    mc_sim_amount: int = 500,
    ef_sim_override: Optional[int] = None,
    verbose: bool = True
) -> Tuple[bool, Dict]:
    """
    Run the full portfolio construction integration test.
    
    Steps:
    1. Query DB for tickers with stored predictions + Monte Carlo results
    2. Reconstruct StockPredictionResult from DB data
    3. Optionally run fresh Monte Carlo simulations 
    4. Collect forecasted prices into portfolio price matrix
    5. Run efficient frontier optimization
    6. Validate outputs
    7. Optionally save portfolio run to DB
    
    Args:
        profile_name: Investor profile name ('conservative', 'balanced', 'aggressive').
        max_stocks: Maximum number of stocks to include (None = all available).
        save_to_db: Whether to store the test portfolio run in the database.
        run_fresh_mc: Whether to run fresh Monte Carlo simulations on reconstructed forecasts.
        mc_sim_amount: Number of MC simulations if run_fresh_mc is True.
        ef_sim_override: Override the number of frontier simulations (default uses 750k).
        verbose: Print detailed output.
        
    Returns:
        Tuple of (success: bool, report: dict).
    """
    overall_start = time.time()
    report = {
        'profile': profile_name,
        'steps': {},
        'validation': {},
        'overall_passed': False,
        'execution_time': 0.0,
        'errors': []
    }

    print("\n" + "=" * 80)
    print("PORTFOLIO CONSTRUCTION INTEGRATION TEST")
    print("=" * 80)

    # ---- Step 1: Configure investor profile ----
    print(f"\n[Step 1] Configuring investor profile: {profile_name}")
    try:
        investor_profile = get_default_profile(profile_name)
        print(f"  Risk Level:       {investor_profile.risk_level.value.upper()}")
        print(f"  Investment Years:  {investor_profile.investment_years}")
        print(f"  Portfolio Size:    {investor_profile.portfolio_size}")
        print(f"  Volatility Cap:    {investor_profile.get_volatility_cap():.0%}")
        print(f"  Min Sharpe Ratio:  {investor_profile.get_min_sharpe_ratio()}")
        report['steps']['profile_config'] = 'OK'
    except Exception as e:
        report['errors'].append(f"Profile config failed: {e}")
        report['steps']['profile_config'] = f'FAILED: {e}'
        print(f"  [FAIL] {e}")
        return False, report

    # ---- Step 2: Load stored predictions from DB ----
    print(f"\n[Step 2] Loading stored predictions from database...")
    use_historical_fallback = False
    try:
        predictions_df = db_interactions.import_stock_predictions_extended()
        mc_df = db_interactions.import_monte_carlo_results()

        pred_tickers = sorted(predictions_df['ticker'].unique().tolist()) if predictions_df is not None and not predictions_df.empty else []
        mc_tickers = sorted(mc_df['ticker'].unique().tolist()) if mc_df is not None and not mc_df.empty else []

        # Use tickers that have both predictions and MC data, or at minimum predictions
        all_available = sorted(set(pred_tickers))
        both_available = sorted(set(pred_tickers) & set(mc_tickers))

        print(f"  Tickers with predictions:  {len(pred_tickers)}")
        print(f"  Tickers with Monte Carlo:  {len(mc_tickers)}")
        print(f"  Tickers with both:         {len(both_available)}")
        print(f"  Total available tickers:   {len(all_available)}")

        if len(all_available) < 2:
            print(f"  [INFO] Not enough stored predictions. Falling back to historical data.")
            use_historical_fallback = True
        else:
            # Apply max_stocks limit
            selected_tickers = all_available
            if max_stocks and max_stocks < len(selected_tickers):
                # Prefer tickers that have both predictions and MC
                selected_tickers = both_available[:max_stocks]
                if len(selected_tickers) < max_stocks:
                    remaining = [t for t in all_available if t not in selected_tickers]
                    selected_tickers += remaining[:max_stocks - len(selected_tickers)]

            # Also limit to portfolio_size
            if len(selected_tickers) > investor_profile.portfolio_size:
                selected_tickers = selected_tickers[:investor_profile.portfolio_size]

            print(f"  Selected tickers:          {len(selected_tickers)}")
            print(f"  Tickers: {selected_tickers}")

        report['steps']['load_predictions'] = (
            f'OK ({len(all_available)} tickers found)'
            if not use_historical_fallback
            else 'FALLBACK (using historical data)'
        )

    except Exception as e:
        report['errors'].append(f"Loading predictions failed: {e}")
        report['steps']['load_predictions'] = f'FAILED: {e}'
        print(f"  [WARN] DB prediction query failed: {e}")
        print(f"  [INFO] Falling back to historical data.")
        use_historical_fallback = True
        predictions_df = pd.DataFrame()
        mc_df = pd.DataFrame()
        traceback.print_exc()

    # ---- Step 2b: Fallback — select tickers from stock_info_data ----
    if use_historical_fallback:
        print(f"\n[Step 2b] Selecting tickers from database for historical fallback...")
        try:
            all_tickers = db_interactions.import_ticker_list()
            print(f"  Total tickers in database: {len(all_tickers)}")

            # Determine how many to select
            target_count = max_stocks if max_stocks else min(investor_profile.portfolio_size, 15)
            # Pick evenly spaced tickers for diversity
            if len(all_tickers) > target_count:
                step = len(all_tickers) // target_count
                selected_tickers = [all_tickers[i * step] for i in range(target_count)]
            else:
                selected_tickers = all_tickers

            print(f"  Selected {len(selected_tickers)} tickers for testing")
            print(f"  Tickers: {selected_tickers}")
            report['steps']['select_fallback_tickers'] = f'OK ({len(selected_tickers)} tickers)'

        except Exception as e:
            msg = f"Could not load ticker list: {e}"
            report['errors'].append(msg)
            report['steps']['select_fallback_tickers'] = f'FAILED: {msg}'
            print(f"  [FAIL] {msg}")
            return False, report

    # ---- Step 3: Reconstruct StockPredictionResult objects ----
    print(f"\n[Step 3] {'Building forecasts from historical data' if use_historical_fallback else 'Reconstructing prediction results from stored data'}...")
    try:
        if use_historical_fallback:
            results = build_forecasts_from_historical_data(
                tickers=selected_tickers,
                forecast_days=90
            )
        else:
            results = reconstruct_prediction_results(
                tickers=selected_tickers,
                predictions_df=predictions_df,
                mc_df=mc_df if mc_df is not None else pd.DataFrame(),
                forecast_days=90
            )

        successful = sum(1 for r in results.values() if r.success)
        failed = sum(1 for r in results.values() if not r.success)
        print(f"  Successfully reconstructed: {successful}")
        print(f"  Failed: {failed}")

        if successful < 2:
            msg = f"Need at least 2 successful reconstructions, got {successful}"
            report['errors'].append(msg)
            report['steps']['reconstruct'] = f'FAILED: {msg}'
            print(f"  [FAIL] {msg}")
            return False, report

        report['steps']['reconstruct'] = f'OK ({successful} successful, {failed} failed)'

    except Exception as e:
        report['errors'].append(f"Reconstruction failed: {e}")
        report['steps']['reconstruct'] = f'FAILED: {e}'
        print(f"  [FAIL] {e}")
        traceback.print_exc()
        return False, report

    # ---- Step 3b (optional): Run fresh Monte Carlo simulations ----
    if run_fresh_mc:
        print(f"\n[Step 3b] Running fresh Monte Carlo simulations ({mc_sim_amount} sims)...")
        mc_count = 0
        for ticker, result in results.items():
            if result.success:
                print(f"  Running MC for {ticker}...")
                try:
                    results[ticker] = run_monte_carlo_on_reconstructed(
                        ticker=ticker,
                        result=result,
                        investor_profile=investor_profile,
                        sim_amount=mc_sim_amount
                    )
                    mc_count += 1
                except Exception as e:
                    print(f"    [MC-SKIP] {ticker}: {e}")
        print(f"  Fresh MC completed for {mc_count} stocks")
        report['steps']['fresh_mc'] = f'OK ({mc_count} stocks)'
    else:
        report['steps']['fresh_mc'] = 'SKIPPED'

    # ---- Step 4: Collect portfolio prices ----
    print(f"\n[Step 4] Collecting forecasted prices for portfolio optimization...")
    try:
        pf_prices = collect_portfolio_prices(results, prediction_days=90)

        if pf_prices.empty or len(pf_prices.columns) < 2:
            msg = (f"Insufficient price data for portfolio optimization: "
                   f"{len(pf_prices.columns)} stocks")
            report['errors'].append(msg)
            report['steps']['collect_prices'] = f'FAILED: {msg}'
            print(f"  [FAIL] {msg}")
            return False, report

        print(f"  Price matrix shape: {pf_prices.shape}")
        print(f"  Stocks included: {list(pf_prices.columns)}")
        print(f"  Date range: day 0 to day {len(pf_prices)-1}")

        # Quick sanity check on prices
        for col in pf_prices.columns:
            prices = pf_prices[col]
            print(f"    {col}: start={prices.iloc[0]:.2f}, "
                  f"end={prices.iloc[-1]:.2f}, "
                  f"min={prices.min():.2f}, max={prices.max():.2f}")

        report['steps']['collect_prices'] = f'OK ({pf_prices.shape})'

    except Exception as e:
        report['errors'].append(f"Price collection failed: {e}")
        report['steps']['collect_prices'] = f'FAILED: {e}'
        print(f"  [FAIL] {e}")
        traceback.print_exc()
        return False, report

    # ---- Step 5: Run Efficient Frontier Optimization ----
    print(f"\n[Step 5] Running efficient frontier optimization...")
    print(f"  Stocks in optimization: {len(pf_prices.columns)}")

    # Optionally reduce simulation count for faster testing
    portfolio_df = None
    try:
        if ef_sim_override:
            # Monkey-patch the simulation count for testing speed
            print(f"  Using reduced simulation count: {ef_sim_override}")
            portfolio_df = _run_efficient_frontier_reduced(pf_prices, ef_sim_override)
        else:
            print(f"  Using full 750,000 simulations (this may take a while)...")
            portfolio_df = efficient_frontier.efficient_frontier_sim(pf_prices)

        if portfolio_df is None or portfolio_df.empty:
            msg = "Efficient frontier returned empty result"
            report['errors'].append(msg)
            report['steps']['efficient_frontier'] = f'FAILED: {msg}'
            print(f"  [FAIL] {msg}")
            return False, report

        print(f"  Efficient frontier complete: {len(portfolio_df)} portfolios")
        print(f"  Columns: {list(portfolio_df.columns)}")
        if verbose:
            print(f"\n  Top 5 portfolios by return:")
            top5 = portfolio_df.nlargest(5, 'Return') if 'Return' in portfolio_df.columns else portfolio_df.head(5)
            print(top5.to_string(index=False))

        report['steps']['efficient_frontier'] = f'OK ({len(portfolio_df)} portfolios)'

    except Exception as e:
        report['errors'].append(f"Efficient frontier failed: {e}")
        report['steps']['efficient_frontier'] = f'FAILED: {e}'
        print(f"  [FAIL] {e}")
        traceback.print_exc()
        return False, report

    # ---- Step 6: Validate portfolio outputs ----
    print(f"\n[Step 6] Validating portfolio outputs...")
    try:
        validation = validate_portfolio_output(
            portfolio_df=portfolio_df,
            investor_profile=investor_profile,
            num_stocks=len(pf_prices.columns)
        )

        report['validation'] = validation

        for check_name, check_result in validation['checks'].items():
            status = "PASS" if check_result['passed'] else "FAIL"
            print(f"  [{status}] {check_name}: {check_result['detail']}")

        if validation['best_portfolio']:
            bp = validation['best_portfolio']
            print(f"\n  Best Portfolio (by Sharpe ratio):")
            print(f"    Expected Return:    {bp['return']:.4f} ({bp['return']:.2%})")
            print(f"    Expected Volatility: {bp['volatility']:.4f} ({bp['volatility']:.2%})")
            print(f"    Sharpe Ratio:        {bp['sharpe']:.4f}")
            print(f"    Top Holdings:")
            sorted_weights = sorted(bp['weights'].items(), key=lambda x: x[1], reverse=True)
            for ticker, weight in sorted_weights[:10]:
                print(f"      {ticker:15s} {weight:6.2%}")

        # Validate Monte Carlo
        mc_validation = validate_monte_carlo_results(results, investor_profile)
        report['mc_validation'] = mc_validation
        print(f"\n  Monte Carlo Coverage:")
        print(f"    Stocks with MC data:    {mc_validation['stocks_with_mc']}")
        print(f"    Stocks without MC data: {mc_validation['stocks_without_mc']}")

    except Exception as e:
        report['errors'].append(f"Validation failed: {e}")
        print(f"  [FAIL] Validation error: {e}")
        traceback.print_exc()

    # ---- Step 7 (optional): Save to database ----
    if save_to_db:
        print(f"\n[Step 7] Saving test portfolio run to database...")
        try:
            run_id = db_interactions.create_portfolio_run(
                risk_level=investor_profile.risk_level.value,
                investment_years=investor_profile.investment_years,
                portfolio_size=investor_profile.portfolio_size,
                run_name=f"TEST_RUN_{profile_name}_{datetime.date.today()}"
            )
            print(f"  Created portfolio run ID: {run_id}")

            successful_count = sum(1 for r in results.values() if r.success)
            failed_count = sum(1 for r in results.values() if not r.success)

            bp = validation.get('best_portfolio', {})
            db_interactions.update_portfolio_run(
                run_id=run_id,
                total_stocks_analyzed=len(selected_tickers),
                successful_predictions=successful_count,
                failed_predictions=failed_count,
                expected_return=bp.get('return'),
                expected_volatility=bp.get('volatility'),
                sharpe_ratio=bp.get('sharpe'),
                status='completed',
                execution_time_seconds=time.time() - overall_start
            )
            print(f"  Updated run with results")

            # Save portfolio holdings
            if validation.get('best_portfolio') and validation['best_portfolio'].get('weights'):
                holdings_data = []
                for rank, (ticker, weight) in enumerate(
                    sorted(validation['best_portfolio']['weights'].items(),
                           key=lambda x: x[1], reverse=True), 1
                ):
                    holdings_data.append({
                        'ticker': ticker,
                        'weight': weight,
                        'rank': rank
                    })

                holdings_df = pd.DataFrame(holdings_data)
                rows_inserted = db_interactions.export_portfolio_holdings(run_id, holdings_df)
                print(f"  Saved {rows_inserted} portfolio holdings")

            report['steps']['save_to_db'] = f'OK (run_id={run_id})'

        except Exception as e:
            report['errors'].append(f"DB save failed: {e}")
            report['steps']['save_to_db'] = f'FAILED: {e}'
            print(f"  [WARNING] Could not save to DB: {e}")
    else:
        report['steps']['save_to_db'] = 'SKIPPED'

    # ---- Final Summary ----
    overall_time = time.time() - overall_start
    report['execution_time'] = overall_time

    all_checks_passed = validation.get('passed', False) if 'validation' in report and report['validation'] else False
    report['overall_passed'] = all_checks_passed and len(report['errors']) == 0

    print("\n" + "=" * 80)
    print("TEST RESULTS SUMMARY")
    print("=" * 80)
    print(f"  Profile:          {profile_name}")
    print(f"  Stocks tested:    {len(selected_tickers)}")
    print(f"  Execution time:   {overall_time:.2f}s")
    print(f"  Errors:           {len(report['errors'])}")
    print()
    for step, status in report['steps'].items():
        icon = "+" if 'OK' in str(status) or 'SKIP' in str(status) else "x"
        print(f"  [{icon}] {step}: {status}")
    print()
    overall_status = "PASSED" if report['overall_passed'] else "FAILED"
    print(f"  OVERALL: {overall_status}")
    print("=" * 80)

    if report['errors']:
        print("\n  Errors encountered:")
        for err in report['errors']:
            print(f"    - {err}")

    return report['overall_passed'], report


def _run_efficient_frontier_reduced(price_df: pd.DataFrame, n_simulations: int) -> pd.DataFrame:
    """
    Run a reduced-simulation version of efficient frontier for faster testing.
    
    This reimplements the core efficient frontier logic with a configurable
    number of simulations instead of the hardcoded 750,000.
    
    Args:
        price_df: Price DataFrame (tickers as columns).
        n_simulations: Number of Monte Carlo portfolio simulations.
        
    Returns:
        Portfolio DataFrame with weights, returns, and volatilities.
    """
    print(f"  Running reduced efficient frontier ({n_simulations:,} simulations)...")

    log_returns_df = np.log(1 + price_df.pct_change(1).dropna())
    log_returns_mean = log_returns_df.mean() * 252
    log_returns_mean = log_returns_mean.to_frame().rename(columns={0: "Mean"}).transpose()
    log_returns_df = log_returns_df[log_returns_mean.columns]
    log_returns_cov = log_returns_df.cov() * 252

    portefolio_number = []
    portefolio_weight = []
    portfolio_returns = []
    portfolio_volatilities = []

    for sim in range(n_simulations):
        portefolio_number.append(sim)
        weights = np.random.random(len(log_returns_mean.columns))
        weights /= np.sum(weights)
        portefolio_weight.append(weights)
        portfolio_returns.append(np.sum(weights * np.array(log_returns_mean)))
        portfolio_volatilities.append(
            np.sqrt(np.dot(weights.T, np.dot(log_returns_cov, weights)))
        )

        if sim % 10000 == 0 and sim > 0:
            print(f"    Simulations: {sim:,}")

    portefolio_number_df = pd.DataFrame(portefolio_number, columns=["Portefolio number"])
    portefolio_weight_df = pd.DataFrame(portefolio_weight, columns=log_returns_mean.columns)
    portfolio_returns_df = pd.DataFrame(
        {'Return': portfolio_returns, 'Volatility': portfolio_volatilities}
    )

    portefolio_df = pd.concat(
        [portefolio_number_df, portefolio_weight_df, portfolio_returns_df], axis=1
    )
    portefolio_df = portefolio_df.sort_values(by="Volatility", ascending=True)

    # Reduce frontier (remove dominated portfolios)
    if len(price_df.columns) > 2:
        portefolio_df["Volatility"] = portefolio_df["Volatility"].round(3)
        
        # For each volatility bucket, keep the row with the highest Return
        # Use idxmax on Return to get the index of the best row per group
        best_idx = portefolio_df.groupby("Volatility")["Return"].idxmax()
        portefolio_df = portefolio_df.loc[best_idx]
        
        columns = list(portefolio_df.columns)
        if "Volatility" not in columns:
            columns.append("Volatility")
        portefolio_df = portefolio_df.reset_index(drop=True)

        loop = True
        while loop:
            x = 0
            drop_list = []
            for index, row in portefolio_df.iterrows():
                if index > 0:
                    if row["Return"] < portefolio_df.iloc[index - 1]["Return"]:
                        drop_list.append(index)
                        x += 1
            portefolio_df = portefolio_df.drop(drop_list)
            portefolio_df = portefolio_df.reset_index(drop=True)
            if x == 0:
                loop = False

    return portefolio_df


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Test portfolio construction pipeline using stored DB predictions"
    )
    parser.add_argument(
        '--profile', type=str, default='balanced',
        choices=['conservative', 'balanced', 'aggressive'],
        help='Investor profile to use (default: balanced)'
    )
    parser.add_argument(
        '--max-stocks', type=int, default=None,
        help='Maximum number of stocks to include (default: all available)'
    )
    parser.add_argument(
        '--save-to-db', action='store_true', default=False,
        help='Save the test portfolio run to the database'
    )
    parser.add_argument(
        '--run-fresh-mc', action='store_true', default=False,
        help='Run fresh Monte Carlo simulations on reconstructed forecasts'
    )
    parser.add_argument(
        '--mc-sims', type=int, default=500,
        help='Number of Monte Carlo simulations if --run-fresh-mc (default: 500)'
    )
    parser.add_argument(
        '--ef-sims', type=int, default=50000,
        help='Number of efficient frontier simulations (default: 50000 for testing speed, '
             'production uses 750000). Set to 0 to use the full production count.'
    )
    parser.add_argument(
        '--verbose', action='store_true', default=True,
        help='Print detailed output (default: True)'
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    ef_override = args.ef_sims if args.ef_sims > 0 else None

    success, report = run_portfolio_construction_test(
        profile_name=args.profile,
        max_stocks=args.max_stocks,
        save_to_db=args.save_to_db,
        run_fresh_mc=args.run_fresh_mc,
        mc_sim_amount=args.mc_sims,
        ef_sim_override=ef_override,
        verbose=args.verbose
    )

    sys.exit(0 if success else 1)
