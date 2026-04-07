"""
Portfolio Builder Module — Phase 3 of the portfolio pipeline.

Constructs an optimized portfolio from existing stock predictions in the database.
Uses efficient frontier optimization and portfolio-level Monte Carlo simulation.

Execution strategy:
    1. Query database for tickers with recent predictions
    2. Rank stocks by predicted return / risk score based on investor risk level
    3. Select top N stocks for portfolio
    4. Fetch historical prices for selected stocks
    5. Run efficient frontier optimization (risk-aware)
    6. Run portfolio-level Monte Carlo simulation
    7. Export portfolio holdings and metrics to database

This module can be run independently of model_trainer.py and price_predictor.py.
It only reads from the database — no ML training or prediction is performed here.

Usage:
    # Build a balanced portfolio from existing predictions
    python portfolio_builder.py

    # Or import and call programmatically
    from portfolio_builder import run_portfolio_construction
    from portfolio_config import InvestorProfile, RiskLevel
    profile = InvestorProfile(risk_level=RiskLevel.MEDIUM, investment_years=7, portfolio_size=25)
    result = run_portfolio_construction(investor_profile=profile)
"""

import os
import sys
import time
import traceback
from typing import List, Optional, Dict

import numpy as np
import pandas as pd

import fetch_secrets
import db_connectors
import db_interactions
import efficient_frontier
import monte_carlo_sim
from portfolio_config import (
    InvestorProfile,
    RiskLevel,
    InvestmentStrategy,
    get_default_profile,
    DEFAULT_RISK_FREE_RATE
)
from blacklist_manager import get_blacklist_manager


# ─── Strategy-aware fundamental scoring ──────────────────────────────

def _get_fundamental_score(
    ticker: str,
    strategy: InvestmentStrategy,
    current_price: float,
    price_df_cache: dict,
) -> float:
    """
    Compute a normalised fundamental bonus in [-1, +1] for *ticker* under the
    given *strategy*.  Returns 0.0 when data is missing so that the ML
    prediction score still dominates.
    """
    if strategy == InvestmentStrategy.BALANCED:
        return 0.0                       # pure ML — no fundamental overlay

    score = 0.0
    components = 0                       # track how many signals contributed

    # --- ratios (P/E, P/S, P/B, P/FCF) ---
    try:
        ratio_df = db_interactions.import_stock_ratio_data(amount=1, stock_ticker=ticker)
    except Exception:
        ratio_df = pd.DataFrame()
    pe = ratio_df['p_e'].iloc[-1] if (not ratio_df.empty and 'p_e' in ratio_df.columns and pd.notna(ratio_df['p_e'].iloc[-1])) else None
    ps = ratio_df['p_s'].iloc[-1] if (not ratio_df.empty and 'p_s' in ratio_df.columns and pd.notna(ratio_df['p_s'].iloc[-1])) else None
    pb = ratio_df['p_b'].iloc[-1] if (not ratio_df.empty and 'p_b' in ratio_df.columns and pd.notna(ratio_df['p_b'].iloc[-1])) else None

    # --- quarterly income (TTM growth, margins) ---
    try:
        qi = db_interactions.import_quarterly_income_data(ticker)
    except Exception:
        qi = pd.DataFrame()
    rev_g = qi['revenue_growth_yoy'].iloc[0] if (not qi.empty and 'revenue_growth_yoy' in qi.columns and pd.notna(qi['revenue_growth_yoy'].iloc[0])) else None
    eps_g = qi['eps_growth_yoy'].iloc[0] if (not qi.empty and 'eps_growth_yoy' in qi.columns and pd.notna(qi['eps_growth_yoy'].iloc[0])) else None
    op_margin = qi['operating_margin_ttm'].iloc[0] if (not qi.empty and 'operating_margin_ttm' in qi.columns and pd.notna(qi['operating_margin_ttm'].iloc[0])) else None
    net_margin = qi['net_margin_ttm'].iloc[0] if (not qi.empty and 'net_margin_ttm' in qi.columns and pd.notna(qi['net_margin_ttm'].iloc[0])) else None

    # --- quarterly balance sheet (ROE, D/E) ---
    try:
        qb = db_interactions.import_quarterly_balancesheet_data(ticker)
    except Exception:
        qb = pd.DataFrame()
    roe = qb['roe_ttm'].iloc[0] if (not qb.empty and 'roe_ttm' in qb.columns and pd.notna(qb['roe_ttm'].iloc[0])) else None
    de = qb['debt_to_equity'].iloc[0] if (not qb.empty and 'debt_to_equity' in qb.columns and pd.notna(qb['debt_to_equity'].iloc[0])) else None

    # --- quarterly cash flow (dividends) ---
    try:
        qc = db_interactions.import_quarterly_cashflow_data(ticker)
    except Exception:
        qc = pd.DataFrame()
    div_ttm = qc['dividends_paid_ttm'].iloc[0] if (not qc.empty and 'dividends_paid_ttm' in qc.columns and pd.notna(qc['dividends_paid_ttm'].iloc[0])) else None
    shares = qi['shares_diluted'].iloc[0] if (not qi.empty and 'shares_diluted' in qi.columns and pd.notna(qi['shares_diluted'].iloc[0])) else None

    # --- price momentum ---
    def _price_momentum(ticker_sym):
        """Return (3M return, 6M return) or (None, None)."""
        if ticker_sym in price_df_cache:
            pdf = price_df_cache[ticker_sym]
        else:
            try:
                pdf = db_interactions.import_stock_price_data(amount=252, stock_ticker=ticker_sym)
                price_df_cache[ticker_sym] = pdf
            except Exception:
                return None, None
        if pdf.empty or 'close_Price' not in pdf.columns or len(pdf) < 63:
            return None, None
        closes = pdf['close_Price'].dropna()
        m3 = (closes.iloc[-1] / closes.iloc[-63] - 1) if len(closes) >= 63 else None
        m6 = (closes.iloc[-1] / closes.iloc[-126] - 1) if len(closes) >= 126 else None
        return m3, m6

    # ── Strategy scoring ────────────────────────────────────────
    if strategy == InvestmentStrategy.DIVIDEND:
        # Positive if company pays dividends (dividends_paid_ttm is negative cash outflow)
        if div_ttm is not None and shares and current_price and current_price > 0:
            dps = abs(div_ttm) / shares
            div_yield = dps / current_price
            if div_yield > 0:
                score += min(div_yield / 0.05, 1.0)   # cap at 5% yield → 1.0
                components += 1
        # Penalise very high P/E (unsustainable)
        if pe is not None and pe > 0:
            score += max(1.0 - pe / 30.0, -0.5)
            components += 1

    elif strategy == InvestmentStrategy.GROWTH:
        if rev_g is not None:
            score += np.clip(rev_g / 0.25, -1, 1)     # 25% rev growth → 1.0
            components += 1
        if eps_g is not None:
            score += np.clip(eps_g / 0.30, -1, 1)
            components += 1

    elif strategy == InvestmentStrategy.VALUE:
        if pe is not None and pe > 0:
            score += np.clip(1.0 - pe / 25.0, -1, 1)  # P/E 12.5 → 0.5
            components += 1
        if pb is not None and pb > 0:
            score += np.clip(1.0 - pb / 3.0, -1, 1)
            components += 1
        if ps is not None and ps > 0:
            score += np.clip(1.0 - ps / 4.0, -1, 1)
            components += 1

    elif strategy == InvestmentStrategy.GARP:
        # PEG-style: want low P/E relative to earnings growth
        if pe is not None and eps_g is not None and eps_g > 0 and pe > 0:
            peg = pe / (eps_g * 100)                    # eps_g is decimal
            score += np.clip(1.0 - peg / 2.0, -1, 1)   # PEG 1.0 → 0.5
            components += 1
        if rev_g is not None:
            score += np.clip(rev_g / 0.20, -0.5, 0.5)
            components += 1

    elif strategy == InvestmentStrategy.QUALITY:
        if roe is not None:
            score += np.clip(roe / 0.20, -1, 1)        # 20% ROE → 1.0
            components += 1
        if op_margin is not None:
            score += np.clip(op_margin / 0.20, -0.5, 0.5)
            components += 1
        if de is not None:
            score += np.clip(1.0 - de / 2.0, -1, 1)   # D/E 1.0 → 0.5
            components += 1

    elif strategy == InvestmentStrategy.MOMENTUM:
        m3, m6 = _price_momentum(ticker)
        if m3 is not None:
            score += np.clip(m3 / 0.15, -1, 1)         # 15% 3M return → 1.0
            components += 1
        if m6 is not None:
            score += np.clip(m6 / 0.30, -1, 1)
            components += 1

    # Average & normalise to [-1, 1]
    if components > 0:
        score = score / components
    return np.clip(score, -1.0, 1.0)


def rank_and_select_stocks(
    prediction_tickers: List[str],
    portfolio_size: int = 25,
    risk_level: RiskLevel = RiskLevel.MEDIUM,
    max_prediction_age_days: int = 7,
    strategy: InvestmentStrategy = InvestmentStrategy.BALANCED,
) -> List[str]:
    """
    Rank stock tickers by their predicted return/risk from stored predictions,
    optionally blended with a fundamental strategy score, and select the top N.
    
    Ranking strategy by risk level:
        - LOW: Favors stocks with tighter confidence bounds (lower uncertainty)
        - MEDIUM: Balances predicted return with uncertainty-adjusted score
        - HIGH: Favors highest predicted returns
    
    Investment strategy overlay:
        The ``strategy`` parameter (default BALANCED = ML-only) adds a
        fundamental bonus/penalty to the composite score.  The bonus is
        weighted at 30 % of the final score to keep ML predictions dominant.
    
    Args:
        prediction_tickers: List of tickers with available predictions
        portfolio_size: Maximum number of stocks to select
        risk_level: Investor risk level
        max_prediction_age_days: Maximum age of predictions to consider
        strategy: Investment strategy for fundamental scoring
        
    Returns:
        List of selected ticker symbols, ranked best to worst
    """
    rankings = []
    price_df_cache: dict = {}  # shared cache for momentum lookups

    for ticker in prediction_tickers:
        summary = db_interactions.get_prediction_summary(
            ticker, max_age_days=max_prediction_age_days
        )
        if summary is None:
            continue

        # Find the 252-day (1-year) horizon, falling back to longest available
        horizons = summary['horizons']
        best_horizon = None
        for h in horizons:
            if h['prediction_horizon_days'] == 252:
                best_horizon = h
                break
        if best_horizon is None and horizons:
            best_horizon = max(horizons, key=lambda x: x['prediction_horizon_days'])

        if best_horizon is None or best_horizon.get('predicted_return') is None:
            continue

        predicted_return = best_horizon['predicted_return']
        pred_std = best_horizon.get('prediction_std')
        conf_lower_5 = best_horizon.get('confidence_lower_5')
        conf_upper_95 = best_horizon.get('confidence_upper_95')
        current_price = best_horizon.get('current_price', 0)

        # Calculate downside risk proxy from confidence bounds
        downside_risk = None
        if conf_lower_5 is not None and current_price and current_price > 0:
            downside_risk = (conf_lower_5 - current_price) / current_price

        upside_potential = None
        if conf_upper_95 is not None and current_price and current_price > 0:
            upside_potential = (conf_upper_95 - current_price) / current_price

        # Calculate composite score based on risk level
        if risk_level == RiskLevel.LOW:
            downside_penalty = min(downside_risk or -0.5, 0)
            score = predicted_return + 2.0 * downside_penalty
        elif risk_level == RiskLevel.HIGH:
            upside_bonus = max(upside_potential or 0, 0) * 0.2
            score = predicted_return + upside_bonus
        else:  # MEDIUM
            downside_penalty = min(downside_risk or -0.3, 0) * 0.5
            score = predicted_return + downside_penalty

        # Blend in fundamental strategy score (30 % weight)
        fund_score = _get_fundamental_score(
            ticker, strategy, current_price or 0, price_df_cache
        )
        # ML score is normalised to roughly [-1, 1] via predicted_return,
        # so blend: 70 % ML + 30 % fundamental overlay
        score = 0.70 * score + 0.30 * fund_score

        rankings.append({
            'ticker': ticker,
            'predicted_return': predicted_return,
            'downside_risk': downside_risk,
            'upside_potential': upside_potential,
            'pred_std': pred_std,
            'fund_score': fund_score,
            'score': score
        })

    if not rankings:
        return []

    # Sort by score descending
    rankings.sort(key=lambda x: x['score'], reverse=True)

    # Select top N
    selected = rankings[:portfolio_size]

    # Print ranking report
    print(f"\n{'=' * 90}")
    print(f"STOCK RANKING & SELECTION ({risk_level.value.upper()} risk, {strategy.value.upper()} strategy)")
    print(f"{'=' * 90}")
    print(f"Eligible stocks: {len(rankings)}, Selecting top {min(portfolio_size, len(rankings))}")
    print(f"\n{'Rank':<6} {'Ticker':<14} {'Pred Return':<14} {'Downside':<12} {'Upside':<12} {'Fund':<8} {'Score':<10} {'Sel'}")
    print(f"{'-' * 90}")

    for i, r in enumerate(rankings[:min(30, len(rankings))]):
        is_selected = "  *" if i < portfolio_size else ""
        ds = f"{r['downside_risk']:.2%}" if r['downside_risk'] is not None else "N/A"
        us = f"{r['upside_potential']:.2%}" if r['upside_potential'] is not None else "N/A"
        fs = f"{r['fund_score']:+.2f}"
        print(f"{i+1:<6} {r['ticker']:<14} {r['predicted_return']:<14.2%} "
              f"{ds:<12} {us:<12} {fs:<8} {r['score']:<10.4f}{is_selected}")

    print(f"{'=' * 90}")

    return [r['ticker'] for r in selected]


def collect_historical_prices(
    selected_tickers: List[str],
    min_days: int = 504
) -> pd.DataFrame:
    """
    Collect historical closing prices from the database for selected tickers.
    
    Uses actual historical data (not forecasts) for portfolio optimization,
    as the efficient frontier should be built on historical return distributions.
    
    Args:
        selected_tickers: List of ticker symbols to fetch
        min_days: Minimum number of trading days required per stock
        
    Returns:
        DataFrame with tickers as columns and dates as index (closing prices)
    """
    price_data = {}
    dropped = []

    for ticker in selected_tickers:
        try:
            stock_df = db_interactions.import_stock_dataset(ticker)
            if stock_df is None or len(stock_df) < min_days:
                dropped.append(ticker)
                continue

            stock_df["date"] = pd.to_datetime(stock_df["date"])
            stock_df = stock_df.set_index("date").sort_index()

            if 'close_Price' in stock_df.columns:
                prices = stock_df['close_Price'].dropna()
            elif stock_df.shape[1] >= 5:
                prices = stock_df.iloc[:, 4].dropna()
            else:
                dropped.append(ticker)
                continue

            if len(prices) >= min_days:
                price_data[ticker] = prices
            else:
                dropped.append(ticker)

        except Exception as e:
            print(f"[WARNING] Could not fetch historical data for {ticker}: {e}")
            dropped.append(ticker)

    if dropped:
        print(f"[INFO] Dropped {len(dropped)} tickers with insufficient historical data: {dropped}")

    if not price_data:
        return pd.DataFrame()

    price_df = pd.DataFrame(price_data)
    price_df = price_df.dropna(how='any')

    print(f"[INFO] Historical price matrix: {len(price_df)} trading days x {len(price_df.columns)} stocks")

    return price_df


def run_portfolio_construction(
    investor_profile: InvestorProfile = None,
    max_prediction_age_days: int = 7,
    custom_excluded_tickers: Optional[List[str]] = None
) -> Dict:
    """
    Main entry point: construct an optimized portfolio from existing predictions.
    
    This does NOT train models or generate predictions. It uses whatever predictions
    already exist in the database.
    
    Pipeline:
        1. Find tickers with recent predictions
        2. Rank and select top stocks based on risk profile
        3. Fetch historical prices for selected stocks
        4. Run efficient frontier optimization
        5. Run portfolio-level Monte Carlo simulation
        6. Export holdings and metrics to database
    
    Args:
        investor_profile: Investor configuration (defaults to balanced if None)
        max_prediction_age_days: Max age of predictions to use
        custom_excluded_tickers: Additional tickers to exclude
        
    Returns:
        dict with 'ef_result', 'mc_result', 'selected_tickers', 'run_id'
    """
    overall_start = time.time()

    if investor_profile is None:
        investor_profile = get_default_profile("balanced")

    print("\n" + "=" * 70)
    print("PORTFOLIO BUILDER — Phase 3")
    print("=" * 70)
    print(f"Risk Level: {investor_profile.risk_level.value.upper()}")
    print(f"Investment Period: {investor_profile.investment_years} years")
    print(f"Target Portfolio Size: {investor_profile.portfolio_size} stocks")
    print(f"Volatility Cap: {investor_profile.get_volatility_cap():.0%}")
    print(f"Max Prediction Age: {max_prediction_age_days} days")
    print("=" * 70 + "\n")

    # Combine exclusions
    blacklisted = get_blacklist_manager().get_blacklist()
    all_excluded = list(set(
        investor_profile.excluded_tickers +
        blacklisted +
        (custom_excluded_tickers or [])
    ))

    # Step 1: Find tickers with recent predictions
    prediction_tickers = db_interactions.get_tickers_with_predictions(
        max_age_days=max_prediction_age_days
    )
    prediction_tickers = [t for t in prediction_tickers if t not in all_excluded]

    print(f"[INFO] Found {len(prediction_tickers)} tickers with recent predictions")
    print(f"[INFO] Excluded {len(all_excluded)} tickers\n")

    if len(prediction_tickers) < 2:
        print("[ERROR] Not enough tickers with predictions for portfolio construction (need >= 2)")
        return {'ef_result': None, 'mc_result': None, 'selected_tickers': [], 'run_id': None}

    # Create portfolio run record in database
    run_id = None
    try:
        run_id = db_interactions.create_portfolio_run(
            risk_level=investor_profile.risk_level.value,
            investment_years=investor_profile.investment_years,
            portfolio_size=investor_profile.portfolio_size,
            excluded_tickers=all_excluded
        )
        print(f"[DB] Created portfolio run ID: {run_id}")
    except Exception as e:
        print(f"[WARNING] Could not create portfolio run record: {e}")

    # Step 2: Rank and select stocks
    print("\n[STEP 2] Ranking stocks by predicted return / risk...")
    selected_tickers = rank_and_select_stocks(
        prediction_tickers=prediction_tickers,
        portfolio_size=investor_profile.portfolio_size,
        risk_level=investor_profile.risk_level,
        max_prediction_age_days=max_prediction_age_days
    )

    if len(selected_tickers) < 2:
        print("[ERROR] Not enough selected stocks for portfolio optimization")
        if run_id:
            try:
                db_interactions.update_portfolio_run(
                    run_id=run_id, status='failed',
                    error_message="Not enough stocks for optimization"
                )
            except Exception:
                pass
        return {'ef_result': None, 'mc_result': None, 'selected_tickers': selected_tickers, 'run_id': run_id}

    # Update run with prediction counts
    if run_id:
        try:
            db_interactions.update_portfolio_run(
                run_id=run_id,
                total_stocks_analyzed=len(prediction_tickers),
                successful_predictions=len(selected_tickers),
                failed_predictions=len(prediction_tickers) - len(selected_tickers)
            )
        except Exception as e:
            print(f"[WARNING] Could not update portfolio run: {e}")

    # Step 3: Fetch historical prices
    print(f"\n[STEP 3] Fetching historical prices for {len(selected_tickers)} selected stocks...")
    historical_prices = collect_historical_prices(selected_tickers)

    if historical_prices.empty or len(historical_prices.columns) < 2:
        print("[ERROR] Not enough historical price data for portfolio optimization")
        if run_id:
            try:
                db_interactions.update_portfolio_run(
                    run_id=run_id, status='failed',
                    error_message="Insufficient historical price data"
                )
            except Exception:
                pass
        return {'ef_result': None, 'mc_result': None, 'selected_tickers': selected_tickers, 'run_id': run_id}

    # Step 4: Efficient frontier optimization
    print(f"\n[STEP 4] Running efficient frontier optimization...")
    ef_result = None
    pf_mc_result = None

    try:
        risk_free_rate = DEFAULT_RISK_FREE_RATE
        volatility_cap = investor_profile.get_volatility_cap()

        max_weight = min(0.25, 1.0 / max(len(historical_prices.columns) * 0.5, 1))
        max_weight = max(max_weight, 0.05)

        ef_result = efficient_frontier.optimize_portfolio(
            price_df=historical_prices,
            risk_level=investor_profile.risk_level.value,
            volatility_cap=volatility_cap,
            risk_free_rate=risk_free_rate,
            max_weight_per_stock=max_weight,
            min_weight_per_stock=0.0,
            mc_simulations=100000,
            plot=True
        )

        print("[OK] Efficient frontier optimization complete!")

    except Exception as ef_error:
        print(f"[ERROR] Efficient frontier optimization failed: {ef_error}")
        traceback.print_exc()
        if run_id:
            try:
                db_interactions.update_portfolio_run(
                    run_id=run_id, status='failed',
                    error_message=f"EF optimization failed: {str(ef_error)}"
                )
            except Exception:
                pass
        return {'ef_result': None, 'mc_result': None, 'selected_tickers': selected_tickers, 'run_id': run_id}

    # Step 5: Portfolio-level Monte Carlo
    print(f"\n[STEP 5] Running portfolio Monte Carlo simulation...")
    try:
        pf_mc_result = monte_carlo_sim.portfolio_monte_carlo(
            weights=list(ef_result['optimal_weights'].values()),
            mean_returns=ef_result['mean_returns'].values,
            cov_matrix=ef_result['cov_matrix'].values,
            initial_investment=100000,
            years=investor_profile.investment_years,
            num_simulations=5000,
            seed=42
        )
        print("[OK] Portfolio Monte Carlo complete!")
    except Exception as mc_error:
        print(f"[ERROR] Portfolio Monte Carlo failed: {mc_error}")
        traceback.print_exc()

    # Step 6: Export to database
    if run_id and ef_result is not None:
        # Export holdings
        if ef_result.get('holdings_df') is not None:
            try:
                holdings_exported = db_interactions.export_portfolio_holdings(
                    run_id=run_id,
                    holdings_df=ef_result['holdings_df']
                )
                print(f"[DB] Exported {holdings_exported} portfolio holdings")
            except Exception as db_error:
                print(f"[WARNING] Could not export holdings to DB: {db_error}")

        # Update portfolio run with metrics
        try:
            update_kwargs = {
                'run_id': run_id,
                'expected_return': ef_result['expected_return'],
                'expected_volatility': ef_result['expected_volatility'],
                'sharpe_ratio': ef_result['sharpe_ratio'],
                'status': 'completed',
            }

            if pf_mc_result is not None:
                update_kwargs['mc_return_p5'] = pf_mc_result['final_return_p5']
                update_kwargs['mc_return_mean'] = pf_mc_result['final_return_mean']
                update_kwargs['mc_return_p95'] = pf_mc_result['final_return_p95']

            db_interactions.update_portfolio_run(**update_kwargs)
            print("[DB] Updated portfolio run with optimization and MC metrics")
        except Exception as db_error:
            print(f"[WARNING] Could not update portfolio run metrics: {db_error}")

    # Finalize timing
    overall_time = time.time() - overall_start
    if run_id:
        try:
            db_interactions.update_portfolio_run(
                run_id=run_id,
                execution_time_seconds=overall_time
            )
        except Exception:
            pass

    # Print final summary
    if ef_result is not None:
        print(f"\n{'=' * 70}")
        print(f"OPTIMIZED PORTFOLIO ({investor_profile.risk_level.value.upper()} RISK)")
        print(f"{'=' * 70}")
        print(f"Expected Annual Return:  {ef_result['expected_return']:.2%}")
        print(f"Expected Volatility:     {ef_result['expected_volatility']:.2%}")
        print(f"Sharpe Ratio:            {ef_result['sharpe_ratio']:.4f}")
        print(f"Number of Holdings:      {len(ef_result['holdings_df'])}")

        if pf_mc_result is not None:
            print(f"\n{investor_profile.investment_years}-Year Monte Carlo Outlook:")
            print(f"  Mean Return:           {pf_mc_result['final_return_mean']:.2%}")
            print(f"  Downside (5th pct):    {pf_mc_result['final_return_p5']:.2%}")
            print(f"  Upside (95th pct):     {pf_mc_result['final_return_p95']:.2%}")
            print(f"  Mean Final Value:      {pf_mc_result['final_value_mean']:,.0f} (from 100,000)")

        print(f"\nTotal execution time: {overall_time:.1f}s")
        print(f"{'=' * 70}")

    return {
        'ef_result': ef_result,
        'mc_result': pf_mc_result,
        'selected_tickers': selected_tickers,
        'run_id': run_id
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build optimized stock portfolio")
    parser.add_argument("--risk", choices=["low", "medium", "high"], default="medium",
                        help="Risk level (default: medium)")
    parser.add_argument("--years", type=int, default=7,
                        help="Investment horizon in years (default: 7)")
    parser.add_argument("--size", type=int, default=25,
                        help="Portfolio size (default: 25)")
    parser.add_argument("--max-pred-age", type=int, default=7,
                        help="Max prediction age in days (default: 7)")
    parser.add_argument("--profile", choices=["conservative", "balanced", "aggressive"],
                        default=None,
                        help="Use a preset profile instead of --risk/--years/--size")

    args = parser.parse_args()

    if args.profile:
        profile = get_default_profile(args.profile)
    else:
        profile = InvestorProfile(
            risk_level=RiskLevel(args.risk),
            investment_years=args.years,
            portfolio_size=args.size
        )

    result = run_portfolio_construction(
        investor_profile=profile,
        max_prediction_age_days=args.max_pred_age
    )
