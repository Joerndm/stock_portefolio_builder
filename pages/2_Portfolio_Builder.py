"""
Portfolio Builder — Interactive Portfolio Construction

Allows the user to:
1. Configure investor profile (risk level, years, portfolio size)
2. Exclude specific tickers
3. Run portfolio optimization (efficient frontier + Monte Carlo)
4. View results interactively
5. Refine by adjusting parameters and re-running
"""
import os
import sys
import time
import datetime

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import gui_data
from portfolio_config import (
    InvestorProfile, RiskLevel, InvestmentStrategy,
    VOLATILITY_CAPS, DEFAULT_RISK_FREE_RATE,
    STRATEGY_LABELS, STRATEGY_DESCRIPTIONS,
)

# ─── Page header ─────────────────────────────────────────────────────
st.title("🔧 Portfolio Builder")
st.caption("Configure your investor profile and build an optimized portfolio")

# ─── Sidebar: Investor Profile Configuration ─────────────────────────
with st.sidebar:
    st.subheader("Investor Profile")

    risk_options = {"Conservative (Low)": "low", "Balanced (Medium)": "medium", "Aggressive (High)": "high"}
    risk_label = st.selectbox(
        "Risk Tolerance",
        options=list(risk_options.keys()),
        index=1,
        help="LOW = minimize volatility, MEDIUM = maximize Sharpe ratio, HIGH = maximize returns"
    )
    risk_level_str = risk_options[risk_label]
    risk_level = RiskLevel(risk_level_str)

    # Investment strategy
    strategy_options = {STRATEGY_LABELS[s]: s for s in InvestmentStrategy}
    strategy_label = st.selectbox(
        "Investment Strategy",
        options=list(strategy_options.keys()),
        index=0,
        help="Blends a fundamental overlay (30 %) with ML predictions (70 %)"
    )
    strategy = strategy_options[strategy_label]
    st.caption(STRATEGY_DESCRIPTIONS[strategy])

    investment_years = st.slider(
        "Investment Horizon (years)",
        min_value=1, max_value=10, value=7,
        help="How long you plan to hold this portfolio"
    )

    portfolio_size = st.slider(
        "Portfolio Size (stocks)",
        min_value=10, max_value=30, value=25,
        help="Number of stocks in the portfolio"
    )

    max_pred_age = st.slider(
        "Max Prediction Age (days)",
        min_value=1, max_value=60, value=30,
        help="Only use predictions newer than this"
    )

    st.divider()
    vol_cap = VOLATILITY_CAPS[risk_level]
    st.caption(f"Volatility cap: **{vol_cap:.0%}**")
    st.caption(f"Risk-free rate: **{DEFAULT_RISK_FREE_RATE:.1%}**")

# ─── Main area: Ticker selection & exclusion ─────────────────────────
st.subheader("Stock Universe")

# Load available data
all_tickers = gui_data.get_all_tickers()
pred_tickers = gui_data.get_tickers_with_predictions(max_age_days=max_pred_age)
stock_info = gui_data.get_stock_info()

col_info, col_exclude = st.columns([2, 1])

with col_info:
    st.info(
        f"**{len(all_tickers)}** total tickers in database  •  "
        f"**{len(pred_tickers)}** with predictions (≤{max_pred_age} days old)"
    )

with col_exclude:
    st.markdown("")  # spacer

# Ticker exclusion
excluded_tickers = st.multiselect(
    "Exclude tickers",
    options=sorted(pred_tickers),
    default=[],
    help="Select tickers to exclude from portfolio construction",
    placeholder="Search and select tickers to exclude..."
)

# Show prediction overview
if pred_tickers:
    with st.expander(f"Preview: Available Predictions ({len(pred_tickers)} tickers)", expanded=False):
        preview_data = []
        # Limit preview to first 30 to avoid slow loading
        for ticker in sorted(pred_tickers)[:30]:
            summary = gui_data.get_prediction_summary(ticker, max_age_days=max_pred_age)
            if summary and summary.get('horizons'):
                best = max(summary['horizons'], key=lambda h: h.get('prediction_horizon_days', 0))
                preview_data.append({
                    'Ticker': ticker,
                    'Pred. Date': summary.get('prediction_date', ''),
                    'Price': f"{summary.get('current_price', 0):.2f}",
                    'Horizon (days)': best.get('prediction_horizon_days', ''),
                    'Pred. Return': f"{best.get('predicted_return', 0):.2%}" if best.get('predicted_return') else '',
                    'Model': best.get('model_type', ''),
                })
        if preview_data:
            st.dataframe(pd.DataFrame(preview_data), width="stretch", hide_index=True)
        if len(pred_tickers) > 30:
            st.caption(f"Showing first 30 of {len(pred_tickers)} tickers")

# ─── Run Portfolio Optimization ──────────────────────────────────────
st.markdown("---")
st.subheader("Build Portfolio")

eligible_count = len([t for t in pred_tickers if t not in excluded_tickers])

if eligible_count < 2:
    st.error(f"Need at least 2 stocks with predictions. Currently have {eligible_count}.")
    st.stop()

st.markdown(
    f"Ready to build portfolio with **{eligible_count}** eligible stocks, "
    f"selecting top **{portfolio_size}**, at **{risk_label}** risk, "
    f"**{strategy_label}** strategy."
)

if st.button("🚀 Build Optimized Portfolio", type="primary", use_container_width=True):
    # Build the profile
    try:
        profile = InvestorProfile(
            risk_level=risk_level,
            investment_years=investment_years,
            portfolio_size=portfolio_size,
            excluded_tickers=excluded_tickers,
            strategy=strategy
        )
    except ValueError as e:
        st.error(f"Invalid profile: {e}")
        st.stop()

    # Run optimization
    with st.spinner("Running portfolio optimization... This may take a minute."):
        progress = st.progress(0, text="Initializing...")

        try:
            import portfolio_builder

            progress.progress(10, text="Ranking stocks by predicted return...")

            # Step 1: Get eligible tickers
            eligible = [t for t in pred_tickers if t not in excluded_tickers]

            # Step 2: Rank and select
            progress.progress(20, text="Selecting top stocks...")
            selected = portfolio_builder.rank_and_select_stocks(
                prediction_tickers=eligible,
                portfolio_size=profile.portfolio_size,
                risk_level=profile.risk_level,
                max_prediction_age_days=max_pred_age,
                strategy=profile.strategy
            )

            if len(selected) < 2:
                st.error("Not enough stocks passed ranking. Try increasing prediction age or reducing exclusions.")
                st.stop()

            progress.progress(40, text=f"Fetching historical prices for {len(selected)} stocks...")

            # Step 3: Historical prices
            historical_prices = portfolio_builder.collect_historical_prices(selected)
            if historical_prices.empty or len(historical_prices.columns) < 2:
                st.error("Not enough historical price data for selected stocks.")
                st.stop()

            progress.progress(60, text="Running efficient frontier optimization...")

            # Step 4: Efficient frontier
            import efficient_frontier
            vol_cap = profile.get_volatility_cap()
            max_weight = min(0.25, 1.0 / max(len(historical_prices.columns) * 0.5, 1))
            max_weight = max(max_weight, 0.05)

            ef_result = efficient_frontier.optimize_portfolio(
                price_df=historical_prices,
                risk_level=profile.risk_level.value,
                volatility_cap=vol_cap,
                risk_free_rate=DEFAULT_RISK_FREE_RATE,
                max_weight_per_stock=max_weight,
                min_weight_per_stock=0.0,
                mc_simulations=50000,  # Reduced for GUI responsiveness
                plot=True
            )

            progress.progress(80, text="Running portfolio Monte Carlo...")

            # Step 5: Portfolio Monte Carlo
            import monte_carlo_sim
            mc_result = monte_carlo_sim.portfolio_monte_carlo(
                weights=list(ef_result['optimal_weights'].values()),
                mean_returns=ef_result['mean_returns'].values,
                cov_matrix=ef_result['cov_matrix'].values,
                initial_investment=100000,
                years=profile.investment_years,
                num_simulations=3000,  # Reduced for GUI
                seed=42
            )

            progress.progress(90, text="Saving results to database...")

            # Step 6: Save to DB
            import db_interactions
            run_id = None
            try:
                run_id = db_interactions.create_portfolio_run(
                    risk_level=profile.risk_level.value,
                    investment_years=profile.investment_years,
                    portfolio_size=profile.portfolio_size,
                    excluded_tickers=excluded_tickers
                )

                if ef_result.get('holdings_df') is not None:
                    db_interactions.export_portfolio_holdings(run_id, ef_result['holdings_df'])

                update_kwargs = {
                    'run_id': run_id,
                    'total_stocks_analyzed': len(eligible),
                    'successful_predictions': len(selected),
                    'failed_predictions': len(eligible) - len(selected),
                    'expected_return': ef_result['expected_return'],
                    'expected_volatility': ef_result['expected_volatility'],
                    'sharpe_ratio': ef_result['sharpe_ratio'],
                    'status': 'completed',
                }
                if mc_result:
                    update_kwargs['mc_return_p5'] = mc_result['final_return_p5']
                    update_kwargs['mc_return_mean'] = mc_result['final_return_mean']
                    update_kwargs['mc_return_p95'] = mc_result['final_return_p95']

                db_interactions.update_portfolio_run(**update_kwargs)
            except Exception as db_err:
                st.warning(f"Could not save to database: {db_err}")

            progress.progress(100, text="Done!")

            # Store results in session state
            st.session_state['last_ef_result'] = ef_result
            st.session_state['last_mc_result'] = mc_result
            st.session_state['last_run_id'] = run_id
            st.session_state['last_profile'] = profile

            # Clear caches so Dashboard picks up new run
            gui_data.get_portfolio_runs.clear()
            gui_data.get_portfolio_holdings.clear()

            st.success(
                f"Portfolio optimized! Sharpe: {ef_result['sharpe_ratio']:.3f}, "
                f"Return: {ef_result['expected_return']:.2%}, "
                f"Volatility: {ef_result['expected_volatility']:.2%}"
            )

        except Exception as e:
            st.error(f"Portfolio optimization failed: {e}")
            import traceback
            st.code(traceback.format_exc())
            st.stop()

# ─── Display Results (if available in session state) ──────────────────
if 'last_ef_result' in st.session_state:
    ef_result = st.session_state['last_ef_result']
    mc_result = st.session_state.get('last_mc_result')
    profile = st.session_state.get('last_profile')

    st.markdown("---")
    st.subheader("Optimization Results")

    # KPI row
    r_cols = st.columns(4)
    with r_cols[0]:
        st.metric("Expected Return", f"{ef_result['expected_return']:.2%}")
    with r_cols[1]:
        st.metric("Volatility", f"{ef_result['expected_volatility']:.2%}")
    with r_cols[2]:
        st.metric("Sharpe Ratio", f"{ef_result['sharpe_ratio']:.3f}")
    with r_cols[3]:
        st.metric("Holdings", len(ef_result['holdings_df']))

    # Charts
    chart_col, data_col = st.columns([1, 1])

    with chart_col:
        # Interactive weight bar chart
        hdf = ef_result['holdings_df'].copy().sort_values('weight', ascending=True)
        fig_bar = px.bar(
            hdf, x='weight', y='ticker', orientation='h',
            color='weight',
            color_continuous_scale='Viridis',
            labels={'weight': 'Weight', 'ticker': 'Ticker'},
            title='Portfolio Weights'
        )
        fig_bar.update_layout(
            height=max(300, len(hdf) * 25 + 60),
            margin=dict(l=0, r=0, t=40, b=0),
            coloraxis_showscale=False,
            yaxis={'categoryorder': 'total ascending'},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        fig_bar.update_traces(
            hovertemplate='<b>%{y}</b><br>Weight: %{x:.2%}<extra></extra>'
        )
        st.plotly_chart(fig_bar, width="stretch")

    with data_col:
        # Efficient frontier image
        risk_str = ef_result.get('risk_level', 'medium')
        ef_path = gui_data.get_graph_path(f"Efficient_frontier_{risk_str}.png")
        if ef_path:
            st.image(ef_path, caption="Efficient Frontier", width="stretch")

    # Holdings table
    st.subheader("Portfolio Holdings")
    display_df = ef_result['holdings_df'].copy()

    col_config = {
        'weight': st.column_config.ProgressColumn(
            "Weight", format="%.2f%%", min_value=0, max_value=display_df['weight'].max() * 100
        ),
    }

    # Format percentages for display
    for col in ['expected_return', 'volatility']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:.2%}" if pd.notna(x) else ""
            )
    if 'sharpe_ratio' in display_df.columns:
        display_df['sharpe_ratio'] = display_df['sharpe_ratio'].apply(
            lambda x: f"{x:.3f}" if pd.notna(x) else ""
        )

    st.dataframe(display_df, width="stretch", hide_index=True)

    # Monte Carlo results
    if mc_result:
        st.subheader("Monte Carlo Portfolio Simulation")

        mc_cols = st.columns([2, 1])

        with mc_cols[0]:
            # Plot yearly stats as a fan chart
            ys = mc_result['yearly_stats']
            fig_mc = go.Figure()

            fig_mc.add_trace(go.Scatter(
                x=ys.index, y=ys['95th Percentile'],
                mode='lines', line=dict(width=0), showlegend=False, name='95th'
            ))
            fig_mc.add_trace(go.Scatter(
                x=ys.index, y=ys['5th Percentile'],
                mode='lines', line=dict(width=0), fill='tonexty',
                fillcolor='rgba(68, 114, 196, 0.15)',
                showlegend=True, name='5th–95th Percentile'
            ))
            fig_mc.add_trace(go.Scatter(
                x=ys.index, y=ys['84th Percentile'],
                mode='lines', line=dict(width=0), showlegend=False, name='84th'
            ))
            fig_mc.add_trace(go.Scatter(
                x=ys.index, y=ys['16th Percentile'],
                mode='lines', line=dict(width=0), fill='tonexty',
                fillcolor='rgba(68, 114, 196, 0.3)',
                showlegend=True, name='16th–84th Percentile'
            ))
            fig_mc.add_trace(go.Scatter(
                x=ys.index, y=ys['Mean'],
                mode='lines+markers', line=dict(color='#4472C4', width=2.5),
                name='Mean'
            ))
            fig_mc.add_hline(
                y=100000, line_dash="dash", line_color="gray",
                annotation_text="Initial Investment"
            )

            fig_mc.update_layout(
                title=f"Portfolio Value — {mc_result['years']}-Year Monte Carlo",
                xaxis_title="Year",
                yaxis_title="Portfolio Value (DKK)",
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                yaxis_tickformat=',',
            )
            st.plotly_chart(fig_mc, width="stretch")

        with mc_cols[1]:
            st.markdown("#### Scenario Analysis")
            st.markdown(f"**{mc_result['years']}-year horizon, 100,000 DKK**")
            st.markdown("")

            for label, key in [
                ("🔴 Pessimistic (5th)", 'final_return_p5'),
                ("🟡 Expected (Mean)", 'final_return_mean'),
                ("🟢 Optimistic (95th)", 'final_return_p95'),
            ]:
                ret = mc_result[key]
                final = 100_000 * (1 + ret)
                st.markdown(f"**{label}**")
                st.markdown(f"Return: {ret:.2%} → **{final:,.0f} DKK**")
                st.markdown("")

    # Link to go to Dashboard
    st.markdown("---")
    st.info("💡 Your portfolio has been saved. Switch to the **Dashboard** page to see it.")
