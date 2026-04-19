"""
Dashboard — Portfolio Overview

Inspired by Nordnet's brokerage dashboard layout:
- Top: Key portfolio metrics (value, return, volatility, Sharpe)
- Center: Portfolio composition chart + Efficient frontier
- Bottom: Holdings table + Monte Carlo outlook
- Sidebar: Latest run info and quick stats
"""
import os
import sys

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Ensure project root is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import gui_data

# ─── Page header ─────────────────────────────────────────────────────
st.title("📊 Portfolio Dashboard")

# ─── Load portfolio runs ─────────────────────────────────────────────
runs_df = gui_data.get_portfolio_runs()

if runs_df.empty:
    st.info(
        "No portfolio runs found in the database. "
        "Go to **Portfolio Builder** to create your first optimized portfolio."
    )
    st.stop()

# ─── Sidebar: run selector ───────────────────────────────────────────
with st.sidebar:
    st.subheader("Portfolio Run")

    completed_runs = runs_df[runs_df['status'] == 'completed']
    if completed_runs.empty:
        st.warning("No completed portfolio runs.")
        st.stop()

    # Build display labels
    run_labels = []
    for _, row in completed_runs.iterrows():
        date_str = pd.to_datetime(row['run_date']).strftime('%Y-%m-%d %H:%M')
        risk = row.get('risk_level', '?').upper()
        sharpe = row.get('sharpe_ratio')
        sharpe_str = f"S={sharpe:.2f}" if pd.notna(sharpe) else ""
        label = f"#{row['run_id']}  {date_str}  {risk}  {sharpe_str}"
        run_labels.append(label)

    selected_label = st.selectbox("Select run", run_labels, index=0)
    selected_run_id = int(selected_label.split("#")[1].split()[0])
    run = completed_runs[completed_runs['run_id'] == selected_run_id].iloc[0]

    st.divider()
    st.caption(f"Risk: **{run.get('risk_level', 'N/A').upper()}**")
    st.caption(f"Horizon: **{run.get('investment_years', 'N/A')} years**")
    st.caption(f"Target size: **{run.get('portfolio_size', 'N/A')} stocks**")
    if pd.notna(run.get('execution_time_seconds')):
        st.caption(f"Run time: **{run['execution_time_seconds']:.0f}s**")

# ─── Load holdings for this run ──────────────────────────────────────
holdings_df = gui_data.get_portfolio_holdings(selected_run_id)

# ─── Top KPI row (Nordnet-style metric cards) ────────────────────────
st.markdown("---")

kpi_cols = st.columns(5)

with kpi_cols[0]:
    exp_ret = run.get('expected_return')
    st.metric(
        "Expected Return",
        f"{exp_ret:.2%}" if pd.notna(exp_ret) else "N/A",
    )

with kpi_cols[1]:
    exp_vol = run.get('expected_volatility')
    st.metric(
        "Volatility",
        f"{exp_vol:.2%}" if pd.notna(exp_vol) else "N/A",
    )

with kpi_cols[2]:
    sharpe = run.get('sharpe_ratio')
    st.metric(
        "Sharpe Ratio",
        f"{sharpe:.3f}" if pd.notna(sharpe) else "N/A",
        help="Sharpe ratio as calculated during portfolio optimization. "
             "Country-specific risk-free rate can be set in Portfolio Builder."
    )

with kpi_cols[3]:
    mc_mean = run.get('mc_return_mean')
    mc_p5 = run.get('mc_return_p5')
    delta_str = f"5th pct: {mc_p5:.1%}" if pd.notna(mc_p5) else None
    st.metric(
        "MC Mean Return",
        f"{mc_mean:.2%}" if pd.notna(mc_mean) else "N/A",
        delta=delta_str,
        delta_color="normal"
    )

with kpi_cols[4]:
    mc_p95 = run.get('mc_return_p95')
    st.metric(
        "MC Upside (95th)",
        f"{mc_p95:.2%}" if pd.notna(mc_p95) else "N/A",
    )

# ─── Portfolio Composition & Efficient Frontier ──────────────────────
st.markdown("---")

if not holdings_df.empty:
    chart_col, frontier_col = st.columns([1, 1])

    # ── Donut chart: Holdings by weight ──
    with chart_col:
        st.subheader("Portfolio Composition")
        fig_pie = px.pie(
            holdings_df,
            values='weight',
            names='ticker',
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_pie.update_traces(
            textposition='inside',
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>Weight: %{value:.2%}<extra></extra>'
        )
        fig_pie.update_layout(
            showlegend=False,
            margin=dict(t=20, b=20, l=20, r=20),
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_pie, width="stretch")

    # ── Efficient Frontier graph ──
    with frontier_col:
        st.subheader("Efficient Frontier")
        risk_level = run.get('risk_level', 'medium')
        ef_path = gui_data.get_graph_path(f"Efficient_frontier_{risk_level}.png")
        if ef_path:
            st.image(ef_path, width="stretch")
        else:
            # Fallback: try any EF graph
            for rl in ['medium', 'low', 'high']:
                p = gui_data.get_graph_path(f"Efficient_frontier_{rl}.png")
                if p:
                    st.image(p, width="stretch")
                    break
            else:
                st.info("No efficient frontier graph available for this run.")

    # ── Holdings table (Nordnet "Beholdning" style) ──
    st.subheader("Holdings")

    display_df = holdings_df.copy()

    # Format columns for display
    format_map = {
        'weight': lambda x: f"{x:.2%}" if pd.notna(x) else "",
        'expected_return': lambda x: f"{x:.2%}" if pd.notna(x) else "",
        'volatility': lambda x: f"{x:.2%}" if pd.notna(x) else "",
        'sharpe_ratio': lambda x: f"{x:.3f}" if pd.notna(x) else "",
    }

    display_cols = ['rank', 'ticker', 'weight', 'expected_return', 'volatility', 'sharpe_ratio']
    if 'industry' in display_df.columns:
        display_cols.append('industry')

    available_cols = [c for c in display_cols if c in display_df.columns]
    display_df = display_df[available_cols]

    rename_map = {
        'rank': 'Rank',
        'ticker': 'Ticker',
        'weight': 'Weight',
        'expected_return': 'Exp. Return',
        'volatility': 'Volatility',
        'sharpe_ratio': 'Sharpe',
        'industry': 'Industry',
    }
    display_df = display_df.rename(columns=rename_map)

    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
        height=min(400, 35 * len(display_df) + 38),
    )
else:
    st.warning("No holdings data available for this portfolio run.")

# ─── Monte Carlo Outlook ──────────────────────────────────────────────
st.markdown("---")
st.subheader("Monte Carlo Portfolio Outlook")

mc_col1, mc_col2 = st.columns([1, 1])

with mc_col1:
    years = run.get('investment_years', 7)
    mc_graph_path = gui_data.get_graph_path(f"Portfolio_Monte_Carlo_{years}yr.png")
    if mc_graph_path:
        st.image(mc_graph_path, width="stretch")
    else:
        st.info("No portfolio Monte Carlo graph available.")

with mc_col2:
    st.markdown("#### Scenario Analysis")

    initial = 100_000
    scenarios = {
        "Pessimistic (5th pct)": run.get('mc_return_p5'),
        "Expected (Mean)": run.get('mc_return_mean'),
        "Optimistic (95th pct)": run.get('mc_return_p95'),
    }

    for label, ret in scenarios.items():
        if pd.notna(ret):
            final_value = initial * (1 + ret)
            profit = final_value - initial
            color = "green" if profit >= 0 else "red"
            st.markdown(
                f"**{label}**  \n"
                f"Return: `{ret:.2%}` → Final value: "
                f"<span style='color:{color}'>{final_value:,.0f} DKK</span>  \n"
                f"Profit: <span style='color:{color}'>{profit:+,.0f} DKK</span>",
                unsafe_allow_html=True
            )
            st.markdown("")

    st.caption(f"Based on {years}-year horizon with 100,000 DKK initial investment")

# ─── Pipeline Status Summary ──────────────────────────────────────────
st.markdown("---")

with st.expander("📋 Pipeline Status", expanded=False):
    status_cols = st.columns(3)

    all_tickers = gui_data.get_all_tickers()
    pred_tickers = gui_data.get_tickers_with_predictions(max_age_days=30)

    with status_cols[0]:
        st.metric("Total Tickers", len(all_tickers))

    with status_cols[1]:
        st.metric("With Predictions (30d)", len(pred_tickers))

    with status_cols[2]:
        analyzed = run.get('total_stocks_analyzed')
        st.metric("Stocks in Last Run", analyzed if pd.notna(analyzed) else "N/A")

    model_status = gui_data.get_model_status(max_age_days=30)
    if not model_status.empty:
        fresh = model_status[model_status['is_fresh'] == True]
        st.caption(
            f"Fresh models (≤30 days): {len(fresh)} across "
            f"{fresh['ticker'].nunique() if not fresh.empty else 0} tickers"
        )
