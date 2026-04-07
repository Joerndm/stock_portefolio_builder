"""
Data Quality Monitor

Surfaces database health metrics so data issues are visible
without running scripts:
- Stale tickers (no recent price data)
- Missing quarterly financials
- NULL ratio coverage
- Extreme/outlier ratio values
- Table coverage overview
"""
import os
import sys

import streamlit as st
import pandas as pd

# Ensure project root is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import gui_data

# ─── Page header ─────────────────────────────────────────────────────
st.title("🩺 Data Quality Monitor")
st.caption("Live health checks against the stock database")

# ─── Load data ───────────────────────────────────────────────────────
with st.spinner("Running data quality checks..."):
    dq = gui_data.get_data_quality_summary()

# ─── Top KPI row ─────────────────────────────────────────────────────
st.markdown("---")
cols = st.columns(5)

with cols[0]:
    st.metric("Total Tickers", dq['total_tickers'])

with cols[1]:
    stale_count = len(dq['stale_tickers'])
    st.metric("Stale Tickers", stale_count,
              delta=f">{7} days old" if stale_count else "All fresh",
              delta_color="inverse")

with cols[2]:
    st.metric("Missing Quarterly", len(dq['missing_quarterly']),
              delta_color="inverse")

with cols[3]:
    null_ratio_count = len(dq['null_ratio_tickers'])
    st.metric("NULL Ratio Tickers", null_ratio_count,
              help=">50% NULL in at least one ratio column",
              delta_color="inverse")

with cols[4]:
    st.metric("Extreme Ratios", dq['extreme_ratio_rows'],
              help="Rows with |ratio| > 10000",
              delta_color="inverse")

# ─── Table Coverage ──────────────────────────────────────────────────
st.markdown("---")
st.subheader("📊 Table Coverage")
st.caption("How many distinct tickers have data in each table")

coverage = dq['table_coverage']
total = dq['total_tickers']

coverage_df = pd.DataFrame([
    {
        'Table': table.replace('stock_', '').replace('_data', '').replace('_', ' ').title(),
        'Tickers': count,
        'Coverage': f"{count / total * 100:.0f}%" if total > 0 else "N/A"
    }
    for table, count in coverage.items()
])
st.dataframe(coverage_df, use_container_width=True, hide_index=True)

# ─── Stale Tickers ───────────────────────────────────────────────────
st.markdown("---")
st.subheader("⏰ Stale Tickers")
st.caption("Tickers with no new price data for >7 days")

stale_df = dq['stale_tickers']
if stale_df.empty:
    st.success("No stale tickers found.")
else:
    stale_df = stale_df.copy()
    stale_df['last_date'] = pd.to_datetime(stale_df['last_date']).dt.date
    st.dataframe(
        stale_df.rename(columns={
            'ticker': 'Ticker',
            'last_date': 'Last Price Date',
            'days_stale': 'Days Stale'
        }),
        use_container_width=True,
        hide_index=True
    )
    st.info(
        f"Run `python stock_orchestrator.py --retry-stale {7}` to re-fetch these tickers."
    )

# ─── NULL Ratio Coverage ─────────────────────────────────────────────
st.markdown("---")
st.subheader("📉 Tickers with High NULL Ratios")
st.caption("Tickers where >50% of a ratio column is NULL")

null_df = dq['null_ratio_tickers']
if null_df.empty:
    st.success("All tickers have good ratio coverage.")
else:
    st.dataframe(
        null_df.rename(columns={
            'ticker': 'Ticker',
            'total_rows': 'Total Rows',
            'pct_null_pe': '% NULL P/E',
            'pct_null_pb': '% NULL P/B',
            'pct_null_ps': '% NULL P/S',
            'pct_null_pfcf': '% NULL P/FCF'
        }),
        use_container_width=True,
        hide_index=True,
        height=400
    )
    st.info(
        f"Run `python batch_refresh_ttm.py --execute` to recalculate TTM data and ratios for these tickers."
    )

# ─── Missing Quarterly ───────────────────────────────────────────────
st.markdown("---")
st.subheader("📋 Missing Quarterly Income Data")
st.caption("Tickers with price data but no quarterly income statement")

missing = dq['missing_quarterly']
if not missing:
    st.success("All tickers have quarterly income data.")
else:
    # Show in columns of 10
    n_cols = 6
    cols = st.columns(n_cols)
    for i, ticker in enumerate(sorted(missing)):
        cols[i % n_cols].code(ticker)
    st.caption(f"Total: {len(missing)} tickers")

# ─── Additional Info ─────────────────────────────────────────────────
st.markdown("---")
st.subheader("ℹ️ Additional Metrics")

info_cols = st.columns(2)
with info_cols[0]:
    st.metric(
        "Tickers with NULL financial_date_used",
        dq['null_financial_date_used_tickers'],
        help="Legacy ratio data without financial date tracking"
    )

with info_cols[1]:
    st.metric(
        "Extreme Ratio Rows",
        dq['extreme_ratio_rows'],
        help="Rows with |ratio| > 10000 (should be 0 after cleanup)"
    )

st.markdown("---")
st.caption("Data refreshes every hour. Click ↻ to force refresh.")
