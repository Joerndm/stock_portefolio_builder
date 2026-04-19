"""
Stock Explorer — Nordnet-Inspired Individual Stock Deep Dive

Layout inspired by Nordnet's stock detail page with two tabs:
  • Oversigt  (Overview)  — chart with overlay toggles, period returns,
    key ratios grid, ML price targets, Monte Carlo simulation, technicals
  • Om virksomheden (About the Company) — financials, revenue / earnings
    history, generated graphs
"""
import os
import sys
import datetime

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import gui_data

# ─── Helper ──────────────────────────────────────────────────────────

def _fmt(value, fmt="{:.2f}", fallback="–"):
    """Safe format a value."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return fallback
    try:
        return fmt.format(value)
    except (ValueError, TypeError):
        return str(value)


def _pct_color(val):
    """Return green / red span for a percentage."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "–"
    color = "#00CC66" if val >= 0 else "#FF4444"
    return f"<span style='color:{color}'>{val:+.2%}</span>"


# ─── Load shared data ────────────────────────────────────────────────

all_tickers = gui_data.get_all_tickers()
pred_tickers = gui_data.get_tickers_with_predictions(max_age_days=60)
stock_info_df = gui_data.get_stock_info()

# Build ticker → company-name lookup
ticker_names = {}
if not stock_info_df.empty:
    for _, row in stock_info_df.iterrows():
        t = row.get('ticker', '')
        name = row.get('company_Name', '')
        ticker_names[t] = f"{t}  —  {name}" if name else t

# ─── Sidebar: stock selector ─────────────────────────────────────────

with st.sidebar:
    st.subheader("Select Stock")

    show_all = st.checkbox("Show all tickers", value=False,
                           help="If unchecked, only tickers with recent predictions are shown")
    ticker_pool = all_tickers if show_all else (pred_tickers if pred_tickers else all_tickers)

    display_labels = [ticker_names.get(t, t) for t in sorted(ticker_pool)]
    label_to_ticker = {ticker_names.get(t, t): t for t in sorted(ticker_pool)}

    if not display_labels:
        st.warning("No tickers available.")
        st.stop()

    selected_label = st.selectbox("Ticker", display_labels, index=0)
    ticker = label_to_ticker[selected_label]

    st.divider()

    # Stock info card
    info_row = stock_info_df[stock_info_df['ticker'] == ticker]
    company_name = ticker
    industry = "N/A"
    if not info_row.empty:
        info = info_row.iloc[0]
        company_name = info.get('company_Name', ticker) or ticker
        industry = info.get('industry', 'N/A') or 'N/A'
        st.caption(f"**{company_name}**")
        st.caption(f"Industry: {industry}")

    has_prediction = ticker in pred_tickers
    if has_prediction:
        st.success("Has ML predictions")
    else:
        st.warning("No recent predictions")

# ─── Stock Header (Nordnet-style) ────────────────────────────────────

price_df = gui_data.get_stock_prices(ticker, amount=2000)

if not price_df.empty:
    price_df['date'] = pd.to_datetime(price_df['date'])
    price_df = price_df.sort_values('date')

    latest = price_df.iloc[-1]
    prev = price_df.iloc[-2] if len(price_df) > 1 else latest

    cur_price = latest.get('close_Price', 0)
    prev_close = prev.get('close_Price', cur_price)
    day_change_abs = cur_price - prev_close
    day_change_pct = day_change_abs / prev_close if prev_close else 0

    chg_color = "#00CC66" if day_change_abs >= 0 else "#FF4444"

    st.markdown(f"## {company_name}")
    st.markdown(
        f"<h2 style='margin-top:-0.5em'>{cur_price:.2f}"
        f"  <small style='color:{chg_color}'>"
        f"{day_change_pct:+.2%} ({day_change_abs:+.2f})</small></h2>",
        unsafe_allow_html=True
    )

    head_cols = st.columns(4)
    with head_cols[0]:
        high = latest.get('high_Price')
        st.caption(f"High: **{_fmt(high)}**")
    with head_cols[1]:
        low = latest.get('low_Price')
        st.caption(f"Low: **{_fmt(low)}**")
    with head_cols[2]:
        vol = latest.get('trade_Volume')
        st.caption(f"Volume: **{_fmt(vol, '{:,.0f}')}**")
    with head_cols[3]:
        date_str = latest['date'].strftime('%Y-%m-%d') if pd.notna(latest['date']) else ''
        st.caption(f"Date: **{date_str}**")

else:
    st.markdown(f"## {company_name}")
    st.warning(f"No price data found for {ticker}")
    st.stop()

# ─── Nordnet-style two-tab layout ────────────────────────────────────

tab_overview, tab_company = st.tabs(["📊 Oversigt", "🏢 Om virksomheden"])

# ══════════════════════════════════════════════════════════════════════
#  TAB 1 — OVERSIGT (Overview)
# ══════════════════════════════════════════════════════════════════════

with tab_overview:

    # ── Chart overlay toggles (Nordnet style toolbar) ─────────────
    st.markdown("##### Chart Overlays")
    ovl_cols = st.columns(6)
    with ovl_cols[0]:
        show_sma20 = st.checkbox("SMA 20", value=True)
    with ovl_cols[1]:
        show_sma200 = st.checkbox("SMA 200", value=True)
    with ovl_cols[2]:
        show_ema5 = st.checkbox("EMA 5", value=False)
    with ovl_cols[3]:
        show_ema20 = st.checkbox("EMA 20", value=False)
    with ovl_cols[4]:
        show_bollinger = st.checkbox("Bollinger", value=False)
    with ovl_cols[5]:
        show_rsi = st.checkbox("RSI", value=True)

    # ── Period selector with return badges ────────────────────────
    periods = {
        "1M": 21, "3M": 63, "6M": 126,
        "YTD": None,  # special handling
        "1Y": 252, "2Y": 504, "5Y": 1260, "Max": len(price_df),
    }

    # Compute period returns for badges
    period_returns_md = []
    for pname, pdays in periods.items():
        if pname == "YTD":
            ytd_mask = price_df['date'] >= pd.Timestamp(datetime.date(datetime.date.today().year, 1, 1))
            p_slice = price_df[ytd_mask]
        else:
            p_slice = price_df.tail(pdays)
        if len(p_slice) >= 2:
            p_first = p_slice['close_Price'].iloc[0]
            p_last = p_slice['close_Price'].iloc[-1]
            ret = (p_last - p_first) / p_first if p_first else 0
            period_returns_md.append(f"**{pname}** {_pct_color(ret)}")
        else:
            period_returns_md.append(f"**{pname}** –")

    st.markdown(" &nbsp;|&nbsp; ".join(period_returns_md), unsafe_allow_html=True)

    # ── Period tabs → chart ───────────────────────────────────────
    period_tabs = st.tabs(list(periods.keys()))

    for tab, (period_name, days) in zip(period_tabs, periods.items()):
        with tab:
            if period_name == "YTD":
                plot_df = price_df[price_df['date'] >= pd.Timestamp(
                    datetime.date(datetime.date.today().year, 1, 1)
                )].copy()
            else:
                plot_df = price_df.tail(days).copy()

            if plot_df.empty or 'close_Price' not in plot_df.columns:
                st.info("Not enough data for this period.")
                continue

            # Determine if we need a RSI row
            row_heights = [0.75, 0.25] if show_rsi else [1.0]
            n_rows = 2 if show_rsi else 1
            fig = make_subplots(
                rows=n_rows, cols=1, shared_xaxes=True,
                row_heights=row_heights, vertical_spacing=0.03,
            )

            # -- Candlestick main chart --
            if all(c in plot_df.columns for c in ['open_Price', 'high_Price', 'low_Price', 'close_Price']):
                fig.add_trace(go.Candlestick(
                    x=plot_df['date'],
                    open=plot_df['open_Price'],
                    high=plot_df['high_Price'],
                    low=plot_df['low_Price'],
                    close=plot_df['close_Price'],
                    name='OHLC',
                    increasing_line_color='#00CC66',
                    decreasing_line_color='#FF4444',
                ), row=1, col=1)
            else:
                fig.add_trace(go.Scatter(
                    x=plot_df['date'], y=plot_df['close_Price'],
                    mode='lines', name='Close',
                    line=dict(color='#4472C4', width=1.5)
                ), row=1, col=1)

            # -- Overlays --
            overlay_map = [
                (show_sma20, 'sma_20', 'SMA 20', '#FFA500', 'dot'),
                (show_sma200, 'sma_200', 'SMA 200', '#4472C4', 'dot'),
                (show_ema5, 'ema_5', 'EMA 5', '#E91E63', 'dash'),
                (show_ema20, 'ema_20', 'EMA 20', '#9C27B0', 'dash'),
            ]
            for enabled, col_name, label, color, dash in overlay_map:
                if enabled and col_name in plot_df.columns:
                    fig.add_trace(go.Scatter(
                        x=plot_df['date'], y=plot_df[col_name],
                        mode='lines', name=label,
                        line=dict(color=color, width=1, dash=dash),
                        opacity=0.8,
                    ), row=1, col=1)

            # -- Bollinger Bands (computed from SMA + bandwidth/2) --
            if show_bollinger and 'sma_20' in plot_df.columns and 'bollinger_Band_20_2STD' in plot_df.columns:
                bb_upper = plot_df['sma_20'] + plot_df['bollinger_Band_20_2STD'] / 2
                bb_lower = plot_df['sma_20'] - plot_df['bollinger_Band_20_2STD'] / 2
                fig.add_trace(go.Scatter(
                    x=plot_df['date'], y=bb_upper,
                    mode='lines', name='BB Upper',
                    line=dict(color='rgba(156,39,176,0.4)', width=1),
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=plot_df['date'], y=bb_lower,
                    mode='lines', name='BB Lower',
                    line=dict(color='rgba(156,39,176,0.4)', width=1),
                    fill='tonexty', fillcolor='rgba(156,39,176,0.05)',
                ), row=1, col=1)

            # -- RSI sub-chart --
            if show_rsi and 'rsi_14' in plot_df.columns:
                fig.add_trace(go.Scatter(
                    x=plot_df['date'], y=plot_df['rsi_14'],
                    mode='lines', name='RSI (14)',
                    line=dict(color='#E040FB', width=1.2),
                ), row=2, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="rgba(255,68,68,0.4)",
                              row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="rgba(0,204,102,0.4)",
                              row=2, col=1)
                fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)

            fig.update_layout(
                height=550 if show_rsi else 450,
                margin=dict(l=0, r=0, t=20, b=0),
                xaxis_rangeslider_visible=False,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(gridcolor='rgba(128,128,128,0.1)'),
                yaxis=dict(gridcolor='rgba(128,128,128,0.1)'),
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
            )
            st.plotly_chart(fig, width="stretch")

    # ── Volume (collapsed) ────────────────────────────────────────
    if 'trade_Volume' in price_df.columns:
        with st.expander("📊 Volume"):
            vol_df = price_df.tail(252).copy()
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Bar(
                x=vol_df['date'], y=vol_df['trade_Volume'],
                marker_color='rgba(68, 114, 196, 0.5)', name='Volume',
            ))
            fig_vol.update_layout(
                height=200, margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                yaxis_tickformat=',',
            )
            st.plotly_chart(fig_vol, width="stretch")

    # ── Nøgletal (Key Ratios) — Nordnet-style grid ────────────────
    st.markdown("---")
    st.subheader("Nøgletal")

    ratio_df = gui_data.get_stock_ratios(ticker, amount=1)
    financial_df = gui_data.get_financial_data(ticker, amount=1)

    # Try to get quarterly TTM data for EPS / Revenue / EBIT
    quarterly_income_df = gui_data.get_quarterly_income(ticker)
    quarterly_bs_df = gui_data.get_quarterly_balancesheet(ticker)

    # Gather latest ratio values
    pe_val = ratio_df['p_e'].iloc[-1] if (not ratio_df.empty and 'p_e' in ratio_df.columns) else None
    ps_val = ratio_df['p_s'].iloc[-1] if (not ratio_df.empty and 'p_s' in ratio_df.columns) else None
    pb_val = ratio_df['p_b'].iloc[-1] if (not ratio_df.empty and 'p_b' in ratio_df.columns) else None
    pfcf_val = ratio_df['p_fcf'].iloc[-1] if (not ratio_df.empty and 'p_fcf' in ratio_df.columns) else None

    # Gather financial values (TTM preferred, fall back to annual)
    eps_val = None
    revenue_val = None
    ebit_val = None
    book_val_ps = None

    if not quarterly_income_df.empty:
        qi = quarterly_income_df.iloc[-1]
        eps_val = qi.get('eps_diluted_ttm') or qi.get('eps_basic_ttm')
        revenue_val = qi.get('revenue_ttm')
        ebit_val = qi.get('operating_income_ttm')

    if not quarterly_bs_df.empty:
        qb = quarterly_bs_df.iloc[-1]
        book_val_ps = qb.get('book_value_per_share')

    # Fall back to annual if TTM not available
    if not financial_df.empty:
        fin = financial_df.iloc[-1]
        if eps_val is None:
            eps_val = fin.get('eps')
        if revenue_val is None:
            revenue_val = fin.get('revenue')
        if ebit_val is None:
            ebit_val = fin.get('operating_Earning')
        if book_val_ps is None:
            book_val_ps = fin.get('book_Value_Per_Share')

    # Display as a 4-column grid (Nordnet nøgletal style)
    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    with r1c1:
        st.metric("P/E", _fmt(pe_val))
    with r1c2:
        st.metric("P/S", _fmt(ps_val))
    with r1c3:
        st.metric("P/B", _fmt(pb_val))
    with r1c4:
        st.metric("EPS", _fmt(eps_val))

    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    with r2c1:
        st.metric("P/FCF", _fmt(pfcf_val))
    with r2c2:
        if revenue_val and revenue_val > 1e9:
            st.metric("Revenue", f"{revenue_val / 1e9:.1f} B")
        elif revenue_val and revenue_val > 1e6:
            st.metric("Revenue", f"{revenue_val / 1e6:.0f} M")
        else:
            st.metric("Revenue", _fmt(revenue_val, '{:,.0f}'))
    with r2c3:
        if ebit_val and abs(ebit_val) > 1e9:
            st.metric("EBIT", f"{ebit_val / 1e9:.1f} B")
        elif ebit_val and abs(ebit_val) > 1e6:
            st.metric("EBIT", f"{ebit_val / 1e6:.0f} M")
        else:
            st.metric("EBIT", _fmt(ebit_val, '{:,.0f}'))
    with r2c4:
        st.metric("Book/Share", _fmt(book_val_ps))

    # ── Beta (Stock vs Market Index) ──────────────────────────────
    st.markdown("---")
    st.subheader("Beta  (vs Market Index)")

    beta_indices = gui_data.get_available_beta_indices(ticker)

    if beta_indices:
        # Let the user pick which index to compare against
        beta_col1, beta_col2 = st.columns([1, 3])
        with beta_col1:
            selected_beta_index = st.selectbox(
                "Benchmark Index",
                options=beta_indices,
                index=0,
                help="Select a market index to see this stock's beta against it",
                key="beta_index_selector",
            )

        beta_df = gui_data.get_stock_beta(ticker, index_code=selected_beta_index)

        if not beta_df.empty:
            b = beta_df.iloc[0]
            beta_date = b.get('date', '')
            if hasattr(beta_date, 'strftime'):
                beta_date = beta_date.strftime('%Y-%m-%d')

            with beta_col2:
                st.caption(f"As of **{beta_date}**  •  Benchmark: **{selected_beta_index}**")

            bc1, bc2, bc3, bc4, bc5 = st.columns(5)
            with bc1:
                st.metric("Beta (60d)", _fmt(b.get('beta_60d'), '{:.2f}'))
            with bc2:
                st.metric("Beta (120d)", _fmt(b.get('beta_120d'), '{:.2f}'))
            with bc3:
                st.metric("Beta (1Y)", _fmt(b.get('beta_252d'), '{:.2f}'))
            with bc4:
                st.metric("Correlation", _fmt(b.get('correlation_252d'), '{:.2f}'))
            with bc5:
                st.metric("R²", _fmt(b.get('r_squared_252d'), '{:.2f}'))

            # Interpretive note
            beta_val = b.get('beta_252d')
            if beta_val is not None and not (isinstance(beta_val, float) and np.isnan(beta_val)):
                if beta_val > 1.5:
                    st.caption("⚡ **High beta** — significantly more volatile than the market")
                elif beta_val > 1.0:
                    st.caption("📈 **Above-market beta** — amplifies market movements")
                elif beta_val > 0.5:
                    st.caption("🛡️ **Defensive** — less volatile than the market")
                elif beta_val > 0:
                    st.caption("🏠 **Very defensive** — largely independent of market swings")
                else:
                    st.caption("🔄 **Negative beta** — tends to move opposite to the market")

        # Show all-index comparison table (latest beta across all indices)
        all_beta_df = gui_data.get_stock_beta(ticker)
        if not all_beta_df.empty and len(all_beta_df) > 1:
            with st.expander("Compare across all indices", expanded=False):
                display_beta = all_beta_df[['index_code', 'beta_60d', 'beta_120d',
                                             'beta_252d', 'correlation_252d']].copy()
                display_beta = display_beta.rename(columns={
                    'index_code': 'Index',
                    'beta_60d': 'Beta (60d)',
                    'beta_120d': 'Beta (120d)',
                    'beta_252d': 'Beta (1Y)',
                    'correlation_252d': 'Correlation',
                })
                for col in ['Beta (60d)', 'Beta (120d)', 'Beta (1Y)', 'Correlation']:
                    display_beta[col] = display_beta[col].apply(
                        lambda x: f"{x:.2f}" if pd.notna(x) else "–"
                    )
                st.dataframe(display_beta, width="stretch", hide_index=True)
    else:
        st.info(
            f"No beta data for {ticker}. Beta will be calculated when the data pipeline "
            f"runs next, or you can trigger it manually."
        )

    # ── Kursmål (ML Price Targets) — Nordnet-style forecast chart ─
    st.markdown("---")
    st.subheader("Kursmål  (ML Price Targets)")

    if has_prediction:
        summary = gui_data.get_prediction_summary(ticker, max_age_days=60)

        if summary and summary.get('horizons'):
            pred_date = summary.get('prediction_date', '')
            current_price = summary.get('current_price', 0)

            horizons = summary['horizons']

            # Nordnet-style: show range text (e.g. "-3.34% til +28.88%")
            returns = [h.get('predicted_return', 0) for h in horizons if h.get('predicted_return') is not None]
            if returns:
                min_ret, max_ret = min(returns), max(returns)
                st.markdown(
                    f"**{min_ret:+.2%}** til **{max_ret:+.2%}**  ·  "
                    f"Prediction date: {pred_date}",
                    unsafe_allow_html=True
                )

            # Horizons summary table
            horizon_data = []
            for h in horizons:
                horizon_data.append({
                    'Horizon': f"{h.get('prediction_horizon_days', '?')}d",
                    'Predicted Price': _fmt(h.get('predicted_price'), '{:.2f}'),
                    'Return': _fmt(h.get('predicted_return'), '{:+.2%}'),
                    'Low (5%)': _fmt(h.get('confidence_lower_5'), '{:.2f}'),
                    'High (95%)': _fmt(h.get('confidence_upper_95'), '{:.2f}'),
                    'Model': h.get('model_type', ''),
                })
            st.dataframe(pd.DataFrame(horizon_data), width="stretch", hide_index=True)

            # Nordnet-inspired forecast chart: historical + predicted price curve
            if 'close_Price' in price_df.columns:
                fig_tgt = go.Figure()

                hist_slice = price_df.tail(126).copy()
                fig_tgt.add_trace(go.Scatter(
                    x=hist_slice['date'], y=hist_slice['close_Price'],
                    mode='lines', name='Historical',
                    line=dict(color='#4472C4', width=1.5),
                ))

                last_date = hist_slice['date'].iloc[-1]

                # Build forecast points sorted by horizon
                sorted_horizons = sorted(horizons, key=lambda x: x.get('prediction_horizon_days', 0))
                fc_days, fc_prices, fc_lo5, fc_hi95, fc_lo16, fc_hi84 = [], [], [], [], [], []
                fc_dates = []

                for h in sorted_horizons:
                    d = h.get('prediction_horizon_days', 0)
                    pp = h.get('predicted_price')
                    if pp is None:
                        continue
                    tgt_date = last_date + pd.Timedelta(days=int(d * 365 / 252))
                    fc_days.append(d)
                    fc_dates.append(tgt_date)
                    fc_prices.append(pp)
                    fc_lo5.append(h.get('confidence_lower_5') or pp * 0.95)
                    fc_hi95.append(h.get('confidence_upper_95') or pp * 1.05)
                    fc_lo16.append(h.get('confidence_lower_16') or pp * 0.97)
                    fc_hi84.append(h.get('confidence_upper_84') or pp * 1.03)

                if fc_dates:
                    # Anchor at current price
                    all_dates = [last_date] + fc_dates
                    all_prices = [current_price] + fc_prices
                    all_lo5 = [current_price] + fc_lo5
                    all_hi95 = [current_price] + fc_hi95
                    all_lo16 = [current_price] + fc_lo16
                    all_hi84 = [current_price] + fc_hi84

                    # Interpolate to daily points for a smooth curve
                    day_offsets = [0] + [int((d - last_date).days) for d in fc_dates]
                    max_day = max(day_offsets)
                    interp_days = list(range(0, max_day + 1))
                    interp_dates = [last_date + pd.Timedelta(days=d) for d in interp_days]

                    interp_prices = np.interp(interp_days, day_offsets, all_prices)
                    interp_lo5 = np.interp(interp_days, day_offsets, all_lo5)
                    interp_hi95 = np.interp(interp_days, day_offsets, all_hi95)
                    interp_lo16 = np.interp(interp_days, day_offsets, all_lo16)
                    interp_hi84 = np.interp(interp_days, day_offsets, all_hi84)

                    # 5th–95th confidence band (outer, lighter)
                    fig_tgt.add_trace(go.Scatter(
                        x=interp_dates, y=interp_hi95,
                        mode='lines', line=dict(width=0), showlegend=False,
                    ))
                    fig_tgt.add_trace(go.Scatter(
                        x=interp_dates, y=interp_lo5,
                        mode='lines', line=dict(width=0),
                        fill='tonexty', fillcolor='rgba(0,204,102,0.08)',
                        name='5th–95th Confidence',
                    ))

                    # 16th–84th confidence band (inner, darker)
                    fig_tgt.add_trace(go.Scatter(
                        x=interp_dates, y=interp_hi84,
                        mode='lines', line=dict(width=0), showlegend=False,
                    ))
                    fig_tgt.add_trace(go.Scatter(
                        x=interp_dates, y=interp_lo16,
                        mode='lines', line=dict(width=0),
                        fill='tonexty', fillcolor='rgba(0,204,102,0.18)',
                        name='16th–84th Confidence',
                    ))

                    # Predicted price curve (smooth line)
                    fig_tgt.add_trace(go.Scatter(
                        x=interp_dates, y=interp_prices,
                        mode='lines', name='Predicted Price',
                        line=dict(color='#00CC66', width=2, dash='dot'),
                    ))

                    # Diamond markers at actual horizon points
                    for i, h in enumerate(sorted_horizons):
                        pp = fc_prices[i] if i < len(fc_prices) else None
                        if pp is None:
                            continue
                        d = h.get('prediction_horizon_days', 0)
                        fig_tgt.add_trace(go.Scatter(
                            x=[fc_dates[i]], y=[pp],
                            mode='markers+text', name=f'{d}d target',
                            marker=dict(color='#00CC66', size=10, symbol='diamond'),
                            text=[f"{pp:.1f}"],
                            textposition='top center',
                            textfont=dict(size=10),
                            showlegend=False,
                        ))

                # Current-price reference line
                fig_tgt.add_hline(y=current_price, line_dash="dash",
                                  line_color="gray",
                                  annotation_text=f"Current: {current_price:.2f}")

                fig_tgt.update_layout(
                    height=380,
                    margin=dict(l=0, r=0, t=20, b=0),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(gridcolor='rgba(128,128,128,0.1)'),
                    yaxis=dict(gridcolor='rgba(128,128,128,0.1)'),
                    legend=dict(orientation='h', y=-0.15),
                )
                st.plotly_chart(fig_tgt, width="stretch")

            # Generated prediction graph (expandable)
            pred_graph = gui_data.get_graph_path(f"future_stock_prediction_of_{ticker}.png")
            if pred_graph:
                with st.expander("ML Prediction Graph (generated)", expanded=False):
                    st.image(pred_graph, width="stretch")
        else:
            st.info("No prediction details available.")
    else:
        st.info(f"No recent predictions for {ticker}. Run the ML pipeline first.")

    # ── Monte Carlo Simulation ────────────────────────────────────
    st.markdown("---")
    st.subheader("Monte Carlo Simulation")

    mc_df = gui_data.get_monte_carlo_results(ticker=ticker)

    if not mc_df.empty:
        latest_date = mc_df['simulation_date'].max()
        mc_latest = mc_df[mc_df['simulation_date'] == latest_date].sort_values('simulation_year')

        num_sims = mc_latest.iloc[0].get('num_simulations', 'N/A')
        st.caption(f"Simulation date: **{latest_date}** | Simulations: **{num_sims}**")

        mc_chart_col, mc_table_col = st.columns([2, 1])

        with mc_chart_col:
            fig_mc = go.Figure()

            if 'percentile_95' in mc_latest.columns and 'percentile_5' in mc_latest.columns:
                fig_mc.add_trace(go.Scatter(
                    x=mc_latest['simulation_year'], y=mc_latest['percentile_95'],
                    mode='lines', line=dict(width=0), showlegend=False,
                ))
                fig_mc.add_trace(go.Scatter(
                    x=mc_latest['simulation_year'], y=mc_latest['percentile_5'],
                    mode='lines', line=dict(width=0), fill='tonexty',
                    fillcolor='rgba(68,114,196,0.15)',
                    name='5th–95th Percentile',
                ))
            if 'percentile_84' in mc_latest.columns and 'percentile_16' in mc_latest.columns:
                fig_mc.add_trace(go.Scatter(
                    x=mc_latest['simulation_year'], y=mc_latest['percentile_84'],
                    mode='lines', line=dict(width=0), showlegend=False,
                ))
                fig_mc.add_trace(go.Scatter(
                    x=mc_latest['simulation_year'], y=mc_latest['percentile_16'],
                    mode='lines', line=dict(width=0), fill='tonexty',
                    fillcolor='rgba(68,114,196,0.3)',
                    name='16th–84th Percentile',
                ))
            if 'mean_price' in mc_latest.columns:
                fig_mc.add_trace(go.Scatter(
                    x=mc_latest['simulation_year'], y=mc_latest['mean_price'],
                    mode='lines+markers', line=dict(color='#4472C4', width=2),
                    name='Mean Price',
                ))
            if 'starting_price' in mc_latest.columns:
                start = mc_latest['starting_price'].iloc[0]
                if pd.notna(start):
                    fig_mc.add_hline(y=start, line_dash="dash", line_color="gray",
                                     annotation_text="Current Price")

            fig_mc.update_layout(
                title=f"Monte Carlo — {ticker}",
                xaxis_title="Year", yaxis_title="Price",
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(gridcolor='rgba(128,128,128,0.1)'),
                yaxis=dict(gridcolor='rgba(128,128,128,0.1)'),
            )
            st.plotly_chart(fig_mc, width="stretch")

        with mc_table_col:
            avail_cols = [c for c in ['simulation_year', 'percentile_5', 'mean_price', 'percentile_95']
                          if c in mc_latest.columns]
            display_mc = mc_latest[avail_cols].copy()
            rename = {'simulation_year': 'Year', 'percentile_5': '5th Pct',
                      'mean_price': 'Mean', 'percentile_95': '95th Pct'}
            display_mc = display_mc.rename(columns=rename)
            for col in ['5th Pct', 'Mean', '95th Pct']:
                if col in display_mc.columns:
                    display_mc[col] = display_mc[col].apply(
                        lambda x: f"{x:,.2f}" if pd.notna(x) else ""
                    )
            st.dataframe(display_mc, width="stretch", hide_index=True)

            # Risk metrics
            if 'var_95' in mc_latest.columns:
                yr1 = mc_latest[mc_latest['simulation_year'] == 1]
                if not yr1.empty:
                    r = yr1.iloc[0]
                    if pd.notna(r.get('var_95')):
                        st.metric("VaR (95%, 1Y)", f"{r['var_95']:.2f}")
                    if pd.notna(r.get('cvar_95')):
                        st.metric("CVaR (95%, 1Y)", f"{r['cvar_95']:.2f}")

        # MC graph
        mc_graph = gui_data.get_graph_path(f"Monte_Carlo_Sim_of_{ticker}.png")
        if mc_graph:
            with st.expander("Monte Carlo Graph (generated)", expanded=False):
                st.image(mc_graph, width="stretch")
    else:
        st.info(f"No Monte Carlo results for {ticker}.")
        mc_graph = gui_data.get_graph_path(f"Monte_Carlo_Sim_of_{ticker}.png")
        if mc_graph:
            st.image(mc_graph, caption="Monte Carlo Simulation", width="stretch")

    # ── Technical Indicators (expanded by default) ────────────────
    st.markdown("---")
    st.subheader("📊 Technical Indicators")

    if not price_df.empty:
        lt = price_df.iloc[-1]
        indicators = [
            ("RSI (14)", 'rsi_14', '{:.1f}'),
            ("MACD", 'macd', '{:.4f}'),
            ("MACD Signal", 'macd_signal', '{:.4f}'),
            ("MACD Hist", 'macd_histogram', '{:.4f}'),
            ("ATR (14)", 'atr_14', '{:.4f}'),
            ("Momentum", 'momentum', '{:.4f}'),
            ("Vol 5D", 'volatility_5d', '{:.4f}'),
            ("Vol 20D", 'volatility_20d', '{:.4f}'),
            ("Vol 60D", 'volatility_60d', '{:.4f}'),
            ("Volume Ratio", 'volume_ratio', '{:.2f}'),
            ("BB Width (20)", 'bollinger_Band_20_2STD', '{:.2f}'),
        ]

        ind_cols = st.columns(4)
        idx = 0
        for label, col_name, fmt in indicators:
            if col_name in lt.index and pd.notna(lt[col_name]):
                with ind_cols[idx % 4]:
                    st.metric(label, fmt.format(lt[col_name]))
                idx += 1

# ══════════════════════════════════════════════════════════════════════
#  TAB 2 — OM VIRKSOMHEDEN (About the Company)
# ══════════════════════════════════════════════════════════════════════

with tab_company:

    # ── Company Details (Nordnet "Detaljer" section) ──────────────
    st.subheader("Detaljer")

    det_col1, det_col2 = st.columns(2)
    with det_col1:
        st.markdown(f"**Company:** {company_name}")
        st.markdown(f"**Ticker:** {ticker}")
        st.markdown(f"**Industry:** {industry}")
    with det_col2:
        # Show the last known financial date
        if not financial_df.empty:
            fin = financial_df.iloc[-1]
            if 'date' in fin.index and pd.notna(fin['date']):
                st.markdown(f"**Latest Annual Report:** {fin['date']}")

    # ── Revenue & Earnings History ────────────────────────────────
    st.markdown("---")
    st.subheader("Revenue & Earnings History")

    if not financial_df.empty and len(financial_df) >= 2:
        hist_fin = financial_df.sort_values('date') if 'date' in financial_df.columns else financial_df

        fig_fin = go.Figure()

        if 'revenue' in hist_fin.columns:
            fig_fin.add_trace(go.Bar(
                x=hist_fin['date'] if 'date' in hist_fin.columns else hist_fin.index,
                y=hist_fin['revenue'],
                name='Revenue',
                marker_color='#4472C4',
                opacity=0.7,
            ))
        if 'operating_Earning' in hist_fin.columns:
            fig_fin.add_trace(go.Bar(
                x=hist_fin['date'] if 'date' in hist_fin.columns else hist_fin.index,
                y=hist_fin['operating_Earning'],
                name='EBIT',
                marker_color='#ED7D31',
                opacity=0.7,
            ))
        if 'net_Income' in hist_fin.columns:
            fig_fin.add_trace(go.Scatter(
                x=hist_fin['date'] if 'date' in hist_fin.columns else hist_fin.index,
                y=hist_fin['net_Income'],
                mode='lines+markers', name='Net Income',
                line=dict(color='#00CC66', width=2),
            ))

        fig_fin.update_layout(
            barmode='group',
            height=350,
            margin=dict(l=0, r=0, t=20, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis_tickformat=',',
            legend=dict(orientation='h', y=-0.15),
        )
        st.plotly_chart(fig_fin, width="stretch")
    else:
        st.info("Not enough annual financial data to show history.")

    # ── Annual Financial Highlights Table ─────────────────────────
    if not financial_df.empty:
        st.subheader("Financial Highlights")

        show_cols = [c for c in [
            'date', 'revenue', 'revenue_Growth', 'gross_Profit',
            'operating_Earning', 'net_Income', 'eps', 'eps_Growth',
            'return_On_Equity', 'return_On_Assets', 'current_Ratio',
            'debt_To_Equity', 'free_Cash_Flow',
        ] if c in financial_df.columns]

        if show_cols:
            highlights = financial_df[show_cols].copy()
            rename_cols = {
                'date': 'Year', 'revenue': 'Revenue',
                'revenue_Growth': 'Rev Growth', 'gross_Profit': 'Gross Profit',
                'operating_Earning': 'EBIT', 'net_Income': 'Net Income',
                'eps': 'EPS', 'eps_Growth': 'EPS Growth',
                'return_On_Equity': 'ROE', 'return_On_Assets': 'ROA',
                'current_Ratio': 'Current Ratio',
                'debt_To_Equity': 'D/E', 'free_Cash_Flow': 'FCF',
            }
            highlights = highlights.rename(columns=rename_cols)
            st.dataframe(highlights, width="stretch", hide_index=True)

    # ── Quarterly TTM Snapshot ────────────────────────────────────
    if not quarterly_income_df.empty:
        st.subheader("Quarterly TTM Snapshot")

        qi_latest = quarterly_income_df.iloc[-1]
        ttm_items = [
            ('Revenue TTM', qi_latest.get('revenue_ttm')),
            ('EBIT TTM', qi_latest.get('operating_income_ttm')),
            ('Net Income TTM', qi_latest.get('net_income_ttm')),
            ('EPS (Diluted) TTM', qi_latest.get('eps_diluted_ttm')),
            ('EBITDA TTM', qi_latest.get('ebitda_ttm')),
        ]

        tc = st.columns(len(ttm_items))
        for i, (lbl, val) in enumerate(ttm_items):
            with tc[i]:
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    if abs(val) > 1e9:
                        st.metric(lbl, f"{val / 1e9:.2f} B")
                    elif abs(val) > 1e6:
                        st.metric(lbl, f"{val / 1e6:.1f} M")
                    else:
                        st.metric(lbl, _fmt(val))
                else:
                    st.metric(lbl, "–")

    # ── Generated Graphs ──────────────────────────────────────────
    st.markdown("---")
    st.subheader("Generated Graphs")

    graph_names = [
        (f"stock_prediction_of_{ticker}.png", "Prediction (Training)"),
        (f"future_stock_prediction_of_{ticker}.png", "Future Prediction"),
        (f"Monte_Carlo_Sim_of_{ticker}.png", "Monte Carlo Simulation"),
    ]

    gcols = st.columns(len(graph_names))
    for gcol, (gfile, glabel) in zip(gcols, graph_names):
        gpath = gui_data.get_graph_path(gfile)
        with gcol:
            if gpath:
                st.image(gpath, caption=glabel, width="stretch")
            else:
                st.caption(f"{glabel}: not generated")

