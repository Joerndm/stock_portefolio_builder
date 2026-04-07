"""
Shared data loading layer for the Streamlit GUI.

Provides cached data access to the database via db_interactions.
All functions use @st.cache_data or @st.cache_resource for performance.
"""
import os
import sys
import datetime
from typing import Optional, List, Dict

import pandas as pd
import streamlit as st

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import db_interactions


# ─── Cache Configuration ─────────────────────────────────────────────
# Default data refresh interval in seconds.  Change this single constant
# (or expose it through a sidebar selector) to adjust how often the GUI
# re-queries the database.  3600 = 1 hour.
CACHE_TTL_SECONDS: int = 3600


# ─── Ticker & Stock Info ─────────────────────────────────────────────

@st.cache_data(ttl=CACHE_TTL_SECONDS)
def get_all_tickers() -> List[str]:
    """Get all non-index ticker symbols from the database."""
    return db_interactions.import_ticker_list()


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def get_stock_info(ticker: str = "") -> pd.DataFrame:
    """Get stock info (ticker, company_Name, industry). Empty ticker = all."""
    return db_interactions.import_stock_info_data(ticker)


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def get_tickers_with_predictions(max_age_days: int = 30) -> List[str]:
    """Get tickers that have recent predictions."""
    return db_interactions.get_tickers_with_predictions(max_age_days=max_age_days)


# ─── Price Data ──────────────────────────────────────────────────────

@st.cache_data(ttl=CACHE_TTL_SECONDS)
def get_stock_prices(ticker: str, amount: int = 1000) -> pd.DataFrame:
    """Get historical price data for a single ticker."""
    return db_interactions.import_stock_price_data(amount=amount, stock_ticker=ticker)


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def get_full_dataset(ticker: str) -> pd.DataFrame:
    """Get the complete combined dataset for a ticker (price + financials + ratios)."""
    return db_interactions.import_stock_dataset(ticker)


# ─── Predictions & Monte Carlo ───────────────────────────────────────

@st.cache_data(ttl=CACHE_TTL_SECONDS)
def get_prediction_summary(ticker: str, max_age_days: int = 30) -> Optional[Dict]:
    """Get prediction summary for a ticker."""
    return db_interactions.get_prediction_summary(ticker, max_age_days=max_age_days)


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def get_predictions_extended(
    ticker: str = None,
    prediction_date: datetime.date = None
) -> pd.DataFrame:
    """Get extended prediction data."""
    return db_interactions.import_stock_predictions_extended(
        ticker=ticker, prediction_date=prediction_date
    )


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def get_monte_carlo_results(
    ticker: str = None,
    simulation_date: datetime.date = None
) -> pd.DataFrame:
    """Get Monte Carlo simulation results."""
    return db_interactions.import_monte_carlo_results(
        ticker=ticker, simulation_date=simulation_date
    )


# ─── Portfolio Runs & Holdings ───────────────────────────────────────

@st.cache_data(ttl=CACHE_TTL_SECONDS)
def get_portfolio_runs() -> pd.DataFrame:
    """Get all portfolio runs from the database."""
    try:
        import fetch_secrets
        import db_connectors
        db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()
        engine = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)
        try:
            df = pd.read_sql("SELECT * FROM portfolio_runs ORDER BY run_date DESC", engine)
            return df
        finally:
            engine.dispose()
    except Exception as e:
        st.warning(f"Could not load portfolio runs: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def get_portfolio_holdings(run_id: int) -> pd.DataFrame:
    """Get holdings for a specific portfolio run."""
    try:
        import fetch_secrets
        import db_connectors
        db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()
        engine = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)
        try:
            df = pd.read_sql(
                "SELECT * FROM portfolio_holdings WHERE run_id = %s ORDER BY `rank` ASC",
                engine, params=(int(run_id),)
            )
            return df
        finally:
            engine.dispose()
    except Exception as e:
        st.warning(f"Could not load portfolio holdings: {e}")
        return pd.DataFrame()


# ─── Model Status ────────────────────────────────────────────────────

@st.cache_data(ttl=CACHE_TTL_SECONDS)
def get_model_status(max_age_days: int = 30) -> pd.DataFrame:
    """Get model training status for all tickers."""
    return db_interactions.get_model_status_for_all_tickers(max_age_days=max_age_days)


# ─── Financial Ratios & Fundamentals ─────────────────────────────────

@st.cache_data(ttl=CACHE_TTL_SECONDS)
def get_stock_ratios(ticker: str, amount: int = 1) -> pd.DataFrame:
    """Get valuation ratios (P/E, P/S, P/B, P/FCF) for a ticker."""
    return db_interactions.import_stock_ratio_data(amount=amount, stock_ticker=ticker)


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def get_financial_data(ticker: str, amount: int = 5) -> pd.DataFrame:
    """Get merged annual financial data (income + balance sheet + cash flow)."""
    return db_interactions.import_stock_financial_data(amount=amount, stock_ticker=ticker)


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def get_quarterly_income(ticker: str) -> pd.DataFrame:
    """Get quarterly income statement data (includes TTM aggregates)."""
    return db_interactions.import_quarterly_income_data(ticker)


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def get_quarterly_balancesheet(ticker: str) -> pd.DataFrame:
    """Get quarterly balance sheet data."""
    return db_interactions.import_quarterly_balancesheet_data(ticker)


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def get_quarterly_cashflow(ticker: str) -> pd.DataFrame:
    """Get quarterly cash flow data."""
    return db_interactions.import_quarterly_cashflow_data(ticker)


# ─── Utility ─────────────────────────────────────────────────────────

GRAPHS_DIR = os.path.join(PROJECT_ROOT, "generated_graphs")


def get_graph_path(filename: str) -> Optional[str]:
    """Get the full path to a generated graph file, or None if it doesn't exist."""
    path = os.path.join(GRAPHS_DIR, filename)
    return path if os.path.isfile(path) else None


def list_available_graphs() -> List[str]:
    """List all generated graph filenames."""
    if not os.path.isdir(GRAPHS_DIR):
        return []
    return sorted(f for f in os.listdir(GRAPHS_DIR) if f.endswith('.png'))


def get_tickers_with_graphs() -> List[str]:
    """Get unique tickers that have generated prediction graphs."""
    graphs = list_available_graphs()
    tickers = set()
    for g in graphs:
        if g.startswith("future_stock_prediction_of_"):
            ticker = g.replace("future_stock_prediction_of_", "").replace(".png", "")
            tickers.add(ticker)
    return sorted(tickers)


# ─── Data Quality ────────────────────────────────────────────────────

@st.cache_data(ttl=CACHE_TTL_SECONDS)
def get_data_quality_summary() -> Dict:
    """
    Run lightweight data quality checks and return a summary dict.
    
    Returns a dict with keys:
        total_tickers, stale_tickers, missing_quarterly, 
        null_ratio_tickers, extreme_ratios, table_coverage
    """
    import fetch_secrets
    import db_connectors
    
    db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()
    engine = db_connectors.pandas_mysql_connector(db_host, db_user, db_pass, db_name)
    
    try:
        return _run_data_quality_queries(engine)
    finally:
        engine.dispose()


def _run_data_quality_queries(engine) -> Dict:
    """Execute all data quality queries (split out for finally-safe disposal)."""
    from sqlalchemy import text as sa_text

    result = {}
    
    # Total tickers
    r = pd.read_sql("SELECT COUNT(DISTINCT ticker) AS cnt FROM stock_info_data WHERE industry != 'Index'", engine)
    result['total_tickers'] = int(r['cnt'].iloc[0])
    
    # Stale tickers (>7 days without new price data)
    stale_q = sa_text("""
        SELECT ticker, MAX(date) AS last_date,
               DATEDIFF(CURDATE(), MAX(date)) AS days_stale
        FROM stock_price_data
        GROUP BY ticker
        HAVING days_stale > 7
        ORDER BY days_stale DESC
    """)
    result['stale_tickers'] = pd.read_sql(stale_q, engine)
    
    # Table coverage (how many tickers have data in each table)
    tables = [
        'stock_price_data', 'stock_income_stmt_data', 'stock_balancesheet_data',
        'stock_cash_flow_data', 'stock_ratio_data',
        'stock_income_stmt_quarterly', 'stock_balancesheet_quarterly', 'stock_cashflow_quarterly'
    ]
    coverage = {}
    for table in tables:
        try:
            r = pd.read_sql(f"SELECT COUNT(DISTINCT ticker) AS cnt FROM {table}", engine)
            coverage[table] = int(r['cnt'].iloc[0])
        except Exception:
            coverage[table] = 0
    result['table_coverage'] = coverage
    
    # Tickers with >50% NULL ratios
    null_ratio_q = sa_text("""
        SELECT ticker,
               COUNT(*) AS total_rows,
               ROUND(SUM(CASE WHEN p_e IS NULL THEN 1 ELSE 0 END) / COUNT(*) * 100) AS pct_null_pe,
               ROUND(SUM(CASE WHEN p_b IS NULL THEN 1 ELSE 0 END) / COUNT(*) * 100) AS pct_null_pb,
               ROUND(SUM(CASE WHEN p_s IS NULL THEN 1 ELSE 0 END) / COUNT(*) * 100) AS pct_null_ps,
               ROUND(SUM(CASE WHEN p_fcf IS NULL THEN 1 ELSE 0 END) / COUNT(*) * 100) AS pct_null_pfcf
        FROM stock_ratio_data
        GROUP BY ticker
        HAVING pct_null_pe > 50 OR pct_null_pb > 50 OR pct_null_pfcf > 50
        ORDER BY pct_null_pe DESC
    """)
    result['null_ratio_tickers'] = pd.read_sql(null_ratio_q, engine)
    
    # Extreme ratio values still in DB
    extreme_q = sa_text("""
        SELECT COUNT(*) AS cnt FROM stock_ratio_data
        WHERE ABS(p_e) > 10000 OR ABS(p_b) > 10000 OR ABS(p_s) > 10000 OR ABS(p_fcf) > 10000
    """)
    r = pd.read_sql(extreme_q, engine)
    result['extreme_ratio_rows'] = int(r['cnt'].iloc[0])
    
    # Missing quarterly income data (have price data but no quarterly income)
    missing_q = sa_text("""
        SELECT p.ticker
        FROM (SELECT DISTINCT ticker FROM stock_price_data) p
        LEFT JOIN (SELECT DISTINCT ticker FROM stock_income_stmt_quarterly) q
            ON p.ticker = q.ticker
        LEFT JOIN stock_info_data si ON p.ticker = si.ticker
        WHERE q.ticker IS NULL AND si.industry != 'Index'
    """)
    result['missing_quarterly'] = pd.read_sql(missing_q, engine)['ticker'].tolist()
    
    # Null financial_date_used
    fdu_q = sa_text("""
        SELECT COUNT(DISTINCT ticker) AS cnt
        FROM stock_ratio_data
        WHERE financial_date_used IS NULL
    """)
    r = pd.read_sql(fdu_q, engine)
    result['null_financial_date_used_tickers'] = int(r['cnt'].iloc[0])
    
    return result
