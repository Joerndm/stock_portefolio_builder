"""
Stock Portfolio Builder — Streamlit GUI

Launch with:
    streamlit run streamlit_app.py

Multi-page app with sidebar navigation.
"""
import streamlit as st

# ─── Page Config (must be first Streamlit call) ─────────────────────
st.set_page_config(
    page_title="Stock Portfolio Builder",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Dark theme custom CSS (inspired by Nordnet) ────────────────────
st.markdown("""
<style>
    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #1a1a2e;
        border: 1px solid #2d2d44;
        border-radius: 8px;
        padding: 12px 16px;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.4rem;
    }
    [data-testid="stMetricDelta"] > div {
        font-size: 0.9rem;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #0e1117;
    }

    /* Table header */
    thead tr th {
        background-color: #1a1a2e !important;
        color: #e0e0e0 !important;
    }

    /* Reduce padding on main block */
    .block-container {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Define pages ────────────────────────────────────────────────────
dashboard = st.Page("pages/1_Dashboard.py", title="Dashboard", icon="📊", default=True)
builder = st.Page("pages/2_Portfolio_Builder.py", title="Portfolio Builder", icon="🔧")
explorer = st.Page("pages/3_Stock_Explorer.py", title="Stock Explorer", icon="📈")

pg = st.navigation([dashboard, builder, explorer])

# ─── Sidebar branding ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📈 Portfolio Builder")
    st.caption("ML-powered stock portfolio optimization")
    st.divider()

pg.run()
