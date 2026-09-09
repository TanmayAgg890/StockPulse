"""
app.py — StockPulse: Celestial Financial Intelligence Workspace.

Design System: Google Stitch ("The Celestial Financial Architect").
Features:
- Robust Ticker Navigation (Form-isolated search + Instant Category Discovery).
- Interactive Timeframe Selector (1M, 3M, 6M, 1Y).
- Visual Price Range Bar (Period Low to High Gauge).
- One-Click Quick Question Chips for AI Advisor.
- Permanent Bottom-Pinned Conversational AI Chat with Dynamic Context Re-Feeding.
- Tactile Liquid "Filling-Up" Button Hover Animations.
"""

from typing import List, Dict, Any
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from tickers import (
    INDIA_TRENDING,
    US_POPULAR,
    GLOBAL_INDICES,
    DEFAULT_WATCHLIST,
    get_ticker_display_name,
    get_ticker_region,
)
from model import (
    get_stock_data,
    add_moving_averages,
    compute_market_stats,
    train_and_predict_models,
    evaluate_models,
    compare_stocks,
)
from ai_advisor import (
    build_context,
    get_ai_insight,
    answer_followup,
)

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="StockPulse — Celestial Financial Workspace",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# 2. Session State Initialization
# -----------------------------------------------------------------------------
if "selected_ticker" not in st.session_state:
    st.session_state["selected_ticker"] = "RELIANCE.NS"

if "history" not in st.session_state:
    st.session_state["history"] = []

if "watchlist" not in st.session_state:
    st.session_state["watchlist"] = list(DEFAULT_WATCHLIST)

if "ai_cache" not in st.session_state:
    st.session_state["ai_cache"] = {}

if "chat_histories" not in st.session_state:
    st.session_state["chat_histories"] = {}

if "user_groq_key" not in st.session_state:
    st.session_state["user_groq_key"] = ""

if "timeframe" not in st.session_state:
    st.session_state["timeframe"] = "6M"

if "quick_question_trigger" not in st.session_state:
    st.session_state["quick_question_trigger"] = None


def navigate_to_ticker(new_ticker: str):
    """Reliably navigate to a new ticker symbol."""
    clean = new_ticker.strip().upper()
    if clean:
        cur = st.session_state.get("selected_ticker", "")
        if cur and cur != clean:
            if not st.session_state["history"] or st.session_state["history"][-1] != cur:
                st.session_state["history"].append(cur)
                if len(st.session_state["history"]) > 15:
                    st.session_state["history"].pop(0)
        st.session_state["selected_ticker"] = clean
        st.rerun()


def go_back():
    """Pop previous ticker from history and return."""
    if st.session_state["history"]:
        prev = st.session_state["history"].pop()
        st.session_state["selected_ticker"] = prev
        st.rerun()


active_ticker = st.session_state["selected_ticker"]
active_region = get_ticker_region(active_ticker)


# -----------------------------------------------------------------------------
# 3. Stitch Design System Styles: The Celestial Architect
# -----------------------------------------------------------------------------
if active_region == "INDIA":
    theme_gradient = (
        "radial-gradient(circle at 10% 15%, rgba(255, 184, 0, 0.09) 0%, transparent 45%), "
        "radial-gradient(circle at 90% 85%, rgba(56, 239, 125, 0.08) 0%, transparent 45%), "
        "linear-gradient(180deg, #070E1F 0%, #0C1324 100%)"
    )
    theme_accent = "#FFB800"
    theme_tag = "🇮🇳 DALAL STREET • NATIONAL STOCK EXCHANGE (NSE) REGIME"
elif active_region == "US":
    theme_gradient = (
        "radial-gradient(circle at 15% 15%, rgba(41, 121, 255, 0.1) 0%, transparent 45%), "
        "radial-gradient(circle at 85% 85%, rgba(255, 184, 0, 0.07) 0%, transparent 45%), "
        "linear-gradient(180deg, #070E1F 0%, #0C1324 100%)"
    )
    theme_accent = "#FFDCA1"
    theme_tag = "🇺🇸 WALL STREET & SILICON VALLEY • US EQUITY REGIME"
else:
    theme_gradient = (
        "radial-gradient(circle at 50% 10%, rgba(128, 250, 133, 0.07) 0%, transparent 50%), "
        "radial-gradient(circle at 50% 90%, rgba(0, 176, 255, 0.07) 0%, transparent 50%), "
        "linear-gradient(180deg, #070E1F 0%, #0C1324 100%)"
    )
    theme_accent = "#80FA85"
    theme_tag = "🌍 GLOBAL MARKET INDICES • CROSS-BORDER BENCHMARK REGIME"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    .stApp {{
        background: {theme_gradient} !important;
        background-attachment: fixed !important;
        color: #DBE2FB;
    }}

    /* Global Selection */
    ::selection {{
        background: {theme_accent} !important;
        color: #0C1324 !important;
    }}

    /* 1. BUTTONS & INTERACTIVE ELEMENTS: GUARANTEED POINTER CURSOR */
    button,
    button *,
    [role="button"],
    [role="button"] *,
    .stButton,
    .stButton *,
    .stButton > button,
    .stButton > button *,
    .stTabs [role="tab"],
    .stTabs [role="tab"] *,
    .stRadio label,
    .stRadio label * {{
        cursor: pointer !important;
    }}

    /* 2. TEXT SELECTION CURSOR (EXCLUDING BUTTONS) */
    .stMarkdown:not(.stButton *):not(button *) p,
    .stMarkdown:not(.stButton *):not(button *) span,
    h1, h2, h3, h4, .insight-body, .mono-label, .metric-val {{
        cursor: text;
    }}

    /* 3. TACTILE LIQUID "FILLING-UP" BUTTON HOVER EFFECT */
    .stButton > button {{
        position: relative !important;
        background-color: #141B2D !important;
        color: #DBE2FB !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 6px !important;
        overflow: hidden !important;
        transition: color 0.25s ease, border-color 0.25s ease, transform 0.2s ease, box-shadow 0.3s ease, background-size 0.35s cubic-bezier(0.4, 0, 0.2, 1) !important;
        background-image: linear-gradient(to top, {theme_accent}44 0%, {theme_accent}77 100%) !important;
        background-repeat: no-repeat !important;
        background-size: 100% 0% !important;
        background-position: bottom !important;
        z-index: 1 !important;
    }}

    .stButton > button:hover {{
        background-size: 100% 100% !important;
        border-color: {theme_accent} !important;
        color: #FFFFFF !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 18px {theme_accent}33 !important;
    }}

    .stButton > button:active {{
        transform: translateY(0px) !important;
    }}

    /* Header Container */
    .header-container {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 0.6rem;
        margin-bottom: 0.8rem;
    }}
    .brand-title {{
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.45rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #FFFFFF;
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    .theme-banner {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        background: rgba(255, 184, 0, 0.08);
        color: {theme_accent};
        border: 1px solid {theme_accent}44;
        padding: 3px 8px;
        border-radius: 6px;
        letter-spacing: 0.5px;
    }}
    .status-indicator {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: #38EF7D;
        display: flex;
        align-items: center;
        gap: 6px;
    }}
    .pulse-dot {{
        width: 8px;
        height: 8px;
        background-color: #38EF7D;
        border-radius: 50%;
        box-shadow: 0 0 8px #38EF7D;
        animation: pulse 2s infinite ease-in-out;
    }}
    @keyframes pulse {{
        0% {{ transform: scale(0.95); opacity: 0.7; }}
        50% {{ transform: scale(1.2); opacity: 1; }}
        100% {{ transform: scale(0.95); opacity: 0.7; }}
    }}

    /* Bento Metric Cards (No Wrapping, Tactile) */
    .bento-card {{
        background: #141B2D;
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 8px;
        padding: 0.85rem 0.85rem;
        margin-bottom: 0.8rem;
        overflow: hidden;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }}
    .bento-card:hover {{
        border-color: {theme_accent}88;
        transform: translateY(-1px);
    }}
    .mono-label {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        color: #9E8F78;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 0.2rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}
    .metric-val {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.25rem;
        font-weight: 700;
        color: #FFFFFF;
        letter-spacing: -0.5px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        line-height: 1.25;
    }}
    .metric-delta {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        font-weight: 500;
        margin-top: 0.25rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}
    .delta-pos {{ color: #38EF7D; }}
    .delta-neg {{ color: #FF6B6B; }}

    /* Range Bar Container */
    .range-bar-box {{
        background: #0E1424;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 6px;
        padding: 0.5rem 0.8rem;
        margin-bottom: 0.8rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
    }}
    .range-progress-bg {{
        width: 100%;
        height: 6px;
        background: #18233C;
        border-radius: 3px;
        margin: 6px 0;
        position: relative;
    }}
    .range-progress-fill {{
        height: 100%;
        background: linear-gradient(90deg, #38EF7D 0%, {theme_accent} 100%);
        border-radius: 3px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 4. Cached Data Pipelines
# -----------------------------------------------------------------------------
TIMEFRAME_MAP = {"1M": "1mo", "3M": "3mo", "6M": "6mo", "1Y": "1y"}

@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_stock_data(ticker_symbol: str, tf_code: str) -> pd.DataFrame:
    return get_stock_data(ticker_symbol, period=tf_code)


@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_comparison(tickers: List[str]):
    return compare_stocks(tickers)


# -----------------------------------------------------------------------------
# 5. Sidebar: Watchlist, Controls, & Force Refresh
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="brand-title">📈 StockPulse</div>', unsafe_allow_html=True)
    st.caption("Celestial Financial Intelligence Workspace")

    # History Navigation Back Button
    if st.session_state["history"]:
        prev_tick = st.session_state["history"][-1]
        if st.button(f"⬅ Back to {prev_tick}", use_container_width=True):
            go_back()

    st.divider()

    # Active Watchlist
    st.markdown('<div class="mono-label">★ ACTIVE WATCHLIST</div>', unsafe_allow_html=True)
    wl_cols = st.columns(2)
    for idx, wl_ticker in enumerate(st.session_state["watchlist"]):
        col = wl_cols[idx % 2]
        is_cur = wl_ticker == active_ticker
        btn_txt = f"● {wl_ticker}" if is_cur else wl_ticker
        if col.button(btn_txt, key=f"side_wl_{wl_ticker}", use_container_width=True):
            navigate_to_ticker(wl_ticker)

    st.divider()

    # Chart Style
    st.markdown('<div class="mono-label">VISUALIZATION MODE</div>', unsafe_allow_html=True)
    chart_mode = st.radio(
        "Chart Style",
        options=["Line (MA7 + MA30)", "Candlestick (OHLC + MAs)"],
        label_visibility="collapsed",
    )

    st.divider()

    # Data Refresh Button (User Comfort)
    if st.button("🔄 Force Refresh Market Feed", use_container_width=True):
        st.cache_data.clear()
        st.session_state["ai_cache"].clear()
        st.rerun()

    # Optional Groq Config
    with st.expander("🔑 AI Advisor API Config", expanded=False):
        st.caption("Configured in `.streamlit/secrets.toml`. Enter override key if needed:")
        override_key = st.text_input("Override Groq Key:", type="password", placeholder="gsk_...")
        if override_key and override_key != st.session_state["user_groq_key"]:
            st.session_state["user_groq_key"] = override_key
            st.session_state["ai_cache"].clear()
            st.rerun()

    st.caption("Google Stitch Design System • Groq AI • scikit-learn")


# -----------------------------------------------------------------------------
# 6. Top Command Header & Discovery Area
# -----------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="header-container">
        <div class="brand-title">
            STOCKPULSE <span class="theme-banner">{theme_tag}</span>
        </div>
        <div class="status-indicator">
            <div class="pulse-dot"></div> DATA CONNECTION READY
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Search Bar Form (Isolated so typing never overrides button clicks)
nav_col1, nav_col2, nav_col3 = st.columns([1.2, 4.5, 1.3])

with nav_col1:
    if st.session_state["history"]:
        prev_sym = st.session_state["history"][-1]
        if st.button(f"⬅ Back ({prev_sym})", use_container_width=True):
            go_back()
    else:
        st.button("⬅ Back", disabled=True, use_container_width=True)

with nav_col2:
    with st.form("search_input_form", clear_on_submit=False):
        sf_input, sf_btn = st.columns([4.8, 1.2])
        with sf_input:
            search_val = st.text_input(
                "Search Ticker:",
                value=active_ticker,
                placeholder="Type symbol (e.g. TCS.NS, NVDA, AAPL) & search...",
                label_visibility="collapsed",
            )
        with sf_btn:
            submitted_search = st.form_submit_button("🔍 Search", use_container_width=True)

    if submitted_search and search_val:
        navigate_to_ticker(search_val)

with nav_col3:
    in_wl = active_ticker in st.session_state["watchlist"]
    wl_btn_label = "★ In Watchlist" if in_wl else "☆ Watchlist"
    if st.button(wl_btn_label, use_container_width=True):
        if in_wl:
            st.session_state["watchlist"].remove(active_ticker)
        else:
            st.session_state["watchlist"].append(active_ticker)
        st.rerun()

# Category Discovery Buttons
st.markdown('<div class="mono-label" style="margin-top: 0.2rem;">QUICK DISCOVERY CATEGORIES</div>', unsafe_allow_html=True)
c_tab1, c_tab2, c_tab3 = st.tabs(["🇮🇳 India Trending", "🇺🇸 US Popular", "🌍 Global Benchmarks"])

def render_category_buttons(items: List[Dict[str, str]], prefix: str):
    cols = st.columns(len(items))
    for i, item in enumerate(items):
        sym = item["symbol"]
        is_cur = sym == active_ticker
        lbl = f"● {sym}" if is_cur else sym
        if cols[i].button(lbl, key=f"{prefix}_{sym}", use_container_width=True):
            navigate_to_ticker(sym)

with c_tab1:
    render_category_buttons(INDIA_TRENDING, "btn_ind")

with c_tab2:
    render_category_buttons(US_POPULAR, "btn_us")

with c_tab3:
    render_category_buttons(GLOBAL_INDICES, "btn_glob")

st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 7. Main Navigation: Workspace Tabs
# -----------------------------------------------------------------------------
tab_main, tab_comparator = st.tabs([
    f"📊 {active_ticker} Workspace & Grounded AI",
    "⚖️ Multi-Asset Comparative Benchmarking",
])


# =============================================================================
# TAB 1: WORKSPACE & AI ADVISOR
# =============================================================================
with tab_main:
    tf_selected = st.session_state["timeframe"]
    tf_code = TIMEFRAME_MAP.get(tf_selected, "6mo")

    with st.spinner(f"Fetching market data for {active_ticker} ({tf_selected})..."):
        try:
            raw_data = get_cached_stock_data(active_ticker, tf_code)
        except ValueError as val_e:
            st.error(f"❌ **Market Data Error:** {val_e}")
            raw_data = None
        except ConnectionError as conn_e:
            st.error(f"🌐 **Connection Timeout:** {conn_e}")
            raw_data = None
        except Exception as exc:
            st.error(f"⚠️ Unable to retrieve data for '{active_ticker}': {exc}")
            raw_data = None

    if raw_data is not None:
        df_processed = add_moving_averages(raw_data)
        stats = compute_market_stats(df_processed, active_ticker)
        curr = stats["currency"]

        # Run ML Models
        try:
            ml_predictions = train_and_predict_models(df_processed)
            ml_evals = evaluate_models(df_processed)
        except Exception as ml_err:
            ml_predictions = {"Linear Regression": stats["current_price"], "Random Forest": stats["current_price"]}
            ml_evals = {
                "Linear Regression": {"rmse": 0.0, "r2": 0.0},
                "Random Forest": {"rmse": 0.0, "r2": 0.0},
            }

        # Build Grounded RAG Context
        rag_context = build_context(active_ticker, stats, ml_predictions, ml_evals)

        # ---------------------------------------------------------------------
        # Bento Grid: Left Column (63%) | Right Column (37%)
        # ---------------------------------------------------------------------
        left_col, right_col = st.columns([63, 37], gap="medium")

        with left_col:
            # Row 1: 4 Metric Cards with No-Wrap Numeric Formatting
            m1, m2, m3, m4 = st.columns(4)

            with m1:
                st.markdown(
                    f"""
                    <div class="bento-card">
                        <div class="mono-label">CURRENT CLOSE</div>
                        <div class="metric-val">{curr}{stats['current_price']:,.2f}</div>
                        <div class="metric-delta {'delta-pos' if stats['period_change'] >= 0 else 'delta-neg'}">
                            {stats['period_change']:+,.2f} ({stats['period_change_pct']:+.2f}%)
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with m2:
                st.markdown(
                    f"""
                    <div class="bento-card">
                        <div class="mono-label">{tf_selected} PERIOD CHANGE</div>
                        <div class="metric-val {'delta-pos' if stats['period_change_pct'] >= 0 else 'delta-neg'}">
                            {stats['period_change_pct']:+.2f}%
                        </div>
                        <div class="metric-delta" style="color: #9E8F78;">{stats['data_points']} Trading Days</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with m3:
                st.markdown(
                    f"""
                    <div class="bento-card">
                        <div class="mono-label">{tf_selected} HIGH</div>
                        <div class="metric-val">{curr}{stats['period_high']:,.2f}</div>
                        <div class="metric-delta" style="color: #9E8F78;">Peak Observed</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with m4:
                st.markdown(
                    f"""
                    <div class="bento-card">
                        <div class="mono-label">{tf_selected} LOW</div>
                        <div class="metric-val">{curr}{stats['period_low']:,.2f}</div>
                        <div class="metric-delta" style="color: #9E8F78;">Trough Observed</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Comfy Feature: Visual Price Range Position Bar
            st.markdown(
                f"""
                <div class="range-bar-box">
                    <div style="display: flex; justify-content: space-between;">
                        <span>🟢 {tf_selected} Low: <b>{curr}{stats['period_low']:,.2f}</b></span>
                        <span>Current: <b>{curr}{stats['current_price']:,.2f}</b> ({stats['range_position']}%)</span>
                        <span>🔴 {tf_selected} High: <b>{curr}{stats['period_high']:,.2f}</b></span>
                    </div>
                    <div class="range-progress-bg">
                        <div class="range-progress-fill" style="width: {stats['range_position']}%;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Row 2: Chart Controls (Timeframe Selector) & Plotly Chart
            chart_header = get_ticker_display_name(active_ticker)
            
            c_header_col, c_tf_col = st.columns([3.5, 1.5])
            with c_header_col:
                st.markdown(f"### 📊 {chart_header}")
            with c_tf_col:
                # Comfy Timeframe Selector
                tf_cols = st.columns(4)
                for i, tf_lbl in enumerate(["1M", "3M", "6M", "1Y"]):
                    is_active_tf = tf_lbl == st.session_state["timeframe"]
                    tf_btn_txt = f"[{tf_lbl}]" if is_active_tf else tf_lbl
                    if tf_cols[i].button(tf_btn_txt, key=f"tf_{tf_lbl}", use_container_width=True):
                        st.session_state["timeframe"] = tf_lbl
                        st.rerun()

            fig = go.Figure()

            if "Candlestick" in chart_mode:
                fig.add_trace(
                    go.Candlestick(
                        x=df_processed.index,
                        open=df_processed["Open"],
                        high=df_processed["High"],
                        low=df_processed["Low"],
                        close=df_processed["Close"],
                        name="OHLC Price",
                        increasing_line_color="#38EF7D",
                        decreasing_line_color="#FF6B6B",
                    )
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=df_processed.index,
                        y=df_processed["Close"],
                        mode="lines",
                        name="Close Price",
                        line=dict(color="#FFB800", width=2.2),
                        hovertemplate=f"<b>Date:</b> %{{x|%Y-%m-%d}}<br><b>Close:</b> {curr}%{{y:,.2f}}<extra></extra>",
                    )
                )

            fig.add_trace(
                go.Scatter(
                    x=df_processed.index,
                    y=df_processed["MA7"],
                    mode="lines",
                    name="7-Day MA",
                    line=dict(color="#FFDCA1", width=1.6, dash="dash"),
                    hovertemplate=f"<b>MA7:</b> {curr}%{{y:,.2f}}<extra></extra>",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=df_processed.index,
                    y=df_processed["MA30"],
                    mode="lines",
                    name="30-Day MA",
                    line=dict(color="#38EF7D", width=1.6, dash="dot"),
                    hovertemplate=f"<b>MA30:</b> {curr}%{{y:,.2f}}<extra></extra>",
                )
            )

            fig.update_layout(
                paper_bgcolor="rgba(20, 27, 45, 0.7)",
                plot_bgcolor="rgba(20, 27, 45, 0.7)",
                xaxis=dict(
                    title=None,
                    showgrid=True,
                    gridcolor="rgba(255, 255, 255, 0.05)",
                    rangeslider=dict(visible=False),
                    tickfont=dict(color="#9E8F78", size=10),
                ),
                yaxis=dict(
                    title=dict(text=f"Price ({curr})", font=dict(color="#9E8F78", size=11)),
                    showgrid=True,
                    gridcolor="rgba(255, 255, 255, 0.05)",
                    tickfont=dict(color="#9E8F78", size=10),
                ),
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    font=dict(color="#DBE2FB", size=10),
                ),
                margin=dict(l=15, r=15, t=35, b=25),
                height=430,
            )

            st.plotly_chart(fig, use_container_width=True)

            # Technical Momentum Note
            ma_msg = (
                "🟢 **Momentum Note:** Short-term 7-day moving average is tracking above the 30-day baseline."
                if stats["ma7"] > stats["ma30"]
                else "🔴 **Momentum Note:** Short-term 7-day moving average is tracking below the 30-day baseline."
            )
            st.caption(ma_msg)

            # Row 3: ML Benchmarking Cards
            st.markdown('<div class="mono-label" style="margin-top: 0.8rem;">AUTOREGRESSIVE PREDICTION BENCHMARK</div>', unsafe_allow_html=True)
            ml_c1, ml_c2 = st.columns(2)

            lr_p = ml_predictions["Linear Regression"]
            lr_d = lr_p - stats["current_price"]
            lr_d_pct = (lr_d / stats["current_price"]) * 100
            lr_rmse = ml_evals["Linear Regression"]["rmse"]
            lr_r2 = ml_evals["Linear Regression"]["r2"]

            with ml_c1:
                st.markdown(
                    f"""
                    <div class="bento-card">
                        <div class="mono-label">🔹 LINEAR REGRESSION (BASELINE)</div>
                        <div class="metric-val">{curr}{lr_p:,.2f}</div>
                        <div class="metric-delta {'delta-pos' if lr_d >= 0 else 'delta-neg'}">
                            {lr_d:+,.2f} ({lr_d_pct:+.2f}%)
                        </div>
                        <div style="margin-top: 0.5rem; font-size: 0.75rem; color: #9E8F78; font-family: 'JetBrains Mono', monospace;">
                            Test RMSE: {curr}{lr_rmse:.2f} &nbsp;|&nbsp; R²: {lr_r2:.4f}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            rf_p = ml_predictions["Random Forest"]
            rf_d = rf_p - stats["current_price"]
            rf_d_pct = (rf_d / stats["current_price"]) * 100
            rf_rmse = ml_evals["Random Forest"]["rmse"]
            rf_r2 = ml_evals["Random Forest"]["r2"]

            with ml_c2:
                st.markdown(
                    f"""
                    <div class="bento-card">
                        <div class="mono-label">🌲 RANDOM FOREST (100 TREES)</div>
                        <div class="metric-val">{curr}{rf_p:,.2f}</div>
                        <div class="metric-delta {'delta-pos' if rf_d >= 0 else 'delta-neg'}">
                            {rf_d:+,.2f} ({rf_d_pct:+.2f}%)
                        </div>
                        <div style="margin-top: 0.5rem; font-size: 0.75rem; color: #9E8F78; font-family: 'JetBrains Mono', monospace;">
                            Test RMSE: {curr}{rf_rmse:.2f} &nbsp;|&nbsp; R²: {rf_r2:.4f}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with st.expander("🔬 Model Transparency & Methodology", expanded=False):
                st.markdown(
                    """
                    - **Lag Features:** Uses closing price lags ($t-1, t-2, t-3$) as supervised inputs.
                    - **Strict Chronological Split:** First 80% of days used for training, most recent 20% for testing.
                    - **Decoupled AI:** The Large Language Model does **not** generate numerical prices.
                    """
                )

        # ---------------------------------------------------------------------
        # Right Column: Conversational AI Advisor with Pinned Bottom Input
        # ---------------------------------------------------------------------
        with right_col:
            st.markdown(
                f"""
                <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 0.4rem; margin-bottom: 0.6rem;">
                    <span style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 0.95rem; font-weight: 700; color: {theme_accent};">
                        🧠 AI ADVISOR
                    </span>
                    <span style="font-size: 0.68rem; color: #38EF7D; font-family: 'JetBrains Mono', monospace;">
                        ● GROUNDED RAG-LITE
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Auto-align chat history with active ticker
            if active_ticker not in st.session_state["chat_histories"]:
                st.session_state["chat_histories"][active_ticker] = []

            # Retrieve / Generate baseline insight for current stock
            if active_ticker not in st.session_state["ai_cache"]:
                with st.spinner(f"Analyzing {active_ticker}..."):
                    base_insight = get_ai_insight(rag_context)
                    st.session_state["ai_cache"][active_ticker] = base_insight
            else:
                base_insight = st.session_state["ai_cache"][active_ticker]

            # 1. Scrollable Chat Feed (Chat flows downwards)
            chat_container = st.container(height=360)
            with chat_container:
                with st.chat_message("assistant", avatar="🧠"):
                    st.markdown(base_insight)

                stock_chat = st.session_state["chat_histories"][active_ticker]
                for msg in stock_chat:
                    with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🧠"):
                        st.markdown(msg["content"])

            # Comfy Feature: Quick-Question Clickable Chips
            st.markdown('<div class="mono-label" style="margin-top: 0.3rem;">QUICK ANALYSIS PROMPTS</div>', unsafe_allow_html=True)
            q_cols = st.columns(2)
            quick_prompts = [
                ("💡 Explain Trend", f"Explain {active_ticker}'s technical trend based on MA7 vs MA30."),
                ("🤖 Compare Models", f"Compare the Linear Regression vs Random Forest projections for {active_ticker}."),
                ("⚠️ Risk Summary", f"What are the main risk factors and limitations of {active_ticker}'s prediction?"),
                ("📊 What is R²?", f"What does the R² and RMSE score tell us about {active_ticker}'s predictability?"),
            ]

            quick_clicked_question = None
            for idx, (chip_label, prompt_text) in enumerate(quick_prompts):
                target_col = q_cols[idx % 2]
                if target_col.button(chip_label, key=f"chip_{active_ticker}_{idx}", use_container_width=True):
                    quick_clicked_question = prompt_text

            # 2. Permanent Bottom Input Form (Always visible at the bottom)
            with st.form(key=f"bottom_chat_form_{active_ticker}", clear_on_submit=True):
                c_input_col, c_btn_col = st.columns([4, 1])
                with c_input_col:
                    user_q = st.text_input(
                        "Ask AI Advisor",
                        placeholder=f"Ask about {active_ticker} (or click a prompt above)...",
                        label_visibility="collapsed",
                    )
                with c_btn_col:
                    send_clicked = st.form_submit_button("Send ➔", use_container_width=True)

            effective_question = quick_clicked_question or (user_q if send_clicked else None)

            if effective_question:
                st.session_state["chat_histories"][active_ticker].append({"role": "user", "content": effective_question})

                with st.spinner(f"Generating grounded answer for {active_ticker}..."):
                    ai_reply = answer_followup(
                        context=rag_context,
                        chat_history=st.session_state["chat_histories"][active_ticker],
                        question=effective_question,
                    )

                st.session_state["chat_histories"][active_ticker].append({"role": "assistant", "content": ai_reply})
                st.rerun()

            with st.expander("🔍 Inspect Retrieved Context (RAG Block)", expanded=False):
                st.caption("Factual context passed to Groq for strict grounding:")
                st.code(rag_context, language="text")

            st.caption("⚠️ **Educational Project:** Not financial advice. Past performance does not guarantee future results.")


# =============================================================================
# TAB 2: MULTI-ASSET COMPARATOR
# =============================================================================
with tab_comparator:
    st.markdown('<div class="mono-label">CROSS-ASSET PERFORMANCE BENCHMARK</div>', unsafe_allow_html=True)
    st.caption("Normalize and benchmark cumulative 6-month percentage returns across multiple equities simultaneously.")

    comp_selection = st.multiselect(
        "Select Assets to Benchmark:",
        options=[
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
            "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "TSLA", "META",
        ],
        default=["RELIANCE.NS", "TCS.NS", "AAPL"],
    )

    if len(comp_selection) < 2:
        st.info("Select at least 2 tickers above to generate cross-asset performance curves.")
    else:
        with st.spinner("Synthesizing multi-asset comparison..."):
            ret_df, comp_stats = get_cached_comparison(comp_selection)

        if not ret_df.empty:
            c_fig = go.Figure()
            palette = ["#FFB800", "#38EF7D", "#2979FF", "#D500F9", "#00B0FF", "#FFD600"]

            for i, asset in enumerate(ret_df.columns):
                c_fig.add_trace(
                    go.Scatter(
                        x=ret_df.index,
                        y=ret_df[asset],
                        mode="lines",
                        name=asset,
                        line=dict(width=2.2, color=palette[i % len(palette)]),
                        hovertemplate=f"<b>{asset}:</b> %{{y:+.2f}}%<extra></extra>",
                    )
                )

            c_fig.update_layout(
                paper_bgcolor="rgba(20, 27, 45, 0.7)",
                plot_bgcolor="rgba(20, 27, 45, 0.7)",
                title=dict(
                    text="<b>Cumulative Growth (% from Baseline)</b>",
                    x=0.01,
                    font=dict(family="Plus Jakarta Sans", size=14, color="#FFFFFF"),
                ),
                xaxis=dict(showgrid=True, gridcolor="rgba(255, 255, 255, 0.05)", tickfont=dict(color="#9E8F78", size=10)),
                yaxis=dict(
                    title=dict(text="Cumulative Return (%)", font=dict(color="#9E8F78", size=11)),
                    showgrid=True,
                    gridcolor="rgba(255, 255, 255, 0.05)",
                    tickfont=dict(color="#9E8F78", size=10),
                ),
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#DBE2FB", size=11)),
                margin=dict(l=15, r=15, t=50, b=25),
                height=450,
            )

            st.plotly_chart(c_fig, use_container_width=True)

            st.markdown('<div class="mono-label" style="margin-top: 1rem;">BENCHMARK SUMMARY METRICS</div>', unsafe_allow_html=True)
            table_records = []
            for t_name, t_meta in comp_stats.items():
                table_records.append({
                    "Symbol": t_name,
                    "Current Price": f"{t_meta['current_price']:,.2f}",
                    "Cumulative Return": f"{t_meta['total_return']:+.2f}%",
                    "Annualized Volatility": f"{t_meta['volatility']:.2f}%",
                })
            st.dataframe(pd.DataFrame(table_records), use_container_width=True, hide_index=True)
