"""
app.py — StockPulse: Market Intelligence & Grounded AI Analytics Workspace.

Key Enhancements:
- Themed Dynamic Ambient Backgrounds (🇮🇳 Dalal Street/India, 🇺🇸 Wall Street/US, 🌍 Global).
- Navigation History Stack with prominent "⬅ Back" button.
- Modern Bottom-Pinned AI Chat with downward message stream and automatic context re-alignment upon ticker switch.
- Robust Text Selection Cursors and Pointer styling.
- Responsive Bento Metric Cards with no-wrap typography preventing digit-breaking.
- Plotly Chart Layout Optimization eliminating title-legend overlap.
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
    page_title="StockPulse — Market Intelligence Workspace",
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


def navigate_to_ticker(new_ticker: str):
    """Navigate to a new ticker while recording the previous one in history."""
    clean = new_ticker.strip().upper()
    if clean and clean != st.session_state["selected_ticker"]:
        # Push current ticker to history stack (max 15 items)
        cur = st.session_state["selected_ticker"]
        if not st.session_state["history"] or st.session_state["history"][-1] != cur:
            st.session_state["history"].append(cur)
            if len(st.session_state["history"]) > 15:
                st.session_state["history"].pop(0)
        st.session_state["selected_ticker"] = clean
        st.rerun()


def go_back():
    """Navigate back to the previous ticker in the history stack."""
    if st.session_state["history"]:
        prev = st.session_state["history"].pop()
        st.session_state["selected_ticker"] = prev
        st.rerun()


active_ticker = st.session_state["selected_ticker"]
active_region = get_ticker_region(active_ticker)


# -----------------------------------------------------------------------------
# 3. Dynamic Themed Background & Precision Typography CSS
# -----------------------------------------------------------------------------
if active_region == "INDIA":
    theme_gradient = (
        "radial-gradient(circle at 10% 15%, rgba(255, 153, 51, 0.08) 0%, transparent 45%), "
        "radial-gradient(circle at 90% 85%, rgba(18, 136, 7, 0.08) 0%, transparent 45%), "
        "linear-gradient(180deg, #07090E 0%, #0A0F18 100%)"
    )
    theme_accent = "#FF9933"
    theme_tag = "🇮🇳 DALAL STREET • NATIONAL STOCK EXCHANGE (NSE) REGIME"
elif active_region == "US":
    theme_gradient = (
        "radial-gradient(circle at 15% 15%, rgba(41, 121, 255, 0.09) 0%, transparent 45%), "
        "radial-gradient(circle at 85% 85%, rgba(0, 230, 118, 0.07) 0%, transparent 45%), "
        "linear-gradient(180deg, #07090E 0%, #090E18 100%)"
    )
    theme_accent = "#2979FF"
    theme_tag = "🇺🇸 WALL STREET & SILICON VALLEY • US EQUITY REGIME"
else:
    theme_gradient = (
        "radial-gradient(circle at 50% 10%, rgba(213, 0, 249, 0.07) 0%, transparent 50%), "
        "radial-gradient(circle at 50% 90%, rgba(0, 176, 255, 0.07) 0%, transparent 50%), "
        "linear-gradient(180deg, #07090E 0%, #0B0E17 100%)"
    )
    theme_accent = "#00B0FF"
    theme_tag = "🌍 GLOBAL INDICES & MULTI-EXCHANGE BENCHMARK REGIME"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    .stApp {{
        background: {theme_gradient} !important;
        background-attachment: fixed !important;
        color: #E6EDF3;
    }}

    /* Global Text Selection & Cursor Behaviors */
    ::selection {{
        background: {theme_accent} !important;
        color: #000000 !important;
    }}

    p, span, h1, h2, h3, h4, .selectable-text, .insight-body, .stMarkdown {{
        cursor: text;
    }}

    button, [role="button"], .stButton > button, input, select, .stRadio label {{
        cursor: pointer !important;
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
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.35rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: #FFFFFF;
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    .theme-banner {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        background: rgba(255, 255, 255, 0.03);
        color: {theme_accent};
        border: 1px solid {theme_accent}44;
        padding: 3px 8px;
        border-radius: 4px;
        letter-spacing: 0.5px;
    }}
    .status-indicator {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        color: #00E676;
        display: flex;
        align-items: center;
        gap: 6px;
    }}
    .pulse-dot {{
        width: 8px;
        height: 8px;
        background-color: #00E676;
        border-radius: 50%;
        box-shadow: 0 0 8px #00E676;
        animation: pulse 2s infinite ease-in-out;
    }}
    @keyframes pulse {{
        0% {{ transform: scale(0.95); opacity: 0.7; }}
        50% {{ transform: scale(1.2); opacity: 1; }}
        100% {{ transform: scale(0.95); opacity: 0.7; }}
    }}

    /* Bento Metric Cards (No-wrap, Compact) */
    .bento-card {{
        background: #0D111A;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 0.85rem 0.8rem;
        margin-bottom: 0.8rem;
        transition: border 0.2s ease;
        overflow: hidden;
    }}
    .bento-card:hover {{
        border-color: {theme_accent}88;
    }}
    .mono-label {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
        color: #8B949E;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 0.2rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}
    .metric-val {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.25rem;
        font-weight: 700;
        color: #FFFFFF;
        letter-spacing: -0.5px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        line-height: 1.2;
    }}
    .metric-delta {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        font-weight: 500;
        margin-top: 0.2rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}
    .delta-pos {{ color: #00E676; }}
    .delta-neg {{ color: #FF5252; }}

    /* AI Advisor Scrollable Feed */
    .chat-scroll-container {{
        max-height: 480px;
        overflow-y: auto;
        padding-right: 6px;
        margin-bottom: 0.8rem;
    }}
    .chat-bubble-ai {{
        background: #0B0F19;
        border-left: 3px solid {theme_accent};
        border-radius: 6px;
        padding: 0.75rem 0.9rem;
        margin-bottom: 0.75rem;
        font-size: 0.88rem;
        line-height: 1.45;
    }}
    .chat-bubble-user {{
        background: #141A26;
        border-right: 3px solid #00E676;
        border-radius: 6px;
        padding: 0.65rem 0.85rem;
        margin-bottom: 0.75rem;
        font-size: 0.85rem;
        text-align: right;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 4. Cached Data Ingestion
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_stock_data(ticker_symbol: str) -> pd.DataFrame:
    return get_stock_data(ticker_symbol)


@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_comparison(tickers: List[str]):
    return compare_stocks(tickers)


# -----------------------------------------------------------------------------
# 5. Sidebar: Watchlist, Back Button, & Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="brand-title">📈 StockPulse</div>', unsafe_allow_html=True)
    st.caption("AI-Powered Financial Analytics Workspace")

    # History Navigation Back Button
    if st.session_state["history"]:
        prev_tick = st.session_state["history"][-1]
        if st.button(f"⬅ Back to {prev_tick}", use_container_width=True):
            go_back()

    st.divider()

    # Watchlist
    st.markdown('<div class="mono-label">★ ACTIVE WATCHLIST</div>', unsafe_allow_html=True)
    wl_cols = st.columns(2)
    for idx, wl_ticker in enumerate(st.session_state["watchlist"]):
        col = wl_cols[idx % 2]
        is_cur = wl_ticker == active_ticker
        btn_txt = f"● {wl_ticker}" if is_cur else wl_ticker
        if col.button(btn_txt, key=f"side_wl_{wl_ticker}", use_container_width=True):
            navigate_to_ticker(wl_ticker)

    st.divider()

    # Chart Mode
    st.markdown('<div class="mono-label">VISUALIZATION MODE</div>', unsafe_allow_html=True)
    chart_mode = st.radio(
        "Chart Style",
        options=["Line (MA7 + MA30)", "Candlestick (OHLC + MAs)"],
        label_visibility="collapsed",
    )

    st.divider()

    # Groq Config
    with st.expander("🔑 AI Advisor API Config", expanded=False):
        st.caption("Configured locally in `.streamlit/secrets.toml`. Enter an override key below if desired:")
        override_key = st.text_input("Override Groq Key:", type="password", placeholder="gsk_...")
        if override_key and override_key != st.session_state["user_groq_key"]:
            st.session_state["user_groq_key"] = override_key
            st.session_state["ai_cache"].clear()
            st.rerun()

    st.caption("StockPulse Pro • Built with Streamlit, pandas, scikit-learn & Groq")


# -----------------------------------------------------------------------------
# 6. Top Command Header
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

# Search & Navigation Controls Row
nav_col1, nav_col2, nav_col3 = st.columns([1.2, 4.5, 1.3])

with nav_col1:
    # Top Back Button
    if st.session_state["history"]:
        prev_sym = st.session_state["history"][-1]
        if st.button(f"⬅ Back ({prev_sym})", use_container_width=True):
            go_back()
    else:
        st.button("⬅ Back", disabled=True, use_container_width=True)

with nav_col2:
    typed_ticker = st.text_input(
        "Search Ticker Symbol:",
        value=active_ticker,
        placeholder="Enter symbol (e.g. RELIANCE.NS, TCS.NS, AAPL, NVDA)...",
        label_visibility="collapsed",
    ).strip().upper()

    if typed_ticker and typed_ticker != active_ticker:
        navigate_to_ticker(typed_ticker)

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
st.markdown('<div class="mono-label" style="margin-top: 0.3rem;">QUICK DISCOVERY CATEGORIES</div>', unsafe_allow_html=True)
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

st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)


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
    with st.spinner(f"Fetching market data for {active_ticker}..."):
        try:
            raw_data = get_cached_stock_data(active_ticker)
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
        # Bento Grid: Left Column (65%) | Right Column (35%)
        # ---------------------------------------------------------------------
        left_col, right_col = st.columns([65, 35], gap="medium")

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
                        <div class="mono-label">6-MONTH CHANGE</div>
                        <div class="metric-val {'delta-pos' if stats['period_change_pct'] >= 0 else 'delta-neg'}">
                            {stats['period_change_pct']:+.2f}%
                        </div>
                        <div class="metric-delta" style="color: #8B949E;">{stats['data_points']} Trading Days</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with m3:
                st.markdown(
                    f"""
                    <div class="bento-card">
                        <div class="mono-label">6-MONTH HIGH</div>
                        <div class="metric-val">{curr}{stats['period_high']:,.2f}</div>
                        <div class="metric-delta" style="color: #8B949E;">Peak Observed</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with m4:
                st.markdown(
                    f"""
                    <div class="bento-card">
                        <div class="mono-label">6-MONTH LOW</div>
                        <div class="metric-val">{curr}{stats['period_low']:,.2f}</div>
                        <div class="metric-delta" style="color: #8B949E;">Trough Observed</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Row 2: Interactive Plotly Chart (Clean Title & Non-Overlapping Legend)
            chart_header = get_ticker_display_name(active_ticker)
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
                        increasing_line_color="#00E676",
                        decreasing_line_color="#FF5252",
                    )
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=df_processed.index,
                        y=df_processed["Close"],
                        mode="lines",
                        name="Close Price",
                        line=dict(color="#2979FF", width=2.2),
                        hovertemplate=f"<b>Date:</b> %{{x|%Y-%m-%d}}<br><b>Close:</b> {curr}%{{y:,.2f}}<extra></extra>",
                    )
                )

            fig.add_trace(
                go.Scatter(
                    x=df_processed.index,
                    y=df_processed["MA7"],
                    mode="lines",
                    name="7-Day MA",
                    line=dict(color="#FF9933", width=1.6, dash="dash"),
                    hovertemplate=f"<b>MA7:</b> {curr}%{{y:,.2f}}<extra></extra>",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=df_processed.index,
                    y=df_processed["MA30"],
                    mode="lines",
                    name="30-Day MA",
                    line=dict(color="#00E676", width=1.6, dash="dot"),
                    hovertemplate=f"<b>MA30:</b> {curr}%{{y:,.2f}}<extra></extra>",
                )
            )

            fig.update_layout(
                paper_bgcolor="rgba(10, 13, 21, 0.7)",
                plot_bgcolor="rgba(10, 13, 21, 0.7)",
                title=dict(
                    text=f"<b>{chart_header} — 6-Month Technical Trend</b>",
                    x=0.01,
                    y=0.96,
                    font=dict(family="IBM Plex Mono", size=13, color="#FFFFFF"),
                ),
                xaxis=dict(
                    title=None,
                    showgrid=True,
                    gridcolor="rgba(255, 255, 255, 0.05)",
                    rangeslider=dict(visible=False),
                    tickfont=dict(color="#8B949E", size=10),
                ),
                yaxis=dict(
                    title=dict(text=f"Price ({curr})", font=dict(color="#8B949E", size=11)),
                    showgrid=True,
                    gridcolor="rgba(255, 255, 255, 0.05)",
                    tickfont=dict(color="#8B949E", size=10),
                ),
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.03,
                    xanchor="right",
                    x=1,
                    font=dict(color="#E6EDF3", size=10),
                ),
                margin=dict(l=15, r=15, t=65, b=25),
                height=450,
            )

            st.plotly_chart(fig, use_container_width=True)

            # Technical Momentum Note
            ma_msg = (
                "🟢 **Momentum Note:** Short-term 7-day average is tracking above the 30-day baseline."
                if stats["ma7"] > stats["ma30"]
                else "🔴 **Momentum Note:** Short-term 7-day average is tracking below the 30-day baseline."
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
                        <div style="margin-top: 0.5rem; font-size: 0.75rem; color: #8B949E; font-family: 'IBM Plex Mono', monospace;">
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
                        <div style="margin-top: 0.5rem; font-size: 0.75rem; color: #8B949E; font-family: 'IBM Plex Mono', monospace;">
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
        # Right Column: Modern Conversational AI Advisor (Bottom-Pinned Input)
        # ---------------------------------------------------------------------
        with right_col:
            st.markdown(
                f"""
                <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 0.4rem; margin-bottom: 0.8rem;">
                    <span style="font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem; font-weight: 600; color: {theme_accent};">
                        🧠 AI ADVISOR
                    </span>
                    <span style="font-size: 0.68rem; color: #00E676; font-family: 'IBM Plex Mono', monospace;">
                        ● GROUNDED RAG-LITE
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Ensure ticker-specific chat history exists
            if active_ticker not in st.session_state["chat_histories"]:
                st.session_state["chat_histories"][active_ticker] = []

            # Retrieve / Generate baseline insight for current stock
            if active_ticker not in st.session_state["ai_cache"]:
                with st.spinner(f"Analyzing {active_ticker}..."):
                    base_insight = get_ai_insight(rag_context)
                    st.session_state["ai_cache"][active_ticker] = base_insight
            else:
                base_insight = st.session_state["ai_cache"][active_ticker]

            # Scrollable Chat Container (Chat flows downwards)
            chat_container = st.container(height=490)

            with chat_container:
                # 1. Primary Grounded Analysis
                with st.chat_message("assistant", avatar="🧠"):
                    st.markdown(base_insight)

                # 2. Render all historical follow-up Q&A turns for this stock
                current_chat = st.session_state["chat_histories"][active_ticker]
                for msg in current_chat:
                    with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🧠"):
                        st.markdown(msg["content"])

            # 3. Bottom Chat Input (Always at bottom)
            user_question = st.chat_input(
                placeholder=f"Ask AI Advisor about {active_ticker}...",
                key=f"chat_in_{active_ticker}",
            )

            if user_question:
                # Add user message to history
                current_chat.append({"role": "user", "content": user_question})

                # Generate grounded response
                with st.spinner(f"Generating grounded answer for {active_ticker}..."):
                    answer = answer_followup(
                        context=rag_context,
                        chat_history=current_chat,
                        question=user_question,
                    )

                # Add assistant response to history
                current_chat.append({"role": "assistant", "content": answer})
                st.rerun()

            # Context Inspector Expander
            with st.expander("🔍 Inspect Retrieved Context (RAG Block)", expanded=False):
                st.caption("Factual context passed to Groq for strict grounding:")
                st.code(rag_context, language="text")

            st.caption("⚠️ **Educational Project:** Not financial advice. Past performance is non-guaranteed.")


# =============================================================================
# TAB 2: MULTI-ASSET COMPARATOR
# =============================================================================
with tab_comparator:
    st.markdown('<div class="mono-label">CROSS-ASSET PERFORMANCE BENCHMARK</div>', unsafe_allow_html=True)
    st.caption("Normalize and benchmark cumulative 6-month percentage returns across multiple equities simultaneously.")

    # Return to Workspace button
    if st.button("⬅ Return to Single Stock Workspace"):
        # Select active tab 0 via rerun or state
        st.session_state["selected_ticker"] = active_ticker
        st.rerun()

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
            palette = ["#2979FF", "#00E676", "#FF9933", "#D500F9", "#00B0FF", "#FFD600"]

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
                paper_bgcolor="rgba(10, 13, 21, 0.7)",
                plot_bgcolor="rgba(10, 13, 21, 0.7)",
                title=dict(
                    text="<b>6-Month Cumulative Growth (% from Baseline)</b>",
                    x=0.01,
                    font=dict(family="IBM Plex Mono", size=14, color="#FFFFFF"),
                ),
                xaxis=dict(showgrid=True, gridcolor="rgba(255, 255, 255, 0.05)", tickfont=dict(color="#8B949E", size=10)),
                yaxis=dict(
                    title=dict(text="Cumulative Return (%)", font=dict(color="#8B949E", size=11)),
                    showgrid=True,
                    gridcolor="rgba(255, 255, 255, 0.05)",
                    tickfont=dict(color="#8B949E", size=10),
                ),
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#E6EDF3", size=11)),
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
                    "6-Month Cumulative Return": f"{t_meta['total_return']:+.2f}%",
                    "Annualized Volatility": f"{t_meta['volatility']:.2f}%",
                })
            st.dataframe(pd.DataFrame(table_records), use_container_width=True, hide_index=True)
