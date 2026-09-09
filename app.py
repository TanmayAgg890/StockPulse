"""
app.py — StockPulse: Market Intelligence & Grounded AI Analytics Workspace.

Architecture:
- Bento-Style Dark Technical Interface (Bloomberg / Modern AI OS aesthetic).
- Category Discovery (India Trending, US Popular, Global Indices) via tickers.py.
- Session-Based Watchlist & AI Caching.
- Multi-Model ML Forecasting (Linear Regression vs. Random Forest) with Chronological Evaluation.
- Grounded AI Advisor (RAG-lite) using Groq API with zero-key fallback.
- Interactive Plotly Visualizations (Line with Moving Averages & Candlestick OHLC).
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
    resolve_api_key,
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
# 2. Styling System: Dark, Restrained, Technical, Tactile
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background-color: #07090E;
        color: #E6EDF3;
    }

    /* Top Command Header */
    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 0.8rem;
        margin-bottom: 1.2rem;
    }
    .brand-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.4rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: #FFFFFF;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .brand-badge {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        background: rgba(255, 138, 76, 0.15);
        color: #FF8A4C;
        border: 1px solid rgba(255, 138, 76, 0.3);
        padding: 2px 6px;
        border-radius: 4px;
        letter-spacing: 0.5px;
    }
    .status-indicator {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        color: #00E676;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .pulse-dot {
        width: 8px;
        height: 8px;
        background-color: #00E676;
        border-radius: 50%;
        box-shadow: 0 0 8px #00E676;
        animation: pulse 2s infinite ease-in-out;
    }
    @keyframes pulse {
        0% { transform: scale(0.95); opacity: 0.7; }
        50% { transform: scale(1.2); opacity: 1; }
        100% { transform: scale(0.95); opacity: 0.7; }
    }

    /* Bento Panel Styling */
    .bento-card {
        background: #0D111A;
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 8px;
        padding: 1.1rem;
        margin-bottom: 1rem;
        transition: border 0.2s ease, transform 0.2s ease;
    }
    .bento-card:hover {
        border-color: rgba(255, 255, 255, 0.15);
    }
    .mono-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        color: #8B949E;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 0.3rem;
    }
    .metric-val {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.6rem;
        font-weight: 600;
        color: #FFFFFF;
        letter-spacing: -0.5px;
    }
    .metric-delta {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.82rem;
        font-weight: 500;
        margin-top: 0.2rem;
    }
    .delta-pos { color: #00E676; }
    .delta-neg { color: #FF5252; }

    /* AI Advisor Panel */
    .ai-panel {
        background: #0B0E17;
        border: 1px solid rgba(255, 138, 76, 0.25);
        border-radius: 8px;
        padding: 1.2rem;
        height: 100%;
    }
    .ai-header {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
        color: #FF8A4C;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.8rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        padding-bottom: 0.5rem;
    }
    .disclaimer-strip {
        font-size: 0.78rem;
        color: #8B949E;
        background: rgba(255, 255, 255, 0.02);
        border-left: 3px solid #FF8A4C;
        padding: 0.5rem 0.8rem;
        border-radius: 2px;
        margin-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 3. Session State Initialization
# -----------------------------------------------------------------------------
if "selected_ticker" not in st.session_state:
    st.session_state["selected_ticker"] = "AAPL"

if "watchlist" not in st.session_state:
    st.session_state["watchlist"] = list(DEFAULT_WATCHLIST)

if "ai_cache" not in st.session_state:
    st.session_state["ai_cache"] = {}

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "user_groq_key" not in st.session_state:
    st.session_state["user_groq_key"] = ""


# -----------------------------------------------------------------------------
# 4. Cached Data Pipelines
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_stock_data(ticker_symbol: str) -> pd.DataFrame:
    """Fetch and cache stock market data for 1 hour."""
    return get_stock_data(ticker_symbol)


@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_comparison(tickers: List[str]):
    """Fetch and cache multi-stock comparison returns."""
    return compare_stocks(tickers)


# -----------------------------------------------------------------------------
# 5. Sidebar: Watchlist & Settings
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="brand-title">📈 StockPulse</div>', unsafe_allow_html=True)
    st.caption("AI-Powered Financial Analytics Workspace")
    st.divider()

    # Watchlist Drawer
    st.markdown('<div class="mono-label">★ ACTIVE WATCHLIST</div>', unsafe_allow_html=True)
    watchlist_cols = st.columns(2)
    for i, w_ticker in enumerate(st.session_state["watchlist"]):
        col = watchlist_cols[i % 2]
        is_active = w_ticker == st.session_state["selected_ticker"]
        btn_label = f"▸ {w_ticker}" if is_active else w_ticker
        if col.button(btn_label, key=f"wl_{w_ticker}", use_container_width=True):
            st.session_state["selected_ticker"] = w_ticker
            st.rerun()

    st.divider()

    # Chart Setting
    st.markdown('<div class="mono-label">VISUALIZATION MODE</div>', unsafe_allow_html=True)
    chart_mode = st.radio(
        "Chart Style",
        options=["Line (MA7 + MA30)", "Candlestick (OHLC + MAs)"],
        label_visibility="collapsed",
    )

    st.divider()

    # Optional Groq API Key Configuration
    with st.expander("🔑 AI Advisor API Config", expanded=False):
        st.caption(
            "Free Llama-3.1 via Groq. If left blank, StockPulse runs in "
            "**Zero-Key Grounded Mode** with local factual synthesis."
        )
        groq_input = st.text_input(
            "Groq API Key:",
            value=st.session_state["user_groq_key"],
            type="password",
            placeholder="gsk_...",
        )
        if groq_input != st.session_state["user_groq_key"]:
            st.session_state["user_groq_key"] = groq_input
            # Clear AI cache to refresh with live LLM
            st.session_state["ai_cache"].clear()
            st.rerun()

    st.caption("StockPulse v2.0 • Built with Streamlit, pandas, scikit-learn & Groq")


# -----------------------------------------------------------------------------
# 6. Top Command Header & Discovery Area
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="header-container">
        <div class="brand-title">
            STOCKPULSE <span class="brand-badge">PRO EDITION</span>
        </div>
        <div class="status-indicator">
            <div class="pulse-dot"></div> DATA CONNECTION READY
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Search & Ticker Discovery Row
search_col, toggle_col = st.columns([3, 1])

with search_col:
    typed_ticker = st.text_input(
        "Search Ticker Symbol:",
        value=st.session_state["selected_ticker"],
        placeholder="Enter symbol (e.g. AAPL, NVDA, TCS.NS, RELIANCE.NS)...",
        label_visibility="collapsed",
    ).strip().upper()

    if typed_ticker and typed_ticker != st.session_state["selected_ticker"]:
        st.session_state["selected_ticker"] = typed_ticker
        st.rerun()

active_ticker = st.session_state["selected_ticker"]

with toggle_col:
    # Watchlist toggle for current ticker
    in_watchlist = active_ticker in st.session_state["watchlist"]
    wl_btn_text = "★ In Watchlist" if in_watchlist else "☆ Add to Watchlist"
    if st.button(wl_btn_text, use_container_width=True):
        if in_watchlist:
            st.session_state["watchlist"].remove(active_ticker)
        else:
            st.session_state["watchlist"].append(active_ticker)
        st.rerun()

# Category Discovery Buttons
st.markdown('<div class="mono-label" style="margin-top: 0.4rem;">QUICK DISCOVERY CATEGORIES</div>', unsafe_allow_html=True)
cat_tab1, cat_tab2, cat_tab3 = st.tabs(["🇮🇳 India Trending", "🇺🇸 US Popular", "🌍 Global Benchmarks"])

def render_category_buttons(items: List[Dict[str, str]], key_prefix: str):
    cols = st.columns(len(items))
    for idx, item in enumerate(items):
        sym = item["symbol"]
        is_cur = sym == active_ticker
        label = f"● {sym}" if is_cur else sym
        if cols[idx].button(label, key=f"{key_prefix}_{sym}", use_container_width=True):
            st.session_state["selected_ticker"] = sym
            st.rerun()

with cat_tab1:
    render_category_buttons(INDIA_TRENDING, "cat_ind")

with cat_tab2:
    render_category_buttons(US_POPULAR, "cat_us")

with cat_tab3:
    render_category_buttons(GLOBAL_INDICES, "cat_glob")

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 7. Main Navigation: Workspace Tabs
# -----------------------------------------------------------------------------
tab_workspace, tab_comparator = st.tabs([
    f"📊 {active_ticker} Workspace & Grounded AI",
    "⚖️ Multi-Asset Comparative Benchmarking",
])


# =============================================================================
# TAB 1: WORKSPACE (DATA + CHART + ML + GROUNDED AI ADVISOR)
# =============================================================================
with tab_workspace:
    # Data Fetching
    with st.spinner(f"Connecting to market data feed for {active_ticker}..."):
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
        # Preprocessing & Statistics
        df_processed = add_moving_averages(raw_data)
        stats = compute_market_stats(df_processed, active_ticker)
        curr = stats["currency"]

        # Run ML Models (Linear Regression & Random Forest)
        try:
            ml_predictions = train_and_predict_models(df_processed)
            ml_evals = evaluate_models(df_processed)
        except Exception as ml_err:
            st.warning(f"ML Pipeline Notice: {ml_err}")
            ml_predictions = {"Linear Regression": stats["current_price"], "Random Forest": stats["current_price"]}
            ml_evals = {
                "Linear Regression": {"rmse": 0.0, "r2": 0.0},
                "Random Forest": {"rmse": 0.0, "r2": 0.0},
            }

        # Build Grounded RAG Context
        rag_context = build_context(active_ticker, stats, ml_predictions, ml_evals)

        # ---------------------------------------------------------------------
        # 8. Bento Layout: Primary Workspace (Left 68%) + AI Advisor (Right 32%)
        # ---------------------------------------------------------------------
        left_col, right_col = st.columns([68, 32], gap="medium")

        with left_col:
            # Row 1: 4 Key Metric Cards
            k1, k2, k3, k4 = st.columns(4)

            with k1:
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

            with k2:
                st.markdown(
                    f"""
                    <div class="bento-card">
                        <div class="mono-label">6-MONTH CHANGE</div>
                        <div class="metric-val {'delta-pos' if stats['period_change_pct'] >= 0 else 'delta-neg'}">
                            {stats['period_change_pct']:+.2f}%
                        </div>
                        <div class="metric-delta" style="color: #8B949E;">Over {stats['data_points']} Trading Days</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with k3:
                st.markdown(
                    f"""
                    <div class="bento-card">
                        <div class="mono-label">6-MONTH HIGH</div>
                        <div class="metric-val">{curr}{stats['period_high']:,.2f}</div>
                        <div class="metric-delta" style="color: #8B949E;">Peak Observed Price</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with k4:
                st.markdown(
                    f"""
                    <div class="bento-card">
                        <div class="mono-label">6-MONTH LOW</div>
                        <div class="metric-val">{curr}{stats['period_low']:,.2f}</div>
                        <div class="metric-delta" style="color: #8B949E;">Trough Observed Price</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Row 2: Interactive Plotly Chart
            chart_title = f"{get_ticker_display_name(active_ticker)} — 6-Month Technical Trend"
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

            # Technical Moving Averages
            fig.add_trace(
                go.Scatter(
                    x=df_processed.index,
                    y=df_processed["MA7"],
                    mode="lines",
                    name="7-Day MA",
                    line=dict(color="#FF8A4C", width=1.6, dash="dash"),
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
                paper_bgcolor="#0A0D15",
                plot_bgcolor="#0A0D15",
                title=dict(
                    text=f"<b>{chart_title}</b>",
                    x=0.01,
                    font=dict(family="IBM Plex Mono", size=15, color="#FFFFFF"),
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
                    y=1.02,
                    xanchor="right",
                    x=1,
                    font=dict(color="#E6EDF3", size=11),
                ),
                margin=dict(l=15, r=15, t=50, b=25),
                height=460,
            )

            st.plotly_chart(fig, use_container_width=True)

            # Momentum Interpretation Banner
            ma_msg = (
                "🟢 **Technical Momentum Note:** 7-Day Moving Average is tracking above the 30-Day average, "
                "suggesting stronger short-term price momentum."
                if stats["ma7"] > stats["ma30"]
                else "🔴 **Technical Momentum Note:** 7-Day Moving Average is tracking below the 30-Day average, "
                "suggesting softening short-term momentum."
            )
            st.caption(ma_msg)

            # Row 3: ML Model Benchmarking (Linear Regression vs. Random Forest)
            st.markdown('<div class="mono-label" style="margin-top: 1rem;">AUTOREGRESSIVE PREDICTION BENCHMARK</div>', unsafe_allow_html=True)
            ml_col1, ml_col2 = st.columns(2)

            lr_p = ml_predictions["Linear Regression"]
            lr_delta = lr_p - stats["current_price"]
            lr_delta_pct = (lr_delta / stats["current_price"]) * 100
            lr_rmse = ml_evals["Linear Regression"]["rmse"]
            lr_r2 = ml_evals["Linear Regression"]["r2"]

            with ml_col1:
                st.markdown(
                    f"""
                    <div class="bento-card">
                        <div class="mono-label">🔹 LINEAR REGRESSION (BASELINE)</div>
                        <div class="metric-val">{curr}{lr_p:,.2f}</div>
                        <div class="metric-delta {'delta-pos' if lr_delta >= 0 else 'delta-neg'}">
                            {lr_delta:+,.2f} ({lr_delta_pct:+.2f}%)
                        </div>
                        <div style="margin-top: 0.6rem; font-size: 0.8rem; color: #8B949E; font-family: 'IBM Plex Mono', monospace;">
                            Test RMSE: {curr}{lr_rmse:.2f} &nbsp;|&nbsp; R² Score: {lr_r2:.4f}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            rf_p = ml_predictions["Random Forest"]
            rf_delta = rf_p - stats["current_price"]
            rf_delta_pct = (rf_delta / stats["current_price"]) * 100
            rf_rmse = ml_evals["Random Forest"]["rmse"]
            rf_r2 = ml_evals["Random Forest"]["r2"]

            with ml_col2:
                st.markdown(
                    f"""
                    <div class="bento-card">
                        <div class="mono-label">🌲 RANDOM FOREST (100 TREES)</div>
                        <div class="metric-val">{curr}{rf_p:,.2f}</div>
                        <div class="metric-delta {'delta-pos' if rf_delta >= 0 else 'delta-neg'}">
                            {rf_delta:+,.2f} ({rf_delta_pct:+.2f}%)
                        </div>
                        <div style="margin-top: 0.6rem; font-size: 0.8rem; color: #8B949E; font-family: 'IBM Plex Mono', monospace;">
                            Test RMSE: {curr}{rf_rmse:.2f} &nbsp;|&nbsp; R² Score: {rf_r2:.4f}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # ML Transparency Expander
            with st.expander("🔬 Model Transparency & Engineering Rationale", expanded=False):
                st.markdown(
                    """
                    **How is this prediction made?**
                    - **Feature Construction:** StockPulse creates 3 autoregressive lag features from closing prices:
                      `Close_Lag1` ($t-1$), `Close_Lag2` ($t-2$), and `Close_Lag3` ($t-3$).
                    - **Zero Data Leakage:** The evaluation strictly adheres to an un-shuffled chronological split
                      (first 80% of days for training, most recent 20% for testing). Shuffling time-series observations
                      causes lookahead bias.
                    - **LLM Decoupling:** The Large Language Model (Groq / Llama) **does not compute** numerical predictions.
                      Numerical forecasts are computed purely by scikit-learn statistical regressors.
                    """
                )

        # ---------------------------------------------------------------------
        # 9. Right Column: Grounded AI Advisor (RAG-Lite)
        # ---------------------------------------------------------------------
        with right_col:
            st.markdown(
                """
                <div class="ai-header">
                    <span>🧠 AI ADVISOR</span>
                    <span style="font-size: 0.7rem; color: #00E676;">● GROUNDED RAG-LITE</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Retrieve or generate AI insight (with session-based caching)
            if active_ticker not in st.session_state["ai_cache"]:
                with st.spinner("Synthesizing grounded market intelligence..."):
                    insight_text = get_ai_insight(rag_context)
                    st.session_state["ai_cache"][active_ticker] = insight_text
            else:
                insight_text = st.session_state["ai_cache"][active_ticker]

            # Render Formatted AI Analysis
            st.markdown(insight_text)

            # Grounded Context Inspection Expander
            with st.expander("🔍 Inspect Retrieved Context (RAG Block)", expanded=False):
                st.caption("This exact structured context block was passed to the AI model to guarantee factual grounding:")
                st.code(rag_context, language="text")

            st.markdown('<div class="mono-label" style="margin-top: 1.2rem;">💬 ASK ABOUT THIS ANALYSIS</div>', unsafe_allow_html=True)
            user_q = st.text_input(
                "Ask a follow-up question:",
                placeholder="e.g. Why is MA7 above MA30? What does RMSE mean?",
                label_visibility="collapsed",
                key=f"q_{active_ticker}",
            )

            if user_q:
                with st.spinner("Formulating grounded response..."):
                    ans = answer_followup(rag_context, st.session_state["chat_history"], user_q)
                    st.session_state["chat_history"].append({"role": "user", "content": user_q})
                    st.session_state["chat_history"].append({"role": "assistant", "content": ans})

                st.markdown(f"**You:** {user_q}")
                st.markdown(f"**Advisor:** {ans}")

            st.markdown(
                """
                <div class="disclaimer-strip">
                    ⚠️ <b>Educational Project:</b> Not financial advice. Past price patterns do not guarantee future trajectory.
                </div>
                """,
                unsafe_allow_html=True,
            )


# =============================================================================
# TAB 2: MULTI-ASSET COMPARATOR
# =============================================================================
with tab_comparator:
    st.markdown('<div class="mono-label">CROSS-ASSET PERFORMANCE BENCHMARK</div>', unsafe_allow_html=True)
    st.caption("Normalize and benchmark cumulative 6-month percentage returns across multiple equities simultaneously.")

    comp_selection = st.multiselect(
        "Select Assets to Benchmark:",
        options=[
            "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "TSLA", "META",
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        ],
        default=["AAPL", "MSFT", "NVDA"],
    )

    if len(comp_selection) < 2:
        st.info("Select at least 2 tickers above to generate cross-asset performance curves.")
    else:
        with st.spinner("Synthesizing multi-asset comparison..."):
            ret_df, comp_stats = get_cached_comparison(comp_selection)

        if not ret_df.empty:
            c_fig = go.Figure()
            palette = ["#2979FF", "#00E676", "#FF8A4C", "#D500F9", "#00B0FF", "#FFD600"]

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
                paper_bgcolor="#0A0D15",
                plot_bgcolor="#0A0D15",
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

            # Asset Comparison Summary Table
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
