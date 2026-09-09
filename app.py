"""
app.py — Interactive Streamlit Dashboard for StockPulse.

Features:
- Single Stock Analysis with interactive Line and Candlestick (OHLC) charting.
- 7-Day and 30-Day Moving Average trend overlays.
- Multi-Model Machine Learning: Side-by-side comparison between Linear Regression and Random Forest Regressor.
- Chronological Model Diagnostics (RMSE, R2) for both models without lookahead leakage.
- Multi-Stock Performance Comparator: Compare cumulative percentage returns across multiple assets.
- Defensive error handling and Streamlit caching.
"""

from typing import List
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from model import (
    get_stock_data,
    add_moving_averages,
    train_and_predict_models,
    evaluate_models,
    compare_stocks,
)

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="StockPulse — Stock Trend Visualizer & Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #888888;
        font-size: 1.05rem;
        margin-bottom: 1.2rem;
    }
    .disclaimer-box {
        background-color: rgba(255, 179, 0, 0.08);
        border-left: 4px solid #ffb300;
        padding: 0.8rem 1.2rem;
        border-radius: 4px;
        margin-top: 2rem;
        font-size: 0.9rem;
    }
    .model-card {
        background-color: rgba(128, 128, 128, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 2. Cached Helper Functions
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_cached_stock_data(ticker_symbol: str) -> pd.DataFrame:
    """Fetch and cache stock market data for 1 hour."""
    return get_stock_data(ticker_symbol)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_cached_stock_comparison(tickers: List[str]):
    """Fetch and cache multi-stock comparison returns."""
    return compare_stocks(tickers)


def get_currency_symbol(ticker: str) -> str:
    """Determine representative currency symbol based on exchange suffix."""
    ticker_upper = ticker.strip().upper()
    if ticker_upper.endswith(".NS") or ticker_upper.endswith(".BO"):
        return "₹"
    elif ticker_upper.endswith(".L"):
        return "£"
    elif ticker_upper.endswith(".TO"):
        return "C$"
    elif ticker_upper.endswith(".DE") or ticker_upper.endswith(".PA"):
        return "€"
    return "$"


# -----------------------------------------------------------------------------
# 3. Sidebar Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    ticker_input = st.text_input(
        "Primary Stock Ticker:",
        value="AAPL",
        help="Type any public ticker symbol (e.g. AAPL, MSFT, TCS.NS, RELIANCE.NS)",
    ).strip().upper()

    chart_type = st.radio(
        "Chart Style:",
        options=["Line Chart (with MAs)", "Candlestick (OHLC with MAs)"],
        index=0,
    )

    st.divider()
    st.markdown("**Popular Ticker Examples:**")
    st.markdown(
        """
        - 🇺🇸 `AAPL` (Apple)
        - 🇺🇸 `MSFT` (Microsoft)
        - 🇺🇸 `GOOGL` (Alphabet)
        - 🇺🇸 `TSLA` (Tesla)
        - 🇮🇳 `TCS.NS` (Tata Consultancy)
        - 🇮🇳 `RELIANCE.NS` (Reliance)
        - 🇮🇳 `INFY.NS` (Infosys)
        - 🇮🇳 `HDFCBANK.NS` (HDFC Bank)
        """
    )
    st.divider()
    st.caption("Developed with Streamlit, pandas, scikit-learn, and Plotly.")


# -----------------------------------------------------------------------------
# 4. Main Header & Navigation Tabs
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">📈 StockPulse</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Stock Trend Visualizer & Machine Learning Next-Day Price Predictor</div>',
    unsafe_allow_html=True,
)

tab_single, tab_compare = st.tabs(["📊 Single Stock Analysis & ML", "⚖️ Multi-Stock Performance Comparator"])


# =============================================================================
# TAB 1: SINGLE STOCK ANALYSIS & MULTI-MODEL PREDICTIONS
# =============================================================================
with tab_single:
    if not ticker_input:
        st.info("👋 Please enter a stock ticker in the sidebar to get started.")
    else:
        with st.spinner(f"Fetching market data for {ticker_input}..."):
            try:
                raw_df = fetch_cached_stock_data(ticker_input)
            except ValueError as val_err:
                st.error(f"❌ **Invalid Ticker / Data Issue:** {val_err}")
                raw_df = None
            except ConnectionError as conn_err:
                st.error(f"🌐 **Network Error:** {conn_err}")
                raw_df = None
            except Exception as general_err:
                st.error(f"⚠️ An unexpected error occurred: {general_err}")
                raw_df = None

        if raw_df is not None:
            df = add_moving_averages(raw_df)
            currency = get_currency_symbol(ticker_input)

            # --- Key Market Metrics ---
            current_price = float(df["Close"].iloc[-1])
            first_price = float(df["Close"].iloc[0])
            period_change = current_price - first_price
            period_change_pct = (period_change / first_price) * 100
            period_high = float(df["High"].max())
            period_low = float(df["Low"].min())

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    label="Current Close",
                    value=f"{currency}{current_price:,.2f}",
                    delta=f"{period_change:+,.2f} ({period_change_pct:+.2f}%)",
                )
            with col2:
                st.metric(
                    label="6-Month Period Change",
                    value=f"{period_change_pct:+.2f}%",
                    delta="Past 6 Months",
                    delta_color="off",
                )
            with col3:
                st.metric(
                    label="6-Month High",
                    value=f"{currency}{period_high:,.2f}",
                )
            with col4:
                st.metric(
                    label="6-Month Low",
                    value=f"{currency}{period_low:,.2f}",
                )

            st.divider()

            # --- Interactive Chart (Line vs. Candlestick) ---
            st.subheader(f"📊 {ticker_input} — Market Trend & Moving Averages")

            fig = go.Figure()

            if chart_type == "Candlestick (OHLC with MAs)":
                # Candlestick Trace
                fig.add_trace(
                    go.Candlestick(
                        x=df.index,
                        open=df["Open"],
                        high=df["High"],
                        low=df["Low"],
                        close=df["Close"],
                        name="OHLC Candlestick",
                        increasing_line_color="#26A69A",
                        decreasing_line_color="#EF5350",
                    )
                )
            else:
                # Line Trace
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df["Close"],
                        mode="lines",
                        name="Closing Price",
                        line=dict(color="#2962FF", width=2.2),
                        hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Close:</b> "
                        + currency
                        + "%{y:,.2f}<extra></extra>",
                    )
                )

            # Moving Averages overlays
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["MA7"],
                    mode="lines",
                    name="7-Day MA",
                    line=dict(color="#FF6D00", width=1.6, dash="dash"),
                    hovertemplate="<b>7-Day MA:</b> " + currency + "%{y:,.2f}<extra></extra>",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["MA30"],
                    mode="lines",
                    name="30-Day MA",
                    line=dict(color="#00C853", width=1.6, dash="dot"),
                    hovertemplate="<b>30-Day MA:</b> " + currency + "%{y:,.2f}<extra></extra>",
                )
            )

            fig.update_layout(
                title=dict(
                    text=f"<b>{ticker_input}</b> — 6-Month Trend with 7-Day & 30-Day Moving Averages",
                    x=0.01,
                    font=dict(size=18),
                ),
                xaxis=dict(
                    title="Date",
                    showgrid=True,
                    gridcolor="rgba(128, 128, 128, 0.2)",
                    rangeslider=dict(visible=False),
                ),
                yaxis=dict(
                    title=f"Price ({currency})",
                    showgrid=True,
                    gridcolor="rgba(128, 128, 128, 0.2)",
                ),
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                ),
                margin=dict(l=20, r=20, t=60, b=30),
                height=500,
            )

            st.plotly_chart(fig, use_container_width=True)

            st.divider()

            # --- Multi-Model Machine Learning Predictions ---
            st.subheader("🤖 Next-Day Price Predictions (Multi-Model Comparison)")
            st.caption(
                "Comparing a linear baseline (Linear Regression) against a non-linear ensemble (Random Forest Regressor) "
                "trained on historical 3-day price lags."
            )

            try:
                preds = train_and_predict_models(df)
                evals = evaluate_models(df)

                m_col1, m_col2 = st.columns(2, gap="large")

                # Model 1: Linear Regression
                with m_col1:
                    lr_pred = preds["Linear Regression"]
                    lr_delta = lr_pred - current_price
                    lr_delta_pct = (lr_delta / current_price) * 100
                    lr_rmse = evals["Linear Regression"]["rmse"]
                    lr_r2 = evals["Linear Regression"]["r2"]

                    st.markdown("### 🔹 Linear Regression (Baseline)")
                    st.metric(
                        label="Predicted Next-Day Close",
                        value=f"{currency}{lr_pred:,.2f}",
                        delta=f"{lr_delta:+,.2f} ({lr_delta_pct:+.2f}%)",
                    )
                    sub1, sub2 = st.columns(2)
                    sub1.metric("Chronological RMSE", f"{currency}{lr_rmse:,.2f}")
                    sub2.metric("R² Score", f"{lr_r2:.4f}")
                    st.caption("Fast parametric model assuming a linear relationship across consecutive price lags.")

                # Model 2: Random Forest
                with m_col2:
                    rf_pred = preds["Random Forest"]
                    rf_delta = rf_pred - current_price
                    rf_delta_pct = (rf_delta / current_price) * 100
                    rf_rmse = evals["Random Forest"]["rmse"]
                    rf_r2 = evals["Random Forest"]["r2"]

                    st.markdown("### 🌲 Random Forest Regressor (Ensemble)")
                    st.metric(
                        label="Predicted Next-Day Close",
                        value=f"{currency}{rf_pred:,.2f}",
                        delta=f"{rf_delta:+,.2f} ({rf_delta_pct:+.2f}%)",
                    )
                    sub3, sub4 = st.columns(2)
                    sub3.metric("Chronological RMSE", f"{currency}{rf_rmse:,.2f}")
                    sub4.metric("R² Score", f"{rf_r2:.4f}")
                    st.caption("Non-parametric ensemble of 100 decision trees capable of capturing non-linear regimes.")

                # Comparative Takeaway
                if rf_rmse < lr_rmse:
                    better_model = "🌲 Random Forest"
                    diff = lr_rmse - rf_rmse
                else:
                    better_model = "🔹 Linear Regression"
                    diff = rf_rmse - lr_rmse

                st.info(
                    f"💡 **Model Comparison Summary:** On the chronological test slice (last 20% of trading days), "
                    f"**{better_model}** demonstrated lower test error by **{currency}{diff:.2f} RMSE**. "
                    "Notice that ensemble trees can bound extremes and avoid runaway linear extrapolation."
                )

            except Exception as ml_err:
                st.error(f"Could not compute multi-model forecasts: {ml_err}")


# =============================================================================
# TAB 2: MULTI-STOCK PERFORMANCE COMPARATOR
# =============================================================================
with tab_compare:
    st.subheader("⚖️ Multi-Stock Cumulative Return Comparator")
    st.caption("Normalize and compare historical 6-month growth percentages across multiple equities simultaneously.")

    default_tickers = ["AAPL", "MSFT", "GOOGL", "TSLA"]
    selected_tickers = st.multiselect(
        "Select Tickers to Compare:",
        options=["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "AMZN", "TCS.NS", "RELIANCE.NS", "INFY.NS", "HDFCBANK.NS"],
        default=["AAPL", "MSFT", "GOOGL"],
    )

    custom_ticker = st.text_input("Or add any custom ticker (e.g. NVDA, META):").strip().upper()
    if custom_ticker and custom_ticker not in selected_tickers:
        selected_tickers.append(custom_ticker)

    if len(selected_tickers) < 2:
        st.warning("⚠️ Please select at least 2 tickers to generate comparative performance charts.")
    else:
        with st.spinner("Fetching comparison data..."):
            returns_df, stats = fetch_cached_stock_comparison(selected_tickers)

        if not returns_df.empty:
            # Comparative Return Chart
            comp_fig = go.Figure()
            colors = ["#2962FF", "#00C853", "#FF6D00", "#D500F9", "#00B0FF", "#FFD600"]

            for idx, col in enumerate(returns_df.columns):
                comp_fig.add_trace(
                    go.Scatter(
                        x=returns_df.index,
                        y=returns_df[col],
                        mode="lines",
                        name=col,
                        line=dict(width=2.2, color=colors[idx % len(colors)]),
                        hovertemplate=f"<b>{col}:</b> %{{y:+.2f}}%<extra></extra>",
                    )
                )

            comp_fig.update_layout(
                title=dict(
                    text="<b>6-Month Cumulative Percentage Returns (% Growth from Day 0)</b>",
                    x=0.01,
                    font=dict(size=16),
                ),
                xaxis=dict(title="Date", showgrid=True, gridcolor="rgba(128, 128, 128, 0.2)"),
                yaxis=dict(title="Cumulative Return (%)", showgrid=True, gridcolor="rgba(128, 128, 128, 0.2)"),
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=20, r=20, t=50, b=30),
                height=460,
            )

            st.plotly_chart(comp_fig, use_container_width=True)

            # Comparative Statistics Table
            st.markdown("### 📋 Asset Comparison Summary")
            stat_rows = []
            for t, s in stats.items():
                stat_rows.append({
                    "Ticker": t,
                    "Current Price": f"{s['current_price']:,.2f}",
                    "6-Month Total Return": f"{s['total_return']:+.2f}%",
                    "Annualized Volatility": f"{s['volatility']:.2f}%",
                })

            if stat_rows:
                stats_df = pd.DataFrame(stat_rows)
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
        else:
            st.error("Could not fetch data for the selected ticker set. Please verify the symbols.")


# -----------------------------------------------------------------------------
# 5. Educational Disclaimer
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="disclaimer-box">
        ⚠️ <b>Educational & Research Demonstration Only:</b> This application is built for machine learning 
        portfolio and academic purposes. Stock markets are non-stationary and influenced by complex macroeconomic 
        factors, breaking news, order book dynamics, and sentiment. Linear and tree regression models on price lags 
        cannot reliably forecast real-world market movements. <b>Never base actual investment decisions on these predictions.</b>
    </div>
    """,
    unsafe_allow_html=True,
)
