"""
app.py — Interactive Streamlit Dashboard for StockPulse.

Features:
- Search bar for global and Indian market tickers.
- Top KPI summary metrics (Current Price, Period % Change, 6-Month High/Low).
- Interactive Plotly time-series chart with 7-day and 30-day Moving Averages.
- Machine Learning panel displaying next-day price forecast and delta.
- Educational diagnostics panel explaining RMSE and R2 scores.
- Comprehensive defensive error handling without ugly tracebacks.
"""

from typing import Tuple
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from model import (
    get_stock_data,
    add_moving_averages,
    train_and_predict,
    evaluate_model,
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

# Custom CSS for styling cards and layout
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
        margin-bottom: 1.5rem;
    }
    .disclaimer-box {
        background-color: rgba(255, 179, 0, 0.1);
        border-left: 4px solid #ffb300;
        padding: 0.8rem 1.2rem;
        border-radius: 4px;
        margin-top: 2rem;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 2. Cached Data Fetching
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_cached_stock_data(ticker_symbol: str) -> pd.DataFrame:
    """
    Fetch and cache stock market data for 1 hour to optimize performance
    and avoid unnecessary duplicate network requests.
    """
    return get_stock_data(ticker_symbol)


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
    st.header("⚙️ Configuration")
    ticker_input = st.text_input(
        "Enter Stock Ticker:",
        value="AAPL",
        help="Type any public ticker symbol (e.g. AAPL, MSFT, TCS.NS, RELIANCE.NS)",
    ).strip().upper()

    st.markdown("**Popular Ticker Examples:**")
    st.markdown(
        """
        - 🇺🇸 `AAPL` (Apple)
        - 🇺🇸 `MSFT` (Microsoft)
        - 🇺🇸 `GOOGL` (Alphabet)
        - 🇺🇸 `TSLA` (Tesla)
        - 🇮🇳 `TCS.NS` (Tata Consultancy Services)
        - 🇮🇳 `RELIANCE.NS` (Reliance Industries)
        - 🇮🇳 `INFY.NS` (Infosys)
        - 🇮🇳 `HDFCBANK.NS` (HDFC Bank)
        """
    )

    st.divider()
    st.caption("Developed with Streamlit, pandas, scikit-learn, and Plotly.")


# -----------------------------------------------------------------------------
# 4. Main Header
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">📈 StockPulse</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Stock Trend Visualizer & Machine Learning Next-Day Price Predictor</div>',
    unsafe_allow_html=True,
)

if not ticker_input:
    st.info("👋 Please enter a stock ticker in the sidebar to get started.")
    st.stop()

# -----------------------------------------------------------------------------
# 5. Data Retrieval & Processing
# -----------------------------------------------------------------------------
with st.spinner(f"Fetching market data for {ticker_input}..."):
    try:
        raw_df = fetch_cached_stock_data(ticker_input)
    except ValueError as val_err:
        st.error(f"❌ **Invalid Ticker / Data Issue:** {val_err}")
        st.stop()
    except ConnectionError as conn_err:
        st.error(f"🌐 **Network Error:** {conn_err}")
        st.stop()
    except Exception as general_err:
        st.error(f"⚠️ An unexpected error occurred while fetching data: {general_err}")
        st.stop()

# Calculate Moving Averages
df = add_moving_averages(raw_df)
currency = get_currency_symbol(ticker_input)

# -----------------------------------------------------------------------------
# 6. Key Market Metrics (4 Columns)
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# 7. Interactive Price Chart (Plotly)
# -----------------------------------------------------------------------------
st.subheader("📊 Market Trend & Moving Averages")

fig = go.Figure()

# Close Price Line
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

# 7-Day Moving Average
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

# 30-Day Moving Average
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
        text=f"<b>{ticker_input}</b> — 6-Month Price Trend with Moving Averages",
        x=0.01,
        font=dict(size=18),
    ),
    xaxis=dict(
        title="Date",
        showgrid=True,
        gridcolor="rgba(128, 128, 128, 0.2)",
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
    height=480,
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# -----------------------------------------------------------------------------
# 8. Machine Learning: Prediction & Diagnostics
# -----------------------------------------------------------------------------
pred_col, eval_col = st.columns([1, 1], gap="medium")

with pred_col:
    st.subheader("🤖 Next-Day Price Prediction")
    st.caption("Autoregressive Linear Regression trained on 3-day historical price lags.")

    try:
        predicted_price = train_and_predict(df)
        pred_delta = predicted_price - current_price
        pred_delta_pct = (pred_delta / current_price) * 100

        st.metric(
            label="Predicted Next-Day Close",
            value=f"{currency}{predicted_price:,.2f}",
            delta=f"{pred_delta:+,.2f} ({pred_delta_pct:+.2f}%)",
        )

        if pred_delta > 0:
            st.success(f"📈 Model projects a potential upward movement of **{pred_delta_pct:+.2f}%**.")
        else:
            st.warning(f"📉 Model projects a potential downward movement of **{pred_delta_pct:+.2f}%**.")

    except Exception as ml_err:
        st.error(f"Could not compute prediction: {ml_err}")

with eval_col:
    st.subheader("🧪 Model Diagnostics")
    st.caption("Evaluated on an un-shuffled chronological test split (most recent 20%).")

    try:
        metrics = evaluate_model(df)
        metric_col1, metric_col2 = st.columns(2)

        with metric_col1:
            st.metric(
                label="RMSE (Typical Error)",
                value=f"{currency}{metrics['rmse']:.2f}",
                help="Root Mean Squared Error measures the average prediction discrepancy in price units.",
            )
        with metric_col2:
            st.metric(
                label="R² Score (Fit)",
                value=f"{metrics['r2']:.4f}",
                help="Coefficient of Determination measures the proportion of variance explained by lag features.",
            )

        st.info(
            f"💡 **Interpretation:** On the recent test period, predictions differed from actual closes by an "
            f"average of **{currency}{metrics['rmse']:.2f}**. An R² of **{metrics['r2']:.4f}** indicates strong "
            f"short-term autocorrelation, though this alone does not guarantee future profitability."
        )

    except Exception as eval_err:
        st.error(f"Could not compute evaluation: {eval_err}")

# -----------------------------------------------------------------------------
# 9. Educational Disclaimer
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="disclaimer-box">
        ⚠️ <b>Educational & Research Demonstration Only:</b> This application is built for machine learning 
        portfolio and academic purposes. Stock markets are non-stationary and influenced by complex macroeconomic 
        factors, breaking news, order book dynamics, and sentiment. Linear Regression on price lags cannot reliably 
        forecast market movements. <b>Never base actual investment decisions on these predictions.</b>
    </div>
    """,
    unsafe_allow_html=True,
)
