# 📈 StockPulse

> **Stock Trend Visualizer & Machine Learning Next-Day Price Predictor**

StockPulse is an interactive financial analytics and educational machine learning dashboard built with **Streamlit**, **Python**, **Plotly**, and **scikit-learn**. It enables users to analyze ~6 months of historical daily stock data for both US and Indian equities, track key technical moving averages, and experiment with a baseline autoregressive linear regression model for next-day price prediction.

---

## 📸 Demo Preview

![App Screenshot](assets/screenshot.png)

---

## ✨ Features

* **Global & Indian Market Support:** Query any public equity symbol from Yahoo Finance (e.g., `AAPL`, `MSFT`, `GOOGL`, `TSLA`, `TCS.NS`, `RELIANCE.NS`, `INFY.NS`).
* **Key Financial Metrics:** Instant display of Current Close, 6-Month Period Change %, 6-Month High, and 6-Month Low.
* **Interactive Charting:** Dynamic Plotly visualization of Closing Price alongside **7-Day (MA7)** and **30-Day (MA30)** Moving Averages with unified hover tooltips and responsive zoom.
* **Autoregressive ML Prediction:** Supervised Linear Regression model predicting the next trading day's closing price based on 3-day historical price lags ($t-1, t-2, t-3$).
* **Chronological Model Diagnostics:** Transparent evaluation reporting **Root Mean Squared Error (RMSE)** and **$R^2$ Score** on an un-shuffled chronological test split (preventing future lookahead leakage).
* **Defensive Engineering & Caching:** Streamlit caching (`@st.cache_data`) for network optimization, with graceful handling of invalid tickers and network drops.
* **Educational Disclaimer:** Clear banner emphasizing experimental educational use—not financial advice.

---

## 🛠️ Tech Stack

| Technology | Purpose |
| :--- | :--- |
| **Python 3.10+** | Primary programming language |
| **Streamlit** | Reactive web application framework |
| **yfinance** | Historical financial market data ingestion |
| **pandas** | Time-series data wrangling, rolling calculations, and lag feature creation |
| **Plotly** | High-performance interactive visualizations |
| **scikit-learn** | Supervised Linear Regression modeling and metrics computation |

---

## 🔄 How It Works

Data flows linearly from user input to interactive prediction:

```text
User Enters Ticker (e.g., "AAPL", "TCS.NS")
               │
               ▼
      Yahoo Finance API
               │
               ▼
   Historical Daily OHLCV Data (~6 Months)
               │
               ▼
      Feature Engineering
   ├── 7-day & 30-day Moving Averages (MA7, MA30)
   └── Lag Features (t-1, t-2, t-3 Closing Prices)
               │
               ▼
   Supervised Linear Regression
   ├── Chronological Train/Test Split (80% Train, 20% Test)
   ├── Model Fitting (X_train -> y_train)
   └── Performance Evaluation (RMSE, R²)
               │
               ▼
      Next-Day Price Inference
               │
               ▼
     Interactive Streamlit UI
```

---

## 📁 Project Structure

```text
StockPulse/
├── app.py              # Frontend: Streamlit dashboard, Plotly charts, UI layout
├── model.py            # Backend: Data fetching, feature engineering, ML model, evaluation
├── requirements.txt    # Application dependencies
├── README.md           # Project documentation and interview guide
├── .gitignore          # Excludes virtual envs, caches, and IDE files
├── assets/
│   └── screenshot.png  # Application screenshot preview
└── tests/
    └── test_model.py   # Unit test suite verifying feature lags, predictions, and metrics
```

---

## 🤖 Machine Learning Approach

### 1. Autoregressive Lag Features
Financial time series cannot be fed into standard regression algorithms without transforming temporal sequences into supervised tabular pairs. For each trading day $t$:
* $\text{Close\_Lag1} = \text{Close}_{t-1}$ (Yesterday's close)
* $\text{Close\_Lag2} = \text{Close}_{t-2}$ (2 days ago)
* $\text{Close\_Lag3} = \text{Close}_{t-3}$ (3 days ago)
* $\text{Target} = \text{Close}_{t}$ (Today's close)

To predict unobserved trading day $T+1$ (tomorrow), the model takes $[ \text{Close}_{T}, \text{Close}_{T-1}, \text{Close}_{T-2} ]$.

### 2. Chronological Train/Test Split
Standard cross-validation randomly shuffles observations. Doing this in financial forecasting causes **lookahead bias (data leakage)**—the model trains on future data points to predict the past. StockPulse enforces a **strict chronological split**:
* First 80% of historical days $\rightarrow$ Training set
* Most recent 20% of historical days $\rightarrow$ Out-of-sample Test set

### 3. Evaluation Metrics
* **RMSE (Root Mean Squared Error):** Expressed directly in currency units, representing the average magnitude of prediction error.
* **$R^2$ Score (Coefficient of Determination):** Quantifies the proportion of variance explained by the lag features.

---

## ⚠️ Limitations

* **Non-Stationarity & Random Walk:** Stock prices frequently exhibit random-walk characteristics. High $R^2$ on short-term price lags often simply reflects yesterday's price persisting, rather than predictive power over market direction.
* **Absence of Exogenous Data:** The model currently relies exclusively on historical closing prices and ignores macroeconomic data, interest rates, company fundamentals, and earnings releases.
* **No Market Sentiment:** Market movements are heavily influenced by breaking news, social sentiment, and institutional order flow, which are not captured in price lags alone.
* **Linear Assumption:** Linear Regression cannot capture complex non-linear volatility regimes or cyclical market shocks.

---

## 🚀 Future Scope

### Beginner
- [ ] Incorporate trading volume trends and Moving Average Convergence Divergence (MACD).
- [ ] Add Relative Strength Index (RSI) momentum indicator.

### Intermediate
- [ ] Compare non-linear regressors (Random Forest Regressor, XGBoost).
- [ ] Multi-stock comparison dashboard (e.g., compare `AAPL` vs `MSFT`).
- [ ] Candlestick chart overlay option using Plotly.

### Advanced
- [ ] Recurrent Neural Networks (LSTM / GRU) for sequence learning.
- [ ] Real-time financial news sentiment extraction via FinBERT or LLMs.
- [ ] Automated backtesting simulation with risk metrics (Sharpe Ratio, Max Drawdown).

---

## 💻 Installation & Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/TanmayAgg890/StockPulse.git
cd StockPulse
```

### 2. Create and Activate Virtual Environment
**On Windows:**
```powershell
python -m venv venv
venv\Scripts\activate
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Automated Tests
```bash
python -m unittest tests/test_model.py
```

### 5. Launch the Streamlit App
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## ⚖️ Disclaimer

**This software is an educational and academic demonstration only.** It is not financial advice, trading advice, or a recommendation to buy or sell securities. Always conduct your own research before making financial investments.
