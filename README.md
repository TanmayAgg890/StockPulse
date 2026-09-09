# 📈 StockPulse — Market Intelligence & Grounded AI Workspace

> **Bloomberg-Inspired Financial Analytics • Multi-Model Machine Learning • Grounded RAG-Lite AI Advisor**

StockPulse is a modern, dark-themed financial analytics workspace built with **Streamlit**, **Python**, **Plotly**, **scikit-learn**, and **Groq (Llama-3.1)**. It bridges the gap between empirical quantitative finance and modern generative AI by combining real-time market data, autoregressive machine learning models, and a **strictly grounded RAG-lite AI Advisor** that cannot hallucinate market statistics.

---

## 📸 Workspace Interface Preview

![StockPulse Dashboard](assets/screenshot.png)

---

## ✨ Features & Capabilities

### 1. 🏛️ Market Discovery & Categorization
* **India (NSE) Trending:** Quick-jump buttons for Indian mega-caps (`RELIANCE.NS`, `TCS.NS`, `HDFCBANK.NS`, `INFY.NS`, `ICICIBANK.NS`, `SBIN.NS`, `ITC.NS`, `LT.NS`).
* **US Tech & Popular:** Pre-configured discovery for `AAPL`, `MSFT`, `NVDA`, `GOOGL`, `AMZN`, `TSLA`, `META`.
* **Global & Domestic Indices:** Track benchmarks like Nifty 50 (`^NSEI`), BSE Sensex (`^BSESN`), S&P 500 (`^GSPC`), and NASDAQ (`^IXIC`).
* **Active Watchlist:** Session-based pinning and rapid switching across bookmarked equities.

### 2. 📊 Interactive Quantitative Visualizations
* **Chart Style Toggle:** Switch seamlessly between standard **Line Charts (with MAs)** and high-definition **OHLC Candlestick Charts**.
* **Technical Moving Averages:** Dynamically overlay **7-Day (MA7)** and **30-Day (MA30)** rolling averages with automated momentum relationship commentary.
* **KPI Metric Cards:** Live tracking of Current Close, 6-Month Period % Change, 6-Month High, and 6-Month Low.

### 3. 🤖 Multi-Model ML Benchmarking (Zero Lookahead Leakage)
* **Linear Regression (Baseline):** Parametric autoregressive model trained on 3-day historical price lags ($t-1, t-2, t-3$).
* **Random Forest Regressor (Ensemble):** Non-parametric ensemble of 100 decorrelated decision trees to handle non-linear volatility regimes.
* **Strict Chronological Train/Test Split:** 80% past data for training, 20% recent slice for testing—preventing future lookahead data leakage.
* **Model Diagnostics:** Reports Root Mean Squared Error (**RMSE** in currency units) and **$R^2$ Score**.

### 4. 🧠 Grounded AI Advisor (RAG-Lite)
* **Strict Factual Grounding:** The LLM does **not** compute numerical predictions or invent market facts. A structured factual context block is extracted from the data pipeline and fed into Groq's `llama-3.1-8b-instant`.
* **Zero-Key Grounded Mode:** Operates immediately out-of-the-box without requiring an API key via deterministic analytical synthesis.
* **Interactive Follow-Up Q&A:** Ask targeted questions about the stock's indicators or metrics. If an answer lies outside the retrieved context, the advisor politely refuses to speculate.
* **Context Transparency:** Includes an expandable *"Inspect Retrieved Context"* viewer showing the exact factual data supplied to the LLM.

### 5. ⚖️ Cross-Asset Performance Comparator
* Benchmark normalized 6-month percentage growth curves and annualized volatility across multiple equities simultaneously.

---

## 🛠️ Tech Stack

| Domain | Technology | Role |
| :--- | :--- | :--- |
| **Application & UI** | Python 3.10+, Streamlit | High-performance reactive web dashboard |
| **Data Ingestion** | yfinance | Historical daily OHLCV equity and index data |
| **Data Manipulation** | pandas, numpy | Rolling window statistics and lag feature creation |
| **Visualizations** | Plotly Graph Objects | Dark-themed Line and Candlestick interactive charts |
| **Machine Learning** | scikit-learn | Linear Regression, Random Forest Regressor, RMSE, $R^2$ |
| **Generative AI** | Groq SDK (`llama-3.1-8b-instant`) | Grounded market intelligence & follow-up Q&A |

---

## 🏗️ Architecture & Data Pipeline

```text
┌─────────────────────────────────────────────────────────────┐
│                    STOCKPULSE WORKSPACE                     │
│    Bento-Grid Dark UI • Ticker Discovery • Watchlist        │
└──────────────┬───────────────────────────────┬──────────────┘
               │                               │
               ▼                               ▼
    ┌──────────────────────┐        ┌──────────────────────┐
    │     DATA & ML        │        │      AI ADVISOR      │
    │     (model.py)       │        │   (ai_advisor.py)    │
    └──────────┬───────────┘        └──────────┬───────────┘
               │                               │
       ┌───────┴───────┐                       │
       ▼               ▼                       ▼
┌──────────────┐ ┌──────────────┐    ┌──────────────────┐
│ yfinance     │ │ scikit-learn │    │ Structured       │
│ OHLCV Data   │ │ ML Pipeline  │───>│ Context Builder  │
└──────────────┘ └──────────────┘    └─────────┬────────┘
                                               │
                                               ▼
                                     ┌──────────────────┐
                                     │ Groq Llama-3.1   │
                                     │ (Grounded RAG)   │
                                     └──────────────────┘
```

---

## 📁 Project Structure

```text
StockPulse/
├── app.py                     # Main dashboard, UI layout, session state, components
├── model.py                   # Data ingestion, MA computation, ML training & evaluation
├── ai_advisor.py              # Grounded context builder, Groq client, fallback engine
├── tickers.py                 # Categorized symbol directories (India, US, Global)
├── requirements.txt           # Project dependencies
├── README.md                  # Comprehensive documentation & architecture guide
├── .gitignore                 # Prevents committing venvs, caches, and secret keys
│
├── .streamlit/
│   └── secrets.toml.example   # Template for configuring free Groq API keys
│
├── assets/
│   └── screenshot.png         # High-resolution dashboard preview
│
└── tests/
    ├── test_model.py          # Unit tests for data pipeline and ML models
    └── test_ai_advisor.py     # Unit tests for RAG context building and fallbacks
```

---

## 🤖 Machine Learning Approach & Methodology

### 1. Autoregressive Lag Modeling
Standard supervised regressors require fixed tabular inputs. For each historical trading day $t$:
* $\text{Close\_Lag1} = \text{Close}_{t-1}$ (Previous trading day)
* $\text{Close\_Lag2} = \text{Close}_{t-2}$ (2 days prior)
* $\text{Close\_Lag3} = \text{Close}_{t-3}$ (3 days prior)
* $\text{Target} = \text{Close}_{t}$

To forecast unobserved trading day $T+1$ (tomorrow), the models ingest $[ \text{Close}_{T}, \text{Close}_{T-1}, \text{Close}_{T-2} ]$.

### 2. Time-Series Splitting & Data Leakage
Standard random splits (`train_test_split(shuffle=True)`) cause **lookahead bias**, allowing models to train on future observations to predict the past. StockPulse enforces a **strict chronological split**:
* First 80% of historical trading days $\rightarrow$ Training Set
* Final 20% of trading days $\rightarrow$ Out-of-sample Test Set

---

## 🧠 Grounded AI Advisor (RAG-Lite) Architecture

To prevent LLM hallucination:
1. **Computed Context Only:** The LLM receives a formatted text block containing only actual calculated numbers (Current Price, Period Change, Moving Averages, ML Predictions, RMSE, $R^2$).
2. **Zero Inventions:** The system prompt explicitly forbids inventing news, price targets, or financial events.
3. **Decoupled Predictions:** The AI Advisor never generates numerical forecasts; all numbers are calculated by scikit-learn.
4. **Zero-Key Mode:** If no Groq API key is present, the app falls back to deterministic rule-based synthesis without crashing.

---

## 💻 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/TanmayAgg890/StockPulse.git
cd StockPulse
```

### 2. Set Up Virtual Environment
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

### 4. Configure Groq API Key (Optional)
To enable live generative intelligence with Groq Llama-3.1:
1. Obtain a free API key at [https://console.groq.com/keys](https://console.groq.com/keys)
2. Create `.streamlit/secrets.toml`:
```toml
GROQ_API_KEY = "gsk_your_groq_api_key_here"
```
*(Or enter it directly into the app sidebar during runtime!)*

### 5. Run Automated Tests
```bash
python -m unittest discover tests/
```

### 6. Launch the Application
```bash
streamlit run app.py
```
Access the application at `http://localhost:8501`.

---

## ⚠️ Limitations

* **Non-Stationarity:** Stock prices exhibit random walk characteristics. Strong $R^2$ on short-term price lags primarily reflects price autocorrelation rather than guaranteed directional forecasting power.
* **Price-Only Scope:** The models currently analyze price lags and moving averages; macroeconomic interest rate decisions, earnings surprises, and supply chain shocks are not included.
* **Single-Step Forecasting:** Predictions are generated for the next immediate trading day. Multi-step compounding forecasts accumulate variance rapidly.

---

## ⚖️ Disclaimer

**Educational & Research Demonstration Only.** This software is an academic project and does not constitute financial, investment, or trading advice. Always perform independent due diligence before making financial decisions.
