"""
model.py — Machine Learning and Data Logic Layer for StockPulse.

Responsibilities:
- Fetch historical market data from Yahoo Finance via yfinance.
- Clean and validate incoming financial time-series.
- Compute technical indicators (7-day & 30-day Moving Averages).
- Construct lag features for autoregressive supervised learning.
- Train baseline Linear Regression and Random Forest Regressors with chronological splits.
- Multi-model price forecasting, evaluation, and cross-asset comparison.
- Statistical summarization for UI and Grounded AI Context.
"""

from typing import Dict, Tuple, List, Any
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


def get_stock_data(ticker: str, period: str = "6mo") -> pd.DataFrame:
    """
    Fetch historical daily stock data for a given ticker symbol.

    Args:
        ticker: Stock symbol (e.g., 'AAPL', 'MSFT', 'TCS.NS').
        period: Historical lookback duration (default: '6mo').

    Returns:
        pd.DataFrame: Historical OHLCV data with DatetimeIndex.

    Raises:
        ValueError: If ticker is invalid, delisted, or returns no data.
        ConnectionError: If network or data provider request fails.
    """
    clean_ticker = ticker.strip().upper()
    if not clean_ticker:
        raise ValueError("Ticker symbol cannot be empty.")

    try:
        stock = yf.Ticker(clean_ticker)
        df = stock.history(period=period)
    except Exception as exc:
        raise ConnectionError(f"Failed to fetch data for '{clean_ticker}': {exc}") from exc

    if df is None or df.empty:
        raise ValueError(
            f"No market data found for ticker '{clean_ticker}'. "
            "Please check if the symbol is valid and active on Yahoo Finance."
        )

    if "Close" not in df.columns:
        raise ValueError(f"Market data for '{clean_ticker}' does not contain required 'Close' prices.")

    # Drop non-trading days or missing closing prices
    df = df.dropna(subset=["Close"])
    if len(df) < 35:
        raise ValueError(
            f"Insufficient historical data ({len(df)} trading days). "
            "At least 35 trading days are required to calculate 30-day moving averages and lag features."
        )

    return df


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 7-day and 30-day simple moving averages of the Close price.

    Args:
        df: Input DataFrame containing a 'Close' column.

    Returns:
        pd.DataFrame: A copy of df with 'MA7' and 'MA30' columns added.
    """
    processed = df.copy()
    processed["MA7"] = processed["Close"].rolling(window=7).mean()
    processed["MA30"] = processed["Close"].rolling(window=30).mean()
    return processed


def compute_market_stats(df: pd.DataFrame, ticker: str) -> Dict[str, Any]:
    """
    Compute key financial statistics, trend metrics, and sparkline points
    from processed historical data.

    Args:
        df: DataFrame containing OHLC and moving average columns.
        ticker: Ticker symbol string.

    Returns:
        Dict[str, Any]: Structured dictionary of factual market metrics.
    """
    clean_ticker = ticker.strip().upper()

    # Currency determination
    if clean_ticker.endswith(".NS") or clean_ticker.endswith(".BO"):
        currency = "₹"
    elif clean_ticker.endswith(".L"):
        currency = "£"
    elif clean_ticker.endswith(".TO"):
        currency = "C$"
    elif clean_ticker.endswith(".DE") or clean_ticker.endswith(".PA"):
        currency = "€"
    else:
        currency = "$"

    close_series = df["Close"]
    current_price = float(close_series.iloc[-1])
    first_price = float(close_series.iloc[0])
    period_change = current_price - first_price
    period_change_pct = (period_change / first_price) * 100
    period_high = float(df["High"].max())
    period_low = float(df["Low"].min())

    # Latest MA readings
    ma7_val = float(df["MA7"].iloc[-1]) if "MA7" in df.columns and not np.isnan(df["MA7"].iloc[-1]) else current_price
    ma30_val = float(df["MA30"].iloc[-1]) if "MA30" in df.columns and not np.isnan(df["MA30"].iloc[-1]) else current_price

    if ma7_val > ma30_val:
        trend_status = "MA7 > MA30 (short-term moving average above longer rolling trend)"
    elif ma7_val < ma30_val:
        trend_status = "MA7 < MA30 (short-term moving average below longer rolling trend)"
    else:
        trend_status = "MA7 == MA30 (moving averages converging)"

    # Recent 12 closes for sparkline
    sparkline_points = [round(float(p), 2) for p in close_series.iloc[-12:].tolist()]

    return {
        "ticker": clean_ticker,
        "currency": currency,
        "current_price": round(current_price, 2),
        "first_price": round(first_price, 2),
        "period_change": round(period_change, 2),
        "period_change_pct": round(period_change_pct, 2),
        "period_high": round(period_high, 2),
        "period_low": round(period_low, 2),
        "ma7": round(ma7_val, 2),
        "ma30": round(ma30_val, 2),
        "trend_status": trend_status,
        "data_points": len(df),
        "sparkline": sparkline_points,
    }


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Construct autoregressive lag features for next-day price prediction.

    Features:
        Close_Lag1: Closing price 1 day prior to target date
        Close_Lag2: Closing price 2 days prior to target date
        Close_Lag3: Closing price 3 days prior to target date

    Target:
        Target: Closing price on target date

    Latest Features:
        The 3 most recent historical closes used to predict the future next-day close.

    Returns:
        Tuple containing (X, y, latest_features):
        - X: Feature matrix DataFrame of historical lag values.
        - y: Target Series of corresponding closing prices.
        - latest_features: 1-row DataFrame containing the most recent 3 closes.
    """
    data = df.copy()

    # Lag 1 = t-1, Lag 2 = t-2, Lag 3 = t-3 relative to the row's Close
    data["Close_Lag1"] = data["Close"].shift(1)
    data["Close_Lag2"] = data["Close"].shift(2)
    data["Close_Lag3"] = data["Close"].shift(3)
    data["Target"] = data["Close"]

    feature_cols = ["Close_Lag1", "Close_Lag2", "Close_Lag3"]

    # Tomorrow's Lag1 = Today's Close, Lag2 = Yesterday's Close, Lag3 = 2-days-ago Close
    latest_closes = df["Close"].iloc[-3:].values
    latest_features = pd.DataFrame(
        [[latest_closes[2], latest_closes[1], latest_closes[0]]],
        columns=feature_cols
    )

    clean_data = data.dropna(subset=feature_cols + ["Target"])

    X = clean_data[feature_cols]
    y = clean_data["Target"]

    return X, y, latest_features


def train_and_predict_models(df: pd.DataFrame) -> Dict[str, float]:
    """
    Train both Linear Regression and Random Forest Regressor models on historical
    lag features, returning next-day predictions for each.

    Args:
        df: DataFrame containing historical 'Close' price data.

    Returns:
        Dict[str, float]: Predictions keyed by model name.
    """
    X, y, latest_features = prepare_features(df)

    if len(X) < 10:
        raise ValueError("Not enough clean historical samples to train regression models.")

    # 1. Linear Regression Baseline
    lr = LinearRegression()
    lr.fit(X, y)
    lr_pred = float(lr.predict(latest_features)[0])

    # 2. Random Forest Regressor (Non-linear Ensemble)
    rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
    rf.fit(X, y)
    rf_pred = float(rf.predict(latest_features)[0])

    return {
        "Linear Regression": round(lr_pred, 2),
        "Random Forest": round(rf_pred, 2)
    }


def train_and_predict(df: pd.DataFrame) -> float:
    """Legacy wrapper for single Linear Regression prediction."""
    predictions = train_and_predict_models(df)
    return predictions["Linear Regression"]


def evaluate_models(df: pd.DataFrame, test_size: float = 0.2) -> Dict[str, Dict[str, float]]:
    """
    Evaluate both Linear Regression and Random Forest models using a
    chronological train/test split.

    Important:
        Time-series data must NOT be randomly shuffled to prevent lookahead bias.

    Args:
        df: DataFrame containing historical 'Close' price data.
        test_size: Proportion of recent observations reserved for testing (default: 0.2).

    Returns:
        Dict[str, Dict[str, float]]: Model names mapped to dict of 'rmse' and 'r2'.
    """
    X, y, _ = prepare_features(df)

    total_samples = len(X)
    split_index = int(total_samples * (1 - test_size))

    if split_index < 5 or (total_samples - split_index) < 5:
        raise ValueError("Insufficient data points for a meaningful train/test split.")

    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    results = {}

    # 1. Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    lr_rmse = float(np.sqrt(mean_squared_error(y_test, lr_preds)))
    lr_r2 = float(r2_score(y_test, lr_preds))
    results["Linear Regression"] = {
        "rmse": round(lr_rmse, 2),
        "r2": round(lr_r2, 4)
    }

    # 2. Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_rmse = float(np.sqrt(mean_squared_error(y_test, rf_preds)))
    rf_r2 = float(r2_score(y_test, rf_preds))
    results["Random Forest"] = {
        "rmse": round(rf_rmse, 2),
        "r2": round(rf_r2, 4)
    }

    return results


def evaluate_model(df: pd.DataFrame, test_size: float = 0.2) -> Dict[str, float]:
    """Legacy wrapper returning evaluation metrics for Linear Regression."""
    evals = evaluate_models(df, test_size=test_size)
    return evals["Linear Regression"]


def compare_stocks(tickers: List[str], period: str = "6mo") -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Fetch and normalize closing prices for multiple stocks to compare cumulative
    percentage returns over time.

    Args:
        tickers: List of ticker symbols (e.g. ['AAPL', 'MSFT', 'GOOGL']).
        period: Time window to fetch.

    Returns:
        Tuple:
        - pd.DataFrame of cumulative percentage changes indexed by Date.
        - Dict of summary metrics per ticker (Total Return %, Volatility %).
    """
    returns_df = pd.DataFrame()
    stats = {}

    for ticker in tickers:
        clean = ticker.strip().upper()
        if not clean:
            continue
        try:
            df = get_stock_data(clean, period=period)
            close = df["Close"]
            base_price = close.iloc[0]
            pct_series = ((close - base_price) / base_price) * 100
            returns_df[clean] = pct_series

            daily_pct = close.pct_change().dropna()
            stats[clean] = {
                "total_return": round(float(pct_series.iloc[-1]), 2),
                "volatility": round(float(daily_pct.std() * np.sqrt(252) * 100), 2),
                "current_price": round(float(close.iloc[-1]), 2),
            }
        except Exception:
            continue

    return returns_df, stats
