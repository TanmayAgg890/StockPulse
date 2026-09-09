"""
model.py — Machine Learning and Data Logic Layer for StockPulse.

Responsibilities:
- Fetch historical market data from Yahoo Finance via yfinance.
- Clean and validate incoming financial time-series.
- Compute technical indicators (7-day & 30-day Moving Averages).
- Construct lag features for autoregressive supervised learning.
- Train a Linear Regression model with a chronological train/test split.
- Generate next-day price predictions and evaluation metrics (RMSE, R2).
"""

from typing import Dict, Tuple, Any
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
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

    # The latest row in df (before dropping NaNs) contains the latest 3 closes
    # to forecast tomorrow's unobserved price:
    # Tomorrow's Lag1 = Today's Close, Lag2 = Yesterday's Close, Lag3 = 2-days-ago Close
    latest_closes = df["Close"].iloc[-3:].values
    latest_features = pd.DataFrame(
        [[latest_closes[2], latest_closes[1], latest_closes[0]]],
        columns=feature_cols
    )

    # Clean dataset for supervised model training
    clean_data = data.dropna(subset=feature_cols + ["Target"])

    X = clean_data[feature_cols]
    y = clean_data["Target"]

    return X, y, latest_features


def train_and_predict(df: pd.DataFrame) -> float:
    """
    Train a Linear Regression model on historical lag features and predict
    the next trading day's closing price.

    Args:
        df: DataFrame containing historical 'Close' price data.

    Returns:
        float: Predicted closing price for the next trading day.
    """
    X, y, latest_features = prepare_features(df)

    if len(X) < 10:
        raise ValueError("Not enough clean historical samples to train regression model.")

    model = LinearRegression()
    model.fit(X, y)

    prediction = model.predict(latest_features)[0]
    return float(prediction)


def evaluate_model(df: pd.DataFrame, test_size: float = 0.2) -> Dict[str, float]:
    """
    Evaluate the Linear Regression model using a chronological train/test split.

    Important:
        Time-series data must NOT be randomly shuffled to prevent data leakage
        from future observations into the training phase.

    Args:
        df: DataFrame containing historical 'Close' price data.
        test_size: Proportion of recent observations reserved for testing (default: 0.2).

    Returns:
        Dict[str, float]: Dictionary with 'rmse' and 'r2' scores.
    """
    X, y, _ = prepare_features(df)

    total_samples = len(X)
    split_index = int(total_samples * (1 - test_size))

    if split_index < 5 or (total_samples - split_index) < 5:
        raise ValueError("Insufficient data points for a meaningful train/test split.")

    # Chronological split: past for training, recent slice for testing
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # Calculate metrics
    # Note: scikit-learn >= 1.4 deprecated squared=False in mean_squared_error
    mse = mean_squared_error(y_test, predictions)
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test, predictions))

    return {
        "rmse": round(rmse, 2),
        "r2": round(r2, 4)
    }
