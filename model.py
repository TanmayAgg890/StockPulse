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
        period: Historical lookback duration ('1mo', '3mo', '6mo', '1y').

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

    # Drop non-trading days or unfinalized empty bars (NaN in Close/Open)
    df = df.dropna(subset=["Close", "Open", "High", "Low"])
    
    # Check minimum required samples
    min_days = 15 if period == "1mo" else 25
    if len(df) < min_days:
        raise ValueError(
            f"Insufficient historical data ({len(df)} trading days). "
            f"At least {min_days} trading days are required for moving averages and lag features."
        )

    return df


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 7-day and 30-day simple moving averages of the Close price.
    For shorter timeframes (e.g. 1mo), MA30 gracefully uses available observations.
    """
    processed = df.copy()
    processed["MA7"] = processed["Close"].rolling(window=7).mean()
    processed["MA30"] = processed["Close"].rolling(window=min(30, max(10, len(df)))).mean()
    return processed


def compute_market_stats(df: pd.DataFrame, ticker: str) -> Dict[str, Any]:
    """
    Compute key financial statistics, trend metrics, and sparkline points.
    """
    clean_ticker = ticker.strip().upper()

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

    ma7_val = float(df["MA7"].iloc[-1]) if "MA7" in df.columns and not np.isnan(df["MA7"].iloc[-1]) else current_price
    ma30_val = float(df["MA30"].iloc[-1]) if "MA30" in df.columns and not np.isnan(df["MA30"].iloc[-1]) else current_price

    if ma7_val > ma30_val:
        trend_status = "MA7 > MA30 (short-term moving average above longer rolling trend)"
    elif ma7_val < ma30_val:
        trend_status = "MA7 < MA30 (short-term moving average below longer rolling trend)"
    else:
        trend_status = "MA7 == MA30 (moving averages converging)"

    # Price Range Position (0 to 100%)
    price_range = period_high - period_low
    if price_range > 0:
        range_position = min(100.0, max(0.0, ((current_price - period_low) / price_range) * 100))
    else:
        range_position = 50.0

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
        "range_position": round(range_position, 1),
        "ma7": round(ma7_val, 2),
        "ma30": round(ma30_val, 2),
        "trend_status": trend_status,
        "data_points": len(df),
        "sparkline": sparkline_points,
    }


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Construct autoregressive lag features for next-day price prediction.
    """
    data = df.copy()

    data["Close_Lag1"] = data["Close"].shift(1)
    data["Close_Lag2"] = data["Close"].shift(2)
    data["Close_Lag3"] = data["Close"].shift(3)
    data["Target"] = data["Close"]

    feature_cols = ["Close_Lag1", "Close_Lag2", "Close_Lag3"]

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
    Train Linear Regression and Random Forest Regressor models on historical lag features.
    """
    X, y, latest_features = prepare_features(df)

    if len(X) < 8:
        raise ValueError("Not enough clean historical samples to train regression models.")

    # 1. Linear Regression
    lr = LinearRegression()
    lr.fit(X, y)
    lr_pred = float(lr.predict(latest_features)[0])

    # 2. Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
    rf.fit(X, y)
    rf_pred = float(rf.predict(latest_features)[0])

    return {
        "Linear Regression": round(lr_pred, 2),
        "Random Forest": round(rf_pred, 2)
    }


def train_and_predict(df: pd.DataFrame) -> float:
    predictions = train_and_predict_models(df)
    return predictions["Linear Regression"]


def evaluate_models(df: pd.DataFrame, test_size: float = 0.2) -> Dict[str, Dict[str, float]]:
    """
    Evaluate models using a chronological train/test split.
    """
    X, y, _ = prepare_features(df)

    total_samples = len(X)
    split_index = int(total_samples * (1 - test_size))

    if split_index < 4 or (total_samples - split_index) < 4:
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
    evals = evaluate_models(df, test_size=test_size)
    return evals["Linear Regression"]


def compare_stocks(tickers: List[str], period: str = "6mo") -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Fetch and normalize closing prices for multiple stocks to compare cumulative percentage returns.
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
