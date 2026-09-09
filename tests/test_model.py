"""
tests/test_model.py — Unit Tests for StockPulse Machine Learning Layer.

Verifies:
1. Feature generation and lag column creation
2. Handling and removal of NaN values from shifting
3. Next-day prediction returning a valid float
4. Model evaluation returning non-empty RMSE and R2 metrics
5. Invalid ticker handling throwing appropriate exceptions
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd

# Add parent directory to path so model can be imported directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model import add_moving_averages, prepare_features, train_and_predict, evaluate_model, get_stock_data


class TestStockPulseModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a synthetic 60-day price trend dataset for fast, deterministic testing
        np.random.seed(42)
        dates = pd.date_range(start="2026-01-01", periods=60, freq="B")
        base_price = 100.0
        # Simulating random walk
        changes = np.random.normal(loc=0.5, scale=1.5, size=60)
        prices = base_price + np.cumsum(changes)

        cls.synthetic_df = pd.DataFrame(
            {
                "Open": prices - 0.5,
                "High": prices + 1.0,
                "Low": prices - 1.0,
                "Close": prices,
                "Volume": np.random.randint(100000, 500000, size=60),
            },
            index=dates,
        )

    def test_moving_averages(self):
        df_ma = add_moving_averages(self.synthetic_df)
        self.assertIn("MA7", df_ma.columns)
        self.assertIn("MA30", df_ma.columns)
        # Check that first 6 values of MA7 are NaN (rolling window of 7)
        self.assertTrue(df_ma["MA7"].iloc[:6].isna().all())
        self.assertFalse(np.isnan(df_ma["MA7"].iloc[6]))
        # Original df must not be mutated
        self.assertNotIn("MA7", self.synthetic_df.columns)

    def test_feature_engineering_lags(self):
        X, y, latest = prepare_features(self.synthetic_df)
        expected_cols = ["Close_Lag1", "Close_Lag2", "Close_Lag3"]
        self.assertListEqual(list(X.columns), expected_cols)
        # Feature rows and target rows must match in length
        self.assertEqual(len(X), len(y))
        # No NaNs in training X or y
        self.assertEqual(X.isna().sum().sum(), 0)
        self.assertEqual(y.isna().sum(), 0)
        # latest_features must have exactly 1 row and 3 columns
        self.assertEqual(latest.shape, (1, 3))

    def test_train_and_predict(self):
        pred = train_and_predict(self.synthetic_df)
        self.assertIsInstance(pred, float)
        # Prediction should be reasonable (within range of prices)
        min_p = self.synthetic_df["Close"].min()
        max_p = self.synthetic_df["Close"].max()
        self.assertGreater(pred, min_p * 0.5)
        self.assertLess(pred, max_p * 1.5)

    def test_evaluation_metrics(self):
        metrics = evaluate_model(self.synthetic_df)
        self.assertIn("rmse", metrics)
        self.assertIn("r2", metrics)
        self.assertIsInstance(metrics["rmse"], float)
        self.assertIsInstance(metrics["r2"], float)
        self.assertGreaterEqual(metrics["rmse"], 0.0)

    def test_invalid_ticker_handling(self):
        with self.assertRaises(ValueError):
            get_stock_data("NONEXISTENT_TICKER_XYZ_999")


if __name__ == "__main__":
    unittest.main()
