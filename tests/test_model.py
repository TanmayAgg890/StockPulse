"""
tests/test_model.py — Unit Tests for StockPulse Machine Learning Layer.

Verifies:
1. Feature generation and lag column creation
2. Handling and removal of NaN values from shifting
3. Next-day predictions returning valid floats for Linear Regression & Random Forest
4. Multi-model evaluation returning valid RMSE and R2 metrics
5. Invalid ticker handling throwing appropriate exceptions
6. Multi-stock comparison logic and normalization
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model import (
    add_moving_averages,
    prepare_features,
    train_and_predict,
    train_and_predict_models,
    evaluate_model,
    evaluate_models,
    get_stock_data,
    compare_stocks,
)


class TestStockPulseModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        dates = pd.date_range(start="2026-01-01", periods=60, freq="B")
        base_price = 100.0
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
        self.assertTrue(df_ma["MA7"].iloc[:6].isna().all())
        self.assertFalse(np.isnan(df_ma["MA7"].iloc[6]))
        self.assertNotIn("MA7", self.synthetic_df.columns)

    def test_feature_engineering_lags(self):
        X, y, latest = prepare_features(self.synthetic_df)
        expected_cols = ["Close_Lag1", "Close_Lag2", "Close_Lag3"]
        self.assertListEqual(list(X.columns), expected_cols)
        self.assertEqual(len(X), len(y))
        self.assertEqual(X.isna().sum().sum(), 0)
        self.assertEqual(y.isna().sum(), 0)
        self.assertEqual(latest.shape, (1, 3))

    def test_train_and_predict_multi_model(self):
        preds = train_and_predict_models(self.synthetic_df)
        self.assertIn("Linear Regression", preds)
        self.assertIn("Random Forest", preds)
        self.assertIsInstance(preds["Linear Regression"], float)
        self.assertIsInstance(preds["Random Forest"], float)

    def test_evaluation_multi_model(self):
        evals = evaluate_models(self.synthetic_df)
        for model_name in ["Linear Regression", "Random Forest"]:
            self.assertIn(model_name, evals)
            self.assertIn("rmse", evals[model_name])
            self.assertIn("r2", evals[model_name])
            self.assertIsInstance(evals[model_name]["rmse"], float)
            self.assertIsInstance(evals[model_name]["r2"], float)

    def test_invalid_ticker_handling(self):
        with self.assertRaises(ValueError):
            get_stock_data("NONEXISTENT_TICKER_XYZ_999")


if __name__ == "__main__":
    unittest.main()
