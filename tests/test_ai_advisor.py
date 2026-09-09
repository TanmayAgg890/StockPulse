"""
tests/test_ai_advisor.py — Unit tests for StockPulse AI Advisor & Grounded Context Builder.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_advisor import build_context, get_ai_insight, answer_followup


class TestAIAdvisor(unittest.TestCase):
    def setUp(self):
        self.sample_stats = {
            "ticker": "AAPL",
            "currency": "$",
            "current_price": 230.50,
            "period_change": 15.20,
            "period_change_pct": 7.05,
            "period_high": 235.00,
            "period_low": 200.00,
            "ma7": 228.00,
            "ma30": 222.00,
            "trend_status": "MA7 > MA30",
            "data_points": 125,
        }
        self.sample_preds = {
            "Linear Regression": 232.00,
            "Random Forest": 231.50,
        }
        self.sample_evals = {
            "Linear Regression": {"rmse": 2.85, "r2": 0.65},
            "Random Forest": {"rmse": 2.70, "r2": 0.68},
        }

    def test_build_context(self):
        context = build_context(
            "AAPL",
            self.sample_stats,
            self.sample_preds,
            self.sample_evals,
        )
        self.assertIn("Ticker Symbol: AAPL", context)
        self.assertIn("Current Close Price: $230.50", context)
        self.assertIn("Linear Regression Estimate: $232.00", context)
        self.assertIn("Random Forest Regressor Estimate: $231.50", context)
        self.assertIn("Test RMSE: $2.85", context)

    def test_fallback_insight_generation(self):
        context = build_context(
            "AAPL",
            self.sample_stats,
            self.sample_preds,
            self.sample_evals,
        )
        # Without providing an API key, it should safely return deterministic fallback
        insight = get_ai_insight(context, custom_api_key=None)
        self.assertIsInstance(insight, str)
        self.assertIn("Trend Assessment", insight)
        self.assertIn("Educational analysis only", insight)

    def test_followup_without_key(self):
        context = build_context(
            "AAPL",
            self.sample_stats,
            self.sample_preds,
            self.sample_evals,
        )
        ans = answer_followup(context, [], "Why is MA7 above MA30?", custom_api_key=None)
        self.assertIn("Grounded Mode", ans)


if __name__ == "__main__":
    unittest.main()
