"""
tests/test_pipeline.py
Unit tests for data generation, feature engineering, and model training.
Run: python tests/test_pipeline.py
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_generator import generate_retail_data
from src.feature_engineering import (
    add_calendar_features, add_lag_features, add_rolling_features,
    engineer_all_features, get_feature_columns,
)
from src.model import evaluate, get_feature_importance
from sklearn.ensemble import RandomForestRegressor


class TestDataGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = generate_retail_data(
            start="2023-01-01", end="2023-03-31",
            n_products=3, seed=0
        )

    def test_row_count(self):
        # 90 days × 3 products
        self.assertEqual(len(self.df), 90 * 3)

    def test_columns_present(self):
        for col in ["date", "product_id", "sales", "price", "promotion", "category"]:
            self.assertIn(col, self.df.columns)

    def test_sales_non_negative(self):
        self.assertTrue((self.df["sales"] >= 0).all())

    def test_date_type(self):
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(self.df["date"]))

    def test_three_products(self):
        self.assertEqual(self.df["product_id"].nunique(), 3)

    def test_promotion_binary(self):
        self.assertTrue(self.df["promotion"].isin([0, 1]).all())

    def test_price_positive(self):
        self.assertTrue((self.df["price"] > 0).all())


class TestFeatureEngineering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = generate_retail_data(
            start="2023-01-01", end="2023-06-30",
            n_products=2, seed=1
        )

    def test_calendar_features_added(self):
        out = add_calendar_features(self.df)
        for col in ["day_of_week", "month", "quarter", "is_weekend",
                    "dow_sin", "dow_cos", "month_sin", "month_cos"]:
            self.assertIn(col, out.columns)

    def test_lag_features_added(self):
        out = add_lag_features(self.df, lags=[1, 7])
        self.assertIn("lag_1", out.columns)
        self.assertIn("lag_7", out.columns)

    def test_lag_introduces_nan(self):
        out = add_lag_features(self.df, lags=[1])
        self.assertTrue(out["lag_1"].isna().any())

    def test_rolling_features_added(self):
        out = add_rolling_features(self.df, windows=[7])
        self.assertIn("roll_mean_7", out.columns)
        self.assertIn("roll_std_7",  out.columns)

    def test_cyclical_encoding_range(self):
        out = add_calendar_features(self.df)
        self.assertTrue((out["dow_sin"].between(-1, 1)).all())
        self.assertTrue((out["month_cos"].between(-1, 1)).all())

    def test_full_pipeline_returns_feature_cols(self):
        out = engineer_all_features(self.df)
        feature_cols = get_feature_columns()
        for col in feature_cols:
            self.assertIn(col, out.columns)

    def test_feature_col_count(self):
        cols = get_feature_columns()
        self.assertGreater(len(cols), 30)

    def test_no_original_data_dropped(self):
        out = engineer_all_features(self.df)
        self.assertIn("sales", out.columns)
        self.assertIn("date",  out.columns)


class TestMetrics(unittest.TestCase):
    def _arrays(self):
        np.random.seed(42)
        y_true = np.random.randint(10, 200, size=100).astype(float)
        y_pred = y_true + np.random.normal(0, 10, 100)
        return y_true, y_pred

    def test_evaluate_returns_dict(self):
        y_true, y_pred = self._arrays()
        result = evaluate(y_true, y_pred)
        self.assertIsInstance(result, dict)

    def test_evaluate_keys(self):
        y_true, y_pred = self._arrays()
        result = evaluate(y_true, y_pred)
        for key in ["mae", "rmse", "r2", "mape"]:
            self.assertIn(key, result)

    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = evaluate(y, y)
        self.assertAlmostEqual(result["r2"], 1.0, places=4)
        self.assertAlmostEqual(result["mae"], 0.0, places=4)

    def test_rmse_non_negative(self):
        y_true, y_pred = self._arrays()
        result = evaluate(y_true, y_pred)
        self.assertGreaterEqual(result["rmse"], 0)

    def test_r2_below_one(self):
        y_true, y_pred = self._arrays()
        result = evaluate(y_true, y_pred)
        self.assertLessEqual(result["r2"], 1.0)


class TestFeatureImportance(unittest.TestCase):
    def test_returns_dataframe(self):
        X = pd.DataFrame(np.random.rand(100, 5),
                         columns=[f"f{i}" for i in range(5)])
        y = np.random.rand(100)
        rf = RandomForestRegressor(n_estimators=10, random_state=0)
        rf.fit(X, y)
        fi = get_feature_importance(rf, X.columns.tolist(), top_n=3)
        self.assertIsInstance(fi, pd.DataFrame)
        self.assertLessEqual(len(fi), 3)

    def test_importance_columns(self):
        X = pd.DataFrame(np.random.rand(50, 4),
                         columns=["a", "b", "c", "d"])
        y = np.random.rand(50)
        rf = RandomForestRegressor(n_estimators=5, random_state=0)
        rf.fit(X, y)
        fi = get_feature_importance(rf, ["a", "b", "c", "d"])
        self.assertIn("feature",    fi.columns)
        self.assertIn("importance", fi.columns)

    def test_sorted_descending(self):
        X = pd.DataFrame(np.random.rand(80, 5),
                         columns=[f"x{i}" for i in range(5)])
        y = X["x0"] * 3 + np.random.rand(80) * 0.1
        rf = RandomForestRegressor(n_estimators=20, random_state=0)
        rf.fit(X, y)
        fi = get_feature_importance(rf, X.columns.tolist())
        imps = fi["importance"].tolist()
        self.assertEqual(imps, sorted(imps, reverse=True))


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
