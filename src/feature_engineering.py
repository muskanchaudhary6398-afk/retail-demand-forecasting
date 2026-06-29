"""
feature_engineering.py
Lag features, rolling statistics, and calendar features for demand forecasting.
"""

import pandas as pd
import numpy as np


def add_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Extract temporal signals from the date column."""
    df = df.copy()
    d = df[date_col]
    df["day_of_week"]    = d.dt.dayofweek
    df["day_of_month"]   = d.dt.day
    df["day_of_year"]    = d.dt.dayofyear
    df["week_of_year"]   = d.dt.isocalendar().week.astype(int)
    df["month"]          = d.dt.month
    df["quarter"]        = d.dt.quarter
    df["year"]           = d.dt.year
    df["is_weekend"]     = (d.dt.dayofweek >= 5).astype(int)
    df["is_month_start"] = d.dt.is_month_start.astype(int)
    df["is_month_end"]   = d.dt.is_month_end.astype(int)

    # Cyclical encoding (preserves periodicity)
    df["dow_sin"]    = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]    = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"]  = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]  = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"]    = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"]    = np.cos(2 * np.pi * df["day_of_year"] / 365)
    return df


def add_lag_features(
    df: pd.DataFrame,
    target_col: str = "sales",
    group_col: str = "product_id",
    lags: list = [1, 7, 14, 21, 28],
) -> pd.DataFrame:
    """Add lagged sales values grouped by product."""
    df = df.copy().sort_values([group_col, "date"])
    for lag in lags:
        df[f"lag_{lag}"] = df.groupby(group_col)[target_col].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str = "sales",
    group_col: str = "product_id",
    windows: list = [7, 14, 28],
) -> pd.DataFrame:
    """Rolling mean, std, min, max for each window size."""
    df = df.copy().sort_values([group_col, "date"])
    for w in windows:
        grp = df.groupby(group_col)[target_col]
        # shift(1) to avoid leakage
        df[f"roll_mean_{w}"] = grp.shift(1).rolling(w).mean().values
        df[f"roll_std_{w}"]  = grp.shift(1).rolling(w).std().values
        df[f"roll_min_{w}"]  = grp.shift(1).rolling(w).min().values
        df[f"roll_max_{w}"]  = grp.shift(1).rolling(w).max().values
    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Price deviation from base price as a ratio."""
    df = df.copy()
    df["price_ratio"] = df["price"] / df["base_price"]
    return df


def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline."""
    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_price_features(df)

    # Category encoding
    df["category_code"] = df["category"].astype("category").cat.codes

    return df


def get_feature_columns() -> list:
    """Canonical list of model input features (no leakage)."""
    calendar = [
        "day_of_week", "day_of_month", "day_of_year",
        "week_of_year", "month", "quarter", "year",
        "is_weekend", "is_month_start", "is_month_end",
        "dow_sin", "dow_cos", "month_sin", "month_cos",
        "doy_sin", "doy_cos",
    ]
    lags = [f"lag_{l}" for l in [1, 7, 14, 21, 28]]
    rolling = [
        f"roll_{stat}_{w}"
        for stat in ["mean", "std", "min", "max"]
        for w in [7, 14, 28]
    ]
    price = ["price", "price_ratio", "promotion", "discount_pct", "is_holiday"]
    other = ["category_code"]
    return calendar + lags + rolling + price + other
