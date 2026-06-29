"""
data_generator.py
Generates a realistic synthetic retail sales dataset (3 years, 10 products).
Includes seasonality, trend, promotions, and noise.
"""

import numpy as np
import pandas as pd
from datetime import datetime


def generate_retail_data(
    start: str = "2022-01-01",
    end: str = "2024-12-31",
    n_products: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    np.random.seed(seed)

    dates = pd.date_range(start, end, freq="D")
    products = [f"P{i:03d}" for i in range(1, n_products + 1)]

    categories = {
        "P001": "Electronics", "P002": "Electronics",
        "P003": "Clothing",    "P004": "Clothing",
        "P005": "Grocery",     "P006": "Grocery",
        "P007": "Home",        "P008": "Home",
        "P009": "Sports",      "P010": "Sports",
    }
    base_prices = {
        "P001": 299, "P002": 149, "P003": 59,  "P004": 39,
        "P005": 12,  "P006": 8,   "P007": 89,  "P008": 45,
        "P009": 79,  "P010": 55,
    }
    base_demand = {
        "P001": 30,  "P002": 55,  "P003": 80,  "P004": 110,
        "P005": 200, "P006": 180, "P007": 50,  "P008": 70,
        "P009": 40,  "P010": 60,
    }

    rows = []
    for product in products:
        cat   = categories[product]
        price = base_prices[product]
        base  = base_demand[product]

        for date in dates:
            doy   = date.day_of_year
            year  = date.year
            dow   = date.dayofweek   # 0=Mon
            month = date.month

            # Long-term trend
            trend = 1 + 0.08 * (year - 2022)

            # Yearly seasonality (different per category)
            if cat == "Electronics":
                season = 1 + 0.4 * np.sin(2 * np.pi * (doy - 330) / 365)
            elif cat == "Clothing":
                season = 1 + 0.3 * np.sin(2 * np.pi * (doy - 90) / 365)
            elif cat == "Grocery":
                season = 1 + 0.1 * np.sin(2 * np.pi * doy / 365)
            elif cat == "Home":
                season = 1 + 0.25 * np.sin(2 * np.pi * (doy - 60) / 365)
            else:  # Sports
                season = 1 + 0.35 * np.sin(2 * np.pi * (doy - 150) / 365)

            # Weekly pattern: weekends higher for non-grocery
            weekly = 1.0
            if cat != "Grocery":
                weekly = 1.2 if dow >= 5 else (0.9 if dow == 0 else 1.0)

            # Holiday bumps
            holiday = 1.0
            if month == 12 and date.day >= 15:   holiday = 2.0
            elif month == 11 and date.day >= 24: holiday = 1.8   # Black Friday
            elif month == 2 and date.day == 14:  holiday = 1.5   # Valentine
            elif month == 7 and date.day == 4:   holiday = 1.3

            # Promotions (random ~10% of days)
            promo = np.random.random() < 0.10
            promo_discount = np.random.uniform(0.1, 0.3) if promo else 0.0
            promo_lift = 1 + promo_discount * 1.8 if promo else 1.0
            actual_price = round(price * (1 - promo_discount), 2)

            # Demand
            mu = base * trend * season * weekly * holiday * promo_lift
            demand = int(max(0, np.random.poisson(mu)))

            rows.append({
                "date":         date,
                "product_id":   product,
                "category":     cat,
                "sales":        demand,
                "price":        actual_price,
                "base_price":   price,
                "promotion":    int(promo),
                "discount_pct": round(promo_discount * 100, 1),
                "is_holiday":   int(holiday > 1.0),
                "is_weekend":   int(dow >= 5),
            })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["product_id", "date"]).reset_index(drop=True)


if __name__ == "__main__":
    df = generate_retail_data()
    df.to_csv("data/retail_sales.csv", index=False)
    print(f"Generated {len(df):,} rows for {df['product_id'].nunique()} products")
    print(df.head())
