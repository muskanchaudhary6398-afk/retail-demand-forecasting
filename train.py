"""
train.py
End-to-end training pipeline: generate data → engineer features → train → evaluate → save.
Run:  python train.py
"""

import sys
import os
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from src.data_generator import generate_retail_data
from src.feature_engineering import engineer_all_features, get_feature_columns
from src.model import (
    train_all_models, get_feature_importance,
    save_artifacts, evaluate
)


def main():
    print("=" * 60)
    print("  Retail Demand Forecasting — Training Pipeline")
    print("=" * 60)

    # 1. Data
    print("\n[1/5] Generating retail dataset …")
    df = generate_retail_data()
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/retail_sales.csv", index=False)
    print(f"  {len(df):,} rows  |  {df['product_id'].nunique()} products  "
          f"|  {df['date'].min().date()} → {df['date'].max().date()}")

    # 2. Feature Engineering
    print("\n[2/5] Engineering features …")
    df_feat = engineer_all_features(df)
    feature_cols = get_feature_columns()

    # Drop rows with NaN (lag warm-up period)
    df_clean = df_feat.dropna(subset=feature_cols).copy()
    print(f"  {len(df_clean):,} usable rows after lag warm-up")
    print(f"  {len(feature_cols)} features: calendar + lag + rolling + price")

    # 3. Train / Test split (time-based — no shuffle!)
    print("\n[3/5] Splitting train / test (last 90 days = test) …")
    cutoff = df_clean["date"].max() - pd.Timedelta(days=90)
    train  = df_clean[df_clean["date"] <= cutoff]
    test   = df_clean[df_clean["date"] >  cutoff]
    X_train, y_train = train[feature_cols], train["sales"]
    X_test,  y_test  = test[feature_cols],  test["sales"]
    print(f"  Train: {len(train):,} rows | Test: {len(test):,} rows")

    # 4. Model Training
    print("\n[4/5] Training & tuning models …")
    metrics_base, metrics_tuned, best_model, scaler = train_all_models(
        X_train, y_train, X_test, y_test, tune=True
    )

    # 5. Feature Importance
    print("\n[5/5] Feature importance (top 10) …")
    fi = get_feature_importance(best_model, feature_cols, top_n=10)
    if not fi.empty:
        for _, row in fi.iterrows():
            bar = "█" * int(row["importance"] / fi["importance"].max() * 20)
            print(f"  {row['feature']:25s} {bar} {row['importance']:.4f}")
        os.makedirs("results", exist_ok=True)
        fi.to_csv("results/feature_importance.csv", index=False)

    # Save full enriched data for dashboard
    df_feat.to_csv("data/retail_sales_features.csv", index=False)

    # Compute baseline RMSE (no-feature linear model) for reduction %
    best_name = max(metrics_tuned, key=lambda k: metrics_tuned[k]["r2"])
    base_rmse = metrics_base["LinearRegression"]["rmse"]
    best_rmse = metrics_tuned[best_name]["rmse"]
    rmse_reduction = (base_rmse - best_rmse) / base_rmse * 100

    summary = {
        "best_model":      best_name,
        "best_r2":         metrics_tuned[best_name]["r2"],
        "best_mae":        metrics_tuned[best_name]["mae"],
        "best_rmse":       metrics_tuned[best_name]["rmse"],
        "best_mape":       metrics_tuned[best_name]["mape"],
        "rmse_reduction_pct": round(rmse_reduction, 2),
        "train_rows":      len(train),
        "test_rows":       len(test),
        "n_features":      len(feature_cols),
        "baseline":        metrics_base,
        "tuned":           metrics_tuned,
    }

    os.makedirs("results", exist_ok=True)
    with open("results/training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    save_artifacts(best_model, scaler, metrics_tuned, feature_cols)

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print(f"  Best model : {best_name}")
    print(f"  R²         : {summary['best_r2']}")
    print(f"  MAE        : {summary['best_mae']}")
    print(f"  RMSE       : {summary['best_rmse']}")
    print(f"  RMSE reduction vs baseline: {rmse_reduction:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
