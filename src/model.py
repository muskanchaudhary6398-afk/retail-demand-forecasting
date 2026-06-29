"""
model.py
Trains Linear Regression, Random Forest, and XGBoost.
Compares MAE / RMSE / R² and saves the best model.
"""

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

def evaluate(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100
    if label:
        print(f"  {label:20s}  MAE={mae:7.2f}  RMSE={rmse:7.2f}  R²={r2:.4f}  MAPE={mape:.2f}%")
    return {"model": label, "mae": round(mae, 4), "rmse": round(rmse, 4),
            "r2": round(r2, 4), "mape": round(mape, 4)}


# ─────────────────────────────────────────────
# TRAIN / EVALUATE PIPELINE
# ─────────────────────────────────────────────

def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    tune: bool = True,
) -> tuple[dict, dict, object]:
    """
    Returns
    -------
    metrics_baseline : dict  — per-model metrics before tuning
    metrics_tuned    : dict  — per-model metrics after tuning
    best_model       : fitted model object
    """
    print("\n── Baseline models ──")

    # ── Linear Regression (needs scaling)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)
    lr = LinearRegression()
    lr.fit(X_tr_s, y_train)
    lr_pred = lr.predict(X_te_s)
    lr_metrics = evaluate(y_test, lr_pred, "LinearRegression")

    # ── Random Forest baseline
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_metrics = evaluate(y_test, rf_pred, "RandomForest (base)")

    # ── XGBoost baseline
    xgb = XGBRegressor(n_estimators=200, learning_rate=0.1,
                       max_depth=6, random_state=42,
                       verbosity=0, n_jobs=-1)
    xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    xgb_pred = xgb.predict(X_test)
    xgb_metrics = evaluate(y_test, xgb_pred, "XGBoost (base)")

    metrics_baseline = {
        "LinearRegression": lr_metrics,
        "RandomForest":     rf_metrics,
        "XGBoost":          xgb_metrics,
    }

    # ── Hyperparameter Tuning ──────────────────
    if tune:
        print("\n── Hyperparameter tuning ──")

        # Random Forest tuning
        rf_params = {
            "n_estimators":    [100, 200, 300],
            "max_depth":       [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "max_features":    ["sqrt", "log2"],
        }
        rf_search = RandomizedSearchCV(
            RandomForestRegressor(random_state=42, n_jobs=-1),
            rf_params, n_iter=12, cv=3, scoring="r2",
            random_state=42, n_jobs=-1
        )
        rf_search.fit(X_train, y_train)
        best_rf = rf_search.best_estimator_
        rf_tuned_pred = best_rf.predict(X_test)
        rf_tuned = evaluate(y_test, rf_tuned_pred, "RandomForest (tuned)")

        # XGBoost tuning
        xgb_params = {
            "n_estimators":  [200, 400, 600],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth":     [4, 6, 8],
            "subsample":     [0.7, 0.8, 1.0],
            "colsample_bytree": [0.7, 0.8, 1.0],
            "reg_alpha":     [0, 0.1, 1.0],
        }
        xgb_search = RandomizedSearchCV(
            XGBRegressor(random_state=42, verbosity=0, n_jobs=-1),
            xgb_params, n_iter=15, cv=3, scoring="r2",
            random_state=42, n_jobs=-1
        )
        xgb_search.fit(X_train, y_train)
        best_xgb = xgb_search.best_estimator_
        xgb_tuned_pred = best_xgb.predict(X_test)
        xgb_tuned = evaluate(y_test, xgb_tuned_pred, "XGBoost (tuned)")

        metrics_tuned = {
            "LinearRegression": lr_metrics,
            "RandomForest":     rf_tuned,
            "XGBoost":          xgb_tuned,
        }

        # Best by R²
        best_name = max(metrics_tuned, key=lambda k: metrics_tuned[k]["r2"])
        best_model = best_rf if best_name == "RandomForest" else (
            best_xgb if best_name == "XGBoost" else lr
        )
        print(f"\n  Best model: {best_name}  R²={metrics_tuned[best_name]['r2']}")
    else:
        metrics_tuned = metrics_baseline
        best_model = xgb

    return metrics_baseline, metrics_tuned, best_model, scaler


# ─────────────────────────────────────────────
# FEATURE IMPORTANCE
# ─────────────────────────────────────────────

def get_feature_importance(model, feature_names: list, top_n: int = 15) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_)
    else:
        return pd.DataFrame()

    df = pd.DataFrame({"feature": feature_names, "importance": imp})
    return df.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)


# ─────────────────────────────────────────────
# PERSIST
# ─────────────────────────────────────────────

def save_artifacts(model, scaler, metrics: dict, feature_cols: list,
                   out_dir: str = "results") -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/best_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(f"{out_dir}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(f"{out_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(f"{out_dir}/feature_cols.json", "w") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"\nArtifacts saved to '{out_dir}/'")


def load_artifacts(out_dir: str = "results"):
    with open(f"{out_dir}/best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(f"{out_dir}/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(f"{out_dir}/metrics.json") as f:
        metrics = json.load(f)
    with open(f"{out_dir}/feature_cols.json") as f:
        feature_cols = json.load(f)
    return model, scaler, metrics, feature_cols
