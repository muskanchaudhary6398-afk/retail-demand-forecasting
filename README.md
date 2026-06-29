# Retail Demand Forecasting & Inventory Planning

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Tests](https://img.shields.io/badge/tests-23%20passed-brightgreen)
![R²](https://img.shields.io/badge/R²-0.9735-success)
![RMSE Reduction](https://img.shields.io/badge/RMSE%20Reduction-44.8%25-orange)
![License](https://img.shields.io/badge/license-MIT-green)

An end-to-end machine learning pipeline for retail demand forecasting. Compares Linear Regression, Random Forest, and XGBoost with systematic hyperparameter tuning, achieving **R² = 0.9735** and a **44.8% RMSE reduction** over the baseline. Includes an interactive Streamlit dashboard for real-time forecasting and inventory planning.

---

## Results

| Model | MAE | RMSE | R² | MAPE |
|---|---|---|---|---|
| Linear Regression | 17.01 | 29.09 | 0.9129 | 15.45% |
| Random Forest (tuned) | 12.84 | 19.96 | 0.9590 | 12.09% |
| **XGBoost (tuned)** | **10.95** | **16.05** | **0.9735** | **10.48%** |

- 10,960 records across 10 products and 6 categories (2022–2024)
- 39 engineered features: lag, rolling statistics, calendar encoding, price/promo signals
- Time-based train/test split (last 90 days held out)

---

## Architecture

```
Raw Sales Data
     │
     ▼
Feature Engineering
  ├── Calendar features (day, month, quarter, cyclical sin/cos)
  ├── Lag features       (lag-1, lag-7, lag-14, lag-21, lag-28)
  ├── Rolling statistics (mean, std, min, max over 7/14/28 days)
  └── Price/promo signals (discount %, holiday flag, weekend flag)
     │
     ▼
Time-Based Train/Test Split
     │
     ├── Linear Regression (StandardScaler)
     ├── Random Forest ──┐
     └── XGBoost        ─┤── RandomizedSearchCV (hyperparameter tuning)
                         │
                         ▼
                  Best Model (XGBoost)
                         │
                    ┌────┴────┐
                    │         │
              Evaluation    Streamlit Dashboard
            MAE/RMSE/R²   (Forecast + Inventory Planning)
```

---

## Project Structure

```
retail-demand-forecasting/
├── src/
│   ├── data_generator.py       # Synthetic retail data with seasonality & promotions
│   ├── feature_engineering.py  # Lag, rolling, calendar, price features
│   └── model.py                # Training, evaluation, hyperparameter tuning
├── tests/
│   └── test_pipeline.py        # 23 unit tests
├── results/
│   ├── training_summary.json   # Benchmark results
│   ├── feature_importance.csv  # Top features by XGBoost importance
│   └── best_model.pkl          # Saved model artifact
├── train.py                    # End-to-end training pipeline
├── dashboard.py                # Streamlit dashboard
└── requirements.txt
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the training pipeline
python train.py

# 3. Launch the dashboard
streamlit run dashboard.py

# 4. Run tests
python tests/test_pipeline.py
```

---

## Key Design Decisions

**Time-based split** — shuffling time-series data leaks future information. The last 90 days are held out as a true test set.

**Lag warm-up** — rows within the first 28 days of each product's history are dropped to avoid NaN leakage into lag features.

**Cyclical encoding** — month and day-of-week are encoded as `sin`/`cos` pairs so the model understands that December → January is a small step, not a large one.

**RandomizedSearchCV** — 12–15 random hyperparameter combinations with 3-fold CV, faster than grid search while exploring a wide space.

---

## Feature Importance (Top 10)

| Feature | Importance |
|---|---|
| roll_mean_7 | 0.3091 |
| roll_min_7 | 0.2667 |
| quarter | 0.0762 |
| roll_mean_28 | 0.0645 |
| lag_1 | 0.0440 |
| lag_21 | 0.0384 |
| roll_mean_14 | 0.0294 |
| price | 0.0269 |
| is_holiday | 0.0239 |
| promotion | 0.0187 |

Short-window rolling statistics dominate, confirming that recent demand history is the strongest signal for next-day forecasting.
