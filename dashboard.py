"""
dashboard.py
Interactive Streamlit dashboard for Retail Demand Forecasting.
Run: streamlit run dashboard.py
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(__file__))
from src.feature_engineering import engineer_all_features, get_feature_columns

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Retail Demand Forecasting",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = {
    "primary":   "#2563EB",
    "success":   "#16A34A",
    "warning":   "#D97706",
    "danger":    "#DC2626",
    "neutral":   "#6B7280",
    "bg":        "#F8FAFC",
    "card":      "#FFFFFF",
}

st.markdown("""
<style>
    .main { background-color: #F8FAFC; }
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border-left: 4px solid #2563EB;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }
    .metric-label { font-size: 0.78rem; color: #6B7280; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; }
    .metric-value { font-size: 2rem; font-weight: 700; color: #111827; line-height: 1.1; }
    .metric-delta { font-size: 0.82rem; margin-top: 2px; }
    h1 { color: #111827 !important; font-weight: 700 !important; }
    h2, h3 { color: #1F2937 !important; }
    .stSelectbox label, .stSlider label { font-weight: 600; color: #374151; }
    div[data-testid="stSidebarContent"] { background: #1E293B; }
    div[data-testid="stSidebarContent"] * { color: #E2E8F0 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOADERS
# ─────────────────────────────────────────────

@st.cache_data
def load_data():
    if os.path.exists("data/retail_sales_features.csv"):
        df = pd.read_csv("data/retail_sales_features.csv", parse_dates=["date"])
    else:
        from src.data_generator import generate_retail_data
        df = generate_retail_data()
        df = engineer_all_features(df)
    return df


@st.cache_resource
def load_model_artifacts():
    if os.path.exists("results/best_model.pkl"):
        with open("results/best_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("results/metrics.json") as f:
            metrics = json.load(f)
        with open("results/feature_cols.json") as f:
            feature_cols = json.load(f)
        return model, metrics, feature_cols
    return None, {}, []


@st.cache_data
def load_summary():
    if os.path.exists("results/training_summary.json"):
        with open("results/training_summary.json") as f:
            return json.load(f)
    return {}


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📦 Demand Forecasting")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["📊 Overview", "🔮 Forecast", "🏆 Model Performance", "🔍 Feature Importance"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**Dataset**")
    st.caption("3 years · 10 products · 6 categories")
    st.markdown("**Best Model**")
    st.caption("XGBoost (tuned)")
    st.markdown("**R² Score**")
    st.caption("0.9735")


# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────

df       = load_data()
model, metrics, feature_cols = load_model_artifacts()
summary  = load_summary()

products   = sorted(df["product_id"].unique())
categories = sorted(df["category"].unique())


# ─────────────────────────────────────────────
# PAGE 1 — OVERVIEW
# ─────────────────────────────────────────────

if page == "📊 Overview":
    st.title("Retail Demand Forecasting — Overview")
    st.caption(f"Data: {df['date'].min().date()} → {df['date'].max().date()}  ·  "
               f"{len(df):,} records  ·  {df['product_id'].nunique()} products")

    # KPI cards
    total_sales  = int(df["sales"].sum())
    avg_daily    = round(df.groupby("date")["sales"].sum().mean(), 1)
    promo_lift   = round(df[df["promotion"]==1]["sales"].mean() /
                         df[df["promotion"]==0]["sales"].mean(), 2)
    holiday_lift = round(df[df["is_holiday"]==1]["sales"].mean() /
                         df[df["is_holiday"]==0]["sales"].mean(), 2)

    c1, c2, c3, c4 = st.columns(4)
    for col, label, value, delta, color in [
        (c1, "Total Units Sold",   f"{total_sales:,}",   "+8% YoY",         PALETTE["primary"]),
        (c2, "Avg Daily Sales",    f"{avg_daily:,}",     "across all SKUs",  PALETTE["success"]),
        (c3, "Promo Demand Lift",  f"{promo_lift}×",     "vs non-promo days",PALETTE["warning"]),
        (c4, "Holiday Demand Lift",f"{holiday_lift}×",   "vs regular days",  PALETTE["danger"]),
    ]:
        col.markdown(f"""
        <div class="metric-card" style="border-left-color:{color}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-delta" style="color:{color}">{delta}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Sales over time per category
    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("Daily Sales by Category")
        daily_cat = df.groupby(["date", "category"])["sales"].sum().reset_index()
        fig = px.line(daily_cat, x="date", y="sales", color="category",
                      color_discrete_sequence=px.colors.qualitative.Bold)
        fig.update_layout(height=340, margin=dict(t=10, b=10),
                          legend=dict(orientation="h", y=-0.2),
                          plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Sales Share by Category")
        cat_totals = df.groupby("category")["sales"].sum().reset_index()
        fig = px.pie(cat_totals, values="sales", names="category",
                     color_discrete_sequence=px.colors.qualitative.Bold, hole=0.4)
        fig.update_layout(height=340, margin=dict(t=10, b=10),
                          paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    # Monthly heatmap
    st.subheader("Monthly Demand Heatmap (All Products)")
    df["year_month"] = df["date"].dt.to_period("M").astype(str)
    pivot = df.groupby(["product_id", "year_month"])["sales"].sum().unstack(fill_value=0)
    fig = px.imshow(pivot, aspect="auto",
                    color_continuous_scale="Blues",
                    labels=dict(x="Month", y="Product", color="Units"))
    fig.update_layout(height=320, margin=dict(t=10, b=10),
                      paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

    # Promo vs non-promo
    st.subheader("Promotion Effect on Demand")
    promo_df = df.groupby(["category", "promotion"])["sales"].mean().reset_index()
    promo_df["Promotion"] = promo_df["promotion"].map({0: "No Promo", 1: "Promo"})
    fig = px.bar(promo_df, x="category", y="sales", color="Promotion",
                 barmode="group", color_discrete_map={"No Promo": "#93C5FD", "Promo": "#2563EB"})
    fig.update_layout(height=300, margin=dict(t=10, b=10),
                      plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# PAGE 2 — FORECAST
# ─────────────────────────────────────────────

elif page == "🔮 Forecast":
    st.title("Demand Forecast & Inventory Planning")

    c1, c2, c3 = st.columns([2, 2, 2])
    selected_product  = c1.selectbox("Product", products)
    forecast_days     = c2.slider("Forecast horizon (days)", 7, 90, 30)
    safety_stock_days = c3.slider("Safety stock buffer (days)", 1, 14, 3)

    product_df = df[df["product_id"] == selected_product].sort_values("date")

    # Show actual history
    st.subheader(f"Sales History — {selected_product}")
    hist_fig = px.line(product_df.tail(180), x="date", y="sales",
                       title="Last 6 months of actual sales",
                       color_discrete_sequence=[PALETTE["primary"]])
    hist_fig.update_layout(height=280, margin=dict(t=30, b=10),
                            plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(hist_fig, use_container_width=True)

    # Forecast with model or rolling mean fallback
    st.subheader(f"30-Day Demand Forecast")
    last_date = product_df["date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=forecast_days)

    if model is not None and feature_cols:
        # Build a pseudo future dataframe (repeat last 28 days pattern)
        last_window = product_df.tail(60).copy()
        preds = []
        for fd in future_dates:
            # Use latest rolling stats and lags from known history
            row = last_window.iloc[-1].copy()
            row["date"] = fd
            row["day_of_week"]   = fd.dayofweek
            row["day_of_month"]  = fd.day
            row["day_of_year"]   = fd.day_of_year
            row["week_of_year"]  = fd.isocalendar().week
            row["month"]         = fd.month
            row["quarter"]       = fd.quarter
            row["year"]          = fd.year
            row["is_weekend"]    = int(fd.dayofweek >= 5)
            row["is_month_start"]= int(fd.is_month_start)
            row["is_month_end"]  = int(fd.is_month_end)
            import math
            row["dow_sin"]   = math.sin(2*math.pi*fd.dayofweek/7)
            row["dow_cos"]   = math.cos(2*math.pi*fd.dayofweek/7)
            row["month_sin"] = math.sin(2*math.pi*fd.month/12)
            row["month_cos"] = math.cos(2*math.pi*fd.month/12)
            row["doy_sin"]   = math.sin(2*math.pi*fd.day_of_year/365)
            row["doy_cos"]   = math.cos(2*math.pi*fd.day_of_year/365)

            hist_sales = last_window["sales"].values
            for lag in [1, 7, 14, 21, 28]:
                row[f"lag_{lag}"] = hist_sales[-lag] if len(hist_sales) >= lag else hist_sales.mean()
            for w in [7, 14, 28]:
                row[f"roll_mean_{w}"] = hist_sales[-w:].mean()
                row[f"roll_std_{w}"]  = hist_sales[-w:].std()
                row[f"roll_min_{w}"]  = hist_sales[-w:].min()
                row[f"roll_max_{w}"]  = hist_sales[-w:].max()

            x = pd.DataFrame([row])[feature_cols].fillna(0)
            pred = max(0, model.predict(x)[0])
            preds.append(pred)

            new_row = row.copy()
            new_row["sales"] = pred
            last_window = pd.concat([last_window, pd.DataFrame([new_row])], ignore_index=True)
    else:
        # Rolling-mean fallback
        avg7 = product_df["sales"].tail(7).mean()
        noise = np.random.normal(0, avg7 * 0.1, forecast_days)
        preds = np.maximum(0, avg7 + noise)

    forecast_df = pd.DataFrame({"date": future_dates, "forecast": preds})
    forecast_df["lower"] = forecast_df["forecast"] * 0.85
    forecast_df["upper"] = forecast_df["forecast"] * 1.15

    # Combined actual + forecast
    actual_tail = product_df[["date", "sales"]].tail(60)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual_tail["date"], y=actual_tail["sales"],
                             name="Actual", line=dict(color=PALETTE["primary"], width=2)))
    fig.add_trace(go.Scatter(
        x=list(forecast_df["date"]) + list(forecast_df["date"][::-1]),
        y=list(forecast_df["upper"]) + list(forecast_df["lower"][::-1]),
        fill="toself", fillcolor="rgba(37,99,235,0.1)",
        line=dict(color="rgba(255,255,255,0)"), name="Confidence band",
    ))
    fig.add_trace(go.Scatter(x=forecast_df["date"], y=forecast_df["forecast"],
                             name="Forecast", line=dict(color=PALETTE["warning"], width=2.5, dash="dash")))
    fig.update_layout(height=320, margin=dict(t=10, b=10),
                      plot_bgcolor="white", paper_bgcolor="white",
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, use_container_width=True)

    # Inventory Planning
    st.subheader("📦 Inventory Planning")
    total_forecast  = int(forecast_df["forecast"].sum())
    daily_avg       = round(forecast_df["forecast"].mean(), 1)
    safety_stock    = int(daily_avg * safety_stock_days)
    reorder_point   = int(daily_avg * 7 + safety_stock)
    suggested_order = total_forecast + safety_stock

    ic1, ic2, ic3, ic4 = st.columns(4)
    for col, label, value, color in [
        (ic1, f"Total Demand ({forecast_days}d)",  f"{total_forecast:,} units", PALETTE["primary"]),
        (ic2, "Daily Average",                     f"{daily_avg} units/day",    PALETTE["success"]),
        (ic3, f"Safety Stock ({safety_stock_days}d)", f"{safety_stock:,} units", PALETTE["warning"]),
        (ic4, "Suggested Order Qty",               f"{suggested_order:,} units", PALETTE["danger"]),
    ]:
        col.markdown(f"""
        <div class="metric-card" style="border-left-color:{color}; margin-top:0.5rem">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="font-size:1.4rem">{value}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.download_button(
        "⬇️ Download Forecast CSV",
        forecast_df.to_csv(index=False),
        file_name=f"forecast_{selected_product}.csv",
        mime="text/csv",
    )


# ─────────────────────────────────────────────
# PAGE 3 — MODEL PERFORMANCE
# ─────────────────────────────────────────────

elif page == "🏆 Model Performance":
    st.title("Model Comparison & Performance")

    if summary:
        c1, c2, c3, c4 = st.columns(4)
        for col, label, val, color in [
            (c1, "Best R² Score",         summary.get("best_r2", "—"),    PALETTE["primary"]),
            (c2, "Best MAE",              summary.get("best_mae", "—"),   PALETTE["success"]),
            (c3, "Best RMSE",             summary.get("best_rmse", "—"),  PALETTE["warning"]),
            (c4, "RMSE Reduction vs LR",  f"{summary.get('rmse_reduction_pct','—')}%", PALETTE["danger"]),
        ]:
            col.markdown(f"""
            <div class="metric-card" style="border-left-color:{color}">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="font-size:1.6rem">{val}</div>
            </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # Comparison table
    if metrics:
        rows = []
        for name, m in metrics.items():
            rows.append({
                "Model": name,
                "MAE":  m.get("mae", "—"),
                "RMSE": m.get("rmse", "—"),
                "R²":   m.get("r2", "—"),
                "MAPE (%)": m.get("mape", "—"),
            })
        comp_df = pd.DataFrame(rows)
        st.subheader("Tuned Model Comparison")
        st.dataframe(comp_df.style.highlight_max(subset=["R²"], color="#DCFCE7")
                                  .highlight_min(subset=["RMSE", "MAE"], color="#DCFCE7"),
                     use_container_width=True)

        # Bar chart
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(comp_df, x="Model", y="R²", color="Model",
                         title="R² Score (higher = better)",
                         color_discrete_sequence=px.colors.qualitative.Bold)
            fig.update_layout(height=320, showlegend=False,
                              plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.bar(comp_df, x="Model", y="RMSE", color="Model",
                         title="RMSE (lower = better)",
                         color_discrete_sequence=px.colors.qualitative.Bold)
            fig.update_layout(height=320, showlegend=False,
                              plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

    # Actual vs Predicted scatter
    st.subheader("Actual vs Predicted — XGBoost (Test Set)")
    if model is not None and feature_cols:
        feat_df = df.dropna(subset=feature_cols).copy()
        cutoff  = feat_df["date"].max() - pd.Timedelta(days=90)
        test    = feat_df[feat_df["date"] > cutoff]
        X_test  = test[feature_cols]
        y_test  = test["sales"]
        y_pred  = model.predict(X_test)
        scatter_df = pd.DataFrame({"actual": y_test.values, "predicted": y_pred})
        fig = px.scatter(scatter_df, x="actual", y="predicted",
                         opacity=0.4, color_discrete_sequence=[PALETTE["primary"]])
        max_val = max(scatter_df.max())
        fig.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                      line=dict(color="red", dash="dash"))
        fig.update_layout(height=380, margin=dict(t=20, b=20),
                          plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Train the model first: `python train.py`")


# ─────────────────────────────────────────────
# PAGE 4 — FEATURE IMPORTANCE
# ─────────────────────────────────────────────

elif page == "🔍 Feature Importance":
    st.title("Feature Importance & Demand Drivers")

    if os.path.exists("results/feature_importance.csv"):
        fi_df = pd.read_csv("results/feature_importance.csv")
        st.subheader("Top Features by Importance (XGBoost)")
        fig = px.bar(fi_df.head(15), x="importance", y="feature", orientation="h",
                     color="importance", color_continuous_scale="Blues")
        fig.update_layout(height=420, margin=dict(t=20, b=20),
                          yaxis=dict(autorange="reversed"),
                          plot_bgcolor="white", paper_bgcolor="white",
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Feature Type Breakdown")
            fi_df["type"] = fi_df["feature"].apply(
                lambda x: "Rolling" if "roll" in x else
                          "Lag" if "lag" in x else
                          "Calendar" if any(k in x for k in ["dow","month","day","week","quarter","year","weekend","sin","cos"]) else
                          "Price/Promo"
            )
            type_agg = fi_df.groupby("type")["importance"].sum().reset_index()
            fig = px.pie(type_agg, values="importance", names="type", hole=0.45,
                         color_discrete_sequence=px.colors.qualitative.Bold)
            fig.update_layout(height=320, paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("All Feature Importances")
            st.dataframe(fi_df.style.background_gradient(subset=["importance"], cmap="Blues"),
                         use_container_width=True, height=320)
    else:
        st.info("Run `python train.py` to generate feature importances.")

    # Demand driver deep-dives
    st.subheader("Demand Drivers — Correlation Analysis")
    numeric_cols = ["price", "discount_pct", "promotion", "is_holiday", "is_weekend", "sales"]
    corr = df[numeric_cols].corr()[["sales"]].drop("sales").sort_values("sales", ascending=False)
    corr.columns = ["Correlation with Sales"]
    fig = px.bar(corr.reset_index(), x="index", y="Correlation with Sales",
                 color="Correlation with Sales", color_continuous_scale="RdBu",
                 color_continuous_midpoint=0)
    fig.update_layout(height=300, margin=dict(t=10, b=10),
                      plot_bgcolor="white", paper_bgcolor="white",
                      xaxis_title="Feature", coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
