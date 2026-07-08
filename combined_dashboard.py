"""
combined_dashboard.py
======================
A single Streamlit app that combines the plots from BOTH of Ben's Original
dashboards into one place, re-ordered according to the requested updates.

Run with:
    streamlit run combined_dashboard.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Ben's Original — Combined Dashboard", layout="wide")

# ---------------------------------------------------------------------------
# Shared brand styling (approximates the look of the two original dashboards)
# ---------------------------------------------------------------------------
BRAND_PRIMARY = "#1C4C74"
BRAND_SECONDARY = "#2E8B99"
BRAND_DARK = "#1F2937"
BRAND_SEQUENCE = [BRAND_PRIMARY, BRAND_SECONDARY, "#6CB4E4", "#648CAC", "#354551", "#9CA3AF"]


def _style(fig: go.Figure, title: str | None = None, y_title: str | None = None) -> go.Figure:
    layout_kwargs = {
        "margin": dict(l=10, r=10, t=50 if title else 15, b=10),
        "font": dict(family="Inter, Segoe UI, sans-serif", size=13, color=BRAND_DARK),
        "legend": dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        "colorway": BRAND_SEQUENCE,
        "hovermode": "x unified",
        "plot_bgcolor": "rgba(0,0,0,0)",
    }
    
    # CRITICAL FIX: Only apply title config if a title is explicitly passed to prevent "undefined" text
    if title:
        layout_kwargs["title"] = dict(text=title, font=dict(size=16, color=BRAND_DARK, family="Inter, Segoe UI, sans-serif"))
        
    fig.update_layout(**layout_kwargs)
    fig.update_xaxes(gridcolor="#E5E7EB", zerolinecolor="#E5E7EB")
    fig.update_yaxes(gridcolor="#E5E7EB", zerolinecolor="#E5E7EB")
    if y_title:
        fig.update_yaxes(title=y_title)
    return fig


# ---------------------------------------------------------------------------
# Dashboard 1 chart builders (row-level marketing data)
# ---------------------------------------------------------------------------
def line_trend(df, x, y, title=None, y_title=None) -> go.Figure:
    fig = px.line(df, x=x, y=y, markers=True)
    fig.update_traces(
        line=dict(color=BRAND_PRIMARY, width=2),
        marker=dict(size=5),
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>%{y:$,.0f}<extra></extra>",
    )
    if not df.empty and y in df.columns:
        max_idx = df[y].idxmax()
        if not pd.isna(max_idx):
            max_row = df.loc[max_idx]
            max_val, max_date = max_row[y], max_row[x]
            val_text = f"${max_val/1_000_000:.1f}M" if max_val > 1_000_000 else (
                f"${max_val/1_000:.1f}k" if max_val > 1_000_000 else f"${max_val:,.0f}")
            fig.add_annotation(
                x=max_date, y=max_val, text=f"Peak: {val_text}", showarrow=True,
                arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor=BRAND_DARK,
                ax=0, ay=-40, font=dict(size=11, color="white"),
                bgcolor=BRAND_DARK, bordercolor=BRAND_DARK, borderwidth=1, borderpad=4, opacity=0.9,
            )
    return _style(fig, title, y_title)


def bar_breakdown(df, x, y, title=None, orientation="v", y_title=None) -> go.Figure:
    if orientation == "h":
        fig = px.bar(df.sort_values(y), x=y, y=x, orientation="h", text_auto=".2s")
        fig.update_traces(marker_color=BRAND_PRIMARY, hovertemplate="<b>%{y}</b><br>%{x:$,.0f}<extra></extra>")
    else:
        fig = px.bar(df, x=x, y=y, text_auto=".2s")
        fig.update_traces(marker_color=BRAND_PRIMARY, hovertemplate="<b>%{x}</b><br>%{y:$,.0f}<extra></extra>")
    return _style(fig, title, y_title)


def joint_spend_sales_bar(df: pd.DataFrame, dimension: str, title: str = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df[dimension], y=df["Spend"], name="Spend", marker_color="#6CB4E4"))
    fig.add_trace(go.Bar(x=df[dimension], y=df["Estimated Sales Contribution"], name="Est. Sales", marker_color=BRAND_PRIMARY))
    fig.update_layout(barmode="group")
    return _style(fig, title, y_title="Amount ($)")


def feature_importance_bar(importance: pd.Series, value_col: str, title=None, top_n=15) -> go.Figure:
    d = importance.head(top_n).reset_index()
    d.columns = ["Feature", value_col]
    fig = px.bar(d.sort_values(value_col), x=value_col, y="Feature", orientation="h")
    fig.update_traces(marker_color=BRAND_SECONDARY)
    return _style(fig, title)


# ---------------------------------------------------------------------------
# Dashboard 2 chart builders (Ben's Original reusable_charts.py, unchanged)
# ---------------------------------------------------------------------------
def _weekly_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Week"] = pd.to_datetime(out["Week"])
    weekly = (
        out.groupby("Week")
        .agg({"Spend": "sum", "Impressions": "sum", "Clicks": "sum", "Sales": "max"})
        .reset_index()
        .sort_values("Week")
    )
    return weekly


def weekly_spend_chart(df: pd.DataFrame) -> go.Figure:
    weekly = _weekly_aggregate(df)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weekly["Week"], y=weekly["Spend"] / 1e3, name="Spend ($k)", line=dict(color="#d62728")))
    fig.update_layout(yaxis_title="Spend ($k)", height=400)
    return _style(fig, title="Weekly Marketing Spend")


def weekly_impressions_chart(df: pd.DataFrame) -> go.Figure:
    weekly = _weekly_aggregate(df)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weekly["Week"], y=weekly["Impressions"] / 1e6, name="Impressions (M)", line=dict(color="#2ca02c")))
    fig.update_layout(yaxis_title="Impressions (M)", height=400)
    return _style(fig, title="Weekly Impressions")


def spend_vs_impressions_chart(df: pd.DataFrame) -> go.Figure:
    weekly = _weekly_aggregate(df)
    r = weekly[["Spend", "Impressions"]].corr().iloc[0, 1]
    fig = px.scatter(weekly, x="Spend", y="Impressions")
    coeffs = np.polyfit(weekly["Spend"], weekly["Impressions"], 1)
    x_range = np.linspace(weekly["Spend"].min(), weekly["Spend"].max(), 50)
    fig.add_trace(go.Scatter(x=x_range, y=coeffs[0] * x_range + coeffs[1], mode="lines", name="Trend",
                              line=dict(color=BRAND_PRIMARY, dash="dash")))
    return _style(fig, title=f"Spend vs Impressions (r = {r:+.2f})")


def correlation_matrix_chart(df: pd.DataFrame) -> go.Figure:
    weekly = _weekly_aggregate(df).set_index("Week")
    corr = weekly[["Sales", "Spend", "Impressions", "Clicks"]].corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    return _style(fig, title="Correlation Matrix (Weekly Aggregated)")


def roas_per_site_chart(df: pd.DataFrame, min_spend: float = 1000) -> go.Figure:
    NON_PAID_CHANNELS = ["Brandwatch", "Infegy"]
    d = df.copy()
    d["Week"] = pd.to_datetime(d["Week"])
    d["Is_Paid"] = ~d["Channel"].isin(NON_PAID_CHANNELS)

    weekly_paid_spend = d[d["Is_Paid"]].groupby("Week")["Spend"].sum().rename("Total_Week_Paid_Spend")
    d = d.merge(weekly_paid_spend, on="Week", how="left")
    d["Total_Week_Paid_Spend"] = d["Total_Week_Paid_Spend"].fillna(0)

    paid_row_counts = d[d["Is_Paid"]].groupby("Week").size().rename("Paid_Row_Count")
    d = d.merge(paid_row_counts, on="Week", how="left")

    d["Spend_Share"] = 0.0
    has_spend_mask = d["Is_Paid"] & (d["Total_Week_Paid_Spend"] > 0)
    d.loc[has_spend_mask, "Spend_Share"] = d.loc[has_spend_mask, "Spend"] / d.loc[has_spend_mask, "Total_Week_Paid_Spend"]
    zero_spend_mask = d["Is_Paid"] & (d["Total_Week_Paid_Spend"] == 0)
    d.loc[zero_spend_mask, "Spend_Share"] = 1.0 / d.loc[zero_spend_mask, "Paid_Row_Count"]

    d["Allocated_Sales"] = d["Sales"] * d["Spend_Share"]

    site_spend = d.groupby("Site")["Spend"].sum().reset_index()
    site_sales = d.groupby("Site")["Allocated_Sales"].sum().reset_index()
    roas_df = pd.merge(site_sales, site_spend, on="Site")
    roas_paid = roas_df[roas_df["Spend"] > min_spend].copy()
    roas_paid["ROAS"] = roas_paid["Allocated_Sales"] / roas_paid["Spend"]
    roas_paid = roas_paid.sort_values("ROAS", ascending=True)

    fig = px.bar(roas_paid, x="ROAS", y="Site", orientation="h",
                 text=roas_paid["ROAS"].map(lambda v: f"{v:,.1f}x"),
                 color="ROAS", color_continuous_scale="Magma")
    fig.update_layout(coloraxis_showscale=False, height=420)
    return _style(fig, title="Return on Ad Spend (ROAS) per Site")


FORECAST_FEATURES = ["Spend", "Impressions", "Clicks", "Sales_Lag1", "Sales_Lag2", "Sales_Roll4", "Month", "WeekOfYear", "Trend"]


def _prepare_forecast_dataset(df: pd.DataFrame) -> pd.DataFrame:
    weekly = _weekly_aggregate(df).reset_index(drop=True)
    weekly = weekly[weekly["Sales"] > 0].reset_index(drop=True)
    weekly["Sales_Lag1"] = weekly["Sales"].shift(1)
    weekly["Sales_Lag2"] = weekly["Sales"].shift(2)
    weekly["Sales_Roll4"] = weekly["Sales"].shift(1).rolling(4).mean()
    weekly["Month"] = weekly["Week"].dt.month
    weekly["WeekOfYear"] = weekly["Week"].dt.isocalendar().week.astype(int)
    weekly["Trend"] = np.arange(len(weekly))
    return weekly.dropna().reset_index(drop=True)


def train_sales_forecast(df: pd.DataFrame, test_frac: float = 0.2):
    model_df = _prepare_forecast_dataset(df)
    split = int(len(model_df) * (1 - test_frac))
    train, test = model_df.iloc[:split], model_df.iloc[split:]
    X_train, y_train = train[FORECAST_FEATURES], train["Sales"]
    X_test, y_test = test[FORECAST_FEATURES], test["Sales"]
    model = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return model_df, {"model": model, "pred": pred, "train": train, "test": test}


def sales_forecast_chart(df: pd.DataFrame, test_frac: float = 0.2) -> go.Figure:
    _, results = train_sales_forecast(df, test_frac=test_frac)
    train, test = results["train"], results["test"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train["Week"], y=train["Sales"] / 1e6, name="Actual (train)", line=dict(color="lightgray")))
    fig.add_trace(go.Scatter(x=test["Week"], y=test["Sales"] / 1e6, name="Actual (test)", line=dict(color="#1f77b4", width=3)))
    fig.add_trace(go.Scatter(x=test["Week"], y=results["pred"] / 1e6, name="Linear Regression Prediction",
                              line=dict(color=BRAND_PRIMARY, dash="dash", width=3)))
    fig.update_layout(yaxis_title="Sales ($M)", height=450)
    return _style(fig)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_row_level(file) -> pd.DataFrame:
    df = pd.read_csv(file, low_memory=False)
    df["Week Starting Sunday"] = pd.to_datetime(df["Week Starting Sunday"], format="mixed")
    return df


@st.cache_data
def load_df_eda(file) -> pd.DataFrame:
    return pd.read_csv(file)


@st.cache_data
def weekly_from_row_level(df: pd.DataFrame) -> pd.DataFrame:
    weekly = (
        df.groupby("Week Starting Sunday")
        .agg(Spend=("Spend", "sum"), Impressions=("Impressions", "sum"), Clicks=("Clicks", "sum"),
             Video_Starts=("Video Starts", "sum"), Sales=("Base Dollar Amount", "max"))
        .reset_index()
        .sort_values("Week Starting Sunday")
        .reset_index(drop=True)
    )
    weekly["CTR"] = np.where(weekly["Impressions"] > 0, weekly["Clicks"] / weekly["Impressions"], 0)
    weekly = weekly[weekly["Sales"] > 0].reset_index(drop=True)
    return weekly


FEATURES = [
    "Sales_Lag_1", "Sales_Lag_2", "Sales_Lag_3", "Sales_Lag_4", "Sales_Lag_5", "Sales_Lag_6", "Sales_Lag_7", "Sales_Lag_8",
    "Sales_EWM_4", "Sales_Rolling_Mean_4", "Sales_Rolling_Mean_8", "Sales_Rolling_Min_8", "Sales_Rolling_Max_8",
    "Sales_Rolling_Std_8", "Sales_Diff_1", "Sales_Ratio_to_Mean_8", "Impressions_Lag_1", "Clicks_Lag_1", "Spend_Lag_1",
    "Video Starts", "CTR", "Month", "Quarter", "WeekOfYear_cos",
]


@st.cache_data
def build_explainability_features(weekly: pd.DataFrame) -> pd.DataFrame:
    w = weekly.copy()
    for lag in range(1, 9):
        w[f"Sales_Lag_{lag}"] = w["Sales"].shift(lag)
    w["Sales_EWM_4"] = w["Sales"].shift(1).ewm(span=4).mean()
    w["Sales_Rolling_Mean_4"] = w["Sales"].shift(1).rolling(4).mean()
    w["Sales_Rolling_Mean_8"] = w["Sales"].shift(1).rolling(8).mean()
    w["Sales_Rolling_Min_8"] = w["Sales"].shift(1).rolling(8).min()
    w["Sales_Rolling_Max_8"] = w["Sales"].shift(1).rolling(8).max()
    w["Sales_Rolling_Std_8"] = w["Sales"].shift(1).rolling(8).std()
    w["Sales_Diff_1"] = w["Sales"].shift(1) - w["Sales"].shift(2)
    w["Sales_Ratio_to_Mean_8"] = w["Sales"].shift(1) / w["Sales_Rolling_Mean_8"]
    w["Impressions_Lag_1"] = w["Impressions"].shift(1)
    w["Clicks_Lag_1"] = w["Clicks"].shift(1)
    w["Spend_Lag_1"] = w["Spend"].shift(1)
    w["Video Starts"] = w["Video_Starts"]
    w["Month"] = w["Week Starting Sunday"].dt.month
    w["Quarter"] = w["Week Starting Sunday"].dt.quarter
    week_of_year = w["Week Starting Sunday"].dt.isocalendar().week.astype(int)
    w["WeekOfYear_cos"] = np.cos(2 * np.pi * week_of_year / 52)
    return w.dropna(subset=FEATURES + ["Sales"]).reset_index(drop=True)


@st.cache_resource
def train_explainability_models(model_df: pd.DataFrame):
    import xgboost as xgb
    import shap

    X = model_df[FEATURES]
    y = np.log(model_df["Sales"])

    lin = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
    lin.fit(X, y)
    coefs = pd.Series(lin.named_steps["model"].coef_, index=FEATURES).sort_values(key=abs, ascending=False)

    xgb_model = xgb.XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
    xgb_model.fit(X, y)
    importances = pd.Series(xgb_model.feature_importances_, index=FEATURES).sort_values(ascending=False)

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X)
    shap_importance = pd.Series(np.abs(shap_values).mean(axis=0), index=FEATURES).sort_values(ascending=False)

    return coefs, importances, shap_values, shap_importance, X


def shap_beeswarm_figure(shap_values, X, max_display=15):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    plt.figure()
    shap.summary_plot(shap_values, X, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Sidebar — data source
# ---------------------------------------------------------------------------
st.sidebar.title("Data Source")
st.sidebar.caption("Defaults to CSVs in the same folder as this script. Upload your own to override.")

row_level_upload = st.sidebar.file_uploader("row_level_clean.csv (Dashboard 1)", type="csv")
df_eda_upload = st.sidebar.file_uploader("df_eda.csv (Dashboard 2)", type="csv")

row_level_source = row_level_upload if row_level_upload is not None else "row_level_clean.csv"
df_eda_source = df_eda_upload if df_eda_upload is not None else "df_eda.csv"

st.title("Ben's Original — Combined Marketing Analytics Dashboard")
st.caption("Combines the plots from both original Streamlit dashboards into a single app.")

# ---------------------------------------------------------------------------
# Target Dashboard Structure Setup
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "📊 Executive Summary",
    "📈 Exploratory Data Analysis",
    "🤖 Model & Explainability"
])

# Try loading both data files up front so metrics/plots can populate their tabs seamlessly
try:
    row_df = load_row_level(row_level_source)
    weekly = weekly_from_row_level(row_df)
except FileNotFoundError:
    row_df = None
    weekly = None

try:
    eda_df = load_df_eda(df_eda_source)
except FileNotFoundError:
    eda_df = None


# ===========================================================================
# TAB 1 — Executive Summary
# ===========================================================================
with tab1:
    # --- Data Quality Overview (Forced at the top) -------------------------
    st.subheader("Data Quality Overview")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Raw Rows", "178,151")
    m2.metric("Columns", "29")
    m3.metric("Distinct Weeks", "219")
    m4.metric("Duplicates", "38,245")
    m5.metric("Duplicate Percentage", "21.46%")

    st.markdown("---")

    if row_df is None:
        st.warning("Couldn't find **row_level_clean.csv** next to this script. Upload it in the sidebar to load elements from Dashboard 1.")
    else:
        # Row 1: Distribution Breakdowns
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Spend Share by Channel")
            channel_share = row_df.groupby("Channel", as_index=False)["Spend"].sum()
            fig_donut = px.pie(channel_share, values="Spend", names="Channel", hole=0.6, color_discrete_sequence=BRAND_SEQUENCE)
            fig_donut.update_traces(textposition="inside", textinfo="none")
            fig_donut.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.05, xanchor="center", x=0.5))
            st.plotly_chart(_style(fig_donut), use_container_width=True, key="exec_spend_share_donut")
            
        with c2:
            st.markdown("##### Estimated Sales Contribution by Media Type")
            mt = row_df.groupby("Media Type", as_index=False)["Estimated Sales Contribution"].sum()
            st.plotly_chart(
                bar_breakdown(mt, "Media Type", "Estimated Sales Contribution"),
                use_container_width=True, key="exec_media_type",
            )

        st.markdown("---")

        # Row 2: Performance Leaders (FIXED: Explicit positional argument title maps to layout title properly now)
        st.subheader("Performance Leaders Breakdown")
        ldr1, ldr2 = st.columns(2)
        with ldr1:
            st.markdown("##### Highest Attributed Sales")
            sales_by_chan = row_df.groupby("Channel", as_index=False)["Estimated Sales Contribution"].sum()
            fig_highest_sales = bar_breakdown(sales_by_chan, x="Channel", y="Estimated Sales Contribution", orientation="h")
            st.plotly_chart(fig_highest_sales, use_container_width=True, key="exec_highest_sales")
            
        with ldr2:
            st.markdown("##### Highest Marketing Spend")
            spend_by_chan = row_df.groupby("Channel", as_index=False)["Spend"].sum()
            fig_highest_spend = bar_breakdown(spend_by_chan, x="Channel", y="Spend", orientation="h")
            st.plotly_chart(fig_highest_spend, use_container_width=True, key="exec_highest_spend")

    st.markdown("---")

    # --- Consolidated Trends View -----------------------------------------
    st.subheader("Weekly System Trends")
    if weekly is not None:
        st.markdown("##### Weekly Sales Trend (company-wide)")
        st.plotly_chart(
            line_trend(weekly, "Week Starting Sunday", "Sales", y_title="Sales ($)"),
            use_container_width=True, key="exec_sales_trend",
        )

    if eda_df is not None:
        r1, r2 = st.columns(2)
        with r1:
            st.plotly_chart(weekly_spend_chart(eda_df), use_container_width=True, key="rc_spend")
        with r2:
            st.plotly_chart(weekly_impressions_chart(eda_df), use_container_width=True, key="rc_impressions")


# ===========================================================================
# TAB 2 — Exploratory Data Analysis
# ===========================================================================
with tab2:
    if eda_df is not None:
        st.subheader("Exploratory Trend & Correlation Analysis")
        
        # Original scatter plots
        r3, r4 = st.columns(2)
        with r3:
            st.plotly_chart(spend_vs_impressions_chart(eda_df), use_container_width=True, key="rc_scatter")
        with r4:
            st.plotly_chart(correlation_matrix_chart(eda_df), use_container_width=True, key="rc_corr")

        # New requested relationship views (Spend vs Clicks & Impressions vs Clicks)
        st.markdown("#### Click Performance Diagnostics")
        r_clicks1, r_clicks2 = st.columns(2)
        weekly_eda_agg = _weekly_aggregate(eda_df)
        
        with r_clicks1:
            r_sc = weekly_eda_agg[["Spend", "Clicks"]].corr().iloc[0, 1]
            fig_sc = px.scatter(weekly_eda_agg, x="Spend", y="Clicks")
            fig_sc.update_traces(marker=dict(color=BRAND_SECONDARY))
            
            if not weekly_eda_agg.empty:
                coeffs_sc = np.polyfit(weekly_eda_agg["Spend"], weekly_eda_agg["Clicks"], 1)
                x_range_sc = np.linspace(weekly_eda_agg["Spend"].min(), weekly_eda_agg["Spend"].max(), 50)
                fig_sc.add_trace(go.Scatter(x=x_range_sc, y=coeffs_sc[0] * x_range_sc + coeffs_sc[1], 
                                            mode="lines", name="Trend", line=dict(color=BRAND_PRIMARY, dash="dash")))
            st.plotly_chart(_style(fig_sc, title=f"Spend vs Clicks (r = {r_sc:+.2f})"), use_container_width=True, key="spend_vs_clicks")
            
        with r_clicks2:
            r_ic = weekly_eda_agg[["Impressions", "Clicks"]].corr().iloc[0, 1]
            fig_ic = px.scatter(weekly_eda_agg, x="Impressions", y="Clicks")
            fig_ic.update_traces(marker=dict(color=BRAND_PRIMARY))
            
            if not weekly_eda_agg.empty:
                coeffs_ic = np.polyfit(weekly_eda_agg["Impressions"], weekly_eda_agg["Clicks"], 1)
                x_range_ic = np.linspace(weekly_eda_agg["Impressions"].min(), weekly_eda_agg["Impressions"].max(), 50)
                fig_ic.add_trace(go.Scatter(x=x_range_ic, y=coeffs_ic[0] * x_range_ic + coeffs_ic[1], 
                                            mode="lines", name="Trend", line=dict(color=BRAND_SECONDARY, dash="dash")))
            st.plotly_chart(_style(fig_ic, title=f"Impressions vs Clicks (r = {r_ic:+.2f})"), use_container_width=True, key="impressions_vs_clicks")

    if row_df is not None:
        # --- Media Efficiency Analysis ----------------------------------------------
        st.subheader("Media Efficiency Analysis")

        def efficiency_table(dim: str) -> pd.DataFrame:
            g = row_df.groupby(dim, as_index=False).agg(
                Spend=("Spend", "sum"), Impressions=("Impressions", "sum"),
                Clicks=("Clicks", "sum"), Engagements=("Engagements", "sum"),
                **{"Estimated Sales Contribution": ("Estimated Sales Contribution", "sum")},
            )
            g["CTR"] = np.where(g["Impressions"] > 0, g["Clicks"] / g["Impressions"], np.nan)
            g["CPC"] = np.where(g["Clicks"] > 0, g["Spend"] / g["Clicks"], np.nan)
            return g

        eff_tabs = st.tabs(["Media Type", "Channel", "Sub-Channel", "Site", "Device"])

        with eff_tabs[0]:
            g = efficiency_table("Media Type")
            st.dataframe(g, use_container_width=True)
            st.plotly_chart(
                joint_spend_sales_bar(g, "Media Type", "Spend vs Est. Sales by Media Type"),
                use_container_width=True, key="eff_media_type_joint",
            )
        with eff_tabs[1]:
            g = efficiency_table("Channel")
            st.dataframe(g, use_container_width=True)
            plot_data = g.dropna(subset=["CTR"])
            st.plotly_chart(
                bar_breakdown(plot_data, "Channel", "CTR", "Click-Through Rate (CTR) by Channel"),
                use_container_width=True, key="eff_channel_ctr",
            )
        with eff_tabs[2]:
            g = efficiency_table("Sub-Channel")
            st.dataframe(g, use_container_width=True)
            plot_data = g.dropna(subset=["CPC"])
            st.plotly_chart(
                bar_breakdown(plot_data, "Sub-Channel", "CPC", "Cost Per Click (CPC) by Sub-Channel"),
                use_container_width=True, key="eff_subchannel_cpc",
            )
        with eff_tabs[3]:
            g = efficiency_table("Site")
            st.dataframe(g, use_container_width=True)
            plot_data = g.dropna(subset=["CPC"])
            st.plotly_chart(
                bar_breakdown(plot_data, "Site", "CPC", "Cost Per Click (CPC) by Site"),
                use_container_width=True, key="eff_site_cpc",
            )
        with eff_tabs[4]:
            g = efficiency_table("Device")
            st.dataframe(g, use_container_width=True)
            plot_data = g.dropna(subset=["CPC"])
            st.plotly_chart(
                bar_breakdown(plot_data, "Device", "CPC", "Cost Per Click (CPC) by Device"),
                use_container_width=True, key="eff_device_cpc",
            )

        # --- Sales Drivers -------------------------------------------------------
        st.subheader("Sales Drivers Breakdown")
        d1, d2 = st.columns(2)
        with d1:
            mt2 = row_df.groupby("Media Type", as_index=False)["Estimated Sales Contribution"].sum()
            st.plotly_chart(
                bar_breakdown(mt2, "Media Type", "Estimated Sales Contribution", "Sales by Media Type", orientation="h"),
                use_container_width=True, key="drivers_media_type",
            )
        with d2:
            ch2 = row_df.groupby("Channel", as_index=False)["Estimated Sales Contribution"].sum()
            st.plotly_chart(
                bar_breakdown(ch2, "Channel", "Estimated Sales Contribution", "Sales by Channel", orientation="h"),
                use_container_width=True, key="drivers_channel",
            )

        # --- Top 20 Creatives ------------------------------------------------------
        st.subheader("Top 20 Creatives by Estimated Sales Contribution")
        top20 = (
            row_df.groupby("Creative", as_index=False)["Estimated Sales Contribution"]
            .sum().sort_values("Estimated Sales Contribution", ascending=False).head(20)
        )
        st.plotly_chart(
            bar_breakdown(top20, "Creative", "Estimated Sales Contribution", orientation="h"),
            use_container_width=True, key="top20_creatives",
        )

    if eda_df is not None:
        st.markdown("### Return on Ad Spend (ROAS) per Site")
        min_spend = st.slider("Minimum total spend to include a site", min_value=0, max_value=50000, value=1000, step=500)
        st.plotly_chart(roas_per_site_chart(eda_df, min_spend=min_spend), use_container_width=True, key="rc_roas")


# ===========================================================================
# TAB 3 — Model & Explainability
# ===========================================================================
with tab3:
    # --- Simple Spend-to-Sales Estimator -------------------------------------
    if weekly is not None:
        st.subheader("🧮 Simple Spend-to-Sales Estimator")
        st.caption("A naive linear estimate based purely on the historical weekly correlation (ignoring momentum and other features).")
        coeffs = np.polyfit(weekly["Spend"], weekly["Sales"], 1)
        slope, intercept = coeffs[0], coeffs[1]
        spend_input = st.number_input("Enter hypothetical Weekly Spend ($)", min_value=0, value=int(weekly["Spend"].median()), step=1000)
        est_sales = slope * spend_input + intercept
        st.metric("Estimated Weekly Sales", f"${est_sales:,.0f}")
        st.caption(f"Based on naive linear trend (y = {slope:.2f}x + {intercept:,.0f})")

    # --- Sales Forecast (Linear Regression) ----------------------------------
    if eda_df is not None:
        st.markdown("### Sales Forecast Model (Linear Regression)")
        test_frac = st.slider("Test set fraction", min_value=0.1, max_value=0.4, value=0.2, step=0.05, key="rc_test_frac")
        _, results = train_sales_forecast(eda_df, test_frac=test_frac)
        c1, c2, c3 = st.columns(3)
        y_test = results["test"]["Sales"]
        pred = results["pred"]
        mae = float(np.mean(np.abs(y_test - pred)))
        rmse = float(np.sqrt(np.mean((y_test - pred) ** 2)))
        r2 = float(1 - np.sum((y_test - pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2))
        c1.metric("MAE", f"${mae:,.0f}")
        c2.metric("RMSE", f"${rmse:,.0f}")
        c3.metric("R²", f"{r2:.3f}")
        st.plotly_chart(sales_forecast_chart(eda_df, test_frac=test_frac), use_container_width=True, key="rc_forecast")

    # --- Advanced Explainability ---------------------------------------------
    if weekly is not None:
        st.subheader("Explainability — What Drives Sales?")
        st.caption(
            "Trained here on the fly (Linear Regression + XGBoost, log-Sales space) using weekly lag/rolling "
            "features derived from row_level_clean.csv."
        )
        top_n_features = st.slider("Number of top features to display", min_value=5, max_value=24, value=15)

        model_df = build_explainability_features(weekly)
        if len(model_df) < 10:
            st.info("Not enough weekly history after feature engineering to train the explainability models.")
        else:
            coefs, importances, shap_values, shap_importance, X = train_explainability_models(model_df)

            e1, e2 = st.columns(2)
            with e1:
                st.markdown("**📐 Linear Model: Standardized Coefficients**")
                st.plotly_chart(
                    feature_importance_bar(coefs, "Standardized Coefficient", "Top Linear Drivers (log-Sales space)", top_n=top_n_features),
                    use_container_width=True, key="exp_linear",
                )
            with e2:
                st.markdown("**🌳 XGBoost: Feature Importances**")
                st.plotly_chart(
                    feature_importance_bar(importances, "Importance", "Top XGBoost Drivers", top_n=top_n_features),
                    use_container_width=True, key="exp_xgb",
                )

            st.markdown("**🧬 SHAP Global Feature Importance (XGBoost)**")
            s1, s2 = st.columns(2)
            with s1:
                st.plotly_chart(
                    feature_importance_bar(shap_importance, "Mean |SHAP value|", "Mean |SHAP value| (log-Sales space)", top_n=top_n_features),
                    use_container_width=True, key="exp_shap_bar",
                )
            with s2:
                fig = shap_beeswarm_figure(shap_values, X, max_display=top_n_features)
                st.pyplot(fig)