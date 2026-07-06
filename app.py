"""
Ben's Original — Marketing Analytics & Sales Forecasting Dashboard
====================================================================
Run locally with:
    pip install -r requirements.txt
    streamlit run app.py

Expects `df_eda.csv` and `master_synthesized_data.csv` in the same folder.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Ben's Original | Marketing Analytics",
    page_icon="📈",
    layout="wide",
)

PRIMARY = "#B8341B"   # Ben's Original brand-adjacent warm red
ACCENT = "#F4A300"
BG_GRID = "#EDEDED"

px.defaults.color_discrete_sequence = px.colors.qualitative.Set2
px.defaults.template = "plotly_white"


# ----------------------------------------------------------------------------
# DATA LOADING
# ----------------------------------------------------------------------------
@st.cache_data
def load_eda_data():
    df = pd.read_csv("df_eda.csv")
    df["Week"] = pd.to_datetime(df["Week"])
    return df


@st.cache_data
def load_master_data():
    df = pd.read_csv("master_synthesized_data.csv")
    df["Week"] = pd.to_datetime(df["Week"])
    return df


@st.cache_data
def build_weekly_timeline(master_df: pd.DataFrame):
    """Collapse the site-level master file into one row per week and
    drop trailing weeks where Sales == 0 (a data-collection gap, not a
    real zero-sales week)."""
    weekly = (
        master_df.groupby("Week")
        .agg({"Spend": "sum", "Impressions": "sum", "Clicks": "sum", "Sales": "max"})
        .reset_index()
        .sort_values("Week")
        .reset_index(drop=True)
    )
    n_before = len(weekly)
    weekly = weekly[weekly["Sales"] > 0].reset_index(drop=True)
    dropped = n_before - len(weekly)
    return weekly, dropped


@st.cache_data
def engineer_features(weekly: pd.DataFrame):
    df = weekly.copy()
    df["Sales_Lag1"] = df["Sales"].shift(1)
    df["Sales_Lag2"] = df["Sales"].shift(2)
    df["Sales_Roll4"] = df["Sales"].shift(1).rolling(4).mean()
    df["Month"] = df["Week"].dt.month
    df["WeekOfYear"] = df["Week"].dt.isocalendar().week.astype(int)
    df["Trend"] = np.arange(len(df))
    return df.dropna().reset_index(drop=True)


FEATURES = [
    "Spend", "Impressions", "Clicks",
    "Sales_Lag1", "Sales_Lag2", "Sales_Roll4",
    "Month", "WeekOfYear", "Trend",
]


@st.cache_resource
def train_models(model_df: pd.DataFrame, test_frac: float = 0.2):
    split = int(len(model_df) * (1 - test_frac))
    train, test = model_df.iloc[:split], model_df.iloc[split:]

    X_train, y_train = train[FEATURES], train["Sales"]
    X_test, y_test = test[FEATURES], test["Sales"]

    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=42),
    }

    results = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results[name] = {
            "model": model,
            "pred": pred,
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "R2": r2_score(y_test, pred),
            "MAPE": float(np.mean(np.abs((y_test - pred) / y_test)) * 100),
        }

    # Naive persistence baseline: "predict last week's sales"
    naive_pred = test["Sales_Lag1"].values
    results["Naive Baseline (last week's sales)"] = {
        "model": None,
        "pred": naive_pred,
        "MAE": mean_absolute_error(y_test, naive_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, naive_pred)),
        "R2": r2_score(y_test, naive_pred),
        "MAPE": float(np.mean(np.abs((y_test - naive_pred) / y_test)) * 100),
    }

    return results, train, test, X_train, y_train


# ----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------------------------------------------
st.sidebar.title("🥫 Ben's Original")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "📊 EDA Dashboard", "🤖 Sales Forecasting Model"],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Data: weekly marketing activity (Spend, Impressions, Clicks) "
    "and Sales across Paid Social, Display and TV, Jan 2023 – early 2025."
)

df_eda = load_eda_data()
df_master = load_master_data()
weekly_raw, dropped_weeks = build_weekly_timeline(df_master)
model_df = engineer_features(weekly_raw)

# ============================================================================
# PAGE 1 — OVERVIEW
# ============================================================================
if page == "🏠 Overview":
    st.title("Ben's Original — Marketing & Sales Analytics")
    st.markdown(
        """
        This dashboard summarizes marketing performance for **Ben's Original**
        and forecasts future weekly sales from marketing activity.

        Use the sidebar to move between:
        - **EDA Dashboard** — explore historic sales, spend efficiency, and channel performance
        - **Sales Forecasting Model** — see how the predictive model works, its accuracy, and try it on hypothetical spend
        """
    )

    c1, c2, c3, c4 = st.columns(4)
    total_sales = df_eda.groupby(["Week", "Site"])["Sales"].max().reset_index()
    total_sales = total_sales.groupby("Week")["Sales"].max().sum()
    total_spend = df_eda["Spend"].sum()
    total_impr = df_eda["Impressions"].sum()
    weeks_covered = df_eda["Week"].nunique()

    c1.metric("Total Sales (period)", f"${total_sales/1e6:,.1f}M")
    c2.metric("Total Marketing Spend", f"${total_spend/1e6:,.1f}M")
    c3.metric("Total Impressions", f"{total_impr/1e9:,.2f}B")
    c4.metric("Weeks of Data", f"{weeks_covered}")

    st.markdown("---")
    st.subheader("What's in the data?")
    st.markdown(
        f"""
        - **Granularity:** weekly, broken down by Site (e.g. Facebook/Instagram, YouTube, Amazon, TV networks) and Device
        - **Timeframe:** {df_eda['Week'].min().date()} to {df_eda['Week'].max().date()}
        - **Cleaning applied:** missing Impressions/Clicks/Spend were synthesized from historical
          median CTR/CPM ratios per Channel × Site, so every paid row has internally consistent economics
          (no "spend with zero impressions" or "impressions with zero spend" rows)
        - **Note:** the last {dropped_weeks} weeks in the raw feed have Sales = 0, which reflects a data
          collection gap rather than a real drop to zero — these are excluded from the forecasting model
          so they don't distort accuracy.
        """
    )

# ============================================================================
# PAGE 2 — EDA DASHBOARD
# ============================================================================
elif page == "📊 EDA Dashboard":
    st.title("📊 Exploratory Data Analysis")
    st.caption("Source: `df_eda.csv` — Week × Site × Device level, 2023–2025")

    sites_all = sorted(df_eda["Site"].unique())
    with st.expander("🔧 Filters", expanded=False):
        selected_sites = st.multiselect("Filter by Site (optional)", sites_all, default=[])
    df_f = df_eda[df_eda["Site"].isin(selected_sites)] if selected_sites else df_eda

    # ---- 1. Company sales during weeks each site was active -----------------
    st.markdown("### 1. Average Company Sales During Weeks Each Site Was Active")
    site_weekly_sales = df_f.groupby(["Week", "Site"])["Sales"].max().reset_index()
    site_weekly_sales_active = site_weekly_sales[site_weekly_sales["Sales"] > 0]
    site_summary = (
        site_weekly_sales_active.groupby("Site")
        .agg(Avg_Sales=("Sales", "mean"), Weeks_Active=("Sales", "count"))
        .reset_index()
        .sort_values("Avg_Sales", ascending=True)
    )
    site_summary["Avg_Sales_M"] = site_summary["Avg_Sales"] / 1e6
    fig1 = px.bar(
        site_summary, x="Avg_Sales_M", y="Site", orientation="h",
        text=site_summary["Avg_Sales_M"].map(lambda v: f"${v:,.1f}M"),
        color="Avg_Sales_M", color_continuous_scale="Viridis",
        hover_data={"Weeks_Active": True, "Avg_Sales_M": ":.1f"},
        labels={"Avg_Sales_M": "Average Weekly Company Sales ($ Millions)"},
    )
    fig1.update_layout(coloraxis_showscale=False, height=450)
    st.plotly_chart(fig1, use_container_width=True)
    st.info(
        "📌 **Reading this chart:** Sales is a single **company-wide weekly figure**, not something "
        "tracked per site — so it can't be summed across sites without double-counting (a site active "
        "in 100 different weeks would rack up ~100x the true company total). To avoid that distortion, "
        "this chart shows the **average** weekly company sales during the weeks each site was active "
        "(hover a bar to see how many weeks that covers). It tells you which sites tended to be running "
        "during stronger sales periods — it is **not** attributing that revenue to the site itself. "
        "The KPI on the Overview page is the true, correctly deduplicated total across all weeks."
    )

    st.markdown("---")

    # ---- 2. ROAS per site --------------------------------------------------
    st.markdown("### 2. Return on Ad Spend (ROAS) per Site")
    site_spend = df_f.groupby("Site")["Spend"].sum().reset_index()
    site_sales = site_weekly_sales.groupby("Site")["Sales"].sum().reset_index()
    roas_df = pd.merge(site_sales, site_spend, on="Site")
    roas_paid = roas_df[roas_df["Spend"] > 1000].copy()
    roas_paid["ROAS"] = roas_paid["Sales"] / roas_paid["Spend"]
    roas_paid = roas_paid.sort_values("ROAS", ascending=True)
    fig2 = px.bar(
        roas_paid, x="ROAS", y="Site", orientation="h",
        text=roas_paid["ROAS"].map(lambda v: f"{v:,.1f}x"),
        color="ROAS", color_continuous_scale="Magma",
    )
    fig2.update_layout(coloraxis_showscale=False, height=420)
    st.plotly_chart(fig2, use_container_width=True)
    st.warning(
        "⚠️ **Caveat:** ROAS = total company sales ÷ site spend. Because Sales isn't tracked "
        "per-site in this dataset, sites with very small spend can show enormous, misleading ROAS. "
        "Only sites with >$1,000 total spend are shown to reduce (not eliminate) this distortion — "
        "treat these as *directional* efficiency signals, not exact attribution."
    )

    st.markdown("---")

    # ---- 3. Spend by channel (pie) + top sites by spend --------------------
    st.markdown("### 3. Spend Allocation")
    col_a, col_b = st.columns(2)
    with col_a:
        spend_by_channel = df_f.groupby("Channel")["Spend"].sum().reset_index()
        spend_by_channel = spend_by_channel[spend_by_channel["Spend"] > 0]
        fig3 = px.pie(spend_by_channel, names="Channel", values="Spend", hole=0.4,
                      title="Total Spend by Channel Type")
        st.plotly_chart(fig3, use_container_width=True)
    with col_b:
        top_spend_sites = (
            df_f.groupby("Site")["Spend"].sum().reset_index()
            .sort_values("Spend", ascending=False).head(10)
        )
        fig3b = px.bar(top_spend_sites, x="Spend", y="Site", orientation="h",
                       title="Top 10 Sites by Total Spend", color_discrete_sequence=[PRIMARY])
        fig3b.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig3b, use_container_width=True)
    st.info(
        "📌 Spend is concentrated in a handful of channels/sites — useful for spotting where "
        "budget is committed versus where it's spread thin."
    )

    st.markdown("---")

    # ---- 4. Weekly trends ---------------------------------------------------
    st.markdown("### 4. Weekly Trends: Sales, Spend & Impressions")
    weekly_df = df_f.groupby("Week").agg(
        {"Sales": "max", "Spend": "sum", "Impressions": "sum"}
    ).reset_index().sort_values("Week")
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=weekly_df["Week"], y=weekly_df["Sales"]/1e6,
                               name="Sales ($M)", line=dict(color="#1f77b4")))
    fig4.update_layout(yaxis_title="Sales ($M)", height=300, title="Weekly Sales")
    st.plotly_chart(fig4, use_container_width=True)

    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(x=weekly_df["Week"], y=weekly_df["Spend"]/1e3,
                               name="Spend ($k)", line=dict(color="#d62728")))
    fig5.update_layout(yaxis_title="Spend ($k)", height=300, title="Weekly Marketing Spend")
    st.plotly_chart(fig5, use_container_width=True)

    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(x=weekly_df["Week"], y=weekly_df["Impressions"]/1e6,
                               name="Impressions (M)", line=dict(color="#2ca02c")))
    fig6.update_layout(yaxis_title="Impressions (M)", height=300, title="Weekly Impressions")
    st.plotly_chart(fig6, use_container_width=True)
    st.info(
        "📌 Watch for lag: spend and impression spikes often *precede* sales spikes rather than "
        "moving in perfect lockstep — that's the intuition behind adding lag features in the "
        "forecasting model."
    )

    st.markdown("---")

    # ---- 5. Stacked spend by site / channel ---------------------------------
    st.markdown("### 5. Weekly Spend Composition")
    weekly_spend_site = df_f.groupby(["Week", "Site"])["Spend"].sum().unstack(fill_value=0)
    top6 = df_f.groupby("Site")["Spend"].sum().sort_values(ascending=False).index[:6]
    site_plot_df = weekly_spend_site[top6].copy()
    other_sites = [c for c in weekly_spend_site.columns if c not in top6]
    if other_sites:
        site_plot_df["Other Sites"] = weekly_spend_site[other_sites].sum(axis=1)
    site_plot_df = site_plot_df / 1e6
    fig7 = px.area(site_plot_df, labels={"value": "Spend ($M)", "Week": "Week"},
                   title="Weekly Spend Stacked by Site")
    st.plotly_chart(fig7, use_container_width=True)

    weekly_spend_channel = df_f.groupby(["Week", "Channel"])["Spend"].sum().unstack(fill_value=0) / 1e6
    fig8 = px.area(weekly_spend_channel, labels={"value": "Spend ($M)", "Week": "Week"},
                   title="Weekly Spend Stacked by Channel")
    st.plotly_chart(fig8, use_container_width=True)

    st.markdown("---")

    # ---- 6. Correlation heatmap --------------------------------------------
    st.markdown("### 6. Correlation Between Weekly Metrics")
    weekly_metrics = df_f.groupby("Week").agg(
        {"Sales": "max", "Spend": "sum", "Impressions": "sum", "Clicks": "sum"}
    )
    corr = weekly_metrics.corr()
    fig9 = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                     title="Correlation Matrix (Weekly Aggregated)")
    st.plotly_chart(fig9, use_container_width=True)

    col_c, col_d, col_e = st.columns(3)
    for col, (x, y, title) in zip(
        [col_c, col_d, col_e],
        [("Spend", "Impressions", "Spend vs Impressions"),
         ("Impressions", "Sales", "Impressions vs Sales"),
         ("Spend", "Sales", "Spend vs Sales")],
    ):
        r = weekly_metrics[[x, y]].corr().iloc[0, 1]
        wm = weekly_metrics.reset_index()
        fig = px.scatter(wm, x=x, y=y, title=f"{title} (r = {r:+.2f})")
        # manual linear trendline (avoids requiring statsmodels)
        coeffs = np.polyfit(wm[x], wm[y], 1)
        x_range = np.linspace(wm[x].min(), wm[x].max(), 50)
        fig.add_trace(go.Scatter(x=x_range, y=coeffs[0] * x_range + coeffs[1],
                                  mode="lines", name="Trend", line=dict(color=PRIMARY, dash="dash")))
        col.plotly_chart(fig, use_container_width=True)
    st.info(
        "📌 **Interpretation:** Spend correlates strongly with Impressions (more budget buys more "
        "reach, as expected). The link from Impressions/Spend to Sales is positive but moderate — "
        "sales are also driven by factors outside this dataset (seasonality, pricing, distribution, "
        "competitor activity), which is exactly why the forecasting model leans on **lag features** "
        "(recent sales momentum) in addition to marketing inputs."
    )

    st.markdown("---")

    # ---- 7. Efficiency metrics ------------------------------------------------
    st.markdown("### 7. Site Efficiency: CTR, CPC, CPM (Top 10 by Spend)")
    site_totals = df_f.groupby("Site").agg(
        {"Spend": "sum", "Impressions": "sum", "Clicks": "sum"}
    ).reset_index()
    site_totals = site_totals[(site_totals["Impressions"] > 0) & (site_totals["Clicks"] > 0)].copy()
    site_totals["CTR_%"] = site_totals["Clicks"] / site_totals["Impressions"] * 100
    site_totals["CPC"] = site_totals["Spend"] / site_totals["Clicks"]
    site_totals["CPM"] = site_totals["Spend"] / (site_totals["Impressions"] / 1000)
    top10 = site_totals.sort_values("Spend", ascending=False).head(10)

    tab1, tab2, tab3 = st.tabs(["CTR %", "CPC ($)", "CPM ($)"])
    with tab1:
        f = px.bar(top10.sort_values("CTR_%"), x="CTR_%", y="Site", orientation="h",
                   color_discrete_sequence=["#1f77b4"])
        st.plotly_chart(f, use_container_width=True)
    with tab2:
        f = px.bar(top10.sort_values("CPC", ascending=False), x="CPC", y="Site", orientation="h",
                   color_discrete_sequence=["#2ca02c"])
        st.plotly_chart(f, use_container_width=True)
    with tab3:
        f = px.bar(top10.sort_values("CPM", ascending=False), x="CPM", y="Site", orientation="h",
                   color_discrete_sequence=["#e377c2"])
        st.plotly_chart(f, use_container_width=True)
    st.info(
        "📌 Lower CPC/CPM is generally more efficient; higher CTR generally signals stronger "
        "creative/audience fit. Compare these against the ROAS chart above to spot sites that are "
        "cheap to run *and* tied to strong sales periods."
    )

# ============================================================================
# PAGE 3 — MODEL
# ============================================================================
else:
    st.title("🤖 Sales Forecasting Model")
    st.caption("Source: `master_synthesized_data.csv`, aggregated to a weekly company-level timeline")

    st.markdown(
        """
        ### The problem
        Predict **next week's total company Sales** using that week's planned marketing
        activity (Spend, Impressions, Clicks) plus recent sales momentum.

        ### Why lag features matter
        The EDA showed marketing spend explains only part of the variation in Sales
        (moderate correlation, r ≈ 0.37–0.41). The rest is driven by momentum, seasonality,
        and factors outside this dataset. So alongside the marketing inputs, the model is
        given:
        - **Sales_Lag1 / Sales_Lag2** — sales from the previous 1–2 weeks
        - **Sales_Roll4** — the trailing 4-week average sales (smooths out noise)
        - **Month / Week-of-year** — seasonality signals
        - **Trend** — a simple week index to capture long-run drift
        """
    )

    st.markdown(f"""
    **Data used for training:** {len(model_df)} weeks (out of {len(weekly_raw) + dropped_weeks} raw weeks —
    excluding {dropped_weeks} trailing weeks with a Sales data gap, and the first 2 weeks used only to
    build the initial lag features).
    """)

    with st.expander("👀 Preview the model-ready dataset"):
        st.dataframe(model_df[["Week", "Sales"] + FEATURES], use_container_width=True)

    results, train, test, X_train, y_train = train_models(model_df)

    st.markdown("---")
    st.subheader("Model Comparison")
    st.markdown(
        "Models are trained on the first 80% of weeks (chronologically) and evaluated on "
        "the most recent 20% — this mimics genuinely forecasting the future rather than "
        "interpolating the past. A **naive baseline** (\"just predict last week's sales\") "
        "is included so the models have to prove they add real value."
    )

    metrics_df = pd.DataFrame({
        name: {"MAE ($)": r["MAE"], "RMSE ($)": r["RMSE"], "R²": r["R2"], "MAPE (%)": r["MAPE"]}
        for name, r in results.items()
    }).T.round(2)
    metrics_df["MAE ($)"] = metrics_df["MAE ($)"].map(lambda v: f"${v:,.0f}")
    metrics_df["RMSE ($)"] = metrics_df["RMSE ($)"].map(lambda v: f"${v:,.0f}")
    st.dataframe(metrics_df, use_container_width=True)

    best_name = min(
        [k for k in results if k != "Naive Baseline (last week's sales)"],
        key=lambda k: results[k]["MAPE"],
    )
    naive_mape = results["Naive Baseline (last week's sales)"]["MAPE"]
    best_mape = results[best_name]["MAPE"]
    if best_mape < naive_mape:
        st.success(
            f"✅ **{best_name}** is the best performer (MAPE {best_mape:.2f}%), beating the naive "
            f"baseline ({naive_mape:.2f}% MAPE) — it's picking up a real, learnable signal on top "
            "of simple persistence."
        )
    else:
        st.warning(
            f"⚠️ The naive baseline (MAPE {naive_mape:.2f}%) is competitive with or better than "
            f"the trained models (best: {best_name} at {best_mape:.2f}%). With ~{len(model_df)} weeks "
            "of data, sales momentum alone is a hard benchmark to beat — more history or additional "
            "predictors (pricing, promotions, seasonality events) would likely help."
        )

    st.markdown("""
    **What the metrics mean:**
    - **MAE** — average dollar error per week, in the same units as Sales
    - **RMSE** — like MAE but penalizes large misses more heavily
    - **R²** — share of week-to-week sales variation explained by the model (1.0 = perfect, 0 = no better than predicting the average, negative = worse than that)
    - **MAPE** — average error as a percentage of actual sales, easiest to communicate to non-technical stakeholders
    """)

    st.markdown("---")
    st.subheader(f"Actual vs. Predicted Sales — {best_name}")
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(x=train["Week"], y=train["Sales"]/1e6, name="Actual (train)",
                                   line=dict(color="lightgray")))
    fig_pred.add_trace(go.Scatter(x=test["Week"], y=test["Sales"]/1e6, name="Actual (test)",
                                   line=dict(color="#1f77b4", width=3)))
    fig_pred.add_trace(go.Scatter(x=test["Week"], y=results[best_name]["pred"]/1e6, name=f"{best_name} Prediction",
                                   line=dict(color=PRIMARY, dash="dash", width=3)))
    fig_pred.update_layout(yaxis_title="Sales ($M)", height=450)
    st.plotly_chart(fig_pred, use_container_width=True)

    if results[best_name]["model"] is not None and hasattr(results[best_name]["model"], "feature_importances_"):
        st.markdown("---")
        st.subheader("What drives the forecast? Feature Importance")
        importances = pd.DataFrame({
            "Feature": FEATURES,
            "Importance": results[best_name]["model"].feature_importances_,
        }).sort_values("Importance", ascending=True)
        fig_imp = px.bar(importances, x="Importance", y="Feature", orientation="h",
                         color_discrete_sequence=[PRIMARY])
        st.plotly_chart(fig_imp, use_container_width=True)
    elif isinstance(results[best_name]["model"], LinearRegression):
        st.markdown("---")
        st.subheader("What drives the forecast? Model Coefficients")
        coefs = pd.DataFrame({
            "Feature": FEATURES,
            "Coefficient": results[best_name]["model"].coef_,
        }).sort_values("Coefficient", key=abs, ascending=True)
        fig_imp = px.bar(coefs, x="Coefficient", y="Feature", orientation="h",
                         color_discrete_sequence=[PRIMARY])
        st.plotly_chart(fig_imp, use_container_width=True)
        st.caption(
            "Positive coefficients push next-week sales up; negative ones pull it down. "
            "Because features are on different scales (dollars vs. counts vs. week numbers), "
            "compare direction and relative size loosely rather than as exact dollar effects."
        )

    st.markdown("---")
    st.subheader("Limitations & Next Steps")
    st.markdown(
        """
        - Only **~2 years of weekly data** (~110 usable weeks) — not enough to fully learn
          annual seasonality (like holiday spikes) with confidence.
        - Sales is a **single company-wide number** repeated across sites/rows in the raw
          data — the model forecasts total company sales, not sales by site or channel.
        - No pricing, promotion calendar, distribution, or competitor data is included;
          these often matter more than media spend for CPG sales.
        - Recommended next steps: bring in at least 3 years of history, add
          promotional/holiday flags, and validate with rolling-origin (walk-forward)
          cross-validation rather than a single train/test split.
        """
    )
