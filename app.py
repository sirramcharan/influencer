import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# --------- LOAD DATA ---------
@st.cache_data
def load_data(path="influencer_roi_dummy.csv"):
    df = pd.read_csv(path)
    df["roas"] = df["roas"].clip(lower=0)
    return df

df = load_data()

st.title("Influencer ROI Planner (Prototype)")
st.write(
    "Upload your own influencer campaign dataset or use the sample model to explore "
    "how budget and influencer selection affect expected ROI."
)

# --------- SIDEBAR: DATA SOURCE ---------
source = st.sidebar.radio(
    "Data source",
    ["Use sample dataset", "Upload my campaign CSV"],
)

if source == "Upload my campaign CSV":
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        df_user = pd.read_csv(uploaded)
        st.success(f"Loaded {len(df_user)} rows from your file.")
        # Expect same column names as the dummy dataset; otherwise add mapping logic
        df = df_user

# --------- FILTERS ---------
col1, col2 = st.columns(2)
with col1:
    category = st.selectbox(
        "Select brand category",
        sorted(df["brand_category"].dropna().unique()),
    )
with col2:
    platform = st.selectbox(
        "Select platform",
        ["All"] + sorted(df["platform"].dropna().unique()),
    )

filtered = df[df["brand_category"] == category].copy()
if platform != "All":
    filtered = filtered[filtered["platform"] == platform]

if filtered.empty:
    st.warning("No data for this combination. Try another category/platform.")
    st.stop()

# --------- SIMPLE INFLUENCER SUMMARY MODEL ---------
inf_summary = (
    filtered.groupby(["influencer_id", "influencer_handle", "influencer_tier"])
    .agg(
        mean_roas=("roas", "mean"),
        mean_cpa=("cpa", "mean"),
        mean_total_spend=("total_spend", "mean"),
        mean_revenue=("revenue", "mean"),
    )
    .reset_index()
)

inf_summary["mean_roas"] = inf_summary["mean_roas"].fillna(0)
inf_summary["mean_cpa"] = (
    inf_summary["mean_cpa"].replace([np.inf, -np.inf], np.nan).fillna(0)
)

# --------- BUDGET SLIDER ---------
min_budget = max(10_000, int(filtered["total_spend"].quantile(0.1)))
max_budget = int(filtered["total_spend"].quantile(0.9) * 5)

budget = st.slider(
    "Total campaign budget",
    min_value=min_budget,
    max_value=max_budget,
    value=int((min_budget + max_budget) / 2),
    step=10_000,
    format="₹ %d",
)

# --------- BUDGET ALLOCATION LOGIC ---------
st.subheader("Recommended influencers for this budget")

top_n = st.slider("Max number of influencers in combo", 1, 5, 3)

# Rank by mean ROAS
ranked = inf_summary.sort_values("mean_roas", ascending=False).reset_index(drop=True)

# Single best influencer
best_single = ranked.iloc[0]
single_rev = budget * best_single["mean_roas"]

# Best combination of top_n influencers with equal split
combo = ranked.head(top_n).copy()
combo["allocated_budget"] = budget / top_n
combo["expected_revenue"] = combo["allocated_budget"] * combo["mean_roas"]
total_combo_rev = combo["expected_revenue"].sum()
combo_roas = total_combo_rev / budget if budget > 0 else 0

colA, colB = st.columns(2)

with colA:
    st.markdown("**Best single influencer (by modelled ROAS)**")
    st.write(
        f"{best_single['influencer_handle']} ({best_single['influencer_tier']}), "
        f"mean ROAS ≈ {best_single['mean_roas']:.2f}x"
    )
    st.write(f"Budget: ₹{budget:,.0f} → Expected revenue ≈ ₹{single_rev:,.0f}")

with colB:
    st.markdown(f"**Best {top_n}-influencer combination (equal split)**")
    st.dataframe(
        combo[
            [
                "influencer_handle",
                "influencer_tier",
                "mean_roas",
                "allocated_budget",
                "expected_revenue",
            ]
        ]
        .rename(
            columns={
                "influencer_handle": "Influencer",
                "influencer_tier": "Tier",
                "mean_roas": "Mean ROAS",
                "allocated_budget": "Budget (₹)",
                "expected_revenue": "Expected revenue (₹)",
            }
        )
        .style.format(
            {
                "Mean ROAS": "{:.2f}",
                "Budget (₹)": "₹{:.0f}",
                "Expected revenue (₹)": "₹{:.0f}",
            }
        )
    )
    st.write(
        f"Total combo expected revenue ≈ ₹{total_combo_rev:,.0f} "
        f"(ROAS ≈ {combo_roas:.2f}x)"
    )

# --------- DYNAMIC GRAPH: BUDGET vs EXPECTED REVENUE ---------
st.subheader("Budget vs expected revenue (dynamic)")

# Budget grid
budgets = np.linspace(min_budget, max_budget, 20)

single_curve = budgets * best_single["mean_roas"]
combo_curve = budgets * combo_roas

# Build tidy dataframe for Altair
plot_df = pd.DataFrame(
    {
        "budget": budgets.astype(float),
        "single": single_curve.astype(float),
        "combo": combo_curve.astype(float),
    }
)

plot_long = plot_df.melt(id_vars="budget", var_name="Scenario", value_name="revenue")
plot_long = plot_long.replace([np.inf, -np.inf], np.nan).dropna(
    subset=["budget", "revenue"]
)

chart = (
    alt.Chart(plot_long)
    .mark_line()
    .encode(
        x=alt.X("budget:Q", title="Budget (₹)"),
        y=alt.Y("revenue:Q", title="Expected revenue (₹)"),
        color=alt.Color("Scenario:N", title="Scenario"),
        tooltip=["budget", "Scenario", "revenue"],
    )
    .interactive()
)

st.altair_chart(chart, use_container_width=True)

st.caption(
    "Prototype: ROAS is estimated from historical-like dummy data. "
    "In production, replace the simple mean-ROAS heuristic with a trained model "
    "and add constraints (min/max spend per influencer, frequency caps, etc.)."
)
