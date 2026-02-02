import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ============================================================
#                 HELPER: RESPONSE CURVE
# ============================================================

def spend_to_revenue(spend_array, k, alpha, beta):
    """
    Non-linear response curve for influencer revenue.
    revenue = k * (scaled_spend^alpha) * exp(-beta * scaled_spend),
    scaled_spend = spend / 1e6.
    """
    spend_array = np.array(spend_array, dtype=float)
    spend_scaled = spend_array / 1_000_000.0
    base = k * (spend_scaled ** alpha) * np.exp(-beta * spend_scaled)
    base = np.maximum(0.0, base)
    return base * 1_000_000.0


# ============================================================
#                        LOAD DATA
# ============================================================

@st.cache_data
def load_data(path="influencer_roi_dummy.csv"):
    df = pd.read_csv(path)
    df["roas"] = df["roas"].clip(lower=0)
    if {
        "avg_likes_per_post",
        "avg_comments_per_post",
        "followers_start",
    }.issubset(df.columns):
        df["engagement_rate_est"] = (
            (df["avg_likes_per_post"] + df["avg_comments_per_post"])
            / df["followers_start"].replace(0, np.nan)
        )
    else:
        df["engagement_rate_est"] = np.nan
    return df


st.set_page_config(page_title="Influencer ROI Analytics", layout="wide")
df = load_data()

st.title("Influencer ROI Analytics")

# ============================================================
#                 DATA SOURCE (SIDEBAR)
# ============================================================

source = st.sidebar.radio(
    "Data source",
    ["Use sample dataset", "Upload my campaign CSV"],
)

if source == "Upload my campaign CSV":
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        df_user = pd.read_csv(uploaded)
        st.success(f"Loaded {len(df_user)} rows from your file.")
        df = df_user

if "engagement_rate_est" not in df.columns and {
    "avg_likes_per_post",
    "avg_comments_per_post",
    "followers_start",
}.issubset(df.columns):
    df["engagement_rate_est"] = (
        (df["avg_likes_per_post"] + df["avg_comments_per_post"])
        / df["followers_start"].replace(0, np.nan)
    )

# ============================================================
#                 MODE TOGGLE (DASHBOARD / PLANNER)
# ============================================================

mode = st.segmented_control(
    "View",
    options=["Dashboard", "Planner"],
    default="Dashboard",
)

# ============================================================
#                         DASHBOARD
# ============================================================

if mode == "Dashboard":
    st.subheader("Overview dashboard")

    total_campaigns = len(df)
    unique_influencers = df["influencer_id"].nunique() if "influencer_id" in df.columns else np.nan
    unique_brands = df["brand_name"].nunique() if "brand_name" in df.columns else np.nan
    avg_roas = df["roas"].replace([np.inf, -np.inf], np.nan).mean()
    avg_eng_rate = df["engagement_rate_est"].replace([np.inf, -np.inf], np.nan).mean()

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Total campaigns", f"{total_campaigns:,}")
    if not np.isnan(unique_influencers):
        kpi2.metric("Unique influencers", f"{unique_influencers:,}")
    else:
        kpi2.metric("Unique influencers", "N/A")
    if not np.isnan(unique_brands):
        kpi3.metric("Unique brands", f"{unique_brands:,}")
    else:
        kpi3.metric("Average ROAS", f"{avg_roas:.2f}x")

    kpi4, kpi5 = st.columns(2)
    kpi4.metric("Average ROAS", f"{avg_roas:.2f}x")
    kpi5.metric("Avg est. engagement rate", f"{avg_eng_rate*100:.2f}%")

    st.markdown("### Engagement and ROAS by platform")

    if "platform" in df.columns:
        plat_group = (
            df.groupby("platform")
            .agg(
                avg_eng_rate=("engagement_rate_est", "mean"),
                avg_roas=("roas", "mean"),
                campaigns=("campaign_id", "count"),
            )
            .reset_index()
        )

        eng_chart = (
            alt.Chart(plat_group)
            .mark_bar()
            .encode(
                x=alt.X("platform:N", title="Platform"),
                y=alt.Y(
                    "avg_eng_rate:Q",
                    title="Avg engagement rate",
                    axis=alt.Axis(format="%"),
                ),
                tooltip=["platform", "avg_eng_rate", "campaigns"],
                color="platform:N",
            )
        )

        roas_chart = (
            alt.Chart(plat_group)
            .mark_bar()
            .encode(
                x=alt.X("platform:N", title="Platform"),
                y=alt.Y("avg_roas:Q", title="Avg ROAS (x)"),
                tooltip=["platform", "avg_roas", "campaigns"],
                color="platform:N",
            )
        )

        c1, c2 = st.columns(2)
        with c1:
            st.altair_chart(eng_chart, use_container_width=True)
        with c2:
            st.altair_chart(roas_chart, use_container_width=True)
    else:
        st.info("Platform column not found in data; cannot show platform breakdown.")

    st.markdown("### Top performers (per influencer)")

    if {"influencer_id", "influencer_handle"}.issubset(df.columns):
        inf_agg = (
            df.groupby(["influencer_id", "influencer_handle", "influencer_tier"])
            .agg(
                mean_roas=("roas", "mean"),
                mean_eng_rate=("engagement_rate_est", "mean"),
                campaigns=("campaign_id", "count"),
            )
            .reset_index()
        )

        top_roas = inf_agg.sort_values("mean_roas", ascending=False).head(10)
        top_eng = inf_agg.sort_values("mean_eng_rate", ascending=False).head(10)

        c3, c4 = st.columns(2)
        with c3:
            st.write("Top 10 influencers by average ROAS")
            st.dataframe(
                top_roas[
                    ["influencer_handle", "influencer_tier", "mean_roas", "campaigns"]
                ]
                .rename(
                    columns={
                        "influencer_handle": "Influencer",
                        "influencer_tier": "Tier",
                        "mean_roas": "Mean ROAS",
                        "campaigns": "Campaigns",
                    }
                )
                .style.format({"Mean ROAS": "{:.2f}"})
            )
        with c4:
            st.write("Top 10 influencers by average engagement rate")
            st.dataframe(
                top_eng[
                    ["influencer_handle", "influencer_tier", "mean_eng_rate", "campaigns"]
                ]
                .rename(
                    columns={
                        "influencer_handle": "Influencer",
                        "influencer_tier": "Tier",
                        "mean_eng_rate": "Mean engagement rate",
                        "campaigns": "Campaigns",
                    }
                )
                .style.format({"Mean engagement rate": "{:.2%}"})
            )
    else:
        st.info("Influencer columns not found; cannot show top performers.")

    st.markdown("### ROAS distribution")
    roas_clean = df["roas"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(roas_clean) > 0:
        roas_df = pd.DataFrame({"roas": roas_clean})
        roas_chart = (
            alt.Chart(roas_df)
            .mark_bar()
            .encode(
                x=alt.X("roas:Q", bin=alt.Bin(maxbins=30), title="ROAS (x)"),
                y=alt.Y("count():Q", title="Number of campaigns"),
            )
        )
        st.altair_chart(roas_chart, use_container_width=True)
    else:
        st.info("No valid ROAS values to plot.")

# ============================================================
#                         PLANNER
# ============================================================

else:
    st.subheader("Budget planner")

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

    # Influencer summary from filtered campaigns
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

    # Bring curve parameters from campaigns
    if {"curve_k", "curve_alpha", "curve_beta"}.issubset(filtered.columns):
        curve_params = (
            filtered.groupby("influencer_id")
            .agg(
                k=("curve_k", "median"),
                alpha=("curve_alpha", "median"),
                beta=("curve_beta", "median"),
            )
            .reset_index()
        )
        inf_summary = inf_summary.merge(curve_params, on="influencer_id", how="left")
    else:
        inf_summary["k"] = 0.0
        inf_summary["alpha"] = 1.0
        inf_summary["beta"] = 0.0

    inf_summary[["k", "alpha", "beta"]] = inf_summary[["k", "alpha", "beta"]].fillna(0.0)
    inf_summary["mean_roas"] = inf_summary["mean_roas"].fillna(0)
    inf_summary["mean_cpa"] = (
        inf_summary["mean_cpa"].replace([np.inf, -np.inf], np.nan).fillna(0)
    )

    # Budget inputs: numeric for real budget, slider for graph range
    min_budget = max(10_000, int(filtered["total_spend"].quantile(0.1)))
    max_budget_default = int(filtered["total_spend"].quantile(0.9) * 5)

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        budget = st.number_input(
            "Enter your total campaign budget (₹)",
            min_value=float(min_budget),
            value=float((min_budget + max_budget_default) / 2),
            step=10_000.0,
            format="%.0f",
        )
    with col_b2:
        graph_max_budget = st.slider(
            "Max budget shown on graph (x-axis)",
            min_value=min_budget,
            max_value=max_budget_default,
            value=max_budget_default,
            step=10_000,
            format="₹ %d",
        )

    st.markdown("#### Recommended influencers for this budget")

    top_n = st.slider("Max number of influencers in combo", 1, 5, 3)

    def expected_rev_for_row(row, spend):
        return float(spend_to_revenue(spend, row["k"], row["alpha"], row["beta"]))

    inf_summary["expected_revenue_at_budget"] = inf_summary.apply(
        lambda r: expected_rev_for_row(r, budget), axis=1
    )
    inf_summary["expected_roas_at_budget"] = (
        inf_summary["expected_revenue_at_budget"] / budget if budget > 0 else 0.0
    )

    ranked = inf_summary.sort_values(
        "expected_revenue_at_budget", ascending=False
    ).reset_index(drop=True)

    best_single = ranked.iloc[0]
    single_rev = best_single["expected_revenue_at_budget"]

    combo = ranked.head(top_n).copy()
    combo["allocated_budget"] = budget / top_n
    combo["expected_revenue"] = combo.apply(
        lambda r: expected_rev_for_row(r, r["allocated_budget"]), axis=1
    )
    total_combo_rev = combo["expected_revenue"].sum()
    combo_roas = total_combo_rev / budget if budget > 0 else 0

    colA, colB = st.columns(2)

    with colA:
        st.markdown("**Best single influencer (by expected revenue at this budget)**")
        st.write(
            f"{best_single['influencer_handle']} ({best_single['influencer_tier']}), "
            f"expected ROAS ≈ {best_single['expected_roas_at_budget']:.2f}x"
        )
        st.write(f"Budget: ₹{budget:,.0f} → Expected revenue ≈ ₹{single_rev:,.0f}")

    with colB:
        st.markdown(f"**Best {top_n}-influencer combination (equal split)**")
        st.dataframe(
            combo[
                [
                    "influencer_handle",
                    "influencer_tier",
                    "expected_roas_at_budget",
                    "allocated_budget",
                    "expected_revenue",
                ]
            ]
            .rename(
                columns={
                    "influencer_handle": "Influencer",
                    "influencer_tier": "Tier",
                    "expected_roas_at_budget": "ROAS at allocated budget",
                    "allocated_budget": "Budget (₹)",
                    "expected_revenue": "Expected revenue (₹)",
                }
            )
            .style.format(
                {
                    "ROAS at allocated budget": "{:.2f}",
                    "Budget (₹)": "₹{:.0f}",
                    "Expected revenue (₹)": "₹{:.0f}",
                }
            )
        )
        st.write(
            f"Total combo expected revenue ≈ ₹{total_combo_rev:,.0f} "
            f"(ROAS ≈ {combo_roas:.2f}x)"
        )

    st.markdown("#### Budget vs expected revenue")

    budgets = np.linspace(min_budget, graph_max_budget, 30)
    single_curve = spend_to_revenue(
        budgets, best_single["k"], best_single["alpha"], best_single["beta"]
    )
    combo_curve = budgets * combo_roas  # approximation for combo

    plot_df = pd.DataFrame(
        {
            "budget": budgets.astype(float),
            "Best single": single_curve.astype(float),
            f"Top {top_n} combo": combo_curve.astype(float),
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
        "Curves use non-linear response functions fitted per influencer. "
        "At higher budgets you may see diminishing or declining returns, "
        "so the app can recommend switching to other influencers or combos."
    )
