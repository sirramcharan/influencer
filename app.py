import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from scipy.optimize import curve_fit

# ============================================================
#                   CONFIG & HELPER FUNCTIONS
# ============================================================

st.set_page_config(page_title="Influencer ROI Optimizer", layout="wide")

# Custom CSS for "Metric Card" styling
st.markdown("""
<style>
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        border: 1px solid #e6e9ef;
        padding: 5% 5% 5% 10%;
        border-radius: 5px;
        color: rgb(30, 30, 30);
        overflow-wrap: break-word;
    }
    /* Break long metric labels */
    div[data-testid="metric-container"] > label {
        font-size: 1rem;
        color: rgb(49, 51, 63);
    }
    div[data-testid="metric-container"] > div {
        font-size: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """
    Load the dataset. 
    In a real app, you might connect to a database or cloud storage.
    Here we expect 'influencer_dataset.csv' in the same folder.
    """
    try:
        df = pd.read_csv("influencer_dataset.csv")
        
        # Ensure date column is datetime
        if 'Date_Posted' in df.columns:
            df['Date_Posted'] = pd.to_datetime(df['Date_Posted'])
            
        # Calculate derived metrics if they don't exist
        if 'ROAS' not in df.columns and 'Total_Revenue_INR' in df.columns and 'Cost_Fee_INR' in df.columns:
            df['ROAS'] = df['Total_Revenue_INR'] / df['Cost_Fee_INR']
            
        if 'Engagement_Rate' not in df.columns and 'Impressions' in df.columns:
            # Avoid division by zero
            df['Engagement_Rate'] = (
                (df['Likes'] + df['Comments'] + df['Shares']) / df['Impressions'].replace(0, np.nan)
            ).fillna(0)

        return df
    except FileNotFoundError:
        return None

# Non-linear response curve function: Revenue = k * (Spend^alpha) * exp(-beta * Spend)
# Normalized Spend (Spend / 1M) is used to keep numbers stable for optimization
def response_curve(spend, k, alpha, beta):
    spend_norm = spend / 1_000_000 # Normalize to Millions
    # Prevent negative or complex results
    return np.maximum(0, k * (spend_norm ** alpha) * np.exp(-beta * spend_norm))

@st.cache_data
def fit_curves(df):
    """
    Fit a response curve for every influencer in the dataset.
    Returns a dictionary of parameters {influencer_id: (k, alpha, beta)}
    """
    curve_params = {}
    
    # We need enough data points per influencer to fit a curve.
    # If not enough, we use a "Category Average" fallback.
    
    # Group by Influencer
    grouped = df.groupby('Influencer_ID')
    
    for influencer, group in grouped:
        # We need at least 3 points to fit 3 parameters. 
        # But for robustness, let's say 5. 
        # If synthetic data is small, we might need to relax this or use synthetic points.
        
        if len(group) < 3:
            # Fallback: Use generic parameters based on their Tier if possible, 
            # or just generic defaults.
            # (k, alpha, beta)
            curve_params[influencer] = (1000000, 0.8, 0.1) 
            continue
            
        X = group['Cost_Fee_INR'].values
        y = group['Total_Revenue_INR'].values
        
        # Bounds: k>0, 0<alpha<2 (diminishing returns), beta>=0 (saturation)
        try:
            popt, _ = curve_fit(
                response_curve, 
                X, 
                y, 
                p0=[500000, 0.9, 0.05], # Initial guess
                bounds=([1000, 0.1, 0.0], [np.inf, 2.0, 5.0]),
                maxfev=5000
            )
            curve_params[influencer] = popt
        except:
             # Fallback if fit fails
            curve_params[influencer] = (1000000, 0.8, 0.1)
            
    return curve_params

# ============================================================
#                   MAIN APP LOGIC
# ============================================================

df = load_data()

if df is None:
    st.error("Dataset 'influencer_dataset.csv' not found. Please upload it to your GitHub repo.")
    st.stop()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["📊 Dashboard", "🧠 Planner"])

# ============================================================
#                   PAGE 1: DASHBOARD
# ============================================================
if page == "📊 Dashboard":
    st.title("Historical Campaign Performance")
    
    # --- Top Level KPIs ---
    total_spend = df['Cost_Fee_INR'].sum()
    total_rev = df['Total_Revenue_INR'].sum()
    blended_roas = total_rev / total_spend if total_spend > 0 else 0
    total_impressions = df['Impressions'].sum()
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Spend", f"₹{total_spend:,.0f}")
    c2.metric("Total Revenue", f"₹{total_rev:,.0f}")
    c3.metric("Blended ROAS", f"{blended_roas:.2f}x", delta_color="normal")
    c4.metric("Total Impressions", f"{total_impressions:,.0f}")
    
    st.markdown("---")
    
    # --- Charts Row 1 ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Spend vs Revenue Trend")
        # Aggregating by Month
        df['Month'] = df['Date_Posted'].dt.to_period('M').astype(str)
        monthly_data = df.groupby('Month')[['Cost_Fee_INR', 'Total_Revenue_INR']].sum().reset_index()
        
        # Melt for Altair
        monthly_melt = monthly_data.melt('Month', var_name='Metric', value_name='Amount')
        
        chart = alt.Chart(monthly_melt).mark_line(point=True).encode(
            x='Month',
            y='Amount',
            color='Metric',
            tooltip=['Month', 'Metric', 'Amount']
        ).interactive()
        st.altair_chart(chart, use_container_width=True)
        
    with col2:
        st.subheader("ROAS by Platform")
        plat_metrics = df.groupby('Platform').agg(
            Spend=('Cost_Fee_INR', 'sum'),
            Revenue=('Total_Revenue_INR', 'sum')
        ).reset_index()
        plat_metrics['ROAS'] = plat_metrics['Revenue'] / plat_metrics['Spend']
        
        bar_chart = alt.Chart(plat_metrics).mark_bar().encode(
            x=alt.X('ROAS', title='ROAS (x)'),
            y=alt.Y('Platform', sort='-x'),
            color=alt.Color('ROAS', scale=alt.Scale(scheme='greens')),
            tooltip=['Platform', 'ROAS', 'Spend', 'Revenue']
        ).interactive()
        st.altair_chart(bar_chart, use_container_width=True)
        
    # --- Leaderboard ---
    st.subheader("Top Performing Influencers")
    
    # Group by Influencer to get averages/sums
    leaderboard = df.groupby(['Influencer_ID', 'Influencer_Tier']).agg(
        Total_Spend=('Cost_Fee_INR', 'sum'),
        Total_Revenue=('Total_Revenue_INR', 'sum'),
        Avg_ROAS=('ROAS', 'mean'),
        Campaign_Count=('Campaign_ID', 'count')
    ).reset_index()
    
    # Filter out one-off wonders (optional: keep only those with >1 campaign if data allows)
    # leaderboard = leaderboard[leaderboard['Campaign_Count'] > 1]
    
    top_n = leaderboard.sort_values('Avg_ROAS', ascending=False).head(10)
    
    st.dataframe(
        top_n.style.format({
            "Total_Spend": "₹{:,.0f}", 
            "Total_Revenue": "₹{:,.0f}", 
            "Avg_ROAS": "{:.2f}x"
        }),
        use_container_width=True
    )

# ============================================================
#                   PAGE 2: PLANNER
# ============================================================
elif page == "🧠 Planner":
    st.title("Budget Optimization Engine")
    
    st.markdown("""
    > **How this works:** This tool uses non-linear regression (diminishing returns) to calculate the 
    > optimal way to split your budget. It compares putting all eggs in one basket (Single Strategy) 
    > vs. diversifying across a portfolio (Combinatorial Strategy).
    """)
    
    # --- 1. Inputs ---
    with st.sidebar:
        st.header("Campaign Settings")
        budget = st.number_input("Total Budget (₹)", min_value=10000, max_value=5000000, value=500000, step=10000)
        
        platform_filter = st.multiselect(
            "Filter Platforms", 
            options=df['Platform'].unique(),
            default=df['Platform'].unique()
        )
        
        tier_filter = st.multiselect(
            "Filter Tiers",
            options=df['Influencer_Tier'].unique(),
            default=df['Influencer_Tier'].unique()
        )
    
    # --- 2. Filter Data & Fit Curves ---
    filtered_df = df[
        (df['Platform'].isin(platform_filter)) & 
        (df['Influencer_Tier'].isin(tier_filter))
    ]
    
    if filtered_df.empty:
        st.warning("No data matches your filters. Please adjust.")
        st.stop()
        
    curve_params = fit_curves(filtered_df)
    
    # --- 3. Optimization Logic ---
    
    # Calculate Expected Revenue for EVERY influencer at full budget
    influencer_scores = []
    
    for inf_id, params in curve_params.items():
        k, alpha, beta = params
        expected_rev = response_curve(budget, k, alpha, beta)
        
        # Get metadata for display
        meta = filtered_df[filtered_df['Influencer_ID'] == inf_id].iloc[0]
        
        influencer_scores.append({
            "Influencer_ID": inf_id,
            "Tier": meta['Influencer_Tier'],
            "Platform": meta['Platform'],
            "Params": params,
            "Expected_Rev_Full_Budget": expected_rev
        })
        
    scores_df = pd.DataFrame(influencer_scores)
    
    # STRATEGY A: BEST SINGLE
    best_single = scores_df.sort_values('Expected_Rev_Full_Budget', ascending=False).iloc[0]
    rev_single = best_single['Expected_Rev_Full_Budget']
    roas_single = rev_single / budget
    
    # STRATEGY B: TOP 3 COMBO (Simple Equal Split Heuristic)
    # In a full thesis, you'd use scipy.optimize.minimize to find exact split.
    # Here we simulate an equal split for robustness.
    
    split_budget = budget / 3
    
    # Recalculate revenue for everyone at 1/3 budget
    scores_df['Expected_Rev_Split'] = scores_df['Params'].apply(
        lambda p: response_curve(split_budget, p[0], p[1], p[2])
    )
    
    top_3_combo = scores_df.sort_values('Expected_Rev_Split', ascending=False).head(3)
    rev_combo = top_3_combo['Expected_Rev_Split'].sum()
    roas_combo = rev_combo / budget
    
    # --- 4. Display Results ---
    
    col_a, col_b = st.columns(2)
    
    # Card A
    with col_a:
        st.subheader("Strategy A: Concentrated")
        st.caption("Invest 100% in Top Performer")
        st.metric("Expected Revenue", f"₹{rev_single:,.0f}")
        st.metric("Expected ROAS", f"{roas_single:.2f}x", delta_color="off")
        st.write(f"**Influencer:** {best_single['Influencer_ID']} ({best_single['Tier']})")
        
    # Card B
    with col_b:
        st.subheader("Strategy B: Diversified (Top 3)")
        st.caption("Split Budget Equally (33% each)")
        st.metric("Expected Revenue", f"₹{rev_combo:,.0f}")
        
        # Highlight the winner
        delta_val = roas_combo - roas_single
        st.metric("Expected ROAS", f"{roas_combo:.2f}x", delta=f"{delta_val:.2f}x vs Single")
        
        st.write("**Portfolio:**")
        for _, row in top_3_combo.iterrows():
            st.write(f"- {row['Influencer_ID']} ({row['Tier']})")

    st.markdown("---")
    
    # --- 5. Visualization: The Diminishing Returns Curve ---
    st.subheader("The 'Why': Diminishing Returns Analysis")
    
    # Generate curve points for plotting
    x_points = np.linspace(0, budget * 1.5, 50) # Plot up to 1.5x budget to show future saturation
    
    # Single Curve
    k, a, b = best_single['Params']
    y_single = response_curve(x_points, k, a, b)
    
    # Combo Curve (Sum of top 3 curves at x/3 spend each)
    y_combo = np.zeros_like(x_points)
    for _, row in top_3_combo.iterrows():
        kp, ap, bp = row['Params']
        # We assume if total spend is X, each gets X/3
        y_combo += response_curve(x_points/3, kp, ap, bp)
        
    plot_data = pd.DataFrame({
        'Budget': np.tile(x_points, 2),
        'Revenue': np.concatenate([y_single, y_combo]),
        'Strategy': ['Best Single'] * 50 + ['Top 3 Combo'] * 50
    })
    
    line_chart = alt.Chart(plot_data).mark_line().encode(
        x=alt.X('Budget', axis=alt.Axis(format='₹~s')),
        y=alt.Y('Revenue', axis=alt.Axis(format='₹~s')),
        color='Strategy',
        tooltip=['Strategy', 'Budget', 'Revenue']
    ).properties(height=400).interactive()
    
    st.altair_chart(line_chart, use_container_width=True)
    
    st.info(f"""
    **Analysis:**
    - The **Blue Line** (Single) shows how revenue grows if you give everything to {best_single['Influencer_ID']}.
    - The **Orange Line** (Combo) shows the revenue from splitting budget across 3 influencers.
    - If the lines cross, that is your **Optimization Tipping Point**.
    """)
