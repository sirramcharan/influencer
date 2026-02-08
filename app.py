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
def load_data(uploaded_file=None):
    """
    Load data from an uploaded file OR fall back to the default repo file.
    """
    df = None
    
    # 1. Try loading uploaded file if present
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")
            return None
            
    # 2. If no upload, try loading default from repo
    elif df is None:
        try:
            df = pd.read_csv("influencer_dataset.csv")
        except FileNotFoundError:
            # Return None to handle gracefully in main app
            return None
            
    # 3. Preprocessing (Standardize columns)
    if df is not None:
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

# Non-linear response curve function
def response_curve(spend, k, alpha, beta):
    spend_norm = spend / 1_000_000 # Normalize to Millions
    return np.maximum(0, k * (spend_norm ** alpha) * np.exp(-beta * spend_norm))

@st.cache_data
def fit_curves(df):
    """Fit a response curve for every influencer in the dataset."""
    curve_params = {}
    grouped = df.groupby('Influencer_ID')
    
    for influencer, group in grouped:
        if len(group) < 3:
            curve_params[influencer] = (1000000, 0.8, 0.1) 
            continue
            
        X = group['Cost_Fee_INR'].values
        y = group['Total_Revenue_INR'].values
        
        try:
            popt, _ = curve_fit(
                response_curve, 
                X, y, 
                p0=[500000, 0.9, 0.05],
                bounds=([1000, 0.1, 0.0], [np.inf, 2.0, 5.0]),
                maxfev=5000
            )
            curve_params[influencer] = popt
        except:
            curve_params[influencer] = (1000000, 0.8, 0.1)
            
    return curve_params

# ============================================================
#                   MAIN APP LOGIC
# ============================================================

# --- SIDEBAR: DATA UPLOAD ---
st.sidebar.title("Data Settings")
uploaded_file = st.sidebar.file_uploader("Upload Campaign CSV", type=["csv"])

# Load data based on user input or default
df = load_data(uploaded_file)

if df is None:
    st.info("👋 **Welcome!**")
    st.warning("⚠️ Default dataset 'influencer_dataset.csv' not found.")
    st.markdown("""
    **To fix this:**
    1. Ensure `influencer_dataset.csv` is in your GitHub repo (root folder).
    2. OR Upload your own CSV file in the sidebar.
    """)
    st.stop()
else:
    if uploaded_file is None:
        st.sidebar.success("✅ Loaded Default Demo Data")
    else:
        st.sidebar.success("✅ Loaded Your Uploaded Data")

# --- NAVIGATION ---
st.sidebar.markdown("---")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["📊 Dashboard", "🧠 Planner"])

# ============================================================
#                   PAGE 1: DASHBOARD
# ============================================================
if page == "📊 Dashboard":
    st.title("Historical Campaign Performance")
    
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
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Spend vs Revenue Trend")
        if 'Date_Posted' in df.columns:
            df['Month'] = df['Date_Posted'].dt.to_period('M').astype(str)
            monthly_data = df.groupby('Month')[['Cost_Fee_INR', 'Total_Revenue_INR']].sum().reset_index()
            monthly_melt = monthly_data.melt('Month', var_name='Metric', value_name='Amount')
            
            chart = alt.Chart(monthly_melt).mark_line(point=True).encode(
                x='Month', y='Amount', color='Metric', tooltip=['Month', 'Metric', 'Amount']
            ).interactive()
            st.altair_chart(chart, use_container_width=True)
        else:
            st.warning("Date column not found, skipping trend chart.")
        
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
        
    st.subheader("Top Performing Influencers")
    leaderboard = df.groupby(['Influencer_ID', 'Influencer_Tier']).agg(
        Total_Spend=('Cost_Fee_INR', 'sum'),
        Total_Revenue=('Total_Revenue_INR', 'sum'),
        Avg_ROAS=('ROAS', 'mean'),
        Campaign_Count=('Campaign_ID', 'count')
    ).reset_index()
    
    top_n = leaderboard.sort_values('Avg_ROAS', ascending=False).head(10)
    st.dataframe(
        top_n.style.format({
            "Total_Spend": "₹{:,.0f}", 
            "Total_Revenue": "₹{:,.0f}", 
            "Avg_ROAS": "{:.2f}x"
        }), use_container_width=True
    )

# ============================================================
#                   PAGE 2: PLANNER
# ============================================================
elif page == "🧠 Planner":
    st.title("Budget Optimization Engine")
    
    with st.sidebar:
        st.header("Optimization Settings")
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
    
    filtered_df = df[
        (df['Platform'].isin(platform_filter)) & 
        (df['Influencer_Tier'].isin(tier_filter))
    ]
    
    if filtered_df.empty:
        st.warning("No data matches your filters. Please adjust.")
        st.stop()
        
    curve_params = fit_curves(filtered_df)
    
    influencer_scores = []
    for inf_id, params in curve_params.items():
        k, alpha, beta = params
        expected_rev = response_curve(budget, k, alpha, beta)
        meta = filtered_df[filtered_df['Influencer_ID'] == inf_id].iloc[0]
        influencer_scores.append({
            "Influencer_ID": inf_id,
            "Tier": meta['Influencer_Tier'],
            "Platform": meta['Platform'],
            "Params": params,
            "Expected_Rev_Full_Budget": expected_rev
        })
        
    scores_df = pd.DataFrame(influencer_scores)
    
    # STRATEGIES
    best_single = scores_df.sort_values('Expected_Rev_Full_Budget', ascending=False).iloc[0]
    rev_single = best_single['Expected_Rev_Full_Budget']
    roas_single = rev_single / budget
    
    split_budget = budget / 3
    scores_df['Expected_Rev_Split'] = scores_df['Params'].apply(
        lambda p: response_curve(split_budget, p[0], p[1], p[2])
    )
    
    top_3_combo = scores_df.sort_values('Expected_Rev_Split', ascending=False).head(3)
    rev_combo = top_3_combo['Expected_Rev_Split'].sum()
    roas_combo = rev_combo / budget
    
    # DISPLAY
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Strategy A: Concentrated")
        st.caption("Invest 100% in Top Performer")
        st.metric("Expected Revenue", f"₹{rev_single:,.0f}")
        st.metric("Expected ROAS", f"{roas_single:.2f}x")
        st.write(f"**Influencer:** {best_single['Influencer_ID']} ({best_single['Tier']})")
        
    with col_b:
        st.subheader("Strategy B: Diversified (Top 3)")
        st.caption("Split Budget Equally (33% each)")
        st.metric("Expected Revenue", f"₹{rev_combo:,.0f}")
        delta_val = roas_combo - roas_single
        st.metric("Expected ROAS", f"{roas_combo:.2f}x", delta=f"{delta_val:.2f}x vs Single")
        st.write("**Portfolio:**")
        for _, row in top_3_combo.iterrows():
            st.write(f"- {row['Influencer_ID']} ({row['Tier']})")

    st.markdown("---")
    st.subheader("Diminishing Returns Analysis")
    
    x_points = np.linspace(0, budget * 1.5, 50)
    k, a, b = best_single['Params']
    y_single = response_curve(x_points, k, a, b)
    
    y_combo = np.zeros_like(x_points)
    for _, row in top_3_combo.iterrows():
        kp, ap, bp = row['Params']
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
