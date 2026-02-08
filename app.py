import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from scipy.optimize import minimize

# ============================================================
#                   CONFIG & STYLING
# ============================================================

st.set_page_config(page_title="Algorithmic Marketer", layout="wide")

st.markdown("""
<style>
    .metric-card {
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #e0e0e0;
        background-color: #f9f9f9;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
#                   MATH ENGINE
# ============================================================

@st.cache_data
def load_data(uploaded_file=None):
    df = None
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except:
            st.error("File is not a valid CSV.")
    else:
        try:
            df = pd.read_csv("influencer_dataset.csv")
        except FileNotFoundError:
            return None

    if df is not None:
        if 'ROAS' not in df.columns:
            df['ROAS'] = df['Total_Revenue_INR'] / df['Cost_Fee_INR']
        if 'Category' not in df.columns:
            df['Category'] = "General"
        if 'Date_Posted' in df.columns:
            df['Date_Posted'] = pd.to_datetime(df['Date_Posted'])
            
    return df

def response_function(spend, k, alpha, beta):
    # Diminishing Returns Formula
    x = spend / 1_000_000 
    return np.maximum(0, k * (x ** alpha) * np.exp(-(beta + 0.1) * x))

@st.cache_data
def fit_curves_heuristic(df):
    curve_params = {}
    grouped = df.groupby('Influencer_ID')
    
    for inf_id, group in grouped:
        if len(group) < 1: continue
        avg_roas = group['ROAS'].mean()
        avg_cost = group['Cost_Fee_INR'].mean()
        
        k_est = avg_cost * avg_roas * 1.5 
        alpha_est = 0.7 + (np.log1p(avg_roas) / 10.0) 
        beta_est = 0.05 + (10000 / avg_cost) 
        
        curve_params[inf_id] = (k_est, alpha_est, beta_est)
            
    return curve_params

def maximize_revenue(budget, influencers, curve_params):
    n = len(influencers)
    params = [curve_params[i] for i in influencers]
    
    def objective(allocations):
        total_rev = 0
        for i, spend in enumerate(allocations):
            k, a, b = params[i]
            total_rev += response_function(spend, k, a, b)
        return -total_rev 

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - budget})
    bounds = tuple((0, budget) for _ in range(n))
    initial_guess = [budget/n] * n
    
    result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    
    allocation = {inf: round(amt, 2) for inf, amt in zip(influencers, result.x)}
    return allocation

# ============================================================
#                   MAIN APP UI
# ============================================================

st.title("The Algorithmic Marketer")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Data Settings")
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    
df = load_data(uploaded_file)

if df is None:
    st.warning("⚠️ No data found. Please upload 'influencer_dataset.csv'.")
    st.stop()

# --- TABS ---
tab1, tab2 = st.tabs(["📊 Performance Dashboard", "🧠 AI Planner"])

# ============================================================
#                   TAB 1: DASHBOARD (UNTOUCHED)
# ============================================================
with tab1:
    total_spend = df['Cost_Fee_INR'].sum()
    total_rev = df['Total_Revenue_INR'].sum()
    roas = total_rev / total_spend if total_spend > 0 else 0
    orders = df['Total_Orders'].sum()
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Investment", f"₹{total_spend:,.0f}")
    k2.metric("Total Revenue", f"₹{total_rev:,.0f}")
    k3.metric("ROI (ROAS)", f"{roas:.2f}x")
    k4.metric("Total Orders", f"{orders:,.0f}")
    
    st.divider()
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("1. Revenue by Category")
        cat_agg = df.groupby('Category')[['Total_Revenue_INR']].sum().reset_index()
        chart1 = alt.Chart(cat_agg).mark_bar().encode(
            x=alt.X('Category', sort='-y'),
            y='Total_Revenue_INR',
            color='Category',
            tooltip=['Category', 'Total_Revenue_INR']
        ).interactive()
        st.altair_chart(chart1, use_container_width=True)
        
    with c2:
        st.subheader("2. Platform Efficiency (ROAS)")
        plat_agg = df.groupby('Platform')[['Cost_Fee_INR', 'Total_Revenue_INR']].sum().reset_index()
        plat_agg['ROAS'] = plat_agg['Total_Revenue_INR'] / plat_agg['Cost_Fee_INR']
        chart2 = alt.Chart(plat_agg).mark_bar().encode(
            x=alt.X('ROAS', title='ROAS (x)'),
            y=alt.Y('Platform', sort='-x'),
            color='Platform',
            tooltip=['Platform', 'ROAS']
        ).interactive()
        st.altair_chart(chart2, use_container_width=True)
        
    st.divider()

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("3. Marketing Funnel")
        funnel_data = pd.DataFrame({
            'Stage': ['Impressions', 'Clicks', 'Orders'],
            'Count': [df['Impressions'].sum(), df['Link_Clicks'].sum(), df['Total_Orders'].sum()]
        })
        chart3 = alt.Chart(funnel_data).mark_bar().encode(
            y=alt.Y('Stage', sort=['Impressions', 'Clicks', 'Orders']),
            x='Count',
            color='Stage',
            tooltip=['Stage', 'Count']
        ).interactive()
        st.altair_chart(chart3, use_container_width=True)
        
    with c4:
        st.subheader("4. Campaign Trend (Time)")
        if 'Date_Posted' in df.columns:
            monthly = df.set_index('Date_Posted').resample('M')['Total_Revenue_INR'].sum().reset_index()
            chart4 = alt.Chart(monthly).mark_line(point=True).encode(
                x='Date_Posted',
                y='Total_Revenue_INR',
                tooltip=['Date_Posted', 'Total_Revenue_INR']
            ).interactive()
            st.altair_chart(chart4, use_container_width=True)

# ============================================================
#                   TAB 2: PLANNER (FIXED)
# ============================================================
with tab2:
    st.subheader("Budget Optimization Engine")
    
    # --- 1. SETTINGS ---
    col1, col2, col3 = st.columns(3)
    with col1:
        budget = st.number_input("Total Campaign Budget (₹)", 10000, 10000000, 500000, step=50000)
    with col2:
        cats = st.multiselect("Category Filter (Default: All)", df['Category'].unique())
    with col3:
        plats = st.multiselect("Platform Filter (Default: All)", df['Platform'].unique())
        
    # Apply Filters
    filtered = df.copy()
    if cats:
        filtered = filtered[filtered['Category'].isin(cats)]
    if plats:
        filtered = filtered[filtered['Platform'].isin(plats)]
        
    if filtered.empty:
        st.error("No data found for these filters. Please select broader categories.")
        st.stop()
        
    # --- 2. CALCULATIONS ---
    curve_params = fit_curves_heuristic(filtered)
    # Consider top 20 candidates for optimization speed
    candidate_pool = filtered.groupby('Influencer_ID')['Total_Revenue_INR'].mean().nlargest(20).index.tolist()
    
    # A. AI OPTIMIZER
    ai_allocation = maximize_revenue(budget, candidate_pool, curve_params)
    
    ai_rev = 0
    active_influencer_count = 0
    
    for inf, amt in ai_allocation.items():
        if amt > 1000: # Only count if budget > 1000 INR
            k, a, b = curve_params[inf]
            ai_rev += response_function(amt, k, a, b)
            active_influencer_count += 1
            
    ai_roas = ai_rev / budget
    
    # B. MANUAL STRATEGY
    st.write("---")
    st.markdown("#### Compare against Manual Strategy")
    manual_n = st.slider("Select Number of Influencers (Equal Split)", 1, 10, 5)
    
    manual_candidates = candidate_pool[:manual_n]
    manual_budget_per = budget / manual_n
    manual_rev = 0
    for inf in manual_candidates:
        k, a, b = curve_params[inf]
        manual_rev += response_function(manual_budget_per, k, a, b)
        
    manual_roas = manual_rev / budget
    
    # --- 3. RESULTS DISPLAY (FIXED METRICS) ---
    col_res1, col_res2 = st.columns(2)
    with col_res1:
        st.success("🤖 **AI Optimal Strategy**")
        # THIS IS THE METRIC YOU ASKED FOR
        st.metric("Optimal Influencer Count", f"{active_influencer_count}") 
        
        m1, m2 = st.columns(2)
        m1.metric("Expected Revenue", f"₹{ai_rev:,.0f}", delta=f"₹{ai_rev-manual_rev:,.0f}")
        m2.metric("Expected ROAS", f"{ai_roas:.2f}x")

    with col_res2:
        st.warning(f"👤 **Manual Strategy**")
        st.metric("Manual Count", f"{manual_n}")
        
        m3, m4 = st.columns(2)
        m3.metric("Expected Revenue", f"₹{manual_rev:,.0f}")
        m4.metric("Expected ROAS", f"{manual_roas:.2f}x")

    st.divider()

    # --- 4. BUDGET SIMULATOR GRAPH (FIXED) ---
    st.subheader("Budget Simulator Curve")
    st.caption("Revenue projection as budget scales (AI vs Manual)")
    
    # Generate robust data for the graph
    steps = 30
    x_vals = np.linspace(budget * 0.2, budget * 2.5, steps)
    
    y_ai = []
    y_man = []
    
    for x in x_vals:
        # AI Logic
        alloc = maximize_revenue(x, candidate_pool, curve_params)
        rev_s = sum([response_function(amt, *curve_params[i]) for i, amt in alloc.items()])
        y_ai.append(rev_s)
        
        # Manual Logic
        per_bud = x / manual_n
        rev_m = sum([response_function(per_bud, *curve_params[i]) for i in manual_candidates])
        y_man.append(rev_m)
        
    # Create clean DataFrame for Altair
    chart_df = pd.DataFrame({
        'Budget': np.tile(x_vals, 2),
        'Revenue': np.concatenate([y_ai, y_man]),
        'Strategy': ['AI Optimal'] * steps + [f'Manual ({manual_n} Inf)'] * steps
    })
    
    # Robust Altair Chart
    base = alt.Chart(chart_df).mark_line(point=False, strokeWidth=3).encode(
        x=alt.X('Budget', axis=alt.Axis(format='₹~s', title='Invested Budget')),
        y=alt.Y('Revenue', axis=alt.Axis(format='₹~s', title='Expected Revenue')),
        color=alt.Color('Strategy', scale=alt.Scale(domain=['AI Optimal', f'Manual ({manual_n} Inf)'], range=['#2ecc71', '#e74c3c'])),
        tooltip=['Strategy', 'Budget', 'Revenue']
    ).properties(height=400)
    
    # Add a vertical rule for current selected budget
    rule = alt.Chart(pd.DataFrame({'Budget': [budget]})).mark_rule(color='white', strokeDash=[5,5]).encode(x='Budget')

    st.altair_chart((base + rule).interactive(), use_container_width=True)
