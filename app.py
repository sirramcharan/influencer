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
    h3 {
        padding-top: 1rem;
        padding-bottom: 0.5rem;
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
        # Generate Derived Metrics
        if 'ROAS' not in df.columns:
            df['ROAS'] = df['Total_Revenue_INR'] / df['Cost_Fee_INR']
        if 'Category' not in df.columns:
            df['Category'] = "General"
        # Ensure Date column exists for Time Series graph
        if 'Date_Posted' in df.columns:
            df['Date_Posted'] = pd.to_datetime(df['Date_Posted'])
            
    return df

def response_function(spend, k, alpha, beta):
    """
    Diminishing Returns Formula:
    Revenue = k * (Spend^alpha) * e^(-beta * Spend)
    """
    # Scaling to prevent overflow
    x = spend / 1_000_000 
    # We add a small epsilon to beta to force curvature (diminishing returns)
    return np.maximum(0, k * (x ** alpha) * np.exp(-(beta + 0.1) * x))

@st.cache_data
def fit_curves_heuristic(df):
    """
    Estimates curve parameters based on historical averages.
    Tweaked to ensure curves look 'bendy' (non-linear).
    """
    curve_params = {}
    grouped = df.groupby('Influencer_ID')
    
    for inf_id, group in grouped:
        if len(group) < 1: continue
        
        avg_roas = group['ROAS'].mean()
        avg_cost = group['Cost_Fee_INR'].mean()
        
        # Heuristic Logic for Curves:
        # High ROAS = High Alpha (Viral Lift)
        k_est = avg_cost * avg_roas * 1.5 
        alpha_est = 0.7 + (np.log1p(avg_roas) / 10.0) # < 1.0 ensures bending
        beta_est = 0.05 + (10000 / avg_cost) 
        
        curve_params[inf_id] = (k_est, alpha_est, beta_est)
            
    return curve_params

def maximize_revenue(budget, influencers, curve_params):
    """
    MATHEMATICAL OPTIMIZER (SLSQP)
    Finds the exact split of 'budget' across 'influencers' to maximize Total Revenue.
    """
    n = len(influencers)
    params = [curve_params[i] for i in influencers]
    
    # Objective: Minimize negative revenue (maximize revenue)
    def objective(allocations):
        total_rev = 0
        for i, spend in enumerate(allocations):
            k, a, b = params[i]
            total_rev += response_function(spend, k, a, b)
        return -total_rev 

    # Constraint: Sum of allocations must equal budget
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - budget})
    
    # Bounds: Each allocation must be between 0 and budget
    bounds = tuple((0, budget) for _ in range(n))
    
    # Initial Guess: Equal split
    initial_guess = [budget/n] * n
    
    # Run Optimization
    result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    
    allocation = {inf: round(amt, 2) for inf, amt in zip(influencers, result.x)}
    return allocation

# ============================================================
#                   MAIN APP UI
# ============================================================

st.title("The Algorithmic Marketer")
st.markdown("### ROI Optimization & Budget Allocation Engine")

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
#                   TAB 1: DASHBOARD
# ============================================================
with tab1:
    # --- Row 1: KPIs ---
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
    
    # --- Row 2: Charts (Category & Platform) ---
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

    # --- Row 3: Charts (Funnel & Trend) ---
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
        else:
            st.warning("Date column not found.")

# ============================================================
#                   TAB 2: PLANNER
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
        
    # Apply Logic: If empty list, treat as "All"
    filtered = df.copy()
    if cats:
        filtered = filtered[filtered['Category'].isin(cats)]
    if plats:
        filtered = filtered[filtered['Platform'].isin(plats)]
        
    if filtered.empty:
        st.error("No data found for these filters.")
        st.stop()
        
    # --- 2. CALCULATIONS ---
    curve_params = fit_curves_heuristic(filtered)
    candidate_pool = filtered.groupby('Influencer_ID')['Total_Revenue_INR'].mean().nlargest(20).index.tolist()
    
    # A. AI OPTIMIZER
    ai_allocation = maximize_revenue(budget, candidate_pool, curve_params)
    
    ai_rev = 0
    ai_spend = 0
    for inf, amt in ai_allocation.items():
        if amt > 1000:
            k, a, b = curve_params[inf]
            ai_rev += response_function(amt, k, a, b)
            ai_spend += amt
            
    ai_roas = ai_rev / ai_spend if ai_spend > 0 else 0
    
    # B. MANUAL STRATEGY (User Input)
    st.write("---")
    st.markdown("#### Compare against Manual Strategy")
    manual_n = st.slider("Select Number of Influencers for Manual Split", 1, 20, 5)
    
    manual_candidates = candidate_pool[:manual_n]
    manual_budget_per = budget / manual_n
    manual_rev = 0
    for inf in manual_candidates:
        k, a, b = curve_params[inf]
        manual_rev += response_function(manual_budget_per, k, a, b)
        
    manual_roas = manual_rev / budget
    
    # --- 3. RESULTS ---
    col_res1, col_res2 = st.columns(2)
    with col_res1:
        st.success("🤖 **AI Optimal Strategy**")
        st.metric("Expected Revenue", f"₹{ai_rev:,.0f}", delta=f"₹{ai_rev-manual_rev:,.0f}")
        st.metric("Expected ROAS", f"{ai_roas:.2f}x")

    with col_res2:
        st.warning(f"👤 **Manual Strategy** (Top {manual_n} Equal Split)")
        st.metric("Expected Revenue", f"₹{manual_rev:,.0f}")
        st.metric("Expected ROAS", f"{manual_roas:.2f}x")

    st.divider()

    # --- 4. INTERACTIVE BUDGET SIMULATOR ---
    st.subheader("Budget Simulator Curve")
    st.caption("Drag the slider below to see how Revenue and ROAS change at different budget levels.")
    
    # The Slider to "Scrub" the graph
    sim_budget = st.slider("👇 Scrub Budget Level", 100000, int(budget*2.5), int(budget), step=50000)
    
    # Calculate Curve Data
    x_vals = np.linspace(100000, budget * 2.5, 40)
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
    
    # Build Data for Chart
    source = pd.DataFrame({
        'Budget': np.tile(x_vals, 2),
        'Revenue': np.concatenate([y_ai, y_man]),
        'Strategy': ['AI Optimal'] * 40 + ['Manual'] * 40
    })

    # Base Chart
    base = alt.Chart(source).mark_line(point=False).encode(
        x=alt.X('Budget', axis=alt.Axis(format='₹~s')),
        y=alt.Y('Revenue', axis=alt.Axis(format='₹~s')),
        color='Strategy'
    )

    # Vertical Rule (controlled by slider)
    rule = alt.Chart(pd.DataFrame({'Budget': [sim_budget]})).mark_rule(color='red', strokeWidth=2).encode(
        x='Budget'
    )
    
    # Text Label for the Rule
    # Calculate exact revenue at sim_budget for AI to display on graph
    alloc_sim = maximize_revenue(sim_budget, candidate_pool, curve_params)
    rev_sim = sum([response_function(amt, *curve_params[i]) for i, amt in alloc_sim.items()])
    
    text = alt.Chart(pd.DataFrame({'Budget': [sim_budget], 'Revenue': [rev_sim]})).mark_text(
        align='left', dx=5, dy=-5, color='red', text=f"₹{rev_sim:,.0f}"
    ).encode(x='Budget', y='Revenue')

    # Combine
    st.altair_chart((base + rule + text).interactive(), use_container_width=True)
    
    st.info(f"At the selected budget of **₹{sim_budget:,.0f}**, the AI Strategy expects to generate **₹{rev_sim:,.0f}**.")
