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
            
    return df

def response_function(spend, k, alpha, beta):
    """
    Diminishing Returns Formula:
    Revenue = k * (Spend^alpha) * e^(-beta * Spend)
    """
    # Scaling to prevent overflow
    x = spend / 1_000_000 
    return np.maximum(0, k * (x ** alpha) * np.exp(-beta * x))

@st.cache_data
def fit_curves_heuristic(df):
    """
    Estimates curve parameters based on historical averages.
    """
    curve_params = {}
    grouped = df.groupby('Influencer_ID')
    
    for inf_id, group in grouped:
        if len(group) < 1: continue
        
        avg_roas = group['ROAS'].mean()
        avg_cost = group['Cost_Fee_INR'].mean()
        
        # Heuristic Logic:
        # High ROAS = High Alpha (Viral Lift)
        # High Cost = Low Beta (Slower Saturation, can take more money)
        k_est = avg_cost * avg_roas * 1.2 
        alpha_est = 0.8 + (np.log1p(avg_roas) / 10.0) 
        beta_est = 0.02 + (5000 / avg_cost) 
        
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
        return -total_rev # Negative because we minimize

    # Constraint: Sum of allocations must equal budget
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - budget})
    
    # Bounds: Each allocation must be between 0 and budget
    bounds = tuple((0, budget) for _ in range(n))
    
    # Initial Guess: Equal split
    initial_guess = [budget/n] * n
    
    result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    
    allocation = {inf: round(amt, 2) for inf, amt in zip(influencers, result.x)}
    return allocation

# ============================================================
#                   MAIN APP
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
    # Row 1: KPIs
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
    
    # Row 2: Charts
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Efficiency by Category")
        cat_agg = df.groupby('Category')[['Cost_Fee_INR', 'Total_Revenue_INR']].sum().reset_index()
        cat_agg['ROAS'] = cat_agg['Total_Revenue_INR'] / cat_agg['Cost_Fee_INR']
        
        c = alt.Chart(cat_agg).mark_bar().encode(
            x=alt.X('Category', sort='-y'),
            y='ROAS',
            color='Category',
            tooltip=['Category', 'ROAS']
        ).interactive()
        st.altair_chart(c, use_container_width=True)
        
    with c2:
        st.subheader("Spend vs Revenue Scatter")
        s = alt.Chart(df).mark_circle(size=60).encode(
            x='Cost_Fee_INR',
            y='Total_Revenue_INR',
            color='Platform',
            tooltip=['Influencer_ID', 'Platform', 'ROAS']
        ).interactive()
        st.altair_chart(s, use_container_width=True)

# ============================================================
#                   TAB 2: PLANNER
# ============================================================
with tab2:
    st.subheader("Budget Optimization Engine")
    
    # --- 1. FILTERS (Empty by default) ---
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
    # Fit curves
    curve_params = fit_curves_heuristic(filtered)
    
    # Pick Top Candidates (Pool of best performers to optimize within)
    # We take top 15 by historical mean revenue to limit computation time
    candidate_pool = filtered.groupby('Influencer_ID')['Total_Revenue_INR'].mean().nlargest(15).index.tolist()
    
    # A. RUN OPTIMIZER (AI)
    ai_allocation = maximize_revenue(budget, candidate_pool, curve_params)
    
    ai_rev = 0
    ai_spend = 0
    ai_data = []
    for inf, amt in ai_allocation.items():
        if amt > 1000: # Only count significant allocations
            k, a, b = curve_params[inf]
            rev = response_function(amt, k, a, b)
            ai_rev += rev
            ai_spend += amt
            ai_data.append({"Influencer": inf, "Budget": amt, "Revenue": rev})
            
    ai_roas = ai_rev / ai_spend if ai_spend > 0 else 0
    
    # B. RUN MANUAL (Equal Split Top 5)
    manual_n = 5
    manual_candidates = candidate_pool[:manual_n]
    manual_budget_per = budget / manual_n
    manual_rev = 0
    
    for inf in manual_candidates:
        k, a, b = curve_params[inf]
        manual_rev += response_function(manual_budget_per, k, a, b)
        
    manual_roas = manual_rev / budget
    
    # --- 3. RESULTS DISPLAY ---
    st.divider()
    
    col_res1, col_res2 = st.columns(2)
    
    with col_res1:
        st.success("🤖 **AI Optimal Strategy**")
        m1, m2 = st.columns(2)
        m1.metric("Optimal Influencers", f"{len(ai_data)}")
        m2.metric("Budget Utilized", f"₹{ai_spend:,.0f}")
        
        m3, m4 = st.columns(2)
        m3.metric("Expected Revenue", f"₹{ai_rev:,.0f}", delta=f"₹{ai_rev-manual_rev:,.0f}")
        m4.metric("Expected ROAS", f"{ai_roas:.2f}x", delta=f"{ai_roas-manual_roas:.2f}x")
        
        # ALLOCATION CHART
        st.write(" **Budget Allocation (AI)**")
        ai_df = pd.DataFrame(ai_data)
        if not ai_df.empty:
            pie = alt.Chart(ai_df).mark_arc(innerRadius=60).encode(
                theta='Budget',
                color=alt.Color('Influencer', legend=None),
                tooltip=['Influencer', 'Budget', 'Revenue']
            ).properties(height=200)
            st.altair_chart(pie, use_container_width=True)

    with col_res2:
        st.warning(f"👤 **Manual Strategy** (Top {manual_n} Equal Split)")
        m1, m2 = st.columns(2)
        m1.metric("Influencers", f"{manual_n}")
        m2.metric("Budget Allocated", f"₹{budget:,.0f}")
        
        m3, m4 = st.columns(2)
        m3.metric("Expected Revenue", f"₹{manual_rev:,.0f}")
        m4.metric("Expected ROAS", f"{manual_roas:.2f}x")
        
        # COMPARISON CHART
        st.write(" **Strategy Comparison**")
        comp_data = pd.DataFrame({
            'Strategy': ['AI Optimal', 'Manual'],
            'Revenue': [ai_rev, manual_rev]
        })
        bar = alt.Chart(comp_data).mark_bar().encode(
            x='Strategy',
            y='Revenue',
            color='Strategy'
        ).properties(height=200)
        st.altair_chart(bar, use_container_width=True)

    st.divider()

    # --- 4. BUDGET SIMULATOR (Fixed Graph) ---
    st.subheader("Budget Simulator Curve")
    st.caption("Projected Revenue as Budget Scales (AI vs Manual)")
    
    # Calculate points
    steps = 20
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
    
    # Create simple dataframe for line_chart
    chart_data = pd.DataFrame({
        "Budget": x_vals,
        "AI Optimal Revenue": y_ai,
        "Manual Revenue": y_man
    }).set_index("Budget")
    
    # Use standard Streamlit Line Chart (Robust)
    st.line_chart(chart_data)
    
    st.info(f"The Red Line (Manual) assumes you keep splitting budget equally among 5 people. The Blue Line (AI) dynamically shifts money to the best performer at every budget level.")
