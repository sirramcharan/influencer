import streamlit as st
import pandas as pd
import numpy as np
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
        
        # Heuristic Logic
        k_est = avg_cost * avg_roas * 1.5 
        alpha_est = 0.7 + (np.log1p(avg_roas) / 10.0) 
        beta_est = 0.05 + (10000 / avg_cost) 
        
        curve_params[inf_id] = (k_est, alpha_est, beta_est)
            
    return curve_params

def maximize_revenue(budget, influencers, curve_params):
    n = len(influencers)
    params = [curve_params[i] for i in influencers]
    
    # Objective: Minimize negative revenue
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
#                   TAB 1: DASHBOARD
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
    
    st.subheader("Revenue Trends")
    if 'Date_Posted' in df.columns:
        # Simple Line Chart using Streamlit Native (Bulletproof)
        monthly = df.set_index('Date_Posted').resample('M')['Total_Revenue_INR'].sum()
        st.line_chart(monthly)
    else:
        st.warning("No Date column found.")
        
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Revenue by Category")
        cat_data = df.groupby('Category')['Total_Revenue_INR'].sum()
        st.bar_chart(cat_data)
        
    with c2:
        st.subheader("Revenue by Platform")
        plat_data = df.groupby('Platform')['Total_Revenue_INR'].sum()
        st.bar_chart(plat_data)

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
        
    # Apply Filters
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
    # Consider top 15 candidates
    candidate_pool = filtered.groupby('Influencer_ID')['Total_Revenue_INR'].mean().nlargest(15).index.tolist()
    
    # A. AI OPTIMIZER
    # Iterate to find optimal NUMBER of influencers (1 to 10)
    best_n = 1
    best_rev = 0
    best_allocation = {}
    
    for n in range(1, min(11, len(candidate_pool))):
        current_candidates = candidate_pool[:n]
        alloc = maximize_revenue(budget, current_candidates, curve_params)
        rev = sum([response_function(amt, *curve_params[i]) for i, amt in alloc.items()])
        
        if rev > best_rev:
            best_rev = rev
            best_n = n
            best_allocation = alloc
            
    ai_roas = best_rev / budget
    
    # B. MANUAL STRATEGY (User Input)
    st.write("---")
    st.markdown("#### Compare against Manual Strategy")
    manual_n = st.slider("Select Number of Influencers (Equal Split)", 1, 10, best_n)
    
    manual_candidates = candidate_pool[:manual_n]
    manual_budget_per = budget / manual_n
    manual_rev = 0
    for inf in manual_candidates:
        k, a, b = curve_params[inf]
        manual_rev += response_function(manual_budget_per, k, a, b)
        
    manual_roas = manual_rev / budget
    
    # --- 3. RESULTS DISPLAY ---
    col_res1, col_res2 = st.columns(2)
    with col_res1:
        st.success("🤖 **AI Smart Allocation**")
        st.metric("Optimal Influencer Count", f"{best_n}")
        st.metric("Expected Revenue", f"₹{best_rev:,.0f}", delta=f"₹{best_rev-manual_rev:,.0f} vs Manual")
        st.metric("Expected ROAS", f"{ai_roas:.2f}x")
        
        st.markdown("**Allocation Details:**")
        # Creating a clean table for the best allocation
        alloc_list = []
        for inf, amt in best_allocation.items():
            if amt > 100:
                alloc_list.append({"Influencer ID": inf, "Allocated Budget": f"₹{amt:,.0f}"})
        st.table(pd.DataFrame(alloc_list))

    with col_res2:
        st.warning(f"👤 **Manual Strategy**")
        st.metric("Manual Count", f"{manual_n}")
        st.metric("Expected Revenue", f"₹{manual_rev:,.0f}")
        st.metric("Expected ROAS", f"{manual_roas:.2f}x")
        st.write(f"*Note: Manual strategy assumes you split the ₹{budget:,.0f} equally (₹{manual_budget_per:,.0f} each).*")

    st.divider()

    # --- 4. BUDGET SIMULATOR (FIXED WITH LINE CHART) ---
    st.subheader("Budget Simulator Curve")
    st.caption("Revenue projection as budget scales. Use slider below to check specific values.")
    
    # Graph Data Generation
    steps = 40
    x_vals = np.linspace(100000, budget * 3, steps) # Plot up to 3x budget
    
    y_ai = []
    y_man = []
    
    for x in x_vals:
        # AI (Dynamic N)
        # For speed in graph, we stick to the Best N found earlier, but re-optimize allocation
        candidates = candidate_pool[:best_n] 
        alloc = maximize_revenue(x, candidates, curve_params)
        rev_s = sum([response_function(amt, *curve_params[i]) for i, amt in alloc.items()])
        y_ai.append(rev_s)
        
        # Manual (Fixed N)
        per_bud = x / manual_n
        rev_m = sum([response_function(per_bud, *curve_params[i]) for i in manual_candidates])
        y_man.append(rev_m)
        
    chart_data = pd.DataFrame({
        "AI Optimal": y_ai,
        "Manual Equal Split": y_man
    }, index=x_vals)
    
    # NATIVE STREAMLIT CHART (100% VISIBLE)
    st.line_chart(chart_data)
    
    # SLIDER INTERACTION
    sim_budget = st.slider("👇 Drag to Simulate Specific Budget", 100000, int(budget*3), int(budget), step=50000)
    
    # Calculate for slider
    # AI Result at Sim Budget
    sim_alloc_ai = maximize_revenue(sim_budget, candidate_pool[:best_n], curve_params)
    sim_rev_ai = sum([response_function(amt, *curve_params[i]) for i, amt in sim_alloc_ai.items()])
    
    # Manual Result at Sim Budget
    sim_per_bud = sim_budget / manual_n
    sim_rev_man = sum([response_function(sim_per_bud, *curve_params[i]) for i in manual_candidates])
    
    c_s1, c_s2 = st.columns(2)
    c_s1.info(f"AI at ₹{sim_budget:,.0f}: **₹{sim_rev_ai:,.0f}** Revenue")
    c_s2.warning(f"Manual at ₹{sim_budget:,.0f}: **₹{sim_rev_man:,.0f}** Revenue")
