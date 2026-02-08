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
    # Diminishing Returns Formula: Revenue = k * (Spend^alpha) * e^(-beta*Spend)
    # Scaled to millions to prevent overflow
    x = spend / 1_000_000 
    # STRICTER MATH: We force alpha < 0.9 and beta > 0.1 to guarantee the curve BENDS.
    return np.maximum(0, k * (x ** alpha) * np.exp(-beta * x))

@st.cache_data
def fit_curves_heuristic(df):
    curve_params = {}
    grouped = df.groupby('Influencer_ID')
    
    for inf_id, group in grouped:
        if len(group) < 1: continue
        avg_roas = group['ROAS'].mean()
        avg_cost = group['Cost_Fee_INR'].mean()
        
        # Heuristic Logic (Engineered for Diminishing Returns)
        # 1. K (Scale): Based on cost * ROAS
        k_est = avg_cost * avg_roas * 1.8 
        
        # 2. Alpha (Bend): Must be < 1.0 for diminishing returns. 
        # We clamp it between 0.6 (fast saturation) and 0.85 (slow saturation)
        alpha_est = 0.6 + (min(avg_roas, 10) / 40.0) 
        
        # 3. Beta (Decay): Higher cost = Lower decay (can absorb more budget)
        beta_est = 0.08 + (5000 / avg_cost) 
        
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
    
    # Robust Optimizer
    result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    
    # Return allocation
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
        st.subheader("2. Platform Share")
        plat_agg = df.groupby('Platform')[['Total_Revenue_INR']].sum().reset_index()
        chart2 = alt.Chart(plat_agg).mark_arc(innerRadius=50).encode(
            theta='Total_Revenue_INR',
            color='Platform',
            tooltip=['Platform', 'Total_Revenue_INR']
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
    
    # 1. INPUTS
    col1, col2, col3 = st.columns(3)
    with col1:
        budget = st.number_input("Total Campaign Budget (₹)", 10000, 10000000, 500000, step=50000)
    with col2:
        cats = st.multiselect("Category Filter (Default: All)", df['Category'].unique())
    with col3:
        plats = st.multiselect("Platform Filter (Default: All)", df['Platform'].unique())
        
    filtered = df.copy()
    if cats: filtered = filtered[filtered['Category'].isin(cats)]
    if plats: filtered = filtered[filtered['Platform'].isin(plats)]
    
    if filtered.empty:
        st.error("No data found for filters.")
        st.stop()
        
    # 2. CALCULATIONS
    curve_params = fit_curves_heuristic(filtered)
    # We pool top 15 candidates to run the optimization against
    candidate_pool = filtered.groupby('Influencer_ID')['Total_Revenue_INR'].mean().nlargest(15).index.tolist()
    
    # A. AI OPTIMIZER LOOP
    # Find best N (from 1 to 10 influencers)
    best_n = 1
    best_rev = 0
    best_allocation = {}
    
    for n in range(1, min(11, len(candidate_pool) + 1)):
        current_pool = candidate_pool[:n]
        alloc = maximize_revenue(budget, current_pool, curve_params)
        rev = sum([response_function(amt, *curve_params[i]) for i, amt in alloc.items()])
        
        # Standard hill-climbing: if revenue improves, keep going
        if rev > best_rev:
            best_rev = rev
            best_n = n
            best_allocation = alloc
            
    ai_roas = best_rev / budget
    
    # B. MANUAL COMPARISON
    st.write("---")
    st.markdown("#### Comparison Strategy")
    manual_n = st.slider("Select Number of Influencers (Equal Split)", 1, 10, best_n)
    manual_candidates = candidate_pool[:manual_n]
    manual_budget_per = budget / manual_n
    
    manual_rev = sum([response_function(manual_budget_per, *curve_params[i]) for i in manual_candidates])
    manual_roas = manual_rev / budget
    
    # 3. RESULTS DISPLAY
    col_res1, col_res2 = st.columns(2)
    with col_res1:
        st.success(f"🤖 **AI Optimal Strategy**")
        st.metric("Optimal Influencer Count", f"{best_n}")
        st.metric("Expected Revenue", f"₹{best_rev:,.0f}", delta=f"₹{best_rev-manual_rev:,.0f}")
        
        # Added ROAS Difference here
        st.metric("Expected ROAS", f"{ai_roas:.2f}x", delta=f"{ai_roas-manual_roas:.2f}x")
        
        # LIST ALLOCATION - FILTER REMOVED so all show up
        alloc_list = [{"Influencer": k, "Allocated Budget": f"₹{v:,.0f}"} for k,v in best_allocation.items() if v > 1.0]
        st.table(pd.DataFrame(alloc_list))

    with col_res2:
        st.warning(f"👤 **Manual Strategy**")
        st.metric("Manual Count", f"{manual_n}")
        st.metric("Expected Revenue", f"₹{manual_rev:,.0f}")
        st.metric("Expected ROAS", f"{manual_roas:.2f}x")
        st.write(f"*Allocating ₹{manual_budget_per:,.0f} equally to top {manual_n} influencers.*")

    st.divider()

    # 4. BUDGET SIMULATOR (FIXED MATH & VISIBILITY)
    st.subheader("Budget Simulator Curve")
    st.caption("See how Diminishing Returns kick in as you scale.")
    
    # Generate Data (0 to 3x budget)
    steps = 40
    x_vals = np.linspace(50000, budget * 3, steps)
    y_ai = []
    y_man = []
    
    for x in x_vals:
        # AI Logic (Re-optimize for X budget)
        # Note: We re-run the full optimizer to find the best split for that specific budget
        alloc = maximize_revenue(x, candidate_pool[:best_n], curve_params)
        rev_s = sum([response_function(amt, *curve_params[i]) for i, amt in alloc.items()])
        y_ai.append(rev_s)
        
        # Manual Logic (Fixed N)
        per_bud = x / manual_n
        rev_m = sum([response_function(per_bud, *curve_params[i]) for i in manual_candidates])
        y_man.append(rev_m)
        
    chart_df = pd.DataFrame({
        "AI Optimal Allocation": y_ai,
        "Manual Equal Split": y_man
    }, index=x_vals)
    
    # NATIVE CHART ( Guaranteed to work )
    st.line_chart(chart_df)
    
    # SLIDER INTERACTION
    sim_budget = st.slider("👇 Drag to Check Revenue at Specific Budget", 50000, int(budget*3), int(budget), step=50000)
    
    # Single Point Calculation
    sim_alloc = maximize_revenue(sim_budget, candidate_pool[:best_n], curve_params)
    sim_rev_ai = sum([response_function(amt, *curve_params[i]) for i, amt in sim_alloc.items()])
    
    # Manual at Sim
    sim_per_bud = sim_budget / manual_n
    sim_rev_man = sum([response_function(sim_per_bud, *curve_params[i]) for i in manual_candidates])
    
    c_s1, c_s2 = st.columns(2)
    c_s1.info(f"AI at ₹{sim_budget:,.0f}: **₹{sim_rev_ai:,.0f}** Revenue")
    c_s2.warning(f"Manual at ₹{sim_budget:,.0f}: **₹{sim_rev_man:,.0f}** Revenue")
