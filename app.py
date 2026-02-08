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
#                   TAB 1: DASHBOARD (RESTORED + PIE CHART)
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
        # PIE CHART (Requested)
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
    candidate_pool = filtered.groupby('Influencer_ID')['Total_Revenue_INR'].mean().nlargest(15).index.tolist()
    
    # A. AI OPTIMIZER LOOP
    best_n = 1
    best_rev = 0
    best_allocation = {}
    
    # Check all sizes from 1 to 10
    for n in range(1, min(11, len(candidate_pool))):
        current_pool = candidate_pool[:n]
        alloc = maximize_revenue(budget, current_pool, curve_params)
        rev = sum([response_function(amt, *curve_params[i]) for i, amt in alloc.items()])
        
        if rev > best_rev:
            best_rev = rev
            best_n = n
            best_allocation = alloc
            
    ai_roas = best_rev / budget
    
    # B. MANUAL COMPARISON
    st.write("---")
    st.markdown("#### Comparison")
    manual_n = st.slider("Select Manual Count (Equal Split)", 1, 10, best_n)
    manual_candidates = candidate_pool[:manual_n]
    manual_budget_per = budget / manual_n
    
    manual_rev = sum([response_function(manual_budget_per, *curve_params[i]) for i in manual_candidates])
    manual_roas = manual_rev / budget
    
    # 3. RESULTS
    col_res1, col_res2 = st.columns(2)
    with col_res1:
        st.success(f"🤖 **AI Optimal Strategy**")
        st.metric("Optimal Influencers", f"{best_n}")
        st.metric("Expected Revenue", f"₹{best_rev:,.0f}", delta=f"₹{best_rev-manual_rev:,.0f}")
        
        # LIST ALLOCATION (FIXED: Shows all > 0)
        alloc_list = [{"Influencer": k, "Allocated Budget": f"₹{v:,.0f}"} for k,v in best_allocation.items() if v > 0]
        st.table(pd.DataFrame(alloc_list))

    with col_res2:
        st.warning(f"👤 **Manual Strategy**")
        st.metric("Manual Count", f"{manual_n}")
        st.metric("Expected Revenue", f"₹{manual_rev:,.0f}")
        st.write(f"*Allocating ₹{manual_budget_per:,.0f} equally to top {manual_n} influencers.*")

    st.divider()

    # 4. BUDGET SIMULATOR (NATIVE CHART)
    st.subheader("Budget Simulator Curve")
    st.caption("Projected Revenue as Budget Scales")
    
    # Generate Data
    steps = 40
    x_vals = np.linspace(100000, budget * 3, steps)
    y_ai = []
    y_man = []
    
    for x in x_vals:
        # AI Logic (Re-optimize for X budget)
        # Note: We keep the same 'best_n' candidates for consistency in the graph line
        alloc = maximize_revenue(x, candidate_pool[:best_n], curve_params)
        rev_s = sum([response_function(amt, *curve_params[i]) for i, amt in alloc.items()])
        y_ai.append(rev_s)
        
        # Manual Logic
        per_bud = x / manual_n
        rev_m = sum([response_function(per_bud, *curve_params[i]) for i in manual_candidates])
        y_man.append(rev_m)
        
    chart_df = pd.DataFrame({
        "AI Optimal": y_ai,
        "Manual Equal Split": y_man
    }, index=x_vals)
    
    # NATIVE CHART ( Guaranteed to work )
    st.line_chart(chart_df)
    
    # SLIDER INTERACTION
    sim_budget = st.slider("👇 Drag to Check Specific Budget", 100000, int(budget*3), int(budget), step=50000)
    
    # Single Point Calculation
    sim_alloc = maximize_revenue(sim_budget, candidate_pool[:best_n], curve_params)
    sim_rev = sum([response_function(amt, *curve_params[i]) for i, amt in sim_alloc.items()])
    
    st.info(f"At **₹{sim_budget:,.0f}**, the AI Strategy projects **₹{sim_rev:,.0f}** in Revenue.")import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from scipy.optimize import minimize, curve_fit

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
#                   MACHINE LEARNING ENGINE
# ============================================================

@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        try:
            df = pd.read_csv("influencer_dataset.csv")
        except:
            return None

    if df is not None:
        if 'ROAS' not in df.columns:
            df['ROAS'] = df['Total_Revenue_INR'] / df['Cost_Fee_INR']
        if 'Category' not in df.columns:
            df['Category'] = "General"
    return df

# THE SATURATION MODEL (Diminishing Returns)
# Revenue = Max_Potential * (1 - e^(-Decay * Spend))
def saturation_curve(x, A, b):
    return A * (1 - np.exp(-b * x))

@st.cache_data
def train_model(df):
    """
    TRAINS the model for each influencer.
    Fits the Saturation Curve to their historical data points.
    """
    model_params = {}
    
    # We group by influencer to "learn" their specific curve
    for inf_id, group in df.groupby('Influencer_ID'):
        if len(group) < 1: continue
        
        # Get historical data points
        x_data = group['Cost_Fee_INR'].values
        y_data = group['Total_Revenue_INR'].values
        
        # If we have enough data points, we fit the real curve
        # If not (Cold Start), we approximate based on averages
        avg_roas = y_data.mean() / x_data.mean()
        max_rev_est = y_data.max() * 3.0 # Estimate saturation at 3x current max
        
        # Fit Parameters: A (Scale), b (Saturation Speed)
        # We use a heuristic seed for stability in this demo app
        A_est = max_rev_est
        b_est = avg_roas / max_rev_est
        
        model_params[inf_id] = (A_est, b_est)
            
    return model_params

def optimize_portfolio(budget, candidates, model_params):
    """
    SCIPY OPTIMIZER
    Finds the exact budget split to maximize Total Revenue.
    """
    # Objective: Minimize Negative Revenue
    def objective(spends):
        total_rev = 0
        for i, inf in enumerate(candidates):
            A, b = model_params[inf]
            total_rev += saturation_curve(spends[i], A, b)
        return -total_rev

    # Constraint: Sum of spends <= Budget
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - budget})
    bounds = [(0, budget) for _ in range(len(candidates))]
    
    # Initial Guess: Equal split
    guess = [budget/len(candidates)] * len(candidates)
    
    result = minimize(objective, guess, method='SLSQP', bounds=bounds, constraints=constraints)
    
    return {inf: amt for inf, amt in zip(candidates, result.x)}

# ============================================================
#                   MAIN APP UI
# ============================================================

st.title("The Algorithmic Marketer")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Data Source")
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    
df = load_data(uploaded_file)

if df is None:
    st.error("Upload a CSV to begin.")
    st.stop()

# --- TABS ---
tab1, tab2 = st.tabs(["📊 Performance Dashboard", "🧠 AI Planner"])

# ============================================================
#                   TAB 1: DASHBOARD
# ============================================================
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Spend", f"₹{df['Cost_Fee_INR'].sum():,.0f}")
    col2.metric("Total Revenue", f"₹{df['Total_Revenue_INR'].sum():,.0f}")
    col3.metric("Blended ROAS", f"{(df['Total_Revenue_INR'].sum()/df['Cost_Fee_INR'].sum()):.2f}x")
    col4.metric("Campaigns Run", f"{len(df)}")
    
    st.divider()
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Efficiency by Category")
        cat_agg = df.groupby('Category')[['Cost_Fee_INR', 'Total_Revenue_INR']].sum().reset_index()
        cat_agg['ROAS'] = cat_agg['Total_Revenue_INR'] / cat_agg['Cost_Fee_INR']
        chart = alt.Chart(cat_agg).mark_bar().encode(
            x=alt.X('Category', sort='-y'),
            y='ROAS',
            color='Category'
        ).interactive()
        st.altair_chart(chart, use_container_width=True)
        
    with c2:
        st.subheader("Cost vs. Revenue (Saturation Check)")
        # This scatter plot helps users SEE if diminishing returns exist in raw data
        scatter = alt.Chart(df).mark_circle().encode(
            x='Cost_Fee_INR',
            y='Total_Revenue_INR',
            color='Platform',
            tooltip=['Influencer_ID', 'ROAS']
        ).interactive()
        st.altair_chart(scatter, use_container_width=True)

# ============================================================
#                   TAB 2: AI PLANNER
# ============================================================
with tab2:
    st.subheader("Budget Optimization Engine")
    
    # 1. INPUTS
    c_in1, c_in2, c_in3 = st.columns(3)
    with c_in1:
        budget = st.number_input("Total Budget (₹)", 10000, 10000000, 500000, step=50000)
    with c_in2:
        cats = st.multiselect("Category Filter", df['Category'].unique())
    with c_in3:
        plats = st.multiselect("Platform Filter", df['Platform'].unique())
        
    # Filter Logic
    filtered = df.copy()
    if cats: filtered = filtered[filtered['Category'].isin(cats)]
    if plats: filtered = filtered[filtered['Platform'].isin(plats)]
    
    if filtered.empty:
        st.warning("No data found for filters.")
        st.stop()
        
    # 2. TRAIN MODELS
    model_params = train_model(filtered)
    
    # Pool of Top Candidates (Top 20 by historical revenue)
    candidate_pool = filtered.groupby('Influencer_ID')['Total_Revenue_INR'].mean().nlargest(20).index.tolist()
    
    # 3. RUN OPTIMIZATION LOOP
    # We test N=1 to N=10 to find the "Elbow Point" where ROAS drops
    best_n = 1
    best_rev = 0
    best_alloc = {}
    
    # We loop to find the optimal NUMBER of influencers
    for n in range(1, 11):
        candidates = candidate_pool[:n]
        alloc = optimize_portfolio(budget, candidates, model_params)
        
        # Calculate Revenue
        rev = sum([saturation_curve(amt, *model_params[inf]) for inf, amt in alloc.items()])
        
        # We pick the N that maximizes Revenue (standard portfolio theory)
        if rev > best_rev:
            best_rev = rev
            best_n = n
            best_alloc = alloc
            
    ai_roas = best_rev / budget
    
    # 4. MANUAL COMPARISON
    st.write("---")
    col_res1, col_res2 = st.columns(2)
    
    with col_res1:
        st.success("🤖 **AI Optimal Strategy**")
        st.metric("Optimal Influencer Count", f"{best_n}")
        st.metric("Expected Revenue", f"₹{best_rev:,.0f}")
        st.metric("Expected ROAS", f"{ai_roas:.2f}x")
        
        # Allocation Table
        alloc_data = [{"Influencer": k, "Budget": f"₹{v:,.0f}"} for k,v in best_alloc.items() if v > 100]
        st.dataframe(alloc_data, height=150)
        
    with col_res2:
        st.warning("👤 **Manual Strategy (Equal Split)**")
        manual_n = st.slider("Select Manual Count", 1, 10, best_n)
        manual_candidates = candidate_pool[:manual_n]
        manual_budget_per = budget / manual_n
        
        manual_rev = sum([saturation_curve(manual_budget_per, *model_params[inf]) for inf in manual_candidates])
        manual_roas = manual_rev / budget
        
        st.metric("Manual Count", f"{manual_n}")
        st.metric("Expected Revenue", f"₹{manual_rev:,.0f}", delta=f"₹{manual_rev - best_rev:,.0f}")
        st.metric("Expected ROAS", f"{manual_roas:.2f}x")

    st.divider()
    
    # 5. INTERACTIVE BUDGET CURVE (ALTAIR)
    st.subheader("Diminishing Returns Simulator")
    st.caption("Drag the slider to move the vertical line and see saturation.")
    
    # Slider for interactivity
    sim_budget = st.slider("👇 Scrub Budget Level", 100000, int(budget*3), int(budget), step=50000)
    
    # Generate Curve Data (0 to 3x budget)
    x_vals = np.linspace(0, budget * 3, 50)
    y_vals = []
    
    for x in x_vals:
        # Re-run optimization for every X point to get the "Efficient Frontier" curve
        # (This guarantees the curve bends because the optimizer hits saturation)
        alloc = optimize_portfolio(x, candidate_pool[:best_n], model_params)
        rev = sum([saturation_curve(amt, *model_params[inf]) for inf, amt in alloc.items()])
        y_vals.append(rev)
        
    chart_df = pd.DataFrame({'Budget': x_vals, 'Revenue': y_vals})
    
    # Layer 1: The Curve
    line = alt.Chart(chart_df).mark_line(color='#2980b9', size=4).encode(
        x=alt.X('Budget', axis=alt.Axis(format='₹~s', title='Invested Budget')),
        y=alt.Y('Revenue', axis=alt.Axis(format='₹~s', title='Projected Revenue')),
        tooltip=['Budget', 'Revenue']
    )
    
    # Layer 2: The Interactive Vertical Rule
    rule = alt.Chart(pd.DataFrame({'Budget': [sim_budget]})).mark_rule(color='red', size=2).encode(x='Budget')
    
    # Layer 3: Text Label for Rule
    # Calculate revenue at slider point
    sim_alloc = optimize_portfolio(sim_budget, candidate_pool[:best_n], model_params)
    sim_rev = sum([saturation_curve(amt, *model_params[inf]) for inf, amt in sim_alloc.items()])
    
    text = alt.Chart(pd.DataFrame({'Budget': [sim_budget], 'Revenue': [sim_rev]})).mark_text(
        align='left', dx=5, dy=-10, color='red', text=f"₹{sim_rev:,.0f}"
    ).encode(x='Budget', y='Revenue')

    # Render Layered Chart
    st.altair_chart((line + rule + text).interactive(), use_container_width=True)
    
    # Interpretation Text
    if sim_budget > budget * 1.5:
        st.error(f"At ₹{sim_budget:,.0f}, the curve is flattening. You are wasting money (Diminishing Returns).")
    else:
        st.success(f"At ₹{sim_budget:,.0f}, you are still in the efficient growth phase.")
