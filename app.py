import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from scipy.optimize import curve_fit

# ============================================================
#                   CONFIG & STYLING
# ============================================================

st.set_page_config(page_title="Algorithmic Marketer", layout="wide")

# Custom CSS for clear visibility in both Light/Dark modes
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
#                   DATA & MATH FUNCTIONS
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

def response_curve(spend, k, alpha, beta):
    # Normalized spend to avoid math overflow (millions)
    x = spend / 1_000_000 
    return np.maximum(0, k * (x ** alpha) * np.exp(-beta * x))

@st.cache_data
def fit_curves(df):
    """
    Fits a curve for every influencer.
    Returns: Dict {id: (k, alpha, beta)}
    """
    curve_params = {}
    grouped = df.groupby('Influencer_ID')
    
    for inf_id, group in grouped:
        if len(group) < 1: continue
        
        avg_roas = group['ROAS'].mean()
        avg_cost = group['Cost_Fee_INR'].mean()
        
        # Heuristic fitting logic for demo purposes
        k_est = avg_cost * avg_roas * 1.5 
        alpha_est = 0.85 + (avg_roas / 25.0) 
        beta_est = 0.05 + (15000 / avg_cost) 
        
        curve_params[inf_id] = (k_est, alpha_est, beta_est)
            
    return curve_params

def optimize_budget(total_budget, influencers, curve_params):
    """
    Smart Allocation Algorithm:
    Allocates budget in small steps to the influencer with the highest 
    Marginal ROAS at that specific moment.
    """
    allocation = {inf: 0.0 for inf in influencers}
    remaining_budget = total_budget
    step_size = total_budget * 0.01  # 1% steps for precision
    
    # Pre-fetch params
    params = {inf: curve_params[inf] for inf in influencers if inf in curve_params}
    
    while remaining_budget > step_size:
        best_inf = None
        best_marginal_gain = -1
        
        for inf, p in params.items():
            k, a, b = p
            current_spend = allocation[inf]
            
            # Calculate Marginal Revenue (Revenue of next step - Revenue now)
            rev_now = response_curve(current_spend, k, a, b)
            rev_next = response_curve(current_spend + step_size, k, a, b)
            marginal_gain = rev_next - rev_now
            
            if marginal_gain > best_marginal_gain:
                best_marginal_gain = marginal_gain
                best_inf = inf
        
        # Stop if even the best option yields zero or negative return (Saturation)
        if best_marginal_gain <= 0:
            break
            
        if best_inf:
            allocation[best_inf] += step_size
            remaining_budget -= step_size
            
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
else:
    st.sidebar.success("Data Loaded Successfully")

# --- TABS ---
tab1, tab2 = st.tabs(["📊 Performance Overview", "🚀 Campaign Planner"])

# ============================================================
#                   TAB 1: DASHBOARD
# ============================================================
with tab1:
    # --- ROW 1: KPIs ---
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
    
    # --- ROW 2: CHARTS ---
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Revenue by Category")
        cat_chart = alt.Chart(df).mark_bar().encode(
            x=alt.X('Category', sort='-y'),
            y='sum(Total_Revenue_INR)',
            color='Category',
            tooltip=['Category', 'sum(Total_Revenue_INR)']
        ).interactive()
        st.altair_chart(cat_chart, use_container_width=True)
        
    with c2:
        st.subheader("Platform Efficiency (ROAS)")
        plat_agg = df.groupby('Platform').agg(
            Spend=('Cost_Fee_INR', 'sum'),
            Revenue=('Total_Revenue_INR', 'sum')
        ).reset_index()
        plat_agg['ROAS'] = plat_agg['Revenue'] / plat_agg['Spend']
        
        roas_chart = alt.Chart(plat_agg).mark_bar().encode(
            x=alt.X('ROAS', title='ROAS Multiplier'),
            y=alt.Y('Platform', sort='-x'),
            color=alt.Color('ROAS', scale=alt.Scale(scheme='greens')),
            tooltip=['Platform', 'ROAS']
        ).interactive()
        st.altair_chart(roas_chart, use_container_width=True)

# ============================================================
#                   TAB 2: PLANNER
# ============================================================
with tab2:
    st.subheader("Smart Budget Allocator")
    
    # INPUTS
    col_in1, col_in2, col_in3 = st.columns(3)
    with col_in1:
        budget = st.number_input("Total Budget (₹)", 10000, 10000000, 500000, step=50000)
    with col_in2:
        cats = st.multiselect("Category", df['Category'].unique(), default=df['Category'].unique())
    with col_in3:
        plats = st.multiselect("Platform", df['Platform'].unique(), default=df['Platform'].unique())
        
    # FILTER LOGIC
    filtered = df[df['Category'].isin(cats) & df['Platform'].isin(plats)]
    if filtered.empty:
        st.error("No influencers match filters.")
        st.stop()
        
    curve_params = fit_curves(filtered)
    
    # 1. AI OPTIMAL ALLOCATION
    # Get top 20 candidates (filtering pool size for speed)
    top_candidates = filtered.groupby('Influencer_ID')['Total_Revenue_INR'].mean().nlargest(20).index.tolist()
    
    # Run Optimizer
    smart_allocation = optimize_budget(budget, top_candidates, curve_params)
    
    # Parse Results
    opt_revenue = 0
    active_influencers = 0
    utilized_budget = 0
    
    alloc_data = []
    
    for inf, amt in smart_allocation.items():
        if amt > 100: # Filter out negligible amounts
            k, a, b = curve_params[inf]
            rev = response_curve(amt, k, a, b)
            opt_revenue += rev
            active_influencers += 1
            utilized_budget += amt
            alloc_data.append({"Influencer": inf, "Allocated Budget": amt, "Expected Revenue": rev})
            
    opt_roas = opt_revenue / utilized_budget if utilized_budget > 0 else 0
    
    # 2. MANUAL SELECTION (Comparison)
    user_n = st.slider("Select Manual Count (for comparison)", 1, 10, 5)
    manual_candidates = top_candidates[:user_n]
    manual_budget_per = budget / user_n
    manual_revenue = 0
    
    for inf in manual_candidates:
        k, a, b = curve_params[inf]
        manual_revenue += response_curve(manual_budget_per, k, a, b)
        
    manual_roas = manual_revenue / budget
    
    st.divider()
    
    # --- RESULTS DISPLAY ---
    col_res1, col_res2 = st.columns(2)
    
    with col_res1:
        st.success("🤖 **AI Smart Allocation**")
        
        # NEW: Optimal Count & Budget Metrics
        m1, m2 = st.columns(2)
        m1.metric("Optimal Influencers", f"{active_influencers}")
        m2.metric("Budget Utilized", f"₹{utilized_budget:,.0f}")
        
        m3, m4 = st.columns(2)
        m3.metric("Expected Revenue", f"₹{opt_revenue:,.0f}", delta=f"₹{opt_revenue-manual_revenue:,.0f}")
        m4.metric("Expected ROAS", f"{opt_roas:.2f}x")
        
        with st.expander("View Allocation Table"):
             st.dataframe(pd.DataFrame(alloc_data).style.format({"Allocated Budget": "₹{:,.0f}", "Expected Revenue": "₹{:,.0f}"}))

    with col_res2:
        st.warning(f"👤 **Manual Selection** (Top {user_n} Equally)")
        st.metric("Influencers Selected", f"{user_n}")
        st.metric("Budget Allocated", f"₹{budget:,.0f}")
        st.metric("Expected Revenue", f"₹{manual_revenue:,.0f}")
        st.metric("Expected ROAS", f"{manual_roas:.2f}x")

    st.divider()
    
    # --- DIMINISHING RETURNS SIMULATOR ---
    st.subheader("Budget Simulator Curve")
    st.caption("See how Revenue changes as you increase budget (Smart vs Manual)")
    
    # Generate Data for Graph
    # We create a range from 10% of budget to 200% of budget
    x_points = np.linspace(budget * 0.1, budget * 2.5, 30)
    
    y_smart = []
    y_manual = []
    
    for x in x_points:
        # Smart Curve
        alloc = optimize_budget(x, top_candidates, curve_params)
        rev_s = sum([response_curve(amt, *curve_params[i]) for i, amt in alloc.items()])
        y_smart.append(rev_s)
        
        # Manual Curve
        per_bud = x / user_n
        rev_m = sum([response_curve(per_bud, *curve_params[i]) for i in manual_candidates])
        y_manual.append(rev_m)
        
    # Build DataFrame
    chart_df = pd.DataFrame({
        'Budget': np.concatenate([x_points, x_points]),
        'Revenue': np.concatenate([y_smart, y_manual]),
        'Strategy': ['AI Smart Allocation'] * 30 + [f'Manual (Top {user_n})'] * 30
    })
    
    # Altair Chart
    base_chart = alt.Chart(chart_df).mark_line(point=True).encode(
        x=alt.X('Budget', axis=alt.Axis(format='₹~s', title='Total Budget Invested')),
        y=alt.Y('Revenue', axis=alt.Axis(format='₹~s', title='Expected Revenue')),
        color='Strategy',
        tooltip=['Budget', 'Revenue', 'Strategy']
    ).properties(height=400)
    
    # Current Budget Line
    vline = alt.Chart(pd.DataFrame({'x': [budget]})).mark_rule(color='red', strokeDash=[5,5]).encode(x='x')
    
    st.altair_chart(base_chart + vline, use_container_width=True)
