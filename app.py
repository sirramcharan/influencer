import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from scipy.optimize import curve_fit

# ============================================================
#                   CONFIG & STYLING
# ============================================================

st.set_page_config(page_title="Algorithmic Marketer ROI Tool", layout="wide")

# Custom CSS to make metrics look professional
st.markdown("""
<style>
    .metric-card {
        background-color: #f9f9f9;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        border-bottom: 2px solid #ff4b4b;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
#                   DATA & MATH FUNCTIONS
# ============================================================

@st.cache_data
def load_data(uploaded_file=None):
    """
    Load dataset from default CSV or User Upload.
    """
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

    # Pre-processing
    if df is not None:
        if 'ROAS' not in df.columns:
            df['ROAS'] = df['Total_Revenue_INR'] / df['Cost_Fee_INR']
        
        # Ensure Category exists (fallback if old dataset)
        if 'Category' not in df.columns:
            df['Category'] = "General"
            
    return df

# The "Diminishing Returns" Math
def response_curve(spend, k, alpha, beta):
    # Normalized spend to avoid overflow
    x = spend / 1_000_000 
    return np.maximum(0, k * (x ** alpha) * np.exp(-beta * x))

@st.cache_data
def fit_curves(df):
    """
    Fit a custom curve for every influencer in the filtered list.
    Returns: Dict {id: (k, alpha, beta)}
    """
    curve_params = {}
    grouped = df.groupby('Influencer_ID')
    
    for inf_id, group in grouped:
        if len(group) < 1: continue
        
        # In a real app with limited data, we simulate curve fitting 
        # using the influencer's historical average performance as a seed.
        avg_roas = group['ROAS'].mean()
        avg_cost = group['Cost_Fee_INR'].mean()
        
        # Heuristic fitting for demo purposes (since real fitting needs 5+ points per person)
        # Higher ROAS = Higher Alpha (Lift)
        # Higher Cost = Lower Beta (Slower Saturation)
        
        k_est = avg_cost * avg_roas * 1.5 
        alpha_est = 0.8 + (avg_roas / 20.0) # slightly dynamic alpha
        beta_est = 0.05 + (10000 / avg_cost) # expensive influencers saturate slower
        
        curve_params[inf_id] = (k_est, alpha_est, beta_est)
            
    return curve_params

# ============================================================
#                   MAIN APP UI
# ============================================================

st.title("The Algorithmic Marketer")
st.markdown("### ROI Optimization & Budget Allocation Engine")

# --- 1. DATA LOADER ---
with st.sidebar:
    st.header("1. Data Source")
    uploaded_file = st.file_uploader("Upload Campaign CSV", type=['csv'])
    st.caption("If no file is uploaded, the demo dataset is used.")

df = load_data(uploaded_file)

if df is None:
    st.error("⚠️ Dataset not found! Please upload a CSV or ensure 'influencer_dataset.csv' is in the folder.")
    st.stop()
else:
    st.sidebar.success(f"Loaded {len(df):,} rows.")

# --- 2. TOP NAVIGATION ---
tab_dash, tab_plan = st.tabs(["📊 Executive Dashboard", "🧠 Strategic Planner"])

# ============================================================
#                   TAB 1: DASHBOARD
# ============================================================
with tab_dash:
    st.markdown("### Historical Performance Overview")
    
    # KPIs
    total_spend = df['Cost_Fee_INR'].sum()
    total_rev = df['Total_Revenue_INR'].sum()
    roas = total_rev / total_spend
    avg_order = df['Total_Revenue_INR'].sum() / df['Total_Orders'].sum()
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Spend", f"₹{total_spend:,.0f}")
    c2.metric("Total Revenue", f"₹{total_rev:,.0f}")
    c3.metric("Blended ROAS", f"{roas:.2f}x")
    c4.metric("Avg Order Value", f"₹{avg_order:,.0f}")
    
    st.divider()
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Efficiency by Category")
        cat_perf = df.groupby('Category').agg(
            Spend=('Cost_Fee_INR', 'sum'),
            Revenue=('Total_Revenue_INR', 'sum')
        ).reset_index()
        cat_perf['ROAS'] = cat_perf['Revenue'] / cat_perf['Spend']
        
        chart_cat = alt.Chart(cat_perf).mark_bar().encode(
            x=alt.X('Category', sort='-y'),
            y=alt.Y('ROAS', title='ROAS (x)'),
            color=alt.Color('ROAS', scale=alt.Scale(scheme='greens')),
            tooltip=['Category', 'ROAS', 'Spend']
        ).interactive()
        st.altair_chart(chart_cat, use_container_width=True)
        
    with col2:
        st.subheader("Revenue Share by Platform")
        plat_perf = df.groupby('Platform')['Total_Revenue_INR'].sum().reset_index()
        
        chart_pie = alt.Chart(plat_perf).mark_arc(innerRadius=50).encode(
            theta=alt.Theta(field="Total_Revenue_INR", type="quantitative"),
            color=alt.Color(field="Platform", type="nominal"),
            tooltip=['Platform', 'Total_Revenue_INR']
        )
        st.altair_chart(chart_pie, use_container_width=True)

# ============================================================
#                   TAB 2: PLANNER
# ============================================================
with tab_plan:
    st.markdown("### Budget Optimization Engine")
    
    # --- A. INPUTS ---
    with st.container():
        st.write("#### 1. Define Campaign Constraints")
        c1, c2, c3 = st.columns(3)
        
        with c1:
            budget_input = st.number_input("Total Budget (₹)", 10000, 10000000, 500000, step=50000)
        with c2:
            cat_filter = st.multiselect("Category", df['Category'].unique(), default=df['Category'].unique())
        with c3:
            plat_filter = st.multiselect("Platform", df['Platform'].unique(), default=df['Platform'].unique())

    # Filter Data
    filtered_df = df[df['Category'].isin(cat_filter) & df['Platform'].isin(plat_filter)]
    
    if filtered_df.empty:
        st.warning("No influencers found for these filters.")
        st.stop()
        
    # Fit Curves for filtered influencers
    curve_params = fit_curves(filtered_df)
    
    st.divider()
    
    # --- B. SIMULATION (USER CHOICE) ---
    st.write("#### 2. User Scenario Simulation")
    st.caption("What happens if you allocate this budget YOUR way?")
    
    u_col1, u_col2 = st.columns([1, 2])
    with u_col1:
        user_n = st.slider("How many influencers do you want to hire?", 1, 20, 5)
        st.info(f"Budget per influencer: **₹{budget_input/user_n:,.0f}**")
    
    # Calculate User Result (Naive Equal Split)
    # We take the Top N influencers by raw historical revenue and split budget equally
    top_n_user = filtered_df.groupby('Influencer_ID')['Total_Revenue_INR'].mean().nlargest(user_n).index.tolist()
    
    user_rev = 0
    per_inf_budget = budget_input / user_n
    
    for inf in top_n_user:
        if inf in curve_params:
            k, a, b = curve_params[inf]
            user_rev += response_curve(per_inf_budget, k, a, b)
            
    user_roas = user_rev / budget_input if budget_input > 0 else 0
    
    with u_col2:
        m1, m2 = st.columns(2)
        m1.metric("Your Expected Revenue", f"₹{user_rev:,.0f}")
        m2.metric("Your Expected ROAS", f"{user_roas:.2f}x")
        if user_roas < 1.5:
            st.warning("⚠️ Low Efficiency detected. You might be diluting your budget too much.")

    st.divider()

    # --- C. OPTIMIZATION (AI RECOMMENDATION) ---
    st.write("#### 3. AI Optimal Recommendation")
    st.caption("The system analyzed diminishing returns to find the mathematical maximum.")

    # Optimizer Logic (Brute force best N for simplicity in demo)
    best_n = 1
    best_rev = 0
    best_roas = 0
    best_combo_ids = []
    
    # Test N from 1 to 10
    possible_influencers = filtered_df.groupby('Influencer_ID')['Total_Revenue_INR'].mean().sort_values(ascending=False).index.tolist()
    
    # Limit search space
    search_limit = min(len(possible_influencers), 10)
    
    for n in range(1, search_limit + 1):
        test_budget_per = budget_input / n
        current_combo = possible_influencers[:n]
        current_rev = 0
        for inf in current_combo:
            if inf in curve_params:
                k, a, b = curve_params[inf]
                current_rev += response_curve(test_budget_per, k, a, b)
        
        if current_rev > best_rev:
            best_rev = current_rev
            best_n = n
            best_roas = current_rev / budget_input
            best_combo_ids = current_combo

    # Display Optimal
    opt_c1, opt_c2 = st.columns([1, 2])
    
    with opt_c1:
        st.success(f"**Optimal Count:** {best_n} Influencers")
        st.write(f"Allocation: **₹{budget_input/best_n:,.0f}** each")
        
    with opt_c2:
        om1, om2 = st.columns(2)
        om1.metric("Optimal Revenue", f"₹{best_rev:,.0f}", delta=f"₹{best_rev-user_rev:,.0f} vs Yours")
        om2.metric("Optimal ROAS", f"{best_roas:.2f}x", delta=f"{best_roas-user_roas:.2f}x")

    st.divider()
    
    # --- D. DIMINISHING RETURNS VISUALIZER ---
    st.write("#### 4. Diminishing Returns Analysis")
    st.caption("Drag the slider below to see how increasing budget eventually leads to waste.")
    
    # Interactive Slider for Graph
    graph_budget = st.slider("Simulate Budget Scale (₹)", 100000, 5000000, int(budget_input), step=100000)
    
    # Generate Plot Data
    x_range = np.linspace(0, 5000000, 50)
    
    # User Strategy Curve (Fixed N)
    y_user = []
    for x in x_range:
        rev = 0
        b_per = x / user_n
        for inf in top_n_user:
            if inf in curve_params:
                k, a, b = curve_params[inf]
                rev += response_curve(b_per, k, a, b)
        y_user.append(rev)
        
    # Optimal Strategy Curve (Fixed at Best N found earlier)
    y_opt = []
    for x in x_range:
        rev = 0
        b_per = x / best_n
        for inf in best_combo_ids:
            if inf in curve_params:
                k, a, b = curve_params[inf]
                rev += response_curve(b_per, k, a, b)
        y_opt.append(rev)
        
    chart_data = pd.DataFrame({
        'Budget': np.tile(x_range, 2),
        'Revenue': np.concatenate([y_user, y_opt]),
        'Strategy': [f'User Selection ({user_n} Inf)'] * 50 + [f'AI Optimal ({best_n} Inf)'] * 50
    })
    
    base_chart = alt.Chart(chart_data).mark_line().encode(
        x=alt.X('Budget', axis=alt.Axis(format='₹~s')),
        y=alt.Y('Revenue', axis=alt.Axis(format='₹~s')),
        color='Strategy',
        tooltip=['Budget', 'Revenue', 'Strategy']
    ).properties(height=400)
    
    # Add vertical line for current selected budget
    rule = alt.Chart(pd.DataFrame({'x': [graph_budget]})).mark_rule(color='red', strokeDash=[5,5]).encode(x='x')
    
    st.altair_chart(base_chart + rule, use_container_width=True)
    
    st.info(f"""
    **Graph Interpretation:**
    - The **Red Dashed Line** is your currently selected simulation budget (₹{graph_budget:,.0f}).
    - Notice how the curve flattens? That is **Saturation**.
    - If the 'AI Optimal' line is higher than 'User Selection', you are leaving money on the table.
    """)
