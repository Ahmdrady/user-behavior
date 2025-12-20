"""
Customer Behavior Analytics Dashboard
A comprehensive Streamlit dashboard for analyzing customer purchase patterns and behavior.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Page configuration
st.set_page_config(
    page_title="Customer Behavior Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #3B82F6;
    }
    .stat-annotation {
        font-size: 0.85rem;
        color: #6B7280;
        font-style: italic;
    }
    .insight-box {
        background: #f8f9fa;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    .warning-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Helper Functions from Elite Dashboard
# -----------------------------------------------------------------------------

@st.cache_data
def calculate_derived_metrics(df):
    """
    Calculate business metrics and customer segmentation.

    Derived Columns:
    - CLV_Score: Purchase Amount × Previous Purchases
    - Loyalty_Tier: New, Regular, Loyal, VIP (by prev purchases)
    - Age_Group: 18-25, 26-35, 36-50, 51-70
    - Value_Segment: Low/Mid/High Spender (purchase amount tertiles)
    """
    df = df.copy()

    # CLV Score (proxy for customer lifetime value)
    df['CLV_Score'] = df['Purchase Amount (USD)'] * df['Previous Purchases']

    # Loyalty Tiers based on Previous Purchases
    def assign_loyalty_tier(prev_purchases):
        if prev_purchases <= 10:
            return 'New'
        elif prev_purchases <= 25:
            return 'Regular'
        elif prev_purchases <= 40:
            return 'Loyal'
        else:
            return 'VIP'

    df['Loyalty_Tier'] = df['Previous Purchases'].apply(assign_loyalty_tier)

    # Age Groups
    def assign_age_group(age):
        if age <= 25:
            return '18-25'
        elif age <= 35:
            return '26-35'
        elif age <= 50:
            return '36-50'
        else:
            return '51-70'

    df['Age_Group'] = df['Age'].apply(assign_age_group)

    # Value Segments (tertiles of purchase amount)
    try:
        df['Value_Segment'] = pd.qcut(df['Purchase Amount (USD)'],
                                        q=3,
                                        labels=['Low Spender', 'Mid Spender', 'High Spender'])
    except Exception:
         df['Value_Segment'] = 'Mid Spender'

    # Boolean conversions
    df['Is_Subscriber'] = df['Subscription Status'] == 'Yes'
    df['Used_Discount'] = df['Discount Applied'] == 'Yes'
    df['Used_Promo'] = df['Promo Code Used'] == 'Yes'
    df['High_Satisfaction'] = df['Review Rating'] >= 4.0

    return df

def perform_ab_test(group_a, group_b, metric_name='Purchase Amount'):
    """
    Perform statistical A/B test using Mann-Whitney U test.

    Returns:
        Dictionary with test results including p-value, effect size, and conclusion
    """
    # Mann-Whitney U test (non-parametric)
    statistic, p_value = stats.mannwhitneyu(group_a, group_b, alternative='two-sided')

    # Effect size (Cohen's d)
    mean_diff = group_a.mean() - group_b.mean()
    pooled_std = np.sqrt((group_a.std()**2 + group_b.std()**2) / 2)
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

    # Confidence intervals (95%)
    try:
        ci_a = stats.t.interval(0.95, len(group_a)-1, loc=group_a.mean(), scale=stats.sem(group_a))
        ci_b = stats.t.interval(0.95, len(group_b)-1, loc=group_b.mean(), scale=stats.sem(group_b))
    except:
        ci_a = (group_a.mean(), group_a.mean())
        ci_b = (group_b.mean(), group_b.mean())

    # Conclusion
    conclusion = "Statistically Significant" if p_value < 0.05 else "Not Significant"

    return {
        'p_value': p_value,
        'effect_size': cohens_d,
        'mean_a': group_a.mean(),
        'mean_b': group_b.mean(),
        'ci_a': ci_a,
        'ci_b': ci_b,
        'n_a': len(group_a),
        'n_b': len(group_b),
        'conclusion': conclusion
    }

def perform_customer_clustering(df, n_clusters=5):
    """
    Perform K-Means clustering on customer features.

    Returns:
        DataFrame with cluster labels and PCA visualization figure
    """
    # Select features for clustering
    features = ['Age', 'Purchase Amount (USD)', 'Previous Purchases', 'Review Rating']
    X = df[features].copy()

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    # PCA for visualization
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    df['PCA1'] = X_pca[:, 0]
    df['PCA2'] = X_pca[:, 1]

    # Calculate cluster statistics
    cluster_stats = df.groupby('Cluster').agg({
        'Purchase Amount (USD)': 'mean',
        'Previous Purchases': 'mean',
        'Age': 'mean',
        'Review Rating': 'mean',
        'Customer ID': 'count'
    }).round(2)
    cluster_stats.columns = ['Avg Purchase', 'Avg Prev Purchases', 'Avg Age', 'Avg Rating', 'Count']

    return df, cluster_stats, pca.explained_variance_ratio_

# Data loading function with caching
@st.cache_data
def load_data():
    """Load and preprocess customer behavior data."""
    data_path = Path(__file__).parent / "user_behavior_data.csv"
    df = pd.read_csv(data_path)

    # Data preprocessing
    df.columns = df.columns.str.strip()

    # Apply derived metrics (Elite Dashboard Integration)
    df = calculate_derived_metrics(df)

    return df

# Load data
try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Sidebar filters
st.sidebar.header("🎛️ Filters")

# Loyalty tier filter
loyalty_tiers = st.sidebar.multiselect(
    "Customer Loyalty Tier",
    options=['New', 'Regular', 'Loyal', 'VIP'],
    default=['New', 'Regular', 'Loyal', 'VIP'],
    help="Filter by customer purchase history"
)

# Value segment filter
value_segments = st.sidebar.multiselect(
    "Value Segment",
    options=['Low Spender', 'Mid Spender', 'High Spender'],
    default=['Low Spender', 'Mid Spender', 'High Spender']
)

# Subscription filter
subscription_filter = st.sidebar.radio(
    "Subscription Status",
    options=['All', 'Subscribers Only', 'Non-Subscribers Only'],
    index=0
)

# Gender filter
gender_options = ['All'] + sorted(df['Gender'].unique().tolist())
selected_gender = st.sidebar.selectbox("Gender", gender_options, index=0)

# Category filter
category_options = ['All'] + sorted(df['Category'].unique().tolist())
selected_category = st.sidebar.multiselect(
    "Category",
    category_options,
    default=['All']
)

# Season filter
season_options = ['All'] + sorted(df['Season'].unique().tolist())
selected_season = st.sidebar.multiselect(
    "Season",
    season_options,
    default=['All']
)

# Age range filter
min_age, max_age = int(df['Age'].min()), int(df['Age'].max())
age_range = st.sidebar.slider(
    "Age Range",
    min_value=min_age,
    max_value=max_age,
    value=(min_age, max_age)
)

# Purchase amount filter
min_purchase = float(df['Purchase Amount (USD)'].min())
max_purchase = float(df['Purchase Amount (USD)'].max())
purchase_range = st.sidebar.slider(
    "Purchase Amount (USD)",
    min_value=min_purchase,
    max_value=max_purchase,
    value=(min_purchase, max_purchase)
)

# Apply filters
filtered_df = df.copy()

if loyalty_tiers:
    filtered_df = filtered_df[filtered_df['Loyalty_Tier'].isin(loyalty_tiers)]

if value_segments:
    filtered_df = filtered_df[filtered_df['Value_Segment'].isin(value_segments)]

if subscription_filter == 'Subscribers Only':
    filtered_df = filtered_df[filtered_df['Is_Subscriber']]
elif subscription_filter == 'Non-Subscribers Only':
    filtered_df = filtered_df[~filtered_df['Is_Subscriber']]

if selected_gender != 'All':
    filtered_df = filtered_df[filtered_df['Gender'] == selected_gender]

if 'All' not in selected_category:
    filtered_df = filtered_df[filtered_df['Category'].isin(selected_category)]

if 'All' not in selected_season:
    filtered_df = filtered_df[filtered_df['Season'].isin(selected_season)]

filtered_df = filtered_df[
    (filtered_df['Age'] >= age_range[0]) &
    (filtered_df['Age'] <= age_range[1])
]

filtered_df = filtered_df[
    (filtered_df['Purchase Amount (USD)'] >= purchase_range[0]) &
    (filtered_df['Purchase Amount (USD)'] <= purchase_range[1])
]

# Main header
st.markdown('<h1 class="main-header">📊 Customer Behavior Analytics Dashboard</h1>', unsafe_allow_html=True)

st.info(f"📈 Showing **{len(filtered_df):,}** of **{len(df):,}** customer records")

# Key Performance Indicators
st.subheader("📈 Key Metrics")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    total_customers = len(filtered_df)
    st.metric(
        label="Total Customers",
        value=f"{total_customers:,}",
        delta=f"{(total_customers/len(df)*100):.1f}% of total"
    )

with col2:
    avg_purchase = filtered_df['Purchase Amount (USD)'].mean()
    st.metric(
        label="Avg Purchase",
        value=f"${avg_purchase:.2f}"
    )

with col3:
    total_revenue = filtered_df['Purchase Amount (USD)'].sum()
    st.metric(
        label="Total Revenue",
        value=f"${total_revenue:,.2f}"
    )

with col4:
    avg_rating = filtered_df['Review Rating'].mean()
    st.metric(
        label="Avg Rating",
        value=f"{avg_rating:.2f} ⭐"
    )

with col5:
    avg_prev_purchases = filtered_df['Previous Purchases'].mean()
    st.metric(
        label="Avg Previous Purchases",
        value=f"{avg_prev_purchases:.1f}"
    )

# Create tabs for different analysis views
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "📊 Overview",
    "🛍️ Purchase Analysis",
    "👥 Customer Demographics",
    "💰 Revenue Analysis",
    "⭐ Review Analytics",
    "📦 Shipping & Payment",
    "👥 Customer Segmentation",
    "🎁 Promotional Analysis",
    "🔮 Predictive Analytics",
    "📊 Statistical Tests"
])

with tab1:
    st.subheader("Overview Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        # Category distribution
        st.markdown("#### Purchase by Category")
        category_dist = filtered_df['Category'].value_counts().reset_index()
        category_dist.columns = ['Category', 'Count']

        fig_category = px.pie(
            category_dist,
            values='Count',
            names='Category',
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_category.update_layout(height=400)
        st.plotly_chart(fig_category, use_container_width=True)

    with col2:
        # Season distribution
        st.markdown("#### Purchase by Season")
        season_dist = filtered_df['Season'].value_counts().reset_index()
        season_dist.columns = ['Season', 'Count']

        fig_season = px.bar(
            season_dist,
            x='Season',
            y='Count',
            color='Season',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_season.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_season, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        # Top 10 items purchased
        st.markdown("#### Top 10 Items Purchased")
        top_items = filtered_df['Item Purchased'].value_counts().head(10).reset_index()
        top_items.columns = ['Item', 'Count']

        fig_items = px.bar(
            top_items,
            x='Count',
            y='Item',
            orientation='h',
            color='Count',
            color_continuous_scale='Blues'
        )
        fig_items.update_layout(height=400)
        st.plotly_chart(fig_items, use_container_width=True)

    with col4:
        # Top 10 locations
        st.markdown("#### Top 10 Customer Locations")
        top_locations = filtered_df['Location'].value_counts().head(10).reset_index()
        top_locations.columns = ['Location', 'Count']

        fig_locations = px.bar(
            top_locations,
            x='Count',
            y='Location',
            orientation='h',
            color='Count',
            color_continuous_scale='Greens'
        )
        fig_locations.update_layout(height=400)
        st.plotly_chart(fig_locations, use_container_width=True)

with tab2:
    st.subheader("Purchase Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Purchase amount distribution
        st.markdown("#### Purchase Amount Distribution")
        fig_dist = px.histogram(
            filtered_df,
            x='Purchase Amount (USD)',
            nbins=50,
            marginal='box',
            color_discrete_sequence=['#3B82F6']
        )
        fig_dist.update_layout(height=400)
        st.plotly_chart(fig_dist, use_container_width=True)

        # Average purchase by category
        st.markdown("#### Average Purchase by Category")
        avg_by_category = filtered_df.groupby('Category')['Purchase Amount (USD)'].mean().reset_index()
        avg_by_category.columns = ['Category', 'Average Purchase']

        fig_avg_cat = px.bar(
            avg_by_category,
            x='Category',
            y='Average Purchase',
            color='Category',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_avg_cat.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_avg_cat, use_container_width=True)

    with col2:
        # Purchase by season and category
        st.markdown("#### Purchase Amount by Season & Category")
        season_category = filtered_df.groupby(['Season', 'Category'])['Purchase Amount (USD)'].sum().reset_index()

        fig_season_cat = px.sunburst(
            season_category,
            path=['Season', 'Category'],
            values='Purchase Amount (USD)',
            color='Purchase Amount (USD)',
            color_continuous_scale='RdYlGn'
        )
        fig_season_cat.update_layout(height=500)
        st.plotly_chart(fig_season_cat, use_container_width=True)

        # Frequency of purchases
        st.markdown("#### Purchase Frequency Distribution")
        freq_dist = filtered_df['Frequency of Purchases'].value_counts().reset_index()
        freq_dist.columns = ['Frequency', 'Count']

        fig_freq = px.pie(
            freq_dist,
            values='Count',
            names='Frequency',
            hole=0.3
        )
        fig_freq.update_layout(height=350)
        st.plotly_chart(fig_freq, use_container_width=True)

with tab3:
    st.subheader("Customer Demographics")

    col1, col2 = st.columns(2)

    with col1:
        # Age distribution
        st.markdown("#### Age Distribution")
        fig_age = px.histogram(
            filtered_df,
            x='Age',
            nbins=30,
            marginal='box',
            color_discrete_sequence=['#10B981']
        )
        fig_age.update_layout(height=400)
        st.plotly_chart(fig_age, use_container_width=True)

        # Gender distribution
        st.markdown("#### Gender Distribution")
        gender_dist = filtered_df['Gender'].value_counts().reset_index()
        gender_dist.columns = ['Gender', 'Count']

        fig_gender = px.pie(
            gender_dist,
            values='Count',
            names='Gender',
            hole=0.4,
            color_discrete_sequence=['#3B82F6', '#EC4899']
        )
        fig_gender.update_layout(height=350)
        st.plotly_chart(fig_gender, use_container_width=True)

    with col2:
        # Age vs Purchase Amount
        st.markdown("#### Age vs Purchase Amount")
        fig_age_purchase = px.scatter(
            filtered_df,
            x='Age',
            y='Purchase Amount (USD)',
            color='Category',
            size='Review Rating',
            hover_data=['Item Purchased', 'Location'],
            opacity=0.6
        )
        fig_age_purchase.update_layout(height=400)
        st.plotly_chart(fig_age_purchase, use_container_width=True)

        # Previous purchases distribution
        st.markdown("#### Previous Purchases Distribution")
        fig_prev = px.histogram(
            filtered_df,
            x='Previous Purchases',
            nbins=20,
            color_discrete_sequence=['#F59E0B']
        )
        fig_prev.update_layout(height=350)
        st.plotly_chart(fig_prev, use_container_width=True)

with tab4:
    st.subheader("Revenue Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Revenue by category
        st.markdown("#### Revenue by Category")
        revenue_by_cat = filtered_df.groupby('Category')['Purchase Amount (USD)'].sum().reset_index()
        revenue_by_cat.columns = ['Category', 'Revenue']
        revenue_by_cat = revenue_by_cat.sort_values('Revenue', ascending=True)

        fig_rev_cat = px.bar(
            revenue_by_cat,
            x='Revenue',
            y='Category',
            orientation='h',
            color='Revenue',
            color_continuous_scale='Viridis',
            text='Revenue'
        )
        fig_rev_cat.update_traces(texttemplate='$%{text:.2f}', textposition='outside')
        fig_rev_cat.update_layout(height=400)
        st.plotly_chart(fig_rev_cat, use_container_width=True)

        # Revenue by season
        st.markdown("#### Revenue by Season")
        revenue_by_season = filtered_df.groupby('Season')['Purchase Amount (USD)'].sum().reset_index()
        revenue_by_season.columns = ['Season', 'Revenue']

        fig_rev_season = px.bar(
            revenue_by_season,
            x='Season',
            y='Revenue',
            color='Season',
            text='Revenue'
        )
        fig_rev_season.update_traces(texttemplate='$%{text:.2f}', textposition='outside')
        fig_rev_season.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_rev_season, use_container_width=True)

    with col2:
        # Revenue by location (top 15)
        st.markdown("#### Revenue by Top 15 Locations")
        revenue_by_loc = filtered_df.groupby('Location')['Purchase Amount (USD)'].sum().reset_index()
        revenue_by_loc.columns = ['Location', 'Revenue']
        revenue_by_loc = revenue_by_loc.sort_values('Revenue', ascending=False).head(15)

        fig_rev_loc = px.bar(
            revenue_by_loc,
            x='Revenue',
            y='Location',
            orientation='h',
            color='Revenue',
            color_continuous_scale='Oranges'
        )
        fig_rev_loc.update_layout(height=500)
        st.plotly_chart(fig_rev_loc, use_container_width=True)

        # Revenue metrics
        st.markdown("#### Revenue Summary")
        col_a, col_b = st.columns(2)

        with col_a:
            st.metric("Total Revenue", f"${total_revenue:,.2f}")
            st.metric("Median Purchase", f"${filtered_df['Purchase Amount (USD)'].median():.2f}")

        with col_b:
            st.metric("Max Purchase", f"${filtered_df['Purchase Amount (USD)'].max():.2f}")
            st.metric("Min Purchase", f"${filtered_df['Purchase Amount (USD)'].min():.2f}")

with tab5:
    st.subheader("Review Analytics")

    col1, col2 = st.columns(2)

    with col1:
        # Review rating distribution
        st.markdown("#### Review Rating Distribution")
        rating_dist = filtered_df['Review Rating'].value_counts().sort_index().reset_index()
        rating_dist.columns = ['Rating', 'Count']

        fig_rating = px.bar(
            rating_dist,
            x='Rating',
            y='Count',
            color='Rating',
            color_continuous_scale='RdYlGn',
            text='Count'
        )
        fig_rating.update_traces(textposition='outside')
        fig_rating.update_layout(height=400)
        st.plotly_chart(fig_rating, use_container_width=True)

        # Average rating by category
        st.markdown("#### Average Rating by Category")
        avg_rating_cat = filtered_df.groupby('Category')['Review Rating'].mean().reset_index()
        avg_rating_cat.columns = ['Category', 'Average Rating']

        fig_avg_rating_cat = px.bar(
            avg_rating_cat,
            x='Category',
            y='Average Rating',
            color='Average Rating',
            color_continuous_scale='Blues',
            text='Average Rating'
        )
        fig_avg_rating_cat.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig_avg_rating_cat.update_layout(height=350)
        st.plotly_chart(fig_avg_rating_cat, use_container_width=True)

    with col2:
        # Rating vs Purchase Amount
        st.markdown("#### Rating vs Purchase Amount")
        fig_rating_purchase = px.scatter(
            filtered_df,
            x='Review Rating',
            y='Purchase Amount (USD)',
            color='Category',
            opacity=0.6,
            trendline='ols'
        )
        fig_rating_purchase.update_layout(height=400)
        st.plotly_chart(fig_rating_purchase, use_container_width=True)

        # Average rating by season
        st.markdown("#### Average Rating by Season")
        avg_rating_season = filtered_df.groupby('Season')['Review Rating'].mean().reset_index()
        avg_rating_season.columns = ['Season', 'Average Rating']

        fig_avg_rating_season = px.line(
            avg_rating_season,
            x='Season',
            y='Average Rating',
            markers=True,
            color_discrete_sequence=['#EF4444']
        )
        fig_avg_rating_season.update_layout(height=350)
        st.plotly_chart(fig_avg_rating_season, use_container_width=True)

with tab6:
    st.subheader("Shipping & Payment Analytics")

    col1, col2 = st.columns(2)

    with col1:
        # Shipping type distribution
        st.markdown("#### Shipping Type Distribution")
        shipping_dist = filtered_df['Shipping Type'].value_counts().reset_index()
        shipping_dist.columns = ['Shipping Type', 'Count']

        fig_shipping = px.pie(
            shipping_dist,
            values='Count',
            names='Shipping Type',
            hole=0.4
        )
        fig_shipping.update_layout(height=400)
        st.plotly_chart(fig_shipping, use_container_width=True)

        # Payment method distribution
        st.markdown("#### Payment Method Distribution")
        payment_dist = filtered_df['Payment Method'].value_counts().reset_index()
        payment_dist.columns = ['Payment Method', 'Count']

        fig_payment = px.bar(
            payment_dist,
            x='Payment Method',
            y='Count',
            color='Payment Method',
            text='Count'
        )
        fig_payment.update_traces(textposition='outside')
        fig_payment.update_layout(height=350, showlegend=False)
        fig_payment.update_xaxes(tickangle=45)
        st.plotly_chart(fig_payment, use_container_width=True)

    with col2:
        # Discount and promo usage
        st.markdown("#### Discount & Promo Code Usage")

        discount_data = {
            'Type': ['Discount Applied', 'Promo Code Used'],
            'Count': [
                filtered_df['Discount Applied'].value_counts().get('Yes', 0),
                filtered_df['Promo Code Used'].value_counts().get('Yes', 0)
            ]
        }
        discount_df = pd.DataFrame(discount_data)

        fig_discount = px.bar(
            discount_df,
            x='Type',
            y='Count',
            color='Type',
            text='Count',
            color_discrete_sequence=['#8B5CF6', '#EC4899']
        )
        fig_discount.update_traces(textposition='outside')
        fig_discount.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_discount, use_container_width=True)

        # Size distribution
        st.markdown("#### Size Distribution")
        size_dist = filtered_df['Size'].value_counts().reset_index()
        size_dist.columns = ['Size', 'Count']

        # Order sizes properly
        size_order = ['S', 'M', 'L', 'XL']
        size_dist['Size'] = pd.Categorical(size_dist['Size'], categories=size_order, ordered=True)
        size_dist = size_dist.sort_values('Size')

        fig_size = px.bar(
            size_dist,
            x='Size',
            y='Count',
            color='Size',
            text='Count',
            color_discrete_sequence=px.colors.sequential.Teal
        )
        fig_size.update_traces(textposition='outside')
        fig_size.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_size, use_container_width=True)

with tab7:
    st.subheader("Customer Value Matrix")

    with st.expander("📚 Business Methodology"):
        st.markdown("""
        **RFM-like Segmentation:**
        - **Frequency (F):** Previous Purchases (historical behavior)
        - **Monetary (M):** Purchase Amount (current transaction value)
        - **Value Score:** F × M (CLV proxy)

        **Customer Quadrants:**
        - **Champions** (High F, High M): Retain at all costs - VIP programs, exclusive perks
        - **High Potential** (Low F, High M): Upsell opportunities - onboarding campaigns
        - **Loyal Deal-Seekers** (High F, Low M): Margin optimization needed
        - **At Risk** (Low F, Low M): Win-back campaigns, satisfaction surveys
        """)

    # Customer Value Matrix Scatter Plot
    median_purchase = filtered_df['Purchase Amount (USD)'].median()
    median_frequency = filtered_df['Previous Purchases'].median()

    # Assign quadrants
    def assign_quadrant(row):
        if row['Purchase Amount (USD)'] >= median_purchase and row['Previous Purchases'] >= median_frequency:
            return 'Champions'
        elif row['Purchase Amount (USD)'] >= median_purchase and row['Previous Purchases'] < median_frequency:
            return 'High Potential'
        elif row['Purchase Amount (USD)'] < median_purchase and row['Previous Purchases'] >= median_frequency:
            return 'Loyal Deal-Seekers'
        else:
            return 'At Risk'

    filtered_df['Customer_Segment'] = filtered_df.apply(assign_quadrant, axis=1)

    fig_matrix = px.scatter(
        filtered_df,
        x='Previous Purchases',
        y='Purchase Amount (USD)',
        color='Customer_Segment',
        size='Review Rating',
        hover_data=['Customer ID', 'Category', 'CLV_Score', 'Age'],
        color_discrete_map={
            'Champions': '#28a745',
            'High Potential': '#ffc107',
            'Loyal Deal-Seekers': '#17a2b8',
            'At Risk': '#dc3545'
        },
        title='Customer Value Matrix: Purchase Amount vs Previous Purchases'
    )

    # Add quadrant lines
    fig_matrix.add_hline(y=median_purchase, line_dash="dash", line_color="gray",
                         annotation_text="Median Purchase")
    fig_matrix.add_vline(x=median_frequency, line_dash="dash", line_color="gray",
                         annotation_text="Median Frequency")

    fig_matrix.update_layout(height=500)
    st.plotly_chart(fig_matrix, use_container_width=True)

    # Segment distribution
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Segment Distribution & Metrics")
        segment_stats = filtered_df.groupby('Customer_Segment').agg({
            'Customer ID': 'count',
            'Purchase Amount (USD)': 'mean',
            'Previous Purchases': 'mean',
            'CLV_Score': 'mean',
            'Review Rating': 'mean'
        }).round(2)
        segment_stats.columns = ['Count', 'Avg Purchase', 'Avg Frequency', 'Avg CLV', 'Avg Rating']
        segment_stats['% of Total'] = (segment_stats['Count'] / segment_stats['Count'].sum() * 100).round(1)
        st.dataframe(segment_stats.style.background_gradient(cmap='RdYlGn', subset=['Avg CLV']))

    with col2:
        st.markdown("#### Loyalty Tier Distribution")
        loyalty_dist = filtered_df['Loyalty_Tier'].value_counts().reset_index()
        loyalty_dist.columns = ['Tier', 'Count']

        # Define tier order
        tier_order = ['New', 'Regular', 'Loyal', 'VIP']
        loyalty_dist['Tier'] = pd.Categorical(loyalty_dist['Tier'], categories=tier_order, ordered=True)
        loyalty_dist = loyalty_dist.sort_values('Tier')

        fig_loyalty = px.funnel(
            loyalty_dist,
            x='Count',
            y='Tier',
            color='Tier',
            color_discrete_sequence=px.colors.sequential.Blues_r
        )
        fig_loyalty.update_layout(height=300)
        st.plotly_chart(fig_loyalty, use_container_width=True)

with tab8:
    st.subheader("Promotional Effectiveness Analysis")

    # Key finding box
    st.markdown("""
    <div class="warning-box">
    <strong>⚠️ Critical Finding:</strong> Discounts <strong>decrease</strong> average order value while increasing transaction volume.
    <br><br>
    <strong>💡 Business Recommendation:</strong> Test tiered promotions (e.g., "Spend $75, get 10% off")
    to boost basket size while maintaining discount appeal and protecting margins.
    </div>
    """, unsafe_allow_html=True)

    # A/B Test: Discount vs No Discount
    discount_group = filtered_df[filtered_df['Used_Discount']]['Purchase Amount (USD)']
    no_discount_group = filtered_df[~filtered_df['Used_Discount']]['Purchase Amount (USD)']

    if len(discount_group) > 0 and len(no_discount_group) > 0:
        ab_results = perform_ab_test(discount_group, no_discount_group, 'Purchase Amount')

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Discount Group AOV", f"${ab_results['mean_a']:.2f}",
                     delta=f"n={ab_results['n_a']}")

        with col2:
            st.metric("No Discount AOV", f"${ab_results['mean_b']:.2f}",
                     delta=f"n={ab_results['n_b']}")

        with col3:
            st.metric("P-Value", f"{ab_results['p_value']:.4f}",
                     delta=ab_results['conclusion'])

        with col4:
            st.metric("Effect Size (Cohen's d)", f"{ab_results['effect_size']:.3f}")

        # Visualization
        col1, col2 = st.columns(2)

        with col1:
            # Box plot comparison
            promo_data = pd.DataFrame({
                'Purchase Amount': list(discount_group) + list(no_discount_group),
                'Group': ['With Discount']*len(discount_group) + ['No Discount']*len(no_discount_group)
            })

            fig_promo = px.box(
                promo_data,
                x='Group',
                y='Purchase Amount',
                color='Group',
                title='Purchase Amount Distribution: Discount vs No Discount',
                color_discrete_map={'With Discount': '#ffc107', 'No Discount': '#28a745'}
            )
            fig_promo.update_layout(height=400)
            st.plotly_chart(fig_promo, use_container_width=True)

        with col2:
            # Statistical summary
            st.markdown("#### Statistical Test Results")
            st.markdown(f"""
            **Test:** Mann-Whitney U (non-parametric)
            **Null Hypothesis:** No difference in purchase amounts
            **Alternative:** Two-sided test

            **Results:**
            - P-value: `{ab_results['p_value']:.4f}`
            - Effect Size: `{ab_results['effect_size']:.3f}`
            - Conclusion: **{ab_results['conclusion']}**

            **Confidence Intervals (95%):**
            - With Discount: ${ab_results['ci_a'][0]:.2f} - ${ab_results['ci_a'][1]:.2f}
            - No Discount: ${ab_results['ci_b'][0]:.2f} - ${ab_results['ci_b'][1]:.2f}

            **Interpretation:**
            The {'small' if abs(ab_results['effect_size']) < 0.2 else 'moderate' if abs(ab_results['effect_size']) < 0.5 else 'large'}
            effect size suggests discounts have a
            {'minimal' if abs(ab_results['effect_size']) < 0.2 else 'moderate' if abs(ab_results['effect_size']) < 0.5 else 'substantial'}
            impact on purchase behavior.
            """)

with tab9:
    st.subheader("Predictive Analytics & Customer Clustering")

    st.markdown("#### K-Means Customer Segmentation")

    with st.expander("📚 Methodology"):
        st.markdown("""
        **Unsupervised Learning:** K-Means clustering discovers hidden customer segments

        **Features Used:**
        - Age (demographic)
        - Purchase Amount (monetary value)
        - Previous Purchases (frequency/loyalty)
        - Review Rating (satisfaction)

        **Process:**
        1. Feature standardization (zero mean, unit variance)
        2. K-Means clustering with optimal K
        3. PCA reduction to 2D for visualization
        4. Cluster profiling and interpretation
        """)

    n_clusters = st.slider("Number of Clusters", min_value=3, max_value=8, value=5)

    if st.button("🔍 Discover Customer Segments", type="primary"):
        with st.spinner("Running K-Means clustering..."):
            clustered_df, cluster_stats, variance_explained = perform_customer_clustering(
                filtered_df,
                n_clusters=n_clusters
            )

            col1, col2 = st.columns([2, 1])

            with col1:
                # PCA visualization
                fig_cluster = px.scatter(
                    clustered_df,
                    x='PCA1',
                    y='PCA2',
                    color='Cluster',
                    hover_data=['Customer ID', 'Age', 'Purchase Amount (USD)',
                               'Previous Purchases', 'Review Rating'],
                    title=f'Customer Clusters (PCA Visualization)',
                    labels={'PCA1': f'PC1 ({variance_explained[0]*100:.1f}% variance)',
                           'PCA2': f'PC2 ({variance_explained[1]*100:.1f}% variance)'}
                )
                fig_cluster.update_layout(height=500)
                st.plotly_chart(fig_cluster, use_container_width=True)

            with col2:
                st.markdown("#### Cluster Profiles")
                st.dataframe(
                    cluster_stats.style.background_gradient(cmap='YlOrRd', subset=['Avg Purchase']),
                    height=500
                )

            # Insights
            st.markdown("#### Business Insights")
            best_cluster = cluster_stats['Avg Purchase'].idxmax()
            worst_cluster = cluster_stats['Avg Purchase'].idxmin()

            st.markdown(f"""
            <div class="insight-box">
            <strong>🎯 High-Value Cluster:</strong> Cluster {best_cluster} shows the highest average purchase
            of ${cluster_stats.loc[best_cluster, 'Avg Purchase']:.2f} with
            {cluster_stats.loc[best_cluster, 'Count']} customers.
            <br><br>
            <strong>💡 Recommendation:</strong> Focus retention efforts on Cluster {best_cluster}.
            Consider Cluster {worst_cluster} for upselling campaigns.
            </div>
            """, unsafe_allow_html=True)

with tab10:
    st.subheader("Statistical Hypothesis Testing")

    st.markdown("""
    Test business hypotheses with statistical rigor. All tests report p-values, effect sizes,
    and confidence intervals following academic standards.
    """)

    col1, col2 = st.columns(2)

    with col1:
        metric_to_test = st.selectbox(
            "Metric to Test",
            ['Purchase Amount (USD)', 'Review Rating', 'Previous Purchases', 'CLV_Score']
        )

    with col2:
        test_variable = st.selectbox(
            "Group Variable",
            ['Subscription Status', 'Discount Applied', 'Gender', 'Season', 'Loyalty_Tier']
        )

    if st.button("⚡ Run Statistical Test", type="primary"):
        st.markdown(f"### Testing: {metric_to_test} by {test_variable}")

        # Get unique groups
        if test_variable == 'Subscription Status':
            groups = filtered_df.groupby('Is_Subscriber')[metric_to_test]
            group_names = ['Non-Subscribers', 'Subscribers']
        elif test_variable == 'Discount Applied':
            groups = filtered_df.groupby('Used_Discount')[metric_to_test]
            group_names = ['No Discount', 'With Discount']
        else:
            groups = filtered_df.groupby(test_variable)[metric_to_test]
            group_names = groups.groups.keys()

        # For two groups, perform detailed A/B test
        if len(groups) == 2:
            group_data = [group for name, group in groups]
            results = perform_ab_test(group_data[0], group_data[1], metric_to_test)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Group A Mean", f"{results['mean_a']:.2f}",
                         delta=f"n={results['n_a']}")

            with col2:
                st.metric("Group B Mean", f"{results['mean_b']:.2f}",
                         delta=f"n={results['n_b']}")

            with col3:
                st.metric("P-Value", f"{results['p_value']:.4f}",
                         delta=results['conclusion'])

            # Detailed results
            st.markdown("#### Detailed Statistical Results")
            st.markdown(f"""
            **Test:** Mann-Whitney U Test (non-parametric)
            **Significance Level:** α = 0.05
            **Effect Size (Cohen's d):** {results['effect_size']:.3f}

            **Conclusion:** {results['conclusion']} (p = {results['p_value']:.4f})

            **Interpretation:**
            - If p < 0.05: Reject null hypothesis - groups differ significantly
            - If p ≥ 0.05: Fail to reject - no significant difference detected

            **Effect Size Interpretation:**
            - |d| < 0.2: Small effect
            - 0.2 ≤ |d| < 0.5: Medium effect
            - |d| ≥ 0.5: Large effect

            **Business Recommendation:**
            {'Strong evidence of difference - prioritize this segmentation' if results['p_value'] < 0.05 else 'No significant difference - consider other factors'}
            """)
        else:
            st.warning("For more than two groups, please select a variable with only two categories or filter the data first. (Kruskal-Wallis test not implemented yet)")

# Footer
st.markdown("---")
st.markdown("""
📊 **Customer Behavior Analytics Dashboard** | Built with Streamlit & Plotly
💡 Use the sidebar filters to explore different customer segments and analyze purchasing patterns.
""")
