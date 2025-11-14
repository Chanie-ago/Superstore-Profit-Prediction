import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Superstore Profit Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 30px;
    }
    .sub-header {
        font-size: 24px;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 20px;
        margin-bottom: 15px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model_and_encoders():
    try:
        with open('./models/profit_prediction_model.pkl', 'rb') as file:
            model = pickle.load(file)
        with open('./models/label_encoders.pkl', 'rb') as file:
            encoders = pickle.load(file)
        with open('./models/feature_names.pkl', 'rb') as file:
            features = pickle.load(file)
        # metrics optional
        try:
            with open('./models/model_metrics.pkl', 'rb') as file:
                metrics = pickle.load(file)
        except FileNotFoundError:
            metrics = None
        return model, encoders, features, metrics
    except Exception as e:
        st.error(f"⚠️ Error loading models: {e}")
        st.info("Make sure all model files are in the repository.")
        return None, None, None, None

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('./data/raw/Superstore.csv', encoding='latin-1')
        # Try multiple date formats
        try:
            df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d-%m-%Y')
            df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%d-%m-%Y')
        except:
            df['Order Date'] = pd.to_datetime(df['Order Date'])
            df['Ship Date'] = pd.to_datetime(df['Ship Date'])
        return df
    except Exception as e:
        st.error(f"⚠️ Error loading dataset: {e}")
        return None

def main():
    # Header
    st.markdown('<p class="main-header">📊 Superstore Profit Prediction Dashboard</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data and models
    df = load_data()
    model, encoders, feature_names, metrics = load_model_and_encoders()
    
    if df is None or model is None:
        st.stop()
    
    # Sidebar
    st.sidebar.title("🎯 Navigation")
    page = st.sidebar.radio("Choose a page:", 
                            ["📈 Dashboard Overview", 
                             "🔮 Profit Prediction", 
                             "📊 Data Explorer"])
    
    # Page 1: Dashboard Overview
    if page == "📈 Dashboard Overview":
        st.markdown('<p class="sub-header">Business Overview</p>', unsafe_allow_html=True)
        
        # Model info
        if metrics:
            # Get metrics
            model_name = metrics.get('best_model', 'CatBoost Original')
            
            # Get best model metrics
            if 'catboost_original' in metrics and model_name == 'CatBoost Original':
                best_metrics = metrics['catboost_original']
            elif 'best_model_metrics' in metrics:
                best_metrics = metrics['best_model_metrics']
            else:
                # Fallback
                for key in metrics:
                    if isinstance(metrics[key], dict) and 'test_r2' in metrics[key]:
                        best_metrics = metrics[key]
                        break
                else:
                    best_metrics = None
            
            if best_metrics:
                st.info(f"🤖 **Model**: {model_name} | **R² Score**: {best_metrics['test_r2']:.2%} | **RMSE**: ${best_metrics['test_rmse']:.2f} | **MAE**: ${best_metrics['test_mae']:.2f}")
        else:
            st.info("🤖 **Model**: CatBoost Original - Profit prediction model loaded successfully")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_sales = df['Sales'].sum()
            st.metric("💰 Total Sales", f"${total_sales:,.2f}")
        
        with col2:
            total_profit = df['Profit'].sum()
            st.metric("💵 Total Profit", f"${total_profit:,.2f}")
        
        with col3:
            total_orders = df['Order ID'].nunique()
            st.metric("📦 Total Orders", f"{total_orders:,}")
        
        with col4:
            avg_profit = df['Profit'].mean()
            st.metric("📊 Avg Profit/Order", f"${avg_profit:,.2f}")
        
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Sales by Category
            category_sales = df.groupby('Category')['Sales'].sum().reset_index()
            fig1 = px.bar(category_sales, x='Category', y='Sales', 
                         title='Total Sales by Category',
                         color='Sales',
                         color_continuous_scale='Blues')
            fig1.update_layout(showlegend=False)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Profit by Region
            region_profit = df.groupby('Region')['Profit'].sum().reset_index()
            fig2 = px.pie(region_profit, values='Profit', names='Region',
                         title='Profit Distribution by Region',
                         color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig2, use_container_width=True)
        
        # Trend
        st.markdown('<p class="sub-header">Sales Trend Over Time</p>', unsafe_allow_html=True)
        monthly_sales = df.groupby(df['Order Date'].dt.to_period('M')).agg({
            'Sales': 'sum',
            'Profit': 'sum'
        }).reset_index()
        monthly_sales['Order Date'] = monthly_sales['Order Date'].dt.to_timestamp()
        
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=monthly_sales['Order Date'], y=monthly_sales['Sales'],
                                 mode='lines+markers', name='Sales',
                                 line=dict(color='blue', width=2)))
        fig3.add_trace(go.Scatter(x=monthly_sales['Order Date'], y=monthly_sales['Profit'],
                                 mode='lines+markers', name='Profit',
                                 line=dict(color='green', width=2)))
        fig3.update_layout(title='Monthly Sales and Profit Trend',
                          xaxis_title='Date',
                          yaxis_title='Amount ($)',
                          hovermode='x unified')
        st.plotly_chart(fig3, use_container_width=True)
        
        # Top 10
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<p class="sub-header">Top 10 Products by Sales</p>', unsafe_allow_html=True)
            top_products = df.groupby('Product Name')['Sales'].sum().sort_values(ascending=False).head(10)
            st.dataframe(top_products.reset_index().rename(columns={'Product Name': 'Product', 'Sales': 'Total Sales ($)'}),
                        hide_index=True, use_container_width=True)
        
        with col2:
            st.markdown('<p class="sub-header">Top 10 Customers by Profit</p>', unsafe_allow_html=True)
            top_customers = df.groupby('Customer Name')['Profit'].sum().sort_values(ascending=False).head(10)
            st.dataframe(top_customers.reset_index().rename(columns={'Customer Name': 'Customer', 'Profit': 'Total Profit ($)'}),
                        hide_index=True, use_container_width=True)
    
    # Page 2: Profit Prediction
    elif page == "🔮 Profit Prediction":
        st.markdown('<p class="sub-header">Predict Profit for New Order</p>', unsafe_allow_html=True)
        st.write("Enter order details to predict the expected profit.")
        
        # Model info
        if metrics:
            model_name = metrics.get('best_model', 'CatBoost Original')
            
            # Get best model metrics
            if 'catboost_original' in metrics and model_name == 'CatBoost Original':
                best_metrics = metrics['catboost_original']
            elif 'best_model_metrics' in metrics:
                best_metrics = metrics['best_model_metrics']
            else:
                best_metrics = None
            
            if best_metrics:
                with st.expander("ℹ️ Model Information"):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Model", model_name)
                    with col2:
                        st.metric("R² Score", f"{best_metrics['test_r2']:.2%}")
                    with col3:
                        st.metric("RMSE", f"${best_metrics['test_rmse']:.2f}")
                    with col4:
                        st.metric("MAE", f"${best_metrics['test_mae']:.2f}")
                    
                    st.write(f"**Training R²**: {best_metrics['train_r2']:.2%} | **Overfitting**: {best_metrics['overfitting']:.2%}")
                    
                    if best_metrics['overfitting'] < 0.10:  # Less than 10%
                        st.success("✅ Model has excellent generalization with minimal overfitting!")
                    else:
                        st.warning("⚠️ Model shows some overfitting. Predictions may vary on new data.")
        else:
            with st.expander("ℹ️ Model Information"):
                st.info("Model: CatBoost Original - High-performance gradient boosting model for profit prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📦 Product Information**")
            category = st.selectbox("Category", df['Category'].unique())
            # Filter sub-categories based on selected category
            sub_categories = df[df['Category'] == category]['Sub-Category'].unique()
            sub_category = st.selectbox("Sub-Category", sub_categories)
            
            st.markdown("**👥 Customer Information**")
            segment = st.selectbox("Customer Segment", df['Segment'].unique())
            region = st.selectbox("Region", df['Region'].unique())
        
        with col2:
            st.markdown("**💰 Transaction Details**")
            sales = st.number_input("Sales Amount ($)", min_value=0.0, value=100.0, step=10.0)
            quantity = st.number_input("Quantity", min_value=1, value=1, step=1)
            discount = st.slider("Discount", min_value=0.0, max_value=0.8, value=0.0, step=0.05, 
                                help="Higher discount typically reduces profit")
        
        if st.button("🔮 Predict Profit", type="primary"):
            # Input data
            input_data = pd.DataFrame({
                'Segment': [segment],
                'Region': [region],
                'Category': [category],
                'Sub-Category': [sub_category],
                'Sales': [sales],
                'Quantity': [quantity],
                'Discount': [discount],
                'Year': [2024],  # default values
                'Month': [1],
                'Quarter': [1],
                'DayOfWeek': [0],
                'Delivery_Days': [3]
            })
            
            # Feature engineering
            input_data['Profit_Margin'] = 0
            input_data['Sales_Per_Quantity'] = input_data['Sales'] / input_data['Quantity']
            input_data['Is_Discounted'] = (input_data['Discount'] > 0).astype(int)
            input_data['High_Value_Transaction'] = (input_data['Sales'] > 500).astype(int)
            
            # Encode
            for col in ['Segment', 'Region', 'Category', 'Sub-Category']:
                input_data[col] = encoders[col].transform(input_data[col])
            
            # Predict
            prediction = model.predict(input_data)[0]
            
            # Result
            st.markdown("---")
            st.markdown('<p class="sub-header">Prediction Result</p>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("💰 Sales", f"${sales:,.2f}")
            
            with col2:
                profit_margin = (prediction / sales * 100) if sales > 0 else 0
                st.metric("📊 Predicted Profit", f"${prediction:,.2f}")
            
            with col3:
                st.metric("📈 Profit Margin", f"{profit_margin:.2f}%")
            
            # Info
            if prediction > 0:
                st.success(f"✅ This order is expected to be profitable with an estimated profit of ${prediction:,.2f}")
            else:
                st.error(f"⚠️ This order may result in a loss of ${abs(prediction):,.2f}")
            
            # Summary
            st.markdown("---")
            st.markdown('<p class="sub-header">Order Summary</p>', unsafe_allow_html=True)
            
            summary_df = pd.DataFrame({
                'Metric': ['Sales', 'Quantity', 'Discount', 'Predicted Profit', 'Profit Margin'],
                'Value': [f"${sales:,.2f}", quantity, f"{discount*100:.0f}%", 
                         f"${prediction:,.2f}", f"{profit_margin:.2f}%"]
            })
            st.table(summary_df)
    
    # Page 3: Data Explorer
    else:
        st.markdown('<p class="sub-header">Explore Superstore Data</p>', unsafe_allow_html=True)
        
        # Filters
        st.sidebar.markdown("### Filters")
        
        selected_category = st.sidebar.multiselect("Category", df['Category'].unique(), default=df['Category'].unique())
        selected_region = st.sidebar.multiselect("Region", df['Region'].unique(), default=df['Region'].unique())
        
        # Filter data
        filtered_df = df[
            (df['Category'].isin(selected_category)) &
            (df['Region'].isin(selected_region))
        ]
        
        # Show filtered data stats
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📊 Filtered Records", f"{len(filtered_df):,}")
        
        with col2:
            st.metric("💰 Total Sales", f"${filtered_df['Sales'].sum():,.2f}")
        
        with col3:
            st.metric("💵 Total Profit", f"${filtered_df['Profit'].sum():,.2f}")
        
        st.markdown("---")
        
        # Display data
        st.markdown('<p class="sub-header">Data Table</p>', unsafe_allow_html=True)
        
        # Show sample or all data
        show_all = st.checkbox("Show all data", value=False)
        
        if show_all:
            st.dataframe(filtered_df, use_container_width=True, height=400)
        else:
            st.dataframe(filtered_df.head(100), use_container_width=True, height=400)
            st.info(f"Showing first 100 rows of {len(filtered_df):,} total records. Check 'Show all data' to see more.")
        
        # Download button
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv,
            file_name='filtered_superstore_data.csv',
            mime='text/csv',
        )
        
        # Statistical summary
        st.markdown("---")
        st.markdown('<p class="sub-header">Statistical Summary</p>', unsafe_allow_html=True)
        st.dataframe(filtered_df.describe(), use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #7f8c8d; padding: 20px;'>
            <p>📊 Superstore Profit Prediction Dashboard | Built with Streamlit & Machine Learning</p>
            <p>Data Science Portfolio Project</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
