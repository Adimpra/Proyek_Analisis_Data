import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import folium
from streamlit_folium import folium_static
from folium.plugins import HeatMap

# Set page config
st.set_page_config(
    page_title="E-Commerce Data Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title('E-Commerce Data Analysis Dashboard')
st.markdown('''
This dashboard presents key insights from the e-commerce data analysis project.
- **Author:** Ahmad Imam Prasojo
- **Email:** ahmadimam.2020@student.uny.ac.id
''')

# Sidebar navigation
st.sidebar.title('Navigation')
page = st.sidebar.radio('Go to', ['Overview', 'Product Categories Analysis', 'Sales Trends', 'Customer Segmentation', 'Geospatial Analysis'])

# Function to load datasets
@st.cache_data
def load_data():
    try:
        # Load all datasets
        customers_df = pd.read_csv('../Data/customers_dataset.csv')
        geolocation_df = pd.read_csv('../Data/geolocation_dataset.csv')
        order_items_df = pd.read_csv('../Data/order_items_dataset.csv')
        order_payments_df = pd.read_csv('../Data/order_payments_dataset.csv')
        order_reviews_df = pd.read_csv('../Data/order_reviews_dataset.csv')
        orders_df = pd.read_csv('../Data/orders_dataset.csv')
        product_category_translation_df = pd.read_csv('../Data/product_category_name_translation.csv')
        products_df = pd.read_csv('../Data/products_dataset.csv')
        sellers_df = pd.read_csv('../Data/sellers_dataset.csv')
        
        # Clean the data
        # 1. Convert dates in orders_df to datetime
        date_columns = ['order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date', 
                        'order_delivered_customer_date', 'order_estimated_delivery_date']
        for col in date_columns:
            orders_df[col] = pd.to_datetime(orders_df[col], errors='coerce')
        
        # Handle NaN values in orders_df
        orders_df['order_approved_at'] = orders_df['order_approved_at'].fillna(orders_df['order_purchase_timestamp'])
        orders_df['order_delivered_carrier_date'] = orders_df['order_delivered_carrier_date'].fillna(orders_df['order_approved_at'])
        orders_df['order_delivered_customer_date'] = orders_df['order_delivered_customer_date'].fillna(orders_df['order_delivered_carrier_date'])
        
        # Handle missing values in products_df
        products_df['product_category_name'] = products_df['product_category_name'].fillna('unknown')
        
        return {
            'customers': customers_df,
            'geolocation': geolocation_df,
            'order_items': order_items_df,
            'order_payments': order_payments_df,
            'order_reviews': order_reviews_df,
            'orders': orders_df,
            'product_category_translation': product_category_translation_df,
            'products': products_df,
            'sellers': sellers_df
        }
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load data
with st.spinner('Loading data...'):
    data = load_data()

# Check if data is loaded successfully
if data is None:
    st.error("Failed to load data. Please check the data paths and format.")
    st.stop()

# Display overview
if page == 'Overview':
    st.header('Project Overview')
    
    st.subheader('About the Project')
    st.write("""
    This project analyzes e-commerce data to extract valuable insights about customers,
    products, sales patterns, and seller performance. The analysis includes data cleaning,
    exploratory data analysis, and visualization of key metrics.
    """)
    
    st.subheader('Key Business Questions')
    st.write("""
    - Which product categories are the most and least popular?
    - How do sales trends change over time?
    - What are the customer segments based on RFM analysis?
    - How does customer location influence orders and delivery?
    """)
    
    st.subheader('Dataset Overview')
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Customers", f"{data['customers'].shape[0]:,}")
        st.metric("Orders", f"{data['orders'].shape[0]:,}")
        st.metric("Products", f"{data['products'].shape[0]:,}")
    
    with col2:
        st.metric("Sellers", f"{data['sellers'].shape[0]:,}")
        st.metric("Order Items", f"{data['order_items'].shape[0]:,}")
        st.metric("Geolocations", f"{data['geolocation'].shape[0]:,}")

# Product Categories Analysis
elif page == 'Product Categories Analysis':
    st.header('Product Categories Analysis')
    
    # Merge datasets to get product categories in English
    order_items_with_category = pd.merge(
        data['order_items'], 
        data['products'][['product_id', 'product_category_name']], 
        on='product_id', 
        how='left'
    )
    
    # Merge with product_category_translation to get English names
    order_items_with_category = pd.merge(
        order_items_with_category, 
        data['product_category_translation'], 
        on='product_category_name', 
        how='left'
    )
    
    # Group by product_category_name_english and count orders
    category_sales_count = order_items_with_category.groupby('product_category_name_english')['order_id'].count().reset_index()
    category_sales_count.columns = ['product_category_name_english', 'number_of_sales']
    
    # Sort by number_of_sales
    category_sales_sorted = category_sales_count.sort_values(by='number_of_sales', ascending=False)
    
    # Display top and bottom categories
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top 10 Best-Selling Product Categories")
        fig1 = px.bar(
            category_sales_sorted.head(10), 
            y='product_category_name_english', 
            x='number_of_sales',
            orientation='h',
            color='number_of_sales',
            color_continuous_scale='Viridis',
            labels={'product_category_name_english': 'Product Category', 'number_of_sales': 'Number of Sales'}
        )
        fig1.update_layout(height=500)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.subheader("Bottom 10 Least-Selling Product Categories")
        fig2 = px.bar(
            category_sales_sorted.tail(10), 
            y='product_category_name_english', 
            x='number_of_sales',
            orientation='h',
            color='number_of_sales',
            color_continuous_scale='Magma',
            labels={'product_category_name_english': 'Product Category', 'number_of_sales': 'Number of Sales'}
        )
        fig2.update_layout(height=500)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Create pie chart for category distribution
    st.subheader("Sales Distribution Across Product Categories")
    
    # Define threshold for "Other" category (e.g., categories with less than 1% of total sales)
    threshold = st.slider("Threshold for 'Other' category (%)", 0.1, 5.0, 1.0) / 100
    total_sales = category_sales_sorted['number_of_sales'].sum()
    
    # Identify categories with sales below the threshold
    low_sales_categories = category_sales_sorted[category_sales_sorted['number_of_sales'] / total_sales < threshold]
    
    # Combine low-sales categories into "Other" category
    num_categories_in_other = len(low_sales_categories)
    other_sales = low_sales_categories['number_of_sales'].sum()
    
    # Filter out low-sales categories and add the "Other" category
    filtered_categories = category_sales_sorted[category_sales_sorted['number_of_sales'] / total_sales >= threshold].copy()
    other_row = pd.DataFrame({
        'product_category_name_english': [f'Other ({num_categories_in_other} categories)'],
        'number_of_sales': [other_sales]
    })
    filtered_categories = pd.concat([filtered_categories, other_row], ignore_index=True)
    
    # Create pie chart
    fig3 = px.pie(
        filtered_categories, 
        values='number_of_sales', 
        names='product_category_name_english',
        title=f'Sales Distribution (Low-sales categories < {threshold*100}% combined)'
    )
    fig3.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig3, use_container_width=True)

# Sales Trends
elif page == 'Sales Trends':
    st.header('Sales Trends Analysis')
    
    # Filter orders with status "delivered"
    valid_orders = data['orders'][data['orders']['order_status'] == 'delivered']
    
    # Merge with order items to get sales value
    order_sales = pd.merge(
        valid_orders[['order_id', 'order_purchase_timestamp']], 
        data['order_items'][['order_id', 'price', 'freight_value']], 
        on='order_id', 
        how='left'
    )
    
    # Calculate total sales value
    order_sales['total_sales_value'] = order_sales['price'] + order_sales['freight_value']
    
    # Extract year and month
    order_sales['year_month'] = order_sales['order_purchase_timestamp'].dt.strftime('%Y-%m')
    
    # Group by year_month and calculate total sales
    monthly_sales = order_sales.groupby('year_month')['total_sales_value'].sum().reset_index()
    
    # Sort by year_month to ensure chronological order
    monthly_sales['year_month'] = pd.to_datetime(monthly_sales['year_month'] + '-01')
    monthly_sales = monthly_sales.sort_values('year_month')
    monthly_sales['year_month_str'] = monthly_sales['year_month'].dt.strftime('%Y-%m')
    
    # Time range filter
    start_date = monthly_sales['year_month'].min().date()
    end_date = monthly_sales['year_month'].max().date()
    
    date_range = st.date_input(
        "Select date range",
        value=(start_date, end_date),
        min_value=start_date,
        max_value=end_date
    )
    
    if len(date_range) == 2:
        filtered_sales = monthly_sales[
            (monthly_sales['year_month'].dt.date >= date_range[0]) & 
            (monthly_sales['year_month'].dt.date <= date_range[1])
        ]
        
        # Plot sales trend
        fig = px.line(
            filtered_sales, 
            x='year_month', 
            y='total_sales_value',
            markers=True,
            labels={'year_month': 'Month', 'total_sales_value': 'Total Sales Value'},
            title='Monthly Sales Trends'
        )
        fig.update_layout(xaxis_tickformat='%b %Y')
        st.plotly_chart(fig, use_container_width=True)
        
        # Show sales statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Sales", f"${filtered_sales['total_sales_value'].sum():,.2f}")
        with col2:
            st.metric("Average Monthly Sales", f"${filtered_sales['total_sales_value'].mean():,.2f}")
        with col3:
            # Calculate growth rate
            if len(filtered_sales) > 1:
                first_month_sales = filtered_sales.iloc[0]['total_sales_value']
                last_month_sales = filtered_sales.iloc[-1]['total_sales_value']
                growth_rate = ((last_month_sales - first_month_sales) / first_month_sales) * 100
                st.metric("Growth Rate", f"{growth_rate:.2f}%")
            else:
                st.metric("Growth Rate", "N/A")
        
        # Show the data in a table
        st.subheader("Monthly Sales Data")
        display_data = filtered_sales[['year_month_str', 'total_sales_value']].copy()
        display_data.columns = ['Year-Month', 'Total Sales']
        display_data['Total Sales'] = display_data['Total Sales'].map('${:,.2f}'.format)
        st.dataframe(display_data, use_container_width=True)

# Customer Segmentation (RFM Analysis)
elif page == 'Customer Segmentation':
    st.header('Customer Segmentation (RFM Analysis)')
    
    # Merge orders with order payments to get payment value for each order
    customer_orders = pd.merge(
        data['orders'][['order_id', 'customer_id', 'order_purchase_timestamp']], 
        data['order_payments'][['order_id', 'payment_value']], 
        on='order_id', 
        how='left'
    )
    
    # Define current date (most recent order date)
    current_date = customer_orders['order_purchase_timestamp'].max()
    
    # Calculate RFM metrics
    rfm_data = customer_orders.groupby('customer_id').agg({
        'order_purchase_timestamp': lambda x: (current_date - x.max()).days,  # Recency
        'order_id': 'count',  # Frequency
        'payment_value': 'sum'  # Monetary
    }).reset_index()
    
    # Rename columns
    rfm_data.columns = ['customer_id', 'recency', 'frequency', 'monetary']
    
    # Assign RFM scores
    # Recency: lower is better (1 is highest score)
    rfm_data['recency_score'] = pd.qcut(rfm_data['recency'], q=5, labels=[5, 4, 3, 2, 1])
    
    # Define custom bins for frequency
    frequency_bins = [0, 1, 2, 3, 5, float('inf')]
    rfm_data['frequency_score'] = pd.cut(
        rfm_data['frequency'], 
        bins=frequency_bins, 
        labels=[1, 2, 3, 4, 5], 
        right=False
    )
    
    # Monetary: higher is better (5 is highest score)
    rfm_data['monetary_score'] = pd.qcut(rfm_data['monetary'], q=5, labels=[1, 2, 3, 4, 5])
    
    # Combine scores into a single RFM score
    rfm_data['rfm_score'] = rfm_data['recency_score'].astype(str) + rfm_data['frequency_score'].astype(str) + rfm_data['monetary_score'].astype(str)
    
    # Define segmentation rules
    def rfm_segment(row):
        if row['rfm_score'] == '555':
            return 'Best Customers'
        elif row['frequency_score'] >= 4 and row['monetary_score'] >= 4:
            return 'Loyal Customers'
        elif row['recency_score'] <= 2 and row['frequency_score'] >= 3 and row['monetary_score'] >= 3:
            return 'At Risk Customers'
        elif row['recency_score'] <= 2 and row['frequency_score'] <= 2 and row['monetary_score'] <= 2:
            return 'Lost Customers'
        else:
            return 'Other'
    
    # Apply segmentation
    rfm_data['segment'] = rfm_data.apply(rfm_segment, axis=1)
    
    # Count the number of customers in each segment
    segment_counts = rfm_data['segment'].value_counts().reset_index()
    segment_counts.columns = ['segment', 'count']
    
    # Create visualizations
    st.subheader("Customer Segments Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart of segment distribution
        fig1 = px.bar(
            segment_counts, 
            x='segment', 
            y='count',
            color='segment',
            labels={'segment': 'Customer Segment', 'count': 'Number of Customers'},
            title='Customer Segmentation Based on RFM Analysis'
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Pie chart of segment distribution
        fig2 = px.pie(
            segment_counts, 
            values='count', 
            names='segment',
            title='Customer Segment Distribution'
        )
        fig2.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig2, use_container_width=True)
    
    # Display segment characteristics
    st.subheader("Segment Characteristics")
    
    segment_metrics = rfm_data.groupby('segment').agg({
        'recency': 'mean',
        'frequency': 'mean',
        'monetary': 'mean',
        'customer_id': 'count'
    }).reset_index()
    
    segment_metrics.columns = ['Segment', 'Avg. Recency (days)', 'Avg. Frequency', 'Avg. Monetary Value', 'Count']
    segment_metrics = segment_metrics.sort_values('Count', ascending=False)
    
    # Format the metrics for display
    segment_metrics['Avg. Recency (days)'] = segment_metrics['Avg. Recency (days)'].round(1)
    segment_metrics['Avg. Frequency'] = segment_metrics['Avg. Frequency'].round(1)
    segment_metrics['Avg. Monetary Value'] = segment_metrics['Avg. Monetary Value'].round(2)
    
    st.dataframe(segment_metrics, use_container_width=True)
    
    # Allow exploring specific segments
    selected_segment = st.selectbox("Select a segment to explore", rfm_data['segment'].unique())
    
    if selected_segment:
        segment_data = rfm_data[rfm_data['segment'] == selected_segment]
        
        st.subheader(f"Distribution of {selected_segment}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig_recency = px.histogram(
                segment_data, 
                x='recency',
                nbins=20,
                title='Recency Distribution',
                labels={'recency': 'Recency (days)'}
            )
            st.plotly_chart(fig_recency, use_container_width=True)
        
        with col2:
            fig_frequency = px.histogram(
                segment_data, 
                x='frequency',
                nbins=20,
                title='Frequency Distribution',
                labels={'frequency': 'Number of Orders'}
            )
            st.plotly_chart(fig_frequency, use_container_width=True)
        
        with col3:
            fig_monetary = px.histogram(
                segment_data, 
                x='monetary',
                nbins=20,
                title='Monetary Distribution',
                labels={'monetary': 'Total Spend'}
            )
            st.plotly_chart(fig_monetary, use_container_width=True)

# Geospatial Analysis
elif page == 'Geospatial Analysis':
    st.header('Geospatial Analysis')
    
    analysis_type = st.radio("Select Analysis Type", ["Customer Locations", "Order Density by State", "Seller Locations"])
    
    if analysis_type == "Customer Locations":
        st.subheader("Customer Locations Heatmap")
        
        # Merge customers with geolocation
        customer_geolocation = pd.merge(
            data['customers'][['customer_id', 'customer_zip_code_prefix']], 
            data['geolocation'][['geolocation_zip_code_prefix', 'geolocation_lat', 'geolocation_lng']], 
            left_on='customer_zip_code_prefix', 
            right_on='geolocation_zip_code_prefix', 
            how='left'
        ).drop(columns=['geolocation_zip_code_prefix'])
        
        # Drop rows with missing coordinates
        customer_geolocation = customer_geolocation.dropna(subset=['geolocation_lat', 'geolocation_lng'])
        
        # Sample data for better performance
        sample_size = st.slider("Sample Size (%)", 1, 100, 10)
        sampled_customer_geolocation = customer_geolocation.sample(frac=sample_size/100, random_state=42)
        
        st.write(f"Displaying {len(sampled_customer_geolocation):,} customers out of {len(customer_geolocation):,} total.")
        
        # Create map
        m = folium.Map(location=[-14.2350, -51.9253], zoom_start=4)
        
        # Add heatmap layer
        heat_data = sampled_customer_geolocation[['geolocation_lat', 'geolocation_lng']].values.tolist()
        HeatMap(heat_data, radius=10).add_to(m)
        
        # Display map
        folium_static(m)
        
    elif analysis_type == "Order Density by State":
        st.subheader("Order Density by State")
        
        # Merge orders with customer data to get state info
        order_geolocation = pd.merge(data['orders'], data['customers'], on='customer_id', how='left')
        
        # Group orders by state and count
        order_density_by_state = order_geolocation.groupby('customer_state')['order_id'].count().reset_index()
        order_density_by_state.columns = ['state', 'order_count']
        
        # Merge with geolocation data to get coordinates
        state_geolocation = data['geolocation'].groupby('geolocation_state').agg({
            'geolocation_lat': 'mean',
            'geolocation_lng': 'mean'
        }).reset_index()
        state_geolocation.columns = ['state', 'lat', 'lng']
        
        # Merge order density with state geolocation
        order_density_map = pd.merge(order_density_by_state, state_geolocation, on='state', how='left')
        
        # Create map
        m = folium.Map(location=[-14.2350, -51.9253], zoom_start=4)
        
        # Add markers for each state
        for index, row in order_density_map.iterrows():
            folium.CircleMarker(
                location=[row['lat'], row['lng']],
                radius=row['order_count'] / 100,  # Scale the radius based on order count
                color='red',
                fill=True,
                fill_color='red',
                fill_opacity=0.6,
                popup=f"State: {row['state']}, Orders: {row['order_count']:,}"
            ).add_to(m)
        
        # Display map
        folium_static(m)
        
        # Show state order data
        st.subheader("Orders by State")
        sorted_states = order_density_by_state.sort_values('order_count', ascending=False)
        
        fig = px.bar(
            sorted_states,
            x='state',
            y='order_count',
            color='order_count',
            labels={'state': 'State', 'order_count': 'Number of Orders'},
            title='Order Count by State'
        )
        st.plotly_chart(fig, use_container_width=True)
        
    elif analysis_type == "Seller Locations":
        st.subheader("Seller Locations Heatmap")
        
        # Merge sellers with geolocation
        seller_geolocation = pd.merge(
            data['sellers'][['seller_id', 'seller_zip_code_prefix']], 
            data['geolocation'][['geolocation_zip_code_prefix', 'geolocation_lat', 'geolocation_lng']], 
            left_on='seller_zip_code_prefix', 
            right_on='geolocation_zip_code_prefix', 
            how='left'
        ).drop(columns=['geolocation_zip_code_prefix'])
        
        # Drop rows with missing coordinates
        seller_geolocation = seller_geolocation.dropna(subset=['geolocation_lat', 'geolocation_lng'])
        
        # Sample data for better performance
        sample_size = st.slider("Sample Size (%)", 1, 100, 20)
        sampled_seller_geolocation = seller_geolocation.sample(frac=sample_size/100, random_state=42)
        
        st.write(f"Displaying {len(sampled_seller_geolocation):,} sellers out of {len(seller_geolocation):,} total.")
        
        # Create map
        m = folium.Map(location=[-14.2350, -51.9253], zoom_start=4)
        
        # Add heatmap layer
        heat_data = sampled_seller_geolocation[['geolocation_lat', 'geolocation_lng']].values.tolist()
        HeatMap(heat_data, radius=10, gradient={'0.2': 'blue', '0.4': 'lime', '0.6': 'orange', '1': 'red'}).add_to(m)
        
        # Display map
        folium_static(m)

# Add footer information
st.sidebar.markdown("---")
st.sidebar.info("""
#### About this Dashboard
This dashboard was created using Streamlit to visualize the e-commerce data analysis project.

**Author:** Ahmad Imam Prasojo  
**Contact:** ahmadimam.2020@student.uny.ac.id
""")
