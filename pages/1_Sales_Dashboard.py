import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from datetime import datetime
import plotly.graph_objects as go

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def total_sales_by_states(order, product, customer):
    # 合并订单和产品数据，以获取产品价格
    order_product = pd.merge(order, product, on='Product_id', how='left')
    order_product['Total_Price'] = order_product['Quantity'] * order_product['Product_Price']

    # 将合并后的订单产品数据与客户数据合并，以获取州信息
    order_product_customer = pd.merge(order_product, customer, on='Customer_id', how='left')

    state_sales = order_product_customer.groupby('Location')['Total_Price'].sum().reset_index()

    return state_sales

# File uploaders for each dataset
st.sidebar.header("Upload your data files")
uploaded_customer = st.sidebar.file_uploader("Upload Customer Table", type=['csv'])
uploaded_feedback = st.sidebar.file_uploader("Upload Feedback Table", type=['csv'])
uploaded_orders_overall = st.sidebar.file_uploader("Upload Order Overall Table", type=['csv'])
uploaded_products = st.sidebar.file_uploader("Upload Product Table", type=['csv'])

if (uploaded_customer is not None) and (uploaded_feedback is not None) and (uploaded_orders_overall is not None) and (uploaded_products is not None):
    st.session_state['customer'] = load_data(uploaded_customer)
    st.session_state['feedback'] = load_data(uploaded_feedback)
    st.session_state['orders_overall'] = load_data(uploaded_orders_overall)
    st.session_state['products'] = load_data(uploaded_products)

if ('customer' in st.session_state) and ('feedback' in st.session_state) and ('orders_overall' in st.session_state) and ('products' in st.session_state):
    customer = st.session_state['customer']
    feedback = st.session_state['feedback']
    orders_overall = st.session_state['orders_overall']
    products = st.session_state['products']



    #我就是写在这里让你看看大概这个dashboard要怎么搞，具体的细节你自己拿捏，我得去改后面的了

    st.title('Sales Analysis')

    # Ensure the date column is in datetime format
    orders_overall['Order_date'] = pd.to_datetime(orders_overall['Order_date'])
    
    # Filter data for the current year
    current_year = datetime.now().year
    orders_this_year = orders_overall[orders_overall['Order_date'].dt.year == current_year]

    # Merge orders with products to get price data for this year
    order_product_merged_this_year = pd.merge(orders_this_year, products, on="Product_id", how="left")
    order_product_merged_this_year['Total_Sales'] = order_product_merged_this_year['Quantity'] * order_product_merged_this_year['Product_Price']
    
    # Total Sales This Year
    total_sales_this_year = order_product_merged_this_year['Total_Sales'].sum()
    
    # Total Quantity Sold This Year
    total_quantity_sold_this_year = orders_this_year['Quantity'].sum()

    # Total Number of Orders This Year
    total_number_of_orders_this_year = orders_this_year['Order_id'].nunique()

    # Total Number of Customers (cumulative)
    total_number_of_customers = customer['Customer_id'].nunique()

    # Calculate cumulative metrics from all-time data
    order_product_merged_all_time = pd.merge(orders_overall, products, on="Product_id", how="left")
    order_product_merged_all_time['Total_Sales'] = order_product_merged_all_time['Quantity'] * order_product_merged_all_time['Product_Price']

    # Average Order Value (cumulative)
    average_order_value = order_product_merged_all_time.groupby('Order_id')['Total_Sales'].sum().mean()
    
    # Average Number of Orders Per Customer (cumulative)
    average_orders_per_customer = orders_overall['Customer_id'].value_counts().mean()

    # Calculate order completion rate for this year
    if not orders_this_year.empty:
        completed_orders_this_year = (orders_this_year['Status'] == 1).sum()
        total_orders_this_year = len(orders_this_year)
        completion_rate_this_year = (completed_orders_this_year / total_orders_this_year) * 100
    else:
        completion_rate_this_year = 0  # In case there are no orders this year

    # Calculate order completion rate cumulatively
    completed_orders_all_time = (orders_overall['Status'] == 1).sum()
    total_orders_all_time = len(orders_overall)
    completion_rate_all_time = (completed_orders_all_time / total_orders_all_time) * 100


    st.markdown('#### Key Metrics')
    # Row A
    with st.container(border=True):
        col1, col2, col3, col4 = st.columns(4)

        # Displaying the metrics for the current year
        with col1:
            st.metric(label='Total Sales This Year', value=f"${total_sales_this_year:,.2f}")

            st.metric(label='Total Number of Customers (Cumulative)', value=f"{total_number_of_customers}")
            
        with col2:
            st.metric(label='Total Quantity Sold This Year', value=f"{total_quantity_sold_this_year}")
            st.metric(label='Average Number of Orders Per Customer (Cumulative)', value=f"{average_orders_per_customer:.2f}")

        with col3:
            st.metric(label='Total Number of Orders This Year', value=f"{total_number_of_orders_this_year}")
            st.metric(label='Average Order Value (Cumulative)', value=f"{average_order_value:,.2f}")

        with col4:
            st.metric(label='Order Completion Rate This Year', value=f"{completion_rate_this_year:.2f}%")
            st.metric(label='Order Completion Rate (Cumulative)', value=f"{completion_rate_all_time:.2f}%")
   
    # Row B

    # Merge orders with products to get quantity and product details
    merged_data = pd.merge(orders_overall, products, on="Product_id", how="left")
    merged_data['Total_Sales'] = merged_data['Quantity'] * merged_data['Product_Price']

    # Calculate top 3 best selling products by quantity and their total sales
    top_products = merged_data.groupby('Product_name').agg({
        'Quantity': 'sum',
        'Total_Sales': 'sum'
    }).nlargest(3, 'Quantity').reset_index()

    # Calculate best selling product categories by quantity and total sales
    category_ranking = merged_data.groupby('Product_Category').agg({
        'Quantity': 'sum',
        'Total_Sales': 'sum'
    }).sort_values(by='Quantity', ascending=False).reset_index()
    
    # Calculate best selling food types by quantity and total sales
    food_type_ranking = merged_data.groupby('Food_type').agg({
        'Quantity': 'sum',
        'Total_Sales': 'sum'
    }).sort_values(by='Quantity', ascending=False).reset_index()



    with st.container(border=True):

        col1, col2, col3 = st.columns(3)

        with col1:
            fig = px.bar(top_products, x='Product_name', y='Quantity',
                        text=top_products['Total_Sales'].apply(lambda x: f"${x:,.2f}"),
                        labels={'Product_name': 'Product', 'Quantity': 'Quantity Sold'},
                        title="Top 3 Best Selling Products by Quantity")
            fig.update_traces(texttemplate='%{text}', textposition='outside')

            fig.update_layout(
                height=450,
                width=500
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Best Selling Product Categories Plot
            fig = px.bar(category_ranking, x='Product_Category', y='Quantity',
                        text=category_ranking['Total_Sales'].apply(lambda x: f"${x:,.2f}"),
                        labels={'Product_Category': 'Category', 'Quantity': 'Quantity Sold'},
                        title="Best Selling Product Categories by Quantity")
            fig.update_traces(texttemplate='%{text}', textposition='outside')
            
            fig.update_layout(
                height=450,
                width=500
            )

            st.plotly_chart(fig, use_container_width=True)

        with col3:
            # Best Selling Food Types Plot
            fig = px.bar(food_type_ranking, x='Food_type', y='Quantity',
                        text=food_type_ranking['Total_Sales'].apply(lambda x: f"${x:,.2f}"),
                        labels={'Food_type': 'Food Type', 'Quantity': 'Quantity Sold'},
                        title="Best Selling Food Types by Quantity")
            fig.update_traces(texttemplate='%{text}', textposition='outside')
            
            fig.update_layout(
                height=450,
                width=500
            )

            st.plotly_chart(fig, use_container_width=True)

    # Row C 这是我帮你画的图，函数帮你写好了，你自己看看放哪
    fig = px.choropleth(total_sales_by_states(orders_overall, products, customer), locations='Location',
                        locationmode='USA-states', color='Total_Price',
                        scope='usa', title='Total Sales By States',
                        color_continuous_scale='Greens')

    st.markdown("""
    <style>
    .plotly-graph-div {
        display: flex;
        justify-content: center;
    }
    </style>
    """, unsafe_allow_html=True)

    st.plotly_chart(fig)

    # Row D
    # Ensure the date column is in datetime format and prepare data
    orders_overall['Order_date'] = pd.to_datetime(orders_overall['Order_date'])
    orders_overall = pd.merge(orders_overall, products, on="Product_id", how="left")
    orders_overall['Total_Sales'] = orders_overall['Quantity'] * orders_overall['Product_Price']

    # Aggregate sales over time monthly
    monthly_sales = orders_overall.resample('M', on='Order_date')['Total_Sales'].sum()

    # Aggregate quarterly sales
    quarterly_sales = orders_overall.resample('Q', on='Order_date')['Total_Sales'].sum()

    # Plotting the line graph with quarterly sales bars
    fig = go.Figure()

    # Add bar for quarterly sales
    fig.add_trace(go.Bar(x=quarterly_sales.index, y=quarterly_sales, name='Quarterly Sales',
                         marker_color='rgb(158,202,225)', opacity=0.6))

    # Add line for monthly sales
    fig.add_trace(go.Scatter(x=monthly_sales.index, y=monthly_sales, mode='lines', name='Monthly Sales',
                             line=dict(color='royalblue', width=4)))

    fig.update_layout(title_text='Monthly/Quarterly Total Sales Over Time',
                      xaxis_title='Date',
                      yaxis_title='Total Sales',
                      template='plotly_white')

    st.plotly_chart(fig, use_container_width=True)

    # 根据时间，画折线图，一个折线图里面有不同的product total sales随着时间的变化
    # User choice for category breakdown (default: product category)
    breakdown_type = st.selectbox("Choose breakdown type:", ['Product Category', 'Food Type'], index=0)

    # Group data based on selected breakdown and plot over time
    if breakdown_type == 'Product Category':
        sales_over_time = orders_overall.groupby(['Order_date', 'Product_Category']).sum()['Total_Sales'].reset_index()
        category_or_type = 'Product_Category'
    else:
        sales_over_time = orders_overall.groupby(['Order_date', 'Food_type']).sum()['Total_Sales'].reset_index()
        category_or_type = 'Food_type'

    # Resample sales data monthly
    sales_over_time['Order_date'] = pd.to_datetime(sales_over_time['Order_date'])
    monthly_sales = sales_over_time.groupby([pd.Grouper(key='Order_date', freq='M'), category_or_type])['Total_Sales'].sum().unstack().fillna(0)
    
    # Create the line chart for each category or type
    fig = px.line(monthly_sales, x=monthly_sales.index, y=monthly_sales.columns,
                  labels={'value': 'Total Sales', 'variable': category_or_type, 'Order_date': 'Date'},
                  title=f'Total Sales Over Time by {category_or_type}')
    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("Please upload all data files to enable the dashboard.")