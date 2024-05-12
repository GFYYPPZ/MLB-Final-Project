import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return None

st.title("Blue Buffalo Sales Dashboard")

# File uploaders for each dataset
st.sidebar.header("Upload your data files")
uploaded_customer = st.sidebar.file_uploader("Upload Customer Table", type=['csv'])
uploaded_feedback = st.sidebar.file_uploader("Upload Feedback Table", type=['csv'])
uploaded_order_details = st.sidebar.file_uploader("Upload Order Detail Table", type=['csv'])
uploaded_orders_overall = st.sidebar.file_uploader("Upload Order Overall Table", type=['csv'])
uploaded_products = st.sidebar.file_uploader("Upload Product Table", type=['csv'])

# Loading uploaded data
customers = load_data(uploaded_customer)
feedback = load_data(uploaded_feedback)
order_details = load_data(uploaded_order_details)
orders_overall = load_data(uploaded_orders_overall)
products = load_data(uploaded_products)

# Only proceed if data is loaded
if customers is not None and feedback is not None and order_details is not None and orders_overall is not None and products is not None:
    # Calculate total sales per order line
    order_details['Total_Sales'] = order_details['Price'] * order_details['Quantity']

    # Aggregate total sales by product
    total_sales_by_product = order_details.groupby('Product_id')['Total_Sales'].sum()

    # Merge with product table to get product names
    product_sales = pd.merge(total_sales_by_product, products[['Product_id', 'Product_name']], on='Product_id')
    top_products = product_sales.nlargest(3, 'Total_Sales')

    # Key Metrics
    st.header("Key Metrics")
    number_of_unique_customers = customers['Customer_id'].nunique()
    average_order_value = order_details.groupby('Order_id')['Total_Sales'].sum().mean()
    orders_per_customer = orders_overall['Customer_id'].value_counts().mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Number of Unique Customers", number_of_unique_customers)
    col2.metric("Average Order Value", f"${average_order_value:.2f}")
    col3.metric("Average Number of Purchases per Customer", f"{orders_per_customer:.2f}")

    # Top 3 Best Selling Products
    st.header("Top 3 Best Selling Products")
    for index, row in top_products.iterrows():
        st.subheader(f"{row['Product_name']}")
        st.write(f"Total Sales: ${row['Total_Sales']:,.2f}")

    # Dynamic Customer Segmentation Analysis
    st.header("Customer Purchase Behavior Segmentation")
    segment_choice = st.selectbox(
        "Select the customer attribute to segment by:",
        ['Gender', 'Age', 'Income_level', 'Occupation', 'Location', 'Education']
    )

    # Merge customer information with orders and calculate total sales
    customer_orders = pd.merge(customers, orders_overall, on='Customer_id')
    customer_order_details = pd.merge(customer_orders, order_details, on='Order_id')
    sales_by_segment = customer_order_details.groupby(segment_choice)['Total_Sales'].sum().nlargest(3)

    # Displaying the top segments
    fig, ax = plt.subplots()
    sales_by_segment.plot(kind='bar', ax=ax)
    ax.set_title(f"Top Segments by {segment_choice}")
    ax.set_xlabel(segment_choice)
    ax.set_ylabel("Total Sales ($)")
    st.pyplot(fig)

    # Category Breakdown with Interactive Selection
    st.header("Category Breakdown")
    breakdown_type = st.selectbox("Choose the category type to breakdown:", ['Product_Category', 'Food_type'])
    category_data = products[breakdown_type].value_counts(normalize=True)
    fig, ax = plt.subplots()
    ax.pie(category_data, labels=category_data.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

    # Quarterly Sales Overview
    st.header("Quarterly Sales Overview")
    orders_overall['Order_date'] = pd.to_datetime(orders_overall['Order_date'])
    merged_data = pd.merge(orders_overall[['Order_id', 'Order_date']], order_details[['Order_id', 'Total_Sales']], on='Order_id')
    merged_data.set_index('Order_date', inplace=True)
    quarterly_sales = merged_data.resample('Q').sum()
    fig, ax = plt.subplots()
    quarterly_sales['Total_Sales'].plot(kind='bar', ax=ax)
    quarters_labels = [f"{x.year} Q{x.quarter}" for x in quarterly_sales.index]
    ax.set_xticklabels(quarters_labels)
    ax.set_title('Quarterly Sales')
    ax.set_xlabel('Quarter')
    ax.set_ylabel('Total Sales ($)')
    st.pyplot(fig)

    # Revenue Breakdown
    st.header("Revenue Breakdown")
    st.text("??")

    # st.header("Continuous Revenue Breakdown")
    # income_level = st.selectbox("Select Customer Income Level:", sorted(customers['Income_level'].unique()))
    # filtered_by_income = orders_overall[orders_overall['Customer_id'].isin(customers[customers['Income_level'] == income_level]['Customer_id'])]

    # if st.checkbox("Breakdown by Product Category"):
    #     merged_data = pd.merge(filtered_by_income, products, on='Product_id')
    #     product_category = st.selectbox("Select Product Category:", sorted(merged_data['Product_Category'].unique()))
    #     filtered_by_category = merged_data[merged_data['Product_Category'] == product_category]

    #     if st.checkbox("Breakdown by Food Type"):
    #         food_type = st.selectbox("Select Food Type:", sorted(filtered_by_category['Food_type'].unique()))
    #         filtered_by_food_type = filtered_by_category[filtered_by_category['Food_type'] == food_type]
    #         total_revenue = filtered_by_food_type['Total_Sales'].sum()
    #         st.write(f"Total Revenue for {income_level} > {product_category} > {food_type}: ${total_revenue:,.2f}")

    # # Displaying Revenue Breakdown Visualization
    # fig, ax = plt.subplots()
    # if 'total_revenue' in locals():
    #     ax.bar([f"{income_level} > {product_category} > {food_type}"], [total_revenue])
    #     ax.set_title('Detailed Revenue Breakdown')
    #     ax.set_ylabel('Total Sales ($)')
    #     st.pyplot(fig)

else:
    st.warning("Please upload all data files to enable the dashboard.")
