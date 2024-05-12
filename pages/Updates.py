import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

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
    # Row A
    with st.container(border=True):
        st.markdown('### Metrics')
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label='Total Sales', value=100)
        with col2:
            st.metric(label='Total Quantity Sold', value=100)
        with col3:
            st.metric(label='Total Number of Customers', value=100)

    # Row B
    col1, col2 = st.columns(2)
    with col1:
        pass
    with col2:
        pass

    # Row C 这是我帮你画的图，函数帮你写好了，你自己看看放哪
    fig = px.choropleth(total_sales_by_states(orders_overall, products, customer), locations='Location',
                        locationmode='USA-states', color='Total_Price',
                        scope='usa', title='Total Sales By States',
                        color_continuous_scale='Greens')
    st.plotly_chart(fig)

    # 根据时间，看看sales的变化，
    # 根据时间，画折线图，一个折线图里面有不同的product total sales随着时间的变化

    # top product
    # top category

    