import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px


def total_sales_by_states(order, product, customer):
    # 合并订单和产品数据，以获取产品价格
    order_product = pd.merge(order, product, on='Product_id', how='left')
    order_product['Total_Price'] = order_product['Quantity'] * order_product['Product_Price']

    # 将合并后的订单产品数据与客户数据合并，以获取州信息
    order_product_customer = pd.merge(order_product, customer, on='Customer_id', how='left')

    state_sales = order_product_customer.groupby('Location')['Total_Price'].sum().reset_index()

    return state_sales

def total_customers_by_states(customer):
    state_customer = customer.groupby('Location')['Customer_id'].count().reset_index()
    return state_customer


#### Streamlit ####
st.title('Map Visualization')


if ('customer' in st.session_state) and ('orders_overall' in st.session_state) and ('products' in st.session_state):
    order = st.session_state.orders_overall
    product = st.session_state.products
    customer = st.session_state.customer

    # Row A
    col1, col2 = st.columns(2)
    with col1:
        option = st.selectbox(label='Select the variable you wish to check',
                              index=None,
                              options=['Gender', 'Income_level', 'Education', 'Occupation'])
        fig = px.pie(customer, names=option, title=f'{option} Distribution')
        st.plotly_chart(fig)
    
    with col2:
        fig = px.histogram(customer, x='Age', nbins=20, title='Customer Age Distribution')
        st.plotly_chart(fig)

    # Row B
    col1, col2 = st.columns(2)
    with col1:
        fig = px.choropleth(total_sales_by_states(order, product, customer), locations='Location',
                            locationmode='USA-states', color='Total_Price',
                            scope='usa', title='Total Sales By States',
                            color_continuous_scale='Greens')
        st.plotly_chart(fig)

    with col2:
        fig = px.choropleth(total_customers_by_states(customer), locations='Location',
                            locationmode='USA-states', color='Customer_id',
                            scope='usa', title='Total Number of Customers By States',
                            color_continuous_scale='Greens')
        st.plotly_chart(fig)