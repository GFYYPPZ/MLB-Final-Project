import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px


def aggregate_tables(order, product, customer):
    # 合并订单和产品数据，以获取产品价格
    order_product = pd.merge(order, product, on='Product_id', how='left')
    order_product['Total_Price'] = order_product['Quantity'] * order_product['Product_Price']
    # 将合并后的订单产品数据与客户数据合并
    order_product_customer = pd.merge(order_product, customer, on='Customer_id', how='left')

    return order_product_customer



def total_sales_by_states(order, product, customer):
    order_product_customer = aggregate_tables(order, product, customer)

    state_sales = order_product_customer.groupby('Location')['Total_Price'].sum().reset_index()

    return state_sales


def total_customers_by_states(customer):
    state_customer = customer.groupby('Location')['Customer_id'].count().reset_index()
    return state_customer


def age_bin(customer):
    bins = [18, 29, 44, 59, 100]
    labels = ['Young', 'Adult', 'Middle-aged', 'Senior']
    age_bin = pd.cut(customer['Age'], bins=bins, labels=labels, right=False)
    return age_bin


def sales_by_variables(var):
    order_product_customer = aggregate_tables(order, product, customer)
    sales_by_var = order_product_customer.groupby(var)['Total_Price'].sum().reset_index()
    sales_by_var.columns = ['Var', 'Total']
    return sales_by_var



#### Streamlit ####
st.title('Customer Analysis')


if ('customer' in st.session_state) and ('orders_overall' in st.session_state) and ('products' in st.session_state):
    order = st.session_state.orders_overall
    product = st.session_state.products
    customer = st.session_state.customer

    customer_option_1 = st.selectbox(label='Select the variable you wish to check distribution:',
                    options=['Gender', 'Income_level', 'Education', 'Occupation'])
    # Row A
    col1, col2, col3 = st.columns([0.3, 0.4, 0.4])
    with col1:
        fig = px.pie(customer, names=customer_option_1, title=f'{customer_option_1} Distribution')
        
        fig.update_layout(
            width=300,  # 宽度为500像素
            height=400  # 高度为400像素
        )

        st.plotly_chart(fig)
    
    with col2:
        fig = px.histogram(customer, x='Age', nbins=20, title='Customer Age Distribution')

        fig.update_layout(
            width=500,  # 宽度为500像素
            height=400  # 高度为400像素
        )

        st.plotly_chart(fig)

    with col3:
        customer['Age_Bin'] = age_bin(customer)
        fig = px.pie(customer, names='age_bin', title='Age (Binned) Distribution')

        fig.update_layout(
            width=500,  # 宽度为500像素
            height=400  # 高度为400像素
        )

        st.plotly_chart(fig)

    # Row B
    col1, col2 = st.columns([0.6, 0.5])
    with col1:
        fig = px.choropleth(total_customers_by_states(customer), locations='Location',
                            locationmode='USA-states', color='Customer_id',
                            scope='usa', title='Total Number of Customers By States',
                            color_continuous_scale='Greens', labels={'Customer_id': 'Number of Customer'})
        fig.update_layout(
            height=700,
            width=700
        )
        st.plotly_chart(fig)

    with col2:
        customer_option_2 = st.selectbox(label='Select the variable you wish to look into:',
                                        options=['Gender', 'Income_level', 'Education', 'Occupation', 'Age_Bin'])
        fig = px.bar(sales_by_variables(customer_option_2), x='Var', y='Total', title=f'Total Sales By {customer_option_2}')
        st.plotly_chart(fig)