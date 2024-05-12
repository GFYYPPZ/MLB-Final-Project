import pandas as pd
import numpy as np
import streamlit as st

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

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

    # Row C
