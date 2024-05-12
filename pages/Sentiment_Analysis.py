import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 加载预训练的 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def batch_sentiment_analysis(reviews, tokenizer, model, batch_size=32):
    # 将评论分成多个批次
    batches = [reviews[i:i + batch_size] for i in range(0, len(reviews), batch_size)]
    results = []

    for batch in batches:
        # 编码
        encoded_input = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}  # 移动到 GPU
        
        # 预测
        with torch.no_grad():
            outputs = model(**encoded_input)
        
        # 获取预测的类别索引
        predictions = torch.argmax(outputs.logits, dim=-1)
        results.extend(predictions.cpu().numpy())  # 移回 CPU

    return np.array(results) + 1  # 转换为情感评分


# 预处理评论数据







def show_wordcloud(reviews):
    '''
        reviews：pandas series
    '''
    text = ' '.join(r for r in reviews)
    wordcloud = WordCloud(width=800, height=400, background_color='grey').generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

#### Streamlit ####

st.title('Sentiment Analysis')

# uploading sentiment data
uploaded_file = st.file_uploader('Choose your csv file', type='csv')
if uploaded_file is not None:
    if 'name' not in st.session_state or st.session_state['name'] != uploaded_file.name:
        df = pd.read_csv(uploaded_file)
        st.session_state['data'] = df
        st.session_state['name'] = uploaded_file.name
        st.success('Dataset Loaded')
if 'data' in st.session_state:
    # display the dataset
    st.write(st.session_state.data)

    # button for generating sentiment score
    if st.button('Generate Scores'):
        sentiments = batch_sentiment_analysis(st.session_state.data['reviews'].tolist(), tokenizer, model)
        st.session_state.data['sentiment_scores'] = sentiments
    if 'sentiment_scores' in st.session_state.data.columns:
        st.write('Sample Output:')
        st.write(st.session_state.data.head())
    
    # draw wordcloud
    #选择sentiment score的range
    sentiment_range = st.select_slider(label='Select the range of sentiment score you wish to include',
                                        options=[1,2,3,4,5],
                                        value=5)
    temp_data = st.session_state.data.loc[st.session_state.data['sentiment_scores'] <= sentiment_range, :]

    #设置一级分类
    wordcloud_option_1 = st.selectbox(label='选择一级分类：',
                                    options=['Product_id', 'Product_Category', 'Food_type'],
                                    index=None,
                                    placeholder='Select a column you want to group by')

    #根据一级分类设置的情况，渲染对应的二级选项
    if (wordcloud_option_1 is not None):
        if (wordcloud_option_1 == 'Product_id'):
            product_id = st.number_input('Input the product id you wish to check',
                                        min_value=1,
                                        value=None)
            temp_data = st.session_state.data.loc[st.session_state.data['Product_id'] == product_id, :]

            # 展示词云图
            
        elif wordcloud_option_1 == 'Product_Category':
            product_selection = st.radio(label='选择Product_Category：',
                                        options=st.session_state.data[wordcloud_option_1].value_counts().index,
                                        index=None)
            temp_data = st.session_state.data.loc[st.session_state.data[wordcloud_option_1] == product_selection, :]

            #展示词云图


        elif wordcloud_option_1 == 'Food_type':
            food_selection = st.radio(label='选择Food_Type：',
                                      options=st.session_state.data[wordcloud_option_1].value_counts().index,
                                      index=None)
            temp_data = st.session_state.data.loc[st.session_state.data[wordcloud_option_1] == food_selection, :]

            # 展示词云图

    st.write(temp_data)