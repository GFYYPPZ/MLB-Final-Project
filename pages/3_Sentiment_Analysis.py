import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


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


# 删除标点符号
def remove_punctuation(text):
    '''
        text: 一个字符串
    '''
    translator = str.maketrans('','', string.punctuation)
    text = text.translate(translator)
    return text


# 删除停顿词
def remove_stopwords(text):
    # 设置停顿词
    stop_words = set(stopwords.words('english'))
    # 分词
    word_tokens = word_tokenize(text)
    # 过滤停顿词
    filtered_text = [w for w in word_tokens if w.lower() not in stop_words]
    return ' '.join(filtered_text)


# 文本预处理 - 合并
def text_preprocess(text):
    text = remove_punctuation(text)
    text = remove_stopwords(text)
    return text


# 画wordcloud
def show_wordcloud(reviews):
    '''
        reviews：pandas series
    '''
    text = ' '.join(r for r in reviews)
    wordcloud = WordCloud(width=1200, height=800, background_color='white').generate(text)

    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')

    return fig

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
        # st.write('Sample Output:')
        # st.write(st.session_state.data.head())
    
        # draw wordcloud
        # 创建一个session state的副本
        temp_data = st.session_state.data.copy()
        st.session_state['temp_data'] = temp_data

        #预处理文本数据
        st.session_state.temp_data['reviews'] = st.session_state.temp_data['reviews'].apply(lambda x : text_preprocess(x))

        #选择sentiment score的range
        sentiment_range = st.select_slider(label='Select the range of sentiment score you wish to include',
                                            options=[1,2,3,4,5],
                                            value=5)
        st.session_state.temp_data = st.session_state.temp_data.loc[st.session_state.temp_data['sentiment_scores'] <= sentiment_range, :]

        #设置一级分类
        wordcloud_option_1 = st.selectbox(label='Select the column you wish to generate word cloud：',
                                        options=['Product_id', 'Product_Category', 'Food_type'],
                                        index=None)

        #根据一级分类设置的情况，渲染对应的二级选项
        if (wordcloud_option_1 is not None):
            if (wordcloud_option_1 == 'Product_id'):
                product_id = st.number_input('Input the product id you wish to check',
                                            min_value=1,
                                            value=None)
                st.session_state.temp_data = st.session_state.temp_data.loc[st.session_state.temp_data['Product_id'] == product_id, :]

                if product_id:
                    try:
                        # 展示词云图
                        fig = show_wordcloud(st.session_state.temp_data['reviews'])
                        st.pyplot(fig)
                    except Exception as e:
                        st.write('We need at least 1 word to plot a word cloud (Do not have enough data)')
                
            elif wordcloud_option_1 == 'Product_Category':
                product_selection = st.radio(label='Select Product_Category：',
                                            options=st.session_state.temp_data[wordcloud_option_1].value_counts().index,
                                            index=None)
                st.session_state.temp_data = st.session_state.temp_data.loc[st.session_state.temp_data[wordcloud_option_1] == product_selection, :]

                if product_selection:
                    try:
                        # 展示词云图
                        fig = show_wordcloud(st.session_state.temp_data['reviews'])
                        st.pyplot(fig)
                    except Exception as e:
                        st.write('We need at least 1 word to plot a word cloud (Do not have enough data)')

                    # # 可以接着往下细看不同的food_type
                    # sub_selection = st.radio(label='选择细分Food_Type：',
                    #                          options=temp_data['Food_type'].value_counts().index,
                    #                          index=None)
                    # temp_data = temp_data.loc[temp_data['Food_type'] == sub_selection, :]

                    # if sub_selection:
                    #     #展示细分词云图
                    #     fig_sub = show_wordcloud(temp_data['reviews'])
                    #     st.pyplot(fig_sub)
                

            elif wordcloud_option_1 == 'Food_type':
                food_selection = st.radio(label='Select Food_Type：',
                                        options=st.session_state.temp_data[wordcloud_option_1].value_counts().index,
                                        index=None)
                st.session_state.temp_data = st.session_state.temp_data.loc[st.session_state.temp_data[wordcloud_option_1] == food_selection, :]

                if food_selection:
                    try:
                        # 展示词云图
                        fig = show_wordcloud(st.session_state.temp_data['reviews'])
                        st.pyplot(fig)
                    except Exception as e:
                        st.write('We need at least 1 word to plot a word cloud (Do not have enough data)')

                    # # 可以接着往下细看不同的product_category
                    # sub_selection = st.radio(label='选择细分Product_Category：',
                    #                          options=temp_data['Product_Category'].value_counts().index,
                    #                          index=None)
                    # temp_data = temp_data.loc[temp_data['Product_Category'] == sub_selection, :]

                    # if sub_selection:
                    #     #展示细分词云图
                    #     fig_sub = show_wordcloud(temp_data['reviews'])
                    #     st.pyplot(fig_sub)

            st.write(st.session_state.temp_data)