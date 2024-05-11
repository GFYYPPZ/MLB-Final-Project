import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import streamlit as st

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
        sentiments = batch_sentiment_analysis(st.session_state.data['review_text'].tolist(), tokenizer, model)
        st.session_state.data['sentiment_scores'] = sentiments
    if 'sentiment_scores' in st.session_state.data.columns:
        st.write('Sample Output:')
        st.write(st.session_state.data.head())
    