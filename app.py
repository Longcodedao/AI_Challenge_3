import streamlit as st
from model import *
import pandas as pd
import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else 'cpu'

def inference_text(text, model, tokenizer):
    label = {0: 'Binh thuong',
             1: 'Xuc Pham',
             2: 'Thu Ghet'}
    encoding = tokenizer.encode_plus(
        text,
        max_length = 50,
        add_special_tokens = True,
        padding = 'max_length',
        return_token_type_ids = True,
        truncation = True,
        return_tensors = 'pt'
    ).to(device)

    with torch.no_grad():
        output = model(
            input_ids = encoding['input_ids'],
            token_type_ids = encoding['token_type_ids'],
            attention_mask = encoding['attention_mask']
        ).cpu()

        predicted_label = np.argmax(output, axis = 1).item()

    return predicted_label

st.set_page_config(
    page_title = 'Phân loại tục tĩu',
    page_icon='🤬',
    layout = 'wide'
)

@st.cache_resource
def initialize_model():
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    model = PhanLoaiTucTiu(n_classes = 3).to(device)

    model.load_state_dict(torch.load('phobert_tuctiu.pth', map_location = device))

    return model, tokenizer

model, tokenizer = initialize_model()

st.title('Phân loại tục tĩu')

text = st.text_area('Nhập văn bản cần phân loại', height = 100)
submit = st.button('Submit')

if submit:
    result = inference_text(text, model, tokenizer)
    if result == 0:
        custom_css = '''
            <style>
                .st-emotion-cache-5rimss  {
                    color: green;
                }
            </style>
            '''
        st.markdown(custom_css, unsafe_allow_html = True)
        st.write('Bạn đã nhập bình thường')
    elif result == 1:
        custom_css = '''
            <style>
                .st-emotion-cache-5rimss  {
                    color: red;
                }
            </style>
            '''
        st.write('Ngôn từ của bạn mang hàm ý xúc phạm')
        st.markdown(custom_css, unsafe_allow_html = True)

    elif result == 2:
        custom_css = '''
            <style>
                .st-emotion-cache-5rimss  {
                    color: red;
                    text-transform: uppercase;
                }
            </style>
            '''
        st.write('Ngôn từ của bạn mang hàm ý thù ghét. Xin cẩn trọng lời nói')
        st.markdown(custom_css, unsafe_allow_html = True)

