import streamlit as st
import cv2
import numpy as np

def show_image_preview(uploaded_file):
    bytes_data = uploaded_file.read()
    np_arr = np.frombuffer(bytes_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    st.image(img, caption="Yüklenen Görsel", use_column_width=True)
    return img
