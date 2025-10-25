# src/ui/utils_ui.py
import streamlit as st
import pandas as pd

def display_feature_table(features_dict):
    """Özellikleri tablo şeklinde göster."""
    df = pd.DataFrame([features_dict])
    st.dataframe(df.style.highlight_max(axis=1, color='lightblue'))
