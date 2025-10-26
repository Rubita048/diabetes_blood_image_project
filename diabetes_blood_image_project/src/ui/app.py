import os
import joblib
import streamlit as st

# Mevcut dosya dizini
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# MODEL_PATH = iki seviye yukarı çık ve models klasörünü bul
MODEL_PATH = os.path.join(BASE_DIR, "../../models/final_rf_v2.joblib")

@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Model yüklenemedi: {e}")
        return None

model = load_model()
