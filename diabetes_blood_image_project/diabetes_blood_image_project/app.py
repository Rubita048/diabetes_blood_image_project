import streamlit as st
import joblib
import numpy as np
import os
from PIL import Image
import cv2

# -------------------------------
# 🔹 MODELİ YÜKLE
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "final_rf_v2.joblib")

@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Model yüklenemedi: {e}")
        return None

model = load_model()

# -------------------------------
# 🔹 SAYFA AYARLARI
# -------------------------------
st.set_page_config(
    page_title="Diabetes Detection via Blood Image",
    page_icon="🩸",
    layout="centered"
)

st.title("🩸 Diyabet Tahmin Aracı (Kan Görüntüsü)")
st.markdown("Bu araç, bir kan hücresi görüntüsünden diyabet olasılığını tahmin eder.")

# -------------------------------
# 🔹 GÖRÜNTÜ YÜKLEME ALANI
# -------------------------------
uploaded_file = st.file_uploader("Bir kan görüntüsü yükleyin (JPG veya PNG)", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Yüklenen Görüntü", use_column_width=True)

    # Görüntüyü model için hazırla
    img = np.array(image)
    img_resized = cv2.resize(img, (64, 64))
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    features = img_gray.flatten().reshape(1, -1)

    if model is not None:
        prediction = model.predict(features)[0]
        prob = model.predict_proba(features)[0][int(prediction)]

        st.success(f"🔍 Tahmin: **{'Diyabet' if prediction == 1 else 'Normal'}**")
        st.info(f"Olasılık: %{prob * 100:.2f}")
    else:
        st.error("Model bulunamadı. Lütfen `models/final_rf_v2.joblib` dosyasını kontrol edin.")
