# app.py
import streamlit as st
import joblib
import numpy as np
import os
from PIL import Image
import cv2

# ------------------------------------------------------------
# 1️⃣ Başlık
st.set_page_config(page_title="Diyabet Tahmin Aracı", page_icon="🩸")
st.title("🩸 Diyabet Tahmin Uygulaması (Kan Görüntüsüne Dayalı)")

# ------------------------------------------------------------
# 2️⃣ Modeli yükle
MODEL_PATH = "models/final_rf_v2.joblib"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model bulunamadı. Lütfen `models/final_rf_v2.joblib` dosyasını kontrol edin.")
        return None
    try:
        model = joblib.load(MODEL_PATH)
        st.success("✅ Model başarıyla yüklendi.")
        return model
    except Exception as e:
        st.error(f"Model yüklenemedi: {e}")
        return None

model = load_model()

# ------------------------------------------------------------
# 3️⃣ Görüntü yükleme alanı
uploaded_file = st.file_uploader("📷 Kan görüntüsünü yükle (JPG veya PNG)", type=["jpg", "jpeg", "png"])

# ------------------------------------------------------------
# 4️⃣ Görüntü yüklendiyse
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Yüklenen Görüntü", use_column_width=True)

    # Görseli NumPy dizisine çevir
    img_array = np.array(image)

    # Basit ön işleme
    img_resized = cv2.resize(img_array, (64, 64))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)

    # Görselden basit özellik çıkarımı
    features = np.array([[mean_intensity, std_intensity]])
    st.write("🔍 Görselden çıkarılan örnek özellikler:")
    st.json({
        "Ortalama Yoğunluk": float(mean_intensity),
        "Yoğunluk Std Sapması": float(std_intensity)
    })

    # ------------------------------------------------------------
    # 5️⃣ Tahmin
    if model is not None:
        try:
            prediction = model.predict(features)
            result = "🔴 Diyabet riski yüksek" if prediction[0] == 1 else "🟢 Diyabet riski düşük"
            st.subheader("🎯 Tahmin Sonucu:")
            st.write(result)
        except Exception as e:
            st.error(f"Tahmin yapılamadı: {e}")
else:
    st.info("👆 Bir kan görüntüsü yükleyerek başlayın.")
