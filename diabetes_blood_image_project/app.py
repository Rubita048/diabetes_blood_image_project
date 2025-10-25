import streamlit as st
import os
import joblib
import numpy as np
import cv2
from PIL import Image

# ============================================================
# 📁 Dosya yolları
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "final_rf_v2.joblib")

# ============================================================
# 🧠 Modeli yükleme
# ============================================================
@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Model yüklenemedi: {e}")
        return None

model = load_model()

# ============================================================
# 🔬 Görüntü işleme fonksiyonu
# ============================================================
def extract_features(image):
    """Kullanıcıdan alınan kan görüntüsünden basit istatistiksel özellikler çıkarır."""
    try:
        img_np = np.array(image)
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        img_gray = cv2.resize(img_gray, (64, 64))

        mean_intensity = np.mean(img_gray)
        std_intensity = np.std(img_gray)
        contrast = img_gray.max() - img_gray.min()

        hist = cv2.calcHist([img_gray], [0], None, [16], [0, 256]).flatten()
        hist = hist / np.sum(hist)

        features = np.concatenate(([mean_intensity, std_intensity, contrast], hist))
        return features.reshape(1, -1)

    except Exception as e:
        st.error(f"Özellik çıkarma hatası: {e}")
        return None

# ============================================================
# 🎨 Arayüz tasarımı
# ============================================================
st.set_page_config(page_title="Diyabet Tahmin Sistemi", page_icon="🩸", layout="wide")

st.title("🩸 Kan Görüntüsünden Diyabet Tahmini")
st.markdown("""
Bu uygulama, bir kan örneği görüntüsünden **basit görüntü işleme ve makine öğrenimi** yöntemleriyle
**diyabet olasılığını** tahmin eder.  
*(Bu proje yalnızca eğitim amaçlıdır.)*
""")

# ============================================================
# 📤 Görsel yükleme alanı
# ============================================================
uploaded_file = st.file_uploader("Kan örneği görüntüsünü yükle (JPG, PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Görsel önizleme
    st.image(image, caption="Yüklenen Görsel", use_container_width=True)

    # Özellik çıkarma
    with st.spinner("🔍 Görüntüden özellikler çıkarılıyor..."):
        features = extract_features(image)

    if features is not None and model is not None:
        # Tahmin
        with st.spinner("🧠 Model tahmin yapıyor..."):
            prediction = model.predict(features)[0]
            proba = model.predict_proba(features)[0][prediction] * 100

        # Sonuç gösterimi
        st.success(f"📊 Tahmin Sonucu: {'Diyabetli' if prediction == 1 else 'Diyabetli Değil'}")
        st.metric("Tahmin Güveni", f"%{proba:.2f}")
    else:
        st.warning("⚠️ Görüntüden özellik çıkarılamadı veya model bulunamadı.")

else:
    st.info("Lütfen bir kan görüntüsü yükleyin.")

# ============================================================
# 📎 Alt bilgi
# ============================================================
st.markdown("""
---
👨‍💻 *Bu uygulama eğitim amaçlı olarak geliştirilmiştir.*  
**Yapay zeka modelinin sonuçları tıbbi teşhis olarak değerlendirilmemelidir.**
""")
