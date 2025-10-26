# models/load_model.py
import joblib
import os

def load_model(model_path="models/final_rf_v2.joblib"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model bulunamadı: {model_path}")
    model = joblib.load(model_path)
    print(f"✅ Model yüklendi: {model_path}")
    return model

if __name__ == "__main__":
    model = load_model()
