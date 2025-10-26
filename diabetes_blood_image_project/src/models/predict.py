import joblib
import numpy as np

def load_model(path="models/final_rf_v2.joblib"):
    try:
        return joblib.load(path)
    except Exception as e:
        print("❌ Model yüklenemedi:", e)
        return None

def predict(model, features):
    return model.predict(np.array(features).reshape(1, -1))
