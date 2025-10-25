# src/models/predict.py
import numpy as np

def predict_diabetes(model, features: dict):
    """Verilen özelliklere göre diyabet olasılığı döndürür."""
    feature_values = np.array(list(features.values())).reshape(1, -1)
    prob = model.predict_proba(feature_values)[0, 1]
    return prob
