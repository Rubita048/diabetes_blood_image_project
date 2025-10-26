import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
import os

def train_model(X_train, y_train, save_path="models/final_rf_v2.joblib"):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    dump(model, save_path)
    print(f"✅ Model kaydedildi: {save_path}")
    return model
