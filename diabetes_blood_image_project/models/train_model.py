# models/train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

def train_and_save_model(data_path="data/features_week1.csv", model_path="models/final_rf_v2.joblib"):
    # Veriyi oku
    df = pd.read_csv(data_path)
    X = df.drop("label", axis=1)
    y = df["label"]

    # Eğitim/Test böl
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modeli eğit
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Performans
    y_pred = model.predict(X_test)
    print("\n📊 Model Performansı:")
    print(classification_report(y_test, y_pred))

    # Modeli kaydet
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, model_path)
    print(f"\n✅ Model '{model_path}' dosyasına kaydedildi.")

if __name__ == "__main__":
    train_and_save_model()
