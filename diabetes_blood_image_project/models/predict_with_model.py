# models/predict_with_model.py
import pandas as pd
from models.load_model import load_model

def predict_from_data(input_data):
    model = load_model("models/final_rf_v2.joblib")
    preds = model.predict(input_data)
    return preds

if __name__ == "__main__":
    # Örnek veri
    example = pd.DataFrame({
        "glucose": [120],
        "insulin": [85],
        "bmi": [29],
        "age": [45],
        "blood_pressure": [78],
        "cholesterol": [200]
    })
    result = predict_from_data(example)
    print(f"🔮 Tahmin sonucu: {result[0]}")
