# data/generate_sample_data.py
import pandas as pd
import numpy as np
import os

def generate_synthetic_data(num_samples=200, random_state=42):
    np.random.seed(random_state)

    df = pd.DataFrame({
        "glucose": np.random.normal(100, 15, num_samples),
        "insulin": np.random.normal(80, 10, num_samples),
        "bmi": np.random.normal(27, 5, num_samples),
        "age": np.random.randint(20, 70, num_samples),
        "blood_pressure": np.random.normal(75, 12, num_samples),
        "cholesterol": np.random.normal(180, 30, num_samples),
        "label": np.random.randint(0, 2, num_samples)  # 0=Normal, 1=Diyabetik
    })

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/features_week1.csv", index=False)
    df.to_csv("data/features_week2.csv", index=False)

    print("✅ Örnek veriler data/ klasörüne kaydedildi.")

if __name__ == "__main__":
    generate_synthetic_data()
