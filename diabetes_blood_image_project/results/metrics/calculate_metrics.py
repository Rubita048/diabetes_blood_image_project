import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import os

def save_metrics(y_true, y_pred, model_name="RandomForest", save_path="results/metrics/results_summary.csv"):
    """Tahmin performans metriklerini hesaplar ve kaydeder."""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    prec = precision_score(y_true, y_pred, average="weighted")
    rec = recall_score(y_true, y_pred, average="weighted")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    df = pd.DataFrame([{
        "model": model_name,
        "accuracy": acc,
        "f1_score": f1,
        "precision": prec,
        "recall": rec
    }])

    if os.path.exists(save_path):
        old = pd.read_csv(save_path)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(save_path, index=False)
    print(f"✅ Metrikler kaydedildi: {save_path}")
