import pandas as pd
import os

def load_features(week: int = 1):
    """CSV'den haftalık özellikleri yükler."""
    path = f"data/features_week{week}.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} bulunamadı.")
    return pd.read_csv(path)
