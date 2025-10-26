import numpy as np
import pandas as pd
from .feature_utils import texture_features

def extract_from_dataset(dataset):
    """Veri kümesinden özellikleri çıkarır (örnek)."""
    records = []
    for img, label in dataset:
        contrast, energy = texture_features(img)
        records.append({
            "contrast": contrast,
            "energy": energy,
            "label": label
        })
    return pd.DataFrame(records)
