import pandas as pd

def save_features(features, path="../results/features_train.csv"):
    df = pd.DataFrame(features)
    df.to_csv(path, index=False)
    print(f"✅ Özellikler kaydedildi: {path}")
