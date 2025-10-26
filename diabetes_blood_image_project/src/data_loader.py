from src.data.load_data import load_features
from src.data.preprocessing import clean_data

def get_dataset(week=1):
    df = load_features(week)
    df = clean_data(df)
    X = df.drop(columns=["label"])
    y = df["label"]
    return X, y
