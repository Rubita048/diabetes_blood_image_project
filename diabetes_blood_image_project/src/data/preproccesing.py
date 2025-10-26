import pandas as pd

def clean_data(df: pd.DataFrame):
    """Eksik verileri doldurur ve aykırı değerleri filtreler."""
    df = df.fillna(df.median())
    df = df[(df != 0).all(1)]
    return df
