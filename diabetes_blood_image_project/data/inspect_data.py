# data/inspect_data.py
import pandas as pd

def show_data_summary(file_path):
    df = pd.read_csv(file_path)
    print(f"\n📊 {file_path} için özet:")
    print(df.head())
    print("\n🔹 Temel İstatistikler:")
    print(df.describe())
    print("\n🔹 Eksik Değer Sayısı:")
    print(df.isnull().sum())

if __name__ == "__main__":
    show_data_summary("data/features_week1.csv")
