import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation(df):
    plt.figure(figsize=(10,8))
    sns.heatmap(df.corr(), cmap='coolwarm')
    plt.title("Özellik Korelasyonu")
    plt.show()

def plot_feature_scatter(df, x="circularity", y="contrast"):
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=x, y=y, hue="label", data=df, palette="tab10", alpha=0.7)
    plt.title(f"{x} vs {y}")
    plt.show()
