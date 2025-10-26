import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_feature_importance(importances, feature_names, save_path="results/figures/feature_importance.png"):
    """Özellik önemlerini çubuk grafiği olarak kaydeder."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    sorted_idx = importances.argsort()[::-1]
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(importances)), importances[sorted_idx], color="teal")
    plt.xticks(range(len(importances)), [feature_names[i] for i in sorted_idx], rotation=45, ha="right")
    plt.title("Özellik Önemleri")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Grafik kaydedildi: {save_path}")
