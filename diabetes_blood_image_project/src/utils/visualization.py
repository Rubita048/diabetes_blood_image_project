import matplotlib.pyplot as plt

def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = importances.argsort()[::-1]
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.title("Özellik Önemleri")
    plt.tight_layout()
    plt.show()
