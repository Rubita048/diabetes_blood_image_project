import numpy as np

def normalize_data(X):
    """Veri ölçekleme."""
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
