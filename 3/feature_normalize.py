import numpy as np

def feature_normalize(X):
    mu = np.mean(X)
    X_norm = X - mu
    sigma = np.std(X_norm)
    X_norm = X_norm / sigma
    return X_norm
