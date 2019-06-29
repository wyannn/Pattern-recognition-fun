import numpy as np
from forward_propagation import forward_propagation

def predict(parameters, X):
    
    A2, cache = forward_propagation(X, parameters)
    predictions = np.around(A2)
    
    return predictions
