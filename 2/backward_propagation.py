import numpy as np

def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]
    

    W1 = parameters["W1"]
    W2 = parameters["W2"]
        
    A1 = cache["A1"]
    A2 = cache["A2"]

    
    dZ2= A2 - Y
    dW2 = np.dot(dZ2 , A1.T) / m 
    db2 = np.sum(dZ2, axis = 1, keepdims = True) / m 
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis = 1, keepdims = True) / m 
        
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads
