import numpy as np

def compute_cost(A2, Y, parameters):
    
    m = Y.shape[1] 

    logprobs = np.multiply(np.log(A2),Y) + np.multiply(np.log(1 - A2), (1 - Y))
    cost = - np.sum(logprobs) / m

    
    cost = np.squeeze(cost)    
                                
    assert(isinstance(cost, float))
    
    return cost
