import numpy as np

def evaluate(predictions,Y):
    fp = np.sum((predictions == 1) & (Y == 0))
    fn = np.sum((predictions == 0) & (Y == 1))
    tp = np.sum((predictions == 1) & (Y == 1))
    tn = np.sum((predictions == 0) & (Y == 0))
    ACC = (tp + tn) / (tp + tn + fp + fn)
    SE = tp / (tp + fn)
    SP = tn / (tn + fp)
    return SE, SP, ACC
