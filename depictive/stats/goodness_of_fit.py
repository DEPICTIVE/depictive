import numpy as np



def rsq(true, predictions):
    return 1 - np.mean((true - predictions)**2) / np.var(true)
