__author__ = 'billhuang'

import numpy as np

def sync_X(x):
    N = x.shape[0]
    X = np.hstack((np.ones((N,1)), x))
    return (X)
