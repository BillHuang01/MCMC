__author__ = 'billhuang'

import numpy as np
import linear_regression as lr

# y = 2 x1 + 3 x2 - 5 + eps

np.random.seed(1234)

N = 200
h = 1 

x1 = np.random.uniform(0, 10, size = N)
x2 = np.random.uniform(-5, 15, size = N)
eps = np.random.normal(0, np.sqrt(1./h), size = N)

y = 2 * x1 + 3 * x2 - 5 + eps

x = np.zeros((N,2))
x[:,0] = x1
x[:,1] = x2

params = {'w':{'mu':0, 'h':1},
          'h':{'a':1, 'b':1}}

w, h = lr.BLR(y, x, params)
print(w, h)
