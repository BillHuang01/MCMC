__author__ = 'billhuang'

import numpy as np
import regression_utils as ru

def BLR(y, x, params, iterations = 200):
    regression = LR(y, x, params)
    for i in range(iterations):
        regression.update_params()
    return (regression.w(), regression.h())

class LR(object):
    def __init__(self, y, x, params):
        self.N_ = y.size
        self.y_ = y
        self.X_ = ru.sync_X(x)
        self.D_ = self.X_.shape[1]
        self.init_params(params)

    def init_params(self, params):
        self.init_w(params['w'])
        self.init_h(params['h'])

    def init_w(self, param):
        self.wmu_ = param['mu'] * np.ones(self.D_)
        self.wh_ = param['h'] * np.eye(self.D_)
        self.w_ = np.random.multivariate_normal(self.wmu_, np.linalg.inv(self.wh_))

    def init_h(self, param):
        self.ah_ = param['a']
        self.bh_ = param['b']
        self.h_ = np.random.gamma(self.ah_, 1./self.bh_)

    def update_params(self):
        self.update_w()
        self.update_h()

    def update_w(self):
        whpost_ = self.wh_ + self.h_ * np.dot(self.X_.T, self.X_)
        wSpost_ = np.linalg.inv(whpost_)
        wmunum_ = np.dot(self.wmu_.T, whpost_) + self.h_ * np.dot(self.y_.T, self.X_)
        wmupost_ = np.dot(wmunum_, wSpost_).reshape((self.D_,))
        self.w_ = np.random.multivariate_normal(wmupost_, wSpost_)

    def update_h(self):
        ahpost_ = self.ah_ + self.N_ / 2
        eps = self.y_ - np.dot(self.X_, self.w_)
        bhpost_ = self.bh_ + np.dot(eps.T, eps) / 2
        self.h_ = np.random.gamma(ahpost_, 1./bhpost_)

    def w(self):
        return (self.w_)

    def h(self):
        return (self.h_)
