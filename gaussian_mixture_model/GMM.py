__author__ = 'billhuang'

import numpy as np
import numerical_utils as nu
import kmean
from scipy import stats

def GSGMM(Y_, J_, params, iterations = 1000):
    sampler = GMM(Y_, J_, params)
    for i in range(iterations):
        sampler.update_params()
    return (sampler.z())
        

class GMM(object):
    def __init__(self, Y_, J_, params):
        self.Y_ = Y_
        self.N_, self.D_ = self.Y_.shape
        self.J_ = J_
        self.init_params(params)

    def init_params(self, params):
        self.init_z(params['z'])
        self.init_pi(params['pi'])
        self.init_mu(params['mu'])
        self.init_h(params['h'])

    def init_z(self, param):
        if (param['initializer'] == 'kmean'):
            self.z_ = kmean.kmean(self.Y_, self.J_)
        elif (param['initializer'] == 'random'):
            self.z_ = np.random.choice(self.J_, size = self.N_)

    def init_pi(self, param):
        self.api_ = param['a']
        self.pi_ = np.random.dirichlet(np.repeat(self.api_, self.J_))

    def init_mu(self, param):
        self.mmu_ = param['m']
        self.hmu_ = param['h']
        self.mu_ = np.random.normal(self.mmu_, np.sqrt(1./self.hmu_), size = (self.J_, self.D_))

    def init_h(self, param):
        self.ah_ = param['a']
        self.bh_ = param['b']
        self.h_ = np.random.gamma(self.ah_, 1./self.bh_, size = (self.J_, self.D_))

    def update_params(self):
        self.partition_Y()
        self.update_pi()
        self.update_mu()
        self.update_h()
        self.sync_Q()
        self.sample_z()

    def partition_Y(self):
        self.partition_ = []
        self.ncount_ = np.zeros(self.J_, dtype = int)
        for j in range(self.J_):
            Yj_ = self.Y_[self.z_ == j]
            self.partition_.append(Yj_)
            self.ncount_[j] = Yj_.shape[0]

    def update_pi(self):
        apipost_ = self.api_ + self.ncount_
        self.pi_ = np.random.dirichlet(apipost_)

    def update_mu(self):
        for j in range(self.J_):
            Yj_ = self.partition_[j]
            nj_ = self.ncount_[j]
            if (nj_ == 0):
                Yj_mean_ = np.repeat(0, self.D_)
            else:
                Yj_mean_ = np.mean(Yj_, axis = 0)
            for d in range(self.D_):
                hmupost_ = self.hmu_ + nj_ * self.h_[j,d]
                mmupost_ = (self.hmu_ * self.mmu_ + nj_ * self.h_[j,d] * Yj_mean_[d]) / hmupost_
                self.mu_[j,d] = np.random.normal(mmupost_, np.sqrt(1./hmupost_))

    def update_h(self):
        for j in range(self.J_):
            Yj_ = self.partition_[j]
            nj_ = self.ncount_[j]
            if (nj_ == 0):
                Yj_svar_ = np.repeat(0, self.D_)
            else:
                Yj_svar_ = np.sum(np.square(Yj_ - self.mu_[j,:]), axis = 0)
            for d in range(self.D_):
                ahpost_ = self.ah_ + nj_ / 2
                bhpost_ = self.bh_ + Yj_svar_[d] / 2
                self.h_[j,d] = np.random.gamma(ahpost_, 1./bhpost_)

    def sync_Q(self):
        logQ_ = np.zeros((self.N_, self.J_))
        for j in range(self.J_):
            logQ_[:,j] = stats.multivariate_normal.logpdf(self.Y_, self.mu_[j,:],
                                                         np.diag(1/self.h_[j,:]))
            logQ_[:,j] += nu.log(self.pi_[j])
        self.log_likelihood_ = np.sum(np.logaddexp.reduce(logQ_, axis = 1))
        self.Q_ = nu.normalize_log_across_row(logQ_)

    def sample_z(self):
        self.z_ = np.zeros(self.N_, dtype=int)
        for i in range(self.N_):
            self.z_[i] = np.random.choice(self.J_, p = self.Q_[i,:])

    def z(self):
        return (self.z_)

    def mu(self):
        return (self.mu_)

    def h(self):
        return (self.h_)

    def pi(self):
        return (self.pi_)
        
