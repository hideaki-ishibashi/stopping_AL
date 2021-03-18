import numpy as np
from tqdm import tqdm
import random
import utils

class active_BRR(object):
    def __init__(self, beta, alpha, basis_size,x_range):
        self.beta = beta
        self.alpha = alpha
        self.basis_size = basis_size
        self.basis_func = self.Additive_RBF_function
        self.x_range = x_range

    def fit(self,X,y,fix_hyper_param=True,iterate_num=10):
        Phi = self.basis_func(X)
        self.m = self.basis_size*X.shape[1]
        if fix_hyper_param:
            self.S = np.linalg.inv(self.beta * Phi.T @ Phi + self.alpha * np.identity(Phi.shape[1]))
            self.W = self.beta * self.S @ Phi.T @ y
        else:
            [eval,_] = np.linalg.eigh(Phi.T @ Phi)
            for i in range(iterate_num):
                self.S = np.linalg.inv(self.beta * Phi.T @ Phi + self.alpha * np.identity(Phi.shape[1]))
                self.W = self.beta * self.S @ Phi.T @ y
                lambdas = self.beta*eval
                gamma = (lambdas/(lambdas+self.alpha)).sum()
                self.alpha = gamma/(self.W.T @ self.W)
                self.beta = (Phi.shape[0]-gamma)/(np.linalg.norm(y-Phi @ self.W)**2)

    def predict(self, X_ast,return_std=True):
        Phi_ast = self.Additive_RBF_function(X_ast)
        if not return_std:
            mu = Phi_ast @ self.W
            return mu
        else:
            mu = Phi_ast @ self.W
            std = np.sqrt(np.diag(Phi_ast @ self.S @ Phi_ast.T))
            return [mu,std]

    def get_var(self, X_ast):
        Phi_ast = self.Additive_RBF_function(X_ast)
        var = np.einsum("ik,kl,il->i",Phi_ast,self.S,Phi_ast)
        return var

    def get_pos(self):
        return [self.W,self.S]

    def get_pri(self):
        return [np.zeros(self.m), self.alpha*np.eye(self.m)]

    def data_aquire(self,candidate_data,pool_indecies):
        var = self.get_var(candidate_data[pool_indecies])
        index = pool_indecies[np.argmax(var)]
        return index

    def Additive_RBF_function(self,inputs):
        [node, step] = np.linspace(self.x_range[0], self.x_range[1], self.basis_size, retstep=True)
        length = step / 2
        dist = ((inputs[:, None, :] - node[None, :, None]) ** 2)
        Phi = np.exp(-1 / (2 * length ** 2) * dist).reshape(inputs.shape[0], node.shape[0] * inputs.shape[1])
        return Phi

    def calc_init_sample(self,train_data,n_cv,test_data=None):
        init_sample_size = train_data[0].shape[0]
        KL_pq = np.zeros((n_cv, init_sample_size))
        KL_qp = np.zeros((n_cv, init_sample_size))
        init_test_error = np.zeros((n_cv, init_sample_size))
        sample_indecies = set(range(init_sample_size))
        epsilon = 0
        for n in range(n_cv):
            resample_index = random.sample(sample_indecies, init_sample_size)
            for j in range(init_sample_size):
                if j == 0:
                    pos_old = self.get_pri()
                else:
                    pos_old = self.get_pos()
                self.fit(train_data[0][resample_index[:j + 1]], train_data[1][resample_index[:j + 1]],fix_hyper_param=True)
                pos_new = self.get_pos()
                KL_pq[n, j] = utils.calcKL_gauss(pos_old, pos_new, epsilon)
                KL_qp[n, j] = utils.calcKL_gauss(pos_new, pos_old, epsilon)
                if test_data is not None:
                    pos_test = self.predict(test_data[0])
                    init_test_error[n, j] = utils.calc_train_error_gauss(test_data[1], pos_test, 1 / self.beta)
                if j == 1:
                    KL_pq[n, j - 1] = KL_pq[n, j]
                    KL_qp[n, j - 1] = KL_qp[n, j]
        if test_data is not None:
            return [KL_pq,KL_qp,init_test_error.mean(axis=0)]
        else:
            return [KL_pq,KL_qp]
