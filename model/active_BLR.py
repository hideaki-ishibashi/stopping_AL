import numpy as np
from tqdm import tqdm
import random
import utils
from sklearn.linear_model import LogisticRegression
from scipy.special import xlogy

class active_BLR(LogisticRegression):
    def __init__(self, *args, basis_size=10, x_range, **kwargs):
        super().__init__(*args, **kwargs)
        self.basis_size = basis_size
        self.x_range = x_range
        self.basis_func = self.additive_RBF_function
        print(self.C)


    def data_acquire(self,candidate_data,pool_indecies):
        entropy = self.get_entropy(candidate_data[pool_indecies])
        index = pool_indecies[np.argmax(entropy)]
        return index

    def get_entropy(self,X_ast):
        pos = self.predict_proba(X_ast)
        entropy = -(xlogy(pos,pos)).sum(axis=1)
        return entropy

    def fit(self,X,y,init_W = None,sample_weight=None):
        if init_W is not None:
            self.coef_[0] = init_W
        self.X = X
        self.Phi = self.basis_func(self.X)
        self.M = X.shape[1]
        super().fit(self.Phi,y,sample_weight)

    def predict(self, X):
        Phi = self.basis_func(X)
        return super().predict(Phi)

    def predict_proba(self, X):
        Phi = self.basis_func(X)
        return super().predict_proba(Phi)

    def predict_log_proba(self, X):
        Phi = self.basis_func(X)
        return super().predict_log_proba(Phi)

    def get_pos(self):
        self.W = self.coef_[0]
        R = self.predict_proba(self.X).prod(axis=1)
        H = self.Phi.T @ np.diag(R) @ self.Phi + self.C * np.identity(self.Phi.shape[1])
        self.S = np.linalg.inv(H)
        return [self.W,self.S]

    def additive_RBF_function(self,inputs):
        [node, step] = np.linspace(self.x_range[0], self.x_range[1], self.basis_size, retstep=True)
        length = step / 2
        dist = ((inputs[:, None, :] - node[None, :, None]) ** 2)
        Phi = np.exp(-1 / (2 * length ** 2) * dist).reshape(inputs.shape[0], node.shape[0] * inputs.shape[1])
        return Phi
