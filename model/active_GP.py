import numpy as np
import random
import scipy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.utils.optimize import _check_optimize_result
import utils

class active_GPR(GaussianProcessRegressor):
    def __init__(self, *args, max_iter=2e05, gtol=1e-06, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_iter = max_iter
        self._gtol = gtol

    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        if self.optimizer == "fmin_l_bfgs_b":
            opt_res = scipy.optimize.minimize(obj_func, initial_theta, method="L-BFGS-B", jac=True, bounds=bounds,
                                              options={'maxiter': self._max_iter, 'gtol': self._gtol})
            _check_optimize_result("lbfgs", opt_res)
            theta_opt, func_min = opt_res.x, opt_res.fun
        elif callable(self.optimizer):
            theta_opt, func_min = self.optimizer(obj_func, initial_theta, bounds=bounds)
        else:
            raise ValueError("Unknown optimizer %s." % self.optimizer)
        return theta_opt, func_min

    def data_aquire(self,candidate_data,pool_indecies):
        [mu,var] = self.predict(candidate_data[pool_indecies],return_std=True)
        index = pool_indecies[np.argmax(var)]
        return index

    def predict(self,inputs,return_cov=False,return_std=False):
        # sklearnのgpではノイズまで含めた事後分布推定をしているため，ノイズを除去している
        if return_cov:
            [mu,var] = super().predict(inputs,return_cov=True)
            noise_level = np.exp(self.kernel_.theta)[1]
            var = var - noise_level*np.eye(var.shape[0])
            pos = [mu,var]
        elif return_std:
            [mu,std] = super().predict(inputs,return_std=True)
            noise_level = np.exp(self.kernel_.theta)[1]
            std = np.sqrt(std**2 - noise_level)
            pos = [mu,std]
        else:
            pos = super().predict(inputs)
        return pos

    def get_var(self, X_ast):
        [mu,std] = self.predict(X_ast,return_std=True)
        return std

    def get_pri(self,X):
        return [np.zeros(X.shape[0]), self.kernel_(X,X)]

    def update_kernel(self,kernel):
        self.kernel_ = kernel

    def change_optimizer(self,optimizer):
        self.optimizer = optimizer
        self.kernel = self.kernel_

    def calc_init_sample(self,train_data,n_cv,test_data=None):
        init_sample_size = train_data[0].shape[0]
        KL_pq = np.zeros((n_cv, init_sample_size))
        KL_qp = np.zeros((n_cv, init_sample_size))
        init_test_error = np.zeros((n_cv, init_sample_size))
        sample_indecies = set(range(init_sample_size))
        params = np.exp(self.kernel_.theta)
        epsilon = 0.1
        for n in range(n_cv):
            resample_index = random.sample(sample_indecies, init_sample_size)
            for j in range(init_sample_size):
                if j == 0:
                    K = self.kernel_(train_data[0][resample_index[:j + 1]],train_data[0][resample_index[:j + 1]])
                    pos_old = [np.zeros(1), K]
                else:
                    pos_old = self.predict(train_data[0][resample_index[:j + 1]], return_cov=True)
                self.fit(train_data[0][resample_index[:j + 1]], train_data[1][resample_index[:j + 1]])
                # pos_new = self.predict(train_data[0][resample_index[:j + 1]], return_cov=True)
                KL_pq[n, j] = utils.calcKL_pq_fast(pos_old, train_data[1][resample_index[-1]], 1 / params[2])
                KL_qp[n, j] = utils.calcKL_qp_fast(pos_old, train_data[1][resample_index[-1]], 1 / params[2])
                # KL_pq[n, j] = utils.calcKL_gauss(pos_old, pos_new, epsilon)
                # KL_qp[n, j] = utils.calcKL_gauss(pos_new, pos_old, epsilon)
                if test_data is not None:
                    pos_test = self.predict(test_data[0], return_std=True)
                    init_test_error[n, j] = utils.calc_train_error_gauss(test_data[1], pos_test, params[1])
        if test_data is not None:
            return [KL_pq,KL_qp,init_test_error.mean(axis=0)]
        else:
            return [KL_pq,KL_qp]
