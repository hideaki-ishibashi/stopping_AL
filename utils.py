import numpy as np
import itertools
from scipy.special import xlogy

def calcKL_gauss(pos1, pos2,epsilon=0.00):
    N = pos1[0].shape[0]
    f2 = pos2[0]
    f1 = pos1[0]
    S2 = pos2[1] + epsilon * np.eye(N)
    S1 = pos1[1] + epsilon * np.eye(N)
    S2_inv = np.linalg.inv(S2)
    S = S2_inv @ S1
    trace = np.trace(S)
    logdet = np.log(np.abs(np.linalg.det(S)))
    se = (f2 - f1).T @ S2_inv @ (f2 - f1)
    KL = 0.5 * (trace - logdet + se - N)
    return KL


def calcKL_pq_fast(pos_old,new_output,beta):
    m = pos_old[0][-1]
    k = pos_old[1][-1,-1]
    trace = beta*k
    logdet = np.log(1+beta*k)
    se = 1/(k+1/beta)**2*(k+k*beta*k)*(new_output-m)**2
    KL = 0.5*(trace - logdet + se)
    return KL


def calcKL_qp_fast(pos_old,new_output,beta):
    m = pos_old[0][-1]
    k = pos_old[1][-1,-1]
    trace = k/(k+1/beta)
    logdet = np.log(1+beta*k)
    se = k/((k+1/beta)**2)*(new_output-m)**2
    KL = 0.5*(-trace + logdet + se)
    return KL


def calc_log_normal(output,pos,var):
    [mean,std] = pos
    train_error = (((output - mean) ** 2).sum() + (std**2).sum()) / (2 * var * output.shape[0]) + 0.5*np.log(2*np.pi*var)
    return train_error


def calc_expected_squre_error(output,pos):
    [mean,std] = pos
    train_error = (((output - mean) ** 2).sum() + (std**2).sum()) / output.shape[0]
    return train_error


def calc_squre_error(output,f):
    train_error = (((output - f) ** 2).sum()) / output.shape[0]
    return train_error


def calc_cross_entropy(output,mu):
    train_error = -(xlogy(output,mu)).sum()/output.shape[0]
    return train_error


def calc_min_list(criterion):
    c_min = criterion.max()
    indecies = []
    for i, c in enumerate(criterion):
        if c_min >= c:
            c_min = c
            indecies.append(i)

    return indecies
