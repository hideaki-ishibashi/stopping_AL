import numpy as np
import mpmath as mp
from sklearn.model_selection import KFold
import statistics
import scipy.stats as st


class base_criterion(object):
    def __init__(self,criterion_name):
        self.criterion_name = criterion_name
        self.stop_flags = False
        self.stop_timings = np.nan

class error_stability_criterion(base_criterion):
    def __init__(self,threshold,validate_size=1):
        super(error_stability_criterion, self).__init__("Error stability={0}".format(threshold))
        self.error_ratio = np.empty(0,float)
        self.threshold = threshold
        self.validate_size = validate_size
        self.KL_pq = np.empty(0, float)
        self.KL_qp = np.empty(0, float)
        self.R_upper = np.empty(0, float)
        self.R_lower = np.empty(0, float)
        self.R = np.empty(0, float)

    def check_threshold(self,KL_pq,KL_qp,current_time):
        tol = 1e-10
        self.KL_pq = np.append(self.KL_pq,KL_pq)
        self.KL_qp = np.append(self.KL_qp,KL_qp)
        u_pq = (KL_pq - 1) / (np.exp(1))
        if u_pq > -1/mp.e+tol:
            Lambda_upper = float(mp.lambertw(u_pq) + 1)
        else:
            Lambda_upper = 0
        u_qp = (KL_qp - 1) / (np.exp(1))
        if u_qp > -1/mp.e+tol:
            Lambda_lower = float(mp.lambertw(u_qp) + 1)
        else:
            Lambda_lower = 0
        self.R_upper = np.append(self.R_upper,(np.exp(Lambda_upper) - 1))
        self.R_lower = np.append(self.R_lower,(np.exp(Lambda_lower) - 1))
        self.R = np.append(self.R,self.R_upper[-1] + self.R_lower[-1])
        if self.validate_size <= current_time:
            error_ratio = self.R[-1] / self.R[:self.validate_size+1].min()
            if self.validate_size == current_time:
                error_ratio = 1
            self.error_ratio = np.append(self.error_ratio,error_ratio)
            if self.error_ratio[-1] <= self.threshold and not self.stop_flags:
                self.stop_timings = current_time
                print("{} : {}".format(self.criterion_name, current_time))
                self.stop_flags = True
        else:
            self.error_ratio = np.append(self.error_ratio, 1)


class max_confidence_criterion(base_criterion):
    def __init__(self,threshold):
        super(max_confidence_criterion, self).__init__("Max confidence")
        self.threshold = threshold
        self.max_var = np.empty(0,float)
        self.criterion = np.empty(0,float)

    def check_threshold(self,confidence,current_time):
        if confidence.shape[0] > 0:
            max_confidence = np.max(confidence)
            self.max_confidence = np.append(self.max_var,np.max(confidence))
            self.criterion = np.append(self.criterion,max_confidence/self.max_var[0])
            if self.threshold >= self.criterion[-1] and not self.stop_flags:
                self.stop_timings = current_time
                print("{} : {}".format(self.criterion_name,current_time))
                self.stop_flags = True


class run_criterion(base_criterion):
    def __init__(self,length,threshold,start_time):
        super(run_criterion, self).__init__("Run")
        self.criterion = np.empty(0, float)
        self.threshold = threshold
        self.R = np.empty(0, float)
        self.length = length
        self.start_time = start_time

    def check_threshold(self,KL,current_time):
        self.R = np.append(self.R,KL)
        if self.start_time <= current_time:
            if self.R.shape[0] > self.length:
                s = self.R.shape[0] - self.length
                R = self.R[s:]
            else:
                R = self.R
            prob = self.runTest(R)
            self.criterion = np.append(self.criterion,prob)
            if prob <= self.threshold and not self.stop_flags:
                self.stop_timings = current_time
                print("{} : {}".format(self.criterion_name,current_time))
                self.stop_flags = True

    def runTest(self,sequence):
        median = statistics.median(sequence)
        n_one = 0
        n_zero = 0
        K = 0
        pre_num = -1
        runs = []
        for c in self.R:
            if c - median < 0:
                runs.append(0)
                n_zero += 1
                if pre_num != 0:
                    K += 1
                pre_num = 0
            else:
                runs.append(1)
                n_one += 1
                if pre_num != 1:
                    K += 1
                pre_num = 1
        n = n_zero + n_one
        mu = 2 * n_zero * n_one / n + 1
        if n_zero == 0 or n_one == 0:
            return 1.0
        sigma = 2 * n_zero * n_one * (2 * n_zero * n_one - n) / ((n ** 2) * (n - 1))
        if sigma <= 1e-6:
            return 1.0
        Z = (K - mu) / np.sqrt(sigma)
        cdf = st.norm.cdf(Z)
        return cdf

