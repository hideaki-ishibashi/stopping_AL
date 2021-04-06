import matplotlib.pylab as plt
from stopping_criteria import *
import utils
import draw_result
import get_dataset
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from tqdm import tqdm
from model.active_GP import active_GPR
import random
import os

def main():
    np.random.seed(1)
    random.seed(1)
    data_names = ["power_plant","protein","gas_emission","grid_stability"]
    noise_level_bounds_list = [(1e-3, 1e3),(1e-3, 1e3),(1e-1, 1e3),(1e-1, 1e3),(1e-3, 1e3)]
    corr_list = np.zeros((len(data_names)))
    fontsize = 24
    fontsize_corr = 40
    batch_size = 1

    for i,data_name in enumerate(data_names):
        save_dir = "result/GPR/"
        os.makedirs(save_dir, exist_ok=True)
        [X,y] = get_dataset.get_dataset(data_name)
        [whole_size, dim] = X.shape
        print(X.shape)
        test_size = 2000
        sample_size = 1000
        train_size = min([5000,whole_size - test_size])
        X_test = X[:test_size]
        y_test = y[:test_size]
        X_train = X[test_size:test_size + train_size]
        y_train = y[test_size:test_size + train_size]

        length_scale_bounds = (1e-3, 1e3)
        noise_level_bounds = noise_level_bounds_list[i]
        kernel = RBF(length_scale=1.0, length_scale_bounds=length_scale_bounds) + WhiteKernel(noise_level=1.0,
                                                                                                 noise_level_bounds=noise_level_bounds)
        gp = active_GPR(kernel=kernel, alpha=0.0, optimizer="fmin_l_bfgs_b",n_restarts_optimizer=5)
        # gp = active_GPR(kernel=kernel, alpha=0.0, optimizer=None)
        gp.fit(X_train[:sample_size], y_train[:sample_size])
        gp.change_optimizer(None)
        init_sample_size = 10
        validate_size = 10
        threshold = 0.05
        error_stability1 = error_stability_criterion(threshold,validate_size)
        threshold = 0.04
        error_stability2 = error_stability_criterion(threshold,validate_size)
        threshold = 0.03
        error_stability3 = error_stability_criterion(threshold,validate_size)
        criteria = [error_stability1,error_stability2,error_stability3]

        pool_indecies = set(range(train_size))
        sampled_indecies = set(random.sample(pool_indecies, init_sample_size))
        pool_indecies = list(pool_indecies - sampled_indecies)
        sampled_indecies = list(sampled_indecies)
        gp.fit(X_train[sampled_indecies], y_train[sampled_indecies])
        test_error = np.empty(0,float)

        color = {error_stability1.criterion_name: "r",error_stability2.criterion_name: "g",error_stability3.criterion_name: "b"}
        params = np.exp(gp.kernel_.theta)
        print(params)
        for e in tqdm(range(sample_size)):
            new_data_index = gp.data_aquire(X_train,pool_indecies)
            sampled_indecies.append(new_data_index)
            pool_indecies.remove(new_data_index)

            X_sampled = X_train[sampled_indecies]
            y_sampled = y_train[sampled_indecies]

            pos_old = gp.predict(X_sampled,return_cov = True)
            gp.fit(X_sampled, y_sampled)

            error = utils.calc_expected_squre_error(y_test, gp.predict(X_test, return_std=True))
            test_error = np.append(test_error,error)

            KL_pq = utils.calcKL_pq_fast(pos_old, y_sampled[-1], 1 / params[1])
            KL_qp = utils.calcKL_qp_fast(pos_old, y_sampled[-1], 1 / params[1])

            error_stability1.check_threshold(KL_pq, KL_qp, e)
            error_stability2.check_threshold(KL_pq, KL_qp, e)
            error_stability3.check_threshold(KL_pq, KL_qp, e)

        draw_result.draw_gene_error(test_error, criteria, init_sample_size,batch_size,color, fontsize)
        plt.tight_layout()
        plt.savefig(save_dir+"GPR_gene_error_"+data_name+".pdf")
        draw_result.draw_correlation(test_error[validate_size:], error_stability1.error_ratio[validate_size:], "GPR", "r", fontsize_corr)
        plt.tight_layout()
        plt.savefig(save_dir+"GPR_correlation_"+data_name+".pdf")
        draw_result.draw_epsilon(criteria, init_sample_size,batch_size,color, fontsize)
        plt.tight_layout()
        plt.savefig(save_dir+"GPR_criterion_"+data_name+".pdf")

        indecies = utils.calc_min_list(error_stability1.error_ratio[validate_size:])
        corr_list[i] = np.corrcoef(test_error[validate_size:][indecies], error_stability1.error_ratio[validate_size:][indecies])[1, 0]

    np.savetxt(save_dir+"corr_list.txt",corr_list)
    np.savetxt(save_dir+"loss.txt",np.array(test_error))
    np.savetxt(save_dir+"lambda.txt",error_stability1.error_ratio)

if __name__ == "__main__":
    main()
