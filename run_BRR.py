import matplotlib.pylab as plt
from stopping_criteria import *
import utils
import draw_result
import get_dataset
from tqdm import tqdm
from model.active_BRR import active_BRR
import random
import os

def main():
    np.random.seed(1)
    random.seed(1)
    data_names = ["power_plant","protein","gas_emission","grid_stability"]
    data_sizes = [2000,2000,1000,8000]
    corr_list = np.zeros((len(data_names)))
    fontsize = 24
    fontsize_corr = 40
    batch_size = 1

    for i,data_name in enumerate(data_names):
        save_dir = "result/BRR/"
        os.makedirs(save_dir, exist_ok=True)
        [X,y] = get_dataset.get_dataset(data_name)
        [whole_size, dim] = X.shape
        print(X.shape)
        test_size = 2000
        train_size = min([10000,whole_size - test_size])
        init_sample_size = 10
        sample_size = min([data_sizes[i],train_size - init_sample_size])
        print(train_size)
        X_test = X[:test_size]
        y_test = y[:test_size]
        X_train = X[test_size:test_size + train_size]
        y_train = y[test_size:test_size + train_size]

        basis_size = 10
        x_range = [X_train.min(), X_train.max()]
        brr = active_BRR(beta=0.1,alpha=1.0,basis_size=basis_size,x_range=x_range)
        brr.fit(X_train[:sample_size],y_train[:sample_size],fix_hyper_param=False)
        validate_size = 10
        # threshold = 0.02
        threshold = 0.05
        error_stability1 = error_stability_criterion(threshold,validate_size)
        threshold = 0.04
        # threshold = 0.015
        error_stability2 = error_stability_criterion(threshold,validate_size)
        threshold = 0.03
        # threshold = 0.01
        error_stability3 = error_stability_criterion(threshold,validate_size)
        criteria = [error_stability1,error_stability2,error_stability3]

        pool_indecies = set(range(train_size))
        sampled_indecies = set(random.sample(pool_indecies, init_sample_size))
        pool_indecies = list(pool_indecies - sampled_indecies)
        sampled_indecies = list(sampled_indecies)
        test_error = np.empty(0,float)

        brr.fit(X_train[sampled_indecies], y_train[sampled_indecies],fix_hyper_param=True)
        color = {error_stability1.criterion_name: "r",error_stability2.criterion_name: "g",error_stability3.criterion_name: "b"}
        epsilon = 0
        for e in tqdm(range(sample_size)):
            # new_data_index = random.sample(pool_indecies,1)[0]
            new_data_index = brr.data_aquire(X_train,pool_indecies)
            sampled_indecies.append(new_data_index)
            pool_indecies.remove(new_data_index)

            X_sampled = X_train[sampled_indecies]
            y_sampled = y_train[sampled_indecies]

            pos_old = brr.get_pos()
            brr.fit(X_sampled, y_sampled)
            # brr.fit(X_sampled, y_sampled, fix_hyper_param=False)
            pos_new = brr.get_pos()
            error = utils.calc_expected_squre_error(y_test, brr.predict(X_test, return_std=True))
            test_error = np.append(test_error, error)

            KL_pq = utils.calcKL_gauss(pos_old, pos_new, epsilon)
            KL_qp = utils.calcKL_gauss(pos_new, pos_old, epsilon)

            error_stability1.check_threshold(KL_pq, KL_qp, e)
            error_stability2.check_threshold(KL_pq, KL_qp, e)
            error_stability3.check_threshold(KL_pq, KL_qp, e)

        draw_result.draw_gene_error(test_error, criteria, init_sample_size,batch_size,color, fontsize)
        plt.tight_layout()
        plt.savefig(save_dir + "BRR_gene_error_" + data_name + ".pdf")
        draw_result.draw_correlation(test_error[validate_size:], error_stability1.error_ratio[validate_size:], "BRR", "g", fontsize_corr)
        plt.tight_layout()
        plt.savefig(save_dir + "BRR_correlation_" + data_name + ".pdf")
        draw_result.draw_epsilon(criteria, init_sample_size,batch_size,color, fontsize)
        plt.tight_layout()
        plt.savefig(save_dir + "BRR_criterion_" + data_name + ".pdf")

        indecies = utils.calc_min_list(error_stability1.error_ratio[validate_size:])
        corr_list[i] = np.corrcoef(test_error[validate_size:][indecies], error_stability1.error_ratio[validate_size:][indecies])[1, 0]

    np.savetxt(save_dir+"corr_list.txt", corr_list)
    np.savetxt(save_dir+"loss.txt",np.array(test_error))
    np.savetxt(save_dir+"lambda.txt",error_stability1.error_ratio)

if __name__ == "__main__":
    main()
