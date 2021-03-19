import matplotlib.pylab as plt
import random
from model.active_BLR import active_BLR
from stopping_criteria import *
import utils
import draw_result
import get_dataset
from tqdm import tqdm
import os

def main():
    np.random.seed(1)
    random.seed(1)
    data_names = ["grid_stability_c","skin","HTRU2"]
    corr_list = np.zeros((len(data_names)))

    C_list = [100,0.001,1]
    fontsize = 24
    batch_size = 1

    for i,data_name in enumerate(data_names):
        save_dir = "result/BLR/"
        os.makedirs(save_dir, exist_ok=True)
        [X, y] = get_dataset.get_dataset(data_name)
        [whole_size, dim] = X.shape
        y = y[:,0].astype(int)
        test_size = 5000
        train_size = whole_size - test_size
        init_sample_size = 100
        sample_size = min([3000,train_size - init_sample_size])
        X_test = X[:test_size]
        y_test = y[:test_size]
        X_train = X[test_size:test_size + train_size]
        y_train = y[test_size:test_size + train_size]

        pool_indecies = set(range(train_size))
        sampled_indecies = set(random.sample(pool_indecies, init_sample_size))
        pool_indecies = list(pool_indecies - sampled_indecies)
        sampled_indecies = list(sampled_indecies)

        X_sampled = X_train[sampled_indecies]
        y_sampled = y_train[sampled_indecies]

        basis_size = 5
        x_range = [X_train.min(),X_train.max()]
        blr = active_BLR(basis_size=basis_size,x_range=x_range,C=C_list[i],solver="newton-cg")

        validate_size = 10
        threshold = 0.2
        error_stability1 = error_stability_criterion(threshold,validate_size)
        threshold = 0.15
        error_stability2 = error_stability_criterion(threshold,validate_size)
        threshold = 0.1
        error_stability3 = error_stability_criterion(threshold,validate_size)
        criteria = [error_stability1,error_stability2,error_stability3]

        test_error = np.empty(0,float)
        blr.fit(X_sampled, y_sampled)
        color = {error_stability1.criterion_name: "r",error_stability2.criterion_name: "g",error_stability3.criterion_name: "b"}
        for e in tqdm(range(sample_size)):
            new_data_index = blr.data_acquire(X_train, pool_indecies)
            sampled_indecies.append(new_data_index)
            pool_indecies.remove(new_data_index)

            X_sampled = X_train[sampled_indecies]
            y_sampled = y_train[sampled_indecies]

            pos_old = blr.get_pos()
            blr.fit(X_sampled,y_sampled,blr.coef_[0])
            pos_new = blr.get_pos()
            KL_pq = utils.calcKL_gauss(pos_old, pos_new)
            KL_qp = utils.calcKL_gauss(pos_new, pos_old)
            error = utils.calc_cross_entropy(y_test, blr.predict_proba(X_test)[:, 1])
            test_error = np.append(test_error, error)

            error_stability1.check_threshold(KL_pq, KL_qp, e)
            error_stability2.check_threshold(KL_pq, KL_qp, e)
            error_stability3.check_threshold(KL_pq, KL_qp, e)

        draw_result.draw_gene_error(test_error, criteria, init_sample_size,batch_size,color, fontsize)
        plt.tight_layout()
        plt.savefig(save_dir + "BLR_gene_error_" + data_name + ".pdf")
        draw_result.draw_correlation(test_error[validate_size:], error_stability1.error_ratio[validate_size:], "BLR",  "b", fontsize)
        plt.tight_layout()
        plt.savefig(save_dir + "BLR_correlation_" + data_name + ".pdf")
        draw_result.draw_epsilon(criteria, init_sample_size,batch_size,color, fontsize)
        plt.tight_layout()
        plt.savefig(save_dir + "BLR_criterion_" + data_name + ".pdf")

        indecies = utils.calc_min_list(error_stability1.error_ratio[validate_size:])
        corr_list[i] = np.corrcoef(test_error[validate_size:][indecies], error_stability1.error_ratio[validate_size:][indecies])[1, 0]

    np.savetxt(save_dir+"corr_list.txt", corr_list)

if __name__ == "__main__":
    main()

