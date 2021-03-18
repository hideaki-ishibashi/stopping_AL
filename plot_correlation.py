import matplotlib.pylab as plt
import utils
import draw_result
import get_dataset
from tqdm import tqdm
import random
import os
import numpy as np


fontsize = 18

color = ["r","g","b","purple"]
models = ["GPR","BRR","BLR","BDNN"]
data_name_lists = [["power_plant","protein","gas_emission","grid_stability"],["power_plant","protein","gas_emission","grid_stability"],["grid_stability","skin","HTRU2"], ["MNIST"]]

# for i, model in enumerate(models):
#     data_name_list = data_name_lists[i]
#     for j, data_name in enumerate(data_name_list):
#         BDNN_loss = np.loadtxt("result/BDNN/loss.txt")
#         BDNN_criterion = np.loadtxt("result/BDNN/lambda.txt")
#         BDNN_criterion = utils.calc_min_list(BDNN_criterion)
#         draw_result.draw_correlation(BDNN_loss,BDNN_criterion,"BDNN","purple")
#         plt.savefig("result/BDNN/BDNN_correlation_MNIST.pdf", bbox_inches="tight")

# BDNN_loss = np.loadtxt("result/BDNN/loss.txt")
# BDNN_criterion = np.loadtxt("result/BDNN/lambda.txt")
# BDNN_criterion = utils.calc_min_list(BDNN_criterion)
# draw_result.draw_colleration(BDNN_loss,BDNN_criterion,"BDNN","cyan")
# plt.savefig("result/BDNN/BDNN_correlation_MNIST.pdf", bbox_inches="tight")

GP_corr_list = np.loadtxt("result/GPR/corr_list.txt")
BRR_corr_list = np.loadtxt("result/BRR/corr_list.txt")
BLR_corr_list = np.loadtxt("result/BLR/corr_list.txt")
BDNN_corr_list = np.array([np.loadtxt("result/BDNN/corr.txt")])
corr_list = [GP_corr_list,BRR_corr_list,BLR_corr_list,BDNN_corr_list]
draw_result.draw_result_list(corr_list,"correlation",data_name_lists,color,models,fontsize,1)
plt.savefig("result/correlation_error.pdf", bbox_inches="tight")

# GP_min_list = np.loadtxt("result/GPr/azuma_min_list.txt")
# BRR_min_list = np.loadtxt("result/BRR/azuma_min_list.txt")
# BLR_min_list = np.loadtxt("result/BLR/azuma_min_list.txt")
# BDNN_min_list = np.array([np.loadtxt("result/BDNN/azuma_min.txt")])
# min_list = [GP_min_list,BRR_min_list,BLR_min_list,BDNN_min_list]
# draw_result.draw_result_list(min_list,"minimum value",data_name_lists,color,models,fontsize,0.2)
# plt.savefig("result/min_error.pdf", bbox_inches="tight")

