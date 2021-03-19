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

GP_corr_list = np.loadtxt("result/GPR/corr_list.txt")
BRR_corr_list = np.loadtxt("result/BRR/corr_list.txt")
BLR_corr_list = np.loadtxt("result/BLR/corr_list.txt")
BDNN_corr_list = np.array([np.loadtxt("result/BDNN/corr.txt")])
corr_list = [GP_corr_list,BRR_corr_list,BLR_corr_list,BDNN_corr_list]
draw_result.draw_result_list(corr_list,"correlation",data_name_lists,color,models,fontsize,1)
plt.savefig("result/correlation_error.pdf", bbox_inches="tight")
