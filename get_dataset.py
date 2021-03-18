import numpy as np
import pandas as pd
import random
from sklearn.datasets import *


def get_dataset(data_name):

    # classification
    if data_name == "skin":
        dataset = np.loadtxt("./dataset/skin/Skin_NonSkin.txt")
        input = dataset[:,:-1]
        input = (input - input.mean(axis=0)[np.newaxis,:])/input.std(axis=0)
        output = dataset[:,-1].astype("int")-1
        n_labels = len(np.unique(output))
        output = np.eye(n_labels)[output]
        indecies = random.sample(range(input.shape[0]), input.shape[0])
        return [input[indecies],output[indecies]]

    if data_name == "grid_stability_c":
        df = pd.read_csv('./dataset/electrical_grid_stability/electrical_grid_stability.csv', index_col=0)
        dataset = df.values
        # 最後から2つ目は回帰の予測変数だから取り除く
        input = dataset[:,:-2]
        input = (input - input.mean(axis=0)[np.newaxis,:])/input.std(axis=0)
        output = dataset[:,-1].astype("int")
        n_labels = len(np.unique(output))
        output = np.eye(n_labels)[output]
        indecies = random.sample(range(input.shape[0]), input.shape[0])
        return [input[indecies],output[indecies]]

    if data_name == "HTRU2":
        df = pd.read_csv('./dataset/HTRU2/HTRU_2.csv', index_col=0)
        dataset = df.values
        input = dataset[:,:-1]
        input = (input - input.mean(axis=0)[np.newaxis,:])/input.std(axis=0)
        output = dataset[:,-1].astype("int")-1
        n_labels = len(np.unique(output))
        output = np.eye(n_labels)[output]
        indecies = random.sample(range(input.shape[0]), input.shape[0])
        return [input[indecies],output[indecies]]

    # regression
    if data_name == "grid_stability":
        df = pd.read_csv('./dataset/electrical_grid_stability/electrical_grid_stability.csv', index_col=0)
        dataset = df.values
        # 最後から2つ目は回帰の予測変数だから取り除く
        input = dataset[:,:-2]
        input = (input - input.mean(axis=0)[np.newaxis,:])/input.std(axis=0)
        output = dataset[:,-2]
        indecies = random.sample(range(input.shape[0]), input.shape[0])
        return [input[indecies],output[indecies]]

    if data_name == "power_plant":
        dataset = np.loadtxt("./dataset/power_plant/ccpp.txt")
        input = dataset[:,:-1]
        input = (input - input.mean(axis=0)[np.newaxis,:])/input.std(axis=0)
        output = dataset[:,-1]
        output = (output - output.mean())/output.std()
        indecies = random.sample(range(input.shape[0]), input.shape[0])
        return [input[indecies],output[indecies]]

    if data_name == "gas_emission":
        df = pd.read_csv('./dataset/gas_emission/gt_2011.csv', index_col=0)
        dataset1 = df.values
        df = pd.read_csv('./dataset/gas_emission/gt_2012.csv', index_col=0)
        dataset2 = df.values
        df = pd.read_csv('./dataset/gas_emission/gt_2013.csv', index_col=0)
        dataset3 = df.values
        df = pd.read_csv('./dataset/gas_emission/gt_2014.csv', index_col=0)
        dataset4 = df.values
        df = pd.read_csv('./dataset/gas_emission/gt_2015.csv', index_col=0)
        dataset5 = df.values
        dataset = np.concatenate([dataset1,dataset2,dataset3,dataset4,dataset5],axis=0)
        input = np.concatenate([dataset[:,:7],dataset[:,7:]],axis=1)
        input = (input - input.mean(axis=0)[np.newaxis,:])/input.std(axis=0)
        output = dataset[:,7]
        output = (output - output.mean())/output.std()
        indecies = random.sample(range(input.shape[0]), input.shape[0])
        return [input[indecies],output[indecies]]

    if data_name == "protein":
        dataset = np.loadtxt("./dataset/protein/CASP.txt")
        input = dataset[:,:-1]
        input = (input - input.mean(axis=0)[np.newaxis,:])/input.std(axis=0)
        output = dataset[:,-1]
        output = (output - output.mean())/output.std()
        indecies = random.sample(range(input.shape[0]), input.shape[0])
        return [input[indecies],output[indecies]]

