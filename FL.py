#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os


"""UPLOADING THE DATASETS"""
import sys 

print(
    "dataset - sampling - sim_type - seed - n_SGD - lr - decay - p - force - mu"
)
print(sys.argv[1:]) 

dataset = sys.argv[1]
sampling = sys.argv[2]
sim_type = sys.argv[3]
seed = int(sys.argv[4])
n_SGD = int(sys.argv[5])
lr = float(sys.argv[6])
decay = float(sys.argv[7])
p = float(sys.argv[8])
force = sys.argv[9] == "True"

try:
    mu = float(sys.argv[10])
except:
    mu = 0.0


"""GET THE HYPERPARAMETERS"""
from py_func.hyperparams import get_hyperparams # 如果要从其他python文件中导入函数，就需要加上py_func.hyperparams

n_iter, batch_size, meas_perf_period = get_hyperparams(dataset, n_SGD)
print("number of iterations", n_iter)
print("batch size", batch_size)
print("percentage of sampled clients", p)
print("metric_period", meas_perf_period)
print("regularization term", mu)


"""NAME UNDER WHICH THE EXPERIMENT'S VARIABLES WILL BE SAVED"""
from py_func.hyperparams import get_file_name

file_name = get_file_name(
    dataset, sampling, sim_type, seed, n_SGD, lr, decay, p, mu
)
print(file_name)


"""GET THE DATASETS USED FOR THE FL TRAINING"""
from py_func.read_db import get_dataloaders

list_dls_train, list_dls_test = get_dataloaders(dataset, batch_size) # 从read.db函数中调用get_dataloaders，然后读出来list_dls_train, list_dls_test这两个变量


"""NUMBER OF SAMPLED CLIENTS"""
n_sampled = int(p * len(list_dls_train)) # 计算出抽样参与方的数量
print("number fo sampled clients", n_sampled)


"""LOAD THE INTIAL _GLOBAL MODEL"""
from py_func.create_model import load_model  # 调用create_model函数中的load_model 就是加载函数的函数

model_0 = load_model(dataset, seed)
print(model_0)

## by zzy begin 
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Experiment running on device: {}".format(device))
model_0.to(device=device)
## by zzy end

"""FEDAVG with random sampling"""
if sampling == "random" and (
    not os.path.exists(f"saved_exp_info/acc/{file_name}.pkl") or force
): 

    from py_func.FedProx import FedProx_sampling_random 

    FedProx_sampling_random(
        model_0,
        n_sampled,
        list_dls_train,
        list_dls_test,
        n_iter,
        n_SGD,
        lr,
        file_name,
        decay,
        meas_perf_period,
        mu,
    )


"""Run FEDAVG with clustered sampling"""
if (sampling == "clustered_1" or sampling == "clustered_2") and (
    not os.path.exists(f"saved_exp_info/acc/{file_name}.pkl") or force
):

    from py_func.FedProx import FedProx_clustered_sampling # 针对不同的sampling，导入不同的函数

    FedProx_clustered_sampling(
        sampling,
        model_0,
        n_sampled,
        list_dls_train,
        list_dls_test,
        n_iter,
        n_SGD,
        lr,
        file_name,
        sim_type,
        0,
        decay,
        meas_perf_period,
        mu,
    )


"""RUN FEDAVG with perfect sampling for MNIST-shard"""
if (
    sampling == "perfect"
    and dataset == "MNIST_shard"
    and (not os.path.exists(f"saved_exp_info/acc/{file_name}.pkl") or force)
):

    from py_func.FedProx import FedProx_sampling_target

    FedProx_sampling_target(
        model_0,
        n_sampled,
        list_dls_train,
        list_dls_test,
        n_iter,
        n_SGD,
        lr,
        file_name,
        decay,
        mu,

    )


"""RUN FEDAVG with its original sampling scheme sampling clients uniformly"""
if sampling == "FedAvg" and (
    not os.path.exists(f"saved_exp_info/acc/{file_name}.pkl") or force
):

    from py_func.FedProx import FedProx_FedAvg_sampling

    FedProx_FedAvg_sampling(
        model_0,
        n_sampled,
        list_dls_train,
        list_dls_test,
        n_iter,
        n_SGD,
        lr,
        file_name,
        decay,
        meas_perf_period,
        mu,
    )


print("EXPERIMENT IS FINISHED") # 实验结束
