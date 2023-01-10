import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import sys
import itertools


def rep_inputs(ori_matrix, output_matrix, reps, drop=False):
    """
    Creates a GSA matrix with repeated set of inputs
    :param ori_matrix:
    :param output_matrix:
    :param reps:
    :param drop:
    :return:
    """
    ori_matrix = open(ori_matrix, 'r')
    matrix_rep = open(output_matrix, 'w')
    matrix_rep.close()
    matrix_rep = open(output_matrix, 'a')
    for line in ori_matrix:
        line2 = line
        matrix_rep.write(line2*reps)
    matrix_rep.close()
    ori_matrix.close()
    return


def var_dis(data, n_reps, file_name=0):
    """
    Calculates v_dis for each input set, Mean V_dis for each output, S_dis (optional) for each Output
    V[Y] = E[V[Y|X]] + V[E[Y|X]] = V_dis + V_exp
    :param data: DataFrame of outputs repeated N times
    :param n_reps: Times each output is repeated
    :param file_name: Name of the output, add path to it, if =0 does not write a file
    :return: V_dis & Mean V_dis for each output in DataFrame format
    """
    vy = np.var(data)
    v_dis = pd.DataFrame(np.zeros((int(data.shape[0]/n_reps), data.shape[1])), columns=data.columns)
    for i in range(v_dis.shape[0]):
        r = np.linspace(0 + i*n_reps, n_reps - 1 + i*n_reps, n_reps)
        for j in data.columns:
            v_dis[j].iloc[i] = np.var(data[j].iloc[r])
    m_v_dis = v_dis.mean()
    s_dis = m_v_dis / vy
    if file_name != 0:
        v_dis.to_csv(file_name, index=False)
    return v_dis, m_v_dis, s_dis


def var_exp(data, n_reps, file_name=0):
    """
    Calculates Expected value for each input set, Vpar for each output, Spar (optional) for each Output and saves a file
    with the expected values of each input set
    V[Y] = E[V[Y|X]] + V[E[Y|X]] = Verr + Vpar
    :param data: DataFrame of outputs repeated N times
    :param n_reps: Times each output is repeated
    :param file_name: Name of the output, add path to it, if =0 does not write a file
    :return:
    """
    vy = np.var(data)
    exp_p = pd.DataFrame(np.zeros((int(data.shape[0] / n_reps), data.shape[1])), columns=data.columns)  # Expected value of the repetitions of set of inputs
    for i in range(exp_p.shape[0]):
        r = np.linspace(0 + i * n_reps, n_reps - 1 + i * n_reps, n_reps)
        for j in data.columns:
            exp_p[j].iloc[i] = np.mean(data[j].iloc[r])
    v_exp = exp_p.var(ddof=0)
    s_exp = v_exp/vy
    if file_name != 0:
        exp_p.to_csv(file_name, index=False)
    return exp_p, v_exp, s_exp



def var_exp(data, n_reps, file_name=0):
    """
    Calculates Expected value for each input set, Vpar for each output, Spar (optional) for each Output and saves a file
    with the expected values of each input set
    V[Y] = E[V[Y|X]] + V[E[Y|X]] = Verr + Vpar
    :param data: DataFrame of outputs repeated N times
    :param n_reps: Times each output is repeated
    :param file_name: Name of the output, add path to it, if =0 does not write a file
    :return:
    """
    vy = np.var(data)
    exp_p = pd.DataFrame(np.zeros((int(data.shape[0] / n_reps), data.shape[1])), columns=data.columns)  # Expected value of the repetitions of set of inputs
    for i in range(exp_p.shape[0]):
        r = np.linspace(0 + i * n_reps, n_reps - 1 + i * n_reps, n_reps)
        for j in data.columns:
            exp_p[j].iloc[i] = np.mean(data[j].iloc[r])
    v_exp = exp_p.var(ddof=0)
    s_exp = v_exp/vy
    if file_name != 0:
        exp_p.to_csv(file_name, index=False)
    return exp_p, v_exp, s_exp


def subsample(outputs, n_rep, sub_sample):
    extraction = np.array(np.random.choice(n_rep, sub_sample, replace=False))
    ext = np.zeros(0)
    for i in range(int(outputs.shape[0]/n_rep)):
        tmp = extraction + i * n_rep
        ext = np.append(ext, tmp)
    out = outputs.loc[ext]
    out = out.reset_index(drop=1)
    return out
