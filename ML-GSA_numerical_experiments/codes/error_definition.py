import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from random import sample
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import copy
from GSA_call import GSA, ST_GSA
from test_functions import ishigami_add, ishigami_mult
from test_functions import sobol_g_add, sobol_g_mult
import seaborn as sns
from analytical_solutions import compute_ishigami_si, compute_ishigami_sij, compute_ishigami_sti
from analytical_solutions import compute_sobol_g_si, compute_sobol_g_sij, compute_sobol_g_sti
from analytical_solutions import sobol_y_var, ishigami_y_var
from functools import reduce
from operator import iconcat
import pickle


def sobol_y_var(a_list):
    vy = 1
    for a_i in a_list:
        gi = 1 + (1/3)/(1 + a_i)**2
        vy = vy * gi
    return vy - 1


def ishigami_y_var(a, b):
    vy = 1/2 + a**2/8 + b*np.pi**4/5 + b**2*np.pi**8/18
    return vy


funct = ishigami_mult
n_t = 10000
s_e = 1
n_test = 5000

# a_list = [0,0,0, 0.5,9,9, 9,9,9,9,9]
a_list = [0, 0, 0, 3, 3, 9, 9, 9, 9, 9]
# a_list = [0, 1, 3]
a, b = 3, 2

r2_list = []
var_error_list = []
var_ratio_list = []

for iteration in range(20):
    if (funct == ishigami_add) or (funct == ishigami_mult):
        funct_name = 'ishigami'
        k = 3
        v_y = ishigami_y_var(a, b)
        if funct == ishigami_add:
            error_type = 'additive'
        else:
            error_type = 'multiplicative'
    elif (funct == sobol_g_add) or (funct == sobol_g_mult):
        funct_name = 'sobolG'
        k = len(a_list)
        v_y = sobol_y_var(a_list)
        if funct == sobol_g_add:
            error_type = 'additive'
        else:
            error_type = 'multiplicative'

    df = pd.DataFrame()
    if funct_name == 'ishigami':
        df = np.random.uniform(low=-np.pi, high=np.pi, size=(n_t + n_test) * k)
    if funct_name == 'sobolG':
        df = np.random.uniform(low=0, high=1, size=(n_t + n_test) * k)
    df = df.reshape([(n_t + n_test), k])
    df = pd.DataFrame(df)
    if funct_name == 'sobolG':
        df['y'] = funct(a_list, df.values, s_e=s_e, seed=int(np.random.uniform(0, 1000)))
    if funct_name == 'ishigami':
        df['y'] = funct(df.values, s_e=s_e, a=a, b=b, seed=int(np.random.uniform(0, 1000)))

    in_train = sample(range(df.shape[0]), n_t)
    train = df.loc[in_train].reset_index(drop=True)
    test = df.drop(in_train, axis=0).reset_index(drop=True)

    # RANDOM FOREST
    rg_model = RandomForestRegressor(n_estimators=200)  # TODO: tune the model
    rg_model.fit(train.drop('y', axis=1), train['y'])
    r2_list.append(r2_score(rg_model.predict(test.drop('y', axis=1)), test['y']))
    var_error_list.append(np.var(df['y']) - v_y)
    var_ratio_list.append((np.var(df['y']) - v_y) / v_y)

print('R2', r2_list,
      '\n Mean R2', np.mean(r2_list),
      '\n Mean V_error', np.mean(var_error_list),
      '\n V_ratio', var_ratio_list,
      '\n Mean V_ratio', np.mean(var_ratio_list))

# In the additive case the total variance equals the sum of the error variance and the function variance
# print(np.var(np.random.uniform(0, 3, 100000)) + np.var(np.random.normal(0, 2, 100000)))
# print(np.var(np.random.uniform(0, 3, 100000) + np.random.normal(0, 2, 100000)))

# In the multiplicative case the total variance is dependent of the distribution of y
# sddd = 0.35
# print(sddd**2 / np.var(np.random.normal(0, 3, 100000) * np.random.normal(1, sddd, 100000)))
# print(sddd**2 / np.var(np.random.uniform(0, 3, 100000) * np.random.normal(1, sddd, 100000)))
# r = 0.49

# sigma_e = (r * v_y / (1 - r))**0.5
# print(sigma_e)
