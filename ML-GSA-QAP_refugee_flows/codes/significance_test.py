import pandas as pd
import os
import re
import numpy as np
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import LeavePGroupsOut
from sklearn.model_selection import KFold
# from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
# from xgboost import XGBRegressor
from itertools import combinations
from sklearn.metrics import r2_score
# import timeit
import matplotlib.pyplot as plt
import matplotlib


"""

Calculate significance of the original S based on the distribution of the permutations 

"""

matplotlib.use('Agg')


def create_folders(study_period_list, folder):
    path = folder + '/figures_S/'
    for year in study_period_list:
        if not os.path.exists(path + str(year)):
            os.makedirs(path + str(year))
    return


def recalc_close_indexes(data, var):
    tmp = data.filter(regex='\S*' + var + '\S*').drop(var, axis=1)
    tmp2 = tmp.sub(data[var], axis=0)
    return tmp2


# As it is SEff, subtracting first-order from second-order sensitivity indexes
# folder = 'LHS_4_4'
folder = 'no_arms'
names = 'all_vars'
n_vars = 15

# ori_s_indexes = pd.read_csv(folder + '/original/s_indexes_0_original' + names + '.csv', index_col=0)
ori_s_indexes = pd.read_csv(folder + '/original/s_indexes_1_' + names + '.csv', index_col=0)
study_period = ori_s_indexes.index.to_list()
variables = ori_s_indexes.columns.to_list()
dct = {var: [] for var in variables}

create_folders(study_period, folder)

for var in variables[0:n_vars]:
    bool_list = []
    [bool_list.append(False) for i in range(n_vars)]  # Create a mask to identify interactions that include a variable
    [bool_list.append(boolean) for boolean in ori_s_indexes.columns.str.contains(var)[n_vars:]]
    tmp = recalc_close_indexes(ori_s_indexes, var)
    ori_s_indexes.loc[:, bool_list] = tmp
    ori_s_indexes.clip(lower=0, inplace=True)

for year in study_period:
    qap_s_indexes = pd.read_csv(folder + '/qap_years/' + str(year) + 's_indexes_.csv', index_col=0)
    for var in variables[0:n_vars]:
        bool_list = []
        [bool_list.append(False) for i in range(n_vars)]
        [bool_list.append(boolean) for boolean in qap_s_indexes.columns.str.contains(var)[n_vars:]]
        tmp = recalc_close_indexes(qap_s_indexes, var)
        qap_s_indexes.loc[:, bool_list] = tmp
        qap_s_indexes.clip(lower=0, inplace=True)

    for var in variables:
        fig, ax = plt.subplots()
        plt.title('S ' + var + ' ' + str(year))
        ax2 = ax.twinx()
        n, bins, patches = ax.hist(qap_s_indexes[var],
                                   bins=100,
                                   density=True)  # TODO: add stuff
        ax.plot([ori_s_indexes.loc[year][var], ori_s_indexes.loc[year][var]],
                 [0, max(n)], color='red')
        n, bins, patches = ax2.hist(qap_s_indexes[var],
                                   bins=100,
                                   cumulative=True,
                                   histtype='step',
                                   density=True,
                                   color='tab:orange')  # TODO: add stuff
        ax.set_xlabel('S')
        ax.set_ylabel('Frequency PDF')
        ax2.set_ylabel('CDF')
        values = qap_s_indexes[var].to_list()
        values.append(ori_s_indexes.loc[year][var])
        # ax.set_xlim((ax.get_xlim()[0], max(values)))
        plt.savefig(folder + '/figures_S/' + str(year) + '/' + var + '_' + str(year) + '_qap_dist.png')

        dct[var].append(sum(qap_s_indexes[var] < ori_s_indexes.loc[year][var]) / len(qap_s_indexes[var]))

ori_s_indexes.to_csv(folder + '/intermediate_results/recalc_s_indexes_1_original' + names + '.csv')
p_value_df = pd.DataFrame(dct)
p_value_df.index = study_period
p_value_df.to_csv(folder + '/intermediate_results/' + 'sig_s_indexes_.csv')

# for var in variables:
#     plt.figure(var)
#     plt.title(var)
#     plt.xlabel('year')
#     plt.ylabel('Fraction of lower S')
#     plt.ylim(0, 1.05)
#     plt.plot(study_period, dct[var])
#     plt.savefig(folder + '/figures_S/s_series/' + str(var) + '_kind_of_p.png')


# Plot time series of all S
# for var in ori_s_indexes.columns.to_list():
#     plt.figure(var)
#     plt.title(var)
#     ori_s_indexes[var].plot()
#     plt.savefig(folder + '/tmp/' + var + '.png')
#


##### p-value, tails significance

names = 'all_vars'
alpha = [0.05, 0.1]  # TODO: increase number of QAP runs, due to alpha = 0.001

left_right_sig = p_value_df.copy()

# TODO: a two tailed test would still be significant at alpha not at alpha/2. Replace a/2 for a
b = 1.01
for a in alpha:
    left_right_sig[(left_right_sig > 1 - a/2) & (left_right_sig < b)] = 1 - a/2  # TODO: Correct this
    b = 1 - a/2

b = - 0.01
for a in alpha:
    left_right_sig[(left_right_sig < a/2) & (left_right_sig > b)] = a/2
    b = a/2

left_right_sig[(left_right_sig > max(alpha)/2) & (left_right_sig < 1 - max(alpha)/2)] = 0.5  # TODO: Correct this
# np.unique(left_right_sig.values)

left_right_sig.to_csv(folder + '/intermediate_results/left_right_sig.csv')

