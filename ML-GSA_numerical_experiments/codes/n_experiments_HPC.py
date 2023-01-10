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
from sklearn.inspection import permutation_importance


"""
ANALYSIS DEFINITION 
"""
# funct = sobol_g_add
funct = ishigami_add
order = 2  # Maximum order of sensitivity indexes


"""
PARAMETER DEFINITION
"""
# s_error Moved to the function conditionals
n_train = [50, 100, 200, 400, 800, 1600, 3200, 4800]  # Train sample sizes
# n_train = [200]  # Train sample sizes
n_test = 200  # Test sample size

n_rea = 100  # Number of experiment realizations (points in the boxplots)
n_gsa = 6000  # GSA sampling
order = 2  # GSA maximum order of sensitivity indexes

a, b = 3, 2  # Ishigami parameters
a_list = [0, 0, 0, 3, 3, 9, 9, 9, 9, 9]

ratio_vye = [0, 0.25, 0.5, 1]  # Variance of the error / (Variance of y)
if (funct == ishigami_add) or (funct == ishigami_mult):
    funct_name = 'ishigami'
    k = 3
    if funct == ishigami_add:
        error_type = 'additive'
        s_error = [0, 23.17, 32.78, 46.38]
    else:
        error_type = 'multiplicative'
        s_error = [0, 0.5, 0.71, 1]
elif (funct == sobol_g_add) or (funct == sobol_g_mult):
    funct_name = 'sobolG'
    k = len(a_list)
    if funct == sobol_g_add:
        error_type = 'additive'
        s_error = [0, 0.62, 0.87, 1.23]
    else:
        error_type = 'multiplicative'
        s_error = [0, 0.39, 0.55, 0.78]

input_names = []
for i in range(k):
    input_names.append('X' + str(i + 1))

"""
INITIALIZATION
"""
# TODO: change rf_add, xgb_add, xgb_mult & rf_mult to common dictionaries so I rerun the analysis for each error type
# TODO: need to do that all across the code (especially, when adding the no error dictionaries to the other ones
# No error nested dictionaries
no_e_dict = {str(n_t): [] for n_t in n_train}
rf_no_e = {'S': copy.deepcopy(no_e_dict),
           'R2': copy.deepcopy(no_e_dict),
           'imp': copy.deepcopy(no_e_dict),
           'ST': copy.deepcopy(no_e_dict),
           'Perm_imp': copy.deepcopy(no_e_dict)}
xgb_no_e = copy.deepcopy(rf_no_e)
# Additive error nested dictionaries (size of training + '_' + ratio vye)
e_dict = {str(n_t) + '_' + str(s_e_ratio): [] for n_t in n_train for s_e_ratio in ratio_vye[1:]}
rf_add = {'S': copy.deepcopy(e_dict),
          'R2': copy.deepcopy(e_dict),
          'imp': copy.deepcopy(e_dict),
          'ST': copy.deepcopy(e_dict),
          'Perm_imp': copy.deepcopy(e_dict)}
xgb_add = copy.deepcopy(rf_add)
# Multiplicative error nested dictionaries (size of training + '_' + ratio vye)
rf_mult = copy.deepcopy(rf_add)
xgb_mult = copy.deepcopy(rf_add)

"""
Numerical experiments with ADDITIVE ERROR   
ISHIGAMI FUNCTION
"""
for i in range(n_rea):
    for n_t in n_train:  # n_t: size for training
        for s_e_i, s_e in enumerate(s_error):  # s_e: standard deviation of the error distribution
            s_e_ratio = ratio_vye[s_e_i]
            print(i, n_t, s_e)
            # np.random.seed(i)
            if funct_name == 'ishigami':
                df = np.random.uniform(low=-np.pi, high=np.pi, size=(n_t + n_test) * k)
            if funct_name == 'sobolG':
                df = np.random.uniform(low=0, high=1, size=(n_t + n_test) * k)
            df = df.reshape([(n_t + n_test), k])
            df = pd.DataFrame(df, columns=input_names)
            if funct_name == 'sobolG':
                df['y'] = funct(a_list, df.values, s_e=s_e)
            if funct_name == 'ishigami':
                df['y'] = funct(df.values, s_e=s_e, a=a, b=b)
            # np.random.seed(0)  # TODO: add seed
            in_train = sample(range(df.shape[0]), n_t)
            train = df.loc[in_train].reset_index(drop=True)
            test = df.drop(in_train, axis=0).reset_index(drop=True)

            # RANDOM FOREST
            rg_model = RandomForestRegressor(n_estimators=200)  # TODO: tune the model
            rg_model.fit(train.drop('y', axis=1), train['y'])
            r2_score(rg_model.predict(test.drop('y', axis=1)), test['y'])
            # plt.hist(df['y'])
            # rg_model.feature_importances_
            if s_e == 0:
                rf_no_e['imp'][str(n_t)].append(rg_model.feature_importances_.tolist())
                rf_no_e['R2'][str(n_t)].append(r2_score(rg_model.predict(test.drop('y', axis=1)), test['y']))
                rf_no_e['S'][str(n_t)].append(GSA(rg_model, n_gsa, funct_name, order, k))
                rf_no_e['ST'][str(n_t)].append(ST_GSA(rg_model, n_gsa, funct_name, k))
                rf_no_e['Perm_imp'][str(n_t)].append(permutation_importance(rg_model, test.drop('y', axis=1), test['y'],
                                                                            n_repeats=10))

            else:
                rf_add['imp'][str(n_t) + '_' + str(s_e_ratio)].append(rg_model.feature_importances_.tolist())
                rf_add['R2'][str(n_t) + '_' + str(s_e_ratio)].append(
                    r2_score(rg_model.predict(test.drop('y', axis=1)), test['y']))
                rf_add['S'][str(n_t) + '_' + str(s_e_ratio)].append(GSA(rg_model, n_gsa, funct_name, order, k))
                rf_add['ST'][str(n_t) + '_' + str(s_e_ratio)].append(ST_GSA(rg_model, n_gsa, funct_name, k))
                rf_add['Perm_imp'][str(n_t) + '_' + str(s_e_ratio)].append(permutation_importance(
                    rg_model, test.drop('y', axis=1), test['y'], n_repeats=10))

            # # XGBoost
            rg_model = XGBRegressor()  # TODO: tune the model
            rg_model.fit(train.drop('y', axis=1), train['y'])
            # r2_score(rg_model.predict(test.drop('y', axis=1)), test['y'])
            # rg_model.feature_importances_
            if s_e == 0:
                xgb_no_e['imp'][str(n_t)].append(rg_model.feature_importances_.tolist())
                xgb_no_e['R2'][str(n_t)].append(r2_score(rg_model.predict(test.drop('y', axis=1)), test['y']))
                xgb_no_e['S'][str(n_t)].append(GSA(rg_model, n_gsa, funct_name, order, k))
                xgb_no_e['ST'][str(n_t)].append(ST_GSA(rg_model, n_gsa, funct_name, k))
                xgb_no_e['Perm_imp'][str(n_t)].append(permutation_importance(
                    rg_model, test.drop('y', axis=1), test['y'], n_repeats=10))
            else:
                xgb_add['imp'][str(n_t) + '_' + str(s_e_ratio)].append(rg_model.feature_importances_.tolist())
                xgb_add['R2'][str(n_t) + '_' + str(s_e_ratio)].append(
                    r2_score(rg_model.predict(test.drop('y', axis=1)), test['y']))
                xgb_add['S'][str(n_t) + '_' + str(s_e_ratio)].append(GSA(rg_model, n_gsa, funct_name, order, k))
                xgb_add['ST'][str(n_t) + '_' + str(s_e_ratio)].append(ST_GSA(rg_model, n_gsa, funct_name, k))
                xgb_add['Perm_imp'][str(n_t) + '_' + str(s_e_ratio)].append(permutation_importance(
                    rg_model, test.drop('y', axis=1), test['y'], n_repeats=10))

with open('figures_' + funct_name + '/' + error_type + '/dictionaries/xgb_add.pkl', 'wb') as f:
    pickle.dump(xgb_add, f)

with open('figures_' + funct_name + '/' + error_type + '/dictionaries/xgb_no_e.pkl', 'wb') as f:
    pickle.dump(xgb_no_e, f)

with open('figures_' + funct_name + '/' + error_type + '/dictionaries/rf_add.pkl', 'wb') as f:
    pickle.dump(rf_add, f)

with open('figures_' + funct_name + '/' + error_type + '/dictionaries/rf_no_e.pkl', 'wb') as f:
    pickle.dump(rf_no_e, f)

# with open('figures_' + funct_name + '/' + error_type + '/dictionaries/rf_add.pkl', 'rb') as f:
#     rf_add = pickle.load(f)
