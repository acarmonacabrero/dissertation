import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, KFold, cross_val_score, train_test_split
from itertools import combinations
from sklearn.metrics import r2_score
import subprocess
from math import floor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# plt.ioff()
lag = 0

def log_transformation(x, transf):
    if transf == 'natural_log_transformation':
        return np.log(x + 1)
    elif transf == 'log10_transformation':
        return np.log10(x + 1)
    else:
        return x


def feature_filter(data, year, lag=-1, prior_ref=True):
    # Using year - 1 as predictor. Ideally, a lag per variable
    selected_columns = data.filter(regex=(str(year + lag))).columns.to_list()
    try:
        selected_columns.remove('immigrant_population_' + str(year + lag))
    except:
        pass
    immigrant_pop_year = min(int(round((year + lag)/10, 0) * 10), 2010)
    selected_columns.append('immigrant_population_' + str(immigrant_pop_year))
    selected_columns.append('contiguity_any')
    if not prior_ref:
        selected_columns.remove('ref_flow_' + str(year + lag))
    [selected_columns.append(item) for item in ['state_destination_name', 'state_origin_name']]
    if year > 2015:
        selected_columns.append('trade_2014')
    x = data[sorted(selected_columns)]
    y = data['ref_flow_' + str(year)]
    return x, y


def threshold_filter(X, y, threshold=1):
    if threshold != 0:
        X_y = X.copy()
        X_y['target'] = y
        X_y = X_y[X_y['target'] >= threshold]
        X_y.reset_index(inplace=True, drop=True)
        y = X_y['target']
        X = X_y.drop('target', axis=1)
        return X, y
    else:
        return X, y

# NOT USING PARSER FOR JUST ORIGINAL DATA
# parser = argparse.ArgumentParser()
# parser.add_argument("--run_id", "-id", help="run id. Used to identify output names")
# parser.add_argument('--sam_method', '-sam', help='LHS or RS')
# parser.add_argument('--n_gsa_runs', 'n_gsa', help='definition of gsa size. Different for LHS or RS')
# Read arguments from the command line
# args = parser.parse_args()
# run_id = int(args.run_id)
# sam_method = args.sam_method
# n_gsa_runs = args.n_gsa_runs

# Analysis definition
run_id = 1  # This cannot be 0
year_seed = 1004  # TODO: REMOVE THIS WHEN NECESSARY
sam_method = 'eRS'
n_gsa_runs = 15000  # GSA sample size
nruns = 1
threshold = 0
k = 10  # Number of folds
# TODO: line above


# Analysis definition
data = pd.read_csv('data/data_for_analysis.csv')


kfold = KFold(n_splits=k, shuffle=True, random_state=0)
# grid_search = GridSearchCV(model, param_grid, scoring=["rmse", "r2"], n_jobs=-1, cv=kfold)

params = {'n_estimators': [100, 200, 300],
          'learning_rate': [0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
          'max_depth': [2, 3, 4, 5, 6, 8, 10],
          'min_child_weight': [1, 3, 5, 7],
          'gamma': [0.0, 0.05, 0.1, 0.2, 0.3, 0.4],
          'reg_lambda': [0, 1, 3, 5, 7, 9, 11],
          'colsample_bytree': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
          }

names = 'all_vars'
cv_metric = 'r2'

# study_period = range(1991, 2017)
study_period = range(1995, 2016)
# TODO: line above

# Initialization
feature_importance = []
cv_score = []
mse_cv_score = []
r2_modelfit = []
params_stack = []
first_s_indexes = pd.DataFrame()
bias_si = pd.DataFrame()
min_ci_si = pd.DataFrame()
max_ci_si = pd.DataFrame()
second_s_indexes = pd.DataFrame()
bias_sij = pd.DataFrame()
min_ci_sij = pd.DataFrame()
max_ci_sij = pd.DataFrame()
transf = 'log10_transformation'
for year in study_period:
    # Data Preparation
    X, y = feature_filter(data, year, lag=lag, prior_ref=False)  # No time lag, no prior ref
    X.drop('arms_inverse_' + str(year), axis=1, inplace=True)
    # No prior refugee flow
    # X_ref_transformed = X['ref_flow_' + str(year - 1)].apply(lambda x: log_transformation(x, transf))
    # X = X.drop('ref_flow_' + str(year - 1), axis=1)
    # X['ref_flow_' + str(year - 1)] = X_ref_transformed
    # del X_ref_transformed
    y = y.apply(lambda x: log_transformation(x, transf))
    X = X.drop(['state_destination_name', 'state_origin_name'], axis=1)
    X, y = threshold_filter(X, y, threshold=log_transformation(threshold, transf))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
    data_ = X.copy()
    data_['target'] = y
    data_.to_csv('year_permutation.csv', index=False)
    del data_
    # Model fit and model statistics
    # TODO: Change n_iter
    rg_model = RandomizedSearchCV(XGBRegressor(),
                               param_distributions=params,
                               n_iter=2,
                               scoring='neg_root_mean_squared_error',
                               n_jobs=-1,
                               cv=5,
                               verbose=0,
                               random_state=year + year_seed)
    rg_model.fit(X, y)
    # model.best_estimator_
    # model.best_score_
    model = XGBRegressor(**rg_model.best_params_)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=True)

    r2_modelfit.append(r2_score(y, model.predict(X)))
    cv_score.append(cross_val_score(estimator=model, X=X_test, y=y_test, cv=kfold,
                                    scoring=cv_metric))  # 'neg_root_mean_squared_error'
    mse_cv_score.append(cross_val_score(estimator=model, X=X_test, y=y_test, cv=kfold,
                                         scoring='neg_root_mean_squared_error'))
    feature_importance.append(dict(zip([feature[:-5] for feature in X.columns.to_list()],
                                       model.feature_importances_)))
    params_stack.append(dict(zip(model.get_params().keys(),
                                 model.get_params().values())))

    # FIRST ORDER: Create GSA sample
    subprocess.call(
        'Rscript --vanilla create_sample.R '
        + str(year) + ' ' + names + ' ' + str(run_id) + ' ' + sam_method + ' ' + str(n_gsa_runs),
        shell=True)  # If only doing QAP, all samples should be equal as the
    # input distribution does not change. Hence, run_id is not needed
    gsa_sam = pd.read_csv('gsa_sample.csv')
    gsa_runs = model.predict(gsa_sam)
    gsa_runs = pd.DataFrame(gsa_runs)
    gsa_runs.to_csv('gsa_runs.csv', index=False)
    # FIRST ORDER: GSA of the model
    subprocess.call(
        'Rscript --vanilla analyze_model.R '
        + str(year) + ' ' + names + ' ' + str(run_id) + ' ' + sam_method + ' ' + str(n_gsa_runs),
        shell=True)
    first_s_indexes_temp = pd.read_csv('S_indexes.csv', index_col=0)
    first_s_indexes = first_s_indexes.append(first_s_indexes_temp.loc['original', ])
    bias_si = bias_si.append(first_s_indexes_temp.loc['bias', ])
    min_ci_si = min_ci_si.append(first_s_indexes_temp.loc['min..c.i.', ])
    max_ci_si = max_ci_si.append(first_s_indexes_temp.loc['max..c.i.', ])
    # SECOND ORDER: Create GSA sample
    subprocess.call(
        'Rscript --vanilla create_sample_2nd.R '
        + str(year) + ' ' + names + ' ' + str(run_id) + ' ' + sam_method + ' ' + str(n_gsa_runs),
        shell=True)  # If only doing QAP, all samples should be equal as the
    # input distribution does not change. Hence, run_id is not needed
    gsa_sam = pd.read_csv('gsa_sample.csv')
    gsa_runs = model.predict(gsa_sam)
    gsa_runs = pd.DataFrame(gsa_runs)
    gsa_runs.to_csv('gsa_runs.csv', index=False)
    # SECOND ORDER: GSA of the model
    subprocess.call(
        'Rscript --vanilla analyze_model_2nd.R '
        + str(year) + ' ' + names + ' ' + str(run_id) + ' ' + sam_method + ' ' + str(n_gsa_runs),
        shell=True)
    second_s_indexes_temp = pd.read_csv('S_indexes.csv', index_col=0)
    second_s_indexes = second_s_indexes.append(second_s_indexes_temp.loc['original', ])
    bias_sij = bias_sij.append(second_s_indexes_temp.loc['bias', ])
    min_ci_sij = min_ci_sij.append(second_s_indexes_temp.loc['min..c.i.', ])
    max_ci_sij = max_ci_sij.append(second_s_indexes_temp.loc['max..c.i.', ])
    

comb = [c[:-5] for c in X.columns.to_list()]
for c in combinations(X.columns, 2):
    comb.append(c[0][:-5] + '_X_' + c[1][:-5])
second_s_indexes.columns = range(len(first_s_indexes.columns) + 1,
                                 len(first_s_indexes.columns) + len(second_s_indexes.columns) + 1)
s_indexes = pd.concat([first_s_indexes, second_s_indexes], axis=1)
s_indexes.columns = comb
s_indexes.reset_index(inplace=True, drop=True)
s_indexes.rename(index=dict(zip(s_indexes.index, study_period)), inplace=True)
s_indexes.clip(lower=0, inplace=True)

bias = pd.concat([bias_si, bias_sij], axis=1)
bias.columns = s_indexes.columns
bias.index = study_period
min_ci = pd.concat([min_ci_si, min_ci_sij], axis=1)
min_ci.columns = s_indexes.columns
min_ci.index = study_period
max_ci = pd.concat([max_ci_si, max_ci_sij], axis=1)
max_ci.columns = s_indexes.columns
max_ci.index = study_period

cv_score_columns = [str(fold) + '_' + cv_metric for fold in range(1, k + 1)]
cv_score = pd.DataFrame(cv_score, index=study_period, columns=cv_score_columns)
mse_cv_score_columns = [str(fold) + '_MSE' for fold in range(1, k + 1)]
mse_cv_score = pd.DataFrame(mse_cv_score, index=study_period, columns=mse_cv_score_columns)
r2_modelfit = pd.DataFrame(r2_modelfit, index=study_period, columns=['R2'])
feature_importance = pd.DataFrame(feature_importance, index=study_period)
feature_importance.to_csv('f_imp_' + str(run_id) + '_' + names + '.csv')
cv_score.to_csv('cv_score_' + str(run_id) + '_' + names + '.csv')
r2_modelfit.to_csv('r2_' + str(run_id) + '_' + names + '.csv')
mse_cv_score.to_csv('mse_cv_score_' + str(run_id) + '_' + names + '.csv')
s_indexes.to_csv('s_indexes_' + str(run_id) + '_' + names + '.csv')
bias.to_csv('bias_' + str(run_id) + '_' + names + '.csv')
min_ci.to_csv('min_ci' + str(run_id) + '_' + names + '.csv')
max_ci.to_csv('max_ci' + str(run_id) + '_' + names + '.csv')

plt.figure(1)
plt.title('Train R2')
plt.plot(r2_modelfit.index, r2_modelfit['R2'])
plt.xlabel('Year')
plt.ylabel('R2')
plt.savefig(names + 'train_R2.png')

plt.figure(2)
plt.title(r'10-fold $R^2$')
cv_score.T.boxplot(rot=45)
plt.xlabel('Year')
plt.ylabel('R2')
plt.savefig(names + 'CV_R2.png')


# fig1 = plt.figure(3, figsize=[9, 6])
# ax1 = fig1.add_subplot(111)
# number_of_Si = s_indexes.shape[1] - second_s_indexes.shape[1]
# colormap = plt.cm.nipy_spectral
# colors = [colormap(i) for i in np.linspace(0, 1, number_of_Si)]
# ax1.set_prop_cycle('color', colors)
# for i in range(number_of_Si):
#     ax1.fill_between(study_period, min_ci.iloc[:, i], max_ci.iloc[:, i], alpha=0.5)
#     ax1.plot(study_period, s_indexes.iloc[:, i], label=s_indexes.columns[i])
# # ax1.legend(loc=2)
# ax1.legend(loc=(1.04, 0))
# plt.tight_layout()
# plt.xlabel('Year')
# plt.ylabel('$S_i$')
#
# plt.savefig('Si.png')
statistics = s_indexes.describe().T
statistics.to_csv(names + 'S_stats.csv')


from random import shuffle, seed
vars_i = [r'Alliance def', r'Inv arms flow', r'Conf dead $\Delta$', r'Conf dead$_i$', r'Contig',
          r'Flood displ$_i$', r'Flood max sev$_i$', r'ND-Gain $\Delta$', r'GDPPC $\Delta$',
          r'Immig pop', r'Min SPEI$_j$', r'Min SPEI$_i$', r'Democ $\Delta$', r'PTS $\Delta$',
          r'Rivalry', r'Trade']
fig3, ax = plt.subplots(1, 3, sharey=True, sharex=True, figsize=[8, 7])

# number_of_Si = s_indexes.shape[1] - second_s_indexes.shape[1]
number_of_Si = 16
colormap = plt.cm.nipy_spectral
colors = [colormap(i) for i in np.linspace(0, 1, number_of_Si)]
seed(0)
shuffle(colors)
ax[0].set_prop_cycle('color', colors[0:5])
ax[1].set_prop_cycle('color', colors[5:10])
ax[2].set_prop_cycle('color', colors[10:])
for i in range(number_of_Si):
    if i < 5:
        ax[0].plot(s_indexes.index, s_indexes.iloc[:, i], label=vars_i[i])
    if (i >= 5) & (i < 10):
        ax[1].plot(s_indexes.index, s_indexes.iloc[:, i], label=vars_i[i])
    if i >= 10:
        ax[2].plot(s_indexes.index, s_indexes.iloc[:, i], label=vars_i[i])
# ax1.legend(loc=2)
ax[0].legend(loc='lower center', bbox_to_anchor=(0.5, -.335))
ax[1].legend(loc='lower center', bbox_to_anchor=(0.5, -.335))
ax[2].legend(loc='lower center', bbox_to_anchor=(0.5, -.335))

ax[1].set_xlabel('Year')
ax[0].set_ylabel('$S_i$')
plt.tight_layout()
plt.savefig('Si.png')