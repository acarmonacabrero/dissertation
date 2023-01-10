import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, KFold, cross_val_score, train_test_split
from itertools import combinations
from sklearn.metrics import r2_score
import subprocess
from math import floor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

plt.ioff()

run_id = 1  # This cannot be 0
sam_method = 'eRS'
n_gsa_runs = 2000  # GSA sample size
k = 10  # Number of folds
# TODO: line above


# Analysis definition
X = pd.read_csv('X_test_python.csv')
y = pd.read_csv('y_test_python.csv')
y.columns = ['y']

kfold = KFold(n_splits=k, shuffle=True, random_state=0)
# grid_search = GridSearchCV(model, param_grid, scoring=["rmse", "r2"], n_jobs=-1, cv=kfold)

params = {'learning_rate': [0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
          'max_depth': [2, 3, 4, 5, 6, 8, 10],
          'min_child_weight': [1, 3, 5, 7],
          'gamma': [0.0, 0.05, 0.1, 0.2, 0.3, 0.4],
          'reg_lambda': [0, 1, 3, 5, 7, 9, 11],
          'colsample_bytree': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
          }


rg_model = RandomizedSearchCV(XGBRegressor(),
                           param_distributions=params,
                           n_iter=2,
                           scoring='neg_root_mean_squared_error',
                           n_jobs=-1,
                           cv=5,
                           verbose=0,
                           random_state=12)
rg_model.fit(X, y)
    # model.best_estimator_
    # model.best_score_
model = XGBRegressor(**rg_model.best_params_)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=True)
data_ = X.copy()
data_['target'] = y
data_.to_csv('year_permutation.csv', index=False)

subprocess.call('Rscript --vanilla create_sample.R ' + str(1000) + ' ' + 'python_test' + ' ' + str(run_id) +
                ' ' + sam_method + ' ' + str(n_gsa_runs), shell=True)
gsa_sam = pd.read_csv('gsa_sample.csv')
gsa_runs = model.predict(gsa_sam)
gsa_runs = pd.DataFrame(gsa_runs)
gsa_runs.to_csv('gsa_runs.csv', index=False)
    # FIRST ORDER: GSA of the model
subprocess.call('Rscript --vanilla analyze_model.R ' + str(1000) + ' ' + 'python_test' + ' ' + str(run_id) +
                ' ' + sam_method + ' ' + str(n_gsa_runs), shell=True)
first_s_indexes_temp = pd.read_csv('S_indexes.csv')
plt.scatter(range(1, 9), first_s_indexes_temp.values[0])
first_s_indexes = first_s_indexes.append(first_s_indexes_temp)
compute_sobol_g_si(a_list)

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
    second_s_indexes_temp = pd.read_csv('S_indexes.csv')
    second_s_indexes = second_s_indexes.append(second_s_indexes_temp)

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


fig1 = plt.figure(3, figsize=[9, 6])
ax1 = fig1.add_subplot(111)
number_of_Si = s_indexes.shape[1] - second_s_indexes.shape[1]
colormap = plt.cm.nipy_spectral
colors = [colormap(i) for i in np.linspace(0, 1, number_of_Si)]
ax1.set_prop_cycle('color', colors)
for i in range(number_of_Si):
    ax1.plot(study_period, s_indexes.iloc[:, i], label=s_indexes.columns[i])
# ax1.legend(loc=2)
ax1.legend(loc=(1.04, 0))
plt.tight_layout()
plt.xlabel('Year')

plt.ylabel('$S_i$')
plt.savefig('Si.png')
statistics = s_indexes.describe().T
statistics.to_csv(names + 'S_stats.csv')

