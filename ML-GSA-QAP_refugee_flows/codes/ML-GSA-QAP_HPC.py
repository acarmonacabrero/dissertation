import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, KFold, cross_val_score, train_test_split
# from sklearn.preprocessing import LabelEncoder
from itertools import combinations
from sklearn.metrics import r2_score
# import timeit
import random
import subprocess
from xgboost import XGBRegressor
import argparse

lag = 0
# parser = argparse.ArgumentParser()
# parser.add_argument('--run_id', '-id', help='run id. Used to identify output names')
# parser.add_argument('--sam_method', '-sam', help='LHS, RS or eRS')
# parser.add_argument('--n_gsa_runs', '-n_gsa', help='definition of gsa size. Different for LHS or RS')
# # Read arguments from the command line
# args = parser.parse_args()
# run_id = int(args.run_id)
# sam_method = args.sam_method
# n_gsa_runs = int(args.n_gsa_runs)
nruns = 2  # Number of permutations per core
threshold = 0
k = 5
# TODO: line above
run_id = 1
sam_method = 'eRS'
n_gsa_runs = 1000


def log_transformation(x):
    return np.log10(x + 1)


def node_permutation(df, column_origin, column_destination, column_y, run_id, nruns, i_run, n=1):
    for i in range(n):
        df2 = df.copy()
        # random.seed(i + run_id)  # previous one
        random.seed((run_id - 1) * nruns + i_run)
        node_mapping = dict(zip(df[column_origin].unique(),
                                sorted(df[column_origin].unique(),
                                       key=lambda k: random.random())))
        df2[column_origin] = df[column_origin].map(node_mapping)
        df2[column_destination] = df[column_destination].map(node_mapping)
        df2 = df2.rename(columns={column_y: column_y + '_p_' + str(i)})
        if i == 0:
            df_p = pd.merge(df, df2, on=[column_origin, column_destination])
        else:
            df_p = pd.merge(df_p, df2, on=[column_origin, column_destination])
    return df_p


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


# Analysis definition
data_ori = pd.read_csv('data/data_for_analysis.csv')
data = data_ori.copy()

kfold = KFold(n_splits=k, shuffle=True, random_state=0)
# grid_search = GridSearchCV(model, param_grid, scoring=["rmse", "r2"], n_jobs=-1, cv=kfold)

params = {'learning_rate': [0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
          'max_depth': [2, 3, 4, 5, 6, 8, 10],
          'min_child_weight': [1, 3, 5, 7],
          'gamma': [0.0, 0.05, 0.1, 0.2, 0.3, 0.4],
          'reg_lambda': [0, 1, 3, 5, 7, 9, 11],
          'colsample_bytree': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
          }

names = 'all_vars'
cv_metric = 'r2'

study_period = range(1995, 1997)
# study_period = range(1996, 1998)
# TODO: line above 
# run_id = 1
# sam_method = 'LHS'
# n_gsa_runs = 2
# TODO: 3 lines above
# Initialization
for i_run in range(1, nruns + 1):
    feature_importance = []
    cv_score = []
    mse_cv_score = []
    r2_modelfit = []
    params_stack = []
    s_indexes = pd.DataFrame()
    first_s_indexes = pd.DataFrame()
    bias_si = pd.DataFrame()
    min_ci_si = pd.DataFrame()
    max_ci_si = pd.DataFrame()
    second_s_indexes = pd.DataFrame()
    bias_sij = pd.DataFrame()
    min_ci_sij = pd.DataFrame()
    max_ci_sij = pd.DataFrame()

    for year in study_period:
        # Data Preparation
        X, y = feature_filter(data, year, lag=lag, prior_ref=False)  # No time lag, no prior ref
        X.drop('arms_inverse_' + str(year), axis=1, inplace=True)
        # data_qap = X.copy()
        data_qap = pd.DataFrame(X[['state_origin_name', 'state_destination_name']])
        data_qap['ref_flow_' + str(year)] = y
        y_qap = node_permutation(data_qap, 'state_origin_name', 'state_destination_name',
                                 'ref_flow_' + str(year), run_id, nruns, i_run, n=1)
        y = y_qap['ref_flow_' + str(year) + '_p_0']
        # No prior refugee flow
        # X_ref_transformed = X['ref_flow_' + str(year - 1)].map(log_transformation)
        # X = X.drop('ref_flow_' + str(year - 1), axis=1)
        # X['ref_flow_' + str(year - 1)] = X_ref_transformed
        # del(X_ref_transformed, y_qap, data_qap)
        del(y_qap, data_qap)
        y = y.map(log_transformation)
        X = X.drop(['state_destination_name', 'state_origin_name'], axis=1)
        X, y = threshold_filter(X, y, threshold=log_transformation(threshold))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
        data_ = X.copy()
        data_['target'] = y
        data_.to_csv('year_permutation.csv', index=False)
        del data_
        rg_model = RandomizedSearchCV(XGBRegressor(),
                                   param_distributions=params,
                                   n_iter=2,
                                   scoring='neg_root_mean_squared_error',
                                   n_jobs=-1,
                                   cv=5,
                                   verbose=0,
                                   random_state=year)
        rg_model.fit(X, y)
        # Model fit and model statistics
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
    second_s_indexes.columns = range(len(first_s_indexes.columns) + 1, len(first_s_indexes.columns) + len(second_s_indexes.columns) + 1)
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
    cv_score.to_csv('cv_score_' + str(run_id) + '_' + names + '_' + str(i_run) + '.csv')
    mse_cv_score.to_csv('mse_cv_score_' + str(run_id) + '_' + names + '_' + str(i_run) + '.csv')
    r2_modelfit.to_csv('r2_' + str(run_id) + '_' + names + '_' + str(i_run) + '.csv')
    s_indexes.to_csv('s_indexes_' + str(run_id) + '_' + names + '_' + str(i_run) + '.csv')
    bias.to_csv('bias_' + str(run_id) + '_' + names + '_' + str(i_run) + '.csv')
    min_ci.to_csv('min_ci_' + str(run_id) + '_' + names + '_' + str(i_run) + '.csv')
    max_ci.to_csv('max_ci_' + str(run_id) + '_' + names + '_' + str(i_run) + '.csv')
    np.savetxt('f_imp_' + str(run_id) + '_' + names + '_' + str(i_run) + '.csv', feature_importance,
               delimiter=',', fmt='%.3f')
    feature_importance.to_csv('f_imp_' + str(run_id) + '_' + names + '_' + str(i_run) + '.csv')

