import pandas as pd
import numpy as np
import random
import os


def log_transformation(x, transf):
    if transf == 'natural_log_transformation':
        return np.log(x + 1)
    elif transf == 'log10_transformation':
        return np.log10(x + 1)
    else:
        return x


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


def feature_order_check(data, study_period):
    """
    Checks whether the combination of predictors is correct when creating the interaction names
    :param data: data set
    :param study_period: study period
    :return: correct order of predictors and their interactions
    """
    X_names = []
    for year in study_period:
        X, y = feature_filter(data, year)
        X_names.append([feature[:-5] for feature in X.columns])
    X_names = pd.DataFrame(X_names)
    return X_names


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

