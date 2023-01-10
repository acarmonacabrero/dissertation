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
# n_train = [50, 100, 200, 400, 800, 1600, 3200, 4800]  # Train sample sizes
n_train = [200]  # Train sample sizes
n_test = 200  # Test sample size

n_rea = 1  # Number of experiment realizations (points in the boxplots)
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
            rg_model = RandomForestRegressor(n_estimators=200)
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
            rg_model = XGBRegressor()
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


# List of experiments in dictionaries
experiment_list = []
[experiment_list.append(str(n_i) + '_' + str(s_i)) for n_i in n_train for s_i in ratio_vye]
# for l, item in enumerate(experiment_list):
#     if item[-2:] == '_0':
#         experiment_list[l] = item[:-2] + '_00'

for key1 in xgb_no_e.keys():
    for key in xgb_no_e[key1].keys():
        xgb_add[key1][key + '_0'] = xgb_no_e[key1][key]
    xgb_add[key1] = {k: xgb_add[key1][k] for k in experiment_list}
for key1 in rf_no_e.keys():
    for key in rf_no_e[key1].keys():
        rf_add[key1][key + '_0'] = rf_no_e[key1][key]
    rf_add[key1] = {k: rf_add[key1][k] for k in experiment_list}



"""
Plots
"""
# sns.color_palette("colorblind")

experiment_list_drop = ['50_2.5', '200_2.5', '800_2.5', '1600_2.5', '3200_0', '3200_2.5', '3200_10', '3200_30',
                        '4800_0', '4800_2.5', '4800_10', '4800_30']

for key1 in xgb_add.keys():
    for experiment in experiment_list_drop:
        xgb_add[key1].pop(experiment, None)
    for experiment in experiment_list_drop:
        rf_add[key1].pop(experiment, None)

# Create labels for effects S (S1, S2, S1,2, ...)
S_names = ['S' + c[1:] for c in input_names]
S_static = ['S' + c[1:] for c in input_names]
for c in combinations(S_static, 2):
    S_names.append(c[0] + ',' + c[1][-1])

#  Get analytic sensitivity indexes
S_analytic = []
if funct_name == 'ishigami':
    [S_analytic.append(s) for s in compute_ishigami_si(a, b)]
    [S_analytic.append(s) for s in compute_ishigami_sij(a, b)]
if funct_name == 'sobolG':
    [S_analytic.append(s) for s in compute_sobol_g_si(a_list)]
    [S_analytic.append(s) for s in compute_sobol_g_sij(a_list)]

# Plots
# *** THEN RUN FROM HERE
plot_list = [rf_add, xgb_add]
plot_list_names = ['rf', 'xgb']

for plot_index, plot_item in enumerate(plot_list):
    # *ADDED THIS TO KEEP CONSISTENCY WITH ST PLOTS IN figures.py
    # if error_type == 'additive':
    #     if funct_name == 'ishigami':
    #         experiment_list_drop = ['50_2.5', '200_2.5', '800_2.5', '1600_2.5', '3200_0', '3200_2.5', '3200_10', '3200_30',
    #                                 '4800_0', '4800_2.5', '4800_10', '4800_30']
    #     else:
    #         experiment_list_drop = ['50_30', '200_0', '200_2.5', '200_10', '200_30', '800_30', '1600_0', '1600_2.5',
    #                                 '1600_10', '1600_30', '3200_30', '4800_30']
    # if error_type == 'multiplicative':
    #     if funct_name == 'ishigami':
    #         experiment_list_drop = ['50_0.85', '200_0.85', '800_2.5', '1600_0.85', '3200_0', '3200_0.5', '3200_0.85', '3200_1.2',
    #                                 '4800_0.85']
    #     else:
    #         experiment_list_drop = ['50_0', '50_0.5', '50_0.85', '50_1.2', '200_0.85', '800_0.85', '1600_0', '1600_0.5',
    #                                 '1600_0.85', '1600_1.2', '3200_0', '3200_0.5', '3200_0.85', '3200_1.2', '4800_0.85']

    # for key1 in xgb_add.keys():
    #     for experiment in experiment_list_drop:
    #         xgb_add[key1].pop(experiment, None)
    #     for experiment in experiment_list_drop:
    #         rf_add[key1].pop(experiment, None)
    # *UNTIL HERE
    df_S = pd.DataFrame(columns=['S', r'$S_{ij}$', 'experiment'])
    for key in plot_item['S'].keys():
        df_tmp = pd.DataFrame(plot_item['S'][key], columns=S_names)
        # df_tmp.clip(lower=0, inplace=True)
        df2_tmp = pd.DataFrame({'S': df_tmp.stack().values,
                            r'$S_{ij}$': [o[1] for o in df_tmp.stack().index.to_list()]})
        df2_tmp['experiment'] = r'$N_T$ = ' + key.split(sep='_')[0] + r', $\sigma_e$ = ' + key.split(sep='_')[1]
        df_S = df_S.append(df2_tmp, ignore_index=True)
        df_S['experiment'] = df_S['experiment'].replace({'= 00': '= 0'}, regex=True)

    sns.set_theme(style="whitegrid")
    sns.set_style("ticks")
    plt.figure(plot_index, figsize=[10, 7])
    ax = sns.boxplot(x=r'$S_{ij}$', y='S', hue='experiment', data=df_S, palette='Set3', linewidth=0.3, fliersize=0.5)
    colors = ['lightblue', 'ivory'] * int((len(S_analytic)/2))
    color_min = []
    color_max = []

    for j, s in enumerate(S_analytic):
        plt.plot([(-1.5) + (j + 1), (-0.5) + (j + 1)], [s, s], '--k')
        color_min.append((-1.5) + (j + 1))
        color_max.append((-0.5) + (j + 1))

    for i in range(len(color_min)):
        ax.axvspan(xmin=color_min[i], xmax=color_max[i], facecolor=colors[i], alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    analytical_line = Line2D([], [], color='black', linestyle='--', label='Analytical value')
    plt.xticks(rotation=45, ha='right')
    handles.append(analytical_line)
    labels.append('Analytical value')
    ax.legend(handles=handles, labels=labels, bbox_to_anchor=(1.01, 1))  # title="experiment"
    sns.despine()
    plt.tight_layout()
    plt.savefig('final_figures/good_quality/S_' + plot_list_names[plot_index] + '_' + funct_name + '_' + error_type + '.png', dpi=500)

    # R2 plot

    df_R2_tmp = pd.DataFrame(plot_item['R2'])
    df_R2 = pd.DataFrame({r'$R^2$': df_R2_tmp.stack().values,
                          r'$N_T$': [i[1].split(sep='_')[0] for i in df_R2_tmp.stack().index.to_list()],
                          r'$\sigma_e$': [i[1].split(sep='_')[1] for i in df_R2_tmp.stack().index.to_list()]})
    plt.figure((plot_index+1) * 100)
    # Error bars = sd of data
    # plt.errorbar(x=df_R2[r'$N_T$'], y=df_R2[r'$R^2$'], yerr=)
    sns.pointplot(x=r'$N_T$', y=r'$R^2$', hue=r'$\sigma_e$', data=df_R2, capsize=0.05, ci='sd')
    plt.xticks(rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.01, 1), loc=2, title=r'$\sigma_e$')
    # plt.ylim(-2.55, 1.1)
    plt.tight_layout()
    plt.savefig('final_figures/good_quality/R2_' + plot_list_names[plot_index] + '_' + funct_name + '_' + error_type + '.png', dpi=500)
# *** UP TO HERE

# ST plot

for plot_index, plot_item in enumerate(plot_list):
    imp_list = {str(i): [] for i in range(k)}
    ml_imp = {str(experiment): copy.deepcopy(imp_list) for experiment in plot_item['imp'].keys()}
    for i in range(k):
        [ml_imp[experiment][str(i)].append(l[i]) for experiment in ml_imp.keys() for l in plot_item['imp'][experiment]]

    plt.figure((plot_index + 1) * 1000)
    plt.title(plot_list_names[plot_index])
    for experiment in ml_imp.keys():
        df_plot = pd.DataFrame(ml_imp[experiment])
        t1 = df_plot.unstack()
        t2 = t1.reset_index().drop('level_1', axis=1)
        if funct_name == 'ishigami':
            analytical_sti = compute_ishigami_sti(a, b, norm=1)
        elif funct_name == 'sobolG':
            analytical_sti = compute_sobol_g_sti(a_list, norm=1)
        analytical_st_list = []
        [analytical_st_list.append([analytical_sti[i]] * n_rea) for i in range(len(analytical_sti))]
        t2.columns = ['Variable', 'Importance']
        t2[r'Analytical $S_T$'] = reduce(iconcat, analytical_st_list)
        # fig, ax = plt.subplots(1, 3, sharey=True)
        plt.plot([-0.05, 1.05], [-.05, 1.05], 'gray')
        plt.errorbar(x=t2.groupby('Variable').mean()[r'Analytical $S_T$'],
                     y=t2.groupby('Variable').mean()[r'Importance'],
                     yerr=pd.DataFrame(np.array([t2['Importance'].to_list()[i*n_rea:(i+1)*n_rea] for i in range(k)])
                                       .T).std(), fmt='o', capsize=5, markeredgewidth=1, label=experiment)
        plt.xticks(rotation=45, ha='right')
        plt.legend(bbox_to_anchor=(1.01, 1))
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel(r'Analytical $S_T$')
        plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig('figures_' + funct_name + '/' + error_type + '/imp_ST/' + plot_list_names[plot_index] + '.png')


# Second-order interactions plot

if funct_name == 'ishigami':
    interaction_list = compute_ishigami_sij(a, b, closed=0)
    total_interactions = np.sum(interaction_list)

if funct_name == 'sobolG':
    interaction_list = compute_sobol_g_sij(a_list, order=2, closed=0)
    total_interactions = np.sum(interaction_list)


for plot_index, plot_item in enumerate(plot_list):
    plt.figure(plot_index)
    plt.title(plot_list_names[plot_index])
    captured_interactions = {experiment: [] for experiment in plot_item['S'].keys()}
    for experiment in captured_interactions:
        for realization in range(n_rea):
            captured_interactions[experiment].append(
                np.sum(plot_item['S'][experiment][realization][-len(interaction_list):]))

    captured_interactions_df = pd.DataFrame(captured_interactions)
    # captured_interactions_df.clip(lower=0, inplace=True)  # TODO: commented this

    original_columns = captured_interactions_df.columns.to_list()
    captured_interactions_df.columns = [r'$N_T$ = ' + col[:-3] + r' $\sigma_e$ = ' + col[-2:]
                                        for col in original_columns]
    captured_interactions_df = captured_interactions_df.stack()
    captured_interactions_df = captured_interactions_df.reset_index().drop('level_0', axis=1)
    captured_interactions_df.columns = ['Experiment', 'Interactions']

    sns.set_theme(style="whitegrid")
    tips = sns.load_dataset("tips")
    plt.figure((plot_index + 1) * 10000)
    plt.title(r'$\Sigma$($2^{nd}$-order interactions)')
    ax = sns.boxplot(x='Experiment', y='Interactions', data=captured_interactions_df, palette='Set3')
    plt.plot([-1.5, 4.5], [total_interactions, total_interactions], '--k')
    plt.xticks(rotation=45, ha='right')

    # handles, labels = ax.get_legend_handles_labels()
    # analytical_line = Line2D([], [], color='black', linestyle='--', label='Analytical value')
    # handles.append(analytical_line)
    # labels.append('Analytical value')
    # ax.legend(handles=handles, labels=labels, bbox_to_anchor=(1.01, 1))  # title="experiment"
    plt.tight_layout()
    plt.savefig('figures_' + funct_name + '/' + error_type + '/sum_int/' + plot_list_names[plot_index] + '.png')
