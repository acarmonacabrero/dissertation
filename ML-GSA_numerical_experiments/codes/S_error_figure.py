import pandas as pd
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from test_functions import ishigami_add, ishigami_mult
from test_functions import sobol_g_add, sobol_g_mult
import seaborn as sns
from analytical_solutions import compute_ishigami_si, compute_ishigami_sij, compute_ishigami_sti
from analytical_solutions import compute_sobol_g_si, compute_sobol_g_sij, compute_sobol_g_sti
from analytical_solutions import sobol_y_var, ishigami_y_var
import pickle
import statsmodels.stats.api as sms
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

order = 2  # Maximum order of sensitivity indexes
n_rea = 100  # Number of experiment realizations (points in the boxplots)
n_gsa = 6000  # GSA sampling
order = 2  # GSA maximum order of sensitivity indexes

a, b = 3, 2  # Ishigami parameters
a_list = [0, 0, 0, 3, 3, 9, 9, 9, 9, 9]
# a_list = [0, 1, 3]

n_train = [50, 100, 200, 400, 800, 1600, 3200, 4800]  # Train sample sizes
# n_train = [50, 200, 1600, 3200, 4800]  # Train sample sizes
n_test = 200  # Test sample size


funct_list = [ishigami_add, ishigami_mult, sobol_g_add, sobol_g_mult]
# Here loop
funct = sobol_g_mult
for i_funct, funct in enumerate(funct_list):
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

    with open('figures_' + funct_name + '_N_e1/' + error_type + '/dictionaries/rf_add.pkl', 'rb') as f:
        rf_add = pickle.load(f)

    with open('figures_' + funct_name + '_N_e1/' + error_type + '/dictionaries/xgb_add.pkl', 'rb') as f:
        xgb_add = pickle.load(f)

    with open('figures_' + funct_name + '_N_e1/' + error_type + '/dictionaries/xgb_no_e.pkl', 'rb') as f:
        xgb_no_e = pickle.load(f)

    with open('figures_' + funct_name + '_N_e1/' + error_type + '/dictionaries/rf_no_e.pkl', 'rb') as f:
        rf_no_e = pickle.load(f)

    experiment_list = []
    [experiment_list.append(str(n_i) + '_' + str(s_i)) for n_i in n_train for s_i in ratio_vye]


    for key1 in xgb_no_e.keys():
        for key in xgb_no_e[key1].keys():
            xgb_add[key1][key + '_0'] = xgb_no_e[key1][key]
        xgb_add[key1] = {k: xgb_add[key1][k] for k in experiment_list}
    for key1 in rf_no_e.keys():
        for key in rf_no_e[key1].keys():
            rf_add[key1][key + '_0'] = rf_no_e[key1][key]
        rf_add[key1] = {k: rf_add[key1][k] for k in experiment_list}
    # TODO: ALL THIS HAS TO BE CHANGED
    experiment_list
    ['50_0', '50_0.25', '50_0.5', '50_1', '100_0', '100_0.25', '100_0.5', '100_1', '200_0', '200_0.25', '200_0.5', '200_1',
     '400_0', '400_0.25', '400_0.5', '400_1', '800_0', '800_0.25', '800_0.5', '800_1', '1600_0', '1600_0.25', '1600_0.5',
     '1600_1', '3200_0', '3200_0.25', '3200_0.5', '3200_1', '4800_0', '4800_0.25', '4800_0.5', '4800_1']


    # # TODO: ALL THIS HAS TO BE CHANGED

    # experiment_list_drop = ['50_0', '50_0.25', '50_0.5', '50_1', '100_0', '100_0.25', '100_0.5', '100_1', '200_0',
    #                         '200_0.25', '200_0.5', '200_1', '400_0', '400_0.25', '400_0.5', '400_1', '800_0', '800_0.25',
    #                         '800_0.5', '800_1', '1600_0', '1600_0.25', '1600_0.5','1600_1', '3200_0', '3200_0.25',
    #                         '3200_0.5', '3200_1', '4800_0', '4800_0.25', '4800_0.5', '4800_1']

    # for key1 in xgb_add.keys():
    #     for experiment in experiment_list_drop:
    #         xgb_add[key1].pop(experiment, None)
    #     for experiment in experiment_list_drop:
    #         rf_add[key1].pop(experiment, None)

    input_names = []
    for i in range(k):
        input_names.append('X' + str(i + 1))

    # Create labels for effects S (S1, S2, S1,2, ...)
    S_names = [r'$S_{' + c[1:] + '}$' for c in input_names]
    S_static = ['S' + c[1:] for c in input_names]
    for c in combinations(S_static, 2):
        S_names.append(r'$S_{' + c[0][1:] + ',' + c[1][1:] + '}$')


    #  Get analytic sensitivity indexes
    S_analytic = []
    if funct_name == 'ishigami':
        [S_analytic.append(s) for s in compute_ishigami_si(a, b)]
        [S_analytic.append(s) for s in compute_ishigami_sij(a, b)]
    if funct_name == 'sobolG':
        [S_analytic.append(s) for s in compute_sobol_g_si(a_list)]
        [S_analytic.append(s) for s in compute_sobol_g_sij(a_list)]

    # ALTERNATIVE
    # plot_list = [rf_add, xgb_add]
    # plot_list_names = ['rf', 'xgb']
    #
    # for plot_index, plot_item in enumerate(plot_list):
    #     df_S_tmp = pd.DataFrame(plot_item['S'])
    #     df_S = pd.DataFrame({'S': df_S_tmp.stack().values,
    #                          r'$N_T$': [i[1].split(sep='_')[0] for i in df_S_tmp.stack().index.to_list()],
    #                          r'$\sigma_\epsilon$': [i[1].split(sep='_')[1] for i in df_S_tmp.stack().index.to_list()]})
    #
    #     plt.figure(plot_index + 10)
    #     for sigma_epsilon in df_S[r'$\sigma_\epsilon$'].unique():
    #         rmse = []
    #         rmse_df = pd.DataFrame()
    #         for N_T in df_S[r'$N_T$'].unique():
    #             sum_rmse = 0
    #             for variable in range(len(S_analytic)):
    #                 si = []
    #
    #                 for i in range(df_S.shape[0]):
    #                     si.append(df_S['S'][i][variable])  # 0 to take the first sensitivity index
    #                 df_S[r'$S_1$'] = si
    #
    #                 sum_rmse += mean_squared_error(df_S[(df_S[r'$N_T$'] == N_T) &
    #                                                     (df_S[r'$\sigma_\epsilon$'] == sigma_epsilon)][r'$S_1$'],
    #                                                np.repeat(S_analytic[variable], n_rea))
    #                 # rmse.append(mean_absolute_error(df_S[(df_S[r'$N_T$'] == N_T) &
    #                 #                                     (df_S[r'$\sigma_\epsilon$'] == sigma_epsilon)][r'$S_1$'],
    #                 #                                np.repeat(S_analytic[0], n_rea)))
    #             rmse.append(sum_rmse)
    #
    #         rmse_df['RMSE'] = rmse
    #         rmse_df.index = df_S[r'$N_T$'].unique()
    #         plt.plot(rmse_df.index, rmse_df, label=r'$\sigma_\epsilon$ = ' + sigma_epsilon)
    #     plt.legend(loc=0)
    #     plt.xlabel(r'$N_T$')
    #     plt.ylabel(r'$\Sigma_iRMSE (S_i, S_{i, j})$')
    #     plt.savefig('old_figures/good_quality/S_RMSE_' + plot_list_names[
    #         plot_index] + '_' + funct_name + '_' + error_type + '.png', dpi=500)


    # ALTERNATIVE
    plot_list = [rf_add, xgb_add]
    plot_list_names = ['rf', 'xgb']

    for plot_index, plot_item in enumerate(plot_list):
        df_S_tmp = pd.DataFrame(plot_item['S'])
        df_S = pd.DataFrame({'S': df_S_tmp.stack().values,
                             r'$N_T$': [i[1].split(sep='_')[0] for i in df_S_tmp.stack().index.to_list()],
                             r'$\sigma_\epsilon$': [i[1].split(sep='_')[1] for i in df_S_tmp.stack().index.to_list()]})

        plt.figure(plot_index + 100 * i_funct)
        for sigma_epsilon in df_S[r'$\sigma_\epsilon$'].unique():
            rmse = []
            rmse_df = pd.DataFrame()
            for N_T in df_S[r'$N_T$'].unique():
                sum_rmse = 0
                for rea in range(n_rea):
                    sum_rmse += mean_squared_error(S_analytic, df_S[(df_S[r'$N_T$'] == N_T) &
                                                        (df_S[r'$\sigma_\epsilon$'] == sigma_epsilon)]['S'].reset_index(drop=True)[rea])
                rmse.append(sum_rmse/n_rea)
            rmse_df['RMSE'] = rmse
            rmse_df.index = df_S[r'$N_T$'].unique()
            plt.plot(rmse_df.index, rmse_df, label=r'$\sigma_\epsilon$ = ' + sigma_epsilon)
        plt.legend(loc=0)
        plt.xlabel(r'$N_T$')
        plt.ylabel(r'$\Sigma_iRMSE (S_i, S_{i, j})$')

        plt.savefig('final_figures/good_quality/S_RMSE_' + plot_list_names[
            plot_index] + '_' + funct_name + '_' + error_type + '.png', dpi=500)
