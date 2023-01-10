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
from sklearn.metrics import r2_score

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

fig, axes = plt.subplots(2, 4, sharex=True, sharey='row', figsize=[16, 8])
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
    plot_list = [xgb_add, rf_add]
    plot_list_names = ['XGB', 'RF']

    for plot_index, plot_item in enumerate(plot_list):
        df_S_tmp = pd.DataFrame(plot_item['S'])
        df_S = pd.DataFrame({'S': df_S_tmp.stack().values,
                             r'$N_T$': [i[1].split(sep='_')[0] for i in df_S_tmp.stack().index.to_list()],
                             r'$R_\epsilon$': [i[1].split(sep='_')[1] for i in df_S_tmp.stack().index.to_list()]})

        for R_epsilon in df_S[r'$R_\epsilon$'].unique():
            rmse = []
            rmse_sd = []
            rmse_df = pd.DataFrame()
            for N_T in df_S[r'$N_T$'].unique():
                sum_rmse = 0
                rmse_rea = []
                for rea in range(n_rea):
                    sum_rmse += mean_squared_error(S_analytic, df_S[(df_S[r'$N_T$'] == N_T) &
                                                        (df_S[r'$R_\epsilon$'] == R_epsilon)]['S'].reset_index(drop=True)[rea])
                    rmse_rea.append(mean_squared_error(S_analytic, df_S[(df_S[r'$N_T$'] == N_T) &
                                                        (df_S[r'$R_\epsilon$'] == R_epsilon)]['S'].reset_index(drop=True)[rea]))
                rmse.append(sum_rmse/n_rea)
                rmse_sd.append(np.std(rmse_rea))
            rmse_df['RMSE'] = rmse
            rmse_df['RMSE_sd'] = rmse_sd
            rmse_df.index = df_S[r'$N_T$'].unique()
            if 2*i_funct + plot_index <= 3:
                # axes[0, 2*i_funct + plot_index].errorbar(x=rmse_df.index, y=rmse_df['RMSE'], yerr=rmse_df['RMSE_sd'],
                #                                          label=r'$R_\epsilon$ = ' + R_epsilon)
                axes[0, 2 * i_funct + plot_index].plot(rmse_df.index, rmse_df['RMSE'], label=r'$R_\epsilon$ = '
                                                                                             + R_epsilon)
            else:
                axes[1, 2 * i_funct + plot_index - 4].plot(rmse_df.index, rmse_df['RMSE'],
                                                           label=r'$R_\epsilon$ = ' + R_epsilon)
                # axes[1, 2 * i_funct + plot_index - 4].errorbar(x=rmse_df.index, y=rmse_df['RMSE'], yerr=rmse_df['RMSE_sd'],
                #                                            label=r'$R_\epsilon$ = ' + R_epsilon)
titles = [r'Ishigami add. error, XGB', r'Ishigami add. error, RF',
          r'Ishigami mult. error, XGB', r'Ishigami mult. error, RF',
          r'Sobol G add. error, XGB', r'Sobol G add. error, RF',
          r'Sobol G mult. error, XGB', r'Sobol G mult. error, RF']
axes[0, 3].legend(loc=0)
for ax_x in range(4):
    axes[1, ax_x].title.set_text(titles[ax_x + 4])
    axes[1, ax_x].set_xlabel(r'$N_T$')
    axes[0, ax_x].title.set_text(titles[ax_x])

for ax_y in range(2):
    axes[ax_y, 0].set_ylabel('Mean RMSE')
plt.tight_layout()

plt.savefig('final_figures/good_quality/S_RMSE_subplots.png', dpi=500)
