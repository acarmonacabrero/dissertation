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

order = 2  # Maximum order of sensitivity indexes
n_rea = 100  # Number of experiment realizations (points in the boxplots)
n_gsa = 6000  # GSA sampling
order = 2  # GSA maximum order of sensitivity indexes

a, b = 3, 2  # Ishigami parameters
a_list = [0, 0, 0, 3, 3, 9, 9, 9, 9, 9]
# a_list = [0, 1, 3]

n_train = [50, 100, 200, 400, 800, 1600, 3200, 4800]  # Train sample sizes
n_test = 200  # Test sample size



funct_list = [ishigami_add, ishigami_mult, sobol_g_add, sobol_g_mult]
# Here loop
funct = ishigami_add
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

# TODO: THIS HAS TO BE CHANGED
experiment_list_drop = ['50_0.5', '100_0', '100_0.25', '100_0.5', '100_1', '200_0',
                        '200_0.25', '200_0.5', '200_1', '400_0', '400_0.25', '400_0.5', '400_1',
                        '800_0.5', '1600_0', '1600_0.25', '1600_0.5', '1600_1',
                        '3200_0.5', '4800_0', '4800_0.25', '4800_0.5', '4800_1']

for key1 in xgb_add.keys():
    for experiment in experiment_list_drop:
        xgb_add[key1].pop(experiment, None)
    for experiment in experiment_list_drop:
        rf_add[key1].pop(experiment, None)

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
if funct_name == 'ishigami':
    analytical_sti = compute_ishigami_sti(a, b, norm=1)
elif funct_name == 'sobolG':
    analytical_sti = compute_sobol_g_sti(a_list, norm=1)



plot_list = [rf_add, xgb_add]
plot_list_names = ['rf', 'xgb']

# TODO: add rf_add for ML_GSA
plot_list = [xgb_add, xgb_add, rf_add]
fig, axes = plt.subplots(1, 3, sharex=True, sharey=True)
for plot_index, plot_item in enumerate(plot_list):
    for experiment in rf_add['S'].keys():
        experiment_label = r'$N_T = ' + experiment.split(sep='_')[0] + ', \sigma_\epsilon = ' + \
                           experiment.split(sep='_')[1] + '$'
        df_imp = pd.DataFrame(columns=['mean', 'ci', 'Analytical ST'])
        mean_imp_variable_list = []
        ci_imp_variable_list = []
        for variable in range(k):
            realization_list = []
            for realization in range(n_rea):
                if plot_index == 0:
                    realization_list.append(plot_item['ST'][experiment][realization][variable] /
                                            np.sum(plot_item['ST'][experiment][realization]))
                else:
                    realization_list.append(plot_item['imp'][experiment][realization][variable])
            mean_imp_variable_list.append(np.mean(realization_list))
            # std_imp_variable_list.append(np.std(realization_list))
            ci_imp_variable_list.append(abs(sms.DescrStatsW(realization_list).tconfint_mean()[0] -
                                            sms.DescrStatsW(realization_list).tconfint_mean()[1]))
            # print(max(realization_list), min(realization_list), ci_imp_variable_list[variable])

        df_imp['mean'] = mean_imp_variable_list
        df_imp['ci'] = ci_imp_variable_list
        df_imp['Analytical ST'] = analytical_sti
        # If sobolG, only plot the three first different indexes (analytically, the rest should be equal to these three)
        if funct_name == 'sobolG':
            df_imp = df_imp.iloc[[0, 3, 5]]
            # df_imp = df_imp.iloc[[0, 1, 2]]
        axes[plot_index].errorbar(x=df_imp['mean'], yerr=df_imp['ci'], y=df_imp['Analytical ST'], fmt='o', capsize=5,
                                  markeredgewidth=1, label=experiment_label)
        axes[plot_index].plot([-0.05, 1.05], [-.05, 1.05], 'gray')
    # plt.plot([0, 1], [0, 1], color='gray')
    # plt.xlim([-0.05, 1.05])
    # plt.ylim([-0.05, 1.05])
axes[2].legend(bbox_to_anchor=(1.01, 1))
axes[0].set_ylabel(r'Analytical $S_T$')
axes[0].set_xlabel(r'$S_T$ XGBoost')
axes[1].set_xlabel('MDI XGBoost')
axes[2].set_xlabel('MDI RF')
plt.tight_layout()
plt.savefig('final_figures/good_quality/MDI_ST_' + funct_name + '_' + error_type + '.png', dpi=500)
