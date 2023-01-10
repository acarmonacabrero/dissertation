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

"""
PARAMETER DEFINITION
"""

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

# if funct_name == 'ishigami':
# experiment_list_drop = ['50_0.5', '100_0', '100_0.25', '100_0.5', '100_1', '200_0',
#                         '200_0.25', '200_0.5', '200_1', '800_0', '800_0.25', '800_0.5', '800_1',
#                         '400_0.5', '1600_0', '1600_0.25', '1600_0.5', '1600_1',
#                         '4800_0.5', '3200_0', '3200_0.25', '3200_0.5', '3200_1']
# else:
experiment_list_drop = ['50_0.5', '100_0', '100_0.25', '100_0.5', '100_1', '200_0',
                        '200_0.25', '200_0.5', '200_1', '400_0', '400_0.25', '400_0.5', '400_1',
                        '800_0.5', '1600_0', '1600_0.25', '1600_0.5', '1600_1',
                        '4800_0.5', '3200_0', '3200_0.25', '3200_0.5', '3200_1']


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

plot_list = [xgb_add, rf_add]
plot_list_names = ['XGB', 'RF']

s_plot_list = ['$S_{1}$', '$S_{4}$', '$S_{6}$', '$S_{1,2}$', '$S_{1,4}$', '$S_{1,6}$', '$S_{4,5}$', '$S_{4,6}$',
               '$S_{6,7}$']

fig, axes = plt.subplots(1, 2, sharex=False, sharey=True, figsize=[14, 6])
for plot_index, plot_item in enumerate(plot_list):
    S_analytic = []
    if funct_name == 'ishigami':
        [S_analytic.append(s) for s in compute_ishigami_si(a, b)]
        [S_analytic.append(s) for s in compute_ishigami_sij(a, b)]
    if funct_name == 'sobolG':
        [S_analytic.append(s) for s in compute_sobol_g_si(a_list)]
        [S_analytic.append(s) for s in compute_sobol_g_sij(a_list)]
    df_S = pd.DataFrame(columns=[r'$S_{i}, S_{i,j}$', r'$S_{ij}$', 'data scenario'])
    for key in plot_item['S'].keys():
        df_tmp = pd.DataFrame(plot_item['S'][key], columns=S_names)
        # df_tmp.clip(lower=0, inplace=True)
        df2_tmp = pd.DataFrame({r'$S_{i}, S_{i,j}$': df_tmp.stack().values,
                            r'$S_{ij}$': [o[1] for o in df_tmp.stack().index.to_list()]})
        df2_tmp['experiment'] = r'$N_T$ = ' + key.split(sep='_')[0] + r', $R_\epsilon$ = ' + key.split(sep='_')[1]
        # TODO: replace \sigma_e for \sigma_\epsilon
        df_S = df_S.append(df2_tmp, ignore_index=True)
        df_S['Data scenario'] = df_S['experiment'].replace({'= 00': '= 0'}, regex=True)

    if funct_name == 'sobolG':
        df_S = df_S[[item in s_plot_list for item in df_S[r'$S_{ij}$'].to_list()]]
        S_analytic_tmp = [S_analytic[i] for i in [0, 3, 5, 10, 12, 14, 34, 35, 45]]
        S_analytic = S_analytic_tmp

    sns.set_theme(style="whitegrid")
    sns.set_style("ticks")
    # plt.figure(plot_index, figsize=[14, 7])
    ax = sns.boxplot(x=r'$S_{ij}$', y=r'$S_{i}, S_{i,j}$', hue='Data scenario', data=df_S, palette='Set3', linewidth=0.3,
                     fliersize=0.5, ax=axes[plot_index])
    if plot_index == 1:
        ax.legend_.remove()
    else:
        ax.set(ylabel=None)
        # ax.set(xlabel=r'$S_{ij}$')
    colors = ['lightblue', 'ivory'] * int((len(S_analytic)/2))
    if len(S_analytic) % 2 != 0:
        colors.append('lightblue')
    color_min = []
    color_max = []

    for j, s in enumerate(S_analytic):
        axes[plot_index].plot([(-1.5) + (j + 1), (-0.5) + (j + 1)], [s, s], '--k')
        color_min.append((-1.5) + (j + 1))
        color_max.append((-0.5) + (j + 1))

    for i in range(len(color_min)):
        ax.axvspan(xmin=color_min[i], xmax=color_max[i], facecolor=colors[i], alpha=0.3)
    # plt.xticks(rotation=45, ha='right')

if funct_name == 'sobolG':
    funct_name = 'Sobol G'
elif funct_name == 'ishigami':
    funct_name = 'Ishigami'
if error_type == 'additive':
    axes[0].title.set_text(funct_name + ' add. error, ' + plot_list_names[0])
    axes[1].title.set_text(funct_name + ' add. error, ' + plot_list_names[1])
if error_type == 'multiplicative':
    axes[0].title.set_text(funct_name + ' mult. error, ' + plot_list_names[0])
    axes[1].title.set_text(funct_name + ' mult. error, ' + plot_list_names[1])

handles, labels = ax.get_legend_handles_labels()
analytical_line = Line2D([], [], color='black', linestyle='--', label='Analytical value')
handles.append(analytical_line)
labels.append('Analytical value')

plt.ylim([-0.03, 0.8])

ax.set(xlabel=None)  # Removes x-axis label
# if funct_name == 'Ishigami':
#     axes[1].legend(handles=handles, labels=labels, bbox_to_anchor=(1.01, 1))
# else:
#     axes[1].legend(handles=handles, labels=labels)
axes[0].legend(handles=handles, labels=labels, loc='upper center', bbox_to_anchor=(1, -0.12), fancybox=True, ncol=5)
fig.subplots_adjust(bottom=0.2)
sns.despine()
# plt.tight_layout()
plt.savefig('final_figures/good_quality/new_S_' + funct_name + '_' + error_type + '.png', dpi=500)
