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
funct = ishigami_mult
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

# TODO: THIS HAS TO BE CHANGED. In this case, the list is longer than in Sij and STi
# experiment_list_drop = ['50_0.5', '200_0.5', '1600_0.5', '3200_0.5', '4800_0.5']


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

plot_list = [xgb_add, rf_add]
plot_list_names = ['XGB', 'RF']

min_y = []

fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=[10, 5])
for plot_index, plot_item in enumerate(plot_list):
    df_S = pd.DataFrame(columns=['S', r'$S_{ij}$', 'experiment'])

    df_R2_tmp = pd.DataFrame(plot_item['R2'])
    df_R2 = pd.DataFrame({r'$R^2$': df_R2_tmp.stack().values,
                          r'$N_T$': [i[1].split(sep='_')[0] for i in df_R2_tmp.stack().index.to_list()],
                          r'$R_\epsilon$': [i[1].split(sep='_')[1] for i in df_R2_tmp.stack().index.to_list()]})
    ax = sns.pointplot(x=r'$N_T$', y=r'$R^2$', hue=r'$R_\epsilon$', data=df_R2, capsize=0.05, ci='sd', scale=0.5,
                       ax=axes[plot_index])
    ax.plot([0, len(n_train)-1], [0, 0], '--k', linewidth=0.5)
    # plt.xticks(rotation=45, ha='right')
    min_y = np.append(min_y, np.floor((df_R2.groupby(['$R_\epsilon$', r'$N_T$']).mean() -
                                       df_R2.groupby(['$R_\epsilon$', r'$N_T$']).std())[r'$R^2$'].min()))
    if plot_index == 0:
        ax.legend_.remove()
    else:
        ax.set(ylabel=None)

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

# axes[0].set_ylim()
plt.yticks(np.append(np.arange(min(min_y), 0, 2), np.array([0, 1])))
axes[1].legend(loc=4, title=r'$R_\epsilon$')
# plt.ylim(min_y, 1.1)
plt.tight_layout()
plt.savefig('final_figures/good_quality/R2_' + funct_name + '_' + error_type + '.png', dpi=500)
