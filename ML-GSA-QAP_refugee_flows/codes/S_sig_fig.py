import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.lines import Line2D
from itertools import combinations

"""

Plot the scatter plot for significance and importance

"""


# folder = 'LHS_4_2.5'
folder = 'no_arms'
names = 'all_vars'
ori_S_df = pd.read_csv(folder + '/intermediate_results/recalc_s_indexes_1_original' + names + '.csv', index_col=0)
p_value_df = pd.read_csv(folder + '/intermediate_results/sig_s_indexes_.csv', index_col=0)
left_right_sig = pd.read_csv(folder + '/intermediate_results/left_right_sig.csv', index_col=0)
# r'$\nabla$'

vars_i = [r'Alliance def', r'Conf dead', r'Conf dead$_i$', r'Contig',
          r'Flood displ$_i$', r'Flood max sev$_i$', r'$\Delta$ ND-Gain', r'$\Delta$ GDPPC',
          r'Immig pop', r'Min SPEI$_j$', r'Min SPEI$_i$', r'$\Delta$ Democ', r'$\Delta$ PTS',
          r'Rivalry', r'Trade']
# for i in range(15):
#     print(vars_i[i], ori_S_df.columns[i])

comb = []
for c in combinations(vars_i, 2):
    comb.append(c[0] + ' X ' + c[1])

var_list = vars_i + comb
# r'$\nabla$ ad'

ori_S_df.columns = var_list
plot_var_list = var_list[15:] + var_list[0:15]
plot_var_list.reverse()
# plot_var_list = var_list
# b = plot_var_list[::-1]
# ori_S_df = ori_S_df[plot_var_list]
# ori_S_df = ori_S_df[plot_var_list[:-70]]  # reduce number of interactions in the plot
df_S_sig = ori_S_df.unstack().reset_index()
df_S_sig.columns = ['var', 'year', 'S']
left_right_sig.columns = var_list
left_right_sig = left_right_sig[plot_var_list]
df_S_sig['sig'] = left_right_sig.unstack().reset_index()[0]
df_S_sig2 = df_S_sig.copy()

df_S_sig2['S'] *= 100
alpha = list(np.unique(df_S_sig2['sig']))
color_mapping = list(np.arange(0, 5, 1)/5)
alpha_mapping = dict(zip(alpha, color_mapping))
df_S_sig2['sig'] = df_S_sig2['sig'].map(alpha_mapping)

cmap = matplotlib.cm.get_cmap('coolwarm')
# s_map = sorted(df_S_sig2['sig'].unique())
s_map = [1, 0.8, 0.5, 0.2, 0]
rgba = cmap(s_map)

# Exploring interactions
# df_S_sig_int = df_S_sig2[~df_S_sig2['var'].isin(vars_i)]
# df_S_sig_int = df_S_sig_int[df_S_sig_int['sig'].isin([0, 0.8])]
# df_S_sig_int.drop(df_S_sig_int[df_S_sig_int['S'] < 1].index, inplace=True)
threshold = 4
int_threshold_set = df_S_sig2.groupby(['var']).max()
int_threshold_set = int_threshold_set[int_threshold_set['S'] > threshold].index
int_threshold_set = list(int_threshold_set)
for var in vars_i:
    if var in int_threshold_set:
        int_threshold_set.remove(var)
int_threshold_set = vars_i + int_threshold_set
df_S_sig3 = df_S_sig2[df_S_sig2['var'].isin(int_threshold_set)].reset_index(drop=True)
# df_S_sig3.drop(df_S_sig3[df_S_sig3['S'] < 1].index, inplace=True)
# df_S_sig_part = df_S_sig2[~df_S_sig2['var'].isin(vars_i)]
# df_S_sig_part.drop(df_S_sig_part[df_S_sig_part['S'] < 4].index, inplace=True)
# df_S_sig3 = df_S_sig3.append(df_S_sig_part).reset_index()

legend_elements = [Line2D([0], [0], marker='o', color='w', label='p-value < 0.05 Right',
                          markerfacecolor=rgba[0], markersize=7),
                   Line2D([0], [0], marker='o', color='w', label='p-value < 0.05 Left',
                          markerfacecolor=rgba[4], markersize=7),
                   Line2D([0], [0], marker='o', color='w', label='p-value < 0.1 Right',
                          markerfacecolor=rgba[1], markersize=8),
                   Line2D([0], [0], marker='o', color='w', label=r'$S$ = 0.4',
                          markerfacecolor='k', markersize=7),
                   Line2D([0], [0], marker='o', color='w', label='Not significant',
                          markerfacecolor=rgba[2], markersize=8),
                   Line2D([0], [0], marker='o', color='w', label=r'$S$ = 0.15',
                          markerfacecolor='k', markersize=5.5),
                   Line2D([0], [0], marker='o', color='w', label='p-value < 0.1 Left',
                          markerfacecolor=rgba[3], markersize=7),
                   Line2D([0], [0], marker='o', color='w', label=r'$S$ = 0.05',
                          markerfacecolor='k', markersize=3.5),
                   ]


fig, ax = plt.subplots(figsize=(7.75, 6))
ax.scatter(x='var',
           y='year',
           s="S",
           data=df_S_sig3,
           c='sig',
           cmap='coolwarm',
           linewidths=0.25,
           edgecolors='face')
ax.plot([14.5, 14.5], [1995, 2015], 'k', linewidth=0.5)  # for lag=0
# ax.plot([15.5, 15.5], [1996, 2015], 'k', linewidth=0.5)  # for lag=-1
plt.margins(.01)
plt.ylabel('Year')
plt.xticks(rotation=45, ha='right')
# plt.yticks([1996, 2001, 2006, 2011, 2015])  # for lag=-1
plt.yticks([1995, 2000, 2005, 2010, 2015])  # for lag=0
# ax.legend(handles=legend_elements, loc=(1.007, 0))
ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=4)
fig.tight_layout()
# plt.show()
plt.savefig(folder + '/final_figs/S_sig.png', dpi=500)


# df = pd.DataFrame([[.3, .2, .4], [.1, .4, .1]], columns=list("ABC"), index=list("XY"))
#
# dfu = df.unstack().reset_index()
# dfu.columns = list("XYS")
#
# dfu["S"] *= 5000
# plt.scatter(x="X", y="Y", s="S", data=dfu, c=[0, 0, 0, 1, 1, 1])
# plt.margins(.4)
# plt.show()
#
# x, y = np.meshgrid(df.columns, df.index)
#
# df *= 5000
# plt.scatter(x=x.flatten(), y=y.flatten(), s=df.values.flatten())
# plt.margins(.4)
# plt.show()

