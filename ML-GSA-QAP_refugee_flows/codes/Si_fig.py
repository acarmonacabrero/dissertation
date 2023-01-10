import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle, seed

folder = 'no_arms'

s_indexes = pd.read_csv(folder + '/original/s_indexes_1_all_vars.csv', index_col=0)
vars_i = [r'Alliance def', r'Conf dead $\Delta$', r'Conf dead$_i$', r'Contig',
          r'Flood displ$_i$', r'Flood max sev$_i$', r'ND-Gain $\Delta$', r'GDPPC $\Delta$',
          r'Immig pop', r'Min SPEI$_j$', r'Min SPEI$_i$', r'Democ $\Delta$', r'PTS $\Delta$',
          r'Rivalry', r'Trade']
fig3, ax = plt.subplots(1, 3, sharey=True, sharex=True, figsize=[8, 7])

# number_of_Si = s_indexes.shape[1] - second_s_indexes.shape[1]
number_of_Si = 15
colormap = plt.cm.nipy_spectral
colors = [colormap(i) for i in np.linspace(0, 1, number_of_Si)]
seed(4)
shuffle(colors)
ax[0].set_prop_cycle('color', colors[0:5])
ax[1].set_prop_cycle('color', colors[5:10])
ax[2].set_prop_cycle('color', colors[10:])
for i in range(number_of_Si):
    if i < 5:
        ax[0].plot(s_indexes.index, s_indexes.iloc[:, i], label=vars_i[i])
    if (i >= 5) & (i < 10):
        ax[1].plot(s_indexes.index, s_indexes.iloc[:, i], label=vars_i[i])
    if i >= 10:
        ax[2].plot(s_indexes.index, s_indexes.iloc[:, i], label=vars_i[i])
# ax1.legend(loc=2)
ax[0].legend(loc='lower center', bbox_to_anchor=(0.5, -.335))
ax[1].legend(loc='lower center', bbox_to_anchor=(0.5, -.335))
ax[2].legend(loc='lower center', bbox_to_anchor=(0.5, -.335))

ax[1].set_xlabel('Year')
ax[0].set_ylabel('$S_i$')
plt.tight_layout()
plt.savefig(folder + '/Si.png')

