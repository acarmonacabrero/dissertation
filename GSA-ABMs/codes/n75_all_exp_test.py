import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os


def intercalate_lists(a, b):
    c = list(zip(a, b))
    return [elt for sublist in c for elt in sublist]


color_dic = {'input': ['meanwater', 'gradient', 'bet11', 'bet12', 'alp1', 'alp2', 'bet21', 'bet22', 'bet23', 'others'],
             'color': [[31/255, 119/255, 180/255], [255/255, 127/255, 14/255], [44/255, 160/255, 44/255],
                       [214/255, 39/255, 40/255], [148/255, 103/255, 189/255], [140/255, 86/255, 75/255],
                       [227/255, 119/255, 194/255], [127/255, 127/255, 127/255], [188/255, 189/255, 34/255], [23/255, 190/255, 207/255]]
             }
color_dic = pd.DataFrame(color_dic)

""" SETTINGS """
outpath = os.getcwd() + '/post_outputs/'
font = {'family': 'serif', 'weight': 'normal', 'size': 14}
matplotlib.rc('font', **font)

file_name, N_n, typeS = 'all_outputsdet_five_cities_SALib_indexes.csv', 75, 'det'  # Det indices

indices = pd.read_csv(outpath + file_name)

pies = pd.DataFrame()
for i in indices['output'].unique():
    tmp = indices[(indices['output'] == i) * (indices['N'] == N_n)]
    group = tmp.groupby('input')
    tmp = group.mean()
    tmp['output'] = i
    pies = pd.concat([pies, tmp])

pies['Interactions'] = pies['ST'] - pies['S1']
pies['input'] = pies.index
pies.columns = ['Si', 'ST', 'S1_conf', 'ST_conf', 'oSi', 'oST', 'N', 'S_exp', 'output', 'Interactions', 'input']
pies['oInt'] = pies['oST'] - pies['oSi']
pies['Interactions'] = pies['Interactions'].clip(lower=0)

# for N = 75 no stochastic component
pies['ST'] = pies['oST']
pies['Si'] = pies['oSi']
pies['Interactions'] = pies['oInt']

for out in pies['output'].unique():
    df = pd.DataFrame(pies[pies['output'] == out])
    group_size = [np.sum(df['oST'])]
    group_names = ['Deterministic']
    # External ring
    a = plt.cm.Blues
    fig, ax = plt.subplots(figsize=[9, 6])
    plt.title('Deterministic ' + ' ' + out)
    ax.axis('equal')
    # mypie, _ = ax.pie(group_size, radius=1.01, labels=group_names, colors=[a(0.8)])
    mypie, _ = ax.pie(group_size, radius=1.01, colors=[a(0.8)])
    mypie[0].set_hatch('/')
    plt.setp(mypie, width=0.15, edgecolor='white')

    # Central ring
    det_ind = 'oST'
    det_values = list(df[df[det_ind] > 0.03][det_ind])
    subgroup_size = det_values
    det_names = list(df[df[det_ind] > 0.03]['input'])

    subgroup_names = det_names
    count_det = len(det_names)
    col = []
    for i in subgroup_names:
        col.append(list(color_dic[color_dic['input'] == i]['color'])[0])
    # mypie2, _ = ax.pie(subgroup_size, radius=1-0.15, labeldistance=0.8, colors=col, labels=subgroup_names)
    mypie2, _ = ax.pie(subgroup_size, radius=1 - 0.15, labeldistance=0.8, colors=col)
    plt.setp(mypie2, width=0.3, edgecolor='white')
    plt.margins(0, 0)

    # Inner ring
    # Interactions
    df['Det interactions'] = df['oST'] - df['oSi']
    det_int = list(df[df[det_ind] > 0.03]['Det interactions'])
    # Direct effects
    det_direct = list(df[df['oST'] > 0.03]['oSi'])

    third_group = intercalate_lists(det_direct, det_int)

    a, b = [plt.cm.Greys, plt.cm.Greys]
    col = []
    for i in range(len(third_group)):
        if i % 2 == 0:
            col.append(a(0.8))
        else:
            col.append(b(0.2))
    cheat = [' ']*len(third_group)
    mypie3, _ = ax.pie(third_group, radius=1-0.45, colors=col, labels=cheat)
    handles, labels = ax.get_legend_handles_labels()
    legs = ['Direct effects', 'Interactions']
    legend = ax.legend(handles[-3:], legs, loc=(0.9, 0.1))
    legend.remove()
    plt.setp(mypie3, width=0.06, edgecolor='white')
    plt.tight_layout()
    plt.savefig(outpath + '0105_' + out + '_piechart' + str(75) + '_no_sto' + '.png')

