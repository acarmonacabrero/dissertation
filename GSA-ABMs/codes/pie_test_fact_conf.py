import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os


def intercalate_lists(a, b):
    c = list(zip(a, b))
    return [elt for sublist in c for elt in sublist]


color_dic = {'input': ['conf', 'meanwater', 'gradient', 'bet11', 'bet12', 'alp1', 'alp2', 'bet21', 'bet22', 'bet23', 'others'],
             'color': [[247/255, 155/255, 62/255], [31/255, 119/255, 180/255], [255/255, 127/255, 14/255], [44/255, 160/255, 44/255],
                       [214/255, 39/255, 40/255], [148/255, 103/255, 189/255], [140/255, 86/255, 75/255],
                       [227/255, 119/255, 194/255], [127/255, 127/255, 127/255], [188/255, 189/255, 34/255], [23/255, 190/255, 207/255]]
             }

color_dic = pd.DataFrame(color_dic)

""" SETTINGS """
outpath = os.getcwd() + '/post_outputs/'
font = {'family': 'serif', 'weight': 'normal', 'size': 14}
matplotlib.rc('font', **font)

file_name = 'fac_S_ind.csv'

indices = pd.read_csv(outpath + file_name)

pies = pd.DataFrame()
for i in indices['output'].unique():
    tmp = indices[(indices['output'] == i)]
    group = tmp.groupby('input')
    tmp = group.mean()
    tmp['output'] = i
    pies = pd.concat([pies, tmp])

pies['Interactions'] = pies['ST'] - pies['S1']
pies['input'] = pies.index
pies.columns = ['Si', 'ST', 'oSi', 'oST', 'S_exp', 'output', 'Interactions', 'input']
pies['oInt'] = pies['oST'] - pies['oSi']
pies['Interactions'] = pies['Interactions'].clip(lower=0)

""" PIE CHARTS WITH DISPERSION COMPONENT FOR 20210703 """

for out in pies['output'].unique():
    df = pd.DataFrame(pies[pies['output'] == out])
    group_size = [df['S_exp'].mean()*100, (1-df['S_exp'].mean())*100]
    group_names = ['Deterministic', 'Stochastic']
    # group_names = ['Dispersion', 'Mean']
    # External ring
    a, b = [plt.cm.Blues, plt.cm.Reds]
    fig, ax = plt.subplots(figsize=[9, 6])
    plt.title('ST ' + out)
    ax.axis('equal')
    patterns = ['/', '\\']
    ring1, _ = ax.pie(group_size, radius=1, colors=[a(0.8), b(0.8)])
    ring1[0].set_hatch(patterns[0])
    ring1[1].set_hatch(patterns[1])
    # ring1, _ = ax.pie(group_size, radius=1, labels=group_names, colors=[b(0.8), a(0.8)])
    plt.setp(ring1, width=0.15, edgecolor='white')

    # Second Ring
    det_values = df[df['ST'] > 0.05]['ST']
    other = (np.sum(df['ST']) - np.sum(det_values))*df['S_exp'].mean()*100/np.sum(df['ST'])
    det_values = list(df[df['ST'] > 0.05]['ST']*df['S_exp'].mean()*100/np.sum(df['ST']))
    det_values.append(other)
    sto_values = [group_size[1]]
    subgroup_size = det_values + sto_values

    det_names = list(df[df['ST'] > 0.05]['input'])
    det_names.append('others')
    sto_names = list(' ')

    subgroup_names = det_names + sto_names

    count_det = len(det_names)
    count_sto = len(sto_names)

    col = []
    for i in subgroup_names:
        if i != ' ':
            col.append(list(color_dic[color_dic['input'] == i]['color'])[0])
        else:
            col.append(b(0.8))
            # col.append(a(0.8))

    ring2, _ = ax.pie(subgroup_size, radius=1-0.15, labeldistance=0.8, colors=col)
    ring2[-1].set_hatch(patterns[1])
    plt.setp(ring2, width=0.3, edgecolor='white')
    plt.margins(0, 0)

    # Third ring
    # Interactions
    df['Interactions'] = df['ST'] - df['Si']
    det_int = list(df[df['ST'] > 0.05]['Interactions'])
    other_int = (np.sum(df['Interactions']) - np.sum(det_int))*df['S_exp'].mean()*100/np.sum(df['ST'])
    det_int = list(df[df['ST'] > 0.05]['Interactions']*df['S_exp'].mean()*100/np.sum(df['ST']))
    det_int.append(other_int)
    # Direct effects
    det_direct = list(df[df['ST'] > 0.05]['Si'])
    other_direct = (np.sum(df['Si']) - np.sum(det_direct))*df['S_exp'].mean()*100/np.sum(df['ST'])
    det_direct = list(df[df['ST'] > 0.05]['Si']*df['S_exp'].mean()*100/np.sum(df['ST']))
    det_direct.append(other_direct)

    third_group = intercalate_lists(det_direct, det_int)
    df['Sto interactions'] = group_size[1]
    sto_int = [group_size[1]]

    third_group_sto = sto_int
    third_group = third_group + third_group_sto

    a, b = [plt.cm.Greys, plt.cm.Greys]
    col = []
    for i in range(len(third_group)-1):
        if i % 2 == 0:
            col.append(a(0.8))
        else:
            col.append(b(0.2))
    col.append(b(0))
    cheat = [' ']*len(third_group)
    third_group = [x if x >= 0 else 0 for x in third_group]
    ring3, _ = ax.pie(third_group, radius=1-0.45, colors=col, labels=cheat)
    handles, labels = ax.get_legend_handles_labels()
    legs = ['Direct effects', 'Interactions']
    plt.setp(ring3, width=0.06, edgecolor='white')
    plt.tight_layout()
    plt.savefig(outpath + '0105_' + 'fac_' + out + '_piechart' + str(75) + '.png')

