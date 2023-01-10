import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os


def intercalate_lists(a, b):
    c = list(zip(a, b))
    return [elt for sublist in c for elt in sublist]


""" SETTINGS """
outpath = os.getcwd() + '/post_outputs/'
font = {'family': 'serif', 'weight': 'normal', 'size': 14}
matplotlib.rc('font', **font)

file_name = 'exp_S_ind.csv'
# file_name = 'var_S_ind.csv'

# for N = 1
# file_name = 'all_outputsdet_five_cities_SALib_indexes.csv'
indices = pd.read_csv(outpath + file_name)

pies = pd.DataFrame()
for i in indices['output'].unique():
    # tmp = indices[(indices['output'] == i) * (indices['N'] == 1)]
    tmp = indices[(indices['output'] == i)]
    group = tmp.groupby('input')
    tmp = group.mean()
    tmp['output'] = i
    pies = pd.concat([pies, tmp])

pies['Interactions'] = pies['ST'] - pies['S1']
pies['input'] = pies.index
# pies.columns = ['Si', 'ST', 'S1_conf', 'ST_conf', 'oSi', 'oST', 'N', 'S_exp', 'output', 'Interactions', 'input']
pies.columns = ['Si', 'ST', 'S1_conf', 'ST_conf', 'oSi', 'oST', 'S_exp', 'output', 'Interactions', 'input']
pies['oInt'] = pies['oST'] - pies['oSi']
pies['Interactions'] = pies['Interactions'].clip(lower=0)
""" PIE CHARTS FOR N=1 """
color_dic = {'input': ['meanwater', 'gradient', 'bet11', 'bet12', 'alp1', 'alp2', 'bet21', 'bet22', 'bet23', 'others'],
             'color': [[31/255, 119/255, 180/255], [255/255, 127/255, 14/255], [44/255, 160/255, 44/255],
                       [214/255, 39/255, 40/255], [148/255, 103/255, 189/255], [140/255, 86/255, 75/255],
                       [227/255, 119/255, 194/255], [127/255, 127/255, 127/255], [188/255, 189/255, 34/255], [23/255, 190/255, 207/255]]
             }
color_dic = pd.DataFrame(color_dic)

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
    # plt.savefig(outpath + out + '_piechart' + str(1) + '.png')


""" PIE CHARTS WITH DISPERSION COMPONENT FOR 20210703 """
n_reps = indices['N'].unique()   # Need to add the final number of repetitions
pies = indices[indices['N'] == indices['N'].unique()[-1]]

for out in pies['output'].unique():
    df = pd.DataFrame(pies[pies['output'] == out])
    group_size = [df['S_exp'].mean()*100, (1-df['S_exp'].mean())*100]
    # group_names = ['Mean', 'Dispersion']
    group_names = ['Dispersion', 'Mean']
    # External ring
    a, b = [plt.cm.Blues, plt.cm.Reds]
    fig, ax = plt.subplots(figsize=[9, 6])
    plt.title('ST ' + out)
    ax.axis('equal')
    # ring1, _ = ax.pie(group_size, radius=1, labels=group_names, colors=[a(0.8), b(0.8)])
    # ring1, _ = ax.pie(group_size, radius=1, colors=[a(0.8), b(0.8)])
    # ring1, _ = ax.pie(group_size, radius=1, labels=group_names, colors=[b(0.8), a(0.8)])
    ring1, _ = ax.pie(group_size, radius=1, colors=[b(0.8), a(0.8)])
    plt.setp(ring1, width=0.15, edgecolor='white')

    # Second Ring
    det_values = df[df['ST'] > 0.03]['ST']
    other = (np.sum(df['ST']) - np.sum(det_values))*df['S_exp'].mean()*100/np.sum(df['ST'])
    det_values = list(df[df['ST'] > 0.03]['ST']*df['S_exp'].mean()*100/np.sum(df['ST']))
    det_values.append(other)
    sto_values = [group_size[1]]
    subgroup_size = det_values + sto_values

    det_names = list(df[df['ST'] > 0.03]['input'])
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
            # col.append(b(0.8))
            col.append(a(0.8))
    if out == 'pop3':
        subgroup_names[0] = 'all'

    # ring2, _ = ax.pie(subgroup_size, radius=1-0.15, labeldistance=0.8, colors=col, labels=subgroup_names)
    ring2, _ = ax.pie(subgroup_size, radius=1 - 0.15, labeldistance=0.8, colors=col)
    plt.setp(ring2, width=0.3, edgecolor='white')
    plt.margins(0, 0)

    # Third ring
    # Interactions
    df['Interactions'] = df['ST'] - df['Si']
    det_int = list(df[df['ST'] > 0.03]['Interactions'])
    other_int = (np.sum(df['Interactions']) - np.sum(det_int))*df['S_exp'].mean()*100/np.sum(df['ST'])
    det_int = list(df[df['ST'] > 0.03]['Interactions']*df['S_exp'].mean()*100/np.sum(df['ST']))
    det_int.append(other_int)
    # Direct effects
    det_direct = list(df[df['ST'] > 0.03]['Si'])
    other_direct = (np.sum(df['Si']) - np.sum(det_direct))*df['S_exp'].mean()*100/np.sum(df['ST'])
    det_direct = list(df[df['ST'] > 0.03]['Si']*df['S_exp'].mean()*100/np.sum(df['ST']))
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
    # ax.legend(handles[-3:], legs, loc=(0.9, 0.1))
    plt.setp(ring3, width=0.06, edgecolor='white')
    plt.tight_layout()
    plt.savefig(outpath + out + '_piechart' + str(75) + '.png')


""" BOXPLOTS """
# indices = pd.read_csv(os.getcwd() + '/post_outputs/' + 'all_outputsdet_five_cities_SALib_indexes.csv')
# indices = pd.read_csv(os.getcwd() + '/post_outputs/' + 'all_outputssto_five_cities_SALib_indexes.csv')
indices = pd.read_csv(os.getcwd() + '/post_outputs/' + 'all_outputssto__150_five_cities_SALib_indexes.csv')
font = {'family': 'serif', 'weight': 'normal', 'size': 14}
matplotlib.rc('font', **font)

""" PLOTS USING Vn """
title = ''

indices.columns = ['Si', 'ST', 'Si_conf', 'ST_conf', 'oSi', 'oST', 'N', 'input', 'output', 'Sdet']

n_list = indices['N'].unique()
for out in indices['output'].unique():
    df = indices[(indices['output'] == out)]

    plt.figure('Si ' + out, figsize=[12, 7])
    plt.title(title + 'Si ' + out)
    for i in range(len(indices['N'].unique()) - 1):
        plt.plot([i + 0.5, i + 0.5], [0, df['ST'].max()], 'grey', linestyle='--', lw=0.5)
    sns_plot = sns.boxplot(y='Si', x='N',
                           data=df,
                           palette="colorblind",
                           hue='input',
                           width=0.9)
    # plt.ylim(0, np.max(df['Si'])+0.15*np.max(df['Si']))
    plt.ylim(0, 0.35)  # range for pop3 and pop5
    plt.xlim(-0.5, len(n_list) - 0.5)
    # plt.legend(loc=1, handlelength=1)
    legend = plt.legend()
    legend.remove()

    plt.savefig(out + '_Si.png')

    df['Interactions'] = df['ST']-df['Si']
    plt.figure('Interactions ' + out, figsize=[12, 7])
    plt.title(title + 'Interactions ' + out)
    sns.boxplot(y='Interactions', x='N',
                data=df,
                palette="colorblind",
                hue='input',
                width=0.9)

    for i in range(len(indices['N'].unique()) - 1):
        plt.plot([i + 0.5, i + 0.5], [0, 2], 'grey', linestyle='--', lw=0.5)
    plt.ylim(0, np.max(df['Interactions'])+0.15*np.max(df['Interactions']))
    plt.xlim(-0.5, len(n_list) - 0.5)
    # plt.legend(loc=2, handlelength=1)
    legend = plt.legend()
    legend.remove()

    plt.savefig(out + '_interactions.png')

    plt.figure('ST ' + out, figsize=[12, 7])
    plt.title(title + 'ST ' + out)
    sns.boxplot(y='ST', x='N',
                data=df,
                palette="colorblind",
                hue='input',
                width=0.9)
    for i in range(len(indices['N'].unique()) - 1):
        plt.plot([i + 0.5, i + 0.5], [0, 2], 'grey', linestyle='--', lw=0.5)
    # plt.ylim(0, np.max(df['ST'])+0.15*np.max(df['ST']))
    plt.ylim(0, 0.8)  # ranges for pop3 and pop5
    plt.xlim(-0.5, len(n_list) - 0.5)
    # plt.legend(loc=0, handlelength=1)
    legend = plt.legend()
    legend.remove()

    plt.savefig(out + '_ST.png')
