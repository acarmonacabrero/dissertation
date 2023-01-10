import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from SALib.sample import saltelli
from SALib.analyze import sobol
import seaborn as sns
sys.path.append('/Users/acarmonacabrero/Dropbox (UFL)/functions')
import functions

font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 14}
matplotlib.rc('font', **font)

""" PLOTS USING Vn """

indices = pd.read_csv('')

# NEED TO CHANGE Si TO S1 OR ST TO St OR Interactions to S1-ST in the code below
set = ''  # '_Verr' or ''
title = ''  # 'Stochastic ' or ''

for out in output.columns:
    df = pd.read_csv(out + set +'_five_cities_SALib_indexes.csv')
    df['n'] = df['n'].astype('int32')
    df = df.replace('meanwater', 'mean water')
    df.columns = ['Si', 'ST', 'oS1', 'oST', 'n', 'Input', 'Output', 'Sdet']

    plt.figure('Si ' + out, figsize=[12, 7])
    plt.title(title + 'Si ' + out)
    for i in range(0, 8):
        plt.plot([i + 0.5, i + 0.5], [0, 1], 'grey', linestyle='--', lw=0.5)
    sns_plot = sns.boxplot(y='Si', x='n',
                           data=df,
                           palette="colorblind",
                           hue='Input',
                           width=0.9)
    plt.ylim(0, np.max(df['Si'])+0.15*np.max(df['Si']))
    plt.legend(loc=0, handlelength=1)

    plt.savefig(out + set +'_Si.png')

    df['Interactions'] = df['ST']-df['Si']
    plt.figure('Interactions ' + out, figsize=[12, 7])
    plt.title(title + 'Interactions ' + out)
    sns.boxplot(y='Interactions', x='n',
                data=df,
                palette="colorblind",
                hue='Input',
                width=0.9)
    for i in range(0, 8):
        plt.plot([i + 0.5, i + 0.5], [0, 1], 'grey', linestyle='--', lw=0.5)
    plt.ylim(0, np.max(df['Interactions'])+0.15*np.max(df['Interactions']))
    plt.legend(loc=0, handlelength=1)
    plt.savefig(out + set + '_interactions.png')

    plt.figure('ST ' + out, figsize=[12, 7])
    plt.title(title +  'ST ' + out)
    sns.boxplot(y='ST', x='n',
                data=df,
                palette="colorblind",
                hue='Input',
                width=0.9)
    for i in range(0, 8):
        plt.plot([i + 0.5, i + 0.5], [0, 1], 'grey', linestyle='--', lw=0.5)
    plt.ylim(0, np.max(df['ST'])+0.15*np.max(df['ST']))
    plt.legend(loc=0, handlelength=1)
    plt.savefig(out + set +'_ST.png')



