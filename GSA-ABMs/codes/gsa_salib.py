import pandas as pd
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('/Users/acarmonacabrero/Dropbox (UFL)/functions')
import functions
from SALib.sample import saltelli
from SALib.analyze import sobol


outpath = '/Users/acarmonacabrero/Dropbox (UFL)/Alvaro_Rafa/five_cities3/ADD+ADD_20210703/c_five_cities3_ADDADD/post_outputs/'

""" COMPUTING S_EXP AND S_STO """
data = pd.read_csv(os.getcwd() + '/pre_outputs/o_75_ADDADD_20210703.txt', header=None)
data.columns = ['pop1', 'pop2', 'pop3', 'pop4', 'pop5', 'exp1', 'exp2', 'exp3', 'exp4', 'exp5',
                'min1', 'min2', 'min3', 'min4', 'min5', 'max1', 'max2', 'max3', 'max4', 'max5',
                'kurt1', 'kurt2', 'kurt3', 'kurt4', 'kurt5']
names = data.columns
matrix = data.to_numpy(dtype='float32')
df = pd.DataFrame(matrix)
exp = pd.DataFrame(df.values.reshape(-1, 75, df.shape[1]).mean(1))
exp.columns = names
var = pd.DataFrame(df.values.reshape(-1, 75, df.shape[1]).var(1))
var.columns = names
vy = data.var()
v_exp = exp.var(ddof=0)
s_exp = v_exp/vy
""" """


""" GENERATING SAMPLE """
problem = {'num_vars': 9,
           'names': ['meanwater', 'gradient', 'bet11', 'bet12', 'alp1', 'alp2', 'bet21', 'bet22', 'bet23'],
           'bounds': [[75, 125], [-12.5, -7.5], [3, 5], [3, 5], [0.225, 0.375], [0.225, 0.375], [3, 5], [3, 5],
                      [3, 5]]
           }

sample_name = 'five_cities3'  # Name without extension
n_rep = 75  # Number of repetitions with the same inputs (N)
sampling_size = 2046  # Sampling size (L). M = L*(K+2)
param_values = saltelli.sample(problem, sampling_size, calc_second_order=False)
param_values.shape
""" """


""" INITIALIZATION OF THE VARIABLES OF THE ANALYSIS """
keys = ['S1', 'ST', 'S1_conf', 'ST_conf']
df_exp, df_var = pd.DataFrame(), pd.DataFrame()
""" """

""" GLOBAL SENSITIVITY ANALYSIS OF THE V[E(Y)] """
aa = pd.read_csv(os.getcwd() + '/post_outputs/exp_75.out', header=None)
aa.columns = names

for out in names:
    sis = sobol.analyze(problem, np.array(aa[out]), calc_second_order=False)

    sit = {x: sis[x] for x in keys}
    sit['oS1'] = sit['S1']
    sit['oST'] = sit['ST']
    sit['input'] = problem['names']
    sit['output'] = [out] * len(sis['S1'])
    temp = pd.DataFrame(sit)

    df_exp = pd.concat([df_exp, temp], ignore_index=1)

del(temp)

""" GLOBAL SENSITIVITY ANALYSIS OF THE E[V(Y)] """
aa = pd.read_csv(os.getcwd() + '/post_outputs/var_75.out', header=None)
aa.columns = names

for out in names:
    sis = sobol.analyze(problem, np.array(aa[out]), calc_second_order=False)

    sit = {x: sis[x] for x in keys}
    sit['oS1'] = sit['S1']
    sit['oST'] = sit['ST']
    sit['input'] = problem['names']
    sit['output'] = [out] * len(sis['S1'])
    temp = pd.DataFrame(sit)

    df_var = pd.concat([df_var, temp], ignore_index=1)

del(temp)
""" """

""" COMPUTING Si AND STi CONSIDERING STOCHASTICITY """
df_exp['S_exp'] = 0
df_var['S_var'] = 0
i = 0
for out in names:
    df_exp['S1'].iloc[np.arange(0, 9, 1) + 9*i] = df_exp['oS1'].iloc[np.arange(0, 9, 1) + 9*i]*s_exp[out]
    df_exp['ST'].iloc[np.arange(0, 9, 1) + 9*i] = df_exp['oST'].iloc[np.arange(0, 9, 1) + 9*i]*s_exp[out]
    df_exp['S_exp'].iloc[np.arange(0, 9, 1) + 9*i] = s_exp[out]
    df_var['S1'].iloc[np.arange(0, 9, 1) + 9*i] = df_var['oS1'].iloc[np.arange(0, 9, 1) + 9*i]*(1-s_exp[out])
    df_var['ST'].iloc[np.arange(0, 9, 1) + 9*i] = df_var['oST'].iloc[np.arange(0, 9, 1) + 9 * i]*(1 - s_exp[out])
    df_var['S_var'].iloc[np.arange(0, 9, 1) + 9 * i] = 1 - s_exp[out]
    i += 1

df_exp[['S1', 'ST', 'S1_conf', 'ST_conf', 'oS1', 'oST']] = df_exp[['S1', 'ST', 'S1_conf', 'ST_conf', 'oS1', 'oST']].clip(lower=0)
df_var[['S1', 'ST', 'S1_conf', 'ST_conf', 'oS1', 'oST']] = df_var[['S1', 'ST', 'S1_conf', 'ST_conf', 'oS1', 'oST']].clip(lower=0)


df_exp.to_csv(outpath + 'exp_S_ind.csv', index=False)
df_var.to_csv(outpath + 'var_S_ind.csv', index=False)
