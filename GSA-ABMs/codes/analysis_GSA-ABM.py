import os
import sys
import numpy as np
import pandas as pd
from SALib.analyze import sobol
import complementary_functions as functions

""" SETTINGS """
problem = {
    'num_vars': 9,
    'names': ['meanwater', 'gradient', 'bet11', 'bet12', 'alp1', 'alp2', 'bet21', 'bet22', 'bet23'],
    'bounds': [[50, 150], [-20, 0], [1.5, 4.5], [2, 6], [0.1, 0.3], [0.25, 0.75], [0.2, 0.6], [0.25, 0.75],
               [-0.15, -0.05]]
}

n_rep = 125  # Number of repetitions with the same inputs (N)
sample_size = ['1', '5', '10', '15', '20', '25', '50', '75', '100', '125']  # Size, N, in which the Sensitivity Indexes, S, are evaluated
box_size = 25  # Sample size for the box plots

file_name = 'five_cities' + str(n_rep)  # Name without extension
outputs = pd.read_csv(file_name + '.out')  # Output file with headers

""" ANALYSIS """
df_ym, df_yd = pd.DataFrame(), pd.DataFrame()  # Initializing DataFrames for Ym and Yd

for n in sample_size:
    for i in range(box_size):
        output = functions.subsample(outputs, n_rep, int(n))
        output = output.reset_index(drop=1)
        if n != '1':
            y_exp, v_exp, s_exp = functions.var_exp(output, int(n))
            y_dis, m_dis, s_dis = functions.var_dis(output, int(n))
        else:
            y_exp = output
        for out in output.columns:
            aa = np.array(y_exp[out])
            s_y_m = sobol.analyze(problem, aa, calc_second_order=True)
            del s_y_m['S2'], s_y_m['S2_conf']
            s_y_m['oSi'] = s_y_m['S1']
            s_y_m['oST'] = s_y_m['ST']
            s_y_m['n'] = np.ones(len(s_y_m['S1'])) * int(n)
            s_y_m['input'] = problem['names']
            s_y_m['output'] = [out]*len(s_y_m['S1'])
            temp = pd.DataFrame(s_y_m)
            if n != '1':
                temp['S_exp'] = np.float(s_exp[out])  # S_exp
                temp['S1'] = temp['S1']*np.float(s_exp[out])
                temp['ST'] = temp['ST']*np.float(s_exp[out])
            df_ym = pd.concat([df_ym, temp], ignore_index=True)

            if n != '1':
                aa = np.array(y_dis[out])
                s_y_d = sobol.analyze(problem, aa, calc_second_order=True)
                del s_y_d['S2'], s_y_d['S2_conf']
                s_y_d['oSi'] = s_y_d['S1']
                s_y_d['oST'] = s_y_d['ST']
                s_y_d['n'] = np.ones(len(s_y_d['S1'])) * int(n)
                s_y_d['input'] = problem['names']
                s_y_d['output'] = [out]*len(s_y_d['S1'])
                temp = pd.DataFrame(s_y_d)
                if n != '1':
                    temp['S_dis'] = np.float(s_dis[out])  # S_dis
                    temp['S1'] = temp['S1']*np.float(s_dis[out])
                    temp['ST'] = temp['ST']*np.float(s_dis[out])
                df_yd = pd.concat([df_yd, temp], ignore_index=True)
        if n == sample_size[-1]:
            break


df_ym['S1'] = df_ym['S1'].clip(lower=0)  # Si >= 0
df_ym['ST'] = df_ym['ST'].clip(lower=0)  # ST >= 0
df_ym = df_ym.rename(columns={'S1': 'Si', 'S1_conf': 'Si_conf'})
df_ym.to_csv(file_name + 'ind_exp.csv', index=False)

df_yd['S1'] = df_yd['S1'].clip(lower=0)  # Si >= 0
df_yd['ST'] = df_yd['ST'].clip(lower=0)  # ST >= 0
df_yd = df_yd.rename(columns={'S1': 'Si', 'S1_conf': 'Si_conf'})
df_yd.to_csv(file_name + 'ind_dis.csv', index=False)

""" FIGURES """
# Stop showing figures plt.ioff()
