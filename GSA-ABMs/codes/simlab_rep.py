import pandas as pd
import sys
import os
sys.path.append('/Users/acarmonacabrero/Dropbox (UFL)/functions')
import functions


matrix = pd.read_csv('/Users/acarmonacabrero/Dropbox (UFL)/Alvaro_Rafa/five_cities3/ADD+ADD/HPC_files/simlab/m_five_cities3.txt', header=None, sep='\t')
matrix = matrix.drop([10, 0], axis=1)
matrix.to_csv('/Users/acarmonacabrero/Dropbox (UFL)/Alvaro_Rafa/five_cities3/factor_conf/HPC_files/simlab/m_five_cities3.txt', header=None, index=False, sep='\t')

functions.rep_inputs('/Users/acarmonacabrero/Dropbox (UFL)/Alvaro_Rafa/five_cities3/factor_conf/HPC_files/simlab/m_five_cities3.txt',
                     os.getcwd()+'/pre_outputs/m_five_cities3_75.txt', reps=75)

matrix = pd.read_csv(os.getcwd()+'/pre_outputs/m_five_cities3_75.txt', header=None, sep='\t')
