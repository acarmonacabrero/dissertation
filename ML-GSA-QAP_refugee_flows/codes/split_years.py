import argparse
import pandas as pd
import os

# parser = argparse.ArgumentParser()
# parser.add_argument('--n_permutations', '-n_qap', help='total number of permutations')
# parser.add_argument('--n_runs_job', '-n_runs', help='number of parameters run per job')

# args = parser.parse_args()
n_qap = 1000
n_runs = 5

names = 'all_vars'
files = 's_indexes_'
folder = 'lag0_160000'
# files = 'f_imp_'
# files = 'r2_'
# files = 'cv_score_'
df = pd.read_csv(folder + '/qap/' + files + '1' + '_' + names + '_1.csv', index_col=0)
study_period = df.index.to_list()
dct = {year: [] for year in study_period}
for i in range(1, int(n_qap/n_runs) + 1):
	for j in range(1, n_runs + 1):
		df = pd.read_csv(folder + '/qap/' + files + str(i) + '_' + names + '_' + str(j) + '.csv', index_col=0)
		for year in study_period:
			dct[year].append(df.loc[year])


for year in study_period:
	df_year = pd.DataFrame(dct[year])
	df_year.to_csv(folder + '/qap_years/' + str(year) + files + '.csv')



# Code to merge two split_year files
# folder1 = 'no_arms/qap1/'
# folder2 = 'no_arms/qap2/'
# study_period = range(1995, 2016)
# for year in study_period:
# 	df1 = pd.read_csv(folder1 + str(year) + 's_indexes_.csv', index_col=0)
# 	df2 = pd.read_csv(folder2 + str(year) + 's_indexes_.csv', index_col=0)
# 	df3 = df1.append(df2)
# 	df3.to_csv('no_arms/qap_years/' + str(year) + 's_indexes_.csv')
