import re
import pandas as pd
import numpy as np

""""""""""""""""""""""""""""""""""""""""""""""""
""" Merging min SPEI data with refugee data """
""""""""""""""""""""""""""""""""""""""""""""""""
data = pd.read_csv('data/all_refugee_variables_average_spei.csv', index_col=0)
data.shape
data_test = data.copy()

# Append origin
nodal_spei = pd.read_csv('data/spei01-annual-min.csv', index_col=0)
nodal_spei.rename(columns={'countriesADMIN': 'state.origin.name'}, inplace=True)
spei_columns_list = ['state.origin.name']
[spei_columns_list.append('min_spei_origin_' + spei_name[1:]) for spei_name in nodal_spei.columns[1:]]
nodal_spei.columns = spei_columns_list
merged_data = pd.merge(data_test, nodal_spei, on='state.origin.name', how='left')

# Append destination
nodal_spei = pd.read_csv('data/spei01-annual-min.csv', index_col=0)
spei_columns_list = ['state.destination.name']
[spei_columns_list.append('min_spei_destination_' + spei_name[1:]) for spei_name in nodal_spei.columns[1:]]
nodal_spei.columns = spei_columns_list
merged_data = pd.merge(merged_data, nodal_spei, on='state.destination.name', how='left')

test_merged_data = merged_data[['state.origin.name', 'state.destination.name',
                                'id', 'min_spei_origin_1990', 'min_spei_destination_1990']]

# Gradient calculation
nodal_spei = pd.read_csv('data/spei01-annual-min.csv', index_col=0)
gradients_df = pd.DataFrame()
for i in nodal_spei.columns[1:]:
    var = i[1:]
    gradients_df['min_spei_gradient_' + var] = merged_data['min_spei_destination_' + var] - \
                                               merged_data['min_spei_origin_' + var]

merged_data[gradients_df.columns] = gradients_df

# test_merged_data = merged_data[['state.origin.name', 'state.destination.name',
#                                 'id', 'min_spei_origin_1990', 'min_spei_destination_1990', 'min_spei_gradient_1990']]
merged_data.shape
merged_data.to_csv('data/all_refugee_variables.csv', index=False)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" Merging old DAOME mod and new data (13 Oct 2022 """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""

new_vars = pd.read_csv('data/all_refugee_variables.csv')
old_vars = pd.read_csv('data/daome_mod.csv')

new_vars.shape

new_var_list = new_vars.columns.to_list()
r = re.compile(".*1990")
annual_var_list = pd.Series(list(filter(r.match, new_var_list))).sort_values()

r = re.compile(".*alliance")
drop_var_list = pd.Series(list(filter(r.match, new_var_list))).to_list()
new_vars.drop(drop_var_list, axis=1, inplace=True)
new_vars.shape

r = re.compile(".*arms")
drop_var_list = pd.Series(list(filter(r.match, new_var_list))).to_list()
new_vars.drop(drop_var_list, axis=1, inplace=True)
new_vars.shape

r = re.compile(".*gdppc")
drop_var_list = pd.Series(list(filter(r.match, new_var_list))).to_list()
new_vars.drop(drop_var_list, axis=1, inplace=True)
new_vars.shape

r = re.compile(".*immigrant")
drop_var_list = pd.Series(list(filter(r.match, new_var_list))).to_list()
new_vars.drop(drop_var_list, axis=1, inplace=True)
new_vars.shape

r = re.compile(".*polyarchy")
drop_var_list = pd.Series(list(filter(r.match, new_var_list))).to_list()
new_vars.drop(drop_var_list, axis=1, inplace=True)
new_vars.shape

r = re.compile(".*pts.gradient")
drop_var_list = pd.Series(list(filter(r.match, new_var_list))).to_list()
new_vars.drop(drop_var_list, axis=1, inplace=True)
new_vars.shape

r = re.compile(".*flow")
drop_var_list = pd.Series(list(filter(r.match, new_var_list))).to_list()
new_vars.drop(drop_var_list, axis=1, inplace=True)
new_vars.shape

r = re.compile(".*rivalry")
drop_var_list = pd.Series(list(filter(r.match, new_var_list))).to_list()
new_vars.drop(drop_var_list, axis=1, inplace=True)
new_vars.shape

r = re.compile(".*riv.strategic")
drop_var_list = pd.Series(list(filter(r.match, new_var_list))).to_list()
new_vars.drop(drop_var_list, axis=1, inplace=True)
new_vars.shape

r = re.compile(".*trade")
drop_var_list = pd.Series(list(filter(r.match, new_var_list))).to_list()
new_vars.drop(drop_var_list, axis=1, inplace=True)
new_vars.shape

new_var_list = new_vars.columns.to_list()
r = re.compile(".*1990")
annual_var_list2 = pd.Series(list(filter(r.match, new_var_list))).sort_values()

new_vars.drop('contiguity.any', axis=1, inplace=True)

# Merge datasets
new_vars.index = new_vars['id']
old_vars.index = old_vars['id']
# merged_data = pd.merge(old_vars, new_vars, on='id', how='left')  # This didn't work
merged_data2 = old_vars.copy()
merged_data2[new_vars.columns.to_list()] = new_vars  # Final dataframe

new_var_list = merged_data2.columns.to_list()
r = re.compile(".*1990")
annual_var_list3 = pd.Series(list(filter(r.match, new_var_list))).sort_values()

col_names = merged_data2.columns

col_names = [col_name.replace('.', '_') for col_name in col_names]
merged_data2.columns = col_names

# test = merged_data2[['id', 'state.origin.name', 'state_origin_name', 'state.destination.name', 'state_destination_name']]
# test.dropna(inplace=True)
# test.shape

stats = merged_data2.describe()
t_stats = stats.T
merged_data2.shape
merged_data3 = merged_data2.loc[:, ~merged_data2.columns.duplicated()].copy()
merged_data3.rename(columns={'remittances_2010': 'remit_2010'}, inplace=True)
merged_data3.shape
country_list = merged_data3['state_origin_name'].unique()
len(country_list)
null_sum = pd.isnull(merged_data3).sum()
null = pd.isnull(merged_data3)

merged_data3.drop(['ccode1', 'ccode2', 'dyad_id', 'icode1', 'icode2', 'min_distance', 'contiguity'], axis=1,
                  inplace=True)
merged_data3.to_csv('data/all_data.csv', index=False)

""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" Selection of factors for analysis 13 Oct 2022 """
""""""""""""""""""""""""""""""""""""""""""""""""""""""

merged_data3 = pd.read_csv('data/all_data.csv')
# test = merged_data3[['id', 'state_origin_name', 'state_destination_name']]

data_for_analysis = pd.DataFrame()
data_for_analysis[['state_origin_name', 'state_destination_name', 'id', 'contiguity_any', 'immigrant_population_1990',
                   'immigrant_population_2000', 'immigrant_population_2010']] = merged_data3[
    ['state_origin_name', 'state_destination_name', 'id', 'contiguity_any', 'immigrant_population_1990',
     'immigrant_population_2000', 'immigrant_population_2010']]

# Variables selected for the analysis with annual values
r = []
r.append(re.compile(".*alliance"))
r.append(re.compile(".*arms"))
r.append(re.compile(".*gdppc"))
r.append(re.compile(".*polyarchy"))
r.append(re.compile(".*pts.gradient"))
r.append(re.compile(".*flow"))
r.append(re.compile(".*riv_strategic"))
r.append(re.compile(".*trade"))
r.append(re.compile(".*gain_gradient"))
r.append(re.compile(".*flood_displaced_origin"))
r.append(re.compile(".*flood_max_severity_origin"))
r.append(re.compile(".*conflicts_dead_gradient"))
r.append(re.compile(".*conflicts_dead_origin"))
r.append(re.compile(".*min_spei_destination"))
r.append(re.compile(".*min_spei_origin"))

all_years_analysis_vars = []
for r_i in r:
    all_years_analysis_vars = all_years_analysis_vars + list(filter(r_i.match, merged_data3.columns))

# Selection of years of analysis
study_period = np.arange(1995, 2016, 1)

r = []
[r.append(re.compile(".*{}".format(str(year))))for year in study_period]

variables_for_analysis = []
for r_i in r:
    variables_for_analysis = variables_for_analysis + list(filter(r_i.match, all_years_analysis_vars))
variables_for_analysis = sorted(variables_for_analysis)

# Merging variables
data_for_analysis[variables_for_analysis] = merged_data3[variables_for_analysis]
data_for_analysis['trade_2015'] = data_for_analysis['trade_2014']  # Copying trade variable as it ends in 2014

#
data_for_analysis.shape
data_for_analysis.dropna(inplace=True)
data_for_analysis.shape

len(data_for_analysis['state_origin_name'].unique())
missing_countries = pd.Series(list(set(merged_data3['state_origin_name'].unique()).difference(
    data_for_analysis['state_origin_name'].unique())))
missing_countries.to_csv('data/diagnosis/missing_countries.csv')

data_for_analysis.to_csv('data/data_for_analysis.csv', index=False)
