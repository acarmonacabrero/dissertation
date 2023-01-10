import pandas as pd
import numpy as np
import re
from statsmodels.stats.outliers_influence import variance_inflation_factor

data = pd.read_csv('data/all_refugee_variables.csv')
stats = data.describe()
t_stats = stats.T
country_list = data['state.origin.name'].unique()
len(country_list)
null_sum = pd.isnull(data).sum()
null = pd.isnull(data)
var = 'ref1990.flow'
var = 'min.distance'
var = 'trade.1992'
nan_id_subset = data[['id', 'state.origin.name']][null[var]]

var_list = data.columns.to_list()
r = re.compile(".*alliance")
a_var = pd.Series(list(filter(r.match, var_list))).sort_values()

######
r = re.compile(".*2007")
a_var = pd.Series(list(filter(r.match, data))).sort_values()

# 2007
reduced_vif = data[a_var]
filt = ['alliance.defense.2007', 'riv.strategic.2007', 'pts.gradient.2007', 'arms.inverse.2007',
        'trade.2007', 'gdppc.2007.gradient', 'polyarchy.additive.2007.gradient', 'gain.gradient.2007',
        'readiness.origin.2007', 'vulnerability.origin.2007', 'hdi.origin.2007', 'hdi.destination.2007',
        'flood.dead.origin.2007', 'flood.displaced.origin.2007', 'flood.max.severity.origin.2007', 'min_spei_origin_2007',
        'min_spei_gradient_2007', 'conflicts.origin.2007', 'conflicts.gradient.2007', 'conflicts.dead.origin.2007',
        'conflicts.dead.gradient.2007']

reduced_vif = reduced_vif[filt]
r2 = reduced_vif.copy()
null_sum = pd.isnull(reduced_vif).sum()

reduced_vif.dropna(inplace=True)
vif = pd.DataFrame()
# for i, variable in enumerate(reduced_vif):
#     print(variable, variance_inflation_factor(reduced_vif.values, i))
vif["VIF Factor"] = [variance_inflation_factor(reduced_vif.values, i) for i in range(reduced_vif.shape[1])]
vif.index = reduced_vif.columns

# Drop HDI destination
reduced_vif.drop('hdi.destination.2007', axis=1, inplace=True)
vif = pd.DataFrame()
null_sum = pd.isnull(reduced_vif).sum()
vif["VIF Factor"] = [variance_inflation_factor(reduced_vif.values, i) for i in range(reduced_vif.shape[1])]
vif.index = reduced_vif.columns

# Drop HDI origin
reduced_vif.drop('hdi.origin.2007', axis=1, inplace=True)
vif = pd.DataFrame()
null_sum = pd.isnull(reduced_vif).sum()
vif["VIF Factor"] = [variance_inflation_factor(reduced_vif.values, i) for i in range(reduced_vif.shape[1])]
vif.index = reduced_vif.columns

# Drop vulnerability origin
reduced_vif.drop('vulnerability.origin.2007', axis=1, inplace=True)
vif = pd.DataFrame()
null_sum = pd.isnull(reduced_vif).sum()
vif["VIF Factor"] = [variance_inflation_factor(reduced_vif.values, i) for i in range(reduced_vif.shape[1])]
vif.index = reduced_vif.columns

# Drop vulnerability origin
reduced_vif.drop('vulnerability.origin.2007', axis=1, inplace=True)
vif = pd.DataFrame()
null_sum = pd.isnull(reduced_vif).sum()
vif["VIF Factor"] = [variance_inflation_factor(reduced_vif.values, i) for i in range(reduced_vif.shape[1])]
vif.index = reduced_vif.columns


# Drop Conflicts origin
reduced_vif.drop('conflicts.origin.2007', axis=1, inplace=True)
vif = pd.DataFrame()
null_sum = pd.isnull(reduced_vif).sum()
vif["VIF Factor"] = [variance_inflation_factor(reduced_vif.values, i) for i in range(reduced_vif.shape[1])]
vif.index = reduced_vif.columns

# Drop conflicts dead gradient
reduced_vif.drop('conflicts.dead.gradient.2007', axis=1, inplace=True)
vif = pd.DataFrame()
null_sum = pd.isnull(reduced_vif).sum()
vif["VIF Factor"] = [variance_inflation_factor(reduced_vif.values, i) for i in range(reduced_vif.shape[1])]
vif.index = reduced_vif.columns


# 2000
r = re.compile(".*2000")
a_var = pd.Series(list(filter(r.match, data))).sort_values()

reduced_vif = data[a_var]
filt = ['alliance.defense.2000', 'riv.strategic.2000', 'pts.gradient.2000', 'arms.inverse.2000',
        'trade.2000', 'gdppc.2000.gradient', 'polyarchy.additive.2000.gradient', 'gain.gradient.2000',
        'readiness.origin.2000', 'vulnerability.origin.2000', 'hdi.origin.2000', 'hdi.destination.2000',
        'flood.dead.origin.2000', 'flood.displaced.origin.2000', 'flood.max.severity.origin.2000', 'min_spei_origin_2000',
        'min_spei_gradient_2000', 'conflicts.origin.2000', 'conflicts.gradient.2000', 'conflicts.dead.origin.2000',
        'conflicts.dead.gradient.2000']

reduced_vif = reduced_vif[filt]
r2 = reduced_vif.copy()
null_sum = pd.isnull(reduced_vif).sum()

reduced_vif.dropna(inplace=True)
vif = pd.DataFrame()
# for i, variable in enumerate(reduced_vif):
#     print(variable, variance_inflation_factor(reduced_vif.values, i))
vif["VIF Factor"] = [variance_inflation_factor(reduced_vif.values, i) for i in range(reduced_vif.shape[1])]
vif.index = reduced_vif.columns

# Drop HDI destination
reduced_vif.drop('hdi.destination.2000', axis=1, inplace=True)
vif = pd.DataFrame()
null_sum = pd.isnull(reduced_vif).sum()
vif["VIF Factor"] = [variance_inflation_factor(reduced_vif.values, i) for i in range(reduced_vif.shape[1])]
vif.index = reduced_vif.columns

# Drop HDI origin
reduced_vif.drop('hdi.origin.2000', axis=1, inplace=True)
vif = pd.DataFrame()
null_sum = pd.isnull(reduced_vif).sum()
vif["VIF Factor"] = [variance_inflation_factor(reduced_vif.values, i) for i in range(reduced_vif.shape[1])]
vif.index = reduced_vif.columns

# Drop vulnerability origin
reduced_vif.drop('vulnerability.origin.2000', axis=1, inplace=True)
vif = pd.DataFrame()
null_sum = pd.isnull(reduced_vif).sum()
vif["VIF Factor"] = [variance_inflation_factor(reduced_vif.values, i) for i in range(reduced_vif.shape[1])]
vif.index = reduced_vif.columns

# Drop flood displaced origin
reduced_vif.drop('flood.displaced.origin.2000', axis=1, inplace=True)
vif = pd.DataFrame()
null_sum = pd.isnull(reduced_vif).sum()
vif["VIF Factor"] = [variance_inflation_factor(reduced_vif.values, i) for i in range(reduced_vif.shape[1])]
vif.index = reduced_vif.columns

# Drop readiness origin
reduced_vif.drop('readiness.origin.2000', axis=1, inplace=True)
vif = pd.DataFrame()
null_sum = pd.isnull(reduced_vif).sum()
vif["VIF Factor"] = [variance_inflation_factor(reduced_vif.values, i) for i in range(reduced_vif.shape[1])]
vif.index = reduced_vif.columns


# Drop Conflicts origin
reduced_vif.drop('conflicts.origin.2000', axis=1, inplace=True)
vif = pd.DataFrame()
null_sum = pd.isnull(reduced_vif).sum()
vif["VIF Factor"] = [variance_inflation_factor(reduced_vif.values, i) for i in range(reduced_vif.shape[1])]
vif.index = reduced_vif.columns

# Drop conflicts dead gradient
reduced_vif.drop('conflicts.dead.gradient.2000', axis=1, inplace=True)
vif = pd.DataFrame()
null_sum = pd.isnull(reduced_vif).sum()
vif["VIF Factor"] = [variance_inflation_factor(reduced_vif.values, i) for i in range(reduced_vif.shape[1])]
vif.index = reduced_vif.columns


# 2011
r = re.compile(".*2011")
a_var = pd.Series(list(filter(r.match, data))).sort_values()

reduced_vif = data[a_var]
filt = ['alliance.defense.2011', 'riv.strategic.2011', 'arms.inverse.2011',
        'trade.2011', 'gdppc.2011.gradient', 'polyarchy.additive.2011.gradient', 'gain.gradient.2011',
        'readiness.origin.2011', 'vulnerability.origin.2011', 'hdi.origin.2011', 'hdi.destination.2011',
        'flood.dead.origin.2011', 'flood.displaced.origin.2011', 'flood.max.severity.origin.2011', 'min_spei_origin_2011',
        'min_spei_gradient_2011', 'conflicts.origin.2011', 'conflicts.gradient.2011', 'conflicts.dead.origin.2011',
        'conflicts.dead.gradient.2011']

reduced_vif = reduced_vif[filt]
r2 = reduced_vif.copy()
null_sum = pd.isnull(reduced_vif).sum()

reduced_vif.dropna(inplace=True)
vif = pd.DataFrame()
# for i, variable in enumerate(reduced_vif):
#     print(variable, variance_inflation_factor(reduced_vif.values, i))
vif["VIF Factor"] = [variance_inflation_factor(reduced_vif.values, i) for i in range(reduced_vif.shape[1])]
vif.index = reduced_vif.columns

# Drop HDI destination
reduced_vif.drop('hdi.destination.2011', axis=1, inplace=True)
vif = pd.DataFrame()
null_sum = pd.isnull(reduced_vif).sum()
vif["VIF Factor"] = [variance_inflation_factor(reduced_vif.values, i) for i in range(reduced_vif.shape[1])]
vif.index = reduced_vif.columns

# Drop HDI origin
reduced_vif.drop('hdi.origin.2011', axis=1, inplace=True)
vif = pd.DataFrame()
null_sum = pd.isnull(reduced_vif).sum()
vif["VIF Factor"] = [variance_inflation_factor(reduced_vif.values, i) for i in range(reduced_vif.shape[1])]
vif.index = reduced_vif.columns

# Drop vulnerability origin
reduced_vif.drop('readiness.origin.2011', axis=1, inplace=True)
vif = pd.DataFrame()
null_sum = pd.isnull(reduced_vif).sum()
vif["VIF Factor"] = [variance_inflation_factor(reduced_vif.values, i) for i in range(reduced_vif.shape[1])]
vif.index = reduced_vif.columns

# Drop flood displaced origin
reduced_vif.drop('flood.displaced.origin.2011', axis=1, inplace=True)
vif = pd.DataFrame()
null_sum = pd.isnull(reduced_vif).sum()
vif["VIF Factor"] = [variance_inflation_factor(reduced_vif.values, i) for i in range(reduced_vif.shape[1])]
vif.index = reduced_vif.columns

# Drop readiness origin
reduced_vif.drop('conflicts.origin.2011', axis=1, inplace=True)
vif = pd.DataFrame()
null_sum = pd.isnull(reduced_vif).sum()
vif["VIF Factor"] = [variance_inflation_factor(reduced_vif.values, i) for i in range(reduced_vif.shape[1])]
vif.index = reduced_vif.columns


# Drop Conflicts origin
reduced_vif.drop('conflicts.origin.2011', axis=1, inplace=True)
vif = pd.DataFrame()
null_sum = pd.isnull(reduced_vif).sum()
vif["VIF Factor"] = [variance_inflation_factor(reduced_vif.values, i) for i in range(reduced_vif.shape[1])]
vif.index = reduced_vif.columns

# Drop conflicts dead gradient
reduced_vif.drop('conflicts.dead.gradient.2011', axis=1, inplace=True)
vif = pd.DataFrame()
null_sum = pd.isnull(reduced_vif).sum()
vif["VIF Factor"] = [variance_inflation_factor(reduced_vif.values, i) for i in range(reduced_vif.shape[1])]
vif.index = reduced_vif.columns
