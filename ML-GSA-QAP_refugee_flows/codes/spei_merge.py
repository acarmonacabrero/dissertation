import pandas as pd

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
