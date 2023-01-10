import pandas as pd
import os
import re
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
# import timeit
from sklearn.ensemble import RandomForestRegressor
os.chdir('/Users/acarmonacabrero/Dropbox (UFL)/Alvaro_Rafa/ML-GSA/ref_flow_ML-GSA')

data_ori = pd.read_csv('data/original_daome/all_rf_iso.csv')
data = data_ori.copy()
sorted(data.columns)
len(data.columns)
drop_cols = data.filter(regex=('trips.*')).columns
data = data.drop(drop_cols, axis=1)
len(data.columns)
drop_cols = data.filter(regex=('arms.2.*')).columns
data = data.drop(drop_cols, axis=1)
len(data.columns)
drop_cols = data.filter(regex=('arms.1.*')).columns
data = data.drop(drop_cols, axis=1)
len(data.columns)
riv_columns = data.filter(regex=('riv.*')).columns
regex = re.compile(r'riv.strategic.2*|riv.strategic.1*')
riv_to_drop = [i for i in riv_columns if not regex.search(i)]
print(sorted(riv_to_drop))
data = data.drop(riv_to_drop, axis=1)
riv_to_drop = data.filter(regex=('riv.strategic.positional.*')).columns
data = data.drop(riv_to_drop, axis=1)
riv_to_drop = data.filter(regex=('riv.strategic.interv.*')).columns
data = data.drop(riv_to_drop, axis=1)
riv_to_drop = data.filter(regex=('riv.strategic.spatial.*')).columns
data = data.drop(riv_to_drop, axis=1)
riv_to_drop = data.filter(regex=('riv.strategic.ideological.*')).columns
data = data.drop(riv_to_drop, axis=1)
sorted(data.columns)
len(data.columns)
drop_cols = data.filter(regex=('remit.*')).columns
data = data.drop(drop_cols, axis=1)
sorted(data.columns)
data = data.drop(['min.distance', 'immigrant.population.1960',
                  'immigrant.population.1970', 'immigrant.population.1980', 'contiguity'], axis=1)
sorted(data.columns)

data.to_csv('data/daome_mod2.csv', index=False)
data = pd.read_csv('data/daome_mod2.csv')
data = data[sorted(data.columns)]

new_columns = ['alliance_defense_1990', 'alliance_defense_1991', 'alliance_defense_1992', 'alliance_defense_1993',
               'alliance_defense_1994', 'alliance_defense_1995', 'alliance_defense_1996', 'alliance_defense_1997',
               'alliance_defense_1998', 'alliance_defense_1999', 'alliance_defense_2000', 'alliance_defense_2001',
               'alliance_defense_2002', 'alliance_defense_2003', 'alliance_defense_2004', 'alliance_defense_2005',
               'alliance_defense_2006', 'alliance_defense_2007', 'alliance_defense_2008', 'alliance_defense_2009',
               'alliance_defense_2010', 'alliance_defense_2011', 'alliance_defense_2012', 'alliance_defense_2013',
               'alliance_defense_2014', 'alliance_defense_2015', 'alliance_defense_2016', 'arms_inverse_1990',
               'arms_inverse_1991', 'arms_inverse_1992', 'arms_inverse_1993', 'arms_inverse_1994', 'arms_inverse_1995',
               'arms_inverse_1996', 'arms_inverse_1997', 'arms_inverse_1998', 'arms_inverse_1999', 'arms_inverse_2000',
               'arms_inverse_2001', 'arms_inverse_2002', 'arms_inverse_2003', 'arms_inverse_2004', 'arms_inverse_2005',
               'arms_inverse_2006', 'arms_inverse_2007', 'arms_inverse_2008', 'arms_inverse_2009', 'arms_inverse_2010',
               'arms_inverse_2011', 'arms_inverse_2012', 'arms_inverse_2013', 'arms_inverse_2014', 'arms_inverse_2015',
               'arms_inverse_2016', 'ccode1', 'ccode2', 'contiguity_any', 'dyad_id', 'gdppc_gradient_1990',
               'gdppc_gradient_1991', 'gdppc_gradient_1992', 'gdppc_gradient_1993', 'gdppc_gradient_1994',
               'gdppc_gradient_1995', 'gdppc_gradient_1996', 'gdppc_gradient_1997', 'gdppc_gradient_1998',
               'gdppc_gradient_1999', 'gdppc_gradient_2000', 'gdppc_gradient_2001', 'gdppc_gradient_2002',
               'gdppc_gradient_2003', 'gdppc_gradient_2004', 'gdppc_gradient_2005', 'gdppc_gradient_2006',
               'gdppc_gradient_2007', 'gdppc_gradient_2008', 'gdppc_gradient_2009', 'gdppc_gradient_2010',
               'gdppc_gradient_2011', 'gdppc_gradient_2012', 'gdppc_gradient_2013', 'gdppc_gradient_2014',
               'gdppc_gradient_2015', 'icode_destination', 'icode_origin', 'icode1', 'icode2', 'id', 'iid',
               'immigrant_population_1990', 'immigrant_population_2000', 'immigrant_population_2010',
               'polyarchy_additive_gradient_1990', 'polyarchy_additive_gradient_1991',
               'polyarchy_additive_gradient_1992', 'polyarchy_additive_gradient_1993',
               'polyarchy_additive_gradient_1994', 'polyarchy_additive_gradient_1995',
               'polyarchy_additive_gradient_1996', 'polyarchy_additive_gradient_1997',
               'polyarchy_additive_gradient_1998', 'polyarchy_additive_gradient_1999',
               'polyarchy_additive_gradient_2000', 'polyarchy_additive_gradient_2001',
               'polyarchy_additive_gradient_2002', 'polyarchy_additive_gradient_2003',
               'polyarchy_additive_gradient_2004', 'polyarchy_additive_gradient_2005',
               'polyarchy_additive_gradient_2006', 'polyarchy_additive_gradient_2007',
               'polyarchy_additive_gradient_2008', 'polyarchy_additive_gradient_2009',
               'polyarchy_additive_gradient_2010', 'polyarchy_additive_gradient_2011',
               'polyarchy_additive_gradient_2012', 'polyarchy_additive_gradient_2013',
               'polyarchy_additive_gradient_2014', 'polyarchy_additive_gradient_2015',
               'polyarchy_additive_gradient_2016', 'pts_gradient_2011', 'pts_gradient_2012', 'pts_gradient_2013',
               'pts_gradient_2014', 'pts_gradient_2015', 'pts_gradient_1990', 'pts_gradient_1991', 'pts_gradient_1992',
               'pts_gradient_1993', 'pts_gradient_1994', 'pts_gradient_1995', 'pts_gradient_1996', 'pts_gradient_1997',
               'pts_gradient_1998', 'pts_gradient_1999', 'pts_gradient_2000', 'pts_gradient_2001', 'pts_gradient_2002',
               'pts_gradient_2003', 'pts_gradient_2004', 'pts_gradient_2005', 'pts_gradient_2006', 'pts_gradient_2007',
               'pts_gradient_2008', 'pts_gradient_2009', 'pts_gradient_2010', 'ref_flow_1990', 'ref_flow_1991',
               'ref_flow_1992', 'ref_flow_1993', 'ref_flow_1994', 'ref_flow_1995', 'ref_flow_1996', 'ref_flow_1997',
               'ref_flow_1998', 'ref_flow_1999', 'ref_flow_2000', 'ref_flow_2001', 'ref_flow_2002', 'ref_flow_2003',
               'ref_flow_2004', 'ref_flow_2005', 'ref_flow_2006', 'ref_flow_2007', 'ref_flow_2008', 'ref_flow_2009',
               'ref_flow_2010', 'ref_flow_2011', 'ref_flow_2012', 'ref_flow_2013', 'ref_flow_2014', 'ref_flow_2015',
               'ref_flow_2016', 'riv_strategic_1990', 'riv_strategic_1991', 'riv_strategic_1992', 'riv_strategic_1993',
               'riv_strategic_1994', 'riv_strategic_1995', 'riv_strategic_1996', 'riv_strategic_1997',
               'riv_strategic_1998', 'riv_strategic_1999', 'riv_strategic_2000', 'riv_strategic_2001',
               'riv_strategic_2002', 'riv_strategic_2003', 'riv_strategic_2004', 'riv_strategic_2005',
               'riv_strategic_2006', 'riv_strategic_2007', 'riv_strategic_2008', 'riv_strategic_2009',
               'riv_strategic_2010', 'riv_strategic_2011', 'riv_strategic_2012', 'riv_strategic_2013',
               'riv_strategic_2014', 'riv_strategic_2015', 'riv_strategic_2016', 'state_destination_abb',
               'state_destination_name', 'state_origin_abb', 'state_origin_name', 'trade_1990', 'trade_1991',
               'trade_1992', 'trade_1993', 'trade_1994', 'trade_1995', 'trade_1996', 'trade_1997', 'trade_1998',
               'trade_1999', 'trade_2000', 'trade_2001', 'trade_2002', 'trade_2003', 'trade_2004', 'trade_2005',
               'trade_2006', 'trade_2007', 'trade_2008', 'trade_2009', 'trade_2010', 'trade_2011', 'trade_2012',
               'trade_2013', 'trade_2014']

data.columns = new_columns
data = data[sorted(new_columns)] # Not run yet
data = data.fillna(0)
data.to_csv('data/daome_mod.csv', index=False)


def feature_filter(data, year, lag=-1, prior_ref=True):
    # Using year - 1 as predictor. Ideally, a lag per variable
    selected_columns = data.filter(regex=(str(year + lag))).columns.to_list()
    immigrant_pop_year = int(round((year + lag)/10, 0) * 10)
    selected_columns.append('immigrant_population_' + str(immigrant_pop_year))
    selected_columns.append('contiguity_any')
    if not prior_ref:
        selected_columns.remove('ref_flow_' + str(year + lag))
    [selected_columns.append(item) for item in ['ccode1', 'ccode2', 'state_destination_abb', 'state_origin_abb']]
    if year > 2015:
        selected_columns.append('trade_2014')
    x = data[sorted(selected_columns)]
    y = data['ref_flow_' + str(year)]
    return x, y


