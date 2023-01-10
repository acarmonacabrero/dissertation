import os
import numpy as np
import sys
import pandas as pd
from SALib.analyze import sobol
sys.path.append('/Users/acarmonacabrero/Dropbox (UFL)/functions')
import functions

""" DEFINITION OF THE SAMPLE """
problem = {'num_vars': 9,
           'names': ['meanwater', 'gradient', 'bet11', 'bet12', 'alp1', 'alp2', 'bet21', 'bet22', 'bet23'],
           'bounds': [[75, 125], [-12.5, -7.5], [3, 5], [3, 5], [0.225, 0.375], [0.225, 0.375], [3, 5], [3, 5],
                      [3, 5]]
           }
""" """

""" Deterministic or Stochastic """
# type = 'det'
type = 'sto'

""" GSA AS N INCREASES """
if type == 'det':
    sample_size = ['1', '5', '10', '25', '75', '150']
elif type == 'sto':
    sample_size = ['5', '10', '25', '75', '150']

# sample_size = ['5']
box_size = 25  # Number of samples in the boxplot

outputs = pd.read_csv(os.getcwd() + '/pre_outputs/o_150_ADDADD_20210703.txt', header=None)
outputs.columns = ['pop1', 'pop2', 'pop3', 'pop4', 'pop5', 'exp1', 'exp2', 'exp3', 'exp4', 'exp5',
                   'min1', 'min2', 'min3', 'min4', 'min5', 'max1', 'max2', 'max3', 'max4', 'max5',
                   'kurt1', 'kurt2', 'kurt3', 'kurt4', 'kurt5']
keys = ['S1', 'ST', 'S1_conf', 'ST_conf']
outputs = outputs[['pop1', 'pop2', 'pop3', 'pop4', 'pop5', 'min1', 'min2', 'min3', 'min4', 'min5',
                   'max1', 'max2', 'max3', 'max4', 'max5']]
df, df1, df2, df3, df4 = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
df5, df6, df7, df8, df9 = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
df10, df11, df12, df13, df14 = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# for n in [150]:
for n in sample_size:
    print(n)
    for i in range(box_size):
        output = functions.extract_n3(outputs, 150, int(n))  # Extraction with replacement
        if n != '1':
            if type == 'det':
                exp_p, spar = functions.var_exp(output, int(n))
                output = exp_p
            elif type == 'sto':
                verr, spar = functions.var_sto(output, int(n))
                output = verr

        for out in output.columns:
            aa = np.array(output[out])
            sis = sobol.analyze(problem, aa, calc_second_order=False)
            sit = {x: sis[x] for x in keys}
            sit['oS1'] = sit['S1']
            sit['oST'] = sit['ST']
            sit['N'] = np.ones(len(sis['S1'])) * int(n)
            sit['input'] = problem['names']
            sit['output'] = [out]*len(sis['S1'])
            temp = pd.DataFrame(sit)
            if n != '1':
                if type == 'det':
                    temp['Sdet'] = spar[out]
                elif type == 'sto':
                    temp['Ssto'] = spar[out]
                temp['S1'] = temp['S1']*np.float(spar[out])
                temp['ST'] = temp['ST']*np.float(spar[out])

            if out == output.columns[0]:
                df = pd.concat([df, temp], ignore_index=True)
            if out == output.columns[1]:
                df1 = pd.concat([df1, temp], ignore_index=True)
            if out == output.columns[2]:
                df2 = pd.concat([df2, temp], ignore_index=True)
            if out == output.columns[3]:
                df3 = pd.concat([df3, temp], ignore_index=True)
            if out == output.columns[4]:
                df4 = pd.concat([df4, temp], ignore_index=True)
            if out == output.columns[5]:
                df5 = pd.concat([df5, temp], ignore_index=True)
            if out == output.columns[6]:
                df6 = pd.concat([df6, temp], ignore_index=True)
            if out == output.columns[7]:
                df7 = pd.concat([df7, temp], ignore_index=True)
            if out == output.columns[8]:
                df8 = pd.concat([df8, temp], ignore_index=True)
            if out == output.columns[9]:
                df9 = pd.concat([df9, temp], ignore_index=True)
            if out == output.columns[10]:
                df10 = pd.concat([df10, temp], ignore_index=True)
            if out == output.columns[11]:
                df11 = pd.concat([df11, temp], ignore_index=True)
            if out == output.columns[12]:
                df12 = pd.concat([df12, temp], ignore_index=True)
            if out == output.columns[13]:
                df13 = pd.concat([df13, temp], ignore_index=True)
            if out == output.columns[14]:
                df14 = pd.concat([df14, temp], ignore_index=True)

df['S1'] = df['S1'].clip(lower=0)
df['ST'] = df['ST'].clip(lower=0)
df1['S1'] = df1['S1'].clip(lower=0)
df1['ST'] = df1['ST'].clip(lower=0)
df2['S1'] = df2['S1'].clip(lower=0)
df2['ST'] = df2['ST'].clip(lower=0)
df3['S1'] = df3['S1'].clip(lower=0)
df3['ST'] = df3['ST'].clip(lower=0)
df4['S1'] = df4['S1'].clip(lower=0)
df4['ST'] = df4['ST'].clip(lower=0)
df5['S1'] = df5['S1'].clip(lower=0)
df5['ST'] = df5['ST'].clip(lower=0)
df6['S1'] = df6['S1'].clip(lower=0)
df6['ST'] = df6['ST'].clip(lower=0)
df7['S1'] = df7['S1'].clip(lower=0)
df7['ST'] = df7['ST'].clip(lower=0)
df8['S1'] = df8['S1'].clip(lower=0)
df8['ST'] = df8['ST'].clip(lower=0)
df9['S1'] = df9['S1'].clip(lower=0)
df9['ST'] = df9['ST'].clip(lower=0)
df10['S1'] = df10['S1'].clip(lower=0)
df10['ST'] = df10['ST'].clip(lower=0)
df11['S1'] = df11['S1'].clip(lower=0)
df11['ST'] = df11['ST'].clip(lower=0)
df12['S1'] = df12['S1'].clip(lower=0)
df12['ST'] = df12['ST'].clip(lower=0)
df13['S1'] = df13['S1'].clip(lower=0)
df13['ST'] = df13['ST'].clip(lower=0)
df14['S1'] = df14['S1'].clip(lower=0)
df14['ST'] = df14['ST'].clip(lower=0)

type = type + '2'

out = outputs.columns[0]
df.to_csv(os.getcwd() + '/post_outputs/' + type + '_' + out + '_150_five_cities_SALib_indexes.csv', index=False)
out = outputs.columns[1]
df1.to_csv(os.getcwd() + '/post_outputs/' + type + '_' + out + '_150_five_cities_SALib_indexes.csv', index=False)
out = outputs.columns[2]
df2.to_csv(os.getcwd() + '/post_outputs/' + type + '_' + out + '_150_five_cities_SALib_indexes.csv', index=False)
out = outputs.columns[3]
df3.to_csv(os.getcwd() + '/post_outputs/' + type + '_' + out + '_150_five_cities_SALib_indexes.csv', index=False)
out = outputs.columns[4]
df4.to_csv(os.getcwd() + '/post_outputs/' + type + '_' + out + '_150_five_cities_SALib_indexes.csv', index=False)
out = outputs.columns[5]
df5.to_csv(os.getcwd() + '/post_outputs/' + type + '_' + out + '_150_five_cities_SALib_indexes.csv', index=False)
out = outputs.columns[6]
df6.to_csv(os.getcwd() + '/post_outputs/' + type + '_' + out + '_150_five_cities_SALib_indexes.csv', index=False)
out = outputs.columns[7]
df7.to_csv(os.getcwd() + '/post_outputs/' + type + '_' + out + '_150_five_cities_SALib_indexes.csv', index=False)
out = outputs.columns[8]
df8.to_csv(os.getcwd() + '/post_outputs/' + type + '_' + out + '_150_five_cities_SALib_indexes.csv', index=False)
out = outputs.columns[9]
df9.to_csv(os.getcwd() + '/post_outputs/' + type + '_' + out + '_150_five_cities_SALib_indexes.csv', index=False)
out = outputs.columns[10]
df10.to_csv(os.getcwd() + '/post_outputs/' + type + '_' + out + '_150_five_cities_SALib_indexes.csv', index=False)
out = outputs.columns[11]
df11.to_csv(os.getcwd() + '/post_outputs/' + type + '_' + out + '_150_five_cities_SALib_indexes.csv', index=False)
out = outputs.columns[12]
df12.to_csv(os.getcwd() + '/post_outputs/' + type + '_' + out + '_150_five_cities_SALib_indexes.csv', index=False)
out = outputs.columns[13]
df13.to_csv(os.getcwd() + '/post_outputs/' + type + '_' + out + '_150_five_cities_SALib_indexes.csv', index=False)
out = outputs.columns[14]
df14.to_csv(os.getcwd() + '/post_outputs/' + type + '_' + out + '_150_five_cities_SALib_indexes.csv', index=False)


dflist = []
for out in outputs.columns[range(0, 5)]:
    df = pd.read_csv(os.getcwd() + '/post_outputs/' + type + '_' + out + '_150_five_cities_SALib_indexes.csv')
    dflist.append(df)

df_t = pd.concat(dflist, ignore_index=True)
df_t.to_csv(os.getcwd() + '/post_outputs/' + 'all_outputs_' + type + '_150_five_cities_SALib_indexes.csv', index=False)
