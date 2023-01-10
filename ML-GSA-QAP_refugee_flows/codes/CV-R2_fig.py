import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

folder = 'no_arms'
cv_score = pd.read_csv(folder + '/original/cv_score_1_all_vars.csv', index_col=0)

# plt.title(r'10-fold $R^2$')
cv_score.T.boxplot(rot=45)
plt.xlabel('Year')
plt.ylabel(r'10-fold CV $R^2$')
plt.tight_layout()
plt.savefig(folder + '/CV_R2.png')
