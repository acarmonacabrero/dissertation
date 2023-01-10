import os
import numpy as np
import pandas as pd
from SALib.sample import saltelli
from SALib.analyze import sobol
import complementary_functions as functions

""" SETTINGS """
problem = {
    'num_vars': 9,
    'names': ['meanwater', 'gradient', 'bet11', 'bet12', 'alp1', 'alp2', 'bet21', 'bet22', 'bet23'],
    'bounds': [[50, 150], [-20, 0], [1.5, 4.5], [2, 6], [0.1, 0.3], [0.25, 0.75], [0.2, 0.6], [0.25, 0.75],
               [-0.15, -0.05]]
}

sample_name = 'five_cities'  # Name without extension
n_rep = 125  # Number of repetitions with the same inputs (N)
sampling_size = 2048  # Sampling size (L). M = L*(K+2)

""" GENERATING SAMPLE """
param_values = saltelli.sample(problem, sampling_size, calc_second_order=False)
np.savetxt(sample_name + '.sam', param_values, delimiter=',')
functions.rep_inputs(sample_name + '.sam', sample_name + str(n_rep) + '.sam', n_rep)
