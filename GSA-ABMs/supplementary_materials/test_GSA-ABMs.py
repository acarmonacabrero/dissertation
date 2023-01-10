import pandas as pd
import numpy as np
from SALib.sample import saltelli
from SALib.analyze import sobol
from supp_functions import ishigami, compute_ishigami_si, compute_ishigami_sij, compute_ishigami_sti, summarize_det_sensitivity

# Parameters
a, b = 7, 0.1  # Ishigami parameters
l = 1024  # GSA sampling intensity
sigma = 2  # Standard deviation of the random process

""" Sample generation """
problem = {'num_vars': 3,
           'names': ['x1', 'x2', 'x3'],
           'bounds': [[-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi]]
           }
gsa_sample = saltelli.sample(problem, l, calc_second_order=True)

""" Approach 1: 1 realization """
n = 1  # Number of realizations
y = ishigami(gsa_sample, a, b, sigma=sigma)
approach_1 = sobol.analyze(problem, y, calc_second_order=True)

""" Approach 2: n realizations but no quantification of stochastic component of the variance """
n = 10  # Number of realizations
gsa_sample_real = np.repeat(gsa_sample, n, axis=0)
y = ishigami(gsa_sample_real, a, b, sigma=sigma)
y_df = pd.DataFrame(y)
y_d = pd.DataFrame(y_df.values.reshape(-1, n, y_df.shape[1]).mean(1))  # Y_d
y_d = y_d.unstack().to_numpy()
approach_2 = sobol.analyze(problem, y_d, calc_second_order=True)


""" Approach 4 (Proposed): n realizations and the stochastic component of the variance is considered """
n = 10  # Number of realizations
# Commented lines are the same as for Approach 2
# gsa_sample = saltelli.sample(problem, l, calc_second_order=True)
# gsa_sample_real = np.repeat(gsa_sample, n, axis=0)
# y = ishigami(gsa_sample_real, a, b)
# y_df = pd.DataFrame(y)
# y_d = pd.DataFrame(y_df.values.reshape(-1, n, y_df.shape[1]).mean(1))  # Y_d
# y_d = y_d.unstack().to_numpy()
y_s = pd.DataFrame(y_df.values.reshape(-1, n, y_df.shape[1]).var(1))  # Y_s
y_s = y_s.unstack().to_numpy()
vy = y_df.var(ddof=0)  # Total variance, V(Y)
""" V(Y) = V(Y|X) + E(Y|X) = Vd + Es """
v_d = y_d.var(ddof=0)  # Deterministic component of the variance, Vd
e_s = y_s.mean()  # Stochastic component of the variance, Es
s_det = v_d/vy.values   # Fraction of total variance attributed to deterministic effects, Vd/V(Y)
s_sto = e_s/vy   # Fraction of total variance attributed to stochastic effects, Es/V(Y)
approach_4 = sobol.analyze(problem, y_d, calc_second_order=True)

summary = summarize_det_sensitivity([approach_1, approach_2, approach_4], s_det)

analytical = np.append(compute_ishigami_si(a, b, sigma), compute_ishigami_sij(a, b, sigma))
analytical = np.append(analytical, compute_ishigami_sti(a, b, sigma))

summary['Analytical'] = analytical
print(summary)

""" In the case of an ABM, the variance of the noise 
is a function of the inputs and can be decomposed """
