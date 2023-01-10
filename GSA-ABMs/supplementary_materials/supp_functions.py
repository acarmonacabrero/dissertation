import numpy as np
import pandas as pd


def ishigami(x, a, b, sigma, seeds=0):
    """
    Ishigami function with random X3. X_i: U[-pi, pi]
    :param x: matrix of inputs
    :param a: alpha parameter
    :param b: beta parameter
    :param sigma: standard deviation of the noise
    # :param seeds: random seed
    :return: evaluation of Ishigami function
    """
    epsilon = np.random.normal(0, sigma, len(x))
    return np.sin(x[:, 0]) + a * np.sin(x[:, 1])**2 + b * x[:, 2]**4 * np.sin(x[:, 0]) + epsilon


def compute_ishigami_si(a, b, sigma):
    vt = (.5 + a**2/8 + b*np.pi**4/5 + b**2*np.pi**8/18) + sigma**2
    s1 = (.5 + b*np.pi**4/5 + b**2*np.pi**8/50) / vt
    s2 = a**2 / (8 * vt)
    s3 = 0
    return [s1, s2, s3]


def compute_ishigami_sij(a, b, sigma):
    vt = (.5 + a**2/8 + b*np.pi**4/5 + b**2*np.pi**8/18) + sigma**2
    s12 = 0
    s13 = (8*b**2*np.pi**8) / (225 * vt)
    s23 = 0
    return [s12, s13, s23]


def compute_ishigami_sti(a, b, sigma):
    si = compute_ishigami_si(a, b, sigma)
    sij = compute_ishigami_sij(a, b, sigma)
    st1 = si[0] + sij[0] + sij[1]
    st2 = si[1] + sij[0] + sij[2]
    st3 = si[2] + sij[1] + sij[2]
    return [st1, st2, st3]


def summarize_det_sensitivity(sensitivity_list, s_det):
    """
    Summarizes results for the different Approaches
    :param sensitivity_list: List of objects from sobol.analyze(...)
    :param: s_det: fraction of the total variance explained by deterministic effects
    :return: DataFrame with sensitivity indexes under the different approaches
    """
    approaches = ['Approach ' + str(i) for i in [1, 2, 3]]
    s_det = [1, 1, float(s_det)]
    indexes = ['S_1', 'S_2', 'S_3', 'S_1,2', 'S_1,3', 'S_2,3', 'ST_1', 'ST_2', 'ST_3']
    sensitivity_df = pd.DataFrame(columns=approaches, index=indexes)

    for i, sensitivity_approach in enumerate(sensitivity_list):
            s_ind = np.append(sensitivity_approach['S1'].tolist()[0:3], sensitivity_approach['S2'][0, 1:3])
            s_ind = np.append(s_ind, sensitivity_approach['S2'][1, 2])
            s_ind = np.append(s_ind, sensitivity_approach['ST'])
            sensitivity_df['Approach ' + str(i + 1)] = s_ind
    sensitivity_df.columns = ['Approach ' + str(i) for i in [1, 2, 4]]
    return sensitivity_df * s_det

