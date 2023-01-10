import numpy as np
import scipy.stats as stats
import pandas as pd


def ishigami_no_e(x, a, b):
    """
    Ishigami function without error. X: U[-pi, pi]
    :param x: matrix of inputs
    :param a: alpha parameter
    :param b: beta parameter
    :return: evaluation of Ishigami function
    """
    return np.sin(x[:, 0]) + a * np.sin(x[:, 1])**2 + b * x[:, 2]**4 * np.sin(x[:, 0])


def ishigami_add(x, a, b, s_e, seed=0):
    """
    Ishigami function with additive normal error (0, s_e). X: U[-pi, pi]
    :param x: matrix of inputs
    :param a: alpha parameter
    :param b: beta parameter
    :param s_e: standard deviation of additive error
    :return: evaluation of Ishigami function
    """
    if s_e == 0:
        return ishigami_no_e(x, a, b)
    else:
        np.random.seed(seed)
        err = np.random.normal(loc=0, scale=s_e, size=x.shape[0])
        return np.sin(x[:, 0]) + a * np.sin(x[:, 1])**2 + b * x[:, 2]**4 * np.sin(x[:, 0]) + err


def ishigami_mult(x, a, b, s_e, seed=0):
    """
    Ishigami function with truncated multiplicative normal error (1, s_e). X: U[-pi, pi]
    :param x: matrix of inputs
    :param a: alpha parameter
    :param b: beta parameter
    :param s_e: standard deviation of multiplicative truncated error
    :return: evaluation of Ishigami function
    """
    if s_e == 0:
        return ishigami_no_e(x, a, b)
    else:
        lower = 0
        upper = 100*s_e
        np.random.seed(seed)
        # err = stats.truncnorm.rvs(a=lower, b=upper, loc=1, scale=s_e, size=x.shape[0])
        err = np.random.normal(loc=1, scale=s_e, size=x.shape[0])
        return (np.sin(x[:, 0]) + a * np.sin(x[:, 1])**2 + b * x[:, 2]**4 * np.sin(x[:, 0])) * err


def sobol_g_no_e(a_list, x):
    """
    Sobol G function without error. X: U[0, 1]
    :param a_list:
    :param x:
    :return:
    """
    g = 1
    for i in range(len(a_list)):
        g = g * (np.abs(4*x[:, i] - 2) + a_list[i]) / (1 + a_list[i])
    return g


def sobol_g_add(a_list, x, s_e, seed=0):
    """
    Sobol G function with additive normal error (0, s_e). X: U[0, 1]
    :param a_list:
    :param x:
    :param s_e:
    :param seed:
    :return:
    """
    if s_e == 0:
        return sobol_g_no_e(a_list, x)
    else:
        np.random.seed(seed)
        err = np.random.normal(loc=0, scale=s_e, size=x.shape[0])
        g = sobol_g_no_e(a_list, x)
    return g + err


def sobol_g_mult(a_list, x, s_e, seed=0):
    """
    Sobol G function with truncated multiplicative normal error (1, s_e). X: U[0, 1]
    :param a_list:
    :param x:
    :param s_e:
    :param seed:
    :return:
    """
    if s_e == 0:
        return sobol_g_no_e(a_list, x)
    else:
        lower = 0
        upper = 100*s_e
        np.random.seed(seed)
        # err = stats.truncnorm.rvs(a=lower, b=upper, loc=1, scale=s_e, size=x.shape[0])
        err = np.random.normal(loc=1, scale=s_e, size=x.shape[0])
        g = sobol_g_no_e(a_list, x)
    return g * err


# import matplotlib.pyplot as plt
# s = 1
# stats.lognorm.stats(s, moments='mvsk')
# err = stats.lognorm.rvs(s=s, size=1000)
# plt.hist(err)
#
# s = 3
# x = np.linspace(stats.lognorm.ppf(0.01, s), stats.lognorm.ppf(0.99, s), 100)
# rv = stats.lognorm.rvs(loc=s, scale=s)
# plt.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')
# # plt.xlim(0, 3)
#
# s = 4
# ttt = stats.lognorm(s=s**0.5, loc=0, scale=np.log(s))
# plt.plot(x, ttt.pdf(x))
