import subprocess
import pandas as pd
from itertools import combinations


def GSA(model, n_gsa_runs, funct_name, order, k):
    """
    Does GSA of the passed model (sampling, y calculation and analysis)
    :param model: regression model to analyze
    :param n_gsa_runs: number of GSA levels
    :param funct_name: function that is analyzed (Ishigami or Sobol G function). Necessary for sampling definition
    :param order: maximum higher-order sensitivity index to be computed
    :param k: number of inputs in the function. Necessary for the Sobol G function
    :return: returns first-order and higher-order sensitivity indexes up to the defined order
    """
    # Create GSA sample
    subprocess.call(
        'Rscript --vanilla create_sample.R ' + funct_name + ' ' + str(n_gsa_runs) + ' ' + str(order) + ' ' + str(k),
        shell=True)
    gsa_sam = pd.read_csv('gsa_sample.csv')
    gsa_runs = model.predict(gsa_sam)
    gsa_runs = pd.DataFrame(gsa_runs)
    gsa_runs.to_csv('gsa_runs.csv', index=False)
    # GSA of the model
    subprocess.call(
        'Rscript --vanilla analyze_model.R ' + str(order), shell=True)
    s_indexes = pd.read_csv('S_indexes.csv')
    return s_indexes.values.tolist()[0]


def ST_GSA(model, n_gsa_runs, funct_name, k):
    """
    Does GSA of the passed model (sampling, y calculation and analysis)
    :param model: regression model to analyze
    :param n_gsa_runs: number of GSA levels
    :param funct_name: function that is analyzed (Ishigami or Sobol G function). Necessary for sampling definition
    :param k: number of inputs in the function. Necessary for the Sobol G function
    :return: returns total-order sensitivity indexes
    """
    # Create GSA sample
    subprocess.call(
        'Rscript --vanilla ST_create_sample.R ' + funct_name + ' ' + str(n_gsa_runs) + ' ' + str(k),
        shell=True)
    gsa_sam = pd.read_csv('gsa_sample.csv')
    gsa_runs = model.predict(gsa_sam)
    gsa_runs = pd.DataFrame(gsa_runs)
    gsa_runs.to_csv('ST_gsa_runs.csv', index=False)
    # GSA of the model
    subprocess.call(
        'Rscript --vanilla ST_analyze_model.R ', shell=True)
    s_indexes = pd.read_csv('ST_indexes.csv')
    return s_indexes.values.tolist()[0]
