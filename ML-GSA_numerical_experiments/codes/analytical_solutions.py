import numpy as np
from itertools import combinations
import copy


def compute_vi(ai):
    """
    Computes first-order variance for Sobol G function
    :param ai:
    :return:
    """
    return (1/3)/(1 + ai)**2


def compute_vint(a_list):
    """
    Computes variance of closed interactions for Sobol G function (Vij includes Vi, Vj and their interaction)
    :param a_list:
    :return:
    """
    vij = 1
    for a in a_list:
        vij = vij * (1 + compute_vi(a))
    vij = vij - 1
    return vij


def compute_vt(v):
    """
    Computes total variance for Sobol G function
    :param v:
    :return:
    """
    total_var = 1
    for v_i in v:
        total_var = total_var * (v_i + 1)
    total_var = total_var - 1
    return total_var


def compute_vti(v_list, i):
    """
    Computes VTi for Sobol G function
    :param v_list:
    :param i:
    :return:
    """
    vti = v_list[i]
    v_l = copy.deepcopy(v_list)
    del v_l[i]
    v_prod = 1
    for v_j in v_l:
        v_prod = v_prod * (1 + v_j)
    vti = vti * v_prod
    return vti


def compute_sobol_g_si(a_list):
    """
    Sobol G function Si
    :param a_list: list of a parameters
    :return: list of Si
    """
    vi_list = [compute_vi(ai) for ai in a_list]
    vt = compute_vt(vi_list)
    return list(np.array(vi_list)/vt)


def compute_sobol_g_sij(a_list, order=2, closed=0):
    """
    Ishigami Sij (no higher than second-order indexes)
    :param a_list: list of a parameters
    :param closed: flag for closed second-order indexes. If non-closed, Sij = (Vij - Vi - Vj) / V. 1: closed
    :return: list of Sij
    """
    a_ij_list = []
    v_ij_list = []
    for c in combinations(a_list, order):
        a_ij_list.append(list(c))
    for a_pair in a_ij_list:
        v_ij_list.append(compute_vint([a_pair[0], a_pair[1]]))  # Closed second-order indexes
    vi_list = [compute_vi(ai) for ai in a_list]
    s_ij_list = []
    ij_comb = []
    for c in combinations(range(len(a_list)), order):
        ij_comb.append(list(c))
    if closed == 0:
        for i in range(len(v_ij_list)):
            s_ij_list.append((v_ij_list[i] - vi_list[ij_comb[i][0]] - vi_list[ij_comb[i][1]]) / compute_vt(vi_list))
        return s_ij_list
    else:
        for i in range(len(v_ij_list)):
            s_ij_list.append((v_ij_list[i]) / compute_vt(vi_list))
        return s_ij_list


def compute_sobol_g_sti(a_list, norm=0):
    """
    Sobol G function STi using non-closed indexes
    :return: list of Sij
    :param a_list: list of a parameters
    :param norm: flag for normalized STi (Sum(STi) = 1). 0: non-normalized.
    :return: list of STi
    """
    vti_list = []
    for i in range(len(a_list)):
        vti = compute_vi(a_list[i])
        vi_list = [compute_vi(a_list[k]) for k in range(len(a_list))]
        v_l = copy.deepcopy(vi_list)
        del v_l[i]
        v_prod = 1
        for v_j in v_l:
            v_prod = v_prod * (1 + v_j)
        vti = vti * v_prod
        vti_list.append(vti)
    st_i_list = []
    for i in range(len(vi_list)):
        st_i_list.append(compute_vti(vi_list, i) / compute_vt(vi_list))
    if norm == 0:
        return st_i_list
    else:
        return [st_i/sum(st_i_list) for st_i in st_i_list]


def compute_ishigami_si(a, b):
    """
    Ishigami Si
    :param a: alpha
    :param b: beta
    :return: list of Si
    """
    s1 = (.5 + b*np.pi**4/5 + b**2*np.pi**8/50) / (.5 + a**2/8 + b*np.pi**4/5 + b**2*np.pi**8/18)
    s2 = a**2 / (8 * (.5 + a**2/8 + b*np.pi**4/5 + b**2*np.pi**8/18))
    s3 = 0
    return [s1, s2, s3]


def compute_ishigami_sij(a, b, closed=0):
    """
    Ishigami Sij (no higher than second-order indexes)
    :param a: alpha
    :param b: beta
    :param closed: flag for closed second-order indexes. If non-closed, Sij = (Vij - Vi - Vj) / V. 1: closed
    :return: list of Sij
    """
    si = compute_ishigami_si(a, b)
    if closed != 0:
        s12 = 0 + si[0] + si[1]
        s13 = (8*b**2*np.pi**8) / (225 * (.5 + a**2/8 + b*np.pi**4/5 + b**2*np.pi**8/18)) + si[0] + si[2]
        s23 = 0 + si[1] + si[2]
        return [s12, s13, s23]
    else:
        s12 = 0
        s13 = (8*b**2*np.pi**8) / (225 * (.5 + a**2/8 + b*np.pi**4/5 + b**2*np.pi**8/18))
        s23 = 0
        return [s12, s13, s23]


def compute_ishigami_sti(a, b, norm=0):
    """
    Ishigami STi using second-order non-closed indexes
    :param a: alpha
    :param b: beta
    :param norm: flag for normalized STi (Sum(STi) = 1). 0: non-normalized.
    :return: list of STi
    """
    si = compute_ishigami_si(a, b)
    sij = compute_ishigami_sij(a, b, closed=0)
    st1 = si[0] + sij[0] + sij[1]
    st2 = si[1] + sij[0] + sij[2]
    st3 = si[2] + sij[1] + sij[2]
    if norm == 0:
        return [st1, st2, st3]
    else:
        sum_st = sum([st1, st2, st3])
        return [st1/sum_st, st2/sum_st, st3/sum_st]


def sobol_y_var(a_list):
    vy = 1
    for a_i in a_list:
        gi = 1 + (1/3)/(1 + a_i)**2
        vy = vy * gi
    return vy - 1


def ishigami_y_var(a, b):
    vy = 1/2 + a**2/8 + b*np.pi**4/5 + b**2*np.pi**8/18
    return vy
