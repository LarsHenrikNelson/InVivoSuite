import numpy as np
from numba import njit
from numpy.linalg import LinAlgError
from scipy import interpolate

from . import kcsd2d


@njit()
def sub_lookup(dist_max, R, h, sigma, basis, dist_table_density, method_type):
    xs = np.logspace(0.0, np.log10(dist_max + 1.0), dist_table_density)
    xs = xs - 1.0  # starting from 0
    dist_table = np.zeros(len(xs))
    if method_type == "kcsc2d":
        forward_model = kcsd2d.forward_model
    for i, pos in enumerate(xs):
        dist_table[i] = forward_model(pos, R, h, sigma, basis)

    return xs, dist_table


@njit()
def values(pots, estimation_table, k_pot, lambd, n_estm, n_time, n_ele, estimate="CSD"):
    """Computes the values of the quantity of interest
    Parameters
    ----------
    estimate : 'CSD' or 'POT'
        What quantity is to be estimated
        Defaults to 'CSD'
    Returns
    -------
    estimation : np.array
        estimated quantity of shape (ngx, ngy, ngz, nt)
    """
    k_inv = np.linalg.inv(k_pot + lambd * np.identity(k_pot.shape[0]))
    estimation = np.zeros((n_estm, n_time))
    for t in range(n_time):
        beta = np.dot(k_inv, pots[:, t])
        for i in range(n_ele):
            estimation[:, t] += estimation_table[:, i] * beta[i]  # C*(x) Eq 18
    return estimation


@njit()
def compute_cverror(lambd, index_generator, k_pot, pots):
    """Useful for Cross validation error calculations
    Parameters
    ----------
    lambd : float
    index_generator : list
    Returns
    -------
    err : float
        the sum of the error computed.
    """
    err = 0
    for idx_train, idx_test in index_generator:
        B_train = k_pot[np.ix_(idx_train, idx_train)]
        V_train = pots[idx_train]
        V_test = pots[idx_test]
        I_matrix = np.identity(len(idx_train))
        B_new = np.matrix(B_train) + (lambd * I_matrix)
        try:
            beta_new = np.dot(np.matrix(B_new).I, np.matrix(V_train))
            B_test = k_pot[np.ix_(idx_test, idx_train)]
            V_est = np.zeros((len(idx_test), pots.shape[1]))
            for ii in range(len(idx_train)):
                for tt in range(pots.shape[1]):
                    V_est[:, tt] += beta_new[ii, tt] * B_test[:, ii]
            err += np.linalg.norm(V_est - V_test)
        except LinAlgError:
            raise LinAlgError(
                "Encoutered Singular Matrix Error: try changing ele_pos slightly"
            )
    return err


# @njit()
# def inner_func(idx_train, pots, beta_new, B_test, V_est):
#     for ii in range(len(idx_train)):
#         for tt in range(pots.shape[1]):
#             V_est[:, tt] += beta_new[ii, tt] * B_test[:, ii]
#     return V_est
