import warnings

import numpy as np
from scipy.signal import convolve2d as conv2

warnings.simplefilter("ignore")  # ignore numpy incompatability warning (harmless)


"""
This code is from ContextLab at: https://github.com/ContextLab/seqnmf
"""


def get_shapes(W, H, force_full=False):
    N = W.shape[0]
    T = H.shape[1]
    K = W.shape[1]
    L = W.shape[2]

    # trim zero padding along the L and K dimensions
    if not force_full:
        W_sum = W.sum(axis=0).sum(axis=1)
        H_sum = H.sum(axis=1)
        K = 1
        for k in np.arange(W.shape[1] - 1, 0, -1):
            if (W_sum[k] > 0) or (H_sum[k] > 0):
                K = k + 1
                break

        L = 2
        for m in np.arange(W.shape[2] - 1, 2, -1):
            W_sum = W.sum(axis=1).sum(axis=0)
            if W_sum[m] > 0:
                L = m + 1
                break

    return N, K, L, T


def trim_shapes(W, H, N, K, L, T):
    return W[:N, :K, :L], H[:K, :T]


def reconstruct(W, H):
    N, K, L, T = get_shapes(W, H, force_full=True)
    W, H = trim_shapes(W, H, N, K, L, T)

    H = np.hstack((np.zeros([K, L]), H, np.zeros([K, L])))
    T += 2 * L
    X_hat = np.zeros([N, T])

    for t in np.arange(L):
        X_hat += np.dot(W[:, :, t], np.roll(H, t - 1, axis=1))

    return X_hat[:, L:-L]


def shift_factors(W, H):
    warnings.simplefilter("ignore")  # ignore warnings for nan-related errors

    N, K, L, T = get_shapes(W, H, force_full=True)
    W, H = trim_shapes(W, H, N, K, L, T)

    if L > 1:
        center = int(np.max([np.floor(L / 2), 1]))
        Wpad = np.concatenate((np.zeros([N, K, L]), W, np.zeros([N, K, L])), axis=2)

        for i in np.arange(K):
            temp = np.sum(np.squeeze(W[:, i, :]), axis=0)
            # return temp, temp
            try:
                cmass = int(
                    np.max(
                        np.floor(np.sum(temp * np.arange(1, L + 1)) / np.sum(temp)),
                        axis=0,
                    )
                )
            except ValueError:
                cmass = center
            Wpad[:, i, :] = np.roll(np.squeeze(Wpad[:, i, :]), center - cmass, axis=1)
            H[i, :] = np.roll(H[i, :], cmass - center, axis=0)

    return Wpad[:, :, L:-L], H


def compute_loadings_percent_power(V, W, H):
    N, K, L, T = get_shapes(W, H)
    W, H = trim_shapes(W, H, N, K, L, T)

    loadings = np.zeros(K)
    var_v = np.sum(np.power(V, 2))

    for i in np.arange(K):
        WH = reconstruct(
            np.reshape(W[:, i, :], [W.shape[0], 1, W.shape[2]]),
            np.reshape(H[i, :], [1, H.shape[1]]),
        )
        loadings[i] = np.divide(
            np.sum(
                np.multiply(2 * V.flatten(), WH.flatten()) - np.power(WH.flatten(), 2)
            ),
            var_v,
        )

    loadings[loadings < 0] = 0
    return loadings


def seqNMF(
    X,
    K=10,
    L=100,
    Lambda=0.001,
    W_init=None,
    H_init=None,
    max_iter=100,
    tol=-np.inf,
    shift=True,
    sort_factors=True,
    lambda_L1W=0,
    lambda_L1H=0,
    lambda_OrthH=0,
    lambda_OrthW=0,
    M=None,
    use_W_update=True,
    W_fixed=False,
):
    """
    :param X: an N (features) by T (timepoints) data matrix to be factorized using seqNMF
    :param K: the (maximum) number of factors to search for; any unused factors will be set to all zeros
    :param L: the (maximum) number of timepoints to consider in each factor; any unused timepoints will be set to zeros
    :param Lambda: regularization parameter (default: 0.001)
    :param W_init: initial factors (if unspecified, use random initialization)
    :param H_init: initial per-timepoint factor loadings (if unspecified, initialize randomly)
    :param plot_it: if True, display progress in each update using a plot (default: False)
    :param max_iter: maximum number of iterations/updates
    :param tol: if cost is within tol of the average of the previous 5 updates, the algorithm will terminate (default: tol = -inf)
    :param shift: allow timepoint shifts in H
    :param sort_factors: sort factors by time
    :param lambda_L1W: regularization parameter for W (default: 0)
    :param lambda_L1H: regularization parameter for H (default: 0)
    :param lambda_OrthH: regularization parameter for H (default: 0)
    :param lambda_OrthW: regularization parameter for W (default: 0)
    :param M: binary mask of the same size as X, used to ignore a subset of the data during training (default: use all data)
    :param use_W_update: set to True for more accurate results; set to False for faster results (default: True)
    :param W_fixed: if true, fix factors (W), e.g. for cross validation (default: False)

    :return:
    :W: N (features) by K (factors) by L (per-factor timepoints) tensor of factors
    :H: K (factors) by T (timepoints) matrix of factor loadings (i.e. factor timecourses)
    :cost: a vector of length (number-of-iterations + 1) containing the initial cost and cost after each update (i.e. the reconstruction error)
    :loadings: the per-factor loadings-- i.e. the explanatory power of each individual factor
    :power: the total power (across all factors) explained by the full reconstruction
    """
    N = X.shape[0]
    T = X.shape[1] + 2 * L
    X = np.concatenate((np.zeros([N, L]), X, np.zeros([N, L])), axis=1)

    if W_init is None:
        W_init = np.max(X) * np.random.rand(N, K, L)
    if H_init is None:
        H_init = np.max(X) * np.random.rand(K, T) / np.sqrt(T / 3)
    if M is None:
        M = np.ones([N, T])

    assert np.all(X >= 0), "all data values must be positive!"

    W = W_init
    H = H_init

    X_hat = reconstruct(W, H)
    mask = M == 0
    X[mask] = X_hat[mask]

    smooth_kernel = np.ones([1, (2 * L) - 1])
    eps = np.max(X) * 1e-6
    last_time = False

    cost = np.zeros([max_iter + 1, 1])
    cost[0] = np.sqrt(np.mean(np.power(X - X_hat, 2)))

    for i in np.arange(max_iter):
        if (i == max_iter - 1) or (
            (i > 6) and (cost[i + 1] + tol) > np.mean(cost[i - 6 : i])
        ):
            cost = cost[: (i + 2)]
            last_time = True
            if i > 0:
                Lambda = 0

        WTX = np.zeros([K, T])
        WTX_hat = np.zeros([K, T])
        for j in np.arange(L):
            X_shifted = np.roll(X, -j + 1, axis=1)
            X_hat_shifted = np.roll(X_hat, -j + 1, axis=1)

            WTX += np.dot(W[:, :, j].T, X_shifted)
            WTX_hat += np.dot(W[:, :, j].T, X_hat_shifted)

        if Lambda > 0:
            dRdH = np.dot(Lambda * (1 - np.eye(K)), conv2(WTX, smooth_kernel, "same"))
        else:
            dRdH = 0

        if lambda_OrthH > 0:
            dHHdH = np.dot(
                lambda_OrthH * (1 - np.eye(K)), conv2(H, smooth_kernel, "same")
            )
        else:
            dHHdH = 0

        dRdH += lambda_L1H + dHHdH

        H *= np.divide(WTX, WTX_hat + dRdH + eps)

        if shift:
            W, H = shift_factors(W, H)
            W += eps

        norms = np.sqrt(np.sum(np.power(H, 2), axis=1)).T
        H = np.dot(np.diag(np.divide(1.0, norms + eps)), H)
        for j in np.arange(L):
            W[:, :, j] = np.dot(W[:, :, j], np.diag(norms))

        if not W_fixed:
            X_hat = reconstruct(W, H)
            mask = M == 0
            X[mask] = X_hat[mask]

            if lambda_OrthW > 0:
                W_flat = np.sum(W, axis=2)
            if (Lambda > 0) and use_W_update:
                XS = conv2(X, smooth_kernel, "same")

            for j in np.arange(L):
                H_shifted = np.roll(H, j - 1, axis=1)
                XHT = np.dot(X, H_shifted.T)
                X_hat_HT = np.dot(X_hat, H_shifted.T)

                if (Lambda > 0) and use_W_update:
                    dRdW = Lambda * np.dot(np.dot(XS, H_shifted.T), (1.0 - np.eye(K)))
                else:
                    dRdW = 0

                if lambda_OrthW > 0:
                    dWWdW = np.dot(lambda_OrthW * W_flat, 1.0 - np.eye(K))
                else:
                    dWWdW = 0

                dRdW += lambda_L1W + dWWdW
                W[:, :, j] *= np.divide(XHT, X_hat_HT + dRdW + eps)

        X_hat = reconstruct(W, H)
        mask = M == 0
        X[mask] = X_hat[mask]
        cost[i + 1] = np.sqrt(np.mean(np.power(X - X_hat, 2)))

        if last_time:
            break

    X = X[:, L:-L]
    X_hat = X_hat[:, L:-L]
    H = H[:, L:-L]

    power = np.divide(
        np.sum(np.power(X, 2)) - np.sum(np.power(X - X_hat, 2)), np.sum(np.power(X, 2))
    )

    loadings = compute_loadings_percent_power(X, W, H)

    if sort_factors:
        inds = np.flip(np.argsort(loadings), 0)
        loadings = loadings[inds]

        W = W[:, inds, :]
        H = H[inds, :]

    return W, H, cost, loadings, power
