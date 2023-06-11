import numpy as np
from numpy.random import default_rng
from scipy import linalg
from .arfit import arfit, arres


def mcarns(u, ip):
    lx, cx = u.shape
    if lx > cx:
        raise AttributeError("Input matrix u might be transposed")
    max_order = 200
    N = np.max(u.shape)

    if ip > max_order:
        raise AttributeError("ip must be less than 200")

    ef = u.copy()
    eb = u.copy()

    pf = u.dot(u.T)
    pb = pf.copy()

    m = 0

    run = True

    A = []
    B = []

    while run:
        pfhat = ef[:, m + 1 :].dot(ef[:, m + 1 :].T)
        pbhat = eb[:, m : N - 1].dot(eb[:, m : N - 1].T)
        pfbhat = ef[:, m + 1 :].dot(eb[:, m : N - 1].T)

        m = m + 1

        #  Calculate estimated partial correlation matrix - Eq. (15.98)
        #             (Nuttall-Strand algorithm only)
        RHO = linalg.solve_sylvester(
            pfhat.dot(np.linalg.inv(pf)), np.linalg.inv(pb).dot(pbhat), -2 * pfbhat
        )

        #  Update forward and backward reflection coefficients
        #  Eqs. (15.73),(15.74),(15.78) (algorithm  by Nuttall-Strand)
        AM = -RHO.dot(np.linalg.inv(pb)) * -1
        BM = -RHO.T.dot(np.linalg.inv(pf)) * -1

        A.append(AM)
        B.append(BM)

        #  Update forward and backward covariance error  - Eqs. (15.75),(15.76)
        pf = pf - np.dot(AM.dot(BM), pf)
        pb = pb - np.dot(BM.dot(AM), pb)

        #  Update forward and backward predictor coefficients - Eqs.(15.84),(15.85)
        if m != 1:
            for k in range(m - 1):
                temp1 = A[k].copy()
                A[k] = A[k] + AM.dot(B[m - k - 2])
                B[m - k - 2] = B[m - k - 2] + BM.dot(temp1)

        #  Update residuals
        Tef = ef.copy()
        ef[:, m:] = np.flip(
            np.flip(Tef[:, m:], axis=1) + AM.dot(np.flip(eb[:, m - 1 : -1], axis=1)),
            axis=1,
        )
        eb[:, m:] = np.flip(
            np.flip(eb[:, m - 1 : -1], axis=1) + BM.dot(np.flip(Tef[:, m:], axis=1)),
            axis=1,
        )

        #  Verify if model order is adequate
        if m == ip:
            A = -np.array(A)
            B = -np.array(B)
            run = False

    return pf, A, pb, B, ef, eb


def zmatrm(Y, p):
    K, T = Y.shape
    y1 = np.zeros(K * p, 1)
    y1 = np.reshape(np.flipud(Y), (K * T, 1))
    Z = np.zeros(K * p, T)
    for i in range(0, T):
        Z[:, i] = np.flipud(y1[K * i : K * i + K * p])
    return Z


def mlsmx(Y, p):
    K, T = Y.shape
    Z = zmatrm(Y, p)
    Gamma = Z @ Z.T
    U1 = np.linalg.inv(Gamma) @ Z

    SU = Y @ Y.T - (Y @ Z.T @ U1 @ Y.T)
    SU = SU / (T - K * p - 1)
    b = np.kron(U1, np.eye(K)).dot(np.reshape(Y, (K * T, 1)))
    nfe = np.reshape(np.reshape(Y, K * T, 1) - np.kron(Z.T, np.eye(K)).dot(b), (K, T))
    return b, SU, nfe


def cmlsm(u, IP):
    m, n = u.shape
    b, SU, nef = mlsmx(u, IP)
    na = np.reshape(b, m, m, IP)
    npf = SU * n
    return npf, na, nef


def mcarvm(u, ip):
    lx, cx = u.shape
    if lx > cx:
        raise AttributeError("Input matrix u might be transposed")
    max_order = 200
    N = np.max(u.shape)

    if ip > max_order:
        raise AttributeError("ip must be less than 200")

    ef = u.copy()
    eb = u.copy()

    pf = u.dot(u.T)
    pb = pf.copy()

    m = 0

    run = True

    A = []
    B = []

    while run:
        pfhat = ef[:, m + 1 :].dot(ef[:, m + 1 :].T)
        pbhat = eb[:, m : N - 1].dot(eb[:, m : N - 1].T)
        pfbhat = ef[:, m + 1 :].dot(eb[:, m : N - 1].T)

        m = m + 1

        spfhat = linalg.sqrtm(pfhat)
        spbhat = linalg.sqrtm(pbhat)
        ispfhat = np.linalg.inv(spfhat)
        ispbhat = np.linalg.inv(spbhat)
        RHO = np.dot(ispfhat.dot(pfbhat), ispbhat.T)

        #  Update forward and backward reflection coefficients
        #  Eqs. (15.73),(15.74),(15.78) (algorithm  by Nuttall-Strand)
        AM = np.dot((-1 * spfhat).dot(RHO), ispbhat)
        BM = np.dot((-1 * spbhat).dot(RHO.T), ispfhat)

        A.append(AM)
        B.append(BM)

        #  Update forward and backward covariance error  - Eqs. (15.75),(15.76)
        pf = pf - np.dot(AM.dot(BM), pf)
        pb = pb - np.dot(BM.dot(AM), pb)

        #  Update forward and backward predictor coefficients - Eqs.(15.84),(15.85)
        if m != 1:
            for k in range(m - 1):
                temp1 = A[k].copy()
                A[k] = temp1 + AM.dot(B[m - k - 2])
                B[m - k - 2] = B[m - k - 2] + BM.dot(temp1)

        #  Update residuals
        Tef = ef.copy()
        ef[:, m:] = np.flip(
            np.flip(Tef[:, m:], axis=1) + AM.dot(np.flip(eb[:, m - 1 : -1], axis=1)),
            axis=1,
        )
        eb[:, m:] = np.flip(
            np.flip(eb[:, m - 1 : -1], axis=1) + BM.dot(np.flip(Tef[:, m:], axis=1)),
            axis=1,
        )

        #  Verify if model order is adequate
        if m == ip:
            A = -np.array(A)
            B = -np.array(B)
            run = False

    return pf, A, pb, B, ef, eb


def arfitcaps(u, IP):
    v = u.T
    w, Au, C, sbc, fpe, th = arfit(v, IP, IP)
    pf = C

    if IP >= 20:
        siglev, res = arres(w, Au, v, IP + 1)
    else:
        siglev, res = arres(w, Au, v)

    # Variable 'siglev' is not used.

    ef = res.T
    A = np.zeros((len(w), len(w), IP))
    rng = default_rng(42)
    for i in range(IP):
        A[:, :, i] = Au[:, (i - 1) * len(w) + 1 : i * len(w)]
        wu = np.ceil(len(ef)) * rng.random(((w.size, w.size)))
        if len(ef) < len(v):
            ef = [ef, ef[:, wu[0]]]
        else:
            ef = ef[:, : len(v)]

    return pf, A, ef
