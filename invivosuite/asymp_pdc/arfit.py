import numpy as np
from scipy import special


def arfit(v, pmin, pmax, selector=None, no_const=None):
    """
    ARFIT	Stepwise least squares estimation of multivariate AR model.

    [w,A,C,SBC,FPE,th]=ARFIT(v,pmin,pmax) produces estimates of the
    parameters of a multivariate AR model of order p,

        v(k,:)' = w' + A1*v(k-1,:)' +...+ Ap*v(k-p,:)' + noise(C),

    where p lies between pmin and pmax and is chosen as the optimizer
    of Schwarz's Bayesian Criterion. The input matrix v must contain
    the time series data, with columns of v representing variables
    and rows of v representing observations.  ARFIT returns least
    squares estimates of the intercept vector w, of the coefficient
    matrices A1,...,Ap (as A=[A1 ... Ap]), and of the noise covariance
    matrix C.

    As order selection criteria, ARFIT computes approximations to
    Schwarz's Bayesian Criterion and to the logarithm of Akaike's Final
    Prediction Error. The order selection criteria for models of order
    pmin:pmax are returned as the vectors SBC and FPE.

    The matrix th contains information needed for the computation of
    confidence intervals. ARMODE and ARCONF require th as input
    arguments.

    If the optional argument SELECTOR is included in the function call,
    as in ARFIT(v,pmin,pmax,SELECTOR), SELECTOR is used as the order
    selection criterion in determining the optimum model order. The
    three letter string SELECTOR must have one of the two values 'sbc'
    or 'fpe'. (By default, Schwarz's criterion SBC is used.) If the
    bounds pmin and pmax coincide, the order of the estimated model
    is p=pmin=pmax.

    If the function call contains the optional argument 'zero' as the
    fourth or fifth argument, a model of the form

           v(k,:)' = A1*v(k-1,:)' +...+ Ap*v(k-p,:)' + noise(C)

    is fitted to the time series data. That is, the intercept vector w
    is taken to be zero, which amounts to assuming that the AR(p)
    process has zero mean.

    Modified 29-Dec-00
    Author: Tapio Schneider
            tapio@gps.caltech.edu

    Args:
        v (np.array): nChan x npoints
        pmin (int): _description_
        pmax (int): _description_
        selector (string, optional): Schwarz's Bayesian Criterion (sbc) or
        logarithm of Akaike's Final Prediction Error (fpe). Defaults to None.
        no_const (string, optional): _description_. Defaults to None.

    Raises:
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    n, m = v.shape

    if pmin != np.round(pmin) | pmax != round(pmax):
        raise ValueError("Order must be integer.")

    if pmax <= pmin:
        raise ValueError("PMAX must be greater than or equal to PMIN.")

    # set defaults and check for optional arguments
    if (selector is None) and (no_const is None):
        mcor = 1
        selector = "sbc"
    elif no_const is None:
        if selector == "zero":
            mcor = 0
            selector = "sbc"
        else:
            mcor = 1
    elif (selector is not None) and (no_const is not None):
        if no_const == "zero":
            mcor = 0
        else:
            raise ValueError("Bad argument")

    ne = n - pmax
    npmax = m * pmax + mcor

    if ne <= npmax:
        raise ValueError("Time series too short.")

    R, scale = arqr(v, pmax, mcor)

    sbc, fpe = arord(R, m, mcor, ne, pmin, pmax)

    if selector == "sbc":
        iopt = np.argmin(sbc)
        # val = sbc[iopt]
    elif selector == "fpe":
        iopt = np.argmin(fpe)
        # val = fpe[iopt]
    else:
        raise ValueError("Selector must be 'sbc' or 'fpe'.")

    popt = pmin + iopt - 1  # estimated optimum order
    nps = m * popt + mcor  # number of parameter vectors of length m

    R11 = R[0:nps, 0:nps]
    R12 = R[0:nps, npmax + 1 : npmax + m]
    R22 = R[nps : npmax + m, npmax + 1 : npmax + m]

    if nps > 0:
        if mcor == 1:
            con = np.max(scale[1 : npmax + m + 1]) / scale[0]
            R11[:, 0] = R11[:, 0] * con
        Aaug, _, _, _ = np.linalg.lstsq(R11, R12)
        Aaug = Aaug.T

        if mcor == 1:
            w = Aaug[:, 0] * con
            A = Aaug[:, : nps + 1]
        else:
            w = np.zeros((m, 1))
            A = Aaug
    else:
        w = np.zeros((m, 1))
        A = Aaug

    # return covariance matrix
    dof = ne - nps
    C = R22.T.dot(R22) / dof

    # for later computation of confidence intervals return in th:
    # (i)  the inverse of U=R11'*R11, which appears in the asymptotic
    #      covariance matrix of the least squares estimator
    # (ii) the number of degrees of freedom of the residual covariance matrix
    invR11 = np.linalg.inv(R11)
    if mcor == 1:
        # undo condition improving scaling
        invR11[0, :] = invR11[0, :] * con
    Uinv = invR11.dot(invR11.T)
    th = np.array([dof, np.zeros(1, Uinv.shape[1] - 1), Uinv])

    return w, A, C, sbc, fpe, th


def arqr(v, p, mcor):
    """
    ARQR	QR factorization for least squares estimation of AR model.

    R, SCALE =ARQR(v,p,mcor) computes the QR factorization needed in
    the least squares estimation of parameters of an AR(p) model. If
    the input flag mcor equals one, a vector of intercept terms is
    being fitted. If mcor equals zero, the process v is assumed to have
    mean zero. The output argument R is the upper triangular matrix
    appearing in the QR factorization of the AR model, and SCALE is a
    vector of scaling factors used to regularize the QR factorization.

    ARQR is called by ARFIT.

    See also ARFIT.

    Modified 29-Dec-99
    Author: Tapio Schneider
            tapio@gps.caltech.edu

    Args:
        v (np.array): nChans x nTimepoints
        p (_type_): _description_
        mcor (_type_): _description_

    Returns:
        _type_: _description_
    """
    n, m = v.shape

    ne = n - p
    nps = m * p + mcor

    K = np.zeros((ne, nps + m))  # initialize K
    if mcor == 1:
        # first column of K consists of ones for estimation of intercept vector w
        K[:, 0] = np.ones(ne)

    # Assemble `predictors' u in K
    for j in range(p):
        K[:, (mcor + m * j) : (mcor + m * (j + 1))] = v[p - j - 1 : n - j - 1, :]

    # Add `observations' v (left hand side of regression model) to K
    K[:, nps : nps + m + 1] = v[p:n, :]

    # Compute regularized QR factorization of K: The regularization
    # parameter delta is chosen according to Higham's (1996) Theorem
    # 10.7 on the stability of a Cholesky factorization. Replace the
    # regularization parameter delta below by a parameter that deps
    # on the observational error if the observational error dominates
    # the rounding error (cf. Neumaier, A. and T. Schneider, 2001:
    # "Estimation of parameters and eigenmodes of multivariate
    # autoregressive models", ACM Trans. Math. Softw., 27, 27--57.).

    q = nps + m  # number of columns of K
    delta = (q**2 + q) * np.finfo(
        np.float64
    ).eps  # Higham's choice for a Cholesky factorization
    scale = np.sqrt(delta) * np.sqrt(np.sum(K**2, axis=0))
    temp = np.zeros((K.shape[0] + scale.size, scale.size))
    temp[: K.shape[0], :] = K
    temp[K.shape[0] :, :] = np.diag(scale)
    _, t = np.linalg.qr(temp)
    t = np.triu(t)
    R = np.zeros(temp.shape)
    R[: t.shape[0], : t.shape[1]] = t

    return R, scale


def arord(R, m, mcor, ne, pmin, pmax):
    imax = pmax - pmin + 1  # maximum index of output vectors

    # initialize output vectors
    sbc = np.zeros((1, imax))  # Schwarz's Bayesian Criterion
    fpe = np.zeros((1, imax))  # log of Akaike's Final Prediction Error
    logdp = np.zeros((1, imax))  # determinant of (scaled) covariance matrix
    nps = np.zeros((1, imax))  # number of parameter vectors of length m
    nps[imax] = m * pmax + mcor

    # Get lower right triangle R22 of R:
    #
    #   | R11  R12 |
    # R=|          |
    #   | 0    R22 |
    #
    R22 = R[nps[imax] + 1 : nps[imax] + m, nps[imax] + 1 : nps[imax] + m]

    # From R22, get inverse of residual cross-product matrix for model
    # of order pmax
    invR22 = np.linagl.inv(R22)
    Mp = invR22.dot(invR22.T)

    # For order selection, get determinant of residual cross-product matrix
    #       logdp = log det(residual cross-product matrix)
    logdp[imax] = 2.0 * np.log(np.abs(np.prod(np.diag(R22))))

    # Compute approximate order selection criteria for models of
    # order pmin:pmax
    i = imax
    for p in range(pmax, pmin):
        nps[i] = m * p + mcor  # number of parameter vectors of length m
    if p < pmax:
        # Downdate determinant of residual cross-product matrix
        # Rp: Part of R to be added to Cholesky factor of covariance matrix
        Rp = R[nps[i] + 1 : nps[i] + m, nps[imax] + 1 : np[imax] + m]

        # Get Mp, the downdated inverse of the residual cross-product
        # matrix, using the Woodbury formula
        L = np.linalg.cholesky(np.eye(m) + np.dot(Rp.dot(Mp), Rp.T)).T
        N = np.linalg.lstsq(L, Rp.dot(Mp))
        Mp = Mp - N.T.dot(N)

        # Get downdated logarithm of determinant
        logdp[i] = logdp[i + 1] + 2.0 * np.log(np.abs(np.prod(np.diag(L))))

    # Schwarz's Bayesian Criterion
    sbc[i] = logdp[i] / m - np.log(ne) * (ne - nps[i]) / ne

    # logarithm of Akaike's Final Prediction Error
    fpe[i] = logdp[i] / m - np.log(ne * (ne - nps[i]) / (ne + nps[i]))

    # Modified Schwarz criterion (MSC):
    # msc[i] = logdp[i]/m - (log(ne) - 2.5) * (1 - 2.5*np[i]/(ne-np[i]))

    i = i - 1  # go to next lower order

    return sbc, fpe, logdp, nps


def arres(w, A, v, k=None):
    """arres computes the time series of residuals:

    res[k,:].T = v[k+p,:].T- w - A1*v[k+p-1,:].T - ... - Ap*v[k,:].T

    of an AR(p) model with A=[A1 ... Ap].

    Also returned is the significance level siglev of the modified
    Li-McLeod portmanteau (LMP) statistic.

    Correlation matrices for the LMP statistic are computed up to lag
    k=20, which can be changed to lag k by using

    Modified 17-Dec-99
    Author: Tapio Schneider
            tapio@gps.caltech.edu

    Reference:
      Li, W. K., and A. I. McLeod, 1981: Distribution of the
          Residual Autocorrelations in Multivariate ARMA Time Series
          Models, J. Roy. Stat. Soc. B, 43, 231--239.

    Args:
        w (_type_): _description_
        A (_type_): _description_
        v (_type_): _description_
        k (_type_, optional): _description_. Defaults to None.

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    m = v.shape[1]  # dimension of state vectors
    p = A.shape[1] / m  # order of model
    n = v.shape[0]  # number of observations
    nres = n - p  # number of residuals

    # Default value for k
    if k is None:
        k = 20
    if k <= p:  # check if k is in valid range
        raise ValueError("Maximum lag of residual correlation matrices too small.")

    if k >= nres:
        raise ValueError("Maximum lag of residual correlation matrices too large.")

    w = w.T  # force w to be row vector

    # Get time series of residuals
    res = np.zeros((nres, v.shape[1]))
    y = np.arange(0, nres - 1)  # vectorized loop l=1,...,nres
    res[y, :] = v[y + p - 1, :] - np.ones((nres, 1)) * w
    for j in range(p):
        res[y, :] = res[y, :] - v[y - j + p, :] * A[:, (j - 1) * m + 1 : j * m].T

    #  of loop over l

    # Center residuals by subtraction of the mean
    res = res - np.ones((nres, 1)) * np.mean(res)

    # Compute lag zero correlation matrix of the residuals
    c0 = res.T.dot(res)
    d = np.diag(c0)
    dd = np.sqrt(d.dot(d.T))
    c0 = c0 / dd

    # Get "covariance matrix" in LMP statistic
    c0_inv = np.linalg.inv(c0)  # inverse of lag 0 correlation matrix
    rr = np.kron(c0_inv, c0_inv)  # "covariance matrix" in LMP statistic

    # Initialize LMP statistic and correlation matrices
    lmp = 0  # LMP statistic
    cl = np.zeros((m, m))  # correlation matrix
    x = np.zeros((m * m, 1))  # correlation matrix arranged as vector

    # Compute modified Li-McLeod portmanteau statistic
    for j in range(k):
        cl = (
            res[1 : nres - j, :].T.dot(res[j + 1 : nres, :])
        ) / dd  # lag l correlation matrix
        x = np.reshape((cl, m * m, 1))  # arrange cl as vector by stacking columns
        lmp = lmp + (x.T.dot(rr), x)  # sum up LMP statistic

    lmp = n * lmp + m ^ 2 * k * (k + 1) / 2 / n  # add remaining term and scale
    dof_lmp = m ^ 2 * (k - p)  # degrees of freedom for LMP statistic

    # Significance level with which hypothesis of uncorrelatedness is rejected
    siglev = 1 - special.gammainc(lmp / 2, dof_lmp / 2)

    return siglev, res
