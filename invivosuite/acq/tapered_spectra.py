import warnings

import numpy as np
from numba import njit, prange
from scipy import linalg, interpolate, fft, signal

"""tapered_spectra is modified version of the multitaper spectral analysis
that I got from https://github.com/nipy/nitime. There are two other ways to
implement multitaper spectral analysis with python; using Scipy or the spectrum
package how ever they create numerical issues with large array sizes. Nitime does
not have the issues but I increased the speed of the code by using numba and removed
small unecessary bits of code.
"""


@njit()
def mtm_cross_spectrum(tx, ty, weights, sides="twosided"):
    r"""

    The cross-spectrum between two tapered time-series, derived from a
    multi-taper spectral estimation.

    Parameters
    ----------

    tx, ty : ndarray (K, ..., N)
       The complex DFTs of the tapered sequence

    weights : ndarray, or 2-tuple or list
       Weights can be specified as a length-2 list of weights for spectra tx
       and ty respectively. Alternatively, if tx is ty and this function is
       computing the spectral density function of a single sequence, the
       weights can be given as an ndarray of weights for the spectrum.
       Weights may be

       * scalars, if the shape of the array is (K, ..., 1)
       * vectors, with the shape of the array being the same as tx or ty

    sides : str in {'onesided', 'twosided'}
       For the symmetric spectra of a real sequence, optionally combine half
       of the frequencies and scale the duplicate frequencies in the range
       (0, F_nyquist).

    Notes
    -----

    spectral densities are always computed as

    :math:`S_{xy}^{mt}(f) = \frac{\sum_k
    [d_k^x(f)s_k^x(f)][d_k^y(f)(s_k^y(f))^{*}]}{[\sum_k
    d_k^x(f)^2]^{\frac{1}{2}}[\sum_k d_k^y(f)^2]^{\frac{1}{2}}}`

    """
    N = tx.shape[-1]
    if ty.shape != tx.shape:
        raise ValueError("shape mismatch between tx, ty")

    # pshape = list(tx.shape)

    if isinstance(weights, (list, tuple)):
        autospectrum = False
        weights_x = weights[0]
        weights_y = weights[1]
        denom = (np.abs(weights_x) ** 2).sum(axis=0) ** 0.5
        denom *= (np.abs(weights_y) ** 2).sum(axis=0) ** 0.5
    else:
        autospectrum = True
        weights_x = weights
        weights_y = weights
        denom = (np.abs(weights) ** 2).sum(axis=0)

    if sides == "onesided":
        # where the nyq freq should be
        Fn = N // 2 + 1
        truncated_slice = [slice(None)] * len(tx.shape)
        truncated_slice[-1] = slice(0, Fn)
        tsl = tuple(truncated_slice)
        tx = tx[tsl]
        ty = ty[tsl]
        # if weights.shape[-1] > 1 then make sure weights are truncated too
        if weights_x.shape[-1] > 1:
            weights_x = weights_x[tsl]
            weights_y = weights_y[tsl]
            denom = denom[tsl[1:]]

    sf = weights_x * tx
    sf *= (weights_y * ty).conj()
    sf = sf.sum(axis=0)
    sf /= denom

    if sides == "onesided":
        # dbl power at duplicated freqs
        Fl = (N + 1) // 2
        sub_slice = [slice(None)] * len(sf.shape)
        sub_slice[-1] = slice(1, Fl)
        sf[tuple(sub_slice)] *= 2

    if autospectrum:
        return sf.real
    return sf


@njit(parallel=True)
def adaptive_weights(yk, eigvals, sides="onesided", max_iter=150):
    r"""
    Perform an iterative procedure to find the optimal weights for K
    direct spectral estimators of DPSS tapered signals.

    Parameters
    ----------

    yk : ndarray (K, N)
       The K DFTs of the tapered sequences
    eigvals : ndarray, length-K
       The eigenvalues of the DPSS tapers
    sides : str
       Whether to compute weights on a one-sided or two-sided spectrum
    max_iter : int
       Maximum number of iterations for weight computation

    Returns
    -------

    weights, nu

       The weights (array like sdfs), and the
       "equivalent degrees of freedom" (array length-L)

    Notes
    -----

    The weights to use for making the multitaper estimate, such that
    :math:`S_{mt} = \sum_{k} |w_k|^2S_k^{mt} / \sum_{k} |w_k|^2`

    If there are less than 3 tapers, then the adaptive weights are not
    found. The square root of the eigenvalues are returned as weights,
    and the degrees of freedom are 2*K

    """
    K = len(eigvals)
    if len(eigvals) < 3:
        print(
            """
        Warning--not adaptively combining the spectral estimators
        due to a low number of tapers.
        """
        )
        # we'll hope this is a correct length for L
        N = yk.shape[-1]
        L = N // 2 + 1 if sides == "onesided" else N
        return (np.multiply.outer(np.sqrt(eigvals), np.ones(L)), 2 * K)
    rt_eig = np.sqrt(eigvals)

    # combine the SDfs in the traditional way in order to estimate
    # the variance of the timeseries
    N = yk.shape[1]
    sdf = mtm_cross_spectrum(yk, yk, eigvals[:, None], sides=sides)
    L = sdf.shape[-1]
    var_est = np.sum(sdf, axis=-1) / N
    bband_sup = (1 - eigvals) * var_est

    # The process is to iteratively switch solving for the following
    # two expressions:
    # (1) Adaptive Multitaper SDF:
    # S^{mt}(f) = [ sum |d_k(f)|^2 S_k(f) ]/ sum |d_k(f)|^2
    #
    # (2) Weights
    # d_k(f) = [sqrt(lam_k) S^{mt}(f)] / [lam_k S^{mt}(f) + E{B_k(f)}]
    #
    # Where lam_k are the eigenvalues corresponding to the DPSS tapers,
    # and the expected value of the broadband bias function
    # E{B_k(f)} is replaced by its full-band integration
    # (1/2pi) int_{-pi}^{pi} E{B_k(f)} = sig^2(1-lam_k)

    # start with an estimate from incomplete data--the first 2 tapers
    sdf_iter = mtm_cross_spectrum(yk[:2], yk[:2], eigvals[:2, None], sides=sides)
    # for numerical considerations, don't bother doing adaptive
    # weighting after 150 dB down

    min_pwr = sdf_iter.max() * 10 ** (-150 / 20.0)
    default_weights = np.where(sdf_iter < min_pwr)[0]
    adaptiv_weights = np.where(sdf_iter >= min_pwr)[0]

    w_def = rt_eig[:, None] * sdf_iter[default_weights]
    w_def /= eigvals[:, None] * sdf_iter[default_weights] + bband_sup[:, None]

    d_sdfs = np.abs(yk[:, adaptiv_weights]) ** 2
    if L < N:
        d_sdfs *= 2
    sdf_iter = sdf_iter[adaptiv_weights]
    yk = yk[:, adaptiv_weights]
    for n in prange(max_iter):
        d_k = rt_eig[:, None] * sdf_iter[None, :]
        d_k /= eigvals[:, None] * sdf_iter[None, :] + bband_sup[:, None]

        # Test for convergence -- this is overly conservative, since
        # iteration only stops when all frequencies have converged.
        # A better approach is to iterate separately for each freq, but
        # that is a nonvectorized algorithm.
        # sdf_iter = mtm_cross_spectrum(yk, yk, d_k, sides=sides)

        sdf_iter = np.sum(d_k**2 * d_sdfs, axis=0)
        sdf_iter /= np.sum(d_k**2, axis=0)
        # Compute the cost function from eq 5.4 in Thomson 1982
        cfn = eigvals[:, None] * (sdf_iter[None, :] - d_sdfs)
        cfn /= (eigvals[:, None] * sdf_iter[None, :] + bband_sup[:, None]) ** 2
        cfn = np.sum(cfn, axis=0)

        # there seem to be some pathological freqs sometimes ..
        # this should be a good heuristic

        if np.percentile(cfn**2, 95) < 1e-12:
            break
    else:
        # If you have reached maximum number of iterations
        # Issue a warning and return non-converged weights:
        warnings.warn("Breaking due to iterative meltdown in adaptive_weights.")
    weights = np.zeros((K, L))
    weights[:, adaptiv_weights] = d_k
    weights[:, default_weights] = w_def
    nu = 2 * (weights**2).sum(axis=-2)
    return weights, nu


def crosscov(x, y, axis=-1, all_lags=False, debias=True, normalize=True):
    r"""Returns the crosscovariance sequence between two ndarrays.
    This is performed by calling fftconvolve on x, y[::-1]

    Parameters
    ----------

    x : ndarray
    y : ndarray
    axis : time axis
    all_lags : {True/False}
       whether to return all nonzero lags, or to clip the length of s_xy
       to be the length of x and y. If False, then the zero lag covariance
       is at index 0. Otherwise, it is found at (len(x) + len(y) - 1)/2
    debias : {True/False}
       Always removes an estimate of the mean along the axis, unless
       told not to (eg X and Y are known zero-mean)

    Returns
    -------

    cxy : ndarray
       The crosscovariance function

    Notes
    -----

    cross covariance of processes x and y is defined as

    .. math::

    C_{xy}[k]=E\{(X(n+k)-E\{X\})(Y(n)-E\{Y\})^{*}\}

    where X and Y are discrete, stationary (or ergodic) random processes

    Also note that this routine is the workhorse for all auto/cross/cov/corr
    functions.
    """

    if x.shape[axis] != y.shape[axis]:
        raise ValueError("crosscov() only works on same-length sequences for now")
    if debias:
        x = x - x.mean()
        y = y - y.mean()
    slicing = [slice(d) for d in x.shape]
    slicing[axis] = slice(None, None, -1)
    cxy = signal.convolve(x, y[tuple(slicing)].conj(), mode="full")
    N = x.shape[axis]
    if normalize:
        cxy /= N
    if all_lags:
        return cxy
    slicing[axis] = slice(N - 1, 2 * N - 1)
    return cxy[tuple(slicing)]


def autocov(x, **kwargs):
    r"""Returns the autocovariance of signal s at all lags.

    Parameters
    ----------

    x : ndarray
    axis : time axis
    all_lags : {True/False}
       whether to return all nonzero lags, or to clip the length of r_xy
       to be the length of x and y. If False, then the zero lag correlation
       is at index 0. Otherwise, it is found at (len(x) + len(y) - 1)/2

    Returns
    -------

    cxx : ndarray
       The autocovariance function

    Notes
    -----

    Adheres to the definition

    .. math::

    C_{xx}[k]=E\{(X[n+k]-E\{X\})(X[n]-E\{X\})^{*}\}

    where X is a discrete, stationary (ergodic) random process
    """
    # only remove the mean once, if needed
    debias = kwargs.pop("debias", True)
    if debias:
        x = x - x.mean()
    kwargs["debias"] = False
    return crosscov(x, x, **kwargs)


def autocorr(x, **kwargs):
    r"""Returns the autocorrelation of signal s at all lags.

    Parameters
    ----------

    x : ndarray
    axis : time axis
    all_lags : {True/False}
       whether to return all nonzero lags, or to clip the length of r_xy
       to be the length of x and y. If False, then the zero lag correlation
       is at index 0. Otherwise, it is found at (len(x) + len(y) - 1)/2

    Notes
    -----

    Adheres to the definition

    .. math::

    R_{xx}[k]=E\{X[n+k]X^{*}[n]\}

    where X is a discrete, stationary (ergodic) random process

    """
    # do same computation as autocovariance,
    # but without subtracting the mean
    kwargs["debias"] = False
    return autocov(x, **kwargs)


@njit(parallel=True)
def tridisolve(d, e, b):
    """
    Symmetric tridiagonal system solver,
    from Golub and Van Loan, Matrix Computations pg 157

    Parameters
    ----------

    d : ndarray
        main diagonal stored in d[:]
    e : ndarray
        superdiagonal stored in e[:-1]
    b : ndarray
        RHS vector

    Returns
    -------

    x : ndarray
        Solution to Ax = b (if overwrite_b is False). Otherwise solution is
        stored in previous RHS vector b

    """
    N = len(b)
    # work vectors
    dw = d.copy()
    ew = e.copy()
    x = b.copy()
    for k in prange(1, N):
        # e^(k-1) = e(k-1) / d(k-1)
        # d(k) = d(k) - e^(k-1)e(k-1) / d(k-1)
        t = ew[k - 1]
        ew[k - 1] = t / dw[k - 1]
        dw[k] = dw[k] - t * ew[k - 1]
    for k in prange(1, N):
        x[k] = x[k] - ew[k - 1] * x[k - 1]
    x[N - 1] = x[N - 1] / dw[N - 1]
    for k in range(N - 2, -1, -1):
        x[k] = x[k] / dw[k] - ew[k] * x[k + 1]

    return x


def tridi_inverse_iteration(d, e, w, x0=None, rtol=1e-8):
    """Perform an inverse iteration to find the eigenvector corresponding
    to the given eigenvalue in a symmetric tridiagonal system.

    Parameters
    ----------

    d : ndarray
      main diagonal of the tridiagonal system
    e : ndarray
      offdiagonal stored in e[:-1]
    w : float
      eigenvalue of the eigenvector
    x0 : ndarray
      initial point to start the iteration
    rtol : float
      tolerance for the norm of the difference of iterates

    Returns
    -------

    e : ndarray
      The converged eigenvector

    """
    eig_diag = d - w
    if x0 is None:
        x0 = np.random.randn(len(d))
    x_prev = np.zeros_like(x0)
    norm_x = np.linalg.norm(x0)
    # the eigenvector is unique up to sign change, so iterate
    # until || |x^(n)| - |x^(n-1)| ||^2 < rtol
    x0 /= norm_x
    while np.linalg.norm(np.abs(x0) - np.abs(x_prev)) > rtol:
        x_prev = x0.copy()
        tridisolve(eig_diag, e, x0)
        norm_x = np.linalg.norm(x0)
        x0 /= norm_x
    return x0


def dpss_windows(N, NW, Kmax, interp_from=None, interp_kind="linear"):
    """
    Returns the Discrete Prolate Spheroidal Sequences of orders [0,Kmax-1]
    for a given frequency-spacing multiple NW and sequence length N.

    Parameters
    ----------
    N : int
        sequence length
    NW : float, unitless
        standardized half bandwidth corresponding to 2NW = BW/f0 = BW*N*dt
        but with dt taken as 1
    Kmax : int
        number of DPSS windows to return is Kmax (orders 0 through Kmax-1)
    interp_from : int (optional)
        The dpss can be calculated using interpolation from a set of dpss
        with the same NW and Kmax, but shorter N. This is the length of this
        shorter set of dpss windows.
    interp_kind : str (optional)
        This input variable is passed to scipy.interpolate.interp1d and
        specifies the kind of interpolation as a string ('linear', 'nearest',
        'zero', 'slinear', 'quadratic, 'cubic') or as an integer specifying the
        order of the spline interpolator to use.


    Returns
    -------
    v, e : tuple,
        v is an array of DPSS windows shaped (Kmax, N)
        e are the eigenvalues

    Notes
    -----
    Tridiagonal form of DPSS calculation from:

    Slepian, D. Prolate spheroidal wave functions, Fourier analysis, and
    uncertainty V: The discrete case. Bell System Technical Journal,
    Volume 57 (1978), 1371430
    """
    Kmax = int(Kmax)
    W = float(NW) / N
    nidx = np.arange(N, dtype="d")

    # In this case, we create the dpss windows of the smaller size
    # (interp_from) and then interpolate to the larger size (N)
    if interp_from is not None:
        if interp_from > N:
            e_s = "In dpss_windows, interp_from is: %s " % interp_from
            e_s += "and N is: %s. " % N
            e_s += "Please enter interp_from smaller than N."
            raise ValueError(e_s)
        dpss = []
        d, e = dpss_windows(interp_from, NW, Kmax)
        for this_d in d:
            x = np.arange(this_d.shape[-1])
            interp_I = interpolate.interp1d(x, this_d, kind=interp_kind)
            d_temp = interp_I(np.linspace(0, this_d.shape[-1] - 1, N, endpoint=False))

            # Rescale:
            d_temp = d_temp / np.sqrt(np.sum(d_temp**2))

            dpss.append(d_temp)

        dpss = np.array(dpss)

    else:
        # here we want to set up an optimization problem to find a sequence
        # whose energy is maximally concentrated within band [-W,W].
        # Thus, the measure lambda(T,W) is the ratio between the energy within
        # that band, and the total energy. This leads to the eigen-system
        # (A - (l1)I)v = 0, where the eigenvector corresponding to the largest
        # eigenvalue is the sequence with maximally concentrated energy. The
        # collection of eigenvectors of this system are called Slepian
        # sequences, or discrete prolate spheroidal sequences (DPSS). Only the
        # first K, K = 2NW/dt orders of DPSS will exhibit good spectral
        # concentration
        # [see http://en.wikipedia.org/wiki/Spectral_concentration_problem]

        # Here I set up an alternative symmetric tri-diagonal eigenvalue
        # problem such that
        # (B - (l2)I)v = 0, and v are our DPSS (but eigenvalues l2 != l1)
        # the main diagonal = ([N-1-2*t]/2)**2 cos(2PIW), t=[0,1,2,...,N-1]
        # and the first off-diagonal = t(N-t)/2, t=[1,2,...,N-1]
        # [see Percival and Walden, 1993]
        diagonal = ((N - 1 - 2 * nidx) / 2.0) ** 2 * np.cos(2 * np.pi * W)
        off_diag = np.zeros_like(nidx)
        off_diag[:-1] = nidx[1:] * (N - nidx[1:]) / 2.0
        # put the diagonals in LAPACK "packed" storage
        ab = np.zeros((2, N), "d")
        ab[1] = diagonal
        ab[0, 1:] = off_diag[:-1]
        # only calculate the highest Kmax eigenvalues
        w = linalg.eigvals_banded(ab, select="i", select_range=(N - Kmax, N - 1))
        w = w[::-1]

        # find the corresponding eigenvectors via inverse iteration
        t = np.linspace(0, np.pi, N)
        dpss = np.zeros((Kmax, N), "d")
        for k in range(Kmax):
            dpss[k] = tridi_inverse_iteration(
                diagonal, off_diag, w[k], x0=np.sin((k + 1) * t)
            )

    # By convention (Percival and Walden, 1993 pg 379)
    # * symmetric tapers (k=0,2,4,...) should have a positive average.
    # * antisymmetric tapers should begin with a positive lobe
    fix_symmetric = dpss[0::2].sum(axis=1) < 0
    for i, f in enumerate(fix_symmetric):
        if f:
            dpss[2 * i] *= -1
    # rather than test the sign of one point, test the sign of the
    # linear slope up to the first (largest) peak
    pk = np.argmax(np.abs(dpss[1::2, : N // 2]), axis=1)
    for i, p in enumerate(pk):
        if np.sum(dpss[2 * i + 1, :p]) < 0:
            dpss[2 * i + 1] *= -1

    # Now find the eigenvalues of the original spectral concentration problem
    # Use the autocorr sequence technique from Percival and Walden, 1993 pg 390
    dpss_rxx = autocorr(dpss) * N
    r = 4 * W * np.sinc(2 * W * nidx)
    r[0] = 2 * W
    eigvals = np.dot(dpss_rxx, r)

    return dpss, eigvals


def tapered_spectra(s, tapers, NFFT=None, low_bias=True):
    """
    Compute the tapered spectra of the rows of s.

    Parameters
    ----------

    s : ndarray, (n_arr, n_pts)
        An array whose rows are timeseries.

    tapers : ndarray or container
        Either the precomputed DPSS tapers, or the pair of parameters
        (NW, K) needed to compute K tapers of length n_pts.

    NFFT : int
        Number of FFT bins to compute

    low_bias : Boolean
        If compute DPSS, automatically select tapers corresponding to
        > 90% energy concentration.

    Returns
    -------

    t_spectra : ndarray, shaped (n_arr, K, NFFT)
      The FFT of the tapered sequences in s. First dimension is squeezed
      out if n_arr is 1.
    eigvals : ndarray
      The eigenvalues are also returned if DPSS are calculated here.

    """
    N = s.shape[-1]
    # XXX: don't allow NFFT < N -- not every implementation is so restrictive!
    if NFFT is None or NFFT < N:
        NFFT = N
    rest_of_dims = s.shape[:-1]

    s = s.reshape(int(np.product(rest_of_dims)), N)
    # de-mean this sucker
    s = s - s.mean()

    if not isinstance(tapers, np.ndarray):
        # then tapers is (NW, K)
        args = (N,) + tuple(tapers)
        dpss, eigvals = dpss_windows(*args)
        if low_bias:
            keepers = eigvals > 0.9
            dpss = dpss[keepers]
            eigvals = eigvals[keepers]
        tapers = dpss
    else:
        eigvals = None
    K = tapers.shape[0]
    sig_sl = [slice(None)] * len(s.shape)
    sig_sl.insert(len(s.shape) - 1, np.newaxis)

    # tapered.shape is (M, Kmax, N)
    tapered = s[tuple(sig_sl)] * tapers

    # compute the y_{i,k}(f) -- full FFT takes ~1.5x longer, but unpacking
    # results of real-valued FFT eats up memory
    t_spectra = fft.fft(tapered, n=NFFT, axis=-1)
    t_spectra.shape = rest_of_dims + (K, NFFT)
    if eigvals is None:
        return t_spectra
    return t_spectra, eigvals


@njit()
def jackknifed_sdf_variance(yk, eigvals, sides="onesided", adaptive=True):
    r"""
    Returns the variance of the log-sdf estimated through jack-knifing
    a group of independent sdf estimates.

    Parameters
    ----------

    yk : ndarray (K, L)
       The K DFTs of the tapered sequences
    eigvals : ndarray (K,)
       The eigenvalues corresponding to the K DPSS tapers
    sides : str, optional
       Compute the jackknife pseudovalues over as one-sided or
       two-sided spectra
    adpative : bool, optional
       Compute the adaptive weighting for each jackknife pseudovalue

    Returns
    -------

    var : The estimate for log-sdf variance

    Notes
    -----

    The jackknifed mean estimate is distributed about the true mean as
    a Student's t-distribution with (K-1) degrees of freedom, and
    standard error equal to sqrt(var). However, Thompson and Chave [1]
    point out that this variance better describes the sample mean.


    [1] Thomson D J, Chave A D (1991) Advances in Spectrum Analysis and Array
    Processing (Prentice-Hall, Englewood Cliffs, NJ), 1, pp 58-113.
    """
    K = yk.shape[0]

    # the samples {S_k} are defined, with or without weights, as
    # S_k = | x_k |**2
    # | x_k |**2 = | y_k * d_k |**2          (with adaptive weights)
    # | x_k |**2 = | y_k * sqrt(eig_k) |**2  (without adaptive weights)

    all_orders = set(range(K))
    jk_sdf = []
    # get the leave-one-out estimates -- ideally, weights are recomputed
    # for each leave-one-out. This is now the case.
    for i in range(K):
        items = list(all_orders.difference([i]))
        spectra_i = np.take(yk, items, axis=0)
        eigs_i = np.take(eigvals, items)
        if adaptive:
            # compute the weights
            weights, _ = adaptive_weights(spectra_i, eigs_i, sides=sides)
        else:
            weights = eigs_i[:, None]
        # this is the leave-one-out estimate of the sdf
        jk_sdf.append(mtm_cross_spectrum(spectra_i, spectra_i, weights, sides=sides))
    # log-transform the leave-one-out estimates and the mean of estimates
    jk_sdf = np.log(jk_sdf)
    # jk_avg should be the mean of the log(jk_sdf(i))
    jk_avg = jk_sdf.mean(axis=0)

    K = float(K)

    jk_var = jk_sdf - jk_avg
    np.power(jk_var, 2, jk_var)
    jk_var = jk_var.sum(axis=0)

    # Thompson's recommended factor, eq 18
    # Jackknifing Multitaper Spectrum Estimates
    # IEEE SIGNAL PROCESSING MAGAZINE [20] JULY 2007
    f = (K - 1) ** 2 / K / (K - 0.5)
    jk_var *= f
    return jk_var


def multi_taper_psd(
    s,
    fs=2 * np.pi,
    NW=None,
    BW=None,
    adaptive=False,
    jackknife=True,
    low_bias=True,
    sides="default",
    NFFT=None,
):
    """Returns an estimate of the PSD function of s using the multitaper
    method. If the NW product, or the BW and fs in Hz are not specified
    by the user, a bandwidth of 4 times the fundamental frequency,
    corresponding to NW = 4 will be used.

    Parameters
    ----------
    s : ndarray
       An array of sampled random processes, where the time axis is assumed to
       be on the last axis

    fs : float
        Sampling rate of the signal

    NW : float
        The normalized half-bandwidth of the data tapers, indicating a
        multiple of the fundamental frequency of the DFT (fs/N).
        Common choices are n/2, for n >= 4. This parameter is unitless
        and more MATLAB compatible. As an alternative, set the BW
        parameter in Hz. See Notes on bandwidth.

    BW : float
        The sampling-relative bandwidth of the data tapers, in Hz.

    adaptive : {True/False}
       Use an adaptive weighting routine to combine the PSD estimates of
       different tapers.

    jackknife : {True/False}
       Use the jackknife method to make an estimate of the PSD variance
       at each point.

    low_bias : {True/False}
       Rather than use 2NW tapers, only use the tapers that have better than
       90% spectral concentration within the bandwidth (still using
       a maximum of 2NW tapers)

    sides : str (optional)   [ 'default' | 'onesided' | 'twosided' ]
         This determines which sides of the spectrum to return.
         For complex-valued inputs, the default is two-sided, for real-valued
         inputs, default is one-sided Indicates whether to return a one-sided
         or two-sided

    Returns
    -------
    (freqs, psd_est, var_or_nu) : ndarrays
        The first two arrays are the frequency points vector and the
        estimated PSD. The last returned array differs depending on whether
        the jackknife was used. It is either

        * The jackknife estimated variance of the log-psd, OR
        * The degrees of freedom in a chi2 model of how the estimated
          PSD is distributed about the true log-PSD (this is either
          2*floor(2*NW), or calculated from adaptive weights)

    Notes
    -----

    The bandwidth of the windowing function will determine the number
    tapers to use. This parameters represents trade-off between frequency
    resolution (lower main lobe BW for the taper) and variance reduction
    (higher BW and number of averaged estimates). Typically, the number of
    tapers is calculated as 2x the bandwidth-to-fundamental-frequency
    ratio, as these eigenfunctions have the best energy concentration.

    """
    # have last axis be time series for now
    N = s.shape[-1]
    M = int(np.product(s.shape[:-1]))

    if BW is not None:
        # BW wins in a contest (since it was the original implementation)
        norm_BW = np.round(BW * N / fs)
        NW = norm_BW / 2.0
    elif NW is None:
        # default NW
        NW = 4
    # (else BW is None and NW is not None) ... all set
    Kmax = int(2 * NW)

    # if the time series is a complex vector, a one sided PSD is invalid:
    if (sides == "default" and np.iscomplexobj(s)) or sides == "twosided":
        sides = "twosided"
    elif sides in ("default", "onesided"):
        sides = "onesided"

    # Find the direct spectral estimators S_k(f) for k tapered signals..
    # don't normalize the periodograms by 1/N as normal.. since the taper
    # windows are orthonormal, they effectively scale the signal by 1/N
    spectra, eigvals = tapered_spectra(s, (NW, Kmax), NFFT=NFFT, low_bias=low_bias)
    NFFT = spectra.shape[-1]
    K = len(eigvals)
    # collapse spectra's shape back down to 3 dimensions
    spectra.shape = (M, K, NFFT)

    last_freq = NFFT // 2 + 1 if sides == "onesided" else NFFT

    # degrees of freedom at each timeseries, at each freq
    nu = np.empty((M, last_freq))
    if adaptive:
        weights = np.empty((M, K, last_freq))
        for i in range(M):
            weights[i], nu[i] = adaptive_weights(spectra[i], eigvals, sides=sides)
    else:
        # let the weights simply be the square-root of the eigenvalues.
        # repeat these values across all n_chan channels of data
        weights = np.tile(np.sqrt(eigvals), M).reshape(M, K, 1)
        nu.fill(2 * K)

    if jackknife:
        jk_var = np.empty_like(nu)
        for i in range(M):
            jk_var[i] = jackknifed_sdf_variance(
                spectra[i], eigvals, sides=sides, adaptive=adaptive
            )

    # Compute the unbiased spectral estimator for S(f) as the sum of
    # the S_k(f) weighted by the function w_k(f)**2, all divided by the
    # sum of the w_k(f)**2 over k

    # 1st, roll the tapers axis forward
    spectra = np.rollaxis(spectra, 1, start=0)
    weights = np.rollaxis(weights, 1, start=0)
    sdf_est = mtm_cross_spectrum(spectra, spectra, weights, sides=sides)
    sdf_est /= fs

    if sides == "onesided":
        freqs = np.linspace(0, fs / 2, NFFT // 2 + 1)
    else:
        freqs = np.linspace(0, fs, NFFT, endpoint=False)

    out_shape = s.shape[:-1] + (len(freqs),)
    sdf_est.shape = out_shape
    if jackknife:
        jk_var.shape = out_shape
        return freqs, sdf_est, jk_var
    else:
        nu.shape = out_shape
        return freqs, sdf_est, nu
