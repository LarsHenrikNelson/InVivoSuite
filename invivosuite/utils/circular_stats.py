import numpy as np
from numba import njit

"""
See: https://github.com/aarchiba/kuiper
See: https://docs.astropy.org/en/stable/index.html
"""


@njit(cache=True)
def periodic_mean_std(angles):
    sines = np.sin(angles)
    cosines = np.cos(angles)
    n_ = len(angles)
    mean = np.arctan2(np.mean(sines), np.mean(cosines))
    R = np.sqrt(np.sum(sines) ** 2 + np.sum(cosines) ** 2) / n_
    std = np.sqrt(-2 * np.log(R))

    return mean, std


@njit(cache=True)
def rayleightest(data):
    n = data.size
    S = np.sum(np.sin(data)) / n
    C = np.sum(np.cos(data)) / n
    Rbar = np.hypot(S, C)
    z = n * Rbar * Rbar

    # see [3] and [4] for the formulae below
    tmp = 1.0
    if n < 50:
        tmp = (
            1.0
            + (2.0 * z - z * z) / (4.0 * n)
            - (24.0 * z - 132.0 * z**2.0 + 76.0 * z**3.0 - 9.0 * z**4.0)
            / (288.0 * n * n)
        )

    p_value = np.exp(-z) * tmp
    return p_value


def h_fpp(H):
    # These values are obtained by fitting to simulations.
    a = 0.9999755
    b = 0.39802
    c = 1.210597
    d = 0.45901
    e = 0.0022900

    if H <= 23:
        return a * np.exp(-b * H)
    elif H < 50:
        return c * np.exp(-d * H + e * H**2)
    else:
        return 4e-8
        # This comes up too often to raise an exception
        raise ValueError(
            f"H={H}>50 not supported; false positive probability less than 4*10**(-8)"
        )


def h_test(events):
    """Apply the H test for uniformity on [0,1).

    The H test is an extension of the Z_m^2 or Rayleigh tests for
    uniformity on the circle. These tests estimate the Fourier coefficients
    of the distribution and compare them with the values predicted for
    a uniform distribution, but they require the user to specify the number
    of harmonics to use. The H test automatically selects the number of
    harmonics to use based on the data. The returned statistic, H, has mean
    and standard deviation approximately 2.51, but its significance should
    be evaluated with the routine h_fpp. This is done automatically in this
    routine.

    Arguments
    ---------

    events : array-like
        events should consist of an array of values to be interpreted as
        values modulo 1. These events will be tested for statistically
        significant deviations from uniformity.

    Returns
    -------

    H : float
        The raw score. Larger numbers indicate more non-uniformity.
    M : int
        The number of harmonics that give the most significant deviation
        from uniformity.
    fpp : float
        The probability of an H score this large arising from sampling a
        uniform distribution.

    Reference
    ---------

    de Jager, O. C., Swanepoel, J. W. H, and Raubenheimer, B. C., "A
    powerful test for weak periodic signals of unknown light curve shape
    in sparse data", Astron. Astrophys. 221, 180-190, 1989.
    """
    max_harmonic = 20
    ev = np.reshape(events, (-1,))
    cs = np.sum(
        np.exp(2.0j * np.pi * np.arange(1, max_harmonic + 1) * ev[:, None]), axis=0
    ) / len(ev)
    Zm2 = 2 * len(ev) * np.cumsum(np.abs(cs) ** 2)
    Hcand = Zm2 - 4 * np.arange(1, max_harmonic + 1) + 4
    M = np.argmax(Hcand) + 1
    H = Hcand[M - 1]
    fpp = h_fpp(H)
    return (H, M, fpp)
