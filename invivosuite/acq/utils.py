from scipy import fft, interpolate
import numpy as np
from numba import njit, prange


def xcorr_fft(array_1: np.ndarray, array_2: np.ndarray):
    array_2 = np.ascontiguousarray(array_2[::-1])

    shape = array_1.size + array_2.size - 1
    fshape = fft.next_fast_len(shape)

    sp1 = fft.rfft(array_1, fshape)
    sp2 = fft.rfft(array_2, fshape)

    ret = fft.irfft(sp1 * sp2, fshape)

    return ret[:shape]


def xcorr_lag(
    array_1: np.ndarray,
    array_2: np.ndarray,
    lag: int,
    norm: bool = True,
    mode: str = "fft",
):
    if norm:
        array_1 = array_1 / np.sqrt(array_1.dot(array_1))
        array_2 = array_2 / np.sqrt(array_2.dot(array_2))
    if mode == "fft":
        output = xcorr_fft(array_1, array_2)
    else:
        output = np.correlate(array_1, array_2, mode="full")
    lags = np.linspace(-lag, lag, num=lag * 2)
    mid = output.size // 2
    return lags, output[int(mid - lag) : int(mid + lag)]


@njit(cache=True)
def convolve(array: np.ndarray, window: np.ndarray):
    # This a tiny bit faster than scipy version
    output = np.zeros(array.size + window.size - 1)
    for i in range(array.size):
        for j in range(window.size):
            output[i + j] += array[i] * window[j]
    return output


@njit(parallel=True)
def cross_corr(acq1: np.ndarray, acq2: np.ndarray, cutoff: int):
    output = np.zeros(cutoff * 2)
    for i in prange(cutoff):
        output[cutoff - i] = np.corrcoef(acq2[i:], acq1[: acq1.size - i])[0, 1]
    for i in prange(cutoff):
        output[i + cutoff] = np.corrcoef(acq1[i:], acq2[: acq2.size - i])[0, 1]
    return output


@njit()
def envelopes_idx(
    s: np.ndarray, dmin: int = 1, dmax: int = 1, split: bool = False, interp=True
):
    """
    Input :
    s: 1d-array, data signal from which to extract high and low envelopes
    dmin, dmax: int, optional, size of chunks, use this if the size of the input signal is too big
    split: bool, optional, if True, split the signal in half along its mean, might help to generate the envelope in some cases
    Output :
    lmin,lmax : high/low envelope idx of input signal s
    """

    # locals min
    lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1
    # locals max
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1

    if split:
        # s_mid is zero if s centered around x-axis or more generally mean of signal
        s_mid = np.mean(s)
        # pre-sorting of locals min based on relative position with respect to s_mid
        lmin = lmin[s[lmin] < s_mid]
        # pre-sorting of local max based on relative position with respect to s_mid
        lmax = lmax[s[lmax] > s_mid]

    # global min of dmin-chunks of locals min
    lmin = lmin[
        [i + np.argmin(s[lmin[i : i + dmin]]) for i in range(0, len(lmin), dmin)]
    ]
    # global max of dmax-chunks of locals max
    lmax = lmax[
        [i + np.argmax(s[lmax[i : i + dmax]]) for i in range(0, len(lmax), dmax)]
    ]
    if interp:
        max_size = lmax.size
        max_start = 0
        max_end = -1
        if lmax[-1] + 1 != s.size:
            max_size += 1
            max_end = s.size - 1
        if lmax[0] != 0:
            max_start += 1
            max_size += 1
        lmax = np.zeros(max_size)
        lmax[max_start:] = lmax
        if max_end != -1:
            lmax[-1] = max_end

        min_size = lmax.size
        min_start = 0
        min_end = -1
        if lmax[-1] + 1 != s.size:
            min_size += 1
            max_end = s.size - 1
        if lmax[0] != 0:
            min_size += 1
            min_start += 1
        lmin = np.zeros(max_size)
        lmin[max_start:] = lmin
        if min_end != -1:
            lmin[-1] = min_end

        cs_max = interpolate.CubicSpline(lmax, s[lmax])
        cs_min = interpolate.CubicSpline(lmin, s[lmin])

        x = np.arange(s.size)
        lmax = cs_max(x)
        lmin = cs_min(x)

    return lmin, lmax
