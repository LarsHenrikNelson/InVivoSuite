import KDEpy
import numpy as np
from numba import njit, prange
from numpy.random import default_rng
from scipy import fft, interpolate, optimize


def xconv_fft(array_1: np.ndarray, array_2: np.ndarray, circular=False):
    shape = array_1.size + array_2.size - 1
    fshape = fft.next_fast_len(shape)

    sp1 = fft.rfft(array_1, fshape)
    sp2 = fft.rfft(array_2, fshape)

    if not circular:
        ret = fft.irfft(sp1 * sp2, fshape)
    else:
        ret = fft.irfft(sp1 * np.conjugate(sp2), fshape)

    return ret[:shape]


def xcorr_fft(array_1: np.ndarray, array_2: np.ndarray, circular=False):
    array_2 = np.ascontiguousarray(array_2[::-1])

    shape = array_1.size + array_2.size - 1
    fshape = fft.next_fast_len(shape)

    sp1 = fft.rfft(array_1, fshape)
    sp2 = fft.rfft(array_2, fshape)

    if not circular:
        ret = fft.irfft(sp1 * sp2, fshape)
    else:
        ret = fft.irfft(sp1 * np.conjugate(sp2), fshape)

    return ret[:shape]


def xcorr_lag(
    array_1: np.ndarray,
    array_2: np.ndarray,
    lag: int,
    norm: bool = True,
    mode: str = "fft",
    remove_mean=False,
    circular=False,
):
    if remove_mean:
        array_1 = array_1 - array_1.mean()
        array_2 = array_2 - array_2.mean()
    if norm:
        array_1 = array_1 / np.sqrt(array_1.dot(array_1))
        array_2 = array_2 / np.sqrt(array_2.dot(array_2))
    if mode == "fft":
        output = xcorr_fft(array_1, array_2, circular=circular)
    else:
        output = np.correlate(array_1, array_2, mode="full")
    lags = np.linspace(-lag, lag, num=lag * 2)
    mid = output.size // 2
    return lags, output[int(mid - lag) : int(mid + lag)]


@njit(cache=True, parallel=True)
def convolve_loop(array_1: np.ndarray, array_2: np.ndarray):
    # This a tiny bit faster than scipy version
    output = np.zeros(array_1.size + array_2.size - 1)
    for i in prange(array_2.size):
        for j in range(array_1.size):
            output[i + j] += array_1[j] * array_2[i]
    return output


def convolve(
    array_1: np.ndarray, array_2: np.ndarray, mode: str = "fft", circular: bool = False
):
    if mode == "fft":
        output = xconv_fft(array_1, array_2, circular=circular)
    else:
        output = convolve(array_1, array_2)
    return output


@njit(parallel=True)
def cross_corr(acq1: np.ndarray, acq2: np.ndarray, cutoff: int):
    output = np.zeros(cutoff * 2)
    for i in prange(cutoff):
        output[cutoff - i] = np.corrcoef(acq2[i:], acq1[: acq1.size - i])[0, 1]
    for i in prange(cutoff):
        output[i + cutoff] = np.corrcoef(acq1[i:], acq2[: acq2.size - i])[0, 1]
    return output


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


def sinfunc(t, A, w, p, c):
    return A * np.sin(w * t + p) + c


def fit_sine(x, y):
    popt, _ = optimize.curve_fit(sinfunc, x, y)
    A, w, p, c = popt
    output = sinfunc(x, A, w, p, c)
    return output, popt


@njit()
def bin_data_sorted(data: np.ndarray, bins: np.ndarray):
    """Similar to Numpy's digitize or search_all function
    but reduces some of the steps and is more straight forward.
    The function assumes that the bins and data are sorted in
    the same manner (i.e. ascending or descending). Useful for
    computing bin sizes for histograms or spike-phase plots.
    The size of the binned data will 1 shorter than the number
    of bins. Uses bins[i] <= value < bins[i+1].

    Args:
        data (np.ndarray): Numpy array of sorted values
        bins (np.nadarray): Numpy array of sorted values

    Returns:
        _type_: _description_
    """
    binned_data = np.zeros(bins.size - 1)
    index = 0
    for i in data:
        if i >= bins[index] and i < bins[int(index + 1)]:
            binned_data[index] += 1
        else:
            if index < binned_data.size:
                binned_data[index] += 1
                index += 1
            else:
                binned_data[binned_data.size - 1] += 1
    return binned_data


@njit()
def binned_data_unsorted(data: np.ndarray, bins: np.ndarray, func: callable):
    binned_data = np.zeros(bins.size - 1)
    for i in range(bins.size - 1):
        indexes = np.where((data >= bins[i]) & (data < bins[i + 1]))
        binned_data = func(data[indexes])
    return binned_data


@njit(cache=True)
def ppc_numba(spike_phases):
    outer_sums = np.zeros(spike_phases.size - 1)
    array1 = np.zeros(2)
    array2 = np.zeros(2)
    for index1 in range(0, spike_phases.size - 1):
        temp_sum = np.zeros(spike_phases.size - index1 + 1)
        array1[0] = np.cos(spike_phases[index1])
        array1[1] = np.sin(spike_phases[index1])
        for index2 in range(index1 + 1, spike_phases.size):
            array2[0] = np.cos(spike_phases[index2])
            array2[1] = np.sin(spike_phases[index2])
            dp = np.dot(array1, array2)
            temp_sum[index2 - index1] = dp
        outer_sums[index1] = temp_sum.sum()
    dp_sum = np.sum(outer_sums)
    ppc_output = dp_sum / int(len(spike_phases) * (len(spike_phases) - 1) / 2)
    return ppc_output


def ppc_sampled(spike_phases, size, iterations, seed=42):
    """This

    Args:
        spike_phases (_type_): _description_
        size (_type_): _description_
        iterations (_type_): _description_
        seed (int, optional): _description_. Defaults to 42.

    Returns:
        _type_: _description_
    """
    rng = default_rng(seed)
    output_array = np.zeros(iterations)
    for i in range(iterations):
        spk_sampled = np.ascontiguousarray(
            rng.choice(spike_phases, size=size, replace=False)
        )
        output_array[i] = ppc_numba(spk_sampled)
    return output_array.mean()


def kde(
    array: np.ndarray,
    kernel: str = "biweight",
    bw_method: str = "ISJ",
    tol: float = 0.001,
):
    if array.ndim > 1:
        array = array.flatten()
    power2 = int(np.ceil(np.log2(array.size)))

    # Calculating x comes from Seaborn KDE
    bw = np.cov(array)
    min_array = array.min() - bw * tol
    max_array = array.max() + bw * tol
    x = np.linspace(min_array, max_array, num=2**power2)
    y = KDEpy.FFTKDE(kernel=kernel, bw=bw_method).fit(array, weights=None).evaluate(x)
    return x, y