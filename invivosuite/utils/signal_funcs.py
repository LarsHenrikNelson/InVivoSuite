from typing import Literal

import KDEpy
import numpy as np
from numba import njit, prange
from scipy import interpolate, optimize, signal

from ..spectral import fft

__all__ = [
    "aligned",
    "bin_data_sorted",
    "bin_data_unsorted",
    "convolve_loop",
    "convolve",
    "corr_acq",
    "cross_corr",
    "envelopes_idx",
    "fit_sine",
    "kde",
    "mad",
    "sinefunc",
    "where_count",
    "whitening_matrix",
    "xconv_fft",
    "xcorr_fft",
    "xcorr_lag",
]


@njit()
def sliding_count(array, window, closed: Literal["left", "right"] = "left"):
    output = np.zeros(array.size, dtype=int)
    if closed == "left":
        for i in range(window):
            output[i] = array[:i].sum()
        for i in range(window, array.size):
            output[i] = array[i : i + window].sum()
    else:
        for i in range(array.size - window):
            output[i] = array[i : i + window].sum()
        for i in range(array.size - window, array.size):
            output[i] = array[i : array.size].sum()
    return output


def mad(array):
    mad = np.median(np.abs(np.median(array) - array))
    return mad / 0.6744897501960817


def whitening_matrix(acquisitions: np.ndarray) -> np.ndarray:
    """ZCA whitening matrix. Assumes data is (nchans, nsamples).
    Can be used for local whitening by passing a subset of
    channels.

    Args:
        acquisitions (np.ndarray): _description_

    Returns:
        np.ndarray: _description_
    """
    nt = acquisitions.shape[0]
    cc = acquisitions @ acquisitions.T
    cc /= nt
    u, s, _ = np.linalg.svd(cc, hermitian=True)
    W = np.dot(u, np.dot(np.diag(1.0 / np.sqrt(s + 1e-8)), u.T))
    return W


def whitening_matrix_local(data: np.ndarray, W: np.ndarray, channel, neighbors):
    # Not used anymore
    nchans = data.shape[0]
    nt = data.shape[1]
    start = max(0, channel - neighbors)
    end = min(channel + neighbors + 1, nchans)
    inds = np.arange(start, end)
    cc = data.T @ data
    cc /= nt
    u, s, _ = np.linalg.svd(cc, hermitian=True)
    W_local = np.dot(u, np.dot(np.diag(1.0 / np.sqrt(s + 1e-8)), u.T))
    ilocal = min(channel, neighbors)
    W[inds, channel] = W_local[inds - inds.min(), ilocal]


@njit(parallel=True)
def corr_acq(acqs):
    size = acqs.shape[0]
    corr_data = np.zeros((size, size))
    for i in prange(size):
        for j in range(i, size):
            corr_data[i, j] = np.corrcoef(acqs[i], acqs[j])[0, 1]
            corr_data[j, i] = corr_data[i, j]
    return corr_data


@njit(cache=True)
def where_count(item, array):
    count = 0
    for i in array:
        if i == item:
            count += 1
    return count


def xconv_fft(array_1: np.ndarray, array_2: np.ndarray, circular=False):
    shape = array_1.size + array_2.size - 1
    fshape = fft.next_fast_len(shape)

    sp1 = fft.r2c_rfft(array_1, fshape)
    sp2 = fft.r2c_rfft(array_2, fshape)

    if not circular:
        ret = fft.ifft(sp1 * sp2, norm="n")
    else:
        ret = fft.ifft(sp1 * np.conjugate(sp2), norm="n")

    return ret[:shape]


def xcorr_fft(array_1: np.ndarray, array_2: np.ndarray, circular=False):
    array_2 = np.ascontiguousarray(array_2[::-1])

    shape = array_1.size + array_2.size - 1

    sp1 = fft.r2c_rfft(array_1, nfft=-1)
    sp2 = fft.r2c_rfft(array_2, nfft=-1)

    if not circular:
        ret = fft.irfft(sp1 * sp2)
    else:
        ret = fft.irfft(sp1 * np.conjugate(sp2))

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
        output = convolve_loop(array_1, array_2)
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


def sinefunc(t, A, w, p, c):
    return A * np.sin(w * t + p) + c


def fit_sine(x, y):
    popt, _ = optimize.curve_fit(sinefunc, x, y)
    A, w, p, c = popt
    output = sinefunc(x, A, w, p, c)
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
def bin_data_unsorted(data: np.ndarray, bins: np.ndarray, func: callable):
    binned_data = np.zeros(bins.size - 1)
    for i in range(bins.size - 1):
        indexes = np.where((data >= bins[i]) & (data < bins[i + 1]))
        binned_data = func(data[indexes])
    return binned_data


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


def aligned(a, alignment=64):
    if (a.ctypes.data % alignment) == 0:
        return a
    assert alignment % a.itemsize == 0
    extra = alignment // a.itemsize
    buf = np.empty(a.size + extra, dtype=a.dtype)
    ofs = (-buf.ctypes.data % alignment) // a.itemsize
    aa = buf[ofs : ofs + a.size].reshape(a.shape)
    np.copyto(aa, a)
    assert aa.ctypes.data % alignment == 0
    return aa


def center_spikes(phy_model, acq_manager, probe="acc", ref_type="cmr", size=45):
    for chan, clust_ids in phy_model.channel_clusters:
        for clust in clust_ids:
            dat = np.where(phy_model.spike_clusters == clust)[0]
            spike_times = phy_model.spike_times[dat].flatten()
            if chan != 64:
                spk_acq = acq_manager.acq(
                    acq_num=chan,
                    acq_type="spike",
                    ref=True,
                    ref_type=ref_type,
                    ref_probe=probe,
                    map_channel=True,
                    probe=probe,
                )
                centered_spks = _center_spikes(
                    spike_times, spk_acq, probe=probe, ref_type=ref_type, size=size
                )
                phy_model.spike_times[dat] = centered_spks


def _center_spikes(spike_times, spk_acq, size=45):
    mod_spk_times = np.zeros(spike_times.size, np.int64)
    for i in range(spike_times.size):
        start = int(spike_times[i] - size)
        end = int(spike_times[i] + size + 1)
        if start < 0:
            start = 0
        if spk_acq.size < end:
            end = spk_acq.size
        b = signal.argrelmin(spk_acq[start:end], order=15)[0]
        if b.size > 1:
            temp = np.argmin(np.abs(b - 45))
            b = b[temp] - size
        else:
            b = b[0] - size
        if b > np.abs(3):
            mod_spk_times[i] = b + spike_times[i]
        else:
            mod_spk_times[i] = spike_times[i]
    return mod_spk_times


def gauss_kernel(sigma):
    """
    Computes a 1-D Gaussian convolution kernel.
    """
    radius = 4 * sigma + 0.5
    sigma2 = sigma * sigma
    x = np.arange(-radius, radius + 1)
    phi_x = np.exp(-0.5 / sigma2 * x**2)
    phi_x = phi_x / phi_x.sum()
    return phi_x
