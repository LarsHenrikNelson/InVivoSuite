from typing import Literal, Union

import numpy as np
from numba import njit, prange
from numpy.polynomial import Polynomial
from scipy import fft, interpolate, signal

from ...spectral import multitaper
from ..signal_functions import kde, short_time_energy

Windows = Literal[
    "hamming",
    "hann",
    "blackmanharris",
    "boxcar",
    "triangle",
    "flattop",
    "parzen",
    "bohman",
    "nuttall",
    "barthann",
    "cosine",
    "exponential",
    "tukey",
    "taylor",
    "lanczos",
    "bartlett",
]


def derivative_baseline(
    array: np.ndarray,
    window: str,
    wlen: float,
    fs: Union[float, int],
    threshold: float = 3.0,
):
    temp = np.gradient(array)
    wlen = int(wlen * fs)
    window = window
    win_array = signal.get_window(window, wlen)
    win_array /= win_array.sum()
    se_array = signal.convolve(temp, win_array)
    se_array = se_array[wlen // 2 : temp.size + wlen // 2]
    baseline = np.cumsum(se_array)
    baseline += np.std(baseline) * threshold
    return baseline


def kde_baseline(
    ste: np.ndarray,
    tol: float = 0.001,
    method: Literal["spline", "fixed", "polynomial"] = "spline",
    deg: int = 90,
    threshold: float = 3.0,
):
    baseline_x = np.arange(0, ste.size, 1)
    x, y = kde(ste, kernel="biweight", bw_method="ISJ", tol=tol)

    # Log tranform distribution since it is all positive values.
    # Find the mean and std.
    mean_x = np.sum(np.log10(x) * y / np.sum(y))
    stand_dev = np.sqrt(np.sum(((np.log10(x) - mean_x) ** 2) * (y / np.sum(y))))

    # Find the max accepted baseline value by back-transforming
    ste_max = 10 ** (mean_x + 3 * stand_dev)
    indices = np.where(ste < ste_max)[0]
    temp = ste[indices]
    if method == "fixed":
        max_index = np.where(x > 10 ** (mean_x))[0][0]
        baseline = np.full(ste.size, x[max_index])
    elif method == "spline":
        knots = np.linspace(0, temp.size, num=deg + 2, dtype=int)[1:-2]
        spl = interpolate.LSQUnivariateSpline(indices, temp, t=indices[knots])
        baseline = spl(baseline_x)
    elif method == "polynomial":
        poly = Polynomial.fit(
            indices,
            temp,
            deg=deg,
            rcond=None,
            full=False,
            w=None,
        )
        baseline = poly(baseline_x)
    else:
        raise ValueError("method must be: fixed, spline, polynomial,")
    baseline += np.std(baseline) * threshold
    return baseline


def clean_bursts(
    bursts_ind: np.ndarray, acq: np.ndarray, minimum_peaks: int = 5, fs: float = 1000.0
):
    cb = np.zeros(bursts_ind.shape)
    index = 0
    min_lag = fs * 0.025
    for i in bursts_ind:
        if (i[1] - i[1] < 20 * fs) or (i[1] - i[0] > 0.2 * fs):
            burst = acq[i[0] : i[1]]
            peaks, _ = signal.find_peaks(
                np.abs(burst),
                distance=min_lag,
                prominence=np.sqrt(np.mean(burst**2)),
            )
            if len(peaks) >= minimum_peaks:
                cb[index] = i
                index += 1
    cb = cb[:index, :]
    return np.asarray(cb, dtype=np.int64)


def _find_bursts(
    ste,
    baseline,
    min_len=0.2,
    max_len=20,
    min_burst_int=0.2,
    pre=3.0,
    post=3.0,
    order=0.1,
    fs=1000,
):
    max_len = max_len * fs
    min_len = min_len * fs
    order = int(order * fs)
    min_burst_int = int(min_burst_int * fs)

    p = np.where(ste > baseline)[0]
    diffs = np.diff(p)
    spl_in = np.where(diffs > min_burst_int)[0] + 1
    bursts_temp = np.split(p, spl_in)
    bursts = []
    for index, i in enumerate(bursts_temp):
        start = i[0]
        end = i[-1] + 1
        end_temp = signal.argrelmin(ste[end : end + int(post * fs)], order=order)[0]
        if end_temp.size == 0:
            end_temp = end
        else:
            end_temp = end_temp[0]
        end = end_temp + end
        if end > ste.size:
            end = ste.size
        start_temp = signal.argrelmin(ste[start - int(pre * fs) : start], order=order)[
            0
        ]
        if start_temp.size == 0:
            start_temp = start
        else:
            start_temp = start_temp[-1]
        start = start_temp + start - int(pre * fs)
        if start < 0:
            start = 0
        blen = end - start
        if blen <= (max_len) and blen >= (min_len):
            bursts.append((int(start), int(end)))
    bursts = np.array(bursts)
    new_bursts = [bursts[0]]
    index = 1
    for i in bursts[1:, :]:
        if new_bursts[index - 1][0] >= i[0]:
            new_bursts[index - 1][1] = i[1]
        elif new_bursts[index - 1][1] >= i[0]:
            new_bursts[index - 1][1] = i[1]
        else:
            new_bursts.append(i)
            index += 1
    bursts = np.array(new_bursts)
    return bursts


def find_bursts(
    acq,
    window: Windows = "hamming",
    min_len: float = 0.2,
    max_len: float = 20.0,
    min_burst_int: float = 0.2,
    minimum_peaks: int = 5,
    wlen: float = 0.2,
    threshold: float = 10.0,
    fs=1000.0,
    pre: float = 3.0,
    post: float = 3.0,
    order: float = 0.1,
    method: Literal["spline", "fixed", "polynomial", "derivative"] = "spline",
    tol: float = 0.001,
    deg: int = 90,
):
    ste = short_time_energy(acq, window=window, wlen=wlen, fs=fs)
    if method == "derivative":
        baseline = derivative_baseline(
            ste, window=window, wlen=tol, fs=fs, threshold=threshold
        )
    else:
        # Multiplying baseline by threshold value accentuates
        # the curves of the baseline. Adding a threshold value
        # Just moves the baseline up.
        baseline = kde_baseline(
            ste, tol=tol, method=method, deg=deg, threshold=threshold
        )
    bursts = _find_bursts(
        ste=ste,
        baseline=baseline,
        min_len=min_len,
        max_len=max_len,
        min_burst_int=min_burst_int,
        pre=pre,
        post=post,
        order=order,
        fs=fs,
    )
    bursts = clean_bursts(bursts, acq, minimum_peaks=minimum_peaks, fs=fs)
    return bursts


@njit()
def burst_baseline_periods(bursts_ind, size):
    if int(bursts_ind[-1, 1]) == size and int(bursts_ind[0, 0]) == 0:
        burst_baseline = np.zeros((bursts_ind.shape[0] - 1, 2), dtype=np.int64)
        burst_baseline[:, 0] = bursts_ind[:-1, 1]
        burst_baseline[:, 1] = bursts_ind[1:, 0]
    elif int(bursts_ind[-1, 1]) != size and int(bursts_ind[0, 0]) != 0:
        burst_baseline = np.zeros((bursts_ind.shape[0] + 1, 2), dtype=np.int64)
        burst_baseline[1:, 0] = bursts_ind[:, 1]
        burst_baseline[:-1, 1] = bursts_ind[:, 0]
        burst_baseline[-1, 1] = size
    elif int(bursts_ind[-1, 1]) == size and int(bursts_ind[0, 0]) != 0:
        burst_baseline = np.zeros((bursts_ind.shape[0], 2), dtype=np.int64)
        burst_baseline[1:, 0] = bursts_ind[:-1, 1]
        burst_baseline[:, 1] = bursts_ind[:, 0]
    else:
        burst_baseline = np.zeros((bursts_ind.shape[0], 2), dtype=np.int64)
        burst_baseline[:, 0] = bursts_ind[:, 1]
        burst_baseline[:-1, 1] = bursts_ind[1:, 0]
        burst_baseline[-1, 1] = size
    return burst_baseline


@njit(parallel=True, cache=True)
def burst_flatness(
    bursts: np.ndarray, acq: np.ndarray, nperseg: int = 200, noverlap: int = 0
) -> np.ndarray:
    flatness = np.zeros(bursts.shape[0])
    for i in prange(bursts.shape[0]):
        nperseg = 200
        noverlap = 0
        burst_size = bursts[i, 1] - bursts[i, 0]
        step = nperseg - noverlap
        num_segs, mod = np.divmod(burst_size, step)
        if mod != 0:
            out_size = num_segs + 1
        else:
            out_size = num_segs
        rms = np.zeros(out_size)
        for j in range(out_size - 1):
            start = (j * step) + bursts[i, 0]
            rms[j] = np.sqrt(np.mean(acq[start : start + nperseg] ** 2))
        rms[-1] = np.sqrt(np.mean(acq[bursts[i, 1] - mod : bursts[i, 1]] ** 2))
        flatness[i] = np.divide(rms.min(), rms.max())
    return flatness


def burst_iei(bursts, acq, fs):
    iei = np.zeros(bursts.shape[0])
    for i in range(bursts.shape[0]):
        burst = acq[bursts[i, 0] : bursts[i, 1]]
        peaks, _ = signal.find_peaks(
            burst * -1, prominence=0.5 * np.sqrt(np.mean(np.square(burst)))
        )
        iei[i] = np.mean(np.diff(peaks) / fs)
    return iei


def burst_power_bands(burst_pxxs, freqs, freq_dict):
    power_dict = {}
    for key, value in freq_dict.items():
        f_ind = np.where((freqs >= value[0]) & (freqs < value[1]))[0]
        power_dict[key] = burst_pxxs[:, f_ind].mean(axis=1)
    return power_dict


@njit(cache=True)
def bursts_rms(bursts, acq):
    rms = np.zeros(bursts.shape[0])
    for i in range(bursts.shape[0]):
        rms[i] = np.sqrt(np.mean(acq[bursts[i, 0] : bursts[i, 1]] ** 2))
    return rms


@njit()
def get_max_from_burst(array):
    maximums = [np.max(array[i]) for i in array]
    return maximums


def burst_stats(
    acq: np.ndarray,
    bursts: np.ndarray,
    baseline_seg: np.ndarray,
    band_dict: dict,
    fs: float,
):
    output_dict = {}
    output_dict["len"] = (bursts[:, 1] - bursts[:, 0]) / fs
    output_dict["iei"] = np.diff(bursts[:, 0]) / fs
    freqs, bursts_pxx = seg_pxx(acq, bursts, fs)
    _, baseline_pxx = seg_pxx(acq, baseline_seg, fs)
    p_dict = burst_power_bands(
        bursts_pxx,
        freqs,
        band_dict,
    )
    p_dict = {f"burst_{key}": value for key, value in p_dict.items()}
    output_dict.update(p_dict)
    b_dict = burst_power_bands(
        baseline_pxx,
        freqs,
        band_dict,
    )
    b_dict = {f"baseline_{key}": value for key, value in b_dict.items()}
    output_dict.update(b_dict)
    output_dict["intra_iei"] = burst_iei(bursts, acq, fs)
    output_dict["flatness"] = burst_flatness(bursts, acq)
    output_dict["rms"] = bursts_rms(bursts, acq)
    return output_dict


def seg_pxx(
    acq: np.ndarray,
    segments: np.ndarray,
    fs: float = 1000.0,
    NW: Union[float, None] = 2.5,
    BW: Union[float, None] = None,
    adaptive=False,
    jackknife=False,
    low_bias=True,
    sides="default",
    NFFT=None,
):
    diff = np.max(segments[:, 1] - segments[:, 0])
    if NFFT is None:
        size = (segments[:, 1] - segments[:, 0]).max()
        NFFT = fft.next_fast_len(size)
    if NFFT < diff:
        raise ValueError("NFFT must be longer than the longest burst.")
    burst_pxx = np.zeros((segments.shape[0], NFFT // 2 + 1))
    for index, i in enumerate(segments):
        if (acq[i[0] : i[1]]).size > 200:
            freq, pxx, _ = multitaper(
                acq[i[0] : i[1]],
                fs=fs,
                NW=NW,
                BW=BW,
                adaptive=adaptive,
                jackknife=jackknife,
                low_bias=low_bias,
                sides=sides,
                NFFT=NFFT,
            )
            burst_pxx[index] = pxx
    return freq, burst_pxx


def get_burst_pxx(
    bursts,
    baselines,
    acq,
    fs: float = 1000.0,
    NW: Union[float, None] = 2.5,
    BW: Union[float, None] = None,
    adaptive=False,
    jackknife=True,
    low_bias=True,
    sides="default",
):
    size1 = (bursts[:, 1] - bursts[:, 0]).max()
    size2 = (baselines[:, 1] - baselines[:, 0]).max()
    NFFT = fft.next_fast_len(np.max((size1, size2)))
    f, bursts_pxx = seg_pxx(
        acq=acq,
        segments=bursts,
        fs=fs,
        NW=NW,
        BW=BW,
        adaptive=adaptive,
        jackknife=jackknife,
        low_bias=low_bias,
        sides=sides,
        NFFT=NFFT,
    )
    _, baseline_pxx = seg_pxx(
        acq=acq,
        segments=baselines,
        fs=fs,
        NW=NW,
        BW=BW,
        adaptive=adaptive,
        jackknife=jackknife,
        low_bias=low_bias,
        sides=sides,
        NFFT=NFFT,
    )
    return f, bursts_pxx, baseline_pxx
