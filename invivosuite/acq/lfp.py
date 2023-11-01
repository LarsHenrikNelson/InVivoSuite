from itertools import combinations
from typing import Literal, Union, TypeAlias

import fcwt
import KDEpy
import numpy as np
import statsmodels.api as sm
from numba import njit, prange
from numpy.polynomial import Polynomial
from scipy import fft, interpolate, signal

from .tapered_spectra import multitaper


__all__ = [
    "get_ave_band_power",
    "get_max_from_burst",
    "split_at_zeros",
    "calc_all_freq_corrs",
    "corr_freqs",
    "create_all_freq_windows",
    "get_freq_window",
    "synchrony_cwt",
    "phase_synchrony",
    "phase_discontinuity_index",
    "stepped_cwt_cohy",
    "find_bursts",
    "short_time_energy",
    "coherence",
    "get_pairwise_coh",
    "kde_baseline",
    "derivative_baseline",
    "burst_stats",
    "seg_pxx",
]

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


def find_logpx_baseline(
    acq,
    fs=1000,
    freqs=(0, 100),
    nperseg=10000,
    noverlap=5000,
    nfft=10000,
    method: Literal[
        "AndrewWave",
        "TrimmedMean",
        "RamsayE",
        "HuberT",
        "TukeyBiweight",
        "Hampel",
        "LeastSquares",
    ] = "AndrewWave",
    window: str = "hann",
    NW: float = 2.5,
    BW: Union[float, None] = None,
    adaptive=False,
    jackknife=True,
    low_bias=True,
    sides="default",
    NFFT=None,
):
    if window != "dpss":
        f, px = signal.welch(acq, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    elif window == "dpss":
        f, px, _ = multitaper(
            acq,
            fs=fs,
            NW=NW,
            BW=BW,
            adaptive=adaptive,
            jackknife=jackknife,
            low_bias=low_bias,
            sides=sides,
            NFFT=NFFT,
        )
    else:
        raise ValueError("Window must be specified.")
    ind = np.where((f <= freqs[1]) & (f >= freqs[0]))[0]
    x = np.ones((ind.size, 2))
    x[:, 1] = f[ind]
    y = np.log10(px[ind]).reshape((ind.size, 1))
    match method:
        case "AndrewWave":
            M = sm.robust.norms.AndrewWave()
        case "TrimmedMean":
            M = sm.robust.norms.TrimmedMean()
        case "HuberT":
            M = sm.robust.norms.HuberT()
        case "RamsayE":
            M = sm.robust.norms.RamsayE()
        case "TukeyBiweight":
            M = sm.robust.norms.TukeyBiweight()
        case "Hampel":
            M = sm.robust.norms.Hampel()
        case "LeastSquares":
            M = sm.robust.norms.LeastSquares()
    rlm_m = sm.RLM(y, x, M=M)
    rlm_r = rlm_m.fit(cov="H2")
    rline = rlm_r.params[1] * f[ind] + rlm_r.params[0]
    return f[ind], px[ind], rline


def get_freq_window(
    pxx: np.ndarray,
    freqs: np.ndarray,
    lower_limit: float,
    upper_limit: float,
    log_transform: bool = True,
    window_type: Literal["sum", "mean"] = "sum",
):
    f_lim = np.where((freqs <= upper_limit) & (freqs >= lower_limit))
    pxx_freq = pxx[f_lim]
    if log_transform:
        pxx_freq = np.log10(pxx_freq)
    if window_type == "sum":
        ave = np.sum(pxx_freq, axis=0)
    elif window_type == "mean":
        ave = np.mean(pxx_freq, axis=0)
    return ave


def create_all_freq_windows(freq_dict, freqs, pxx):
    if np.iscomplexobj(pxx):
        pxx = np.abs(pxx)
    bands = {}
    for key, value in freq_dict.items():
        bands[key] = get_freq_window(pxx, freqs, value[0], value[1])
    return bands


@njit()
def cwt_cohy(cwt_1, cwt_2):
    sxy = cwt_1 * np.conjugate(cwt_2)
    sxx = cwt_1 * np.conjugate(cwt_1)
    syy = cwt_2 * np.conjugate(cwt_2)
    coh = (sxy**2) / (sxx * syy + 1e-8)
    return coh


@njit()
def corr_freqs(freq_band_1, freq_band_2, window):
    freq_band_1_window = np.lib.stride_tricks.sliding_window_view(freq_band_1, window)
    freq_band_2_window = np.lib.stride_tricks.sliding_window_view(freq_band_2, window)
    corrs = [0 for i in range(int((window - 1) / 2))]
    corrs.extend(
        [
            np.corrcoef(i, j)[1][0]
            for i, j in zip(freq_band_1_window, freq_band_2_window)
        ]
    )
    corrs.extend([0 for i in range((int((window - 1) / 2)))])
    return corrs


@njit()
def calc_all_freq_corrs(window, freq_bands):
    freqs = list(freq_bands.keys())
    combos = combinations(freqs, 2)
    for i in combos:
        corr_freqs(i[0], i[1], window)


@njit()
def split_at_zeros(array):
    array_list = []
    temp_list = []
    for index, i in enumerate(array):
        if i > 0:
            temp_list.append(index)
        else:
            if temp_list:
                array_list.append(temp_list)
            temp_list = []
    return array_list


@njit()
def get_max_from_burst(array):
    maximums = [np.max(array[i]) for i in array]
    return maximums


def get_ave_band_power(self, lower_limit, upper_limit, window="hann", type="welch"):
    if type != "welch":
        psd = signal.periodogram(self.array, window=window, fs=self.sample_rate)
    else:
        f, psd = signal.welch(self.array, window=window)
    f_lim = np.where((f <= upper_limit) & (f >= lower_limit))
    ave = np.mean(psd[f_lim])
    return ave


def synchrony_cwt_band(cwt_1, cwt_2, freqs, f0, f1):
    band = np.where((freqs <= f1) & (freqs >= f0))
    band_1 = np.angle(np.mean(cwt_1[band], axis=0))
    band_2 = np.angle(np.mean(cwt_2[band], axis=0))
    band_synchrony = 1 - np.sin(np.abs(band_2 - band_1) / 2)
    return band_synchrony


Bands: TypeAlias = dict[
    str, Union[list[float, float], tuple[float, float], np.ndarray[float, float]]
]


def get_cwt_bands(
    cwt: np.ndarray,
    freqs: np.ndarray,
    bands: Bands,
    ret_type=Literal["angle", "power", "raw"],
):
    band_arrays = {}
    for key, value in bands.items():
        band = np.where((freqs >= value[0]) & (freqs <= value[1]))
        band_arrays[key] = np.mean(cwt[band], axis=0)
    if ret_type == "raw":
        return band_arrays
    elif ret_type == "angle":
        return {
            key: np.angle(value, degree=False) for key, value in band_arrays.items()
        }
    elif ret_type == "power":
        return {key: np.abs(value) for key, value in band_arrays.items()}
    else:
        raise AttributeError(f"{ret_type} not recognized. Use angle, power or raw")


def synchrony_cwt(
    arrays: np.ndarray,
    freq_bands: Bands,
    fs: float = 1000.0,
    f0: float = 1.0,
    f1: float = 110,
    fn: int = 400,
    scaling: Literal["log", "lin"] = "log",
    norm: bool = True,
    nthreads: int = -1,
):
    matrices = []
    for i in range(len(freq_bands)):
        syn_matrix = np.zeros((arrays.shape[0], arrays.shape[0]))
        matrices.append(syn_matrix)
    morl = fcwt.Morlet(2.0)
    if scaling == "log":
        s = fcwt.FCWT_LOGFREQS
    elif scaling == "lin":
        s = fcwt.FCWT_LINFREQS
    else:
        raise AttributeError(f"{scaling} not a valid setting. Use log or lin.")
    scales = fcwt.Scales(morl, s, fs, f0, f1, fn)
    freqs = np.zeros((fn), dtype="single")
    scales.getFrequencies(freqs)
    fcwt_obj1 = fcwt.FCWT(
        morl, nthreads, use_optimization_plan=False, use_normalization=norm
    )
    fcwt_obj2 = fcwt.FCWT(
        morl, nthreads, use_optimization_plan=False, use_normalization=norm
    )
    cwt1 = np.zeros((fn, arrays.shape[1]), dtype=np.complex64)
    cwt2 = np.zeros((fn, arrays.shape[1]), dtype=np.complex64)
    for i in range(arrays.shape[0]):
        fcwt_obj1.cwt(arrays[i], scales, cwt1)
        bands1 = get_cwt_bands(cwt1, freqs, freq_bands, ret_type="angle")
        for j in range(i, arrays.shape[0]):
            fcwt_obj2.cwt(arrays[j], scales, cwt2)
            bands2 = get_cwt_bands(cwt1, freqs, freq_bands, ret_type="angle")

            for m in range(len(freq_bands)):
                band_synchrony = 1 - np.sin(np.abs(bands1[m] - bands2[m]) / 2)
                matrices[m][i, j] = band_synchrony
    return matrices


def synchrony_hilbert(arrays):
    syn_matrix = np.zeros((arrays.shape[0], arrays.shape[0]))
    for i in range(arrays.shape[0]):
        al1 = np.angle(signal.hilbert(arrays[i]), deg=False)
        for j in range(i, arrays.shape[0]):
            al2 = np.angle(signal.hilbert(arrays[j]), deg=False)
            band_synchrony = 1 - np.sin(np.abs(al1 - al2) / 2)
            syn_matrix[i, j] = band_synchrony
    return syn_matrix


def phase_synchrony(array_1: np.ndarray, array_2: np.ndarray):
    al1 = np.angle(array_1, deg=False)
    al2 = np.angle(array_2, deg=False)
    synchrony = 1 - np.sin(np.abs(al1 - al2) / 2)
    return synchrony


def stepped_cwt_cohy(cwt: np.ndarray, size: int):
    noverlap = 0
    nperseg = size
    step = nperseg - noverlap
    shape = cwt.shape[:-1] + ((cwt.shape[-1] - noverlap) // step, nperseg)
    strides = cwt.strides[:-1] + (step * cwt.strides[-1], cwt.strides[-1])
    temp = np.lib.stride_tricks.as_strided(cwt, shape=shape, strides=strides)
    coh = []
    for i in range(1, temp.shape[1], 1):
        sxy = np.mean(temp[:, i - 1, :], axis=1) * np.conjugate(
            np.mean(temp[:, i, :], axis=1)
        )
        sxx = np.mean(temp[:, i - 1, :], axis=1) * np.conjugate(
            np.mean(temp[:, i - 1, :], axis=1)
        )
        syy = np.mean(temp[:, i, :], axis=1) * np.conjugate(
            np.mean(temp[:, i, :], axis=1)
        )
        coh_temp = sxy / np.sqrt(sxx * syy + 1e-8)
        coh.append(coh_temp)
    coh = np.array(coh).T
    return coh


def phase_discontinuity_index(
    cwt: np.ndarray,
    freqs: np.ndarray,
    freq_dict: dict[str, Union[tuple[int, int], tuple[float, float]]],
    size: int = 5000,
    tol: float = 0.01,
) -> dict[str, float]:
    pdi = {}
    coh = stepped_cwt_cohy(cwt, size)
    coh = np.diff(np.angle(coh))
    for key, value in freq_dict.items():
        f_lim = np.where((freqs <= value[1]) & (freqs >= value[0]))[0]
        g = coh[f_lim]
        g = g.flatten()
        power2 = int(np.ceil(np.log2(g.size)))
        bw = np.cov(g)
        min_g = g.min() - bw * tol
        max_g = g.max() + bw * tol
        x = np.linspace(min_g, max_g, num=2**power2)
        y = KDEpy.FFTKDE(bw="ISJ").fit(g).evaluate(x)
        args1 = np.where((x > np.pi / 5) | (x < -np.pi / 5))[0]
        args2 = np.where((x <= np.pi / 5) & (x >= -np.pi / 5))[0]
        pdi[key] = np.sum(y[args1]) / np.sum(y[args2])
    return pdi


def coherence(
    acq1,
    acq2,
    fs=1000,
    noverlap=1000,
    nperseg=10000,
    nfft: Union[int, None] = None,
    window="hamming",
    ret_type: Literal[
        "icohere",
        "mscohere1",
        "mscohere2",
        "lcohere2019",
        "icohere2019",
        "cohy",
        "plv",
        "iplv",
    ] = "icohere",
):
    # Modified version of scipy to work with the imaginary part of coherence
    acq1 = np.asarray(acq1)
    acq2 = np.asarray(acq2)
    step = nperseg - noverlap
    shape = acq1.shape[:-1] + ((acq1.shape[-1] - noverlap) // step, nperseg)
    strides = acq1.strides[:-1] + (step * acq1.strides[-1], acq1.strides[-1])
    win = signal.get_window(window, nperseg)
    scale = 1.0 / (fs * (win * win).sum())
    temp1 = np.lib.stride_tricks.as_strided(acq1, shape=shape, strides=strides)
    temp2 = np.lib.stride_tricks.as_strided(acq2, shape=shape, strides=strides)
    temp1 = temp1 - temp1.mean(axis=1, keepdims=True)
    temp2 = temp2 - temp2.mean(axis=1, keepdims=True)
    temp1 *= win
    temp2 *= win
    if nfft is None:
        nfft = nperseg
    freqs = fft.rfftfreq(nfft, 1 / fs)
    fft1 = fft.rfft(temp1, n=nfft)
    fft2 = fft.rfft(temp2, n=nfft)
    sxx = fft1 * np.conjugate(fft1)
    syy = fft2 * np.conjugate(fft2)
    sxy = fft1 * np.conjugate(fft2)
    sxx *= scale
    sxx = sxx.mean(axis=0)
    syy *= scale
    syy = syy.mean(axis=0)
    sxy *= scale
    sxy = sxy.mean(axis=0)
    if ret_type == "icohere2019":
        # See Nolte et. al. 2004
        output = np.abs((sxy / (np.sqrt(sxx.real * sxy.real) + 1e-18)).imag)
    if ret_type == "icohere":
        # This is from Brainstorm
        cohy = sxy / np.sqrt(sxx * syy)
        output = (cohy.imag**2) / ((1 - cohy.real**2) + 1e-18)
    elif ret_type == "lcohere2019":
        # This is from Brainstorm
        cohy = sxy / np.sqrt(sxx * syy)
        output = np.abs(cohy.imag) / (np.sqrt(1 - cohy.real**2) + 1e-18)
    elif ret_type == "mscohere1":
        output = (np.abs(sxy) ** 2) / (sxx.real * syy.real)
    elif ret_type == "mscohere2":
        output = np.abs(sxy) / np.sqrt(sxx.real * syy.real)
    elif ret_type == "cohy":
        cohy = sxy / np.sqrt((sxx * syy) + 1e-18)
        output = cohy
    elif ret_type == "plv":
        plv = np.abs(sxy / np.sqrt(sxy))
        output = plv
    elif ret_type == "iplv":
        acc = sxy / np.sqrt((sxx * syy) + 1e-18)
        iplv = np.abs(acc.imag)
        rplv = acc.real
        rplv = np.clip(plv, -1, 1)
        mask = np.abs(rplv) == 1
        rplv[mask] = 0
        ciplv = iplv / np.sqrt(1 - rplv**2)
        output = ciplv
    # elif ret_type == "pli":
    #     pli = np.abs(np.sign(sxy.imag))
    #     output = cohy
    # elif ret_type == "upli":
    #     pli = np.abs(np.sign(sxy.imag))
    #     upli = pli**2 - 1
    #     output = upli
    # elif ret_type == "dpli":
    #     dpli = np.heaviside(np.imag(sxy), 0.5)
    #     output = dpli
    # elif ret_type == "wpli":
    #     num = np.abs(sxy.imag)
    #     denom = np.abs(sxy.imag)
    #     z_denom = np.where(denom == 0.0)[0]
    #     denom[z_denom] = 1.0
    #     con = num / denom
    #     con[z_denom] = 0.0
    #     output = con
    # elif ret_type == "dwpli":
    #     sum_abs_im_csd = np.abs(sxy.imag)
    #     sum_sq_im_csd = (sxy.imag) ** 2
    #     denom = sum_abs_im_csd**2 - sum_sq_im_csd
    #     z_denom = np.where(denom == 0.0)
    #     denom[z_denom] = 1.0
    #     con = (sxy**2 - sum_sq_im_csd) / denom
    #     con[z_denom] = 0.0
    #     output = con
    # elif ret_type == "ppc":
    #     denom = np.abs(sxy)
    #     z_denom = np.where(denom == 0.0)
    #     denom[z_denom] = 1.0
    #     this_acc = sxy / denom
    #     this_acc[z_denom] = 0.0
    #     (this_acc * np.conj(this_acc) - 1)  # / (1 * (1 - 1))
    else:
        AttributeError(
            "Return type must be icohere, icohere2019, lcohere2019, mscohere or cohy"
        )
    return freqs, output


def phase_slope_index(
    cohy,
    fs=1000,
    f_band=[4, 10],
):
    freqs = np.linspace(0, fs * 0.5, cohy.size)
    f_ind = np.where((freqs > f_band[0]) & (freqs < f_band[1]))[0]
    psi = np.sum(np.conj(cohy[f_ind[0] : f_ind[-2]] * cohy[f_ind[1] : f_ind[-1]]).imag)
    return psi


def short_time_energy(
    array, window: Windows = "hamming", wlen: int = 0.2, fs: Union[float, int] = 1000.0
):
    wlen = int(wlen * fs)
    win_array = signal.get_window(window, wlen)
    win_array /= win_array.sum()
    se_array = signal.convolve(array**2, win_array)
    se_array = se_array[wlen // 2 : array.size + wlen // 2]
    return se_array


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
    power2 = int(np.ceil(np.log2(ste.size)))
    # min_v = ste.min()
    # max_v = ste.max()
    # x = np.linspace(min_v - np.abs(min_v * tol), max_v + max_v * tol, num=2**power2)

    # Calculating x comes from Seaborn KDE
    bw = np.cov(ste)
    min_ste = ste.min() - bw * tol
    max_ste = ste.max() + bw * tol
    x = np.linspace(min_ste, max_ste, num=2**power2)
    y = KDEpy.FFTKDE(kernel="biweight", bw="ISJ").fit(ste, weights=None).evaluate(x)

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


def get_pairwise_coh(
    acqs,
    ret_type: Literal[
        "icohere", "mscohere", "lcohere2019", "icohere2019", "cohy"
    ] = "mscohere",
):
    gamma = np.zeros((128, 128))
    theta = np.zeros((128, 128))
    beta = np.zeros((128, 128))
    f = fft.rfftfreq(10000, 1 / 1000)
    g = np.where((f <= 80) & (f > 30))[0]
    t = np.where((f <= 10) & (f > 4))[0]
    b = np.where((f <= 30) & (f > 12))[0]
    for i in range(len(acqs)):
        for j in range(i, len(acqs)):
            imag_coh = coherence(acqs[i], acqs[j], ret_type=ret_type)
            # _, imag_coh = signal.coherence(
            #     acq1, acqs[j], window="hamming", nperseg=10000, noverlap=1000
            # )
            gamma[i, j] = np.mean(imag_coh[g])
            theta[i, j] = np.mean(imag_coh[t])
            beta[i, j] = np.mean(imag_coh[b])
    return theta, beta, gamma


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
