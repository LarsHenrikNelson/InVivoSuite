import math

import numpy as np
from KDEpy import TreeKDE
from numba import njit
from scipy import fft
from sklearn.decomposition import PCA

__all__ = [
    "find_bursts",
]


def is_ngb(xcorr_array, fs):
    out_fft = fft.rfft(xcorr_array)
    val = 1.0 / (xcorr_array.size * 1 / fs)
    N = xcorr_array.size // 2 + 1
    freqs = np.arange(0, N, dtype=int) * val
    ngb_indexes = np.where((freqs >= 50) & (freqs <= 70))[0]
    wb_indexes = np.where((freqs >= 40) & (freqs <= 300))[0]
    if np.max(out_fft[ngb_indexes]) > np.max(out_fft[wb_indexes]):
        return True
    else:
        return False


@njit()
def find_bursts(spikes, freq):
    bursts = []
    mean_isi = 1 / freq
    isi_to_start = mean_isi / 2
    isi_to_end = mean_isi
    i = 0
    while i < spikes.size - 2:
        if (
            spikes[i + 1] - spikes[i] < isi_to_start
            and spikes[i + 2] - spikes[i + 1] < isi_to_start
        ):
            bur = []
            bur.extend((i, i + 1, i + 2))
            i += 2
            add_spikes = True
            while add_spikes:
                if spikes[i + 1] - spikes[i] <= isi_to_end:
                    bur.append(spikes[i + 1])
                    i += 1
                else:
                    add_spikes = False
    return bursts


@njit()
def find_fwd_burst(freq, burst: np.ndarray):
    snurprise = []
    for i in range(3, len(burst)):
        p = poisson_surp(freq, burst[:i])
        s = -np.log10(p)
        snurprise.append(s)
    b = np.argmax(snurprise)
    return burst[: 3 + b]


@njit()
def find_bwk_burst(freq, burst: np.ndarray):
    snurprise = []
    for i in range(3, len(burst)):
        p = poisson_surp(freq, burst[:i])
        s = -np.log10(p)
        snurprise.append(s)
    b = np.argmax(snurprise)
    return burst[: 3 + b]


@njit()
def poisson_surp(r, burst):
    rT = r * (burst[-1] - burst[0])
    e_rT = np.exp(-rT)
    temp = ((rT) ** len(burst)) / math.factorial(len(burst))
    p = -np.log10(e_rT + temp)
    return p


def CMA(dist):
    cma = np.empty(dist.size)
    for index in range(1, len(dist)):
        cma[index] = np.mean(dist[:index]) / (dist[:index].size)
    return cma


def find_isi_max(spikes):
    x, y = (
        TreeKDE(kernel="gaussian", bw="ISJ")
        .fit(np.diff(spikes))
        .evaluate(np.diff(spikes).size)
    )
    cma = CMA(y)
    max_index = np.argmax(cma)
    max_isi = x[max_index]
    return max_isi


# def get_spike_cwt(spikes, fs=40000, f0=300, f1=1500, fn=100, bandwidth=2.0):
#     morl = fcwt.Morlet(bandwidth)
#     scales = fcwt.Scales(morl, fcwt.FCWT_LOGSCALES, fs, f0, f1, fn)
#     cwt_obj = fcwt.FCWT(morl, 2, False, True)
#     data = np.zeros((spikes.shape))
#     output = np.zeros((fn, spikes[0].size), dtype=np.complex64)
#     for index, i in enumerate(spikes):
#         if i.dtype != "single":
#             i = i.astype("single")
#         cwt_obj.cwt(i, scales, output)
#         c = PCA(n_components=1).fit_transform(np.abs(output.T))
#         data[index] = c.T
#     return data
