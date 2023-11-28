from typing import Literal

import numpy as np
from scipy import fft

from .tapered_spectra import dpss_windows


def fast_multitaper(
    acq: np.ndarray,
    k: int = 5,
    nw: int = 3,
    fs: float = 1.0,
    nperseg: int = 10000,
    noverlap: int = 5000,
    nfft: int = 10000,
    method: Literal["adapt", "unity", "eigen"] = "adapt",
    ret_type: Literal["pxx", "sxx"] = "sxx",
):
    """DPSS multipaper FFT in Scipy style. Works the same as Scipy Welch except
    that a multitaper window is used. The function is based off of the spectrum python
    package except this is much faster.

    Args:
        acq (np.ndarray): np.array to create multitaper spectrum from
        k (int, optional): _description_. Defaults to 5.
        nw (int, optional): _description_. Defaults to 3.
        fs (float, optional): sample rate. Defaults to 1.0.
        nperseg (int, optional): Length of segments. Defaults to 10000.
        noverlap (int, optional): Overlap of segments. Defaults to 5000.
        nfft (int, optional): Lenght of fft. Defaults to 10000.
        method ("adapt", "unity", "eigen", optional): _description_. Defaults to "adapt".
        ret_type ("pxx", "sxx", optional): Spectral type to return stationary (pxx) or moving (sxx). Defaults to "sxx".

    Raises:
        ValueError: Acq must a single row or column np.array.
        ValueError: noverlap must be less than nperseg.
        ValueError: nfft must be greater than or equal to nperseg.
        ValueError: ret_type must be: sxx or pxx.

    Returns:
        _type_: _description_
    """
    ndim = acq.shape
    if len(ndim) > 2:
        raise ValueError("Acq must a single row or column np.array.")
    if len(ndim) == 2:
        if ndim[1] > ndim[0]:
            acq = acq.flatten()
    if noverlap >= nperseg:
        raise ValueError("noverlap must be less than nperseg.")
    if nperseg > acq.size:
        nperseg = acq.size
    if nfft < nperseg:
        raise ValueError("nfft must be greater than or equal to nperseg.")
    step = nperseg - noverlap
    shape = acq.shape[:-1] + ((acq.shape[-1] - noverlap) // step, nperseg)
    strides = acq.strides[:-1] + (step * acq.strides[-1], acq.strides[-1])
    temp = np.lib.stride_tricks.as_strided(acq, shape=shape, strides=strides)

    # Neither of these functions work with very large nperseg, at least on windows
    # tapers, eigenvalues = signal.windows.dpss(nperseg, nw, k, return_ratios=True)
    # tapers, eigenvalues = dpss(nperseg, NW=3, k=5)

    tapers, eigenvalues = dpss_windows(nperseg, nw, k)
    tapers = tapers.T

    p = np.zeros((temp.shape[0], k, temp.shape[1]))
    for i in range(k):
        p[:, i, :] = temp * tapers[i]
    # w = fft.fft(p * tapers, n=nfft)
    w = fft.fft(p, n=nfft)
    sk = np.abs(w) ** 2

    # Create weights
    weights = create_mt_weights(sk, temp, p, nfft, eigenvalues, method)
    sk *= weights
    Sxx = np.mean(sk, axis=1).T
    if nfft % 2 == 0:
        freqs = np.linspace(0, fs * 0.5, (nfft // 2) + 1)
        Sxx = Sxx[0 : (Sxx.shape[0] // 2) + 1,]
    else:
        freqs = np.linspace(0, fs * 0.5, (nfft + 1) // 2)
        Sxx = Sxx[0 : (Sxx.shape[0] + 1) // 2,]
    if ret_type == "sxx":
        return freqs, Sxx
    elif ret_type == "pxx":
        Pxx = Sxx.mean(axis=1)
        return freqs, Pxx
    else:
        raise ValueError("ret_type must be: sxx or pxx.")


# %%
# Vectorized DPSS FFT
def create_mt_weights(sk, temp, p, nfft, eigenvalues, method):
    if method == "adapt":
        # p = np.zeros((temp.shape[0], eigenvalues.size, temp.shape[1]))
        # for i in range(5):
        #     p[:, i, :] = temp.copy()
        sig = np.diag(np.dot(temp, temp.T) / float(temp[0].size))
        s = (sk.T[:, 0, :] + sk.T[:, 1, :]) / 2  # Initial spectrum estimate
        tols = 0.0005 * sig / float(nfft)
        eig = np.full((temp.shape[0], eigenvalues.size), eigenvalues)
        a = sig.reshape((sig.size, 1)) * (1 - eig)
        weights = np.zeros(sk.shape)
        for index in range(tols.size):
            i = 0
            S = s[:, index].reshape((nfft, 1))
            S1 = np.zeros((nfft, 1))
            wk = np.ones((nfft, 1)) * eigenvalues.T
            while np.sum(np.abs(S - S1)) / nfft > tols[index] and i < 100:
                i = i + 1
                # calculate weights
                b1 = np.multiply(S, np.ones((1, eigenvalues.size)))
                b2 = np.multiply(S, eigenvalues.T) + np.ones((nfft, 1)) * a[index].T
                b3 = b1 / b2

                # calculate new spectral estimate
                wk = (b3**2) * (np.ones((nfft, 1)) * eigenvalues.T)
                S1 = np.sum(wk.T * sk[index], axis=0) / np.sum(wk.T, axis=0)
                S1 = S1.reshape(nfft, 1)
                S, S1 = S1, S  # swap S and S1
            weights[index] = wk.T
    elif method == "unity":
        weights = np.ones((temp.shape[0], eigenvalues.size, temp.shape[1]))
    elif method == "eigen":
        weights = eigenvalues / np.arange(1, eigenvalues.size + 1)
        weights = weights.reshape(1, eigenvalues.size, 1)
    else:
        raise ValueError("method must be adapt, unity, or eigen")
    return weights
