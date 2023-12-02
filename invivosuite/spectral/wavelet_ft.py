# %%
from typing import Literal

import tensorflow as tf
import numpy as np


def compute_cwt(
    array: np.ndarray,
    f0: float,
    f1: float,
    fs: float,
    num: int,
    scaling: str = "linear",
    n_cycles: int = 7,
    sigma: int = -1,
    zero_mean: bool = True,
    order: Literal[1, 2] = 2,
    wavelets: list[np.ndarray[np.complex128]] = None,
    skip_fft=False,
):
    if not isinstance(n_cycles, int):
        n_cycles = np.array(n_cycles).ravel()
    else:
        n_cycles = np.array([n_cycles])

    if scaling == "linear":
        freqs = np.linspace(start=f0, stop=f1, num=num)
    else:
        freqs = np.logspace(start=np.log10(f0), stop=np.log10(f1), num=num)
    if wavelets is None:
        wavelets = create_wavelets(
            freqs=freqs,
            fs=fs,
            n_cycles=n_cycles,
            sigma=sigma,
            zero_mean=zero_mean,
            order=order,
        )
    input_size = array.size
    newsize = input_size + wavelets.shape[1] - 1
    nfft = 1 << int(np.ceil(np.log2(newsize)))

    # workers = multiprocessing.cpu_count()
    # array_fft = pyfftw.interfaces.scipy_fft.fft(array, n=nfft, workers=workers)
    array_fft = tf.signal.rfft(array, n=nfft)

    if not skip_fft:
        # output_f = pyfftw.interfaces.scipy_fft.fft(
        #     wavelets[index], n=nfft, workers=workers
        # )
        wavelet_fft = tf.signal.fft(wavelets, n=nfft)
    else:
        wavelet_fft = wavelets
    # ret = pyfftw.interfaces.scipy_fft.fft(wavelet_fft * array_fft, workers=workers)
    ret = tf.signal.ifft(wavelet_fft * array_fft)

    ret = ret[:, :newsize]
    outsize = input_size
    currsize = ret.shape[1]
    startind = int((currsize - outsize) // 2)
    endind = startind + outsize
    ret = ret[:, startind:endind]

    return freqs, ret


# %%
