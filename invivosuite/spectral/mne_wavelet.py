import os
from typing import Literal, Union

# import joblib
import numpy as np
from numba import njit
import pyfftw

# from scipy import fft

from ..spectral import fft

__all__ = ["mne_wavelets", "mne_cwt", "gen_cwt"]

"""
This is  modified version of MNE tfr: 
https://github.com/mne-tools/mne-python/blob/main/mne/time_frequency/tfr.py
with some modifications such as allowing the guassian sd to be modified
and the normalization order to be change to 1 or 2.
"""


@njit(cache=True)
def mne_wavelets(
    freqs,
    fs,
    n_cycles,
    sigma=-1,
    zero_mean: bool = True,
    gauss_sd=5.0,
    order: Literal[1, 2] = 1,
):
    """Compute Morlet wavelets for the given frequency range.

    Parameters
    ----------
    sfreq : float
        The sampling Frequency.
    freqs : float | array-like, shape (n_freqs,)
        Frequencies to compute Morlet wavelets for.
    n_cycles : float | array-like, shape (n_freqs,)
        Number of cycles. Can be a fixed number (float) or one per frequency
        (array-like).
    sigma : float, default None
        It controls the width of the wavelet ie its temporal
        resolution. If sigma is None the temporal resolution
        is adapted with the frequency like for all wavelet transform.
        The higher the frequency the shorter is the wavelet.
        If sigma is fixed the temporal resolution is fixed
        like for the short time Fourier transform and the number
        of oscillations increases with the frequency.
    zero_mean : bool, default False
        Make sure the wavelet has a mean of zero.

    Returns
    -------
    Ws : list of ndarray | ndarray
        The wavelets time series. If ``freqs`` was a float, a single
        ndarray is returned instead of a list of ndarray.
    """
    inv_fs = 1.0 / fs
    index = 0
    wave = []
    for fc in freqs:
        if sigma == -1:
            sigma_t = n_cycles[index] / (2.0 * np.pi * fc)
        else:
            sigma_t = n_cycles[index] / (2.0 * np.pi * sigma)

        # Go 5 STDEVs out on each side
        num_values = int((gauss_sd * sigma_t) // inv_fs)

        # Alt eq if not using fixed sigma: num_values = int((2*np.pi*fc/fs)*5.0)

        t = np.arange(-num_values, num_values + 1) / fs

        oscillation = np.exp(2.0 * 1j * np.pi * fc * t)

        if zero_mean:
            real_offset = np.exp(-2 * (np.pi * fc * sigma_t) ** 2)
            oscillation -= real_offset
        gaussian_env = np.exp(-(t**2) / (2.0 * sigma_t**2))
        oscillation *= gaussian_env
        oscillation /= np.sqrt(0.5) * np.linalg.norm(oscillation, ord=order)

        """
        This is an alternative run that is easier to see the explicit
        equation similar to what is found on wikipedia however it is
        twice as slow.

        oscillation = np.zeros(num_values * 2 + 1, dtype=np.complex128)
        index = 0
        real_offset = 0.0
        if zero_mean:
            real_offset = np.exp(-2 * (np.pi * freqs[i] * sigma_t) ** 2)
        for k in range(-num_values, num_values + 1):
            ti = k / fs
            oscillation[index] = (
                np.exp(2.0 * 1j * np.pi * freqs[i] * ti) - real_offset
            ) * np.exp(-(ti**2) / (2.0 * sigma_t**2))
            index += 1
        oscillation /= np.sqrt(0.5) * np.linalg.norm(oscillation, ord=order)
        """

        wave.append(oscillation)
        if n_cycles.size != 1:
            index += 1
    return wave


def simple_cwt(
    wavelet: np.ndarray,
    array_fft: np.ndarray,
    array_size: int,
    nfft: int,
    threads: int = -1,
):
    ret = fft.ifft(fft.c2c_fft(wavelet, nfft=nfft) * array_fft, threads=1)
    ret = ret[: wavelet.size + array_size - 1]

    startind = (ret.size - array_size) // 2
    endind = startind + array_size

    return ret[startind:endind]


def gen_cwt(
    array: np.ndarray,
    output: np.ndarray,
    wavelets: Union[list, np.ndarray],
    skip_fft: bool = False,
    threads=-1,
):
    input_size = array.size
    conv_size = input_size + np.max([i.size for i in wavelets]) - 1
    nfft = 1 << int(np.ceil(np.log2(conv_size)))

    array_fft = fft.r2c_fft(array, nfft=nfft, threads=threads)

    for index, ws in enumerate(wavelets):
        ret = fft.c2c_ifft(fft.c2c_fft(ws, nfft=nfft, threads=threads) * array_fft, threads=threads)
        ret = ret[: ws.size + input_size - 1]

        startind = (ret.size - input_size) // 2
        endind = startind + input_size

        output[index, :] = ret[startind:endind]


def mne_cwt(
    array: np.ndarray,
    f0: float,
    f1: float,
    fs: float,
    num: int,
    scaling: Literal["linear", "log"] = "log",
    n_cycles: int = 7,
    sigma: int = -1,
    zero_mean: bool = True,
    order: Literal[1, 2] = 2,
    threads: int = -1,
):
    if not isinstance(n_cycles, int):
        n_cycles = np.array(n_cycles).ravel()
    else:
        n_cycles = np.array([n_cycles])

    if scaling == "linear":
        freqs = np.linspace(start=f0, stop=f1, num=num)
    else:
        freqs = np.logspace(start=np.log10(f0), stop=np.log10(f1), num=num)

    wavelets = mne_wavelets(
        freqs=freqs,
        fs=fs,
        n_cycles=n_cycles,
        sigma=sigma,
        zero_mean=zero_mean,
        order=order,
    )

    if threads == -1:
        threads = os.cpu_count() // 2

    output = np.zeros((len(wavelets), array.size), dtype=np.complex128)

    gen_cwt(array, output, wavelets, skip_fft=False, threads=threads)

    return freqs, output


class mneCWT:
    """mneCWT class to reuse the output buffer and wavelets."""

    def __init__(
        self,
        input_size: int,
        f0: float,
        f1: float,
        fs: float,
        num: int,
        scaling: str = "linear",
        n_cycles: int = 7,
        sigma: int = -1,
        zero_mean: bool = True,
        order: Literal[1, 2] = 1,
        threads: int = -1,
    ):
        self.input_size = input_size
        self.f0 = f0
        self.f1 = f1
        self.fs = fs
        self.num = num
        self.scaling = scaling

        if not isinstance(n_cycles, int):
            self.n_cycles = np.array(n_cycles).ravel()
        else:
            self.n_cycles = np.array([n_cycles])

        self.sigma = sigma
        self.zero_mean = zero_mean
        self.order = order

        if threads == -1:
            self.threads = os.cpu_count() // 2
        else:
            self.threads = threads

        if scaling == "linear":
            self.freqs = np.linspace(start=f0, stop=f1, num=num)
        else:
            self.freqs = np.logspace(start=np.log10(f0), stop=np.log10(f1), num=num)
    
        self.wavelets = mne_wavelets(
            self.freqs,
            self.fs,
            self.n_cycles,
            self.sigma,
            self.zero_mean,
            self.gauss_sd,
            self.order,
        )

    def cwt(self, data):
        output = np.zeros((len(self.wavelets), data.size), dtype=np.complex128)
        gen_cwt(data, output, self.wavelets, skip_fft=False, threads=self.threads)
        return output
