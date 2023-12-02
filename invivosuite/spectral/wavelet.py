from typing import Literal

import multiprocessing

import numpy as np
from numba import njit
from scipy import fft

# import pyfftw

__all__ = ["create_wavelets", "compute_cwt", "s_to_f", "f_to_s"]


@njit(cache=True)
def create_wavelets(
    freqs, fs, n_cycles, sigma=-1, zero_mean: bool = True, order: Literal[1, 2] = 1
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
    for i in range(freqs.size):
        if sigma == -1:
            sigma_t = n_cycles[index] / (2.0 * np.pi * freqs[i])
        else:
            sigma_t = n_cycles[index] / (2.0 * np.pi * sigma)

        # Go 5 STDEVs out on each side
        num_values = int((5.0 * sigma_t) // inv_fs)
        wave_len = num_values * 2 + 1
        t = np.zeros(wave_len)
        for k in range(1, num_values + 1):
            t[k + num_values] = t[k - 1 + num_values] + inv_fs
            t[num_values - k] = t[num_values - k + 1] + inv_fs * -1.0

        # Could also do
        # oscillation.imag = np.sin(2.0 * np.pi* f * t)
        # oscillation.real = np.cos(2.0 * np.pi* f * t)

        oscillation = np.exp(2.0 * 1j * np.pi * freqs[i] * t)

        if zero_mean:
            real_offset = np.exp(-2 * (np.pi * freqs[i] * sigma_t) ** 2)
            oscillation -= real_offset
        gaussian_env = np.exp(-(t**2) / (2.0 * sigma_t**2))
        oscillation *= gaussian_env
        oscillation /= np.sqrt(0.5) * np.linalg.norm(oscillation, ord=order)
        wave.append(oscillation)
        if n_cycles.size != 1:
            index += 1
    return wave


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
    conv_size = input_size + np.max([i.size for i in wavelets]) - 1
    nfft = 1 << int(np.ceil(np.log2(conv_size)))

    workers = multiprocessing.cpu_count()
    # array_fft = pyfftw.interfaces.scipy_fft.fft(array, n=nfft, workers=workers)
    array_fft = fft.fft(array, n=nfft, workers=workers)
    output = np.zeros((num, input_size), dtype=np.complex128)

    for index, ws in enumerate(wavelets):
        if not skip_fft:
            # output_f = pyfftw.interfaces.scipy_fft.fft(
            #     wavelets[index], n=nfft, workers=workers
            # )
            output_f = fft.fft(ws, n=nfft, workers=workers)
        else:
            output_f = ws
        # ret = pyfftw.interfaces.scipy_fft.fft(output_f * array_fft, workers=workers)
        ret = fft.ifft(output_f * array_fft, workers=workers)
        ret = ret[: ws.size + input_size]

        startind = (ret.size - input_size) // 2
        endind = startind + input_size

        output[index, :] = ret[startind:endind]

    return freqs, output


def get_support(fb, scale):
    return fb * scale * 3.0


def f_to_s(freqs, fs):
    return fs / freqs


def s_to_f(scales, fs):
    return fs / scales


def fwhm_sigma(sigma):
    return sigma * 2 * np.sqrt(2 * np.log(2))


def fwhm_freq(freq, n_cycles=7.0):
    sigma = n_cycles / (2.0 * np.pi * freq)
    return sigma * 2 * np.sqrt(2 * np.log(2))


def fwhm_to_cycles(fwhm, freqs):
    return fwhm * np.pi * np.array(freqs) / np.sqrt(2 * np.log(2))


def morlet(w, mu):
    cs = (1 + np.exp(-(mu**2)) - 2 * np.exp(-3 / 4 * mu**2)) ** (-0.5)
    ks = np.exp(-0.5 * mu**2)
    C1 = np.sqrt(2) * cs * np.pi**0.25
    C0 = -0.5
    return C1 * (np.exp(C0 * (w - mu) ** 2) - ks * np.exp(C0 * w**2))
