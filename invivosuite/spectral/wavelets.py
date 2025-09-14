import numpy as np
from numba import njit

import pyfftw

from .wavelet_utils import f_to_s

class Wavelet:
    def __init__(
        self,
        fc: float,
        fs: float,
        n_cycles: float = 7.0,
        gauss_sd: float = 5.0,
        sigma: int = -1,
        zero_mean: bool = True,
        imaginary: bool = True,
    ):
        self.fc = fc
        self.fs = fs
        self.n_cycles = n_cycles
        self.gauss_sd = gauss_sd
        self.sigma = sigma
        self.zero_mean = zero_mean
        self.imaginary = imaginary
        self.scale = f_to_s(self.fc, self.fs, self.n_cycles)

    def time(self):
        inv_fs = 1.0 / self.fs

        # I think this fraction bandwidth in the freq domain
        if self.sigma == -1:
            sigma_t = self.n_cycles / (2.0 * np.pi * self.fc)
        else:
            sigma_t = self.n_cycles / (2.0 * np.pi * self.sigma)

        # Go gauss_sd STDEVs out on each side
        num_values = int((self.gauss_sd * sigma_t) // inv_fs)
        t = np.arange(-num_values, num_values + 1) / self.fs
        oscillation = np.exp(2.0 * 1j * np.pi * self.fc * t)
        if self.zero_mean:
            real_offset = np.exp(-2 * (np.pi * self.fc * sigma_t) ** 2)
            oscillation -= real_offset
        gaussian_env = np.exp(-(t**2) / (2.0 * sigma_t**2))
        oscillation *= gaussian_env
        oscillation /= np.sqrt(0.5) * np.linalg.norm(oscillation, ord=2)
        if self.imaginary:
            return oscillation
        else:
            return oscillation.real

    def frequency(self, size: int):
        wavelet = self.time()
        a = pyfftw.zeros_aligned(size, dtype="complex128")
        b = pyfftw.empty_aligned(size, dtype="complex128")

        a[: wavelet.size] = wavelet

        forward_fft = pyfftw.FFTW(a, b, threads=1)
        forward_fft()
        if self.imaginary:
            return b
        else:
            return np.abs(b)

    def daughter_length(self, fc):
        inv_fs = 1.0 / self.fs

        # I think this fraction bandwidth in the freq domain
        if self.sigma == -1:
            sigma_t = self.n_cycles / (2.0 * np.pi * self.fc)
        else:
            sigma_t = self.n_cycles / (2.0 * np.pi * self.sigma)

        # Go gauss_sd STDEVs out on each side
        num_values = int((self.gauss_sd * sigma_t) // inv_fs)
        return num_values * 2 + 1

def morlet(w, mu):
    cs = (1 + np.exp(-(mu**2)) - 2 * np.exp(-3 / 4 * mu**2)) ** (-0.5)
    ks = np.exp(-0.5 * mu**2)
    C1 = np.sqrt(2) * cs * np.pi**0.25
    C0 = -0.5
    return C1 * (np.exp(C0 * (w - mu) ** 2) - ks * np.exp(C0 * w**2))


@njit(cache=True, parallel=True)
def fcwt_wavelet(mu, size, scale=2.0):
    toradians = (2 * np.pi) / size
    norm = np.sqrt(2 * np.pi) * np.power(np.pi, -1 / 4)

    mother = np.zeros(size)
    for i in range(size):
        temp1 = (scale * i * toradians) * mu - 2.0 * np.pi * mu
        temp1 = -(temp1**2) / 2
        mother[i] = norm * np.exp(temp1)
    return mother


# mne wavelet
def mne_wavelet(
    fc: float,
    fs: float,
    n_cycles: float = 7.0,
    gauss_sd: float = 5.0,
    sigma: int = -1,
    zero_mean: bool = True,
):
    inv_fs = 1.0 / 1000.0

    # I think this fraction bandwidth in the freq domain
    if sigma == -1:
        sigma_t = n_cycles / (2.0 * np.pi * fc)
    else:
        sigma_t = n_cycles / (2.0 * np.pi * sigma)

    # Go gauss_sd STDEVs out on each side
    num_values = int((gauss_sd * sigma_t) // inv_fs)
    t = np.arange(-num_values, num_values + 1) / fs
    oscillation = np.exp(2.0 * 1j * np.pi * fc * t)
    if zero_mean:
        real_offset = np.exp(-2 * (np.pi * fc * sigma_t) ** 2)
        oscillation -= real_offset
    gaussian_env = np.exp(-(t**2) / (2.0 * sigma_t**2))
    oscillation *= gaussian_env
    oscillation /= np.sqrt(0.5) * np.linalg.norm(oscillation, ord=2)
    return oscillation


def scipy_wavelet(fc: float, fs: float, n_cycles: float = 7.0, gauss_sd=5.0):
    inv_fs = 1.0 / fs
    sigma_t = n_cycles / (
        2.0 * np.pi * fc
    )  # I think this fraction bandwidth in the freq domain
    # Go 5 STDEVs out on each side
    num_values = int((gauss_sd * sigma_t) // inv_fs)
    M = num_values * 2 + 1
    s = n_cycles * fs / (2 * fc * np.pi)
    x = np.arange(0, M) - (M - 1.0) / 2
    x /= s
    wavelet = np.exp(1j * n_cycles * x) * np.exp(-0.5 * x**2) * np.pi ** (-0.25)
    output = np.sqrt(1 / s) * wavelet
    return output
