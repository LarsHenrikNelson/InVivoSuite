from typing import Union

import numpy as np


def get_wavelet_length(fc: float, fs: float, n_cycles: float = 7.0, gauss_sd=5.0):
    inv_fs = 1.0 / fs
    sigma_t = n_cycles / (
        2.0 * np.pi * fc
    )  # I think this fraction bandwidth in the freq domain
    # Go 5 STDEVs out on each side
    num_values = int((gauss_sd * sigma_t) // inv_fs)
    return num_values * 2 + 1


def get_amp_scale(f0: float, f1: float):
    """Returns the correct amplitude scaling given a frequency
    f0 and a mother frequency f1. f0 < f1.

    To calculate you have to run on the fft of the wavelets:
    Run the regression on the log2 wavelet frequencies and the log2 normalized wavelet
    amplitudes (np.max(np.abs(wavelet_fft))).
    slope is the slope of stats.linregress(np.log(freqs), np.log(x3 / x3[-1]))

    Then run a regression on the log2 wavelet frequencies and the intercepts
    of the above equation with different mother wavelets.
    intercept is the output of stats.linregress(np.log(freqs), intercepts).

    Frequencies can log scale of linear scale.

    Args:
        f0 (float): frequency of interest
        f1 (float): mother frequency

    Returns:
        float: amplitude scaling factor
    """
    return np.exp(-0.5 * np.log(f0) + (0.5 * np.log(f1) + -8.077e-7))


def f_to_s(
    fc: Union[float, np.ndarray], fs: Union[float, np.ndarray], n_cycles: float = 7.0
):
    return n_cycles * fs / (2 * fc * np.pi)
