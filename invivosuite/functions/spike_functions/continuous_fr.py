from typing import Literal
from functools import cache

import numpy as np
from numba import njit
from scipy import signal

from .binarize_spikes import bin_spikes
from ..signal_functions import create_window


__all__ = ["create_continuous_spikes", "Methods", "Windows"]

Windows = Literal["gaussian", "exponential", "exponential_abi", "boxcar"]
Methods = Literal["convolve", "add", "set"]


def _create_array(array: np.ndarray, window: np.ndarray, method: Methods = "convolve"):
    if method == "convolve":
        sdf = signal.oaconvolve(array, window)
        sdf = sdf[window.size // 2 : array.size + window.size // 2]
    else:
        sdf = _set_array(array, window, method)
    return sdf


@njit(cache=True)
def _set_array(array: np.ndarray, window: np.ndarray, method: Methods):
    sdf = np.zeros(array.size)
    indexes = np.where(array > 0)[0]
    if (window.size // 2) == 0:
        temp = np.zeros(window.size + 1)
        temp[: window.size] = window
        window = temp
    dt = window.size // 2
    for i in indexes:
        start = max(i - dt, 0)
        end = min(array.size, i + dt)
        if start < 0:
            wstart = end - window.size
        else:
            wstart = 0
        wend = min(end - start, window.size)
        if method == "add":
            sdf[start:end] += window[wstart:wend]
        else:
            sdf[start:end] = window[wstart:wend]
    return sdf

def create_continuous_spikes(
    spikes: np.ndarray,
    binary_size: int,
    nperseg: int = 1,
    fs: float = 40000.0,
    window: Windows = "boxcar",
    sigma: float = 0.02,
    method: Methods = "convolve",
) -> np.ndarray:
    """Generates a continuous spike rate function. Data does not have to be binned.

    Args:
        spikes (np.ndarray): Array of spike times in samples
        binary_size (int): length of the recording in samples
        nperseg (int): Number of samples per bin. Use 0 for no binning
        fs (float, optional): Sample rate. Defaults to 40000.0.
        window (Literal[&quot;gaussian&quot;, &quot;exponential&quot;], optional): Convolution filter. Defaults to "gaussian".
        sigma (float, optional): Sigma for the filter. Defaults to 0.02.

    Returns:
        np.ndarray: _description_
    """
    if nperseg > 1:
        temp = bin_spikes(spikes, binary_size=binary_size, nperseg=nperseg)
        sampInt = 1 / (fs / nperseg)
    else:
        temp = np.zeros(binary_size)
        temp[spikes] = 1
        sampInt = 1
    w = create_window(window, sigma, sampInt)
    return _create_array(temp, w, method=method)
