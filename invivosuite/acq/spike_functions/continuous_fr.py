from typing import Literal

import numpy as np
from scipy import signal
from scipy.signal import windows

from .binarize_spikes import bin_spikes


__all__ = ["create_continuous_spikes"]


def create_continuous_spikes(
    spikes: np.ndarray,
    binary_size: int,
    nperseg: int = 0,
    fs: float = 40000.0,
    filt: Literal["gaussian", "exponential"] = "gaussian",
    sigma: float = 0.02,
) -> np.ndarray:
    """Generates a continuous spike rate function. Data has to be binned.

    Args:
        spikes (np.ndarray): Array of spike times in samples
        binary_size (int): length of the recording in samples
        nperseg (int): Number of samples per bin. Use 0 for no binning
        fs (float, optional): Sample rate. Defaults to 40000.0.
        filt (Literal[&quot;gaussian&quot;, &quot;exponential&quot;], optional): Convolution filter. Defaults to "gaussian".
        sigma (float, optional): Sigma for the filter. Defaults to 0.02.

    Returns:
        np.ndarray: _description_
    """
    if nperseg > 0:
        temp = bin_spikes(spikes, binary_size=binary_size, nperseg=nperseg)
        sampInt = 1 / (fs / nperseg)
    else:
        temp = np.zeros(binary_size)
        temp[spikes] = 1
        sampInt = sigma
    if filt == "exponential":
        filtPts = int(5 * sigma / sampInt)
        w = np.zeros(filtPts * 2)
        w[-filtPts:] = windows.exponential(
            filtPts, center=0, tau=sigma / sampInt, sym=False
        )
        wlen = w.size
        w /= w.sum()
    else:
        wlen = int(4.0 * sampInt + 0.5) + 1
        w = windows.gaussian(wlen, sampInt)
        w /= w.sum()
    sdf = signal.convolve(temp, w)
    sdf = sdf[wlen // 2 : temp.size + wlen // 2]
    return sdf
