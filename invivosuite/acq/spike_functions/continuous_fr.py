from typing import Literal

import numpy as np
from scipy import signal
from scipy.signal import windows

from .binarize_spikes import bin_spikes


def getSDF(
    spikes: np.ndarray,
    binary_size: int,
    nperseg: int,
    fs: float = 40000.0,
    filt: Literal["gaussian", "exponential"] = "gaussian",
    sigma: float = 0.02,
):
    binned_spikes = bin_spikes(spikes, binary_size=binary_size, nperseg=nperseg)

    sampInt = 1 / (fs / nperseg)
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
    sdf = signal.convolve(binned_spikes, w)
    sdf = sdf[wlen // 2 : binned_spikes.size + wlen // 2]
    return sdf
