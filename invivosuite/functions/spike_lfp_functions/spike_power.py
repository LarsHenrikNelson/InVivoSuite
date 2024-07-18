import numpy as np
from numba import njit

__all__ = ["spike_triggered_lfp"]


@njit()
def spike_triggered_lfp(indexes: np.ndarray, lfp: np.ndarray, window: int):
    output = np.zeros((indexes.size, (window * 2) + 1))
    for i in range(indexes.size):
        start = int(indexes[i] - window)
        os = 0
        end = int(indexes[i] + window + 1)
        oe = int((window * 2) + 1)
        if start < 0:
            os = start * -1
            start = 0
        if end > lfp.size:
            end = lfp.size
            oe = end - start
        output[i, os:oe] = lfp[start:end]
    return output
