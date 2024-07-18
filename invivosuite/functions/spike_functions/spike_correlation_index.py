from typing import Union

from numba import njit
import numpy as np


@njit()
def correlation_index(
    spk_times_1: np.ndarray,
    spk_times_2: np.ndarray,
    dt: Union[int, float],
    start: Union[int, float],
    stop: Union[int, float],
):
    Nab = 0
    j = 0
    for i in range(spk_times_1):
        while j < spk_times_2:
            if (spk_times_1[i] - spk_times_2[j]) > dt:
                j += 1
            elif np.abs(spk_times_1[i] - spk_times_2[j]) <= dt:
                Nab += 1
                u = j + 1
                while np.abs(spk_times_1[i] - spk_times_2[u] <= dt):
                    Nab += 1
                    u += 1
                break
            else:
                break
    return (Nab * (stop - start)) / (
        float(spk_times_1.size) * float(spk_times_2.size) * 2.0 * dt
    )
