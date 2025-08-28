import functools
import os

import joblib
import numpy as np
from numba import njit
from numpy.random import default_rng

from ..signal_functions import bin_by

def rand_cfc(phi: np.ndarray, amp: np.ndarray, steps: int, seed: int = 42):
    rng = default_rng(seed)
    amp_rand = rng.permutation(amp)
    _, temp = bin_by(phi, amp_rand, steps, -np.pi, np.pi)
    output = temp.max() - temp.min()
    return output


def cfc_pvalue(
    phi: np.ndarray,
    amp: np.ndarray,
    steps: int,
    iterations: int = 1000,
    seed: int = 42,
    parallel: bool = False,
    threads: int = -1,
):
    if not parallel:
        rng = default_rng(seed)
        output = np.zeros(iterations)
        for i in range(iterations):
            amp_rand = rng.permutation(amp)
            _, temp = bin_by(phi, amp_rand, steps, -np.pi, np.pi)
            output[i] = temp.max() - temp.min()
    else:
        pfunc = functools.partial(rand_cfc, phi=phi, amp=amp, steps=steps)
        physical_core_count = os.cpu_count() // 2
        if threads == -1:
            threads = physical_core_count - 1
        elif threads >= physical_core_count:
            threads = physical_core_count - 1
        output = joblib.Parallel(n_jobs=threads)(
            joblib.delayed(pfunc)(num=i) for i in range(iterations)
        )
        output = np.array(output)

    _, cfc_data = bin_by(phi, amp, steps, -np.pi, np.pi)
    cfc_range = cfc_data.max() - cfc_data.min()
    return (output > cfc_range).sum() / iterations


def modulation_index(phi: np.ndarray, amp: np.ndarray, steps: int):
    _, binned_data = bin_by(phi, amp, steps, -np.pi, np.pi)

    normalized = binned_data / binned_data.sum()
    max_entropy = np.log(len(binned_data))
    mi = (max_entropy - (-np.sum(normalized * np.log(normalized)))) / max_entropy
    return mi
