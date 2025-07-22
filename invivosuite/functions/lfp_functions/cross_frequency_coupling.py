import functools
import os

import joblib
import numpy as np
from numpy.random import default_rng
from scipy import stats


def cross_frequency_coupling(phi: np.ndarray, amp: np.ndarray, steps: int):
    """See: https://mark-kramer.github.io/Case-Studies-Python/07.html
    for a good explanation and implementation.

    Args:
        phi (np.ndarray): The computed angle of an analytic signal
        amp (np.ndarray): The computed amplitude of an analytic signal
        steps (int): number of bins for binning the data

    Returns:
        _type_: _description_
    """
    dt = np.pi * 2 / steps
    lower = -np.pi
    upper = -np.pi + dt
    output_bins = np.linspace(-np.pi + dt, np.pi - dt, num=steps)
    binned_data = np.zeros(steps)
    for i in range(steps):
        binned_data[i] = np.mean(amp[(phi < upper) & (phi >= lower)])
        lower += dt
        upper += dt
    return output_bins, binned_data


def rand_cfc(phi: np.ndarray, amp: np.ndarray, steps: int, seed: int = 42):
    rng = default_rng(seed)
    indexes = rng.integers(0, amp.size, size=amp.size)
    amp_rand = amp[indexes]
    _, temp = cross_frequency_coupling(phi, amp_rand, steps)
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
            indexes = rng.integers(0, amp.size, size=amp.size)
            amp_rand = amp[indexes]
            _, temp = cross_frequency_coupling(phi, amp_rand, steps)
            output[i] = temp.max() - temp.min()
    else:
        pfunc = functools.partial(rand_cfc, phi=phi, amp=amp, steps=steps)
        indexes = np.arange(0, iterations)
        physical_core_count = os.cpu_count() // 2
        if threads == -1:
            threads = physical_core_count - 1
        elif threads >= physical_core_count:
            threads = physical_core_count - 1
        output = joblib.Parallel(n_jobs=threads)(
            joblib.delayed(pfunc)(num=i) for i in range(iterations)
        )
        output = np.array(output)

    _, cfc_data = cross_frequency_coupling(phi, amp, steps)
    cfc_range = cfc_data.max() - cfc_data.min()
    num_above = np.where(output > cfc_range)[0]
    return num_above.size / iterations

def modulation_index( phi: np.ndarray,
    amp: np.ndarray,
    steps: int):
    _, binned_data = cross_frequency_coupling(phi, amp, steps)
    
    normalized = binned_data/binned_data.sum()
    max_entropy = np.log(len(binned_data))
    mi = (max_entropy-(-np.sum(normalized)*np.log(normalized)))/max_entropy
    return mi 
