from typing import Union

import numpy as np

__all__ = ["sfa_local_var", "sfa_divisor", "sfa_abi", "sfa_rlocal_var"]


def sfa_local_var(isi: np.ndarray) -> float:
    """
    This function calculates the local variance in spike frequency
    accomadation that was drawn from the paper:
    Shinomoto, Shima and Tanji. (2003). Differences in Spiking Patterns
    Among Cortical Neurons. Neural Computation, 15, 2823-2842.

    Returns
    -------
    None.

    """
    if len(isi) < 2 or isi is np.nan:
        return np.nan
    isi_shift = isi[1:]
    isi_cut = isi[:-1]
    n_minus_1 = len(isi_cut)
    output = (
        np.sum((3 * (isi_cut - isi_shift) ** 2) / (isi_cut + isi_shift) ** 2)
        / n_minus_1
    )
    return output


def sfa_rlocal_var(isi: np.ndarray, R: Union[float, int]) -> float:
    """
    This function calculates the revised local variance in spike frequency
    accomadation that was drawn from the paper:
    Shinomoto, S. et al. Relating Neuronal Firing Patterns to Functional Differentiation
    of Cerebral Cortex. PLoS Comput Biol 5, e1000433 (2009).


    Returns
    -------
    None.

    """
    if len(isi) < 2 or isi is np.nan:
        return np.nan
    isi_plus = isi[1:] + isi[:1]
    isi_mult = (isi[1:] * isi[:1]) * 4
    intermediate = (1 - 4 * isi_mult / (isi_plus**2)) * (1 + 4 * R / isi_plus)
    multiplier = 3 / (isi.size - 1)
    return multiplier * np.sum(intermediate)


def sfa_divisor(isi: list[np.ndarray]) -> np.ndarray[float]:
    """
    The idea for the function was initially inspired by a program called
    Easy Electropysiology (https://github.com/easy-electrophysiology).
    """
    if len(isi) > 1 or np.isnan(isi):
        if isi[-1] > 0:
            return isi[0] / isi[-1]
        else:
            return np.nan
    else:
        return 0.0


def sfa_abi(isi: list[np.ndarray]) -> np.ndarray[float]:
    """
    This function calculates the spike frequency adaptation. A positive
    number means that the spikes are speeding up and a negative number
    means that spikes are slowing down. This function was inspired by the
    Allen Brain Institutes IPFX analysis program
    https://github.com/AllenInstitute/ipfx/tree/
    db47e379f7f9bfac455cf2301def0319291ad361
    """
    if len(isi) <= 1:
        return np.nan
    else:
        if np.allclose((isi[1:] + isi[:-1]), 0.0):
            return 0.0
        else:
            norm_diffs = (isi[1:] - isi[:-1]) / (isi[1:] + isi[:-1])
            norm_diffs[(isi[1:] == 0) & (isi[:-1] == 0)] = 0.0
            return np.nanmean(norm_diffs)
