import numpy as np

__all__ = [
    "sfa_local_var",
    "sfa_peak",
    "sfa_divisor",
    "sfa_abi",
]


def sfa_local_var(bursts: list[np.ndarray]) -> np.ndarray[float]:
    """
    This function calculates the local variance in spike frequency
    accomadation that was drawn from the paper:
    Shinomoto, Shima and Tanji. (2003). Differences in Spiking Patterns
    Among Cortical Neurons. Neural Computation, 15, 2823-2842.

    Returns
    -------
    None.

    """
    output = np.zeros(len(bursts))
    for index, b in enumerate(bursts):
        iei = np.diff(b).astype(float)
        if len(iei) < 2 or iei is np.nan:
            output[index] = np.nan
        else:
            isi_shift = iei[1:]
            isi_cut = iei[:-1]
            n_minus_1 = len(isi_cut)
            output[index] = (
                np.sum((3 * (isi_cut - isi_shift) ** 2) / (isi_cut + isi_shift) ** 2)
                / n_minus_1
            )
    return output


def sfa_peak(sfa_values):
    temp = sfa_values[~np.isnan(sfa_values)]
    bins = temp.size // 4 if temp.size > 4 else temp.size
    if bins > 4:
        b, bb = np.histogram(temp, bins=bins)
        peak = np.argmax(b)
        return np.mean(bb[peak : peak + 1])
    else:
        return np.mean(temp)


def sfa_divisor(bursts: list[np.ndarray]) -> np.ndarray[float]:
    """
    The idea for the function was initially inspired by a program called
    Easy Electropysiology (https://github.com/easy-electrophysiology).
    """
    output = np.zeros(len(bursts))
    for index, b in enumerate(bursts):
        iei = np.diff(b).astype(float)
        if len(iei) > 1 or np.isnan(iei):
            if iei[-1] > 0:
                output[index] = iei[0] / iei[-1]
            else:
                output[index] = np.nan
        else:
            output[index] = 0.0
    return output


def sfa_abi(bursts: list[np.ndarray]) -> np.ndarray[float]:
    """
    This function calculates the spike frequency adaptation. A positive
    number means that the spikes are speeding up and a negative number
    means that spikes are slowing down. This function was inspired by the
    Allen Brain Institutes IPFX analysis program
    https://github.com/AllenInstitute/ipfx/tree/
    db47e379f7f9bfac455cf2301def0319291ad361
    """
    output = np.zeros(len(bursts))
    for index, b in enumerate(bursts):
        if len(b) <= 1:
            output[index] = np.nan
        else:
            iei = np.diff(b).astype(float)
            if np.allclose((iei[1:] + iei[:-1]), 0.0):
                output[index] = 0.0
            else:
                norm_diffs = (iei[1:] - iei[:-1]) / (iei[1:] + iei[:-1])
                norm_diffs[(iei[1:] == 0) & (iei[:-1] == 0)] = 0.0
                output[index] = np.nanmean(norm_diffs)
    return output
