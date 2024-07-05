from typing import Literal, Union

import numpy as np

from .spike_freq_adapt import (
    sfa_abi,
    sfa_divisor,
    sfa_local_var,
    sfa_rlocal_var,
)

__all__ = ["max_int_bursts", "get_burst_data"]


def _sfa_local_var(bursts: list[np.ndarray]) -> np.ndarray[float]:
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
        output[index] = sfa_local_var(b)
    return output


def _sfa_revised_local(bursts: list[np.ndarray], R: Union[float, int]):
    """
    This function calculates the revised local variance in spike frequency
    accomadation that was drawn from the paper:
    Shinomoto, S. et al. Relating Neuronal Firing Patterns to Functional Differentiation
    of Cerebral Cortex. PLoS Comput Biol 5, e1000433 (2009).


    Returns
    -------
    None.

    """
    output = np.zeros(len(bursts))
    for index, b in enumerate(bursts):
        output[index] = sfa_rlocal_var(b, R)
    return output


def _sfa_divisor(bursts: list[np.ndarray]) -> np.ndarray[float]:
    """
    The idea for the function was initially inspired by a program called
    Easy Electropysiology (https://github.com/easy-electrophysiology).
    """
    output = np.zeros(len(bursts))
    for index, b in enumerate(bursts):
        output[index] = sfa_divisor(b)
    return output


def _sfa_abi(bursts: list[np.ndarray]) -> np.ndarray[float]:
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
        output[index] = sfa_abi(b)
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


def inter_burst_iei(bursts: list[np.ndarray]):
    if len(bursts) <= 1:
        return np.array([np.nan])
    diff = np.zeros(len(bursts))
    diff[-1] = np.nan
    for i in range(1, len(bursts)):
        diff[i - 1] = bursts[i][0] - bursts[i - 1][-1]
    return diff


def spikes_per_burst(bursts: list[np.ndarray]) -> np.ndarray[float]:
    if len(bursts) == 0:
        return np.array([])
    output = np.zeros(len(bursts))
    for index, b in enumerate(bursts):
        output[index] = len(b)
    return output


def intra_burst_iei(bursts: list[np.ndarray]) -> np.ndarray[float]:
    """Get the average iei from each burst. Does not correct for sampling rate.

    Parameters
    ----------
    bursts : list-like
        A list of bursts

    Returns
    -------
    float
        The average iei of all the bursts
    """
    if len(bursts) == 0:
        return np.array([])
    output = np.zeros(len(bursts))
    for index, b in enumerate(bursts):
        output[index] = np.mean(np.diff(b))
    return output


def bursts_len(bursts: list[np.ndarray]) -> np.ndarray[float]:
    if len(bursts) == 0:
        return np.array([])
    output = np.zeros(len(bursts))
    for index, b in enumerate(bursts):
        output[index] = b[-1] - b[0]
    return output


def get_burst_data(bursts: list[np.ndarray], R: float) -> tuple[dict, dict]:
    iei_bursts = [np.diff(i) for i in bursts]
    props_dict = {}
    props_dict["intra_burst_iei"] = intra_burst_iei(bursts)
    props_dict["spikes_per_burst"] = spikes_per_burst(bursts)
    props_dict["local_sfa"] = _sfa_local_var(iei_bursts)
    props_dict["rlocal_sfa"] = _sfa_revised_local(iei_bursts, R)
    props_dict["divisor_sfa"] = _sfa_divisor(iei_bursts)
    props_dict["abi_sfa"] = _sfa_abi(iei_bursts)
    props_dict["burst_len"] = bursts_len(bursts)

    mean_dict = {}
    mean_dict["num_bursts"] = len(bursts)
    mean_dict["total_burst_time"] = np.nansum(props_dict["burst_len"])
    mean_dict["inter_burst_iei"] = inter_burst_iei(bursts)
    for key, value in props_dict.items():
        mean_dict[key] = np.nanmean(value)
        if "sfa" in key:
            mean_dict[f"peak_{key}"] = sfa_peak(value)
    return props_dict, mean_dict


def clean_max_int_bursts(bursts: list[np.ndarray], max_int: float):
    cleaned_bursts = []
    i = 1
    if len(bursts) > 1:
        while i < len(bursts):
            temp = []
            temp.extend(bursts[i - 1])
            while i < len(bursts) and (bursts[i][0] - bursts[i - 1][-1]) < max_int:
                temp.extend(bursts[i])
                i += 1
            cleaned_bursts.append(np.array(temp))
            i += 1
    else:
        cleaned_bursts = bursts
    return cleaned_bursts


def max_int_bursts(
    spikes: np.ndarray,
    fs: Union[float, int],
    min_count=5,
    min_dur=0,
    max_start=None,
    max_end=None,
    max_int=None,
    output_type: Literal["sec", "ms", "sample"] = "sec",
):
    if len(spikes) < min_count:
        return []
    bursts = []
    spike_temp = spikes / fs
    freq = 1 / np.mean(np.diff(spike_temp))
    if max_start is None:
        max_start = 1 / freq / 2
    if max_end is None:
        max_end = 1 / freq
    if max_int is None:
        max_int = max_end
    i = 0
    while i < (spike_temp.size - min_count):
        if (spike_temp[i + 1] - spike_temp[i]) < max_start:
            bur = []
            bur.extend((spike_temp[i], spike_temp[i + 1]))
            i += 1
            add_spikes = True
            while add_spikes and i < (spike_temp.size - 2):
                if (spike_temp[i + 1] - spike_temp[i]) <= max_end:
                    bur.append(spike_temp[i + 1])
                    i += 1
                else:
                    add_spikes = False
                    if (len(bur) >= min_count) and ((bur[-1] - bur[0]) > min_dur):
                        bursts.append(np.array(bur))
        else:
            i += 1
    bursts = clean_max_int_bursts(bursts, max_int)
    if output_type == "ms":
        bursts = [(i / fs) * 1000 for i in bursts]
    elif output_type == "sample":
        bursts = [(i * fs).astype(int) for i in bursts]
    return bursts
