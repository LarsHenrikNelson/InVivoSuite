from typing import Literal, TypedDict, Union

import numpy as np

__all__ = ["max_int_bursts", "get_burst_data"]


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
        return 0
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
        return np.array([0.0])
    output = np.zeros(len(bursts))
    for index, b in enumerate(bursts):
        output[index] = np.mean(np.diff(b))
    return output


def bursts_len(bursts: list[np.ndarray]) -> np.ndarray[float]:
    if len(bursts) == 0:
        return np.array([0.0])
    output = np.zeros(len(bursts))
    for index, b in enumerate(bursts):
        output[index] = b[-1] - b[0]
    return output


class BurstProps(TypedDict):
    burst_len: np.ndarray[float]
    intra_burst_iei: np.ndarray[float]
    inter_burst_iei: np.ndarray[float]
    spikes_per_burst: np.ndarray[float]
    local_sfa: np.ndarray[float]
    divisor_sfa: np.ndarray[float]
    abi_sfa: np.ndarray[float]


class BurstPropsMeans(TypedDict):
    num_bursts: int
    ave_burst_len: float
    intra_burst_iei: float
    inter_burst_iei: float
    ave_spikes_per_burst: float
    ave_local_sfa: float
    peak_local_sfa: float
    ave_divisor_sfa: float
    peak_divisor_sfa: float
    ave_abi_sfa: float
    peak_abi_sfa: float
    total_burst_time: float


def get_burst_data(bursts: list[np.ndarray]) -> tuple[BurstProps, BurstPropsMeans]:
    intra_iei = intra_burst_iei(bursts)
    spk_per_burst = spikes_per_burst(bursts)
    inter_iei = inter_burst_iei(bursts)
    local_sfa = sfa_local_var(bursts)
    divisor_sfa = sfa_divisor(bursts)
    abi_sfa = sfa_abi(bursts)
    b_len = bursts_len(bursts)
    props_dict = BurstProps(
        ave_burst_len=b_len,
        intra_burst_iei=intra_iei,
        spikes_per_burst=spk_per_burst,
        inter_burst_iei=inter_iei,
        local_sfa=local_sfa,
        divisor_sfa=divisor_sfa,
        abi_sfa=abi_sfa,
    )
    mean_dict = BurstPropsMeans(
        num_bursts=len(bursts),
        total_burst_time=np.nansum(b_len),
        ave_burst_len=np.nanmean(b_len),
        intra_burst_iei=np.nanmean(intra_iei),
        ave_spikes_per_burst=np.nanmean(spk_per_burst),
        inter_burst_iei=np.nanmean(inter_iei),
        ave_local_sfa=np.nanmean(local_sfa),
        ave_divisor_sfa=np.nanmean(divisor_sfa),
        ave_abi_sfa=np.nanmean(abi_sfa),
        peak_local_sfa=sfa_peak(local_sfa),
        peak_divisor_sfa=sfa_peak(divisor_sfa),
        peak_abi_sfa=sfa_peak(abi_sfa),
    )
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
    while i < spike_temp.size - min_count:
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
