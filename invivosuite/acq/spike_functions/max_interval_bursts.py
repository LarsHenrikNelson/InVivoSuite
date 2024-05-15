from typing import Literal, TypedDict

import numpy as np

__all__ = ["max_int_bursts", "get_burst_data"]


def sfa_local_var(iei):
    """
    This function calculates the local variance in spike frequency
    accomadation that was drawn from the paper:
    Shinomoto, Shima and Tanji. (2003). Differences in Spiking Patterns
    Among Cortical Neurons. Neural Computation, 15, 2823-2842.

    Returns
    -------
    None.

    """
    if len(iei) < 2 or iei is np.nan:
        local_var = np.nan
    else:
        isi_shift = iei[1:]
        isi_cut = iei[:-1]
        n_minus_1 = len(isi_cut)
        local_var = (
            np.sum((3 * (isi_cut - isi_shift) ** 2) / (isi_cut + isi_shift) ** 2)
            / n_minus_1
        )
    return local_var


def sfa_divisor(iei):
    """
    The idea for the function was initially inspired by a program called
    Easy Electropysiology (https://github.com/easy-electrophysiology).
    """
    if len(iei) > 1 or iei is np.nan:
        sfa_divisor = iei[0] / iei[-1]
    else:
        sfa_divisor = np.nan
    return sfa_divisor


def sfa_abi(iei):
    """
    This function calculates the spike frequency adaptation. A positive
    number means that the spikes are speeding up and a negative number
    means that spikes are slowing down. This function was inspired by the
    Allen Brain Institutes IPFX analysis program
    https://github.com/AllenInstitute/ipfx/tree/
    db47e379f7f9bfac455cf2301def0319291ad361
    """
    if len(iei) <= 1:
        spike_adapt = np.nan
    else:
        # iei = iei.astype(float)
        if np.allclose((iei[1:] + iei[:-1]), 0.0):
            spike_adapt = np.nan
        norm_diffs = (iei[1:] - iei[:-1]) / (iei[1:] + iei[:-1])
        norm_diffs[(iei[1:] == 0) & (iei[:-1] == 0)] = 0.0
        spike_adapt = np.nanmean(norm_diffs)
    return spike_adapt


def ave_inter_burst_iei(bursts):
    if len(bursts) <= 1:
        return 0
    diff = []
    for i in range(1, len(bursts)):
        diff.append(bursts[i][0] - bursts[i - 1][-1])
    return np.mean(diff)


def ave_spikes_burst(bursts: list[np.ndarray]):
    if len(bursts) == 0:
        return 0
    ave = 0
    for i in bursts:
        ave += len(i)
    return ave / len(bursts)


def ave_intra_burst_iei(bursts: list[np.ndarray]) -> float:
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
        return 0.0
    ave = 0.0
    for i in bursts:
        ave += np.mean(np.diff(i))
    return ave / len(bursts)


def ave_burst_len(bursts: list[np.ndarray]) -> float:
    if len(bursts) == 0:
        return 0
    ave = 0
    for i in bursts:
        ave += i[-1] - i[0]
    return ave / len(bursts)


class BurstProps(TypedDict):
    num_bursts: int
    ave_burst_len: float
    intra_burst_iei: float
    inter_burst_iei: float
    ave_spikes_burst: float
    local_sfa: float
    divisor_sfa: float
    abi_sfa: float


def get_burst_data(bursts: list[np.ndarray]) -> BurstProps:
    data_dict = BurstProps(
        num_bursts=len(bursts),
        ave_burst_len=ave_burst_len(bursts),
        intra_burst_iei=ave_intra_burst_iei(bursts),
        ave_spikes_burst=ave_spikes_burst(bursts),
        inter_burst_iei=ave_inter_burst_iei(bursts),
        local_sfa=np.mean([sfa_local_var(np.diff(i)) for i in bursts]),
        divisor_sfa=np.mean([sfa_divisor(np.diff(i)) for i in bursts]),
        abi_sfa=np.mean([sfa_abi(np.diff(i)) for i in bursts]),
    )
    return data_dict


def clean_max_int_bursts(bursts, max_int):
    cleaned_bursts = []
    i = 1
    if len(bursts) > 1:
        while i < len(bursts):
            temp = []
            temp.extend(bursts[i - 1])
            while (bursts[i][0] - bursts[i - 1][-1]) < max_int:
                temp.extend(bursts[i])
                i += 1
            cleaned_bursts.append(np.array(temp))
            i += 1
    else:
        cleaned_bursts = bursts
    return cleaned_bursts


def max_int_bursts(
    spikes,
    fs,
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
                    if len(bur) >= min_count and (bur[-1] - bur[0]) > min_dur:
                        bursts.append(np.array(bur))
        else:
            i += 1
    bursts = clean_max_int_bursts(bursts, max_int)
    if output_type == "ms":
        bursts = [(i / fs) * 1000 for i in bursts]
    elif output_type == "sample":
        bursts = [(i * fs).astype(int) for i in bursts]
    return bursts
