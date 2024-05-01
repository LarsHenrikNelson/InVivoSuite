from typing import Literal

import numpy as np

__all__ = ["max_int_bursts", "get_burst_data"]


def ave_inter_burst_iei(bursts):
    if len(bursts) <= 1:
        return 0
    diff = []
    for i in range(1, len(bursts)):
        diff.append(bursts[i][0] - bursts[i - 1][-1])
    return np.mean(diff)


def ave_spikes_burst(bursts: list):
    if len(bursts) == 0:
        return 0
    ave = 0
    for i in bursts:
        ave += len(i)
    return ave / len(bursts)


def ave_intra_burst_iei(bursts: list) -> float:
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


def ave_burst_len(bursts):
    if len(bursts) == 0:
        return 0
    ave = 0
    for i in bursts:
        ave += i[-1] - i[0]
    return ave / len(bursts)


def get_burst_data(bursts):
    data_dict = {}
    data_dict["num_bursts"] = len(bursts)
    data_dict["ave_burst_len"] = ave_burst_len(bursts)
    data_dict["intra_burst_iei"] = ave_intra_burst_iei(bursts)
    data_dict["ave_spks_burst"] = ave_spikes_burst(bursts)
    data_dict["inter_burst_iei"] = ave_inter_burst_iei(bursts)
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
