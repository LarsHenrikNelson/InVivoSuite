import numpy as np

__all__ = ["max_int_bursts"]


def clean_max_int_bursts(spikes, bursts, max_int):
    cleaned_bursts = []
    i = 1
    while i < len(bursts):
        temp = []
        temp.extend(bursts[i - 1])
        while spikes[bursts[i][0]] - spikes[bursts[i - 1][-1]] < max_int:
            temp.extend(bursts[i])
            i += 1
        cleaned_bursts.append(np.array(temp))
        i += 1
    return cleaned_bursts


def max_int_bursts(
    spikes,
    fs,
    min_count=5,
    min_dur=0,
    max_start=None,
    max_end=None,
    max_int=None,
    output_type="index",
):
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
            bur.extend((i, i + 1))
            i += 1
            add_spikes = True
            while add_spikes and i < (spike_temp.size - 2):
                if (spike_temp[i + 1] - spike_temp[i]) <= max_end:
                    bur.append(i + 1)
                    i += 1
                else:
                    add_spikes = False
                    if (len(bur) >= min_count) and (
                        (spike_temp[bur[-1]] - spike_temp[bur[0]]) > min_dur
                    ):
                        bursts.append(np.array(bur))
        else:
            i += 1
    bursts = clean_max_int_bursts(spikes, bursts, max_int)
    if output_type == "time":
        bursts = [np.array([spikes[j] for j in i]) / fs for i in bursts]
    elif output_type == "sample":
        bursts = [np.array([spikes[j] for j in i]) for i in bursts]
    return bursts
