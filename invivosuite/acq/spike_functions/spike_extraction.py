from typing import TypedDict

import numpy as np
from numba import njit, prange
from scipy import signal

from ...utils import where_count


__all__ = [
    "center_spikes",
    "duplicate_spikes",
    "extract_multi_channel_spikes",
    "extract_single_channel_spikes",
    "find_minmax_exponent_3",
    "get_multichan_spikes",
    "get_template_parts",
    "template_properties",
]


class TemplateProperties(TypedDict):
    half_width: float
    half_width_zero: float
    start_to_peak: int
    peak_to_end: int
    trough: float
    peak_Left: float
    peak_Right: float


@njit(cache=True)
def get_template_parts(template: np.ndarray) -> tuple[int, int, int]:
    index = np.argmin(template)
    start_index = index
    end_index = index
    current = template[index]
    previous = current - 1
    while previous < current and start_index > 0:
        previous = current
        start_index -= 1
        current = template[start_index]
    start_index += 1
    current = template[index]
    previous = current - 1
    while previous < current and end_index < template.size - 1:
        previous = current
        end_index += 1
        current = template[end_index]
    end_index -= 1
    amplitude = template[index]
    trough_to_peak = template[end_index] - amplitude
    return start_index, end_index, index, amplitude, trough_to_peak


def template_properties(template: np.ndarray, negative: bool = True):
    if negative:
        multiplier = -1
    else:
        multiplier = 1
    peak_index = np.argmin(template)
    _, Lb, Rb = signal.peak_prominences(template * multiplier, [peak_index])
    widths, _, _, _ = signal.peak_widths(template * multiplier, [peak_index])
    tt = np.where(template > 0, 0, template)
    widths_zero, _, _, _ = signal.peak_widths(tt * multiplier, [peak_index])
    t_props = TemplateProperties(
        half_width=widths[0],
        half_width_zero=widths_zero,
        peak_to_end=Rb - peak_index,
        start_to_peak=peak_index - Lb,
        trough=template[peak_index],
        peak_Left=template[Lb],
        peak_Right=template[Rb],
    )
    return t_props


@njit(cache=True)
def find_minmax_exponent_3(spike_waveforms, min_val, max_val):
    # min_val = np.finfo("d").max
    # max_val = np.finfo("d").min
    for one in range(spike_waveforms.shape[0]):
        for two in range(spike_waveforms.shape[1]):
            for three in range(spike_waveforms.shape[2]):
                if spike_waveforms[one, two, three] != 0:
                    temp_value = np.log10(np.abs(spike_waveforms[one, two, three]))
                    if temp_value < min_val:
                        min_val = temp_value
                    if temp_value > max_val:
                        max_val = temp_value
    return np.floor(min_val), np.floor(max_val)


def duplicate_spikes(chan, cluster_id, acq_manager):
    indexes = acq_manager.get_cluster_spike_indexes(cluster_id)
    spk_acq = acq_manager.acq(
        chan,
        acq_type="spike",
        ref=True,
        ref_type="cmr",
        ref_probe="acc",
        map_channel=True,
        probe="acc",
    )
    nu = center_spikes(indexes, spk_acq)

    return not (np.unique(nu).size == indexes.size), np.unique(nu).size


def get_multichan_spikes(
    chan,
    acq_manager,
    phy_loader,
    size=45,
    nchans=8,
    ref=True,
    ref_type="cmr",
    ref_probe="acc",
    map_channel=True,
    probe="acc",
    center_spikes=False,
):
    total_spikes = 0
    for i in phy_loader.channel_clusters[chan]:
        total_spikes += where_count(i, phy_loader.spike_clusters)
    spk_chan = acq_manager.get_multichans(
        chan=chan,
        nchans=nchans,
        acq_type="spike",
        ref=ref,
        ref_type=ref_type,
        ref_probe=ref_probe,
        map_channel=map_channel,
        probe=probe,
    )
    if spk_chan.ndim == 1:
        outsize = size * 2 + 1
    else:
        outsize = spk_chan.shape[0] * (size * 2 + 1)
    spks = np.zeros((total_spikes, outsize))
    load_index = 0
    for clust in phy_loader.channel_clusters[chan]:
        spike_times = phy_loader.get_cluster_spike_indexes(clust)
        spks_temp = extract_multi_channel_spikes(spike_times, spk_chan, size=size)
        if center_spikes:
            spike_times = center_spikes(spike_times, spk_chan, size=size)
        end_index = load_index + spks_temp.shape[0]
        spks[load_index:end_index, :] = spks_temp
        load_index += spks_temp.shape[0]
    return spks


@njit(parallel=True, cache=True)
def center_spikes(indexes, acq, size=45):
    m = np.zeros(indexes.size, dtype=np.int64)
    for i in range(indexes.size):
        start = int(indexes[i] - size)
        end = int(indexes[i] + size + 1)
        if start < 0:
            start = 0
        if end < acq.size:
            b = acq[start:end]
            max_vel = np.argmin(b[1:] - b[:-1])
            d = np.argmin(b[max_vel : max_vel + 10])
            d += start + max_vel
            m[i] = d
        else:
            m[i] = indexes[i]
    return m


@njit(parallel=True, cache=True)
def extract_single_channel_spikes(
    indexes: np.ndarray, acq: np.ndarray, size: int = 45, center: bool = False
):
    if center:
        indexes = center_spikes(indexes, acq, size=45)
    m = np.zeros((indexes.size, size * 2 + 1))
    for i in prange(indexes.size):
        start = int(indexes[i] - size)
        end = int(indexes[i] + size + 1)
        if start < 0:
            start = 0
        if acq.size < end:
            end = acq.size
        b = acq[start:end]
        m[i, : b.size] = b
    return m


@njit(parallel=True, cache=True)
def extract_multi_channel_spikes(indexes, acqs, size=45):
    if acqs.ndim == 1:
        outsize = size * 2 + 1
        acq_n = acqs.size
    else:
        outsize = acqs.shape[0] * (size * 2 + 1)
        acq_n = acqs.shape[1]
    m = np.zeros((indexes.size, outsize))
    for i in prange(indexes.size):
        start = int(indexes[i] - size)
        end = int(indexes[i] + size + 1)
        if start < 0:
            start = 0
        if acq_n < end:
            end = acq_n
        b = acqs[:, start:end]
        m[i, :] = b.flatten()
    return m
