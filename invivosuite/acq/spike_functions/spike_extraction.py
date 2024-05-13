from typing import TypedDict, Union

import numpy as np
from numba import njit, prange
from scipy import signal
from scipy import interpolate

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
    hwL: float
    hwL_len: float
    hwR: float
    hwR_len: float
    full_width: Union[int, float]
    start: Union[int, float]
    end: Union[int, float]
    center: Union[int, float]
    center_y: float
    start_y: float
    end_y: float
    center_diff: float
    min_x: int
    min_y: float


@njit(cache=True)
def get_template_parts(template: np.ndarray) -> tuple[int, int, int]:
    peak_index = np.argmin(template)
    start_index = peak_index
    end_index = peak_index
    current = template[peak_index]
    previous = current - 1
    while previous < current and start_index > 0:
        previous = current
        start_index -= 1
        current = template[start_index]
    start_index += 1
    current = template[peak_index]
    previous = current - 1
    while previous < current and end_index < template.size - 1:
        previous = current
        end_index += 1
        current = template[end_index]
    end_index -= 1
    return start_index, peak_index, end_index


def simple_interpolation(template: np.ndarray, value: float, index: int):
    y1 = template[index]
    y0 = template[index - 1]
    y = value
    temp1 = np.abs((y1 - y0))
    temp2 = np.abs((y0 - y))
    out = (temp2 / temp1) if temp1 > temp2 else 0
    return out + index - 1


def template_properties(
    template: np.ndarray, center: int, upsample_factor: int = 2, negative: bool = True
):
    # if negative:
    #     multiplier = -1
    # else:
    #     multiplier = 1
    spl = interpolate.CubicSpline(
        np.linspace(0, template.size, num=template.size), template
    )
    yinterp = spl(np.linspace(0, template.size, num=template.size * upsample_factor))
    center *= upsample_factor

    yinterp_diff = np.diff(yinterp)
    min_diff = np.argmin(yinterp_diff)
    min_st = np.argmin(yinterp)
    min_x = np.argmin(yinterp)

    # Find the start and end
    i = signal.argrelmax(yinterp_diff[:center], order=1)[0]
    i = np.array([j for j in i if j < (center - (5 * upsample_factor))])
    i = i[-1] if i.size > 0 else 0
    j = signal.argrelmax(yinterp[:center], order=1)[0]
    j = np.array([m for m in j if m < (center - (5 * upsample_factor))])
    j = j[-1] if j.size > 0 else 0
    Lb = i if i > j else j
    k = np.argmax(yinterp[center:]) + center
    Rb = k if k > (center + (5 * upsample_factor)) else (yinterp.size - 1)

    if ((Rb - Lb) / upsample_factor) > (5 * upsample_factor):
        hwidth_L = yinterp[center] - ((yinterp[center] - yinterp[Lb]) / 2)
        bb = np.where(yinterp[Lb:Rb] < hwidth_L)[0] + Lb
        Lw1 = simple_interpolation(yinterp, hwidth_L, bb[0])
        Lw2 = simple_interpolation(yinterp, hwidth_L, bb[-1] + 1)

        hwidth_R = yinterp[center] - ((yinterp[center] - yinterp[Rb]) / 2)
        bb = np.where(yinterp[Lb:Rb] < hwidth_R)[0] + Lb
        Rw1 = simple_interpolation(yinterp, hwidth_R, bb[0])
        Rw2 = simple_interpolation(yinterp, hwidth_R, bb[-1] + 1)

    else:
        hwidth_L = np.nan
        hwidth_R = np.nan
        Lw1 = np.nan
        Lw2 = np.nan
        Rw1 = np.nan
        Rw2 = np.nan

    t_props = TemplateProperties(
        hwL_len=(Lw2 - Lw1) / upsample_factor,
        hwL=hwidth_L,
        hwR_len=(Rw2 - Rw1) / upsample_factor,
        hwR=hwidth_R,
        full_width=(Rb - Lb) / upsample_factor,
        end=Rb / upsample_factor,
        center=center / upsample_factor,
        start=Lb / upsample_factor,
        center_y=yinterp[center],
        start_y=yinterp[Lb],
        end_y=yinterp[Rb],
        center_diff=min_diff - min_st,
        min_x=(min_x / upsample_factor),
        min_y=yinterp[min_x],
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
