import collections
import math
from typing import Union

import fcwt
import numpy as np
from KDEpy import TreeKDE
from numba import njit, prange
from scipy import fft, signal
from sklearn.decomposition import PCA

from ..utils import where_count

__all__ = [
    "bin_spikes",
    "center_spikes",
    "create_binary_spikes",
    "extract_spikes_multi_channel",
    "extract_spikes_single_channel",
    "find_bursts",
    "get_burst_data",
    "max_int_bursts",
]


def get_channel_spikes(
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

    if chan != 64:
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
    else:
        return None
    if spk_chan.ndim == 1:
        outsize = size * 2 + 1
    else:
        outsize = spk_chan.shape[0] * (size * 2 + 1)
    spks = np.zeros((total_spikes, outsize))
    load_index = 0
    for clust in phy_loader.channel_clusters[chan]:
        spike_times = phy_loader.get_cluster_spike_indexes(clust)
        spks_temp = extract_spikes_multi_channel(spike_times, spk_chan, size=size)
        if center_spikes:
            spike_times = center_spikes(spike_times, spk_chan, size=size)
        end_index = load_index + spks_temp.shape[0]
        spks[load_index:end_index, :] = spks_temp
        load_index += spks_temp.shape[0]
    return spks


@njit(parallel=True, cache=True)
def center_spikes(indexes, acq_array, size=45):
    m = np.zeros(indexes.size, dtype=np.int64)
    for i in prange(indexes.size):
        start = int(indexes[i] - size)
        end = int(indexes[i] + size + 1)
        if start < 0:
            start = 0
        if end < indexes.size:
            b = acq_array[start:end]
            max_vel = np.argmin(np.diff(b))
            d = np.argmin(b[max_vel : max_vel + 10])
            d += start + max_vel
            m[i] = d
        else:
            m[i] = indexes[i]
    return m


@njit(parallel=True, cache=True)
def extract_spikes_single_channel(indexes, acq, size=45):
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
def extract_spikes_multi_channel(indexes, acqs, size=45):
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


@njit()
def run_P(
    spk_times_1: np.ndarray, spk_times_2: np.ndarray, dt: Union[int, float]
) -> int:
    Nab = 0
    j = 0
    k = 0
    add_spk = 0
    for i in range(0, spk_times_1.size):
        while j < spk_times_2.size:
            # Need this for unsigned ints to work with numba
            if spk_times_1[i] < spk_times_2[j]:
                temp = spk_times_2[j] - spk_times_1[i]
                add_spk = 1
            else:
                temp = spk_times_1[i] - spk_times_2[j]
                add_spk = 0
            if np.abs(temp) <= dt:
                Nab = Nab + 1
                k += add_spk
                break
            elif spk_times_2[j] > spk_times_1[i]:
                break
            else:
                j += 1
    return Nab, k


@njit()
def run_T(
    spk_train: np.ndarray,
    dt: Union[float, int],
    start: Union[float, int],
    stop: Union[float, int],
) -> float:
    i = 0
    time_A = 2 * spk_train.size * dt
    if spk_train.size == 1:
        if spk_train[0] - start < dt:
            time_A = time_A - start + spk_train[0] - dt
        elif (spk_train[0] + dt) > stop:
            time_A = time_A - spk_train[0] - dt + stop
    else:
        for i in range(0, spk_train.size - 1):
            diff = spk_train[i + 1] - spk_train[i]
            if diff < (2 * dt):
                time_A = time_A - 2 * dt + diff
        if (spk_train[0] - start) < dt:
            time_A = time_A - start + spk_train[0] - dt
        if (stop - spk_train[-1]) < dt:
            time_A = time_A - spk_train[-1] - dt + stop
    return time_A


@njit()
def sttc(
    spk_times_1: np.ndarray,
    spk_times_2: np.ndarray,
    dt: Union[float, int],
    start: Union[float, int],
    stop: Union[float, int],
) -> tuple[float, int, int, int, int]:
    """This is a Numba accelerated version of spike timing tiling coefficient.
    It is faster than Elephants version by about 50 times. This adds up when
    there are 10000+ comparisons to make. This function can run using unsigned ints
    which is the most numerically precise.

    Args:
        spk_times_1 (np.ndarray): np.array of spike times
        spk_times_2 (np.ndarray): np.array of spike times
        dt (int, float): largest time difference at which two spikes can be considered
        to co-occur
        start (int, float): start of time to assess
        stop (int, float): stop of time to assess, usually the length of the recording

    Returns:
        float: _description_
        int: number of spikes in spk_times_1 that occur within dt of spikes spk_times_2
        int: number of spikes in spk_times_1 that occur before spikes spk_times_2
        int: number of spikes in spk_times_2 that occur within dt of spikes spk_times_1
        int: number of spikes in spk_times_2 that occur before spikes spk_times_1
    """
    if spk_times_1.size == 0 or spk_times_2.size == 0:
        return 0.0, 0, 0, 0, 0
    else:
        dt = float(dt)
        t = stop - start
        tA = run_T(spk_times_1, dt, start, stop)
        tA /= t
        tB = run_T(spk_times_2, dt, start, stop)
        tB /= t
        pA_a, kA = run_P(spk_times_1, spk_times_2, dt)
        pA = pA_a / spk_times_1.size
        pB_b, kB = run_P(spk_times_2, spk_times_1, dt)
        pB = pB_b / spk_times_2.size
        if pA * tB == 1 and pB * tA == 1:
            index = 1.0
        elif pA * tB == 1:
            index = 0.5 + 0.5 * (pB - tA) / (1 - pB * tA)
        elif pB * tA == 1:
            index = 0.5 + 0.5 * (pA - tB) / (1 - pA * tB)
        else:
            index = (0.5 * ((pA - tB) / (1 - pA * tB))) + (
                0.5 * ((pB - tA) / (1 - pB * tA))
            )
        return index, pA_a, kA, pB_b, kB


def run_p(
    spiketrain_j: np.ndarray,
    spiketrain_i: np.ndarray,
    dt: Union[int, float],
) -> float:
    # Create a boolean array where each element represents whether a spike
    # in spiketrain_j lies within +- dt of any spike in spiketrain_i.
    tiled_spikes_j = np.isclose(
        spiketrain_j[:, np.newaxis],
        spiketrain_i,
        atol=dt,
        rtol=0,
    )
    # Determine which spikes in spiketrain_j satisfy the time window
    # condition.
    tiled_spike_indices = np.any(tiled_spikes_j, axis=1)
    # Extract the spike times in spiketrain_j that satisfy the condition.
    tiled_spikes_j = spiketrain_j[tiled_spike_indices]
    # Calculate the ratio of matching spikes in j to the total spikes in j.
    return len(tiled_spikes_j) / len(spiketrain_j)


def run_t(
    spiketrain: np.ndarray,
    dt: Union[int, float],
    t_start: Union[int, float],
    t_stop: Union[int, float],
) -> float:
    dt = dt
    sorted_spikes = spiketrain

    diff_spikes = np.diff(sorted_spikes)

    overlap_durations = diff_spikes[diff_spikes <= 2 * dt]
    covered_time_overlap = np.sum(overlap_durations)

    non_overlap_durations = diff_spikes[diff_spikes > 2 * dt]
    covered_time_non_overlap = len(non_overlap_durations) * 2 * dt

    if sorted_spikes[0] - t_start < dt:
        covered_time_overlap += sorted_spikes[0] - t_start
    else:
        covered_time_non_overlap += dt
    if t_stop - sorted_spikes[-1] < dt:
        covered_time_overlap += t_stop - sorted_spikes[-1]
    else:
        covered_time_non_overlap += dt

    total_time_covered = covered_time_overlap + covered_time_non_overlap
    total_time = t_stop - t_start

    return total_time_covered / total_time


def sttc_ele(spiketrain_i, spiketrain_j, dt, start, stop):
    if len(spiketrain_i) == 0 or len(spiketrain_j) == 0:
        index = np.nan
    else:
        TA = run_t(spiketrain_j, dt, start, stop)
        TB = run_t(spiketrain_i, dt, start, stop)
        PA = run_p(spiketrain_j, spiketrain_i, dt)
        PB = run_p(spiketrain_i, spiketrain_j, dt)

        # check if the P and T values are 1 to avoid division by zero
        # This only happens for TA = PB = 1 and/or TB = PA = 1,
        # which leads to 0/0 in the calculation of the index.
        # In those cases, every spike in the train with P = 1
        # is within dt of a spike in the other train,
        # so we set the respective (partial) index to 1.
        if PA * TB == 1 and PB * TA == 1:
            index = 1.0
        elif PA * TB == 1:
            index = 0.5 + 0.5 * (PB - TA) / (1 - PB * TA)
        elif PB * TA == 1:
            index = 0.5 + 0.5 * (PA - TB) / (1 - PA * TB)
        else:
            index = 0.5 * (PA - TB) / (1 - PA * TB) + 0.5 * (PB - TA) / (1 - PB * TA)
    return index


def find_spikes(
    array: np.ndarray,
    spike_start: int,
    spike_end: int,
    n_threshold: float = 5.0,
    p_threshold: float = 0.0,
    tau: float = 15,
    method: str = "std",
):
    if method == "mad":
        med = np.median(-array)
        mad = np.median(np.abs(-array - med))
        spike_temp, _ = signal.find_peaks(
            -array, height=med + p_threshold * mad, distance=15
        )
    else:
        spike_temp, _ = signal.find_peaks(
            -array, height=np.mean(-array) + n_threshold * np.std(array), distance=15
        )
    spikes = clean_spikes(array, spike_temp, spike_start, spike_end, p_threshold)
    return spikes


@njit()
def clean_spikes(array, spikes, spike_start, spike_end, p_threshold):
    clean_spikes = []
    for i in spikes:
        if i > spike_start and i + spike_end <= array.size:
            if (
                np.max(array[i - spike_start : i + spike_end]) > p_threshold
                and np.argmin(array[i - spike_start : i + spike_end]) == spike_start
            ):
                clean_spikes.append(i)
    return np.array(clean_spikes)


def create_binary_spikes(spikes, size):
    if len(spikes) > 0:
        binary_spikes = np.zeros(shape=(size,))
        binary_spikes[spikes] = 1
        return binary_spikes
    else:
        AttributeError("There are no spikes in the acquisition.")


def _bin_spikes(binary_spks, bin_size):
    noverlap = 0
    nperseg = bin_size
    step = nperseg - noverlap
    shape = binary_spks.shape[:-1] + (
        (binary_spks.shape[-1] - noverlap) // step,
        nperseg,
    )
    strides = binary_spks.strides[:-1] + (
        step * binary_spks.strides[-1],
        binary_spks.strides[-1],
    )
    temp = np.lib.stride_tricks.as_strided(binary_spks, shape=shape, strides=strides)
    output = temp.sum(axis=1)
    return output


def bin_spikes(spikes, binary_size, nperseg):
    binary_spikes = create_binary_spikes(spikes, binary_size)
    binned_spikes = _bin_spikes(binary_spikes, nperseg)
    return binned_spikes


def is_ngb(xcorr_array, fs):
    out_fft = fft.rfft(xcorr_array)
    val = 1.0 / (xcorr_array.size * 1 / fs)
    N = xcorr_array.size // 2 + 1
    freqs = np.arange(0, N, dtype=int) * val
    ngb_indexes = np.where((freqs >= 50) & (freqs <= 70))[0]
    wb_indexes = np.where((freqs >= 40) & (freqs <= 300))[0]
    if np.max(out_fft[ngb_indexes]) > np.max(out_fft[wb_indexes]):
        return True
    else:
        return False


SpkParams = collections.namedtuple("SpkParams", ["peak", "lmi", "rmi", "lm", "rm"])


@njit()
def spike_parameters(spikes):
    data = np.empty((len(spikes), 10))
    for index, i in enumerate(spikes):
        j = np.argmin(i)
        lmi = np.argmax(i[:j])
        rmi = np.argmax(i[j:]) + j
        z1 = np.where(i[:j] < 0)[0][0]
        z2 = np.where(i[j:] < 0)[0][-1] + j
        data[index, 0] = i[j]
        data[index, 1] = lmi
        data[index, 2] = rmi
        data[index, 3] = i[lmi]
        data[index, 4] = i[rmi]
        data[index, 5] = rmi - j
        data[index, 6] = i[lmi] - i[j]
        data[index, 7] = z1
        data[index, 8] = z2
        data[index, 9] = i[z2] - i[z1]
        labels = (
            "m",
            "lmi",
            "rmi",
            "lm",
            "rm",
            "lmi_to_mi",
            "amp",
            "lz",
            "rz",
            "width",
        )
    return data, labels


@njit()
def get_spikes(array, spikes, spike_start=50, spike_end=50):
    spike_array = np.zeros((len(spikes), spike_start + spike_end))
    for index, i in enumerate(spikes):
        spike_array[index] = array[i - spike_start : i + spike_end]
    return spike_array


def get_all_spikes(acq_list: list, spike_start=50, spike_end=50):
    spikes = []
    for i in acq_list:
        spikes.append(i.get_spikes(spike_start, spike_end))
    return spikes


def get_spike_indexes(acq_list: list):
    spk_list = []
    for i in acq_list:
        i.load_hdf5_acq()
        spk_list.append(i.file["spikes"][()])
    return spk_list


def get_spike_freq(acq_list: list):
    spk_freq = []
    for i in acq_list:
        i.load_hdf5_acq()
        spk_freq.append(i.file.attrs["spike_freq"])
    return spk_freq


@njit()
def find_bursts(spikes, freq):
    bursts = []
    mean_isi = 1 / freq
    isi_to_start = mean_isi / 2
    isi_to_end = mean_isi
    i = 0
    while i < spikes.size - 2:
        if (
            spikes[i + 1] - spikes[i] < isi_to_start
            and spikes[i + 2] - spikes[i + 1] < isi_to_start
        ):
            bur = []
            bur.extend((i, i + 1, i + 2))
            i += 2
            add_spikes = True
            while add_spikes:
                if spikes[i + 1] - spikes[i] <= isi_to_end:
                    bur.append(spikes[i + 1])
                    i += 1
                else:
                    add_spikes = False
    return bursts


@njit()
def find_fwd_burst(freq, burst: np.ndarray):
    snurprise = []
    for i in range(3, len(burst)):
        p = poisson_surp(freq, burst[:i])
        s = -np.log10(p)
        snurprise.append(s)
    b = np.argmax(snurprise)
    return burst[: 3 + b]


@njit()
def find_bwk_burst(freq, burst: np.ndarray):
    snurprise = []
    for i in range(3, len(burst)):
        p = poisson_surp(freq, burst[:i])
        s = -np.log10(p)
        snurprise.append(s)
    b = np.argmax(snurprise)
    return burst[: 3 + b]


@njit()
def poisson_surp(r, burst):
    rT = r * (burst[-1] - burst[0])
    e_rT = np.exp(-rT)
    temp = ((rT) ** len(burst)) / math.factorial(len(burst))
    p = -np.log10(e_rT + temp)
    return p


def CMA(dist):
    cma = np.empty(dist.size)
    for index in range(1, len(dist)):
        cma[index] = np.mean(dist[:index]) / (dist[:index].size)
    return cma


def find_isi_max(spikes):
    x, y = (
        TreeKDE(kernel="gaussian", bw="ISJ")
        .fit(np.diff(spikes))
        .evaluate(np.diff(spikes).size)
    )
    cma = CMA(y)
    max_index = np.argmax(cma)
    max_isi = x[max_index]
    return max_isi


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
        return 0
    ave = 0
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


def get_spike_cwt(spikes, fs=40000, f0=300, f1=1500, fn=100, bandwidth=2.0):
    morl = fcwt.Morlet(bandwidth)
    scales = fcwt.Scales(morl, fcwt.FCWT_LOGSCALES, fs, f0, f1, fn)
    cwt_obj = fcwt.FCWT(morl, 2, False, True)
    data = np.zeros((spikes.shape))
    output = np.zeros((fn, spikes[0].size), dtype=np.complex64)
    for index, i in enumerate(spikes):
        if i.dtype != "single":
            i = i.astype("single")
        cwt_obj.cwt(i, scales, output)
        c = PCA(n_components=1).fit_transform(np.abs(output.T))
        data[index] = c.T
    return data
