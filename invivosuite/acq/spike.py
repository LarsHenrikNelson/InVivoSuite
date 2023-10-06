import collections
import math

import fcwt
import numpy as np
from KDEpy import TreeKDE
from numba import njit
from scipy import signal
from sklearn.decomposition import PCA


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


@njit()
def create_binary_spikes(spikes, size):
    if len(spikes) > 0:
        binary_spikes = np.zeros(shape=(size,))
        binary_spikes[spikes] = 1
        return binary_spikes
    else:
        AttributeError("There are no spikes in the acquisition.")


@njit()
def bin_spikes(spikes, binary_size, nperseg, size):
    step_index = np.arange(0, binary_size, nperseg)
    binned_spikes = np.empty(shape=step_index.size)
    binary_spikes = create_binary_spikes(spikes, binary_size)
    for index, i in enumerate(step_index):
        if (i + nperseg) <= size:
            j = i + nperseg
        else:
            j = size
        binned_spikes[index] = np.sum(binary_spikes[i:j])
    return binned_spikes


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


# def compute_whitening_matrix(acqs):
#     acqs = acqs - np.mean(acqs, axis=1)
#     cov = (acqs.T @ acqs).data.shape[0]


if __name__ == "__main__":
    get_spike_freq()
    get_spike_indexes()
    get_spikes()
    spike_parameters()
    find_spikes()
    bin_spikes()
    create_binary_spikes()
    clean_spikes()
    max_int_bursts()
