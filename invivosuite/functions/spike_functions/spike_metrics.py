import math
from typing import TypedDict, Union

import numpy as np
from numba import njit
from scipy import ndimage, stats

from .. import signal_functions

"""These are from 
https://github.com/AllenInstitute/ecephys_spike_sorting/blob/master/ecephys_spike_sorting/modules/quality_metrics/metrics.py
I will likely change the histogram based metrics to KDE for better accuracy.
"""

__all__ = [
    "amplitude_cutoff",
    "firing_rate",
    "isi_violations",
    "presence",
    "rb_violations",
]


class Presence(TypedDict):
    presence_ratio: float
    reg_slope: float
    reg_pvalue: float
    uniform_fit: bool


def presence(data: np.ndarray, start: int = -1, end: int = -1, tol=1e-9) -> Presence:
    """Modified version of AllenInstitutes spike presence. Creates a
    KDE of the spike indices then runs a regression to see if the
    data is skewed and a

    Args:
        data (np.ndarray): data in samples
        nbins (int): _description_
        start (int, optional): _description_. Defaults to -1.
        end (int, optional): _description_. Defaults to -1.
        tol (_type_, optional): _description_. Defaults to 1e-9.

    Returns:
        tuple[float, float, float, bool]: _description_
    """
    if start == -1:
        start = data[0]
    if end == -1:
        end == data[-1]
    nbins = math.ceil(data.size * 0.05)
    if nbins > 2:
        bins = np.linspace(start, end, num=nbins)
        binned = signal_functions.bin_data_sorted(data, bins)
        reg_out = stats.linregress(np.arange(binned.size), binned)
        # _, y = signal_functions.kde(data, tol=tol)z
        fit_out = stats.fit(stats.uniform, binned)
        output = Presence(
            presence_ratio=np.sum(binned > 0) / nbins,
            reg_slope=reg_out.slope,
            reg_pvalue=reg_out.pvalue,
            uniform_fit=fit_out.success,
        )
    else:
        output = Presence(
            presence_ratio=0.0, reg_slope=0.0, reg_pvalue=0.0, uniform_fit=False
        )
    return output


def firing_rate(spike_train, min_time=None, max_time=None):
    """Calculate firing rate for a spike train.

    If no temporal bounds are specified, the first and last spike time are used.

    Inputs:
    -------
    spike_train : numpy.ndarray
        Array of spike times in seconds
    min_time : float
        Time of first possible spike (optional)
    max_time : float
        Time of last possible spike (optional)

    Outputs:
    --------
    fr : float
        Firing rate in Hz

    """

    if min_time is not None and max_time is not None:
        duration = max_time - min_time
    else:
        duration = np.max(spike_train) - np.min(spike_train)

    fr = spike_train.size / duration

    return fr


def amplitude_cutoff(
    amplitudes: np.ndarray,
    num_histogram_bins: int = 500,
    histogram_smoothing_value: Union[float, int] = 3,
):
    """Calculate approximate fraction of spikes missing from a distribution of amplitudes

    Assumes the amplitude histogram is symmetric (not valid in the presence of drift)

    Inspired by metric described in Hill et al. (2011) J Neurosci 31: 8699-8705

    Input:
    ------
    amplitudes : numpy.ndarray
        Array of amplitudes (don't need to be in physical units)

    Output:
    -------
    fraction_missing : float
        Fraction of missing spikes (0-0.5)
        If more than 50% of spikes are missing, an accurate estimate isn't possible

    """

    h, b = np.histogram(amplitudes, num_histogram_bins, density=True)

    pdf = ndimage.gaussian_filter1d(h, histogram_smoothing_value)
    support = b[:-1]

    peak_index = np.argmax(pdf)
    G = np.argmin(np.abs(pdf[peak_index:] - pdf[0])) + peak_index

    bin_size = np.mean(np.diff(support))
    fraction_missing = np.sum(pdf[G:]) * bin_size

    fraction_missing = np.min([fraction_missing, 0.5])

    return fraction_missing


def isi_violations(spike_train, min_time, max_time, isi_threshold=1.5, min_isi=0):
    """Calculate ISI violations for a spike train.

    Based on metric described in Hill et al. (2011) J Neurosci 31: 8699-8705

    modified by Dan Denman from cortex-lab/sortingQuality GitHub by Nick Steinmetz

    Inputs:
    -------
    spike_train : array of spike times
    min_time : minimum time for potential spikes
    max_time : maximum time for potential spikes
    isi_threshold : threshold for isi violation
    min_isi : threshold for duplicate spikes

    Outputs:
    --------
    fpRate : rate of contaminating spikes as a fraction of overall rate
        A perfect unit has a fpRate = 0
        A unit with some contamination has a fpRate < 0.5
        A unit with lots of contamination has a fpRate > 1.0
    num_violations : total number of violations

    """

    duplicate_spikes = np.where(np.diff(spike_train) <= min_isi)[0]

    spike_train = np.delete(spike_train, duplicate_spikes + 1)
    isis = np.diff(spike_train)

    num_violations = sum(isis < isi_threshold)
    violation_time = 2 * spike_train.size * (isi_threshold - min_isi)
    total_rate = firing_rate(spike_train, min_time, max_time)
    violation_rate = num_violations / violation_time
    fpRate = violation_rate / total_rate

    return fpRate, num_violations


@njit(cache=True)
def rb_violations(spike_train, min_time, max_time, isi_threshold, min_isi):

    T = max_time - min_time
    rp_nv = 0
    N = len(spike_train)

    for i in range(N):
        for j in range(i + 1, N):
            diff = spike_train[j] - spike_train[i]

            if diff > isi_threshold:
                break

            rp_nv += 1

    D = 1 - rp_nv * (T - 2 * N * min_isi) / (N**2 * (isi_threshold - min_isi))
    rp_contamination = 1 - math.sqrt(D) if D >= 0 else 1.0

    return rp_contamination, rp_nv
