import numpy as np
from scipy import stats

from .. import utils

"""These are from 
https://github.com/AllenInstitute/ecephys_spike_sorting/blob/master/ecephys_spike_sorting/modules/quality_metrics/metrics.py
I will likely change the histogram based metrics to KDE for better accuracy.
"""

__all__ = [
    "amplitude_cutoff",
    "firing_rate",
    "isi_violations",
    "presence",
]


def presence(
    data: np.ndarray, nbins: int, start: int = -1, stop: int = -1, tol=1e-9
) -> tuple[float, float, float, bool]:
    """Modified version of AllenInstitutes spike presence. Creates a
    KDE of the spike indices then runs a regression to see if the
    data is skewed and a

    Parameters
    ----------
    data : np.ndarray
        _description_
    nbins : int
        _description_
    start : int, optional
        _description_, by default -1
    stop : int, optional
        _description_, by default -1
    tol : float, optional
        _description_, by default 1e-9

    Returns
    -------
    tuple[float, float, float, bool]
        _description_
    """
    if start == -1:
        start = data[0]
    if stop == -1:
        stop == data[-1]
    bins = np.linspace(start, stop, num=nbins)
    binned = utils.bin_data_sorted(data, bins)
    x, y = utils.kde(data, tol=1e-9)
    fit_out = stats.fit(stats.uniform, y)
    reg_out = stats.linregress(np.arange(binned.size), binned)
    return np.sum(binned > 0) / nbins, reg_out.slope, reg_out.pvalue, fit_out.success


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
    kernel: str = "biweight",
    bw_method: str = "ISJ",
    tol: float = 0.001,
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

    h, b = np.histogram(amplitudes, tol, density=True)

    # pdf = ndimage.gaussian_filter1d(h, histogram_smoothing_value)
    support = b[:-1]
    x, pdf = utils.kde(amplitudes, kernel, bw_method, tol)

    peak_index = np.argmax(pdf)
    G = np.argmin(np.abs(pdf[peak_index:] - pdf[0])) + peak_index

    bin_size = np.mean(np.diff(support))
    fraction_missing = np.sum(pdf[G:]) * bin_size

    fraction_missing = np.min([fraction_missing, 0.5])

    return fraction_missing


def isi_violations(spike_train, min_time, max_time, isi_threshold, min_isi=0):
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

    num_spikes = len(spike_train)
    num_violations = sum(isis < isi_threshold)
    violation_time = 2 * num_spikes * (isi_threshold - min_isi)
    total_rate = firing_rate(spike_train, min_time, max_time)
    violation_rate = num_violations / violation_time
    fpRate = violation_rate / total_rate

    return fpRate, num_violations
