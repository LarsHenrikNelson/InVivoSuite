import numpy as np
from scipy import ndimage

"""These are from 
https://github.com/AllenInstitute/ecephys_spike_sorting/blob/master/ecephys_spike_sorting/modules/quality_metrics/metrics.py
I will likely change the histogram based metrics to KDE for better accuracy.
"""


def presence_ratio(spike_train, min_time, max_time, num_bins=100):
    """Calculate fraction of time the unit is present within an epoch.

    Inputs:
    -------
    spike_train : array of spike times
    min_time : minimum time for potential spikes
    max_time : maximum time for potential spikes

    Outputs:
    --------
    presence_ratio : fraction of time bins in which this unit is spiking

    """

    h, b = np.histogram(spike_train, np.linspace(min_time, max_time, num_bins))

    return np.sum(h > 0) / num_bins


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


def amplitude_cutoff(amplitudes, num_histogram_bins=500, histogram_smoothing_value=3):
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


def calculate_isi_violations(
    spike_times, spike_clusters, total_units, isi_threshold, min_isi
):
    cluster_ids = np.unique(spike_clusters)
    viol_rates = np.zeros((total_units,))
    for idx, cluster_id in enumerate(cluster_ids):
        for_this_cluster = spike_clusters == cluster_id
        viol_rates[idx], num_violations = isi_violations(
            spike_times[for_this_cluster],
            min_time=np.min(spike_times),
            max_time=np.max(spike_times),
            isi_threshold=isi_threshold,
            min_isi=min_isi,
        )

    return viol_rates


def calculate_presence_ratio(spike_times, spike_clusters, total_units):
    cluster_ids = np.unique(spike_clusters)
    ratios = np.zeros((total_units,))
    for idx, cluster_id in enumerate(cluster_ids):
        for_this_cluster = spike_clusters == cluster_id
        ratios[idx] = presence_ratio(
            spike_times[for_this_cluster],
            min_time=np.min(spike_times),
            max_time=np.max(spike_times),
        )

    return ratios


def calculate_firing_rate(
    spike_times, spike_clusters, total_units, min_time=-1, max_time=-1
):
    cluster_ids = np.unique(spike_clusters)
    firing_rates = np.zeros((total_units,))
    if min_time == -1:
        min_time = np.min(spike_times)
    if max_time == -1:
        max_time = np.max(spike_times)

    for idx, cluster_id in enumerate(cluster_ids):
        for_this_cluster = spike_clusters == cluster_id
        firing_rates[idx] = firing_rate(
            spike_times[for_this_cluster],
            min_time=np.min(spike_times),
            max_time=np.max(spike_times),
        )

    return firing_rates


def calculate_amplitude_cutoff(spike_clusters, amplitudes, total_units):
    cluster_ids = np.unique(spike_clusters)
    amplitude_cutoffs = np.zeros((total_units,))
    for idx, cluster_id in enumerate(cluster_ids):
        for_this_cluster = spike_clusters == cluster_id
        amplitude_cutoffs[idx] = amplitude_cutoff(amplitudes[for_this_cluster])
    return amplitude_cutoffs


def calculate_metrics(spike_times, spike_clusters, amplitudes, isi_threshold, min_isi, acqs):
    """Calculate metrics for all units on one probe

    Inputs:
    ------
    spike_times : numpy.ndarray (num_spikes x 0)
        Spike times in seconds (same timebase as epochs)
    spike_clusters : numpy.ndarray (num_spikes x 0)
        Cluster IDs for each spike
    spike_templates : numpy.ndarray (num_spikes x 0)
        Original template IDs for each spike time
    amplitudes : numpy.ndarray (num_spikes x 0)
        Amplitude value for each spike time
    params : dict of parameters
        'isi_threshold' : minimum time for isi violations


    Outputs:
    --------
    metrics : pandas.DataFrame
        one column for each metric
        one row per unit per epoch

    """

    total_units = len(np.unique(spike_clusters))
    metrics = np.zeros((total_units, 5))
    labels = [
        "isi_violations",
        "presence_ratio",
        "firing_rate",
        "amplitude_cutoff",
        "cluster_ids",
    ]

    metrics[:, 0] = calculate_isi_violations(
        spike_times,
        spike_clusters,
        total_units,
        isi_threshold,
        min_isi,
    )

    metrics[:, 1] = calculate_presence_ratio(spike_times, spike_clusters, total_units)

    metrics[:, 2] = calculate_firing_rate(spike_times, spike_clusters, total_units)

    metrics[:, 3] = calculate_amplitude_cutoff(spike_clusters, amplitudes, total_units)
    metrics[:, 4] = np.unique(spike_clusters)

    return labels, metrics
