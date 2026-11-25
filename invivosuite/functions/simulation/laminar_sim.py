"""Synthetic laminar recordings for developing/validating spike removal and
CSD-based analyses.

Generates a multi-channel wideband signal with:
- a ground-truth gamma oscillation whose *phase varies linearly with depth*
  (spatially structured, CSD-relevant),
- an optional common-mode theta component,
- gaussian white noise,
- injected multi-channel spike trains (per-cluster rates, amplitude jitter,
  gaussian channel footprints, zero-mean per channel so spikes do not inject
  artificial DC steps into the LFP band).

The generator returns both the contaminated data and the ground-truth
(spikes, templates, clean LFP), so removal quality and phase-recovery can be
quantified exactly.
"""

import numpy as np

__all__ = [
    "TEMPLATE_CENTER",
    "TEMPLATE_LEN",
    "build_laminar_recording",
    "spike_waveform",
]

TEMPLATE_LEN = 82  # Kilosort convention used throughout InVivoSuite
TEMPLATE_CENTER = 41


def spike_waveform(fs, template_center=TEMPLATE_CENTER, template_len=TEMPLATE_LEN):
    """Triphasic spike: sharp negative peak, small positive rebound, slow tail
    (parameters in ms, so shapes carry across sample rates)."""
    t_ms = (np.arange(template_len) - template_center) / (fs / 1000.0)
    neg = -np.exp(-((t_ms + 0.15) ** 2) / (2 * 0.08**2))
    pos = 0.25 * np.exp(-((t_ms - 0.55) ** 2) / (2 * 0.35**2))
    late = 0.08 * np.exp(-((t_ms + 3.0) ** 2) / (2 * 1.5**2))
    return neg + pos + late


def multi_channel_templates(n_chans, centers, footp_sd=1.5, fs=40_000):
    """(n_templates, 82, n_chans) templates with gaussian channel footprints.

    Real action potentials have (nearly) zero net area over the extracted
    window, so each channel's template is demeaned.
    """
    base = spike_waveform(fs)
    templates = np.zeros((len(centers), base.size, n_chans))
    for i, center in enumerate(centers):
        for ch in range(n_chans):
            d = ch - center
            templates[i, :, ch] = base * np.exp(-(d**2) / (2 * footp_sd**2))
        templates[i] -= templates[i].mean(axis=0, keepdims=True)
    return templates


def make_spikes(n_samples, rates, fs, rng, min_sep=TEMPLATE_LEN):
    """Poisson spike trains for each cluster (sorted, cluster-major merged).

    Returns (spike_times, spike_clusters, amplitudes) zero-based and sorted
    by time.
    """
    out_times, out_clusters, out_amps = [], [], []
    for cid, rate in enumerate(rates):
        n = rng.poisson(rate * n_samples / fs)
        times = np.sort(
            rng.integers(min_sep, n_samples - min_sep, n)
        )
        out_times.append(times)
        out_clusters.append(np.full(times.size, cid))
        out_amps.append(0.6 + 0.8 * rng.random(times.size))
    times = np.concatenate(out_times)
    clusters = np.concatenate(out_clusters)
    amplitudes = np.concatenate(out_amps)
    order = np.argsort(times, kind="stable")
    return (
        times[order].astype(np.int64),
        clusters[order],
        amplitudes[order],
    )


def build_laminar_recording(
    n_chans=8,
    fs=40_000,
    duration_s=4.0,
    gamma_hz=60.0,
    gamma_amp=25.0,
    gamma_phase_per_chan=0.2,
    theta_hz=8.0,
    theta_amp=50.0,
    noise_sd=8.0,
    spike_rates=(20.0, 8.0),
    spike_amps=300.0,
    cluster_channels=None,
    seed=42,
):
    """Build a synthetic laminar recording with known ground truth.

    Parameters
    ----------
    n_chans, fs, duration_s : recording geometry/timing
    gamma_hz : ground-truth gamma frequency.
    gamma_amp : gamma amplitude (per channel).
    gamma_phase_per_chan : radians of gamma phase lag per channel of depth.
    theta_hz, theta_amp : common-mode (identical across channels) component.
    spike_rates : per-cluster Poisson rates [Hz]; len() sets n_clusters.
    spike_amps : scale applied to the (unit-peak) templates when injecting.
    cluster_channels : channel index of each cluster's peak. Defaults to
        spreading clusters over the middle of the array.

    Returns
    -------
    data : dict with keys
        ``acqs`` contaminated int16 (n_chans, n_samples),
        ``clean`` float64 LFP+noise before spikes (n_chans, n_samples),
        ``gamma_phase`` ground-truth gamma phase (n_chans, n_samples),
        ``spike_times``, ``spike_clusters``, ``amplitudes``, ``templates``.
    """
    rng = np.random.default_rng(seed)
    n_samples = int(fs * duration_s)
    depth = np.arange(n_chans)
    t = np.arange(n_samples) / fs
    if cluster_channels is None:
        n_clusters = len(spike_rates)
        cluster_channels = np.linspace(
            n_chans // 3, 2 * n_chans // 3, n_clusters
        ).astype(int)
    templates = multi_channel_templates(n_chans, list(cluster_channels), fs=fs)
    gamma_phase = 2 * np.pi * gamma_hz * t[None, :] - gamma_phase_per_chan * depth[:, None]
    clean = np.empty((n_chans, n_samples))
    clean[:] = (
        gamma_amp * np.sin(gamma_phase) + theta_amp * np.sin(2 * np.pi * theta_hz * t)
    )
    clean += rng.normal(0, noise_sd, size=clean.shape)
    spike_times, spike_clusters, amplitudes = make_spikes(
        n_samples, spike_rates, fs, rng
    )
    data = clean.copy()
    for time, cid, amp in zip(spike_times, spike_clusters, amplitudes):
        start = time - TEMPLATE_CENTER
        data[:, start : start + TEMPLATE_LEN] += (
            templates[int(cid)] * amp * spike_amps
        ).T
    return {
        "acqs": data.astype(np.int16),
        "clean": clean,
        "gamma_phase": gamma_phase,
        "spike_times": spike_times,
        "spike_clusters": spike_clusters,
        "amplitudes": amplitudes,
        "templates": templates,
        "cluster_channels": np.asarray(cluster_channels),
        "spike_amps": float(spike_amps),
    }
