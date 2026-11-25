"""Benchmark-driven tests for multichannel spike removal.

Measured on the synthetic laminar ground truth (60 Hz gamma, 45-85 Hz
analysis band; *local* = mean circular phase error within spike windows):

    contaminated            0.0241 rad
    template subtraction    0.0026 rad   (~9x better)
    hybrid + pchip          0.0083 rad
    hybrid + spatial        0.0032 rad   (best with robustness to misses)
    interpolate-only
      linear                0.0213 rad   (barely improves contamination)
      pchip                  0.0272 rad
      cubic                  0.6992 rad   (rings on noisy oscillatory anchors)
      akima                  0.1391 rad

These encode the methodological conclusions (see the module docstring):
subtraction is mandatory for gamma; interpolation's role is repairing
residuals after subtraction; PCHIP is the robust temporal default.
"""

import numpy as np
import pytest
from scipy.signal import hilbert

from invivosuite.functions.filter_functions import butterworth_zero
from invivosuite.functions.simulation.laminar_sim import build_laminar_recording
from invivosuite.functions.spike_lfp_functions.spike_removal import (
    detect_spike_windows,
    interpolate_windows,
    remove_spikes,
    spike_triggered_average,
    spike_windows,
    subtract_templates,
)

FS = 40_000
BAND = (45.0, 85.0)


@pytest.fixture(scope="module")
def rec():
    return build_laminar_recording(seed=42)


@pytest.fixture(scope="module")
def spike_band_fn():
    return lambda x: butterworth_zero(x, 4, FS, highpass=300.0, lowpass=6000.0)


def _band_phase_error(tested, clean, fs, band=BAND, local_mask=None):
    b_clean = butterworth_zero(
        clean.astype(float), 4, fs, highpass=band[0], lowpass=band[1]
    )
    b_tested = butterworth_zero(
        np.asarray(tested, dtype=float), 4, fs, highpass=band[0], lowpass=band[1]
    )
    err = np.abs(
        np.angle(
            np.exp(
                1j * (np.angle(hilbert(b_tested)) - np.angle(hilbert(b_clean)))
            )
        )
    )
    if local_mask is None:
        return float(err.mean())
    return float(err[local_mask].mean())


def _local_mask(recording):
    wins = spike_windows(
        recording["spike_times"], recording["spike_clusters"],
        templates=recording["templates"],
    )
    mask = np.zeros_like(recording["acqs"], dtype=bool)
    for c in range(recording["acqs"].shape[0]):
        for s, e in wins[c]:
            mask[c, max(0, s - 82) : min(mask.shape[1], e + 82)] = True
    return mask

def test_template_subtraction_recovers_ground_truth(rec):
    acqs, clean = rec["acqs"], rec["clean"]
    times, clusters, amps, templates = (
        rec["spike_times"], rec["spike_clusters"], rec["amplitudes"],
        rec["templates"],
    )
    before = np.sqrt(np.mean((acqs - clean) ** 2)) / clean.std()
    cleaned, scales = subtract_templates(
        acqs, times, clusters, templates, spatial_scale="per_channel"
    )
    after = np.sqrt(np.mean((cleaned - clean) ** 2)) / clean.std()
    assert after < 0.2 * before, (before, after)
    # fitted scales recover the injected amplitude scaling
    np.testing.assert_allclose(
        np.mean(scales), rec["spike_amps"] * np.mean(amps), rtol=0.05
    )


def test_hybrid_beats_contamination_gamma(rec, spike_band_fn):
    acqs, clean = rec["acqs"], rec["clean"]
    times, clusters, templates = (
        rec["spike_times"], rec["spike_clusters"], rec["templates"]
    )
    mask = _local_mask(rec)
    err_contaminated = _band_phase_error(acqs, clean, FS, local_mask=mask)
    assert err_contaminated > 0.01, err_contaminated

    hybrid_pchip = remove_spikes(
        acqs, times, clusters, templates, method="hybrid",
        interpolation_kind="pchip", detect_on=spike_band_fn,
    )
    hybrid_spatial = remove_spikes(
        acqs, times, clusters, templates, method="hybrid",
        interpolation_kind="spatial", detect_on=spike_band_fn,
    )
    err_pchip = _band_phase_error(hybrid_pchip, clean, FS, local_mask=mask)
    err_spatial = _band_phase_error(hybrid_spatial, clean, FS, local_mask=mask)
    assert err_pchip < 0.5 * err_contaminated, (err_pchip, err_contaminated)
    assert err_spatial < 0.3 * err_contaminated, (err_spatial, err_contaminated)
    # the cross-channel spatial repair is at least as good as pchip here
    assert err_spatial < err_pchip, (err_spatial, err_pchip)

    # residual spike-triggered average collapses
    sta_before, _ = spike_triggered_average(acqs, times, clusters)
    sta_after, _ = spike_triggered_average(hybrid_spatial, times, clusters)
    assert np.abs(sta_after).max() < 0.1 * np.abs(sta_before).max()


def test_interpolation_only_is_a_fallback_not_a_gamma_solution(rec):
    acqs, clean = rec["acqs"], rec["clean"]
    times, clusters, templates = (
        rec["spike_times"], rec["spike_clusters"], rec["templates"]
    )
    mask = _local_mask(rec)
    err_contaminated = _band_phase_error(acqs, clean, FS, local_mask=mask)

    results = {}
    for kind in ("linear", "pchip", "cubic", "akima", "spatial"):
        cleaned = remove_spikes(
            acqs, times, clusters, templates,
            method="interpolate", interpolation_kind=kind,
        )
        results[kind] = _band_phase_error(cleaned, clean, FS, local_mask=mask)
    # modest, not transformative (documents the <25 Hz consensus)
    assert results["linear"] < 1.1 * err_contaminated, ("linear", results)
    assert results["pchip"] < 1.4 * err_contaminated, ("pchip", results)
    # cubic/akima can ring badly on noisy anchors during long-gap repair
    assert results["cubic"] > 2 * results["linear"], results
    assert results["akima"] > 2 * results["linear"], results


def test_detect_windows_covers_spikes(rec):
    # MAD detection must run on the spike band: on the wideband the LFP
    # dominates the robust scale and hides the spikes.
    acqs = rec["acqs"]
    times, clusters = rec["spike_times"], rec["spike_clusters"]
    templates = rec["templates"]
    spike_band = butterworth_zero(acqs.astype(float), 4, FS, highpass=300.0)
    wins = detect_spike_windows(spike_band, threshold=6.0)
    expected = spike_windows(times, clusters, templates=templates)
    exp_total = 0
    overlap_total = 0
    for c in range(acqs.shape[0]):
        exp_mask = np.zeros(acqs.shape[1], dtype=bool)
        for s, e in expected[c]:
            exp_mask[s:e] = True
        det_mask = np.zeros(acqs.shape[1], dtype=bool)
        for s, e in wins[c]:
            det_mask[s:e] = True
        overlap_total += int((exp_mask & det_mask).sum())
        exp_total += int(exp_mask.sum())
    assert overlap_total > 0.85 * exp_total, (overlap_total, exp_total)


def test_interpolate_windows_basic():
    t = np.arange(2000) / 1000.0
    x = np.sin(2 * np.pi * 10 * t)
    acq = x[None, :].copy()
    acq[0, 500:512] = 0.0  # 12 ms gap (~2% of a cycle, spike-window scale)
    wins = [np.array([[500, 512]])]
    for kind in ("linear", "cubic", "pchip", "akima"):
        fixed = interpolate_windows(acq, wins, kind=kind)
        err = np.sqrt(np.mean((fixed[0, 500:512] - x[500:512]) ** 2)) / x.std()
        assert err < 0.35, (kind, err)


