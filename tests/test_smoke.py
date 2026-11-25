"""Smoke tests for the synthetic fixture and core AcqManager plumbing."""

import numpy as np

import invivosuite
from invivosuite.acq import AcqManager

from .conftest import (
    CLUSTER0_CHAN,
    CLUSTER1_CHAN,
    FS,
    N_CHANS,
    N_SAMPLES,
    THETA_HZ,
)


def test_package_imports():
    for module in ("acq", "functions", "spectral", "csd"):
        __import__(f"invivosuite.{module}")
    assert invivosuite is not None


def test_wideband_acq_matches_file(synthetic_manager):
    wideband = synthetic_manager.acq(0, acq_type="wideband", ref_type="none")
    assert wideband.shape == (N_SAMPLES,)
    with __import__("h5py").File(synthetic_manager.file_path, "r") as f:
        raw = f["acqs"][0] * f["coeffs"][0] - f["means"][0]
    np.testing.assert_allclose(wideband, raw, rtol=1e-12)


def test_lfp_filter_downsamples(synthetic_manager):
    # Channel 7 is far from both spike clusters, so it carries mostly the
    # clean LFP mix for this check.
    lfp = synthetic_manager.acq(
        7, acq_type="lfp", ref_type="none", map_channel=False
    )
    lfp_fs = FS // 40
    assert lfp.shape[0] == N_SAMPLES // 40
    # The theta oscillation must survive the 0.5-300 Hz bandpass. The ceiling is
    # < 1 because the channel also carries gamma and spike-band residue; the
    # ideal reference correlation lands around 0.89 for this signal mix.
    ref_t = np.arange(lfp.size) / lfp_fs
    ref = np.sin(2 * np.pi * THETA_HZ * ref_t)
    corr = np.corrcoef(lfp, ref)[0, 1]
    assert corr > 0.8


def test_cmr_removes_common_mode(synthetic_manager):
    # Theta is identical across channels, so the common-median reference must
    # suppress it relative to the unreferenced channel.
    none = synthetic_manager.acq(0, "lfp", ref_type="none", map_channel=False)
    cmr = synthetic_manager.acq(0, "lfp", ref_type="cmr", map_channel=False)
    ref_t = np.arange(none.size) / (FS // 40)
    ref = np.sin(2 * np.pi * THETA_HZ * ref_t)
    corr_none = np.corrcoef(none, ref)[0, 1]
    corr_cmr = np.corrcoef(cmr, ref)[0, 1]
    assert corr_none > 0.8
    assert abs(corr_cmr) < 0.25


def test_multichan_acq_shape(synthetic_manager):
    multi = synthetic_manager.multichan_acq(
        "wideband", ref_type="none", map_channel=False
    )
    assert multi.shape == (N_CHANS, N_SAMPLES)


def test_spike_times_loaded_matlab_offset(synthetic_manager, synthetic_dataset):
    _, spike_times, spike_clusters, _, _ = synthetic_dataset
    for cid in (0, 1):
        loaded = synthetic_manager.get_cluster_spike_times(cid)
        expected = np.sort(spike_times[spike_clusters == cid])
        np.testing.assert_array_equal(loaded, expected)


def test_channel_clusters(synthetic_manager):
    chan_dict = synthetic_manager.get_channel_clusters()
    assert CLUSTER0_CHAN in chan_dict and chan_dict[CLUSTER0_CHAN] == [0]
    assert CLUSTER1_CHAN in chan_dict and chan_dict[CLUSTER1_CHAN] == [1]


def test_set_file_window(synthetic_manager):
    manager = synthetic_manager
    assert manager.start == 0
    assert manager.end == N_SAMPLES
    manager.set_start(N_SAMPLES // 2)
    assert manager.acq(0, acq_type="wideband").size == N_SAMPLES // 2
