"""Shared pytest fixtures: a small synthetic HDF5 recording with an injected
gamma/theta LFP, spike trains with known templates, and a fake Kilosort
directory. Everything downstream (spike removal, CSD, spike-phase analysis)
can be tested without real recordings. Generation lives in
`invivosuite.functions.simulation.laminar_sim` (single source of truth).
"""

import numpy as np
import pytest

from invivosuite.acq import AcqManager
from invivosuite.functions.simulation.laminar_sim import build_laminar_recording

FS = 40_000
N_CHANS = 8
N_SAMPLES = FS * 4
GAMMA_HZ = 60.0
THETA_HZ = 8.0
CLUSTER0_CHAN = 4
CLUSTER1_CHAN = 2


def build_recording(rng):
    """(n_chans, n_samples) int16 acquisition matrix with LFP + injected
    spikes, plus the bookkeeping arrays."""
    data = build_laminar_recording(
        n_chans=N_CHANS,
        fs=FS,
        duration_s=N_SAMPLES / FS,
        gamma_hz=GAMMA_HZ,
        theta_hz=THETA_HZ,
        gamma_phase_per_chan=1.0,
        spike_rates=(20.0, 8.0),
        cluster_channels=[CLUSTER0_CHAN, CLUSTER1_CHAN],
        seed=42,
    )
    return (
        data["acqs"],
        data["spike_times"],
        data["spike_clusters"],
        data["amplitudes"],
        data["templates"],
    )


@pytest.fixture(scope="session")
def synthetic_dataset():
    rng = np.random.default_rng(42)
    return build_recording(rng)



@pytest.fixture
def synthetic_path(tmp_path, synthetic_dataset):
    """HDF5 file + fake Kilosort directory on disk."""
    acqs, spike_times, spike_clusters, amplitudes, templates = synthetic_dataset
    hdf5_path = tmp_path / "synthetic.hdf5"
    ks_dir = tmp_path / "ks"
    ks_dir.mkdir()
    manager = AcqManager(hdf5_path)
    manager.create_hdf5_file(
        acqs=acqs,
        wb_channels=np.arange(N_CHANS),
        sample_rates=np.full(N_CHANS, FS),
        coeffs=np.ones(N_CHANS),
        timestamps=np.zeros(N_CHANS),
        units=np.array([b"uV"] * N_CHANS),
        enabled=np.ones(N_CHANS, dtype=np.int16),
        identifier="synthetic",
    )
    # Identity channel map: multichan_acq reads it even when map_channel=False.
    manager.set_channel_map(np.arange(N_CHANS), probe="all")
    # Kilosort files. NOTE: spike_times are stored 1-indexed (Matlab style);
    # SpkManager subtracts 1 at load time.
    np.save(ks_dir / "spike_times.npy", (spike_times + 1).astype(np.int64))
    np.save(ks_dir / "spike_clusters.npy", spike_clusters)
    np.save(ks_dir / "spike_templates.npy", spike_clusters)  # 1 template per cluster map
    np.save(ks_dir / "templates.npy", templates)
    np.save(ks_dir / "amplitudes.npy", amplitudes)
    return hdf5_path, ks_dir


@pytest.fixture
def synthetic_manager(tmp_path, synthetic_path):
    """Loaded AcqManager with LFP filter settings and virtual references ready."""
    hdf5_path, ks_dir = synthetic_path
    manager = AcqManager(hdf5_path)
    manager.set_filter(
        "lfp",
        filter_type="butterworth_zero",
        order=4,
        highpass=0.5,
        lowpass=300,
        sample_rate=1000,
    )
    manager.set_filter(
        "spike",
        filter_type="butterworth_zero",
        order=4,
        highpass=300,
        sample_rate=FS,
    )
    manager.compute_virtual_ref("cmr", probe="all")
    manager.compute_virtual_ref("car", probe="all")
    manager.load_kilosort(ks_dir)
    return manager
