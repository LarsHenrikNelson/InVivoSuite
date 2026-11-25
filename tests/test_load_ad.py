"""Regression tests for the HDF5 loading helpers in ``invivosuite.acq.load_ad``.

``load_hdf5_acqs`` builds base ``AcqManager`` instances (the data-access
layer) from HDF5 files on disk; results must be wrapped in
``SpkLFPManager`` before running spike/LFP analyses.
"""

import numpy as np

from invivosuite.acq import AcqManager, load_hdf5_acqs

from .conftest import N_CHANS, N_SAMPLES


def test_load_hdf5_acqs_returns_usable_managers(synthetic_path):
    hdf5_path, _ks_dir = synthetic_path
    managers = load_hdf5_acqs(str(hdf5_path.parent))
    assert len(managers) == 1
    manager = managers[0]
    assert isinstance(manager, AcqManager)
    # A base AcqManager can still read wideband data directly.
    wideband = manager.acq(0, acq_type="wideband", ref_type="none")
    assert wideband.shape == (N_SAMPLES,)
    assert manager.shape == (N_CHANS, N_SAMPLES)