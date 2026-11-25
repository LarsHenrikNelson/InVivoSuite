# AGENTS.md — Guide for AI coding agents working in this repository

InVivoSuite is a Python package (GPLv3) for analyzing in-vivo electrophysiology:
spikes (Kilosort output) and LFPs from multisite silicon probes, stored as HDF5.
It intentionally avoids `neo`/`quantities` unit arrays — everything is plain NumPy.

## Environment & Commands

- Package manager: **uv**. Python >= 3.10 (repo pins 3.12 via `.python-version`).
- `uv sync` — install/sync the venv (creates `.venv`). Dev extras (pytest, ruff) install automatically from the `dev` group.
- `uv run pytest` — run the test suite (also: `.venv\Scripts\python -m pytest`).
- `uv run ruff check .` — lint (config in `pyproject.toml`; F401/F403 ignored).
- Windows is required only for PL2 → HDF5 conversion (Plexon DLL in `invivosuite/acq/bin`). All other functionality is platform independent.

## Repository map

- `invivosuite/acq/` — the manager/data-access layer, organized as an
  inheritance hierarchy (the manager classes are no longer base-less mixins of a
  composed class — see git history around the "analysis classes separate" refactor):
  - `acq_manager.py` — `AcqManager`, the **base class**: HDF5 file creation and
    plumbing (`set/get_file_*`, `set/get_grp_*`), `acq`/`multichan_acq`/`avg_acq`,
    `set_filter`, `compute_virtual_ref`, channel maps, probes, analysis window.
  - `spike_manager.py` — `SpkManager(AcqManager)`: loads Kilosort outputs from a KS directory
    (`spike_times.npy`, `spike_clusters.npy`, `spike_templates.npy`, `templates.npy`,
    `amplitudes.npy`; `accepted_units.npy` and `celltype.csv` auto-created).
  - `lfp_manager.py` — `LFPManager(AcqManager)`: spectral settings (`set_cwt`,
    `set_welch`, ...) and LFP analyses. Settings are stored as HDF5 group attrs
    and read back with `get_spectral_settings`.
  - `spike_lfp_manager.py` — `SpkLFPManager(LFPManager, SpkManager)`: spike–LFP
    analyses (spike phase/power) and the spike-free CSD pipeline (spike removal,
    `csd`, `csd_spike_phase`). **This is the full-stack class: instantiate
    `SpkLFPManager` when you need everything** — the base `AcqManager` no longer
    exposes the analysis methods.
  - `load_ad.py` — PL2/HDF5 loaders. The `pypl2` DLL imports are lazy so the package
    imports on machines without the Plexon DLL.
- `invivosuite/functions/` — pure NumPy algorithm modules (usable without any manager):
  - `lfp_functions/`, `spike_functions/`, `spike_lfp_functions/` (incl. spike removal),
    `simulation/` (Izhikevich; synthetic laminar data), `filter_functions.py`,
    `signal_functions.py`, `circular_stats.py`. Subpackages re-export selected names
    in their `__init__.py`.
- `invivosuite/csd/` — kCSD current-source-density (ported from Elephant's
  `current_source_density_src`, BSD-3, by Chintaluri & Czerwinski; vectorized,
  plain NumPy, no `quantities`). `KCSD2D` is the primary entry point; `KCSD1D`
  is included for single-shank (laminar) probes.
- `invivosuite/spectral/` — FFT (pyFFTW), multitaper, CWT helpers (pyfcwt in managers).
- `invivosuite/asymp_pdc/` — asymptotic partial directed coherence.
- `invivosuite/utils/` — plotting, TSV helpers, dataset utilities.

## HDF5 file schema (created by `create_hdf5_file` / PL2 conversion)

- Datasets (root): `acqs` (n_chans × n_samples, int16), `coeffs`, `means`,
  `sample_rate`, `timestamps`, `units`, `enabled`, `wb_channels`, attrs `id/start/end`.
- The `start`/`end` file attributes define the analysis window. **All `start`/`end`
  arguments in manager methods are relative to this window** (`acq()` adds
  `self.start` internally). This has historically caused bugs (see git history)
  — keep the convention consistent when adding methods.
- Groups: `probes/<name>` — channel index ranges `[chan0, chan1]`;
  `channel_maps/<probe>` — channel remap; `<ref_type>/<probe>` (`cmr`/`car`) —
  reference traces over time; `spike`/`lfp` groups hold filter settings as attrs
  (missing values stored as string `"None"`, converted back by `get_filter`);
  analysis-specific groups (e.g. `lfp_bursts/<channel>`, `freq_bands`, `csd`).

## Conventions

- Array orientation: `(n_chans, n_samples)` everywhere; time along axis 1.
- Spike times are sample indices into `acqs` (sample-rate of the wideband data).
  Matlab Kilosort output is 1-based: `load_kilosort(..., matlab_indexing=True)`
  subtracts 1 at load. The default (`matlab_indexing=False`, no correction) is
  correct for Kilosort 4, whose spike times are already 0-based.
- Binned-time convention for spike-phase methods: `spike_times // nperseg` bins
  spikes into downsampled LFP time bins (wideband fs / nperseg = LFP fs).
- Referencing: `ref_type` in {"none", "cmr" (median), "car" (mean)}. CMR/CAR traces
  are precomputed per probe via `compute_virtual_ref`.
- Long-running loops: prefer chunking (see `extract_waveforms` for the pattern:
  chunk overlap >= template length) and `self.callback` for progress reporting.
- Numba: use `@njit(cache=True)` (and `prange` where safe) for hot numeric kernels
  that only use NumPy. **Never** `@njit` functions calling scipy (`integrate`, etc.).
- Typing: modern syntax (`int | None`), `Literal` for string options; NumPy-style docstrings.
- Pure-function kernels live in `invivosuite/functions/...`; managers only orchestrate
  (data access, chunking, persistence). Keep kernels manager- and platform-independent
  and test them without any HDF5 file (see `tests/`).

## Testing

- `tests/conftest.py` builds a small synthetic HDF5 recording + fake Kilosort
  directory (fixtures `synthetic_manager` / `synthetic_path`), so the manager
  classes can be exercised without real data (`synthetic_manager` is a
  `SpkLFPManager` — the full stack; `synthetic_path` uses the base `AcqManager`
  to build files). Reuse these fixtures rather than mocking HDF5.
- Add regression tests alongside bug fixes (e.g. `tests/test_filters.py` guards the
  IIR-notch coefficient order).

## Known gotchas (verify before assuming)

- `set_filter` stores `None` as the string `"None"` in HDF5 attrs; `get_filter`
  converts back — round-trip only through those helpers.
- `iirnotch_zero` is zero-phase (`filtfilt`); `iirnotch` is one-pass; both expect
  `signal.iirnotch(freq, q, fs)` return order `(b, a)`.
- The old, partially-broken vendored CSD code was replaced by the vectorized port —
  if you find references to `container_classes` from before 2027, they are stale.
- `cwt_spike_phase` builds its `pyfcwt.Wavelet` with the *wideband* fs while
  analyzing the downsampled (wideband/nperseg) trace, so its returned
  frequency labels are off by nperseg. It is kept as-is for backwards
  compatibility; `csd_spike_phase` uses the correct analysis rate.
- `invivosuite/asymp_pdc/pdc.py` and `invivosuite/functions/curve_fit.py`
  contain pre-existing syntax errors (they do not import); they are excluded
  from `ruff check` (see `pyproject.toml`). `load_acq`/`load_hdf5_acqs` in
  `load_ad.py` are stale helpers that call methods that no longer exist.
- Spike removal must run on the raw wideband with `ref_type="none"`
  *before* any CMR/CAR referencing: the virtual references contain spike
  transients and would remix them into clean channels (`csd`, `get_clean_lfp`
  and `remove_spikes_wideband` enforce this ordering).
- The phase convention of the synthetic laminar data: gamma is built as
  `A*sin(phase)`, so its analytic phase is `phase - pi/2` — align conventions
  when asserting phase recovery (see `tests/test_csd_wiring.py`).

## Roadmap

See `ROADMAP.md` for in-flight projects and acceptance criteria per task.
