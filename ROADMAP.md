# InVivoSuite Roadmap

Task decomposition with acceptance criteria, structured so individual tasks are
safe to hand to AI coding agents (each ends with a green test run).

## Project: Spike-free CSD for spike–phase analysis (in flight)

Motivation: spikes bleed broadband power into gamma (30–100 Hz) LFP; CSD (a spatial
high-pass) sharpens single-contact spike artifacts rather than removing them, so the
wideband must be decontaminated *before* taking the spatial derivative. Goal:
`SpkLFPManager.remove_spikes_wideband()` → KCSD → `csd_spike_phase()` reusing the
existing circular-statistics stack.

- [x] **Phase 0 — Agent enablement scaffold.**
      AGENTS.md, ROADMAP.md, pytest infra + synthetic HDF5/Kilosort fixtures,
      latent-bug fixes (IIR notch), lazy PL2 imports, CI workflow.
      *Acceptance:* `uv run pytest` green on a clean checkout; `import invivosuite`
      works without the Plexon DLL.
- [x] **Phase 1 — KCSD port (from Elephant `current_source_density_src`).**
      Plain NumPy (no `quantities`/`neo`), `KCSD2D` primary + `KCSD1D` for laminar
      single-shank probes; `values()` nested loops replaced by one precomputed
      operator `M = K_interp_cross @ inv(K_pot + λI)`; time-chunked application;
      bootstrapped/LOO cross-validation dropped by default (optional, vectorized,
      segment-based). *Acceptance:* vectorized output matches the reference loop
      implementation to ~1e-10; gaussian-dipole reconstruction tests pass; full vs.
      chunked estimates match.
- [x] **Phase 2 — Multichannel spike removal.**
      `functions/spike_lfp_functions/spike_removal.py`: per-spike LSQ-scaled
      template subtraction across all template-covered channels
      (`spatial_scale="global"|"per_channel"`), scipy interpolator family
      (`linear|cubic|pchip|akima`), and cross-channel "spatial" gap repair;
      `hybrid` = subtraction + MAD-detector residual repair (detection on the
      spike band via a `detect_on` callable).
      `functions/simulation/laminar_sim.py`: synthetic ground-truth generator.
      *Benchmark measured* (gamma 45-85 Hz, local circular phase error, rad):
      contaminated 0.024 -> hybrid+spatial 0.003, hybrid+pchip 0.008,
      template 0.0026; interpolation-only is a fallback (linear 0.021,
      pchip 0.027; cubic/akima can ring badly, 0.7/0.14). Recorded in
      `tests/test_spike_removal.py`.
- [x] **Phase 3 — Manager wiring.**
      `SpkLFPManager.remove_spikes_wideband()` (chunked, boundary-safe),
      `get_clean_lfp()` (chunked clean->filter->downsample with edge padding),
      `set_csd_geometry()`/`get_csd_geometry()` (persisted in the `csd` HDF5
      group), `csd()` (KCSD1D/KCSD2D dispatch from geometry, spacing-derived
      defaults, per-electrode traces via the precomputed operator), and
      `csd_spike_phase()` (CSD traces -> CWT -> existing circular stats).
      *Acceptance met:* `tests/test_csd_wiring.py` — per-spike gamma phase from
      the spike-free CSD matches the injected truth to <0.5 rad, end-to-end on
      the synthetic fixture.
- [ ] **Phase 4 — Real-data sanity + polish (user-run).**
      Spike-band PSD collapse after removal; no AP-shaped residual in
      spike-triggered CSD; sensible sink/source reversal across depth;
      docstrings already carry the citations (Potworowski 2012; Zanos 2011;
      David 2010; Banaie Boroujeni 2019; Scheffer-Teixeira 2013).

## Future / optional (not scheduled)

- Adaptive per-spike spectral spike removal (ASR, Banaie Boroujeni 2019) as an
  alternative `method=` backend.
- ICA-based multichannel spike-component rejection.
- Spike-train → LFP Wiener-filter decoupling (David 2010) as a `method=` backend.
- kCSD3D / MoIKCSD ports if new probe geometries arrive.
- Public docs/tutorial notebooks beyond `README.md` and the root example scripts.
