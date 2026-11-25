# InVivoSuite
InVivoSuite is intended for analysis LFPs and spike data recorded using multisite silicon probes and is currently a work in progress with major changes occuring. The primary goal of InVivoSuite is to provide an interface for analyzing baseline in vivo electrophysiology data (i.e. 10 minute recordings of at rest neural activity). InVivoSuite can also convert Plexon AnalogChannels from pl2 files to hdf5 files as long as you use Windows since Plexon does not provide libraries for Linux or Mac. InVivoSuite also can export a binary file for use in Kilosort and Phy. The Phy export function works better than Phy's export-waveforms by allowing subtraction of overlapping waveforms and extraction of asymmetric waveforms (spike waveform decay is longer than the rise). While package is not necessarily intended for broad use there are also many use spike, spike-lfp and lfp functions that can be used independently of the acquisition manager classes. These include maximum interval spike burst finding (most accurate method), circular statistics for spike-phase data, spike metrics (STTC, network analysis, Allen Institute quality assessments), spike waveform metrics, binary to continuous spike conversion, spike binning, synthetic spike train generation, python based CWT and fastCWT (only requires pyFFTW and not fcwt), phase discontinuities, fast LFP power methods, and fast FFT methods (wrapper around pyFFTW). Future additions will include extraction of timestamped spike data (trials, optogenetic pulse trains, etc.) and some spike synchrony/connectivity analysis. Some benefits of these functions over Elephant of Spike Interface are easy of use (no required use of Quantities package or complicated object interfaces) and speed (Elephant STTC is about 10x slower).

InVivoSuite now also includes (see ROADMAP.md for details and acceptance
tests):
- a plain-NumPy, vectorized port of Elephant's kCSD current-source-density
  methods (`invivosuite.csd`: KCSD1D for laminar probes, KCSD2D for planar
  geometries; the estimation operator is precomputed once and applied as a
  single matrix multiplication so long recordings are cheap, and Elephant's
  bootstrapped cross-validation is optional and segment-based);
- multichannel spike removal from the wideband
  (`spike_lfp_functions.spike_removal`: per-spike LSQ-scaled template
  subtraction across template-covered channels plus scipy/PCHIP/spatial gap
  repair) - a prerequisite for gamma-band analyses, since spike power bleeds
  far below the spike band;
- and the wired-together spike-free CSD pipeline on `AcqManager`:
  `remove_spikes_wideband()` -> `get_clean_lfp()` -> `csd()` ->
  `csd_spike_phase()`, producing spike-phase coupling estimates from the
  decontaminated current-source density.

The repository ships `AGENTS.md` (orientation for humans and AI coding
agents), `ROADMAP.md` (task decomposition with acceptance criteria) and a
pytest suite with a synthetic HDF5 + fake-Kilosort fixture, so everything
can be developed and validated without real recordings: `uv run pytest`.


For examples on how to use the package with the AcqManager interface see the file-setup.py, spike-analysis.py and spike-lfp.py. For an example on how to compile and clean your extracted data see final-analysis.py.