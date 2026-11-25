from collections.abc import Iterable
from typing import Literal, Optional

import numpy as np
from joblib import Parallel, delayed

from ..csd import KCSD1D, KCSD2D
from ..functions.circular_stats import periodic_mean, periodic_mean_std, ppc_dot_product
from ..functions.filter_functions import downsample, filter_array
from ..functions.spike_lfp_functions import spike_removal
from ..functions.spike_lfp_functions.spike_phase import (
    analyze_spike_phase,
    cwt_phase_best_frequency,
    extract_spike_phase_data,
)
from ..functions.spike_lfp_functions.spike_power import spike_triggered_lfp

from pyfcwt import Frequencies, PyFCWT, Wavelet
from ..utils import concatenate_dicts, expand_data


class SpkLFPManager:
    def get_cluster_spike_phase(
        self,
        cluster_id: int,
        freq_bands: dict[str, Iterable],
        sxx_type: Literal["cwt", "hilbert"],
        ref_type: Literal["none", "cmr", "car"] = "cmr",
        ref_probe: str = "all",
        map_channel=True,
        probe: str = "all",
        nperseg: int = 40,
        center: Optional[int] = None,
        start: int = 0,
        end: int = 0,
    ) -> tuple[dict]:
        chan = self.get_cluster_channel(cluster_id, center=center)
        band_dict = self.get_sxx_freq_bands(
            sxx_type=sxx_type,
            output_type="phase",
            freq_bands=freq_bands,
            channel=chan,
            ref_type=ref_type,
            ref_probe=ref_probe,
            map_channel=map_channel,
            probe=probe,
            start=start,
            end=end,
        )
        spike_times = self.get_cluster_spike_times(cluster_id) // nperseg
        stats, phases = extract_spike_phase_data(band_dict, spike_times)
        return stats, phases

    def spike_phase(
        self,
        freq_bands: dict[str, Iterable],
        sxx_type: Literal["cwt", "hilbert"],
        ref_type: Literal["none", "cmr", "car"] = "cmr",
        ref_probe: str = "all",
        map_channel=True,
        probe: str = "all",
        nperseg: int = 40,
        start: int = 0,
        end: int = 0,
    ) -> dict[str, np.ndarray]:
        chan_dict = self.get_channel_clusters()
        chans = sorted(list(chan_dict.keys()))
        output_data = []
        analyzed_spk_phase = []
        for chan in chans:
            band_dict = self.get_sxx_freq_bands(
                sxx_type=sxx_type,
                output_type="phase",
                freq_bands=freq_bands,
                channel=chan,
                ref_type=ref_type,
                ref_probe=ref_probe,
                map_channel=map_channel,
                probe=probe,
                start=start,
                end=end,
            )
            for cid in chan_dict[chan]:
                self.callback(f"Extracting spike phase for cluster {cid}.")
                spike_times = self.get_cluster_spike_times(cid) // nperseg
                stats, phases = extract_spike_phase_data(band_dict, spike_times)
                stats["channel"] = chan
                stats["cluster_id"] = cid
                phases["cluster_id"] = np.full(next(iter(phases.values())).size, cid)
                phases["channel"] = np.full(next(iter(phases.values())).size, chan)
                analyzed_spk_phase.append(stats)
                output_data.append(phases)
        output_data = concatenate_dicts(output_data)
        analyzed_spk_phase = concatenate_dicts(analyzed_spk_phase)
        return output_data, analyzed_spk_phase

    def cwt_spike_phase(
        self,
        freq_bands: dict[str, Iterable],
        nperseg: int = 40,
        ref_type: Literal["none", "cmr", "car"] = "cmr",
        ref_probe: str = "all",
        map_channel: bool = False,
        probe: str = "all",
        start: int = 0,
        end: int = 0,
    ) -> dict[str, np.ndarray]:
        sxx_attrs = self.get_grp_attrs("cwt")
        chan_dict = self.get_channel_clusters()
        chans = sorted(list(chan_dict.keys()))

        sample_rate = self.get_file_dataset("sample_rate", rows=0)
        w = Wavelet(sample_rate, imaginary=sxx_attrs["imaginary"])
        f = Frequencies(
            w,
            sxx_attrs["f0"],
            sxx_attrs["f1"],
            sxx_attrs["fn"],
            sample_rate,
            sxx_attrs["scaling"],
        )
        pyf = PyFCWT(
            w,
            f,
            sxx_attrs["nthreads"],
            dtype=sxx_attrs["dtype"],
            norm=sxx_attrs["scaling"],
        )

        cwt_spike_phase = []
        band_specific_phase = []
        for chan in chans:
            wb = self.acq(
                chan,
                acq_type="wideband",
                ref_type=ref_type,
                ref_probe=ref_probe,
                map_channel=map_channel,
                probe=probe,
                start=start,
                end=end,
            )
            acq = downsample(wb, sample_rate, sample_rate / nperseg, 3)
            cwt = pyf.cwt(acq)

            for cid in chan_dict[chan]:
                self.callback(f"Extracting spike phase for cluster {cid}.")
                spike_times = self.get_cluster_spike_times(cid) // nperseg
                temp = np.angle(cwt[:, spike_times])
                stats = Parallel(n_jobs=4, prefer="threads")(
                    delayed(analyze_spike_phase)(temp[i, :])
                    for i in range(temp.shape[0])
                )
                for i, freq in zip(stats, f.f):
                    i["channel"] = chan
                    i["cluster_id"] = cid
                    i["frequency"] = freq

                cwt_spike_phase.extend(stats)
                min_f = f.f.max()
                max_f = f.f.min()
                for key, value in freq_bands.items():
                    min_f = min(min_f, value[0])
                    max_f = max(max_f, value[1])
                    phase_output = cwt_phase_best_frequency(value[0], value[1], f.f, temp)
                    phase_output["band"] = key 
                    phase_output["cluster_id"] = cid
                    phase_output["channel"] = chan
                    band_specific_phase.append(phase_output)
                phase_output = cwt_phase_best_frequency(min_f, max_f, f.f, temp)
                phase_output["band"] = "all" 
                phase_output["cluster_id"] = cid
                phase_output["channel"] = chan
                band_specific_phase.append(phase_output)
        return cwt_spike_phase, band_specific_phase

    def extract_spike_power_data(
        self,
        power_dict: dict[str, np.ndarray],
        cluster_id: int,
        nperseg: int,
        window: int,
    ) -> dict[str, np.ndarray]:
        b_spks = self.get_binned_spike_cluster(cluster_id, nperseg=nperseg)
        output_dict = {}
        spk_indexes = np.where(b_spks > 0)[0]
        output_dict["cluster_id"] = [cluster_id] * spk_indexes.size
        for b_name, power in power_dict.items():
            temp = spike_triggered_lfp(spk_indexes, power, window)
            temp = expand_data(temp, b_spks[spk_indexes])
            output_dict[b_name] = temp
        output_dict["cluster_id"] = np.full(output_dict[b_name].size, cluster_id)
        return output_dict

    def spike_lfp(
        self,
        freq_bands: dict[str, Iterable],
        sxx_type: Literal["cwt", "hilbert"] = "cwt",
        output_type: Literal["power", "frequency"] = "frequency",
        ref_type: Literal["none", "cmr", "car"] = "cmr",
        ref_probe: str = "all",
        map_channel=True,
        probe: str = "all",
        nperseg: int = 40,
        window: int = 100,
        start: int = 0,
        end: int = 0,
    ) -> dict[str, np.ndarray]:
        chan_dict = self.get_channel_clusters()
        chans = sorted(list(chan_dict.keys()))
        output_data = []
        for chan in chans:
            self.callback(f"Starting extraction for channel {chan}.")
            band_dict = self.get_sxx_freq_bands(
                sxx_type=sxx_type,
                output_type=output_type,
                freq_bands=freq_bands,
                channel=chan,
                ref_type=ref_type,
                ref_probe=ref_probe,
                map_channel=map_channel,
                probe=probe,
                start=start,
                end=end,
            )
            for cid in chan_dict[chan]:
                self.callback(f"Extracting spike power for cluster {cid}.")
                output = self.extract_spike_power_data(
                    power_dict=band_dict, cluster_id=cid, nperseg=nperseg, window=window
                )
                output["channel"] = [chan] * output["count"].size
                output["cluster_id"] = [cid] * output["count"].size
                output_data.append(output)
        output_data = concatenate_dicts(output_data)
        mean_data = self.lfp_mean_per_cluster(output_data, list(freq_bands.keys()))
        return output_data, mean_data

    def lfp_mean_per_cluster(self, input, freq_bands):
        cid = np.unique(input["cluster_id"])
        output_dict = {}
        for band in freq_bands:
            output_dict[band] = np.zeros((cid.size, input[band].shape[1]))
        for index, i in enumerate(cid):
            indexes = np.where(input["cluster_id"] == i)[0]
            for band in freq_bands:
                temp = input[band][indexes]
                output_dict[band][index] = temp.mean(axis=0)
        return output_dict

    # ------------------------------------------------------------------
    # Spike-free CSD pipeline (see ROADMAP.md, "Spike-free CSD for
    # spike-phase analysis"). Order matters:
    # raw wideband (ref_type="none"!) -> spike removal -> referencing if
    # desired -> LFP filter/downsample -> kCSD -> phase analysis.
    # Spike removal runs *before* CMR/CAR because the virtual reference
    # would otherwise remix spike transients into every channel.
    # ------------------------------------------------------------------

    def _probe_chans(self, probe: str) -> tuple[int, int]:
        chans = self.get_grp_dataset("probes", probe)
        return int(chans[0]), int(chans[1])

    def _relative_bounds(self, start: int, end: int) -> tuple[int, int]:
        """Relative [start, end) window indices (same convention as `acq`)."""
        rel0 = start if start > 0 else 0
        rel1 = end if end > 0 else self.end - self.start
        return rel0, rel1

    def _spike_filter_callable(self, fs):
        """Build a callable spike-band view for the residual detector."""
        fdict = self.get_filter("spike")
        if fdict["sample_rate"] == "None":
            fdict["sample_rate"] = fs

        def _band(x):
            return filter_array(
                x,
                sample_rate=fs,
                filter_type=fdict["filter_type"],
                order=fdict["order"],
                highpass=fdict["highpass"],
                high_width=fdict["high_width"],
                lowpass=fdict["lowpass"],
                low_width=fdict["low_width"],
                window=fdict["window"],
            )

        return _band

    def _chunks(self, rel0: int, rel1: int, chunk_size: int, pad: int):
        """Yield (cs, ce, rb, re): keep [cs, ce), read padded [rb, re)."""
        total = self.end - self.start
        for cs in range(rel0, rel1, chunk_size):
            ce = min(cs + chunk_size, rel1)
            rb = max(0, cs - pad)
            re_ = min(total, ce + pad)
            yield cs, ce, rb, re_

    def remove_spikes_wideband(
        self,
        probe: str = "all",
        start: int = 0,
        end: int = 0,
        ref_type: Literal["none", "cmr", "car"] = "none",
        method: spike_removal.RemovalMethod = "hybrid",
        interpolation_kind: spike_removal.InterpolationKind = "spatial",
        spatial_scale: spike_removal.SpatialScale = "per_channel",
        chunk_size: int = 240_000,
        detection_threshold: float = 6.0,
    ) -> np.ndarray:
        """Remove every detected spike (all Kilosort clusters) from the
        probe's wideband.

        The wideband is processed in chunks (each read padded by one template
        window so spikes straddling chunk boundaries are fully removed);
        subtraction uses all spikes on their template-covered channels
        (per-spike LSQ scaling), and a hybrid residual pass repairs anything
        left over. Use ``ref_type="none"`` (default): CMR/CAR traces contain
        spike transients and would remix them into clean channels.

        Parameters
        ----------
        probe, start, end : data selection (relative indices, as in `acq`).
        method, interpolation_kind, spatial_scale :
            See ``spike_lfp_functions.spike_removal.remove_spikes``.
        chunk_size : int
            Wideband samples per chunk (default 6 s at 40 kHz).

        Returns
        -------
        np.ndarray, (n_chans, n_rel_samples) float32
        """
        if not hasattr(self, "spike_times"):
            raise AttributeError("No Kilosort data loaded. Call load_kilosort first.")
        rel0, rel1 = self._relative_bounds(start, end)
        chans = self.get_grp_dataset("probes", probe)
        n_chans = int(chans[1] - chans[0])
        templates = np.asarray(self.sparse_templates, dtype=np.float64)
        window = templates.shape[1]
        n_time = rel1 - rel0
        output = np.empty((n_chans, n_time), dtype=np.float32)
        spike_times = np.asarray(self.spike_times)
        spike_idx = np.asarray(self.spike_templates)
        fs = int(self.get_file_dataset("sample_rate", rows=int(chans[0])))
        detect_on = None
        if method == "hybrid":
            detect_on = self._spike_filter_callable(fs)
        for index, (cs, ce, rb, re_) in enumerate(
            self._chunks(rel0, rel1, chunk_size, window)
        ):
            self.callback(f"Removing spikes from chunk {index + 1}: {cs}-{ce}")
            chunk = self.multichan_acq(
                "wideband",
                ref_type=ref_type,
                map_channel=False,
                probe=probe,
                start=rb,
                end=re_,
            )
            in_chunk = (spike_times >= cs) & (spike_times < ce)
            cleaned = spike_removal.remove_spikes(
                chunk,
                spike_times[in_chunk] - rb,
                spike_idx[in_chunk],
                templates,
                method=method,
                spatial_scale=spatial_scale,
                interpolation_kind=interpolation_kind,
                detect_on=detect_on,
                detection_threshold=detection_threshold,
            )
            keep0 = cs - rb
            keep1 = keep0 + (ce - cs)
            out0 = cs - rel0
            output[:, out0 : out0 + (ce - cs)] = cleaned[:, keep0:keep1]
        return output

    def get_clean_lfp(
        self,
        probe: str = "all",
        start: int = 0,
        end: int = 0,
        method: spike_removal.RemovalMethod = "hybrid",
        interpolation_kind: spike_removal.InterpolationKind = "spatial",
        spatial_scale: spike_removal.SpatialScale = "per_channel",
        chunk_size: int = 2_400_000,
        filter_pad_s: float = 3.0,
    ) -> np.ndarray:
        """Spike-free LFP at the stored 'lfp' filter settings, chunked for
        long recordings.

        Per chunk: read padded wideband -> remove spikes -> filter (the
        padding absorbs filter edge transients, which are discarded) ->
        downsample to the stored LFP rate. Peak memory stays proportional to
        the *output* size, so 10-minute, 64-channel recordings are practical.

        Returns
        -------
        np.ndarray, (n_chans, n_lfp_samples) float32
        """
        rel0, rel1 = self._relative_bounds(start, end)
        chans = self.get_grp_dataset("probes", probe)
        n_chans = int(chans[1] - chans[0])
        fs_wb = int(self.get_file_dataset("sample_rate", rows=int(chans[0])))
        fdict = self.get_filter("lfp")
        fs_lfp = fdict["sample_rate"]
        ratio = int(fs_wb // fs_lfp)
        pad = int(filter_pad_s * fs_wb)
        pad = ((pad + ratio - 1) // ratio) * ratio
        n_out = (rel1 - rel0 + ratio - 1) // ratio
        output = np.empty((n_chans, n_out), dtype=np.float32)
        templates = np.asarray(self.sparse_templates, dtype=np.float64)
        spike_times = np.asarray(self.spike_times)
        spike_idx = np.asarray(self.spike_templates)
        detect_on = None
        if method == "hybrid":
            detect_on = self._spike_filter_callable(fs_wb)
        for index, (cs, ce, rb, re_) in enumerate(
            self._chunks(rel0, rel1, chunk_size, pad)
        ):
            self.callback(f"Cleaning LFP chunk {index + 1}: {cs}-{ce}")
            chunk = self.multichan_acq(
                "wideband",
                ref_type="none",
                map_channel=False,
                probe=probe,
                start=rb,
                end=re_,
            )
            in_chunk = (spike_times >= cs) & (spike_times < ce)
            cleaned = spike_removal.remove_spikes(
                chunk,
                spike_times[in_chunk] - rb,
                spike_idx[in_chunk],
                templates,
                method=method,
                spatial_scale=spatial_scale,
                interpolation_kind=interpolation_kind,
                detect_on=detect_on,
            )
            band = filter_array(
                cleaned,
                sample_rate=fs_wb,
                filter_type=fdict["filter_type"],
                order=fdict["order"],
                highpass=fdict["highpass"],
                high_width=fdict["high_width"],
                lowpass=fdict["lowpass"],
                low_width=fdict["low_width"],
                window=fdict["window"],
            )
            ds = downsample(band, fs_wb, fs_lfp, fdict["up_sample"])
            keep0 = (cs - rb) // ratio
            keep1 = keep0 + (ce - cs) // ratio
            out0 = (cs - rel0) // ratio
            filled = out0 + (ce - cs) // ratio
            output[:, out0:filled] = ds[:, keep0:keep1]
        # Tail samples lost to rounding (if any) are dropped, not returned
        # uninitialized.
        return output[:, :filled]

    def set_csd_geometry(self, positions, probe: str = "all"):
        """Store the electrode geometry for CSD estimation.

        Parameters
        ----------
        positions : np.ndarray
            (n_chans, 1)/(n_chans,) depths -> KCSD1D (single-shank laminar
            probes); (n_chans, 2) (x, y) coordinates -> KCSD2D. Any consistent
            length unit (e.g. um); depths must follow channel order.
        """
        positions = np.asarray(positions, dtype=float)
        self.set_grp_dataset("csd", probe, positions)

    def get_csd_geometry(self, probe: str = "all"):
        return self.get_grp_dataset("csd", probe)

    def csd(
        self,
        probe: str = "all",
        start: int = 0,
        end: int = 0,
        lfp: np.ndarray | None = None,
        geometry: np.ndarray | None = None,
        sigma: float = 1.0,
        R_init: float | None = None,
        gdx: float | None = None,
        ext: float | None = None,
        n_src_init: int | None = None,
        lambd: float = 1e-8,
        rows_at: Literal["electrodes", "grid"] = "electrodes",
        clean_kwargs: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Current source density of the spike-free LFP via kCSD.

        Uses the vectorized kCSD port (`invivosuite.csd`): KCSD1D for 1D
        geometries (laminar probes), KCSD2D for planar (x, y) geometries. The
        estimation operator is applied as a single matmul, so long recordings
        cost one BLAS pass. By default the CSD is evaluated at the estimation
        grid points nearest each electrode (one trace per channel; ideal for
        spike-phase analysis); ``rows_at="grid"`` returns the full grid.

        Parameters
        ----------
        lfp : np.ndarray | None
            Precomputed spike-free LFP (skips cleaning; e.g. reuse
            `get_clean_lfp` output across analyses). Defaults to calling
            `get_clean_lfp` with ``clean_kwargs``.
        geometry : np.ndarray | None
            Electrode geometry; defaults to `get_csd_geometry(probe)`
            (call `set_csd_geometry` first).
        sigma : float
            Conductivity (only scales the CSD magnitude).
        R_init, gdx, ext, n_src_init : floats/int
            kCSD parameters; each defaults from the electrode spacing
            (R = 2*spacing, gdx = spacing/2, ext = 2*spacing,
            n_src_init = 300 for 1D / 1000 for 2D).
        rows_at : {'electrodes', 'grid'}

        Returns
        -------
        csd : np.ndarray
            (n_chans, n_time_lfp) at the nearest grid points for
            'electrodes'; (n_estm, n_time_lfp) for 'grid'.
        info : dict
           {'row_indices', 'estm_points', 'geometry', 'kcsd'}.
        """
        if lfp is None:
            lfp = self.get_clean_lfp(
                probe=probe, start=start, end=end, **(clean_kwargs or {})
            )
        if geometry is None:
            geometry = self.get_csd_geometry(probe)
        geometry = np.asarray(geometry, dtype=float)
        pots = np.asarray(lfp, dtype=float)
        if geometry.shape[0] != pots.shape[0]:
            raise ValueError(
                f"geometry has {geometry.shape[0]} positions but LFP has "
                f"{pots.shape[0]} channels"
            )
        spacing = float(np.median(np.diff(np.unique(geometry))))
        R_init = 2 * spacing if R_init is None else R_init
        gdx = spacing / 2 if gdx is None else gdx
        ext = 2 * spacing if ext is None else ext
        if geometry.ndim == 1 or geometry.shape[1] == 1:
            kcsd = KCSD1D(
                geometry.ravel()[:, np.newaxis],
                pots,
                sigma=sigma,
                R_init=R_init,
                gdx=gdx,
                ext_x=ext,
                n_src_init=n_src_init or 300,
                lambd=lambd,
            )
        elif geometry.shape[1] == 2:
            gdy = gdx
            kcsd = KCSD2D(
                geometry,
                pots,
                sigma=sigma,
                R_init=R_init,
                gdx=gdx,
                gdy=gdy,
                ext_x=ext,
                ext_y=ext,
                n_src_init=n_src_init or 1000,
                lambd=lambd,
            )
        else:
            raise ValueError("geometry must be (n, 1) or (n, 2)")
        if rows_at == "electrodes":
            query = geometry if geometry.ndim > 1 else geometry[:, np.newaxis]
            query = query.reshape(-1, kcsd.dim)
            rows, estm_points = kcsd.nearest_rows(query)
            traces = kcsd.apply(pots, rows=rows)
        else:
            traces = kcsd.estimate(pots=pots)
            estm_points = kcsd.estm_points()
            rows = None
        info = {
            "row_indices": rows,
            "estm_points": estm_points,
            "geometry": geometry,
            "kcsd": kcsd,
        }
        return traces, info

    def csd_spike_phase(
        self,
        freq_bands: dict[str, Iterable],
        nperseg: int = 40,
        probe: str = "all",
        start: int = 0,
        end: int = 0,
        csd_kwargs: dict | None = None,
        sxx_attrs: dict | None = None,
    ) -> tuple[list[dict], list[dict]]:
        """Spike-triggered phase of the spike-free CSD (the decontaminated
        companion of `cwt_spike_phase`).

        Wideband fs / nperseg is the phase-analysis rate (same convention as
        `cwt_spike_phase`: spike sample indices ``// nperseg`` index the
        downsampled traces). The CWT is computed on the CSD trace at each
        cluster's channel — gamma-band spike phase without spike bleed-
        through (see the spike_removal module docstring for why removal
        precedes CSD).

        Returns
        -------
        (per_frequency_stats, band_stats) : lists of dicts, one entry per
            cluster x frequency / band, mirroring `cwt_spike_phase`.
        """
        traces, info = self.csd(
            probe=probe, start=start, end=end, **(csd_kwargs or {})
        )
        attrs = sxx_attrs or self.get_grp_attrs("cwt")
        chans = self.get_grp_dataset("probes", probe)
        fs_wb = int(self.get_file_dataset("sample_rate", rows=int(chans[0])))
        fdict = self.get_filter("lfp")
        fs_lfp = int(fdict["sample_rate"])
        analysis_fs = fs_wb / nperseg
        if fs_lfp > analysis_fs:
            traces = downsample(traces, fs_lfp, analysis_fs, fdict["up_sample"])
        elif fs_lfp != analysis_fs:
            raise ValueError(
                "LFP sample rate lower than the requested analysis rate; "
                "increase the stored lfp sample_rate or lower nperseg"
            )
        # NOTE: unlike cwt_spike_phase (which passes the wideband fs to the
        # Wavelet), the Wavelet here uses the *analysis* rate so the
        # frequency axis is labeled correctly.
        w = Wavelet(analysis_fs, imaginary=attrs["imaginary"])
        f = Frequencies(
            w,
            attrs["f0"],
            attrs["f1"],
            attrs["fn"],
            analysis_fs,
            attrs["scaling"],
        )
        pyf = PyFCWT(
            w,
            f,
            attrs["nthreads"],
            dtype=attrs["dtype"],
            norm=attrs["scaling"],
        )
        chan_dict = self.get_channel_clusters()
        per_frequency = []
        band_stats = []
        for chan in sorted(chan_dict.keys()):
            row = int(chan - chans[0])
            trace = np.ascontiguousarray(traces[row], dtype=np.float32)
            cwt = pyf.cwt(trace)
            for cid in chan_dict[chan]:
                self.callback(f"Extracting CSD spike phase for cluster {cid}.")
                spike_times = self.get_cluster_spike_times(cid) // nperseg
                temp = np.angle(cwt[:, spike_times])
                stats = Parallel(n_jobs=4, prefer="threads")(
                    delayed(analyze_spike_phase)(temp[i, :])
                    for i in range(temp.shape[0])
                )
                for stat, freq in zip(stats, f.f):
                    stat["channel"] = chan
                    stat["cluster_id"] = cid
                    stat["frequency"] = freq
                    per_frequency.append(stat)
                min_f, max_f = f.f.max(), f.f.min()
                for key, value in freq_bands.items():
                    min_f = min(min_f, value[0])
                    max_f = max(max_f, value[1])
                    output = cwt_phase_best_frequency(value[0], value[1], f.f, temp)
                    output["band"] = key
                    output["cluster_id"] = cid
                    output["channel"] = chan
                    band_stats.append(output)
                output = cwt_phase_best_frequency(min_f, max_f, f.f, temp)
                output["band"] = "all"
                output["cluster_id"] = cid
                output["channel"] = chan
                band_stats.append(output)
        return per_frequency, band_stats





