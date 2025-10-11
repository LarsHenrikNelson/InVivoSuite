from collections.abc import Iterable
from typing import Literal, Optional

import numpy as np
from joblib import Parallel, delayed

from ..functions.filter_functions import downsample
from ..functions.spike_lfp_functions.spike_phase import (
    analyze_spike_phase,
    extract_spike_phase_data,
    cwt_phase_best_frequency,
)
from ..functions.spike_lfp_functions.spike_power import spike_triggered_lfp
from ..spectral import Frequencies, PyFCWT, Wavelet, get_freq_window, multitaper
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
                    freq_output = cwt_phase_best_frequency(value[0], value[1], f.f, temp)
                    freq_output["key"] = key
                    freq_output["cluster_id"] = cid
                    freq_output["channel"] = chan
                    band_specific_phase.append(freq_output)
                freq_output = cwt_phase_best_frequency(min_f, max_f, f.f, temp)
                freq_output["key"] = "all"
                freq_output["cluster_id"] = cid
                freq_output["channel"] = chan
                band_specific_phase.append(freq_output)
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
