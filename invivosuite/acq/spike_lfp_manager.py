from collections.abc import Callable, Iterable
from typing import Literal, TypedDict

import numpy as np

from ..functions.spike_lfp_functions.circular_stats import (
    h_test,
    mean_vector_length,
    periodic_mean_std,
    rayleightest,
)
from ..functions.spike_lfp_functions.spike_power import spike_triggered_lfp
from ..utils import concatenate_dicts, expand_data

# TODO:  Break down spike-phase and spike-power into smaller components.


class CircStats(TypedDict):
    rayleigh_pval: float
    circ_mean: float
    circ_std: float
    h: float
    m: float
    fpp: float


class SpkLFPManager:

    def get_cwt_freq_bands(
        self,
        freq_bands: dict[str, Iterable],
        channel: int,
        ref: bool = False,
        ref_type: Literal["cmr", "car"] = "cmr",
        ref_probe: str = "all",
        map_channel=True,
        probe: str = "all",
    ) -> dict[str, np.ndarray]:
        freqs, cwt = self.sxx(
            channel=channel,
            sxx_type="cwt",
            ref=ref,
            ref_type=ref_type,
            ref_probe=ref_probe,
            map_channel=map_channel,
            probe=probe,
        )

        band_dict = {}
        for b_name, fr in freq_bands.items():
            band_ind = np.where((freqs > fr[0]) & (freqs < fr[1]))[0]
            cwt_band = cwt[band_ind]
            band_dict[b_name] = cwt_band.mean(axis=0)
        return band_dict

    def get_hilbert_freq_bands(
        self,
        freq_bands: dict[str, Iterable],
        channel: int,
        ref: bool = False,
        ref_type: Literal["cmr", "car"] = "cmr",
        ref_probe: str = "all",
        map_channel=True,
        probe: str = "all",
    ) -> dict[str, np.ndarray]:
        band_dict = {}
        for b_name, fr in freq_bands.items():
            band_dict[b_name] = self.hilbert(
                channel=channel,
                ref=ref,
                ref_type=ref_type,
                ref_probe=ref_probe,
                map_channel=map_channel,
                probe=probe,
                highpass=fr[0],
                lowpass=fr[1],
                filter_type="butterworth_zero",
                order=4,
            )
        return band_dict

    def get_sxx_freq_bands(
        self,
        output_type: Literal["phase", "power", "frequency"],
        freq_bands: dict[str, Iterable],
        sxx_type: Literal["cwt", "hilbert"],
        channel: int,
        ref: bool = False,
        ref_type: Literal["cmr", "car"] = "cmr",
        ref_probe: str = "all",
        map_channel=True,
        probe: str = "all",
    ) -> dict[str, np.ndarray]:
        if sxx_type == "cwt":
            band_dict = self.get_cwt_freq_bands(
                freq_bands=freq_bands,
                channel=channel,
                ref=ref,
                ref_type=ref_type,
                ref_probe=ref_probe,
                map_channel=map_channel,
                probe=probe,
            )
        else:
            band_dict = self.get_hilbert_freq_bands(
                freq_bands=freq_bands,
                channel=channel,
                ref=ref,
                ref_type=ref_type,
                ref_probe=ref_probe,
                map_channel=map_channel,
                probe=probe,
            )
        for key, value in band_dict.items():
            if output_type == "phase":
                band_dict[key] = np.angle(value)
            elif output_type == "power":
                band_dict[key] = np.abs(value)
            elif output_type == "frequency":
                band_dict[key] = value.real
            else:
                raise AttributeError(
                    "output_type not recognized. Must be phase or power."
                )
        return band_dict

    def analyze_spike_phase(self, data: np.ndarray) -> CircStats:
        cm, stdev = periodic_mean_std(data)
        h, m, fpp = h_test(data)
        p = rayleightest(data)
        vlen, vp = mean_vector_length(data)

        stats = CircStats(
            rayleigh_pval=p, circ_mean=cm, circ_std=stdev, h=h, m=m, fpp=fpp
        )
        return stats

    def extract_spike_phase_data(
        self, phase_dict: dict[str, np.ndarray], cluster_id: int, nperseg: int
    ) -> dict[str, np.ndarray]:
        b_spks = self.get_binned_spike_cluster(cluster_id, nperseg=nperseg)
        output_dict = {}
        output_stats = {}
        spk_indexes = np.where(b_spks > 0)[0]
        output_dict["cluster_id"] = [cluster_id] * spk_indexes.size
        for b_name, phase in phase_dict.items():
            b_phases = phase[spk_indexes]
            b_phases = expand_data(b_phases, b_spks[spk_indexes])
            output_dict[b_name] = b_phases
            stats = self.analyze_spike_phase(b_phases)
            output_stats.update(
                {f"{b_name}_{key}": value for key, value in stats.items()}
            )
        output_stats["cluster_id"] = cluster_id
        output_dict["cluster_id"] = np.full(output_dict[b_name].size, cluster_id)
        return (output_stats, output_dict)

    def spike_phase(
        self,
        freq_bands: dict[str, Iterable],
        sxx_type: Literal["cwt", "hilbert"],
        ref: bool = False,
        ref_type: Literal["cmr", "car"] = "cmr",
        ref_probe: str = "all",
        map_channel=True,
        probe: str = "all",
        nperseg: int = 40,
        callback: Callable = print,
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
                ref=ref,
                ref_type=ref_type,
                ref_probe=ref_probe,
                map_channel=map_channel,
                probe=probe,
            )
            for cid in chan_dict[chan]:
                callback(f"Extracting spike phase for cluster {cid}.")
                stats, phases = self.extract_spike_phase_data(band_dict, cid, nperseg)
                phases["channel"] = np.full(phases["cluster_id"].size, chan)
                stats["channel"] = chan
                stats["cluster_id"] = cid
                analyzed_spk_phase.append(stats)
                output_data.append(phases)
        output_data = concatenate_dicts(output_data)
        analyzed_spk_phase = concatenate_dicts(analyzed_spk_phase)
        return output_data, analyzed_spk_phase

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
        ref: bool = False,
        ref_type: Literal["cmr", "car"] = "cmr",
        ref_probe: str = "all",
        map_channel=True,
        probe: str = "all",
        nperseg: int = 40,
        window: int = 100,
        callback: Callable = print,
    ) -> dict[str, np.ndarray]:
        chan_dict = self.get_channel_clusters()
        chans = sorted(list(chan_dict.keys()))
        output_data = []
        for chan in chans:
            callback(f"Starting extraction for channel {chan}.")
            band_dict = self.get_sxx_freq_bands(
                sxx_type=sxx_type,
                output_type=output_type,
                freq_bands=freq_bands,
                channel=chan,
                ref=ref,
                ref_type=ref_type,
                ref_probe=ref_probe,
                map_channel=map_channel,
                probe=probe,
            )
            for cid in chan_dict[chan]:
                callback(f"Extracting spike power for cluster {cid}.")
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
