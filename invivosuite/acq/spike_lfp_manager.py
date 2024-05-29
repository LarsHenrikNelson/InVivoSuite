from collections.abc import Iterable, Callable
from typing import Literal, TypedDict

import numpy as np

from .spike_lfp_functions.circular_stats import h_test, rayleightest, periodic_mean_std
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

    def get_cwt_phase(
        self,
        freq_bands: dict[str, Iterable],
        chan: int,
        ref: bool = False,
        ref_type: Literal["cmr", "car"] = "cmr",
        ref_probe: str = "all",
        map_channel=True,
        probe: str = "all",
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        freqs, cwt = self.sxx(
            acq_num=chan,
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
            band_dict[b_name] = np.angle(cwt_band.mean(axis=0))
        return band_dict

    def analyze_spike_phase(self, data: np.ndarray) -> CircStats:
        cm, stdev = periodic_mean_std(data)
        h, m, fpp = h_test(data)
        p = rayleightest(data)

        stats = CircStats(
            rayleigh_pval=p, circ_mean=cm, circ_std=stdev, h=h, m=m, fpp=fpp
        )
        return stats

    def extract_spike_phase_data(
        self, phase_dict: dict[str, np.ndarray], cluster_id: int, nperseg: int
    ):
        b_spks = self.get_binned_spike_cluster(cluster_id, nperseg=nperseg)
        output_dict = {}
        output_stats = {}
        spk_indexes = np.where(b_spks > 0)[0]
        output_dict["cluster_id"] = [cluster_id] * spk_indexes.size
        output_dict["count"] = b_spks[spk_indexes]
        for b_name, phase in phase_dict.items():
            b_phases = phase[spk_indexes]
            output_dict[b_name] = b_phases
            stats = self.analyze_spike_phase(b_phases)
            output_stats.update(
                {f"{b_name}_{key}": value for key, value in stats.items()}
            )
        output_stats["cluster_id"] = cluster_id
        return (output_stats, output_dict)

    def spike_phase(
        self,
        freq_bands: dict[str, Iterable],
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
            band_dict = self.get_cwt_phase(
                freq_bands=freq_bands,
                chan=chan,
                ref=ref,
                ref_type=ref_type,
                ref_probe=ref_probe,
                map_channel=map_channel,
                probe=probe,
            )
            for cid in chan_dict[chan]:
                callback(f"Extracting spike phase for cluster {cid}.")
                stats, phases = self.extract_spike_phase_data(band_dict, cid, nperseg)
                phases["channel"] = [chan] * phases["count"].size
                phases["cluster_id"] = [chan] * phases["count"].size
                stats["channel"] = chan
                stats["cluster_id"] = cid
                analyzed_spk_phase.append(stats)
                output_data.append(phases)
        output_data = concatenate_dicts(output_data)
        analyzed_spk_phase = concatenate_dicts(analyzed_spk_phase)
        c_size = output_data["count"].size
        n_size = output_data["count"].sum()
        output_data = expand_data(
            data=output_data, column="count", current_size=c_size, new_size=n_size
        )
        return output_data, analyzed_spk_phase
