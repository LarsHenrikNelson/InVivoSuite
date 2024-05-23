from collections.abc import Iterable, Callable
from typing import Literal

import numpy as np

from ..utils import concatenate_dicts, expand_data


class SpkLFPManager:
    def spike_lfp_phase(
        self,
        freq_bands: dict[str, Iterable],
        ref: bool = False,
        ref_type: Literal["cmr", "car"] = "cmr",
        ref_probe: str = "all",
        map_channel=True,
        probe: str = "all",
        nperseg: int = 40,
        callback: Callable = print,
    ):
        chan_dict = self.get_channel_clusters()
        chans = sorted(list(chan_dict.keys()))
        output_data = []
        total_len = 0
        for chan in chans:
            freqs, cwt = self.sxx(
                acq_num=chan,
                sxx_type="cwt",
                ref=ref,
                ref_type=ref_type,
                ref_probe=ref_probe,
                map_channel=map_channel,
                probe=probe,
            )
            for cid in chan_dict[chan]:
                callback(f"Extracting spike phase for cluster {cid}.")
                b_spks = self.get_binned_spike_cluster(cid, nperseg=nperseg)
                temp_dict = {}
                spk_indexes = np.where(b_spks > 0)[0]
                temp_dict["cluster_id"] = [cid] * spk_indexes.size
                temp_dict["channel"] = [chan] * spk_indexes.size
                temp_dict["count"] = b_spks[spk_indexes]
                total_len += spk_indexes.size
                for b_name, fr in freq_bands.items():
                    gamma_ind = np.where((freqs > fr[0]) & (freqs < fr[1]))[0]
                    cwt_band = cwt[gamma_ind]
                    band_phase = np.angle(cwt_band.mean(axis=0))
                    temp_dict[b_name] = band_phase[spk_indexes]
                output_data.append(temp_dict)
        output_data = concatenate_dicts(output_data)
        c_size = output_data["count"].size
        n_size = output_data["count"].sum()
        output_data = expand_data(
            data=output_data, column="count", current_size=c_size, new_size=n_size
        )
        return output_data
