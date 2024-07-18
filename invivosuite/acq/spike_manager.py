from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
from send2trash import send2trash

from ..utils import concatenate_dicts, save_tsv
from ..functions.spike_functions import (
    _sttc_sig,
    _template_channels,
    amplitude_cutoff,
    bin_spikes,
    create_binary_spikes,
    create_continuous_spikes,
    get_burst_data,
    get_template_channels,
    isi_violations,
    max_int_bursts,
    Methods,
    presence,
    rb_violations,
    sfa_abi,
    sfa_divisor,
    sfa_local_var,
    sfa_rlocal_var,
    sttc,
    sttc_ele,
    sttc_python,
    template_properties,
    Windows,
)

Callback = Callable[[str], None]


class SpkManager:
    def load_ks_data(self, load_type: str = "r+"):
        self._load_sparse_templates(load_type=load_type)
        self._load_spike_templates(load_type=load_type)
        self._load_spike_clusters(load_type=load_type)
        self._load_spike_times(load_type=load_type)
        self._load_amplitudes(load_type=load_type)
        self._load_spike_waveforms(load_type=load_type)
        self._load_accepted_units(load_type=load_type)

    def _load_amplitudes(self, load_type: str = "r+"):
        if load_type == "memory":
            self.amplitudes = np.array(
                np.load(self.ks_directory / "amplitudes.npy", "r").flatten()
            )
        else:
            self.amplitudes = np.load(
                self.ks_directory / "amplitudes.npy", load_type
            ).flatten()

    def _remove_file(self, file_ending):
        temp_path = Path(self.ks_directory / file_ending)
        if temp_path.exists():
            send2trash(str(temp_path))

    def _load_sparse_templates(self, load_type: str = "r+"):
        temp_path = self.ks_directory / "templates.npy"
        if temp_path.exists():
            if load_type == "memory":
                self.sparse_templates = np.array(
                    np.load(self.ks_directory / "templates.npy", "r")
                )
            else:
                self.sparse_templates = np.load(
                    self.ks_directory / "templates.npy", load_type
                )
        else:
            self.sparse_templates = np.zeros((0, 0, 0))

    def _load_spike_templates(self, load_type: str = "r+"):
        if load_type == "memory":
            self.spike_templates = np.array(
                np.load(self.ks_directory / "spike_templates.npy", "r").flatten()
            )
        else:
            self.spike_templates = np.load(
                self.ks_directory / "spike_templates.npy", load_type
            ).flatten()
        self.template_ids = np.unique(self.spike_templates)

    def _load_spike_clusters(self, load_type: str = "r+"):
        if load_type == "memory":
            self.spike_clusters = np.array(
                np.load(self.ks_directory / "spike_clusters.npy", "r").flatten()
            )
        else:
            self.spike_clusters = np.load(
                self.ks_directory / "spike_clusters.npy", load_type
            ).flatten()
        self.cluster_ids = np.unique(self.spike_clusters)

    def _create_chan_clusters(self):
        self.chan_clusters = defaultdict(list)
        for cluster, chan in zip(self.cluster_ids, self.cluster_channels):
            self.chan_clusters[chan].append(cluster)

    def _load_spike_times(self, load_type: str = "r+"):
        if load_type == "memory":
            self.spike_times = np.array(
                np.load(self.ks_directory / "spike_times.npy", "r").flatten()
            )
        else:
            self.spike_times = np.load(
                self.ks_directory / "spike_times.npy", load_type
            ).flatten()

    def _load_spike_waveforms(self, load_type: str = "r+"):
        temp_path = self.ks_directory / "_phy_spikes_subset.waveforms.npy"
        if temp_path.exists():
            if load_type == "memory":
                self.spike_waveforms = np.array(np.load(temp_path, "r"))
            else:
                self.spike_waveforms = np.load(temp_path, load_type)
        else:
            self.spike_waveforms = np.zeros((0, 0, 0))
            np.save(temp_path, self.spike_waveforms)
            self._load_spike_waveforms()

    def _load_accepted_units(self, load_type: str = "r+"):
        temp_path = self.ks_directory / "accepted_units.npy"
        if temp_path.exists():
            if load_type == "memory":
                self.accepted_units = np.array(np.load(temp_path, "r"))
            else:
                self.accepted_units = np.load(temp_path, load_type)
        else:
            self.accepted_units = np.zeros(self.cluster_ids.size, dtype=bool)
            self.accepted_units[:] = 1
            np.save(temp_path, self.accepted_units)

        if self.accepted_units.size != self.cluster_ids.size:
            self.accepted_units = np.zeros(self.cluster_ids.size, dtype=bool)
            self.accepted_units[:] = 1
            np.save(temp_path, self.accepted_units)

    def get_cluster_spike_ids(self, cluster_id: int) -> np.ndarray:
        return np.where(self.spike_clusters == cluster_id)[0]

    def get_cluster_templates(self, cluster_id: int) -> np.ndarray:
        cid = np.where(self.spike_clusters == cluster_id)[0]
        return np.unique(self.spike_templates[cid])

    def get_cluster_template_waveforms(self, cluster_id: int) -> np.ndarray:
        return self.sparse_self.sparse_templates[cluster_id, :, :]

    def get_cluster_best_template_waveform(
        self, cluster_id: int, nchans: int = 4
    ) -> np.ndarray:
        template_index = np.where(self.cluster_ids == cluster_id)[0]
        if len(template_index) == 1:
            chan, _, _ = _template_channels(
                self.sparse_templates[template_index[0]],
                nchans=nchans,
                total_chans=self.sparse_templates.shape[2],
            )
            return self.sparse_templates[template_index, :, chan]
        else:
            raise AttributeError(f"Cluster id {cluster_id} is not valid.")

    def get_cluster_spike_times(
        self,
        cluster_id: int,
        fs: Optional[int] = None,
        output_type: Literal["sec", "ms", "samples"] = "samples",
    ) -> np.ndarray:
        spike_ids = np.where(self.spike_clusters == cluster_id)[0]
        if fs is None or output_type == "samples":
            return self.spike_times[spike_ids]
        if output_type == "ms":
            return self.spike_times[spike_ids].flatten() / (fs / 1000)
        elif output_type == "sec":
            return self.spike_times[spike_ids].flatten() / fs

    def get_cluster_spike_amplitudes(self, cluster_id: int) -> np.ndarray:
        spike_ids = np.where(self.spike_clusters == cluster_id)[0]
        return self.amplitudes[spike_ids]

    def get_binary_spike_cluster(
        self,
        cluster_id: int,
    ) -> np.ndarray:
        spike_ids = np.where(self.spike_clusters == cluster_id)[0]
        spike_indexes = self.spike_times[spike_ids]
        return create_binary_spikes(spike_indexes, self.end - self.start)

    def get_continuous_spike_cluster(
        self,
        cluster_id: int,
        fs: float = 40000.0,
        nperseg: int = 0,
        window: Windows = "boxcar",
        sigma: float = 200,
        method: Methods = "convolve",
    ):
        spike_ids = np.where(self.spike_clusters == cluster_id)[0]
        spike_indexes = self.spike_times[spike_ids]
        return create_continuous_spikes(
            spike_indexes,
            self.end - self.start,
            nperseg=nperseg,
            fs=fs,
            window=window,
            sigma=sigma,
            method=method,
        )

    def get_reconstucted_cluster(self, cluster_id: int) -> np.ndarray:
        spike_ids = np.where(self.spike_clusters == cluster_id)[0]
        sample_indexes = self.spike_times[spike_ids]
        output_array = np.zeros(self.end.self.start)
        template = self.get_cluster_best_template_waveform(cluster_id=cluster_id)
        template -= template[0, 0]
        start_len = np.min(template) - 1
        end_len = template.size - start_len
        for i in sample_indexes:
            start = int(i - start_len)
            end = int(i + end_len)
            output_array[start:end] = template
        return output_array

    def get_binned_spike_cluster(self, cluster_id: int, nperseg: int) -> np.ndarray:
        spike_ids = np.where(self.spike_clusters == cluster_id)[0]
        sample_indexes = self.spike_times[spike_ids]
        return bin_spikes(sample_indexes, self.end - self.start, nperseg)

    def get_channel_clusters(self) -> dict[str, list[int]]:
        channel_dict = defaultdict(list)
        for temp_index in range(self.cluster_ids.size):
            cid = self.cluster_ids[temp_index]
            template = self.sparse_templates[temp_index, :, :]
            best_chan = np.argmax(np.sum(np.abs(template), axis=0))
            channel_dict[best_chan].append(cid)
        return channel_dict

    def get_cluster_channels(self) -> np.ndarray:
        channels = np.zeros(self.cluster_ids.size, dtype=int)
        for temp_index in range(self.cluster_ids.size):
            template = self.sparse_templates[temp_index, :, :]
            best_chan = np.argmax(np.sum(np.abs(template), axis=0))
            channels[temp_index] = best_chan
        return channels

    def get_cluster_channel(self, cluster_id: int) -> int:
        temp_index = np.where(self.cluster_ids == cluster_id)[0][0]
        template = self.sparse_templates[temp_index, :, :]
        return np.argmax(np.sum(np.abs(template), axis=0))

    def get_all_binary_spikes(
        self, channel: Optional[int] = None, dt: int = 0
    ) -> np.ndarray:
        chan_cid_dict = self.get_channel_clusters()
        chans = sorted(list(chan_cid_dict.keys()))
        output = np.zeros(
            (self.cluster_ids.size, self.end - self.start), dtype=np.int16
        )
        index = 0
        if channel is not None:
            chans = [channel]
        for chan in chans:
            for cid in chan_cid_dict[chan]:
                output[index] = self.get_binary_spike_cluster(cid, dt=dt)
                index += 1
        return output

    def get_binned_spikes_channel(
        self, nperseg: int, channel: Optional[int] = None
    ) -> np.ndarray:
        chan_cid_dict = self.get_channel_clusters()
        chans = sorted(list(chan_cid_dict.keys()))
        length = (self.end - self.start) // nperseg
        index = 0
        if channel is not None:
            chans = [channel]
            output = np.zeros((len(chan_cid_dict[channel]), length), dtype=np.int16)
        else:
            output = np.zeros((self.cluster_ids.size, length), dtype=np.int16)
        for chan in chans:
            for cid in chan_cid_dict[chan]:
                output[index] = self.get_binned_spike_cluster(cid, nperseg=nperseg)
                index += 1
        return output

    def get_cluster_spike_properties(
        self,
        cluster_id: int,
        fs: int = 40000,
        start: int = -1,
        end: int = -1,
        isi_threshold: float = 0.0015,
        R: Optional[float] = 0.005,
        min_isi: float = 0.0,
        nperseg: int = 40,
        output_type: Literal["sec", "ms"] = "sec",
    ):
        if start == -1:
            start = self.index_to_time(self.start, fs=fs, output_type=output_type)
        if end == -1:
            end = self.index_to_time(self.end, fs=fs, output_type=output_type)

        output_dict = {}
        times = self.get_cluster_spike_times(cluster_id, fs=fs, output_type=output_type)
        amps = self.get_cluster_spike_amplitudes(cluster_id)
        binned = self.get_binned_spike_cluster(cluster_id=cluster_id, nperseg=nperseg)
        output_dict["fano_factor"] = binned.var() / binned.mean()

        if times.size > 2:
            isi = np.diff(times)
            mean_isi = np.mean(isi)
            output_dict["iei"] = mean_isi
            if output_type == "sec":
                output_dict["fr_iei"] = 1 / mean_isi
            else:
                output_dict["fr_iei"] = 1000 / mean_isi
            output_dict["fr"] = times.size / (end - start)
            output_dict["abi_sfa"] = sfa_abi(isi)
            output_dict["local_sfa"] = sfa_local_var(isi)
            output_dict["rlocal_sfa"] = sfa_rlocal_var(isi, R)
            output_dict["divisor_sfa"] = sfa_divisor(isi)
            fpRate, nv = isi_violations(
                spike_train=times,
                min_time=start,
                max_time=end,
                isi_threshold=isi_threshold,
                min_isi=min_isi,
            )
            output_dict["fp_rate"] = fpRate
            output_dict["num_violations"] = nv
            rp_contam, _ = rb_violations(
                spike_train=times,
                min_time=start,
                max_time=end,
                isi_threshold=isi_threshold,
                min_isi=min_isi,
            )
            output_dict["rp_contam"] = rp_contam

        else:
            output_dict["iei"] = 0
            output_dict["fr"] = 0
            output_dict["fr_iei"] = 0
            output_dict["fp_rate"] = np.nan
            output_dict["num_violations"] = np.nan
            output_dict["abi_sfa"] = np.nan
            output_dict["local_sfa"] = np.nan
            output_dict["rlocal_sfa"] = np.nan
            output_dict["divisor_sfa"] = np.nan
            output_dict["rp_contam"] = 1.0
        output_dict["n_spikes"] = times.size
        if output_type == "sec":
            divisor = fs
        else:
            divisor = fs / 1000
        pr = presence(times, self.start / divisor, self.end / divisor)
        output_dict["presence_ratio"] = pr["presence_ratio"]
        cutoff = amplitude_cutoff(amps)
        output_dict["amp_cutoff"] = cutoff
        return output_dict

    def get_spikes_properties(
        self,
        fs: int = 40000,
        start: int = -1,
        end: int = -1,
        isi_threshold: float = 0.0015,
        R: Optional[float] = 0.005,
        min_isi: float = 0.0,
        nperseg: int = 40,
        output_type: Literal["sec", "ms"] = "sec",
    ):
        if start == -1:
            start = self.index_to_time(self.start, fs=fs, output_type=output_type)
        if end == -1:
            end = self.index_to_time(self.end, fs=fs, output_type=output_type)

        output_list = []

        for clust_id in self.cluster_ids:
            data = self.get_cluster_spike_properties(
                cluster_id=clust_id,
                fs=fs,
                start=start,
                end=end,
                R=R,
                isi_threshold=isi_threshold,
                min_isi=min_isi,
                nperseg=nperseg,
                output_type=output_type,
            )
            data["cluster_id"] = clust_id
            output_list.append(data)

        output_list = concatenate_dicts(output_list)
        return output_list

    def get_templates_properties(
        self,
        templates: np.ndarray,
        center: int = 41,
        nchans: int = 4,
        total_chans: int = 64,
        upsample_factor: int = 2,
        negative: bool = True,
    ):
        t_props_list = []
        for temp_index in range(self.cluster_ids.size):
            cluster_id = self.cluster_ids[temp_index]
            chan, start_chan, _ = _template_channels(
                templates[temp_index, :, :], nchans=nchans, total_chans=total_chans
            )
            t_props = template_properties(
                templates[temp_index, :, chan],
                center=center,
                upsample_factor=upsample_factor,
                negative=negative,
            )
            indexes = np.where(self.spike_clusters == cluster_id)[0]
            temp_spikes_waveforms = self.spike_waveforms[indexes]
            template_stdev = np.mean(
                np.std(temp_spikes_waveforms, axis=0)[:, int(chan - start_chan)]
            )
            t_props["cluster_id"] = cluster_id
            t_props["channel"] = chan
            t_props["stdev"] = template_stdev
            t_props_list.append(t_props)
        t_props = concatenate_dicts(t_props_list)
        return t_props

    def get_burst_properties(
        self,
        min_count: int = 5,
        min_dur: float = 0.01,
        max_start: float = 0.170,
        max_int: float = 0.3,
        max_end: float = 0.34,
        R: float = 0.005,
        output_type: Literal["sec", "ms", "sample"] = "sec",
        fs: Union[float, int] = 40000,
    ):
        props_list = []
        mean_list = []
        for clust_id in self.cluster_ids:
            spk_times = self.get_cluster_spike_times(clust_id, output_type="sec", fs=fs)
            b_data = max_int_bursts(
                spk_times,
                fs,
                min_count=min_count,
                min_dur=min_dur,
                max_start=max_start,
                max_int=max_int,
                max_end=max_end,
                output_type=output_type,
            )
            props_dict, burst_dict = get_burst_data(b_data, R)
            burst_dict["cluster_id"] = clust_id
            mean_list.append(burst_dict)
            props_dict["cluster_id"] = np.array([clust_id] * len(b_data))
            props_list.append(props_dict)
        mean_list = concatenate_dicts(mean_list)
        props_list = concatenate_dicts(props_list)
        return props_list, mean_list

    def get_properties(
        self,
        templates: Optional[np.ndarray] = None,
        fs: int = 40000,
        center=41,
        nchans: int = 4,
        total_chans: int = 64,
        upsample_factor: int = 2,
        isi_threshold: float = 0.0015,
        nperseg: int = 40,
        min_isi: float = 0.0,
        min_count: int = 5,
        min_dur: float = 0.01,
        max_start: float = 0.170,
        max_int: float = 0.3,
        max_end: float = 0.34,
        output_type: Literal["sec", "ms", "sample"] = "sec",
    ) -> dict:
        if templates is None:
            templates = self.sparse_templates
        output_dict = {}
        temp_props = self.get_templates_properties(
            templates=templates,
            center=center,
            nchans=nchans,
            total_chans=total_chans,
            upsample_factor=upsample_factor,
        )
        spk_props = self.get_spikes_properties(
            fs=fs, isi_threshold=isi_threshold, min_isi=min_isi, nperseg=nperseg
        )
        other_burst_props, burst_props = self.get_burst_properties(
            min_count=min_count,
            min_dur=min_dur,
            max_start=max_start,
            max_int=max_int,
            max_end=max_end,
            output_type=output_type,
        )
        output_dict.update(temp_props)
        output_dict.update(spk_props)
        output_dict.update(burst_props)
        return output_dict, other_burst_props

    def get_cluster_binary_burst(
        self,
        cluster_id: int,
        fs: float,
        min_count: int = 3,
        min_dur: float = 0.01,
        max_start: float = 0.170,
        max_int: float = 0.3,
        max_end: float = 0.34,
        nperseg: int = 1,
        dt: int = 0,
    ):
        b_data = self.get_cluster_bursts(
            cluster_id=cluster_id,
            min_count=min_count,
            min_dur=min_dur,
            max_start=max_start,
            max_int=max_int,
            max_end=max_end,
            fs=fs,
            output_type="sample",
        )
        length = self.end - self.start
        output_data = np.zeros(length // nperseg, dtype=int)
        for i in b_data:
            start = max(0, i[0] // nperseg - dt)
            end = min(i[1] // nperseg + dt, output_data.size)
            output_data[start:end] = 1
        return output_data

    def get_channel_binary_burst(
        self,
        channel: Optional[int] = None,
        min_count: int = 3,
        min_dur: float = 0.01,
        max_start: float = 0.170,
        max_int: float = 0.3,
        max_end: float = 0.34,
        nperseg: int = 1,
    ):
        length = self.end - self.start
        chan_cid_dict = self.get_channel_clusters()
        chans = sorted(list(chan_cid_dict.keys()))
        length = (self.end - self.start) // nperseg
        index = 0
        if channel is not None:
            chans = [channel]
            output_data = np.zeros(
                (len(chan_cid_dict[channel]), length), dtype=np.int16
            )
        else:
            output_data = np.zeros((self.cluster_ids.size, length), dtype=np.int16)
        for channel in chans:
            for cluster_id in chan_cid_dict[channel]:
                b_data = self.get_cluster_bursts(
                    cluster_id=cluster_id,
                    min_count=min_count,
                    min_dur=min_dur,
                    max_start=max_start,
                    max_int=max_int,
                    max_end=max_end,
                    output_type="sample",
                )
                for i in b_data:
                    start = i[0] // nperseg
                    end = i[1] // nperseg
                    output_data[index, start:end] = 1
                index += 1
        return output_data

    def get_cluster_bursts(
        self,
        cluster_id: int,
        min_count: int = 3,
        min_dur: float = 0.01,
        max_start: float = 0.170,
        max_int: float = 0.3,
        max_end: float = 0.34,
        output_type: Literal["sec", "ms", "sample"] = "sec",
        fs: Union[float, int] = 40000,
    ) -> list[np.ndarray]:
        indexes = self.get_cluster_spike_times(
            cluster_id, output_type=output_type, fs=fs
        )
        b_data = max_int_bursts(
            indexes,
            fs,
            min_count=min_count,
            min_dur=min_dur,
            max_start=max_start,
            max_int=max_int,
            max_end=max_end,
            output_type=output_type,
        )
        return b_data

    def get_cluster_burst_properties(
        self,
        cluster_id: int,
        min_count: int = 3,
        min_dur: float = 0.01,
        max_start: float = 0.170,
        max_int: float = 0.3,
        max_end: float = 0.34,
        output_type: Literal["sec", "ms", "sample"] = "sec",
        fs: Union[float, int] = 40000,
    ) -> tuple[dict[str, Union[float, int]], dict[str, np.ndarray], list[np.ndarray]]:
        indexes = self.get_cluster_spike_times(cluster_id)
        bursts = max_int_bursts(
            indexes,
            fs,
            min_count=min_count,
            min_dur=min_dur,
            max_start=max_start,
            max_int=max_int,
            max_end=max_end,
            output_type=output_type,
        )
        props_dict, burst_dict = get_burst_data(bursts)
        return props_dict, burst_dict, bursts

    def save_properties_phy(
        self,
        fs: int = 40000,
        center: int = 41,
        nchans: int = 4,
        total_chans: int = 64,
        callback: Callback = print,
    ):
        callback("Calculating spike template properties.")
        out_data, _ = self.get_properties(
            fs=fs, center=center, nchans=nchans, total_chans=total_chans
        )
        out_data["ch"] = out_data["channel"]
        out_data["Amplitude"] = out_data["center_y"] - out_data["start_y"]
        out_data["ContamPct"] = [100.0] * len(out_data["cluster_id"])
        del out_data["channel"]

        callback("Saving cluster info.")
        save_path = self.ks_directory / "cluster_info"
        save_tsv(save_path, out_data, mode="w")

        callback("Saving cluster Amplitude.")
        save_path = self.ks_directory / "cluster_Amplitude"
        save_tsv(
            save_path,
            {"cluster_id": out_data["cluster_id"], "Amplitude": out_data["Amplitude"]},
        )

        callback("Saving cluster ContamPct.")
        save_path = self.ks_directory / "cluster_ContamPct"
        save_tsv(
            save_path,
            {
                "cluster_id": out_data["cluster_id"],
                "ContamPct": out_data["ContamPct"],
            },
        )

        callback("Saving cluster KSLabel.")
        save_path = self.ks_directory / "cluster_KSLabel"
        save_tsv(
            save_path,
            {
                "cluster_id": out_data["cluster_id"],
                "KSLabel": ["mua"] * len(out_data["cluster_id"]),
            },
        )
        callback("Finished exporting Phy data.")

    def extract_waveforms_chunk(
        self,
        output: np.ndarray,
        recording_chunk: np.ndarray,
        nchans: int,
        start: int,
        end: int,
        waveform_length: int,
        center: Optional[int],
        peaks: np.ndarray,
        channels: int,
        template_amplitudes: np.ndarray,
        subtract: bool = True,
    ):

        # Get only the current spikes
        current_spikes = np.where(
            (self.spike_times < end) & (self.spike_times > start)
        )[0]

        # Sort the spikes by amplitude
        extract_indexes = np.argsort(template_amplitudes[current_spikes])

        spk_chans = nchans * 2
        if center is None:
            wbegin = waveform_length // 2
            wend = wbegin
            cutoff_end = end - wend
        else:
            wend = waveform_length - center
            wbegin = center
            cutoff_end = end - wend
        for i in extract_indexes:
            # Get the spike info in loop
            curr_spk_index = current_spikes[i]
            cur_spk_time = self.spike_times[curr_spk_index]
            cur_template_index = self.spike_templates[curr_spk_index]
            cur_template = self.sparse_templates[cur_template_index]
            chans = channels[cur_template_index, 1:3]
            peak_chan = channels[cur_template_index, 0]
            if (cur_spk_time < cutoff_end) and ((cur_spk_time - wbegin) > start):
                output[curr_spk_index, :, :spk_chans] = recording_chunk[
                    int(cur_spk_time - start - wbegin) : int(
                        cur_spk_time - start + wend
                    ),
                    chans[0] : chans[1],
                ]
                tt = recording_chunk[int(cur_spk_time - start), peak_chan]
                tj = peaks[cur_template_index, 0] / (tt + 1e-10)
                if subtract:
                    recording_chunk[
                        int(cur_spk_time - start - wbegin) : int(
                            cur_spk_time - start + wend
                        ),
                        chans[0] : chans[1],
                    ] -= cur_template[:, chans[0] : chans[1]] / (tj + 1e-10)
            # else:
            #     size = int(cutoff_end - (cur_spk_time - start - width))
            #     output[curr_spk_index, :size, :spk_chans] = recording_chunk[
            #         int(cur_spk_time - start - width) : int(cutoff_end),
            #         chans[0] : chans[1],
            #     ]

    def extract_waveforms(
        self,
        nchans: int = 4,
        waveform_length: int = 82,
        center: Optional[int] = 41,
        ref: bool = False,
        ref_type: Literal["cmr", "car"] = "cmr",
        ref_probe: str = "all",
        map_channel: bool = False,
        probe: str = "all",
        start: Union[None, int] = None,
        end: Union[None, int] = None,
        chunk_size: int = 240000,
        acq_type: Literal["spike", "lfp", "wideband"] = "spike",
        subtract: bool = False,
        callback: Callback = print,
    ):
        if start is None:
            start = self.get_file_attr("start")
        if end is None:
            end = self.get_file_attr("end")
        self.spike_times = np.array(self.spike_times) + start
        n_chunks = (end - start) // (chunk_size)
        chunk_starts = (np.arange(n_chunks) * chunk_size) + start
        output = np.zeros((len(self.spike_times), waveform_length, (nchans * 2)))

        # Get the best range of channels for each template
        channel_map = self.get_grp_dataset("channel_maps", probe)
        peaks, channels = get_template_channels(
            self.sparse_templates, nchans=nchans, total_chans=channel_map.size
        )

        temps = np.unique(self.spike_templates)
        template_peaks = {key: value for key, value in zip(temps, peaks.flatten())}
        template_amplitudes = np.zeros((self.spike_templates.size))
        for i in range(template_amplitudes.size):
            template_amplitudes[i] = template_peaks[self.spike_templates[i]]

        for index, i in enumerate(chunk_starts):
            callback(
                f"Starting chunk {index+1} start at {i} and ending at {i+chunk_size}."
            )
            chunk_start = max(0, i - waveform_length)
            recording_chunk = self.get_multichans(
                acq_type=acq_type,
                ref=ref,
                ref_probe=ref_probe,
                ref_type=ref_type,
                map_channel=map_channel,
                probe=probe,
                start=chunk_start,
                end=i + chunk_size,
            ).T
            self.extract_waveforms_chunk(
                output=output,
                recording_chunk=recording_chunk,
                nchans=nchans,
                start=chunk_start,
                end=i + chunk_size,
                waveform_length=waveform_length,
                center=center,
                peaks=peaks,
                channels=channels,
                template_amplitudes=template_amplitudes,
                subtract=subtract,
            )
        leftover = (end - start) % (chunk_size)
        if leftover > 0:
            i = end - leftover
            callback(f"Starting chunk {index+1} at sample {i}.")
            recording_chunk = self.get_multichans(
                acq_type=acq_type,
                ref=ref,
                ref_probe=ref_probe,
                ref_type=ref_type,
                map_channel=map_channel,
                probe=probe,
                start=i,
                end=end,
            ).T
            self.extract_waveforms_chunk(
                output=output,
                recording_chunk=recording_chunk,
                nchans=nchans,
                start=i,
                end=end,
                waveform_length=waveform_length,
                center=center,
                peaks=peaks,
                channels=channels,
                template_amplitudes=template_amplitudes,
                subtract=subtract,
            )
        return output

    def _extract_spikes_channels(self, channels):
        size = channels[0, 1] - channels[0, 0]
        output = np.zeros((self.spike_clusters.size, size), dtype=int)
        for index in range(self.cluster_ids.size):
            clust_id = self.cluster_ids[index]
            indexes = np.where(self.spike_clusters == clust_id)[0]
            start = channels[index, 0]
            end = channels[index, 1]
            tt = np.arange(start, end)
            output[indexes, :] = tt
        return output

    def export_phy_waveforms(
        self,
        nchans: int = 4,
        waveform_length: int = 82,
        center: Optional[int] = 41,
        ref: bool = False,
        ref_type: Literal["cmr", "car"] = "cmr",
        ref_probe: str = "all",
        map_channel: bool = False,
        probe: str = "all",
        start: Union[None, int] = None,
        end: Union[None, int] = None,
        chunk_size: int = 240000,
        dtype: Literal["f64", "f32", "f16", "i32", "i16"] = "f64",
        subtract: bool = False,
        callback: Callback = print,
    ):
        callback("Extracting spike waveforms.")

        dtypes = {
            "f64": np.float64,
            "f32": np.float32,
            "f16": np.float16,
            "i32": np.int32,
            "i16": np.int16,
        }

        self.spike_waveforms = self.extract_waveforms(
            nchans=nchans,
            waveform_length=waveform_length,
            center=center,
            ref=ref,
            ref_type=ref_type,
            ref_probe=ref_probe,
            map_channel=map_channel,
            probe=probe,
            start=start,
            end=end,
            chunk_size=chunk_size,
            subtract=subtract,
            callback=callback,
        )
        callback("Spike waveforms extracted.")

        callback("Finding spike channels.")

        if dtype != "f64":
            callback(f"Converting to dtype {dtype}.")
            # min_val, max_val = find_minmax_exponent_3(
            #     output, np.finfo("d").max, np.finfo("d").min
            # )
            # exponent = (np.abs((np.abs(min_val) - np.abs(max_val))) // 4) * 3
            # callback(f"Multiplying by 10**{exponent} to reduce dtype.")
            # output = (output * float(10**exponent)).astype(dtypes[dtype])
            self.spike_waveforms.astype(dtype=dtypes[dtype], copy=False)

        callback("Saving spikes waveforms.")
        self._remove_file("_phy_spikes_subset.waveforms.npy")
        np.save(
            self.ks_directory / "_phy_spikes_subset.waveforms.npy", self.spike_waveforms
        )

        callback("Saving spike times.")
        self._remove_file("_phy_spikes_subset.spikes.npy")
        np.save(
            self.ks_directory / "_phy_spikes_subset.spikes.npy",
            np.arange(self.spike_times.shape[0]),
        )

        callback("Saving template channel indices")
        self._remove_file("templates_ind.npy")
        shape = self.sparse_templates.shape
        temp_ind = np.zeros((shape[0], shape[-1]), dtype=np.int16)
        temp_ind[:, :] += np.arange(shape[-1])
        np.save(
            self.ks_directory / "templates_ind.npy",
            temp_ind,
        )

        callback("Finished exporting data.")
        self._load_spike_waveforms()

    def extract_templates(
        self,
        spike_waveforms: np.ndarray,
        total_chans: int = 64,
        callback: callable = print,
    ):
        nchans = spike_waveforms.shape[2] // 2

        _, channels = get_template_channels(
            self.sparse_templates, nchans=nchans, total_chans=total_chans
        )
        waveform_length = spike_waveforms.shape[1]
        sparse_templates_new = np.zeros((self.cluster_ids.size, waveform_length, 64))
        callback("Beginning template extraction")
        spk_templates = np.zeros(self.spike_clusters.size, dtype=int)
        for template_index in range(self.cluster_ids.size):
            clust_id = self.cluster_ids[template_index]
            callback(f"Extracting cluster {clust_id} template")
            indexes = np.where(self.spike_clusters == clust_id)[0]
            temp_spikes_waveforms = spike_waveforms[indexes]
            test = np.mean(temp_spikes_waveforms, axis=0)
            temp_index = np.unique(self.spike_templates[indexes])[0]
            start_chan = channels[temp_index][0] - nchans
            end_chan = channels[temp_index][0] + nchans
            if start_chan < 0:
                end_chan -= start_chan
                start_chan = 0
            if end_chan > total_chans:
                start_chan -= end_chan - total_chans
                end_chan = total_chans
            best_chans = np.arange(start_chan, end_chan)
            sparse_templates_new[template_index, :, best_chans] = test.T

            indexes = np.where(self.spike_clusters == clust_id)[0]
            spk_templates[indexes] = template_index
        return sparse_templates_new, spk_templates

    def save_templates(
        self,
        nchans: int = 4,
        waveform_length: int = 82,
        center: Optional[int] = 41,
        ref: bool = False,
        ref_type: Literal["cmr", "car"] = "cmr",
        ref_probe: str = "all",
        map_channel: bool = False,
        probe: str = "all",
        start: Union[None, int] = None,
        end: Union[None, int] = None,
        chunk_size: int = 240000,
        dtype: Literal["f64", "f32", "f16", "i32", "i16"] = "f32",
        callback: Callback = print,
    ):
        if self.spike_waveforms.size == 0:
            self.export_to_phy(
                nchans=nchans,
                waveform_length=waveform_length,
                center=center,
                ref=ref,
                ref_type=ref_type,
                ref_probe=ref_probe,
                map_channel=map_channel,
                probe=probe,
                start=start,
                end=end,
                chunk_size=chunk_size,
                dtype=dtype,
                callback=callback,
            )
            self._load_spike_waveforms()

        channel_map = self.get_grp_dataset("channel_maps", probe)

        self.sparse_templates, self.spike_templates = self.extract_templates(
            spike_waveforms=self.spike_waveforms,
            total_chans=channel_map.size,
            callback=callback,
        )
        _, channels = get_template_channels(
            self.sparse_templates, nchans=nchans, total_chans=channel_map.size
        )
        full_spike_channels = self._extract_spikes_channels(channels[:, 1:])

        callback("Saving templates.")
        self._remove_file("templates.npy")
        np.save(
            self.ks_directory / "templates.npy",
            self.sparse_templates.astype(np.float32),
        )

        callback("Saving template similarity.")
        self._remove_file("similar_templates.npy")
        np.save(
            self.ks_directory / "similar_templates.npy",
            np.zeros((self.sparse_templates.shape[0], self.sparse_templates.shape[0])),
        )

        callback("Saving spike templates.")
        self._remove_file("spike_templates.npy")
        np.save(
            self.ks_directory / "spike_templates.npy",
            self.spike_templates,
        )

        callback("Saving spike channels.")
        self._remove_file("_phy_spikes_subset.channels.npy")
        np.save(
            self.ks_directory / "_phy_spikes_subset.channels.npy",
            full_spike_channels,
        )

        self._load_sparse_templates()
        self.save_properties_phy(
            center=center, nchans=nchans, total_chans=channel_map.size
        )
        callback("Finished saving templates.")

    def export_to_phy(
        self,
        nchans: int = 4,
        waveform_length: int = 82,
        center: Optional[int] = 41,
        ref: bool = False,
        ref_type: Literal["cmr", "car"] = "cmr",
        ref_probe: str = "all",
        map_channel: bool = False,
        probe: str = "all",
        start: Union[None, int] = None,
        end: Union[None, int] = None,
        chunk_size: int = 480000,
        dtype: Literal["f64", "f32", "f16", "i32", "i16"] = "f32",
        subtract: bool = False,
        callback: callable = print,
    ):
        self.export_phy_waveforms(
            nchans=nchans,
            waveform_length=waveform_length,
            center=center,
            ref=ref,
            ref_type=ref_type,
            ref_probe=ref_probe,
            map_channel=map_channel,
            probe=probe,
            start=start,
            end=end,
            chunk_size=chunk_size,
            dtype=dtype,
            subtract=subtract,
            callback=callback,
        )
        self.save_templates(
            nchans=nchans,
            waveform_length=waveform_length,
            center=center,
            ref=ref,
            ref_type=ref_type,
            ref_probe=ref_probe,
            map_channel=map_channel,
            probe=probe,
            start=start,
            end=end,
            chunk_size=chunk_size,
            dtype=dtype,
            callback=callback,
        )

    def compute_sttc(
        self,
        dt: Union[float, int] = 200,
        start: Union[float, int] = -1,
        end: Union[float, int] = -1,
        sttc_version: Literal["ivs", "elephant"] = "ivs",
        output_type: Literal["sec", "ms", "samples"] = "ms",
        fs: float = 40000.0,
        test_sig: Optional[Literal["shuffle", "distribution"]] = None,
        reps: int = 1000,
        gen_type: Literal[
            "poisson", "gamma", "inverse_gaussian", "lognormal"
        ] = "poisson",
        callback: Callable = print,
    ):
        output_index = 0
        if start == -1:
            start = self.index_to_time(self.start, fs=fs, output_type=output_type)
        if end == -1:
            end = self.index_to_time(self.end, fs=fs, output_type=output_type)

        # Correct for recordings that don't start at zero b/c Kilosort
        # starts all indexes as zero.
        end -= start
        start -= start
        size = (self.cluster_ids.size * (self.cluster_ids.size - 1)) // 2
        sttc_data = np.zeros(size)
        cluster_ids = np.zeros((size, 4), dtype=int)
        if sttc_version == "ivs":
            num1dt_array = np.zeros(size, dtype=int)
            num2dt_array = np.zeros(size, dtype=int)
            num1_2_array = np.zeros(size, dtype=int)
            num2_1_array = np.zeros(size, dtype=int)

        if test_sig is not None:
            sig_vals = np.zeros(size)

        for index1 in range(self.cluster_ids.size - 1):
            clust_id1 = self.cluster_ids[index1]
            indexes1 = self.get_cluster_spike_times(
                clust_id1, output_type=output_type, fs=fs
            )

            if indexes1.size > 3:
                iei_1 = np.diff(indexes1)

            for index2 in range(index1 + 1, self.cluster_ids.size):
                clust_id2 = self.cluster_ids[index2]
                callback(f"Analyzing sttc for cluster {clust_id1} and {clust_id2}.")
                indexes2 = self.get_cluster_spike_times(
                    clust_id2, output_type=output_type, fs=fs
                )

                if indexes2.size > 3:
                    iei_2 = np.diff(indexes2)

                if sttc_version == "ivs":
                    sttc_index, num1dt, num1_2, num2dt, num2_1 = sttc(
                        indexes1, indexes2, dt=dt, start=start, stop=end
                    )
                    num1dt_array[output_index] = num1dt
                    num2dt_array[output_index] = num2dt
                    num1_2_array[output_index] = num1_2
                    num2_1_array[output_index] = num2_1
                elif sttc_version == "elephant":
                    sttc_index = sttc_ele(
                        indexes1, indexes2, dt=dt, start=start, stop=end
                    )
                else:
                    sttc_index = sttc_python(
                        indexes1,
                        indexes2,
                        indexes1.size,
                        indexes2.size,
                        dt=dt,
                        start=start,
                        stop=end,
                    )

                if test_sig is not None:
                    if indexes1.size > 3 and indexes2.size > 3:
                        sig, _ = _sttc_sig(
                            sttc_value=sttc_index,
                            iei_1=iei_1,
                            iei_2=iei_2,
                            dt=5,
                            start=start,
                            end=end,
                            reps=reps,
                            sttc_version="ivs",
                            test_version=test_sig,
                            gen_type=gen_type,
                            input_type=output_type,
                        )
                        sig_vals[output_index] = sig
                    else:
                        sig_vals[output_index] = 0.9999

                sttc_data[output_index] = sttc_index
                cluster_ids[output_index, 0] = clust_id1
                cluster_ids[output_index, 1] = clust_id2
                cluster_ids[output_index, 2] = indexes1.size
                cluster_ids[output_index, 3] = indexes2.size
                output_index += 1

        data = {}
        data["sttc"] = sttc_data
        data["cluster1_id"] = cluster_ids[:, 0].flatten()
        data["cluster2_id"] = cluster_ids[:, 1].flatten()
        data["cluster1_size"] = cluster_ids[:, 2].flatten()
        data["cluster2_size"] = cluster_ids[:, 3].flatten()
        if sttc_version == "ivs":
            data["1_before_2"] = num1_2_array
            data["2_before_1"] = num2_1_array
            data["1_dt_2"] = num1dt_array
            data["2_dt_1"] = num2dt_array
        if test_sig is not None:
            data["sttc_sig"] = sig_vals
        return data

    def compute_correlation(
        self,
        dt: Union[float, int] = 200,
        start: Union[float, int] = -1,
        end: Union[float, int] = -1,
        output_type: Literal["sec", "ms", "samples"] = "samples",
        fs: float = 40000.0,
        callback: Callable = print,
    ):
        output_index = 0
        if start == -1:
            start = self.index_to_time(self.start, fs=fs, output_type=output_type)
        if end == -1:
            end = self.index_to_time(self.end, fs=fs, output_type=output_type)
        size = (self.cluster_ids.size * (self.cluster_ids.size - 1)) // 2
        correlation_data = np.zeros(size)
        cluster_ids = np.zeros((size, 4), dtype=int)

        for index1 in range(self.cluster_ids.size - 1):
            clust_id1 = self.cluster_ids[index1]
            indexes1 = self.get_cluster_spike_times(
                clust_id1, output_type=output_type, fs=fs
            )

            for index2 in range(index1 + 1, self.cluster_ids.size):
                clust_id2 = self.cluster_ids[index2]
                callback(f"Analyzing sttc for cluster {clust_id1} and {clust_id2}.")
                indexes2 = self.get_cluster_spike_times(
                    clust_id2, output_type=output_type, fs=fs
                )

                cluster_ids[output_index, 0] = clust_id1
                cluster_ids[output_index, 1] = clust_id2
                cluster_ids[output_index, 2] = indexes1.size
                cluster_ids[output_index, 3] = indexes2.size
                correlation_data[output_index] = sttc(
                    indexes1, indexes2, dt=dt, start=start, stop=end
                )
                output_index += 1

        data = {}
        data["sttc"] = correlation_data
        data["cluster1_id"] = cluster_ids[:, 0].flatten()
        data["cluster2_id"] = cluster_ids[:, 1].flatten()
        data["cluster1_size"] = cluster_ids[:, 2].flatten()
        data["cluster2_size"] = cluster_ids[:, 3].flatten()
        return data
