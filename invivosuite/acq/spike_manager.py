from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Literal, Optional, TypedDict, Union

import numpy as np
from scipy import stats
from send2trash import send2trash

from ..utils import concatenate_dicts, save_tsv
from .spike_functions import (
    _template_channels,
    amplitude_cutoff,
    bin_spikes,
    create_binary_spikes,
    create_continuous_spikes,
    fit_iei,
    _gen_bootstrap_sttc,
    get_burst_data,
    get_template_channels,
    isi_violations,
    max_int_bursts,
    presence,
    sfa_abi,
    sfa_divisor,
    sfa_local_var,
    sfa_rlocal_var,
    _shuffle_bootstrap_sttc,
    sttc,
    sttc_ele,
    sttc_python,
    template_properties,
)

Callback = Callable[[str], None]


class SpikeProperties(TypedDict):
    presence_ratio: np.ndarray[float]
    iei: np.ndarray[float]
    n_spikes: np.ndarray[int]
    cluster_id: np.ndarray[int]
    contampct: np.ndarray[float]
    fp_rate: np.ndarray[float]
    num_violations: np.ndarray[float]
    fr: np.ndarray[float]
    fr_iei: np.ndarray[float]
    amplitude_cutoff: np.ndarray[float]
    fano_factor: np.ndarray[float]
    divisor_sfa: np.ndarray[float]
    abi_sfa: np.ndarray[float]
    local_sfa: np.ndarray[float]
    rlocal_sfa: np.ndarray[float]


class TemplateProperties(TypedDict):
    hwL: np.ndarray[float]
    hwL_len: np.ndarray[float]
    hwR: np.ndarray[float]
    hwR_len: np.ndarray[float]
    full_width: np.ndarray[float]
    start: np.ndarray[float]
    end: np.ndarray[float]
    center: np.ndarray[float]
    min_x: np.ndarray[float]
    center_y: np.ndarray[float]
    start_y: np.ndarray[float]
    end_y: np.ndarray[float]
    min_y: np.ndarray[float]
    stdev: np.ndarray[float]
    channel: np.ndarray[int]


class BurstProperties(TypedDict):
    num_bursts: np.ndarray[int]
    ave_burst_len: np.ndarray[float]
    intra_burst_iei: np.ndarray[float]
    inter_burst_iei: np.ndarray[float]
    ave_spikes_burst: np.ndarray[float]
    divisor_sfa_b: np.ndarray[float]
    abi_sfa_b: np.ndarray[float]
    local_sfa_b: np.ndarray[float]
    rlocal_sfa_b: np.ndarray[float]
    peak_divisor_sfa_b: np.ndarray[float]
    peak_abi_sfa_b: np.ndarray[float]
    peak_local_sfa_b: np.ndarray[float]
    peak_rlocal_sfa_b: np.ndarray[float]
    total_time: np.ndarray[float]


class SpkManager:
    def load_ks_data(self, load_type: str = "r+"):
        self._load_sparse_templates(load_type=load_type)
        self._load_spike_templates(load_type=load_type)
        self._load_spike_clusters(load_type=load_type)
        self._load_spike_times(load_type=load_type)
        self._load_amplitudes(load_type=load_type)
        self._load_spike_waveforms(load_type=load_type)

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
        output_type: Literal["sec", "ms", "samples"] = "sec",
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
        dt: int = 0,
    ) -> np.ndarray:
        spike_ids = np.where(self.spike_clusters == cluster_id)[0]
        spike_indexes = self.spike_times[spike_ids]
        return create_binary_spikes(spike_indexes, self.end - self.start, dt=dt)

    def get_continuous_spike_cluster(
        self,
        cluster_id: int,
        fs: float = 40000.0,
        nperseg: int = 0,
        window: Literal["exponential", "gaussian"] = "gaussian",
        sigma: float = 200,
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
    ) -> SpikeProperties:
        if start == -1:
            start = self.start / fs
        if end == -1:
            end = self.end / fs

        size = len(self.cluster_ids)

        presence_ratios = np.zeros(size)
        iei = np.zeros(size)
        n_spikes = np.zeros(size, dtype=int)
        fr_iei = np.zeros(size)
        fr = np.zeros(size)
        frate = np.zeros(size)
        abi_sfa = np.zeros(size)
        local_sfa = np.zeros(size)
        rlocal_sfa = np.zeros(size)
        div_sfa = np.zeros(size)
        num_violations = np.zeros(size, dtype=int)
        amp_cutoff = np.zeros(size)
        ff = np.zeros(size)

        for i in range(size):
            clust_id = self.cluster_ids[i]
            times = self.get_cluster_spike_times(clust_id, fs=fs, output_type="sec")
            amps = self.get_cluster_spike_amplitudes(clust_id)
            binned = self.get_binned_spike_cluster(cluster_id=clust_id, nperseg=nperseg)
            ff[i] = binned.var() / binned.mean()
            if times.size > 2:
                isi = np.diff(times)
                mean_isi = np.mean(isi)
                iei[i] = mean_isi
                if output_type == "sec":
                    fr_iei[i] = 1 / mean_isi
                else:
                    fr_iei[i] = 1000 / mean_isi
                fr[i] = times.size / (end - start)
                abi_sfa[i] = sfa_abi(isi)
                local_sfa[i] = sfa_local_var(isi)
                rlocal_sfa[i] = sfa_rlocal_var(isi, R)
                div_sfa[i] = sfa_divisor(isi)
                fpRate, nv = isi_violations(
                    spike_train=times,
                    min_time=start,
                    max_time=end,
                    isi_threshold=isi_threshold,
                    min_isi=min_isi,
                )
                frate[i] = fpRate
                num_violations[i] = nv
            else:
                iei[i] = 0
                fr[i] = 0
                abi_sfa[i] = np.nan
                local_sfa[i] = np.nan
                rlocal_sfa[i] = np.nan
                div_sfa[i] = np.nan
            n_spikes[i] = times.size
            if output_type == "sec":
                divisor = fs
            else:
                divisor = fs / 1000
            pr = presence(times, self.start / divisor, self.end / divisor)
            presence_ratios[i] = pr["presence_ratio"]
            cutoff = amplitude_cutoff(amps)
            amp_cutoff[i] = cutoff
        data = SpikeProperties(
            presence_ratio=presence_ratios,
            iei=iei,
            fr_iei=fr_iei,
            n_spikes=n_spikes,
            cluster_id=self.cluster_ids,
            fr=fr,
            fp_rate=frate,
            num_violations=num_violations,
            amplitude_cutoff=amp_cutoff,
            fano_factor=ff,
            divisor_sfa=div_sfa,
            abi_sfa=abi_sfa,
            local_sfa=local_sfa,
            rlocal_sfa=rlocal_sfa,
        )
        return data

    def get_templates_properties(
        self,
        templates: np.ndarray,
        center: int = 41,
        nchans: int = 4,
        total_chans: int = 64,
        upsample_factor: int = 2,
        negative: bool = True,
    ) -> TemplateProperties:
        hwL = np.zeros(self.cluster_ids.size)
        hwL_len = np.zeros(self.cluster_ids.size)
        hwR = np.zeros(self.cluster_ids.size)
        hwR_len = np.zeros(self.cluster_ids.size)
        full_width = np.zeros(self.cluster_ids.size)
        start_x = np.zeros(self.cluster_ids.size)
        end_x = np.zeros(self.cluster_ids.size)
        center_x = np.zeros(self.cluster_ids.size)
        min_x = np.zeros(self.cluster_ids.size)
        start_y = np.zeros(self.cluster_ids.size)
        end_y = np.zeros(self.cluster_ids.size)
        center_y = np.zeros(self.cluster_ids.size)
        min_y = np.zeros(self.cluster_ids.size)
        stdevs = np.zeros(self.cluster_ids.size)
        channel = np.zeros(self.cluster_ids.size, dtype=int)
        for temp_index in range(self.cluster_ids.size):
            i = self.cluster_ids[temp_index]
            chan, start_chan, _ = _template_channels(
                templates[temp_index, :, :], nchans=nchans, total_chans=total_chans
            )
            t_props = template_properties(
                templates[temp_index, :, chan],
                center=center,
                upsample_factor=upsample_factor,
                negative=negative,
            )
            hwL[temp_index] = t_props["hwL"]
            hwL_len[temp_index] = t_props["hwL_len"]
            hwR[temp_index] = t_props["hwR"]
            hwR_len[temp_index] = t_props["hwR_len"]
            full_width[temp_index] = t_props["full_width"]
            start_x[temp_index] = t_props["start"]
            end_x[temp_index] = t_props["end"]
            center_x[temp_index] = t_props["center"]
            min_x[temp_index] = t_props["min_x"]
            center_y[temp_index] = t_props["center_y"]
            start_y[temp_index] = t_props["start_y"]
            end_y[temp_index] = t_props["end_y"]
            min_y[temp_index] = t_props["min_y"]
            indexes = np.where(self.spike_clusters == i)[0]
            temp_spikes_waveforms = self.spike_waveforms[indexes]
            template_stdev = np.mean(
                np.std(temp_spikes_waveforms, axis=0)[:, int(chan - start_chan)]
            )
            stdevs[temp_index] = template_stdev
            channel[temp_index] = chan
        temp = TemplateProperties(
            hwL=hwL,
            hwR=hwR,
            hwL_len=hwL_len,
            hwR_len=hwR_len,
            full_width=full_width,
            start=start_x,
            end=end_x,
            center=center_x,
            min_x=min_x,
            start_y=start_y,
            end_y=end_y,
            center_y=center_y,
            min_y=min_y,
            stdev=stdevs,
            channel=channel,
        )
        return temp

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
    ) -> BurstProperties:
        other_props = []
        num_bursts = np.ndarray(self.cluster_ids.size, dtype=int)
        ave_burst_len = np.ndarray(self.cluster_ids.size, dtype=float)
        intra_burst_iei = np.ndarray(self.cluster_ids.size, dtype=float)
        inter_burst_iei = np.ndarray(self.cluster_ids.size, dtype=float)
        ave_spikes_burst = np.ndarray(self.cluster_ids.size, dtype=float)
        ave_divisor = np.ndarray(self.cluster_ids.size, dtype=float)
        ave_abi = np.ndarray(self.cluster_ids.size, dtype=float)
        ave_local = np.ndarray(self.cluster_ids.size, dtype=float)
        ave_rlocal = np.ndarray(self.cluster_ids.size, dtype=float)
        peak_divisor = np.ndarray(self.cluster_ids.size, dtype=float)
        peak_abi = np.ndarray(self.cluster_ids.size, dtype=float)
        peak_local = np.ndarray(self.cluster_ids.size, dtype=float)
        peak_rlocal = np.ndarray(self.cluster_ids.size, dtype=float)
        total_time = np.ndarray(self.cluster_ids.size, dtype=float)
        for cluster_index, clust_id in enumerate(self.cluster_ids):
            spk_times = self.get_cluster_spike_times(clust_id)
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
            num_bursts[cluster_index] = burst_dict["num_bursts"]
            ave_burst_len[cluster_index] = burst_dict["ave_burst_len"]
            intra_burst_iei[cluster_index] = burst_dict["intra_burst_iei"]
            inter_burst_iei[cluster_index] = burst_dict["inter_burst_iei"]
            ave_spikes_burst[cluster_index] = burst_dict["ave_spikes_per_burst"]
            ave_divisor[cluster_index] = burst_dict["ave_divisor_sfa"]
            ave_abi[cluster_index] = burst_dict["ave_abi_sfa"]
            ave_local[cluster_index] = burst_dict["ave_local_sfa"]
            ave_rlocal[cluster_index] = burst_dict["ave_rlocal_sfa"]
            peak_divisor[cluster_index] = burst_dict["peak_divisor_sfa"]
            peak_abi[cluster_index] = burst_dict["peak_abi_sfa"]
            peak_local[cluster_index] = burst_dict["peak_local_sfa"]
            peak_rlocal[cluster_index] = burst_dict["peak_rlocal_sfa"]
            total_time[cluster_index] = burst_dict["total_burst_time"]
            props_dict["cluster_id"] = [clust_id] * len(b_data)
            other_props.append(props_dict)
        burst_props = BurstProperties(
            ave_burst_len=ave_burst_len,
            num_bursts=num_bursts,
            intra_burst_iei=intra_burst_iei,
            inter_burst_iei=inter_burst_iei,
            ave_spikes_burst=ave_spikes_burst,
            divisor_sfa_b=ave_divisor,
            abi_sfa_b=ave_abi,
            local_sfa_b=ave_local,
            rlocal_sfa_b=ave_rlocal,
            total_time=total_time,
            peak_divisor_sfa_b=peak_divisor,
            peak_abi_sfa_b=peak_abi,
            peak_local_sfa_b=peak_local,
            peak_rlocal_sfa_b=peak_rlocal,
        )
        other_props = [i for i in other_props if i["cluster_id"]]
        other_props = concatenate_dicts(other_props)
        return burst_props, other_props

    def get_properties(
        self,
        templates: Optional[np.ndarray] = None,
        fs: int = 40000,
        center=41,
        nchans: int = 4,
        total_chans: int = 64,
        upsample_factor: int = 2,
        isi_threshold: float = 1.5,
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
        burst_props, other_burst_props = self.get_burst_properties(
            min_count=min_count,
            min_dur=min_dur,
            max_start=max_start,
            max_int=max_int,
            max_end=max_end,
            output_type=output_type,
        )
        output_dict["cluster_id"] = self.cluster_ids
        output_dict.update(temp_props)
        output_dict.update(spk_props)
        output_dict.update(burst_props)
        return output_dict, other_burst_props

    def get_cluster_binary_burst(
        self,
        cluster_id: int,
        min_count: int = 3,
        min_dur: float = 0.01,
        max_start: float = 0.170,
        max_int: float = 0.3,
        max_end: float = 0.34,
        nperseg: int = 1,
    ):
        b_data = self.get_cluster_bursts(
            cluster_id=cluster_id,
            min_count=min_count,
            min_dur=min_dur,
            max_start=max_start,
            max_int=max_int,
            max_end=max_end,
            output_type="sample",
        )
        length = self.end - self.start
        output_data = np.zeros(length // nperseg, dtype=int)
        for i in b_data:
            start = i[0] // nperseg
            end = i[1] // nperseg
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
        indexes = self.get_cluster_spike_times(cluster_id)
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
        n_chunks = (end - start) // (chunk_size)
        chunk_starts = np.arange(n_chunks) * chunk_size
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
        test_sig: Optional[Literal["shuffle", "generate"]] = None,
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

        for index1 in range(self.cluster_ids.size - 1):
            clust_id1 = self.cluster_ids[index1]
            indexes1 = self.get_cluster_spike_times(
                clust_id1, output_type=output_type, fs=fs
            )
            iei_1 = np.diff(indexes1)

            for index2 in range(index1 + 1, self.cluster_ids.size):
                clust_id2 = self.cluster_ids[index2]
                callback(f"Analyzing sttc for cluster {clust_id1} and {clust_id2}.")
                indexes2 = self.get_cluster_spike_times(
                    clust_id2, output_type=output_type, fs=fs
                )
                iei_2 = np.diff(indexes2)

                if sttc_version == "ivs":
                    sttc_index, num1dt, num1_2, num2dt, num2_1 = sttc(
                        indexes1, indexes2, dt=dt, start=start, stop=end
                    )
                    sttc_data[output_index] = sttc_index
                    num1dt_array[output_index] = num1dt
                    num2dt_array[output_index] = num2dt
                    num1_2_array[output_index] = num1_2
                    num2_1_array[output_index] = num2_1
                elif sttc_version == "elephant":
                    sttc_index = sttc_ele(
                        indexes1, indexes2, dt=dt, start=start, stop=end
                    )
                    sttc_data[output_index] = sttc_index
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
