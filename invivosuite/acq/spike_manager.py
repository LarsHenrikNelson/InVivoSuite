from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Literal, TypedDict, Union, Optional

import numpy as np

from send2trash import send2trash

from ..utils import save_tsv
from .spike_functions import (
    presence,
    get_burst_data,
    max_int_bursts,
    sttc,
    sttc_ele,
    isi_violations,
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


class TemplateProperties(TypedDict):
    peak_Left: np.ndarray[float]
    peak_Float: np.ndarray[float]
    trough: np.ndarray[float]
    peak_to_end: np.ndarray[int]
    start_to_peak: np.ndarray[int]
    half_width: np.ndarray[float]
    cluster_id: np.ndarray[int]
    stdev: np.ndarray[int]
    channel: np.ndarray[int]
    half_width_zero: np.ndarray[int]


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

    def get_cluster_spike_indexes(self, cluster_id: int) -> np.ndarray:
        # Convenience function, not used for anything.
        spike_ids = np.where(self.spike_clusters == cluster_id)[0]
        indexes = self.spike_times[spike_ids].flatten()
        return indexes

    def get_cluster_spike_ids(self, cluster_id: int) -> np.ndarray:
        return np.where(self.spike_clusters == cluster_id)[0]

    def get_cluster_templates(self, cluster_id: int) -> np.ndarray:
        cid = np.where(self.spike_clusters == cluster_id)[0]
        return np.unique(self.spike_templates[cid])

    def get_cluster_template_waveforms(self, cluster_id: int) -> np.ndarray:
        return self.sparse_self.sparse_templates[cluster_id, :, :]

    def get_cluster_best_template_waveform(
        self, cluster_id: int, nchans: int = 4, total_chans: int = 64
    ) -> np.ndarray:
        template_index = np.where(self.cluster_ids == cluster_id)[0]
        chan, _, _ = self._template_channels(
            self.sparse_templates[template_index],
            nchans=nchans,
            total_chans=total_chans,
        )
        return self.sparse_templates[template_index, :, chan]

    def get_cluster_spike_times(
        self,
        cluster_id: int,
        fs: Optional[int] = None,
        out_type: Literal["sec", "ms", "samples"] = "sec",
    ) -> np.ndarray:
        spike_ids = np.where(self.spike_clusters == cluster_id)[0]
        if fs is None:
            return self.spike_times[spike_ids]
        if out_type == "ms":
            times = self.spike_times[spike_ids].flatten() / (fs / 1000)
        elif out_type == "sec":
            times = self.spike_times[spike_ids].flatten() / fs
        return times

    def get_spikes_properties(
        self,
        fs: int = 40000,
        start: int = -1,
        end: int = -1,
        isi_threshold=1.5,
        min_isi=0,
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
        num_violations = np.zeros(size, dtype=int)

        for i in range(size):
            clust_id = self.cluster_ids[i]
            times = self.get_cluster_spike_times(clust_id, fs=fs, out_type="sec")
            if times.size > 2:
                diffs = np.mean(np.diff(times))
                iei[i] = diffs
                fr_iei[i] = 1 / diffs
                fr[i] = times.size / (end - start)
                fpRate, nv = isi_violations(
                    spike_train=times * 1000,
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
            n_spikes[i] = times.size
            pr = presence(times, self.start / fs, self.end / fs)
            presence_ratios[i] = pr["presence_ratio"]
        data = SpikeProperties(
            presence_ratio=presence_ratios,
            iei=iei,
            fr_iei=fr_iei,
            n_spikes=n_spikes,
            cluster_id=self.cluster_ids,
            fr=fr,
            fp_rate=frate,
            num_violations=num_violations,
        )
        return data

    def get_templates_properties(
        self,
        templates: np.ndarray,
        nchans: int,
        total_chans: int,
        negative: bool = True,
    ) -> TemplateProperties:
        ampR = np.zeros(self.cluster_ids.size)
        ampL = np.zeros(self.cluster_ids.size)
        trough = np.zeros(self.cluster_ids.size)
        hw = np.zeros(self.cluster_ids.size)
        hw_zero = np.zeros(self.cluster_ids.size)
        channel = np.zeros(self.cluster_ids.size, dtype=int)
        cid = np.zeros(self.cluster_ids.size, dtype=int)
        stdevs = np.zeros(self.cluster_ids.size, dtype=float)
        peak_to_end = np.zeros(self.cluster_ids.size, dtype=int)
        start_to_peak = np.zeros(self.cluster_ids.size, dtype=int)
        for temp_index in range(self.cluster_ids.size):
            i = self.cluster_ids[temp_index]
            chan, start_chan, _ = self._template_channels(
                templates[i], nchans=nchans, total_chans=total_chans
            )
            t_props = template_properties(templates[i, :, chan], negative=negative)
            hw[temp_index] = t_props["half_width"]
            ampL[temp_index] = t_props["peak_Left"]
            ampR[temp_index] = t_props["peak_Right"]
            peak_to_end[temp_index] = t_props["peak_to_end"]
            start_to_peak[temp_index] = t_props["start_to_peak"]
            trough[temp_index] = t_props["trough"]
            hw_zero[temp_index] = t_props["half_width_zero"]
            indexes = np.where(self.spike_clusters == i)[0]
            temp_spikes_waveforms = self.spike_waveforms[indexes]
            template_stdev = np.mean(
                np.std(temp_spikes_waveforms, axis=0)[:, int(chan - start_chan)]
            )
            channel[temp_index] = chan
            cid[temp_index] = i
            stdevs[temp_index] = template_stdev
        temp = TemplateProperties(
            peak_Right=ampR,
            peak_Left=ampL,
            half_width=hw,
            channel=channel,
            stdev=stdevs,
            peak_to_end=peak_to_end,
            start_to_peak=start_to_peak,
            cluster_id=cid,
            trough=trough,
            half_width_zero=hw_zero,
        )
        return temp

    def get_properties(self, fs: int = 40000, nchans: int = 4, total_chans: int = 64):
        output_dict = {}
        temp_props = self.get_templates_properties(
            self.sparse_templates, nchans, total_chans
        )
        spk_props = self.get_spikes_properties(fs=fs)
        output_dict.update(temp_props)
        output_dict.update(spk_props)
        return output_dict

    def calculate_spike_bursts(
        self,
        min_dur: float = 0.01,
        max_start: float = 0.170,
        max_int: float = 0.3,
        max_end: float = 0.34,
        output_type: Literal["sec", "ms", "sample"] = "sec",
        fs: Union[float, int] = 40000,
    ):
        output = []
        for clust_id in self.cluster_ids:
            indexes = self.get_cluster_spike_times(
                clust_id,
            )
            data = {}
            if indexes.size > 1:
                temp_mean = np.mean(np.diff(indexes / fs))
                if temp_mean == 0:
                    data["hertz"] = 0.0
                else:
                    data["hertz"] = 1 / temp_mean
            else:
                data["hertz"] = 0.0
            data["num_spikes"] = indexes.size
            b_data = max_int_bursts(
                indexes,
                fs,
                min_dur=min_dur,
                max_start=max_start,
                max_int=max_int,
                max_end=max_end,
                output_type=output_type,
            )
            bursts_dict = get_burst_data(b_data)
            data["id"] = self.ks_directory.stem
            data["cluster_id"] = clust_id
            data.update(bursts_dict)
            output.append(data)
        return output

    def get_cluster_bursts(
        self,
        cluster_id,
        min_dur: float = 0.01,
        max_start: float = 0.170,
        max_int: float = 0.3,
        max_end: float = 0.34,
        output_type: Literal["sec", "ms", "sample"] = "sec",
        fs: Union[float, int] = 40000,
    ):
        indexes = self.get_cluster_spike_times(cluster_id)
        b_data = max_int_bursts(
            indexes,
            fs,
            min_dur=min_dur,
            max_start=max_start,
            max_int=max_int,
            max_end=max_end,
            output_type=output_type,
        )
        return b_data

    def save_properties_phy(
        self,
        file_path=None,
        fs: int = 40000,
        nchans: int = 4,
        total_chans: int = 64,
        callback: Callback = print,
    ):
        callback("Calculating spike template properties.")
        out_data = self.get_properties(fs, nchans, total_chans)
        out_data["ch"] = out_data["channel"]
        out_data["Amplitude"] = out_data["trough"] - out_data["peak_Right"]
        out_data["ContamPct"] = [100.0] * len(out_data["cluster_id"])
        del out_data["channel"]

        callback("Saving cluster info.")
        if file_path is None:
            save_path = self.ks_directory / "cluster_info"
        else:
            save_path = Path(file_path) / "cluster_info"
        save_tsv(save_path, out_data, mode="w")

        callback("Saving cluster Amplitude.")
        if file_path is None:
            save_path = self.ks_directory / "cluster_Amplitude"
        else:
            save_path = Path(file_path) / "cluster_Amplitude"
        save_tsv(
            save_path,
            {"cluster_id": out_data["cluster_id"], "Amplitude": out_data["Amplitude"]},
        )

        callback("Saving cluster ContamPct.")
        if file_path is None:
            save_path = self.ks_directory / "cluster_ContamPct"
        else:
            save_path = Path(file_path) / "cluster_ContamPct"
        save_tsv(
            save_path,
            {
                "cluster_id": out_data["cluster_id"],
                "ContamPct": out_data["ContamPct"],
            },
        )

        callback("Saving cluster KSLabel.")
        if file_path is None:
            save_path = self.ks_directory / "cluster_KSLabel"
        else:
            save_path = Path(file_path) / "cluster_KSLabel"
        save_tsv(
            save_path,
            {
                "cluster_id": out_data["cluster_id"],
                "KSLabel": ["mua"] * len(out_data["cluster_id"]),
            },
        )
        callback("Finished exporting Phy data.")

    def get_template_channels(self, templates, nchans: int, total_chans: int):
        channel_output = np.zeros((templates.shape[0], 3), dtype=int)
        peak_output = np.zeros((templates.shape[0], 1))
        for i in range(templates.shape[0]):
            chan, start_chan, end_chan = self._template_channels(
                templates[i], nchans, total_chans
            )
            # chan = np.argmin(np.min(templates[i], axis=0))
            # start_chan = chan - nchans
            # end_chan = chan + nchans
            # if start_chan < 0:
            #     end_chan -= start_chan
            #     start_chan = 0
            # if end_chan > total_chans:
            #     start_chan -= end_chan - total_chans
            #     end_chan = total_chans
            channel_output[i, 1] = start_chan
            channel_output[i, 2] = end_chan
            channel_output[i, 0] = chan
            peak_output[i, 0] = np.min(templates[i, :, chan])
        return peak_output, channel_output

    def _template_channels(self, template: np.ndarray, nchans: int, total_chans: int):
        best_chan = np.argmin(np.min(template, axis=0))
        start_chan = best_chan - nchans
        end_chan = best_chan + nchans
        if start_chan < 0:
            end_chan -= start_chan
            start_chan = 0
        if end_chan > total_chans:
            start_chan -= end_chan - total_chans
            end_chan = total_chans
        return best_chan, start_chan, end_chan

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
        center: Optional[int] = None,
        ref: bool = False,
        ref_type: Literal["cmr", "car"] = "cmr",
        ref_probe: str = "all",
        map_channel: bool = False,
        probe: str = "all",
        start: Union[None, int] = None,
        end: Union[None, int] = None,
        chunk_size: int = 240000,
        output_chans: int = 16,
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
        output = np.zeros((len(self.spike_times), waveform_length, output_chans))

        # Get the best range of channels for each template
        channel_map = self.get_grp_dataset("channel_maps", probe)
        peaks, channels = self.get_template_channels(
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
        for clust_id in self.cluster_ids:
            indexes = np.where(self.spike_clusters == clust_id)[0]
            start = channels[clust_id, 0]
            end = channels[clust_id, 1]
            tt = np.arange(start, end)
            output[indexes, :] = tt
        return output

    def export_phy_waveforms(
        self,
        nchans: int = 4,
        waveform_length: int = 82,
        center: Optional[int] = None,
        ref: bool = False,
        ref_type: Literal["cmr", "car"] = "cmr",
        ref_probe: str = "all",
        map_channel: bool = False,
        probe: str = "all",
        start: Union[None, int] = None,
        end: Union[None, int] = None,
        chunk_size: int = 240000,
        output_chans: int = 16,
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
            output_chans=output_chans,
            subtract=subtract,
            callback=callback,
        )
        callback("Spike waveforms extracted.")

        callback("Finding spike channels.")
        channel_map = self.get_grp_dataset("channel_maps", probe)
        _, channels = self.get_template_channels(
            self.sparse_templates, nchans=nchans, total_chans=channel_map.size
        )
        full_spike_channels = self._extract_spikes_channels(channels[:, 1:])

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

        callback("Saving spike channels.")
        self._remove_file("_phy_spikes_subset.channels.npy")
        np.save(
            self.ks_directory / "_phy_spikes_subset.channels.npy",
            full_spike_channels,
        )

        callback("Saving template channel indices")
        self._remove_file("templates_ind.npy")
        shape = self.sparse_templates.shape
        temp_ind = np.zeros((shape[0], shape[-1]), dtype=np.int16)
        temp_ind[:, :] += np.arange(shape[-1])

        callback("Finished exporting data.")
        self._load_spike_waveforms()

    def extract_templates(
        self,
        spike_waveforms: np.ndarray,
        nchans: int = 4,
        total_chans: int = 64,
        waveform_length: int = 84,
        callback: callable = print,
    ):
        _, channels = self.get_template_channels(
            self.sparse_templates, nchans=nchans, total_chans=total_chans
        )
        sparse_templates_new = np.zeros((self.cluster_ids.size, waveform_length, 64))
        callback("Beginning template extraction")
        spk_templates = np.zeros(self.cluster_ids.size, dtype=int)
        for i in range(self.cluster_ids.size):
            clust_id = self.cluster_ids[i]
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
            sparse_templates_new[clust_id, :, best_chans] = test.T

            indexes = np.where(self.spike_clusters == clust_id)[0]
            spk_templates[indexes] = i
        return sparse_templates_new, spk_templates

    def save_templates(
        self,
        nchans: int = 4,
        waveform_length: int = 82,
        center: Optional[int] = None,
        ref: bool = False,
        ref_type: Literal["cmr", "car"] = "cmr",
        ref_probe: str = "all",
        map_channel: bool = False,
        probe: str = "all",
        start: Union[None, int] = None,
        end: Union[None, int] = None,
        chunk_size: int = 240000,
        output_chans: int = 16,
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
                output_chans=output_chans,
                dtype=dtype,
                callback=callback,
            )
            self._load_spike_waveforms()

        channel_map = self.get_grp_dataset("channel_maps", probe)

        self.sparse_templates, self.spike_templates = self.extract_templates(
            spike_waveforms=self.spike_waveforms,
            nchans=nchans,
            total_chans=channel_map.size,
            waveform_length=waveform_length,
            callback=callback,
        )

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

        self._load_sparse_templates()
        self.save_properties_phy()
        callback("Finished recomputing templates.")

    def export_to_phy(
        self,
        nchans: int = 4,
        waveform_length: int = 82,
        center: Optional[int] = None,
        ref: bool = False,
        ref_type: Literal["cmr", "car"] = "cmr",
        ref_probe: str = "all",
        map_channel: bool = False,
        probe: str = "all",
        start: Union[None, int] = None,
        end: Union[None, int] = None,
        chunk_size: int = 480000,
        output_chans: int = 16,
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
            output_chans=output_chans,
            dtype=dtype,
            subtract=subtract,
            callback=callback,
        )
        self.save_templates(
            nchans=nchans,
            waveform_length=waveform_length,
            ref=ref,
            ref_type=ref_type,
            ref_probe=ref_probe,
            map_channel=map_channel,
            probe=probe,
            start=start,
            end=end,
            chunk_size=chunk_size,
            output_chans=output_chans,
            dtype=dtype,
            callback=callback,
        )
        self.save_properties_phy()

    def compute_sttc(
        self,
        dt: Union[float, int] = 200,
        start: Union[float, int] = -1,
        end: Union[float, int] = -1,
        sttc_version: Literal["ivs, elephant"] = "ivs",
    ):
        output_index = 0
        if start == -1:
            start = self.start
        if end == -1:
            end = self.end
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
            indexes1 = self.get_cluster_spike_indexes(clust_id1)

            for index2 in range(index1 + 1, self.cluster_ids.size):
                clust_id2 = self.cluster_ids[index2]
                indexes2 = self.get_cluster_spike_indexes(clust_id2)

                if sttc_version == "ivs":
                    sttc_index, num1dt, num1_2, num2dt, num2_1 = sttc(
                        indexes1, indexes2, dt=dt, start=start, stop=end
                    )
                    sttc_data[output_index] = sttc_index
                    num1dt_array[output_index] = num1dt
                    num2dt_array[output_index] = num2dt
                    num1_2_array[output_index] = num1_2
                    num2_1_array[output_index] = num2_1
                else:
                    sttc_index = sttc_ele(
                        indexes1, indexes2, dt=dt, start=start, stop=end
                    )
                    sttc_data[output_index] = sttc_index

                cluster_ids[output_index, 0] = clust_id1
                cluster_ids[output_index, 1] = clust_id2
                cluster_ids[output_index, 2] = indexes1.size
                cluster_ids[output_index, 1] = indexes2.size
                output_index += 1

        data = {}
        data["id"] = [self.ks_directory.stem] * size
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
