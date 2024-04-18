from collections import defaultdict
from pathlib import Path
from typing import Literal, TypedDict, Union

import numpy as np

from send2trash import send2trash

from ..utils import save_tsv
from .spike_functions import get_template_parts, presence


class SpikeProperties(TypedDict):
    presence_ratio: list[float]
    iei: list[float]
    n_spikes: list[int]
    cluster_id: list[int]
    contampct: list[float]
    # depth: list
    # sh: list
    fr: list[float]


class TemplateProperties(TypedDict):
    start_index: list[int]
    peak_index: list[int]
    end_index: list[int]
    peak_value: list[float]
    amplitude: list[float]
    cluster_id: list[int]
    stdev: list[int]
    channel: list[int]


class SpkManager:
    def load_ks_data(self):
        self._load_sparse_templates()
        self._load_spike_templates()
        self._load_spike_clusters()
        self._load_spike_times()
        self._load_amplitudes()
        self._load_spike_waveforms()

    def _load_amplitudes(self, load_type: str = "r+"):
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
            self.sparse_templates = np.load(
                self.ks_directory / "templates.npy", load_type
            )
        else:
            self.sparse_templates = np.zeros((0, 0, 0))

    def _load_spike_templates(self, load_type: str = "r+"):
        self.spike_templates = np.load(
            self.ks_directory / "spike_templates.npy", load_type
        ).flatten()
        self.template_ids = np.unique(self.spike_templates)

    def _load_spike_clusters(self, load_type: str = "r+"):
        self.spike_clusters = np.load(
            self.ks_directory / "spike_clusters.npy", load_type
        ).flatten()
        self.cluster_ids = np.unique(self.spike_clusters)

    def _create_chan_clusters(self):
        self.chan_clusters = defaultdict(list)
        for cluster, chan in zip(self.cluster_ids, self.cluster_channels):
            self.chan_clusters[chan].append(cluster)

    def _load_spike_times(self, load_type: str = "r+"):
        self.spike_times = np.load(
            self.ks_directory / "spike_times.npy", load_type
        ).flatten()

    def _load_spike_waveforms(self, load_type: str = "r+"):
        temp_path = self.ks_directory / "_phy_spikes_subset.waveforms.npy"
        if temp_path.exists():
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

    def get_cluster_spike_times(
        self,
        cluster_id: int,
        fs: int = 40000,
        out_type: Literal["sec", "ms", "samples"] = "sec",
    ) -> np.ndarray:
        spike_ids = np.where(self.spike_clusters == cluster_id)[0]
        if out_type == "ms":
            times = self.spike_times[spike_ids].flatten() / (fs / 1000)
        elif out_type == "sec":
            times = self.spike_times[spike_ids].flatten() / fs
        return times

    def get_spikes_properties(
        self, templates: np.ndarray, fs: int = 40000
    ) -> SpikeProperties:
        _, channels = self.get_template_channels(templates, nchans=8, total_chans=64)
        presence_ratios = []
        iei = []
        n_spikes = []
        # depth = []
        fr = []
        cluster_ids = self.cluster_ids
        for i in cluster_ids:
            times = self.get_cluster_spike_times(i, fs=fs, out_type="sec")
            if times.size > 2:
                diffs = np.mean(np.diff(times))
                iei.append(diffs)
                fr.append(1 / diffs)
            else:
                iei.append(0)
                fr.append(0)
            n_spikes.append(times.size)
            pr = presence(times, self.start / fs, self.end / fs)
            presence_ratios.append(pr["presence_ratio"])
        data = SpikeProperties(
            presence_ratio=presence_ratios,
            iei=iei,
            n_spikes=n_spikes,
            channel=channels[cluster_ids, 0],
            cluster_id=cluster_ids,
            fr=fr,
        )
        return data

    def get_templates_properties(
        self, templates: np.ndarray, nchans, total_chans
    ) -> TemplateProperties:
        amplitude = []
        start = []
        middle = []
        end = []
        trough_to_peak = []
        channel = []
        cid = []
        stdevs = []
        for i in self.cluster_ids:
            chan, start_chan, _ = self._template_channels(
                templates[i], nchans=nchans, total_chans=total_chans
            )
            si, ei, mi, amp, t_to_p = get_template_parts(templates[i, :, chan])
            indexes = np.where(self.spike_clusters == i)[0]
            temp_spikes_waveforms = self.spike_waveforms[indexes]
            template_stdev = np.mean(
                np.std(temp_spikes_waveforms[:, :, int(chan - start_chan)], axis=0)
            )
            start.append(si)
            end.append(ei)
            middle.append(mi)
            amplitude.append(amp * 1000000)
            trough_to_peak.append(t_to_p * 1000000)
            channel.append(chan)
            cid.append(i)
            stdevs.append(template_stdev)
        temp = TemplateProperties(
            start_index=start,
            peak_index=middle,
            end_index=end,
            peak_value=amplitude,
            amplitude=trough_to_peak,
            channel=channel,
            stdev=stdevs,
            cluster_id=cid,
        )
        return temp

    def get_properties(self, fs: int = 40000, nchans: int = 4, total_chans: int = 64):
        output_dict = {}
        temp_props = self.get_templates_properties(
            self.sparse_templates, nchans, total_chans
        )
        spk_props = self.get_spikes_properties(self.sparse_templates, fs=fs)
        output_dict.update(temp_props)
        output_dict.update(spk_props)
        return output_dict

    def save_properties_phy(
        self,
        file_path=None,
        fs: int = 40000,
        nchans: int = 4,
        total_chans: int = 64,
        callback=print,
    ):
        callback("Calculating spike template properties.")
        out_data = self.get_properties(fs, nchans, total_chans)
        out_data["ch"] = out_data["channel"]
        out_data["Amplitude"] = out_data["amplitude"]
        out_data["ContamPct"] = [100.0] * len(out_data["cluster_id"])
        del out_data["amplitude"]
        del out_data["channel"]
        del out_data["start_index"]
        del out_data["end_index"]
        del out_data["peak_index"]

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

    def get_template_channels(self, templates, nchans, total_chans):
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

    def _template_channels(self, template: np.ndarray, nchans, total_chans):
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
        output,
        recording_chunk,
        nchans,
        start,
        end,
        waveform_length,
        peaks,
        channels,
        template_amplitudes,
    ):

        # Get only the current spikes
        current_spikes = np.where(
            (self.spike_times < end) & (self.spike_times > start)
        )[0]

        # Sort the spikes by amplitude
        extract_indexes = np.argsort(template_amplitudes[current_spikes])

        spk_chans = nchans * 2
        width = waveform_length / 2
        cutoff_end = end - width
        for i in extract_indexes:
            # Get the spike info in loop
            curr_spk_index = current_spikes[i]
            cur_spk_time = self.spike_times[curr_spk_index]
            cur_template_index = self.spike_templates[curr_spk_index]
            cur_template = self.sparse_templates[cur_template_index]
            chans = channels[cur_template_index, 1:3]
            peak_chan = channels[cur_template_index, 0]
            if (cur_spk_time < cutoff_end) and ((cur_spk_time - width) > start):
                output[curr_spk_index, :, :spk_chans] = recording_chunk[
                    int(cur_spk_time - start - width) : int(
                        cur_spk_time - start + width
                    ),
                    chans[0] : chans[1],
                ]
                tt = recording_chunk[int(cur_spk_time - start), peak_chan]
                tj = peaks[cur_template_index, 0] / (tt + 1e-10)
                recording_chunk[
                    int(cur_spk_time - start - width) : int(
                        cur_spk_time - start + width
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
        callback=print,
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
                peaks=peaks,
                channels=channels,
                template_amplitudes=template_amplitudes,
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
                peaks=peaks,
                channels=channels,
                template_amplitudes=template_amplitudes,
            )
        return output

    def extract_spike_channels(self, probe, nchans, output_chans):
        channel_map = self.get_grp_dataset("channel_maps", probe)
        _, channels = self.get_template_channels(
            self.sparse_templates, nchans=nchans, total_chans=channel_map.size
        )
        spike_channels = np.full(
            (self.sparse_templates.shape[0], output_chans), fill_value=-1
        )
        for i in range(channels.shape[0]):
            num_channels = channels[i, 2] - channels[i, 1]
            spike_channels[i, :num_channels] = np.arange(channels[i, 1], channels[i, 2])
        full_spike_channels = np.full(
            (self.spike_times.shape[0], output_chans), fill_value=-1
        )
        for i in range(self.spike_templates.shape[0]):
            temp_index = self.spike_templates[i]
            full_spike_channels[i] = spike_channels[temp_index]
        return full_spike_channels

    def export_phy_waveforms(
        self,
        nchans: int = 4,
        waveform_length: int = 82,
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
        callback=print,
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
            ref=ref,
            ref_type=ref_type,
            ref_probe=ref_probe,
            map_channel=map_channel,
            probe=probe,
            start=start,
            end=end,
            chunk_size=chunk_size,
            output_chans=output_chans,
            callback=callback,
        )
        callback("Spike waveforms extracted.")

        callback("Finding spike channels.")
        full_spike_channels = self.extract_spike_channels(
            probe=probe, nchans=nchans, output_chans=output_chans
        )

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

    def recompute_templates(
        self,
        nchans: int = 4,
        waveform_length: int = 82,
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
        callback=print,
    ):
        if self.spike_waveforms.size == 0:
            self.export_to_phy(
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
            self._load_spike_waveforms()

        channel_map = self.get_grp_dataset("channel_maps", probe)
        _, channels = self.get_template_channels(
            self.sparse_templates, nchans=nchans, total_chans=channel_map.size
        )

        sparse_templates_new = np.zeros((self.cluster_ids[-1] + 1, waveform_length, 64))
        callback("Starting to recompute templates")
        self.spike_templates = np.array(self.spike_templates)
        for clust_id in self.cluster_ids:
            callback(f"Recomputing cluster {clust_id} template")
            indexes = np.where(self.spike_clusters == clust_id)[0]
            temp_spikes_waveforms = self.spike_waveforms[indexes]
            test = np.mean(temp_spikes_waveforms, axis=0)
            temp_index = np.unique(self.spike_templates[indexes])[0]
            start_chan = channels[temp_index][0] - nchans
            end_chan = channels[temp_index][0] + nchans
            if start_chan < 0:
                end_chan -= start_chan
                start_chan = 0
            if end_chan > channel_map.size:
                start_chan -= end_chan - channel_map.size
                end_chan = channel_map.size
            best_chans = np.arange(start_chan, end_chan)
            sparse_templates_new[clust_id, :, best_chans] = test.T
            self.spike_templates[indexes] = clust_id
        self.sparse_templates = sparse_templates_new

        callback("Saving templates.")
        self._remove_file("templates.npy")
        np.save(
            self.ks_directory / "templates.npy",
            sparse_templates_new.astype(np.float32),
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
        callback=print,
    ):
        self.export_phy_waveforms(
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
        self.recompute_templates(
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
