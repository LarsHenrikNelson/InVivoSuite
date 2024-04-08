from collections import defaultdict
from typing import Literal, Union


import numpy as np

from .spike_functions import find_minmax_exponent_3


class SpkManager:
    def load_ks_data(self):
        self._load_sparse_templates()
        self._load_spike_templates()
        self._load_spike_clusters()
        self._load_spike_times()
        self._load_amplitudes()

    def _load_amplitudes(self):
        self.amplitudes = np.load(
            str(self.ks_directory / "amplitudes.npy"), "r+"
        ).flatten()

    def _load_sparse_templates(self):
        self.sparse_templates = np.load(str(self.ks_directory / "templates.npy"), "r+")

    def _load_spike_templates(self):
        self.spike_templates = np.load(
            str(self.ks_directory / "spike_templates.npy"), "r+"
        ).flatten()
        self.template_ids = np.unique(self.spike_templates)

    def _load_spike_clusters(self):
        self.spike_clusters = np.load(
            str(self.ks_directory / "spike_clusters.npy"), "r+"
        ).flatten()
        self.cluster_ids = np.unique(self.spike_clusters)

    def _create_chan_clusters(self):
        self.chan_clusters = defaultdict(list)
        for cluster, chan in zip(self.cluster_ids, self.cluster_channels):
            self.chan_clusters[chan].append(cluster)

    def _load_spike_times(self):
        self.spike_times = np.load(
            str(self.ks_directory / "spike_times.npy"), "r+"
        ).flatten()

    def get_template_channels(self, templates, nchans, total_chans):
        channel_output = np.zeros((templates.shape[0], 3), dtype=int)
        peak_output = np.zeros((templates.shape[0], 1))
        for i in range(templates.shape[0]):
            chan = np.argmin(np.min(templates[i], axis=0))
            start_chan = chan - nchans
            end_chan = chan + nchans
            if start_chan < 0:
                end_chan -= start_chan
                start_chan = 0
            if end_chan > total_chans:
                start_chan -= end_chan - total_chans
                end_chan = total_chans
            channel_output[i, 1] = start_chan
            channel_output[i, 2] = end_chan
            channel_output[i, 0] = chan
            peak_output[i, 0] = np.min(templates[i, :, chan])
        return peak_output, channel_output

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
            if (cur_spk_time < cutoff_end) and ((cur_spk_time - width) > start):
                cur_template_index = self.spike_templates[curr_spk_index]
                cur_template = self.sparse_templates[cur_template_index]
                chans = channels[cur_template_index, 1:3]
                peak_chan = channels[cur_template_index, 0]

                output[curr_spk_index, :, :spk_chans] = recording_chunk[
                    int(cur_spk_time - start - width) : int(
                        cur_spk_time - start + width
                    ),
                    chans[0] : chans[1],
                ]
                tt = recording_chunk[
                    int(cur_spk_time - start - width) : int(
                        cur_spk_time - start + width
                    ),
                    peak_chan,
                ]
                tj = peaks[cur_template_index, 0] / (np.min(tt) + 1e-10)
                recording_chunk[
                    int(cur_spk_time - start - width) : int(
                        cur_spk_time - start + width
                    ),
                    chans[0] : chans[1],
                ] -= (
                    cur_template[:, chans[0] : chans[1]] / tj
                )

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
                "spike",
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
            # _extract_waveforms_chunk(
            #     spike_times=self.spike_times,
            #     spike_templates=self.spike_templates,
            #     sparse_templates=self.sparse_templates,
            #     output=output,
            #     recording_chunk=recording_chunk,
            #     nchans=nchans,
            #     start=start,
            #     end=end,
            #     waveform_length=waveform_length,
            #     peaks=peaks,
            #     channels=channels,
            #     template_amplitudes=template_amplitudes,
            # )
        leftover = (end - start) % (chunk_size)
        if leftover > 0:
            i = end - leftover
            callback(f"Starting chunk {index+1} at sample {i}.")
            recording_chunk = self.get_multichans(
                "spike",
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
            # _extract_waveforms_chunk(
            #     spike_times=self.spike_times,
            #     spike_templates=self.spike_templates,
            #     sparse_templates=self.sparse_templates,
            #     output=output,
            #     recording_chunk=recording_chunk,
            #     nchans=nchans,
            #     start=start,
            #     end=end,
            #     waveform_length=waveform_length,
            #     peaks=peaks,
            #     channels=channels,
            #     template_amplitudes=template_amplitudes,
            # )
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
        callback("Extracting spike waveforms.")

        dtypes = {
            "f64": np.float64,
            "f32": np.float32,
            "f16": np.float16,
            "i32": np.int32,
            "i16": np.int16,
        }

        output = self.extract_waveforms(
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
            output.astype(dtype=dtypes[dtype], copy=False)

        callback("Saving spikes waveforms.")
        np.save(self.ks_directory / "_phy_spikes_subset.waveforms.npy", output)

        callback("Saving spike times.")
        np.save(
            self.ks_directory / "_phy_spikes_subset.spikes.npy",
            np.arange(self.spike_times.shape[0]),
        )

        callback("Saving spike channels.")
        np.save(
            self.ks_directory / "_phy_spikes_subset.channels.npy", full_spike_channels
        )
        callback("Finished exporting data.")
