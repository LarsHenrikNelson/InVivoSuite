from collections import defaultdict
from typing import Literal, Union


import numpy as np


class SpkManager:
    def load_ks_data(self):
        self._load_sparse_templates()
        self._load_spike_templates()
        self._load_spike_clusters()
        self._load_spike_times()
        self._load_amplitudes()

    def _load_amplitudes(self):
        self.amplitudes = np.load(str(self.ks_directory / "amplitudes.npy"), "r+")

    def _load_sparse_templates(self):
        self.sparse_templates = np.load(str(self.ks_directory / "templates.npy"), "r+")

    def _load_spike_templates(self):
        self.spike_templates = np.load(
            str(self.ks_directory / "spike_templates.npy"), "r+"
        )
        self.template_ids = np.unique(self.spike_templates)

    def _load_spike_clusters(self):
        self.spike_clusters = np.load(
            str(self.ks_directory / "spike_clusters.npy"), "r+"
        )
        self.cluster_ids = np.unique(self.spike_clusters)

    def _create_chan_clusters(self):
        self.chan_clusters = defaultdict(list)
        for cluster, chan in zip(self.cluster_ids, self.cluster_channels):
            self.chan_clusters[chan].append(cluster)

    def _load_spike_times(self):
        self.spike_times = np.load(str(self.ks_directory / "spike_times.npy"), "r+")

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
                start_chan -= end_chan - total_chans + 1
                end_chan = total_chans - 1
            channel_output[i, 1] = start_chan
            channel_output[i, 2] = end_chan
            channel_output[i, 0] = chan
            peak_output[i, 0] = np.min(templates[i, :, chan])
        return peak_output, channel_output

    def extract_waveforms_chunk(
        self,
        output,
        recording_chunk,
        time_offset,
        nchans,
        total_chans,
        start,
        end,
        waveform_length,
    ):

        # Get only the current spikes
        current_spikes = np.where((self.spike_times < end) & (self.spike_times > start))

        # Sort the spikes by amplitude
        extract_indexes = (
            np.argsort(self.amplitudes[current_spikes], axis=0) + current_spikes[0]
        )

        # Get the best range of channels for each template
        peaks, channels = self.get_template_channels(
            self.sparse_templates, nchans=nchans, total_chans=total_chans
        )
        spk_chans = nchans * 2
        width = waveform_length / 2
        cutoff = end - width
        for i in extract_indexes:
            # Get the spike info in loop
            cur_spk_time = self.spike_times[i, 0]
            if cur_spk_time < cutoff:
                cur_template_index = self.spike_templates[i]
                cur_template = self.sparse_templates[cur_template_index][0]
                chans = channels[cur_template_index, 1:3][0]
                peak_chan = channels[cur_template_index, 0]
                output[i, :, :spk_chans] = recording_chunk[
                    int(cur_spk_time - time_offset - width) : int(
                        cur_spk_time - time_offset + width
                    ),
                    chans[0] : chans[1],
                ]
                tt = recording_chunk[
                    int(cur_spk_time - time_offset - width) : int(
                        cur_spk_time - time_offset + width
                    ),
                    peak_chan,
                ]
                tj = peaks[cur_template_index, 0] / (np.min(tt) + 1e-10)
                recording_chunk[
                    int(cur_spk_time - time_offset - width) : int(
                        cur_spk_time - time_offset + width
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
        callback=print,
    ):
        if start is None:
            start = self.get_file_attr("start")
        if end is None:
            end = self.get_file_attr("end")
        n_chunks = ((end - start) // (chunk_size)) - 1
        chunk_starts = np.arange(n_chunks) * chunk_size
        output = np.zeros((len(self.spike_times), waveform_length, 12))
        for index, i in enumerate(chunk_starts):
            callback(f"Starting chunk {index+1} at {i}.")
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
                output,
                recording_chunk,
                i,
                nchans,
                recording_chunk.shape[-1],
                chunk_start,
                i + chunk_size,
                waveform_length,
            )
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
                output,
                recording_chunk,
                recording_chunk.shape[-1],
                nchans,
                i,
                end,
                waveform_length,
            )
        return output
