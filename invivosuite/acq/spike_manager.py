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
        nchans,
        start,
        end,
        waveform_length,
        peaks,
        channels,
    ):

        # Get only the current spikes
        current_spikes = np.where(
            (self.spike_times < end) & (self.spike_times > start)
        )[0]

        # Sort the spikes by amplitude
        extract_indexes = np.argsort(self.amplitudes[current_spikes])

        spk_chans = nchans * 2
        width = waveform_length / 2
        cutoff_end = end - width
        for i in extract_indexes:
            # Get the spike info in loop
            curr_spk_index = current_spikes[i]
            cur_spk_time = self.spike_times[curr_spk_index]
            if cur_spk_time < cutoff_end and (cur_spk_time - width) > start:
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
        callback=print,
    ):
        if start is None:
            start = self.get_file_attr("start")
        if end is None:
            end = self.get_file_attr("end")
        n_chunks = (end - start) // (chunk_size)
        chunk_starts = np.arange(n_chunks) * chunk_size
        output = np.zeros((len(self.spike_times), waveform_length, 12))

        # Get the best range of channels for each template
        channel_map = self.get_grp_dataset("channel_maps", probe)
        peaks, channels = self.get_template_channels(
            self.sparse_templates, nchans=nchans, total_chans=channel_map.size
        )

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
                output=output,
                recording_chunk=recording_chunk,
                nchans=nchans,
                start=i,
                end=end,
                waveform_length=waveform_length,
                peaks=peaks,
                channels=channels,
            )
        return output

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
        callback=print,
    ):
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
            callback=callback,
        )
        peaks, channels = self.get_template_channels(
            self.sparse_templates, nchans=nchans, total_chans=64
        )
        spike_channels = np.full((self.sparse_templates.shape[0], 12), fill_value=-1)
        for i in range(channels.shape[0]):
            num_channels = channels[i, 2] - channels[i, 1]
            spike_channels[: num_channels + 1] = np.arange(
                channels[i, 1], channels[i, 2] + 1
            )

        np.save(self.ks_directory / "_phy_spikes_subset.waveforms.npy", output)
        np.save(
            self.ks_directory / "_phy_spikes_subset.waveforms.npy", self.spike_times
        )
        np.save(self.ks_directory / "_phy_spikes_subset.channels.npy", spike_channels)
