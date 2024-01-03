from collections import defaultdict
from pathlib import Path

import numpy as np

# from .spike_metrics import calculate_metrics


class SpikeModel:
    def __init__(self, directory, n_channels=64, offset=0):
        self.n_channels = n_channels
        self.offset = offset
        self.directory = Path(directory)
        self.n_closest_channels = 12
        self.amplitude_threshold = 0
        self.load_data()

    def load_data(self):
        self._load_sparse_templates()
        self._load_templates_cols()
        self._load_spike_templates()
        self._load_spike_clusters()
        self._load_spike_times()
        self._load_whitening_matrix()
        self._load_whitening_matrix_inv()
        self._load_channel_map()
        self._load_channel_positions()
        self._load_channel_shanks()
        self._load_amplitudes()
        self._create_merge_map()
        self._load_rec()
        self._set_best_channels()
        self._map_template_channels()
        self._create_chan_clusters()

    def _load_sparse_templates(self):
        self.sparse_templates = np.load(str(self.directory / "templates.npy"), "r+")

    def _load_templates_cols(self):
        # self.template_cols = np.load(str(self.directory / "templates_ind.npy"), "r+")
        # self.num_templates = self.template_cols.shape[0]
        # phylib model does not recognize file and sets template_cols to None
        self.template_cols = None

    def _load_spike_templates(self):
        self.spike_templates = np.load(
            str(self.directory / "spike_templates.npy"), "r+"
        )
        self.template_ids = np.unique(self.spike_templates)

    def _load_spike_clusters(self):
        self.spike_clusters = np.load(str(self.directory / "spike_clusters.npy"), "r+")
        self.cluster_ids = np.unique(self.spike_clusters)

    def _create_chan_clusters(self):
        self.chan_clusters = defaultdict(list)
        for cluster, chan in zip(self.cluster_ids, self.cluster_channels):
            self.chan_clusters[chan].append(cluster)

    def _load_spike_times(self):
        self.spike_times = np.load(str(self.directory / "spike_times.npy"), "r+")

    def _load_whitening_matrix(self):
        self.wm = np.load(str(self.directory / "whitening_mat.npy"), "r+")

    def _load_whitening_matrix_inv(self):
        self.wmi = np.load(str(self.directory / "whitening_mat_inv.npy"), "r+")

    def _load_channel_positions(self):
        self.channel_positions = np.load(
            str(self.directory / "channel_positions.npy"), "r+"
        )

    def _load_channel_map(self):
        self.channel_mapping = np.load(
            str(self.directory / "channel_map.npy"), "r+"
        ).squeeze()
        self.n_channels = self.channel_mapping.shape[0]

    def _load_amplitudes(self):
        self.amplitudes = np.load(str(self.directory / "amplitudes.npy"), "r+")

    def _load_channel_shanks(self):
        self.channel_shanks = np.zeros(self.n_channels, dtype=np.int32)

    def _create_merge_map(self):
        self.merge_map = {key: [] for key in range(np.max(self.spike_clusters) + 1)}
        for temp in np.unique(self.spike_templates):
            idx = np.where(self.spike_templates == temp)[0]
            new_idx = self.spike_clusters[idx]
            mapping = np.unique(new_idx)
            for n in mapping:
                self.merge_map[n].append(temp)

    def _load_rec(self):
        # I will likely just rely on on my acquisition manager to provide
        # the acquisitions since that is already built in.
        bin_path = list(self.directory.glob("*.bin"))[0]
        fsize = bin_path.stat().st_size
        item_size = np.dtype(np.int16).itemsize
        n_samples = (fsize - self.offset) // (item_size * self.n_channels)
        shape = (n_samples, self.n_channels)
        bin_path = list(self.directory.glob("*.bin"))[0]
        self.traces = np.memmap(bin_path, shape=shape, dtype=np.int16)

    def _set_best_channels(self):
        self.cluster_channels = np.array(
            [self.get_cluster_best_channel(i) for i in self.cluster_ids]
        )

    def _map_template_channels(self):
        self.templates_channels = defaultdict(list)
        self.channel_clusters = defaultdict(list)
        for i in range(self.cluster_ids.size):
            self.channel_clusters[self.cluster_channels[i]].append(self.cluster_ids[i])
            self.templates_channels[self.cluster_ids[i]].append(
                self.cluster_channels[i]
            )

    def get_cluster_spike_ids(self, cluster_id):
        # Convenience function, not used for anything.
        spike_ids = np.where(self.spike_clusters == cluster_id)[0]
        return spike_ids

    def get_cluster_spike_indexes(self, cluster_id):
        # Convenience function, not used for anything.
        spike_ids = np.where(self.spike_clusters == cluster_id)[0]
        indexes = self.spike_times[spike_ids].flatten()
        return indexes

    def get_cluster_amplitudes(self, cluster_id):
        spike_ids = np.where(self.spike_clusters == cluster_id)[0]
        amplitudes = self.amplitudes[spike_ids].flatten()
        return amplitudes

    def get_cluster_spike_times(self, cluster_id):
        spike_ids = np.where(self.spike_clusters == cluster_id)[0]
        times = self.spike_times[spike_ids].flatten() / 40000
        return times

    def get_cluster_iei(self, cluster_id):
        spike_ids = np.where(self.spike_clusters == cluster_id)[0]
        times = self.spike_times[spike_ids].flatten() / 40000
        return np.diff(times)

    def get_cluster_channels(
        self, cluster_id, get_best_channel: bool = True, all_channels: bool = False
    ):
        """Return the most relevant channels of a cluster."""
        # template_ids, counts = get_template_ids_from_spikes(
        #     cluster_id, self.spike_clusters, self.spike_templates
        # )
        spike_ids = np.where(self.spike_clusters == cluster_id)
        st = self.spike_templates[spike_ids]
        template_ids, counts = np.unique(st, return_counts=True)
        channels = []
        if get_best_channel:
            template_id = template_ids[np.argmax(counts)]
            _, _, best_channel, channel_ids = get_template(
                template_id=template_id,
                sparse_templates=self.sparse_templates,
                template_cols=self.template_cols,
                wmi=self.wmi,
                amplitude_threshold=self.amplitude_threshold,
                channel_positions=self.channel_positions,
                channel_shanks=self.channel_shanks,
            )
            if all_channels:
                channels.extend(channel_ids)
            else:
                channels.append(best_channel)
        else:
            for template_id in template_ids:
                _, _, best_channel, channel_ids = get_template(
                    template_id,
                    self.sparse_templates,
                    self.template_cols,
                    wmi=self.wmi,
                )
                channels.extend(channel_ids)
                if all_channels:
                    channels.extend(channel_ids)
                else:
                    channels.append(best_channel)
        return channels

    def get_cluster_best_channel(self, cluster_id):
        """Return the most relevant channel of a cluster."""
        spike_ids = np.where(self.spike_clusters == cluster_id)
        st = self.spike_templates[spike_ids]
        template_ids, counts = np.unique(st, return_counts=True)
        # template_ids, counts = get_template_ids_from_spikes(
        #     cluster_id, self.spike_clusters, self.spike_templates
        # )
        template_id = template_ids[np.argmax(counts)]
        if self.template_cols is None:
            _, _, best_channel, _ = _get_template_dense(
                template_id=template_id,
                sparse_templates=self.sparse_templates,
                wmi=self.wmi,
                amplitude_threshold=self.amplitude_threshold,
                channel_positions=self.channel_positions,
                channel_shanks=self.channel_shanks,
            )
        else:
            _, _, best_channel, _ = _get_template_sparse(
                template_id,
                self.sparse_templates,
                wmi=self.wmi,
            )
        return best_channel

    # def calculate_spk_metrics(self, fs, isi_threshold, min_isi):
    #     fs /= 1000
    #     labels, m = calculate_metrics(
    #         self.spike_times / fs,
    #         self.spike_clusters,
    #         self.amplitudes,
    #         isi_threshold,
    #         min_isi,
    #     )
    #     return labels, m


def _unwhiten(wmi, x, channel_ids=None):
    mat = wmi
    if channel_ids is not None:
        mat = mat[np.ix_(channel_ids, channel_ids)]
        assert mat.shape == (len(channel_ids),) * 2
    assert x.shape[1] == mat.shape[0]
    out = np.dot(x, mat) * 1.0
    return np.ascontiguousarray(out)


def get_closest_channels(channel_positions, channel_index, n=None):
    """Get the channels closest to a given channel on the probe."""
    x = channel_positions[:, 0]
    y = channel_positions[:, 1]
    x0, y0 = channel_positions[channel_index]
    d = (x - x0) ** 2 + (y - y0) ** 2
    out = np.argsort(d)
    if n:
        out = out[:n]
    assert out[0] == channel_index
    return out


def _find_best_channels(
    template,
    channel_positions,
    channel_shanks,
    n_closest_channels=12,
    amplitude_threshold=0,
):
    """Find the best channels for a given template."""
    # Compute the template amplitude on each channel.
    assert template.ndim == 2  # shape: (n_samples, n_channels)
    amplitude = template.max(axis=0) - template.min(axis=0)
    assert not np.all(np.isnan(amplitude)), "Template is all NaN!"
    assert amplitude.ndim == 1  # shape: (n_channels,)
    # Find the peak channel.
    best_channel = np.argmax(amplitude)
    max_amp = amplitude[best_channel]
    # Find the channels X% peak.
    peak_channels = np.nonzero(amplitude >= amplitude_threshold * max_amp)[0]
    # Find N closest channels.
    close_channels = get_closest_channels(
        channel_positions, best_channel, n_closest_channels
    )
    assert best_channel in close_channels
    # Restrict to the channels belonging to the best channel's shank.
    if channel_shanks is not None:
        shank = channel_shanks[best_channel]  # shank of best channel
        channels_on_shank = np.nonzero(channel_shanks == shank)[0]
        close_channels = np.intersect1d(close_channels, channels_on_shank)
    # Keep the intersection.
    channel_ids = np.intersect1d(peak_channels, close_channels)
    # Order the channels by decreasing amplitude.
    order = np.argsort(amplitude[channel_ids])[::-1]
    channel_ids = channel_ids[order]
    amplitude = amplitude[order]
    assert best_channel in channel_ids
    assert amplitude.shape == (len(channel_ids),)
    return channel_ids, amplitude, best_channel


def _get_template_dense(
    template_id,
    sparse_templates,
    channel_positions,
    wmi,
    channel_shanks,
    amplitude_threshold=0,
    n_closest_channels=12,
):
    """Return data for one template."""
    template_w = sparse_templates[template_id, ...]
    if wmi is not None:
        template_w = _unwhiten(wmi, template_w)
    assert template_w.ndim == 2
    channel_ids_, amplitude, best_channel = _find_best_channels(
        template=template_w,
        channel_positions=channel_positions,
        channel_shanks=channel_shanks,
        n_closest_channels=n_closest_channels,
        amplitude_threshold=amplitude_threshold,
    )
    template_w = template_w[:, channel_ids_]
    assert template_w.ndim == 2
    assert template_w.shape[1] == channel_ids_.shape[0]
    return template_w, amplitude, best_channel, channel_ids_


def _get_template_sparse(
    template_id,
    sparse_templates,
    template_cols,
    wmi=None,
):
    assert template_cols is not None
    template_w, channel_ids = sparse_templates[template_id], template_cols[template_id]

    # KS2 HACK: dense templates may have been saved as
    # sparse arrays (with all channels),
    # we need to remove channels with no signal.

    # template_w is (n_samples, n_channels)
    template_max = np.abs(template_w).max(axis=0)
    has_signal = template_max > template_max.max() * 1e-6
    channel_ids = channel_ids[has_signal]
    template_w = template_w[:, has_signal]

    # Remove unused channels = -1.
    used = channel_ids != -1
    template_w = template_w[:, used]
    channel_ids = channel_ids[used]
    channel_ids = channel_ids.astype(np.uint32)

    # Unwhiten.
    if wmi is not None:
        template_w = _unwhiten(wmi, template_w, channel_ids=channel_ids)
    template_w = template_w.astype(np.float32)
    assert template_w.ndim == 2
    assert template_w.shape[1] == len(channel_ids)
    amplitude = template_w.max(axis=0) - template_w.min(axis=0)
    best_channel = channel_ids[np.argmax(amplitude)][::-1]
    # NOTE: it is expected that the channel_ids are reordered by decreasing amplitude.
    # To each column of the template_w array corresponds the channel id
    # given by channel_ids.
    channels_reordered = np.argsort(amplitude)[::-1]
    return (
        template_w[..., channels_reordered],
        amplitude,
        best_channel,
        channel_ids[..., channels_reordered],
    )


def get_template(
    template_id,
    sparse_templates,
    channel_ids=None,
    wmi=None,
    template_cols=None,
    channel_positions=None,
    channel_shanks=None,
    amplitude_threshold=0,
):
    if template_cols is None:
        template, amplitude, best_channel, channel_ids = _get_template_dense(
            template_id=template_id,
            sparse_templates=sparse_templates,
            wmi=wmi,
            channel_positions=channel_positions,
            channel_shanks=channel_shanks,
            amplitude_threshold=amplitude_threshold,
        )
    else:
        template, amplitude, best_channel, channel_ids = _get_template_sparse(
            template_id=template_id,
            sparse_templates=sparse_templates,
            template_cols=template_cols,
            wmi=wmi,
        )
    return template, amplitude, best_channel, channel_ids


def _extract_waveform(traces, sample, channel_ids=None, n_samples_waveforms=None):
    """Extract a single spike waveform."""
    nsw = n_samples_waveforms
    assert traces.ndim == 2
    dur = traces.shape[0]
    a = nsw // 2
    b = nsw - a
    assert nsw > 0
    assert a + b == nsw
    if channel_ids is None:
        channel_ids = slice(None, None, None)
        n_channels = traces.shape[1]
    else:
        n_channels = len(channel_ids)
    t0, t1 = int(sample - a), int(sample + b)
    # Extract the waveforms.
    w = traces[max(0, t0) : t1][:, channel_ids]
    if not isinstance(channel_ids, slice):
        w[:, channel_ids == -1] = 0
    # Deal with side effects.
    if t0 < 0:
        w = np.vstack((np.zeros((nsw - w.shape[0], n_channels), dtype=w.dtype), w))
    if t1 > dur:
        w = np.vstack((w, np.zeros((nsw - w.shape[0], n_channels), dtype=w.dtype)))
    assert w.shape == (nsw, n_channels)
    return w


def extract_waveforms(traces, spike_samples, channel_ids, n_samples_waveforms=None):
    """Extract waveforms for a given set of spikes, on certain channels."""
    # Create the output array.
    ns = len(spike_samples)
    nsw = n_samples_waveforms
    assert nsw > 0, "Please specify n_samples_waveforms > 0"
    nc = len(channel_ids)
    # Extract the spike waveforms.
    out = np.zeros((ns, nsw, nc), dtype=traces.dtype)
    for i, ts in enumerate(spike_samples):
        out[i] = _extract_waveform(
            traces, ts, channel_ids=channel_ids, n_samples_waveforms=nsw
        )[np.newaxis, ...]
    return out


def get_waveforms(self, spike_ids, n_samples_waveforms, channel_ids=None):
    """Return spike waveforms on specified channels."""
    if self.traces is None and self.spike_waveforms is None:
        return
    nsw = n_samples_waveforms
    channel_ids = np.arange(self.n_channels) if channel_ids is None else channel_ids

    if self.spike_waveforms is not None:
        # Load from precomputed spikes.
        return None
        # return get_spike_waveforms(
        #     spike_ids,
        #     channel_ids,
        #     spike_waveforms=self.spike_waveforms,
        #     n_samples_waveforms=nsw,
        # )
    else:
        # Or load directly from raw data (slower).
        spike_samples = self.spike_samples[spike_ids]
        return extract_waveforms(
            self.traces, spike_samples, channel_ids, n_samples_waveforms=nsw
        )


def get_template_ids_from_spikes(cluster_id, spike_clusters, spike_templates):
    spike_ids = np.where(spike_clusters == cluster_id)
    st = spike_templates[spike_ids]
    template_ids, counts = np.unique(st, return_counts=True)
    return template_ids, counts


def get_cluster_channels(cluster_id, spike_clusters, spike_ids, spike_templates):
    """Return the most relevant channels of a cluster."""
    # template_ids, counts = get_template_ids_from_spikes(
    #     cluster_id, spike_clusters, spike_templates
    # )
    spike_ids = np.where(spike_clusters == cluster_id)
    st = spike_templates[spike_ids]
    template_ids, counts = np.unique(st, return_counts=True)
    best_template = template_ids[np.argmax(counts)]
    return best_template


# def get_cluster_spike_waveforms(cluster_id):
#     """Return all spike waveforms of a cluster, on the most relevant channels."""
#     spike_ids = get_cluster_spikes(cluster_id)
#     channel_ids = get_cluster_channels(cluster_id)
#     return get_waveforms(spike_ids, channel_ids)


# %%
def _channels(self, sparse):
    """Gets peak channels for each waveform"""
    tmp = sparse.data
    n_templates, n_samples, n_channels = tmp.shape
    if sparse.cols is None:
        template_peak_channels = np.argmax(tmp.max(axis=1) - tmp.min(axis=1), axis=1)
    else:
        # when the templates are sparse, the first channel
        # is the highest amplitude channel
        template_peak_channels = sparse.cols[:, 0]
    assert template_peak_channels.shape == (n_templates,)
    return template_peak_channels
