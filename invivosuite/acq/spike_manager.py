from . import spike


class SpkManager:
    def find_spikes(
        self, spike_start, spike_end, n_threshold=5.0, p_threshold=0.0, method="std"
    ):
        if not self.file_open:
            self.load_hdf5_acq()
        array = self.acq("spike")
        sample_rate = self.file["spike"].attrs["sample_rate"]
        spikes = spike.find_spikes(
            array,
            spike_start,
            spike_end,
            n_threshold=n_threshold,
            p_threshold=p_threshold,
            method=method,
        )
        hertz = len(spikes) / (len(array) / sample_rate)
        if self.file.attrs.get("spike_freq"):
            self.file.attrs["spike_freq"] = hertz
        else:
            self.file.attrs.create("spike_freq", hertz)
        if self.file.get("spikes"):
            del self.file["spikes"]
            self.file.create_dataset("spikes", data=spikes)
        else:
            self.file.create_dataset(
                "spikes", dtype=spikes.dtype, shape=spikes.shape, maxshape=array.shape
            )
            self.file["spikes"].resize(spikes.shape)
            self.file["spikes"][...] = spikes

    def get_spike_indexes(self):
        if not self.file_open:
            self.load_hdf5_acq()
        return self.file["spikes"][()]

    def get_spikes(self, spike_start, spike_end):
        if not self.file_open:
            self.load_hdf5_acq()
        spike_indexes = self.file["spikes"][()]
        spikes = spike.get_spikes(
            self.acq("spike"), spike_indexes, spike_start, spike_end
        )
        return spikes

    def create_binned_spikes(self, nperseg):
        if not self.file_open:
            self.load_hdf5_acq()
        spikes = self.file["spikes"][()]
        size = self.acq("spike").size
        binned_spikes = spike.bin_spikes(spikes, size, nperseg)
        return binned_spikes

    def get_spike_parameters(self, spike_start=50, spike_end=50):
        if not self.file_open:
            self.load_hdf5_acq()
        spikes = self.get_spikes(spike_start, spike_end)
        data, labels = spike.spike_parameters(spikes)
        return data, labels

    def get_binary_spikes(self):
        if not self.file_open:
            self.load_hdf5_acq()
        return spike.create_binary_spikes(
            self.file["spikes"][()], self.file["array"].size
        )

    def find_spike_bursts(self, method="max_int", **kwargs):
        if not self.file_open:
            self.load_hdf5_acq()
        if method == "max_int":
            bursts = spike.max_int_bursts(
                spikes=self.file["spikes"][()],
                freq=self.file.attrs["spike_freq"],
                fs=self.file["spike"].attrs["sample_rate"],
                output_type="time",
                **kwargs,
            )
        self.set_acq_attr("burst_freq", len(bursts) / (self.rec_len / 60))
        self.set_acq_attr("num_bursts", data=len(bursts))
        ave_burst_len = spike.ave_burst_len(bursts)
        self.set_acq_attr("ave_burst_len", ave_burst_len)
        intra_burst_iei = spike.intra_burst_iei(bursts)
        self.set_acq_attr("intra_burst_iei", intra_burst_iei)
        ave_spikes = spike.ave_spikes_burst(bursts)
        self.set_acq_attr("ave_spikes_burst", ave_spikes)
        ave_iei = spike.ave_iei_burst(bursts)
        self.set_acq_attr("ave_iei_burst", ave_iei)

    def get_spike_bursts(self, method="max_int", **kwargs):
        if not self.file_open:
            self.load_hdf5_acq()
        if method == "max_int":
            bursts = spike.max_int_bursts(
                self.file["spikes"][()],
                self.file.attrs["spike_freq"],
                output_type="time",
                **kwargs,
            )
        return bursts
