import os
from pathlib import Path
from typing import Literal, Union

import h5py
import numpy as np
from scipy import signal

from .filtering_functions import Filters, Windows, filter_array, iirnotch_zero
from .lfp_manager import LFPManager
from .spike_manager import SpkManager
from .spike_lfp_manager import SpkLFPManager
from ..functions.signal_functions import envelopes_idx, whitening_matrix


class AcqManager(SpkManager, LFPManager, SpkLFPManager):
    filters = Filters
    windows = Windows

    def __init__(self):
        self.file = None
        self.file_open = False
        self._set_cwt = False

    def create_hdf5_file(
        self,
        acqs,
        wb_channels,
        sample_rates,
        coeffs,
        units,
        enabled,
        identifier="test",
        save_path="",
        ai=None,
    ):
        if save_path == "":
            save_path = os.getcwd()
            self.file_path = f"{save_path}.hdf5"
        else:
            save_path = Path(save_path)
            if not save_path.exists():
                save_path.mkdir()
            save_path = save_path / identifier
            self.file_path = f"{save_path}.hdf5"
        self.file = h5py.File(self.file_path, "a")
        self.file.create_dataset("acqs", data=acqs)
        self.file.create_dataset("coeffs", data=coeffs)
        self.file.create_dataset("sample_rate", data=sample_rates)
        self.file.create_dataset("units", data=units)
        self.file.create_dataset("enabled", data=enabled)
        self.file.create_dataset("wb_channels", data=wb_channels)
        self.set_file_attr("id", identifier)
        self.set_file_attr("start", 0)
        self.set_file_attr("end", acqs.shape[1])
        self.set_probe("all", [0, acqs.shape[0]])
        self.compute_means()
        if ai is not None:
            self.set_grp_dataset("ai", "acq", ai[0])
            self.set_grp_dataset("ai", "fs", ai[1])
            self.set_grp_dataset("ai", "coeffs", ai[2])
            self.set_grp_dataset("ai", "units", ai[3])
        self.close()

    def load_hdf5(self, file_path):
        self.file_path = file_path

    def load_kilosort(self, file_directory, load_type: str = "r+"):
        self.ks_directory = Path(file_directory)
        self.load_ks_data(load_type=load_type)

    def open(self):
        if not self.file_open:
            self.file = h5py.File(self.file_path, "r+")
            self.file_open = True

    def downsample(self, array, sample_rate, resample_freq, up_sample):
        ratio = int(sample_rate / resample_freq)
        resampled = signal.resample_poly(array, up_sample, up_sample * ratio)
        return resampled

    @property
    def n_chans(self):
        self.open()
        channels = self.file["acqs"].shape[0]
        self.close()
        return channels

    @property
    def start(self):
        return self.get_file_attr("start")

    @property
    def end(self):
        return self.get_file_attr("end")

    @property
    def shape(self):
        self.open()
        shape = self.file["acqs"].shape
        self.close()
        return shape

    def index_to_time(self, index, fs, output_type: Literal["samples", "ms", "sec"]):
        if output_type == "samples":
            return index
        if output_type == "ms":
            return index / (fs / 1000)
        elif output_type == "sec":
            return index / fs
        else:
            raise AttributeError("Outputype not recognized.")

    # @property
    # def id(self):
    #     self.open()
    #     id = self.file.attrs["id"]
    #     self.close()
    #     return id

    @property
    def probes(self):
        self.open()
        if "probes" in self.file:
            p = list(self.file["probes"].keys())
            self.close()
            return p
        else:
            return None

    # def channel_map(self):
    #     self.open()
    #     if self.file.get("channel_map"):
    #         channel_map = self.file["channel_map"][()]
    #         self.close()
    #         return channel_map
    #     else:
    #         self.close()
    #         return None

    def set_filter(
        self,
        acq_type: Literal["spike", "lfp"],
        filter_type: Filters = "butterworth_zero",
        order: Union[None, int] = None,
        highpass: Union[int, float, None] = None,
        high_width: Union[int, float, None] = None,
        lowpass: Union[int, float, None] = None,
        low_width: Union[int, float, None] = None,
        window: Windows = "hann",
        polyorder: Union[int, None] = 0,
        sample_rate: Union[float, int] = 40000,
        up_sample=3,
        notch_filter: bool = False,
        notch_freq: float = 60.0,
        notch_q: float = 30.0,
    ):
        input_dict = {
            "filter_type": filter_type,
            "order": order,
            "highpass": highpass,
            "high_width": high_width,
            "lowpass": lowpass,
            "low_width": low_width,
            "window": window,
            "polyorder": polyorder,
            "sample_rate": sample_rate,
            "up_sample": up_sample,
            "notch_filter": notch_filter,
            "notch_freq": notch_freq,
            "notch_q": notch_q,
        }
        for key, value in input_dict.items():
            if value is None:
                value = "None"
            self.set_grp_attr(acq_type, key, value)

    def get_filter(self, acq_type: Literal["spike", "lfp"]):
        self.open()
        try:
            grp = self.file[acq_type]
        except KeyError:
            raise KeyError(
                f"{acq_type} does not exist. Use set_filter to create {acq_type}."
            )
        input_dict = dict(grp.attrs)
        for key, value in input_dict.items():
            if value == "None":
                input_dict[key] = None
        self.close()
        return input_dict

    def compute_means(self):
        chans = self.get_grp_dataset("probes", "all")
        nchans = int(chans[1] - chans[0])
        means = np.zeros(nchans)
        for i in range(nchans):
            array = self.get_file_dataset("acqs", rows=i) * self.get_file_dataset(
                "coeffs", rows=i
            )
            means[i] = np.mean(array)
        self.set_file_dataset("means", means)

    def compute_virtual_ref(
        self,
        ref_type: Literal["cmr", "car"] = "cmr",
        probe: str = "all",
        bin_size: int = 0,
    ):
        start = 0
        self.open()
        end = self.file["acqs"].shape[1]
        self.close()
        if ref_type == "cmr":
            ref = np.median
        elif ref_type == "car":
            ref = np.mean
        else:
            raise AttributeError("ref_type must be cmr or car.")
        if probe == "all":
            chans = (0, self.n_chans)
        else:
            chans = self.get_grp_dataset("probes", probe)
        if bin_size != 0:
            cmr = np.zeros(end - start)
            means = np.zeros((chans[1] - chans[0], 1))
            for i in range((end - start) // bin_size):
                begin = int(start + i * bin_size)
                stop = begin + bin_size
                array = self.get_file_dataset(
                    "acqs", rows=chans, columns=(begin, stop)
                ) * self.get_file_dataset("coeffs", rows=chans).reshape(
                    (chans[1] - chans[0], 1)
                ) - self.get_file_dataset("means", rows=chans).reshape(
                    chans[1] - chans[0], 1
                )
                array -= means
                cmr[begin:stop] = ref(array, axis=0)
            get_the_rest = (start - end) % bin_size
            if get_the_rest > 0:
                array = self.get_file_dataset(
                    "acqs", rows=chans, columns=(end - get_the_rest, end)
                ) * self.get_file_dataset("coeffs", rows=chans).reshape(
                    (chans[1] - chans[0], 1)
                )
                means = array.mean(axis=1, keepdims=True)
                array -= means
                cmr[(end - get_the_rest) :] = ref(array[-get_the_rest:], axis=0)
            self.set_grp_dataset(ref_type, probe, cmr)
        else:
            array = self.get_file_dataset(
                "acqs", rows=chans, columns=(start, end)
            ) * self.get_file_dataset("coeffs", rows=chans).reshape(
                (chans[1] - chans[0], 1)
            ) - self.get_file_dataset("means", rows=chans).reshape(
                chans[1] - chans[0], 1
            )
            cmr = ref(array, axis=0)
            self.set_grp_dataset(ref_type, probe, cmr)

    def compute_whitening_matrix(
        self,
        neighbors: int,
        probe: str = "all",
        acq_type: Literal["spike", "lfp"] = "spike",
        ref: bool = False,
        ref_type: Literal["cmr", "car"] = "cmr",
        ref_probe: str = "all",
        map_channel: bool = True,
    ):
        probe_chans = self.get_grp_dataset("probes", probe)
        probe_chans -= probe_chans[0]
        total_chans = probe_chans[1] - probe_chans[0]
        start = self.get_file_attr("start")
        end = self.get_file_attr("end")
        W = np.zeros((total_chans, total_chans))
        if neighbors != total_chans:
            for channel in range(probe_chans[1]):
                acquisitions = self.get_multichans(
                    "spike",
                    ref=ref,
                    channel=channel,
                    nchans=neighbors,
                    ref_probe=ref_probe,
                    ref_type=ref_type,
                    map_channel=map_channel,
                    probe=probe,
                    start=start,
                    end=end,
                )
                start_chan = int(max(0, channel - neighbors))
                end_chan = int(min(total_chans, channel + neighbors + 1))
                W_temp = whitening_matrix(
                    acquisitions=acquisitions,
                )
                ilocal = min(0, neighbors)
                W[channel, start_chan:end_chan] = W_temp[ilocal, :]
        else:
            acquisitions = np.zeros((probe_chans[1], end - start))
            for i in range(probe_chans[1]):
                acquisitions[i, :] = self.acq(
                    i,
                    acq_type=acq_type,
                    ref=ref,
                    ref_type=ref_type,
                    ref_probe=ref_probe,
                    map_channel=map_channel,
                    probe=probe,
                    start=start,
                    end=end,
                )
            W = whitening_matrix(acquisitions)
        self.set_grp_dataset("whitening_matrix", probe, W)

    def acq(
        self,
        channel: int,
        acq_type: Literal["spike", "lfp", "wideband"],
        ref: bool = False,
        ref_type: Literal["cmr", "car"] = "cmr",
        ref_probe: str = "all",
        map_channel: bool = False,
        probe: str = "all",
        start: int = 0,
        end: int = 0,
    ):
        start = self.start + start
        if end > 0:
            end = self.start + end
        else:
            end = self.end
        channel = self.get_mapped_channel(channel, probe=probe, map_channel=map_channel)
        array = self.get_file_dataset(
            "acqs", rows=int(channel), columns=(start, end)
        ) * self.get_file_dataset("coeffs", rows=int(channel)) - self.get_file_dataset(
            "means", rows=int(channel)
        )
        if ref:
            median = self.get_grp_dataset(ref_type, ref_probe)
            array -= median[start:end]
        if acq_type == "wideband":
            return array
        filter_dict = self.get_filter(acq_type)
        sample_rate = self.get_file_dataset("sample_rate", rows=int(channel))
        acq = filter_array(
            array,
            sample_rate=sample_rate,
            filter_type=filter_dict["filter_type"],
            order=filter_dict["order"],
            highpass=filter_dict["highpass"],
            high_width=filter_dict["high_width"],
            lowpass=filter_dict["lowpass"],
            low_width=filter_dict["low_width"],
            window=filter_dict["window"],
            polyorder=filter_dict["polyorder"],
        )
        if filter_dict["notch_filter"]:
            acq = iirnotch_zero(
                array,
                freq=filter_dict["notch_freq"],
                q=filter_dict["notch_q"],
                fs=sample_rate,
            )
        if filter_dict["sample_rate"] != sample_rate:
            acq = self.downsample(
                acq, sample_rate, filter_dict["sample_rate"], filter_dict["up_sample"]
            )
        return acq

    def get_multichans(
        self,
        acq_type: Literal["spike", "lfp", "wideband"],
        channel: Union[int, None] = None,
        nchans: Union[int, None] = None,
        ref: bool = False,
        ref_type: Literal["cmr", "car"] = "cmr",
        ref_probe: str = "all",
        map_channel: bool = False,
        probe: str = "all",
        whiten: bool = False,
        start: int = 0,
        end: int = 0,
    ):
        data = self.get_grp_dataset("probes", probe)
        total_chans = data[1] - data[0]
        start = self.start + start
        if end > 0:
            end = self.start + end
        else:
            end = self.end

        if channel is not None:
            # start_chan = channel - nchans
            end_chan = channel + nchans
            start_chan = max(0, int(channel - nchans))
            end_chan = min(total_chans, int(channel + nchans))
            multi_acq = np.zeros((end_chan - start_chan, int(end - start)))
        else:
            start_chan = 0
            end_chan = data[1] - data[0]
            multi_acq = np.zeros((end_chan, int(end - start)))
        for channel in range(start_chan, end_chan):
            multi_acq[int(channel - start_chan), :] = self.acq(
                channel=channel,
                acq_type=acq_type,
                ref=ref,
                ref_type=ref_type,
                ref_probe=ref_probe,
                map_channel=map_channel,
                probe=probe,
                start=start,
                end=end,
            )
        if whiten:
            W = self.get_grp_dataset("whitening_matrix", probe)
            multi_acq = W[start_chan:end_chan, start_chan:end_chan] @ multi_acq
        return multi_acq

    def get_groups(self):
        self.open()
        groups = self.file.keys()
        self.close()
        return groups

    def envelope(self, acq: np.ndarray, interp: bool = True):
        env_min, env_max = envelopes_idx(acq, interp=interp)
        return env_min, env_max

    def set_file_attr(self, attr, data):
        self.open()
        if attr not in self.file.attrs:
            self.file.attrs.create(attr, data=data)
        else:
            self.file.attrs[attr] = data
        self.close()

    def set_file_dataset(self, name, data):
        self.open()
        if name in self.file:
            del self.file[name]
            self.file.create_dataset(name, data=data)
        else:
            self.file.create_dataset(name, data=data)
        self.close()

    def set_grp_attr(self, grp_name, attr, data):
        self.open()
        if grp_name in self.file:
            grp = self.file[grp_name]
        else:
            grp = self.file.create_group(grp_name)
        if attr not in self.file.attrs:
            grp.attrs.create(attr, data)
        else:
            grp.attrs[attr] = data
        self.close()

    def set_grp_dataset(self, grp_name, name, data):
        self.open()
        if grp_name in self.file:
            grp = self.file[grp_name]
        else:
            grp = self.file.create_group(grp_name)
        if name in grp:
            del grp[name]
            grp.create_dataset(name, data=data)
        else:
            grp.create_dataset(name, data=data)
        self.close()

    def get_grp_dataset(
        self,
        grp,
        dataset,
        rows: Union[int, tuple[int, int], list[int, int], None] = None,
        columns: Union[int, tuple[int, int], list[int, int], None] = None,
    ):
        self.open()
        if grp in self.file:
            if dataset in self.file[grp]:
                group = self.file[grp]
                file_dataset = self._get_data(group, dataset, rows, columns)
                self.close()
                return file_dataset
            else:
                self.close()
                raise AttributeError(f"{dataset} does not exist.")
        else:
            self.close()
            raise AttributeError(f"{grp} does not exist.")

    def get_grp_attr(self, grp_name, name):
        self.open()
        if grp_name in self.file:
            if name in self.file[grp_name].attrs:
                value = self.file[grp_name].attrs[name]
                self.close()
                return value
            else:
                self.close()
                raise KeyError(f"{name} is not an attribute of {grp_name}.")
        else:
            self.close()
            raise KeyError(f"{grp_name} does not exist.")

    def get_file_attr(self, attr: str):
        self.open()
        if attr in self.file.attrs:
            file_attr = self.file.attrs[attr]
            self.close()
        else:
            self.close()
            raise KeyError(f"{attr} does not exist.")
        return file_attr

    def get_file_dataset(
        self,
        dataset: str,
        rows: Union[int, tuple[int, int], list[int, int], None] = None,
        columns: Union[int, tuple[int, int], list[int, int], None] = None,
    ):
        self.open()
        if dataset in self.file:
            file_dataset = self._get_data(self.file, dataset, rows, columns)
            self.close()
            return file_dataset
        else:
            self.close()
            raise KeyError(f"{dataset} does not exist.")

    def _get_data(
        self,
        grp,
        dataset: str,
        rows: Union[int, tuple[int, int], list[int, int], None] = None,
        columns: Union[int, tuple[int, int], list[int, int], None] = None,
    ):
        if rows is None and columns is None:
            file_dataset = grp[dataset][()]
        elif columns is None:
            if isinstance(rows, int):
                file_dataset = grp[dataset][rows]
            else:
                file_dataset = grp[dataset][rows[0] : rows[1]]
        elif rows is None:
            if isinstance(columns, int):
                file_dataset = grp[dataset][:, columns]
            else:
                file_dataset = grp[dataset][:, columns[0] : columns[1]]
        else:
            if isinstance(columns, int) and isinstance(rows, int):
                file_dataset = grp[dataset][rows, columns]
            elif isinstance(columns, int) and not isinstance(rows, int):
                file_dataset = grp[dataset][rows[0] : rows[1], columns]
            elif not isinstance(columns, int) and isinstance(rows, int):
                file_dataset = grp[dataset][rows, columns[0] : columns[1]]
            else:
                file_dataset = grp[dataset][rows[0] : rows[1], columns[0] : columns[1]]
        return file_dataset

    def get_grp_attrs(self, grp_name: str):
        self.open()
        if grp_name in self.file:
            grp = self.file[grp_name]
            attrs = dict(grp.attrs)
            self.close()
            return attrs
        else:
            self.close()
            raise KeyError(f"{grp_name} settings do not exist in file. Use set_pxx")

    def set_channel_map_from_file(
        self, map_path: Union[str, Path], probe: str = "None"
    ):
        """Set the probe channel map from a .csv or .txt file. Excel files are not accepted. The channel map must be a single row or column of numbers.

        Args:
            map_path (Union[str, Path]): File name to read.
            probe (str, optional): Name of the probe. Defaults to "None".

        Raises:
            NotImplementedError: .xlsx files are not supported.
            NotImplementedError: File type not recognized.
            ValueError: Channel contains more than one column or one row.
        """
        path = Path(map_path)
        if path.suffix == ".xlsx":
            raise NotImplementedError("xlsx files are not supported")
        elif path.suffix == ".csv":
            chan_map = np.loadtxt(path, delimiter=",", dtype=np.int16)
        elif path.suffix == ".txt":
            chan_map = np.loadtxt(path, dtype=np.int16)
        else:
            raise NotImplementedError("File type not recognized.")
        if chan_map.ndim > 1 and chan_map.shape[1] > 1 and chan_map.shape[0] > 1:
            raise ValueError("File can only inlude one column or row.")
        if chan_map.ndim > 1:
            chan_map = chan_map.flatten()
        self.set_grp_dataset("channel_maps", probe, data=chan_map)

    def set_channel_map_from_array(self, chan_map, probe: str = "None"):
        if chan_map.ndim > 1 and chan_map.shape[1] > 1 and chan_map.shape[0] > 1:
            raise ValueError("File can only inlude one column or row.")
        if chan_map.ndim > 1:
            chan_map = chan_map.flatten()
        self.set_grp_dataset("channel_maps", probe, data=chan_map)

    def set_channel_map(
        self, chan_map: Union[str, Path, np.ndarray], probe: str = "None"
    ):
        """Set the probe channel map from a .csv or .txt file. Excel files are not accepted. The channel map must be a single row or column of numbers.

        Args:
            chan_map (Union[str, Path, np.ndarray]): Filename, Path or np.ndarray containing channel map.
            probe (str, optional): Name of the probe. Defaults to "None".

        Raises:
            ValueError: _description_
        """
        if isinstance(chan_map, str) or isinstance(chan_map, Path):
            self.set_channel_map_from_file(chan_map, probe)
        elif isinstance(chan_map, np.ndarray):
            self.set_channel_map_from_array(chan_map, probe)
        else:
            raise ValueError(f"{chan_map} chan_map must be str, Path or np.ndarray")

    def get_mapped_channel(
        self, channel: int, probe: str = "none", map_channel: bool = False
    ):
        probe = probe.lower()
        if probe != "none" and map_channel:
            channel_map = self.get_grp_dataset("channel_maps", probe)
            channel = channel_map[channel]
            data = self.get_grp_dataset("probes", probe)
            channel = int(channel + data[0])
        elif probe != "none":
            data = self.get_grp_dataset("probes", probe)
            channel = int(channel + data[0])
        return channel

    def save_kilosort_bin(
        self,
        rows: Union[int, tuple[int, int], list[int, int], None] = None,
        columns: Union[int, tuple[int, int], list[int, int], None] = None,
        probe: Union[str, None] = None,
        save_path=None,
    ):
        if probe is None:
            if rows is None:
                rows = (0, self.shape[1])
            if columns is None:
                start = self.start
                end = self.end
                columns = (start, end)
            else:
                if columns[0] > 0:
                    start = self.start + start
                else:
                    start = self.start
                if columns[1] > 0:
                    end = self.start + end
                else:
                    end = self.end
                columns = (start, end)
        else:
            if columns is None:
                start = self.start
                end = self.end
                columns = (start, end)
            else:
                if columns[0] > 0:
                    start = self.start + start
                else:
                    start = self.start
                if columns[1] > 0:
                    end = self.start + end
                else:
                    end = self.end
                columns = (start, end)
            if rows is None and probe is None:
                rows = (0, self.shape[1])
            elif probe is not None:
                rows = self.get_grp_dataset("probes", probe)
        acqs = self.get_file_dataset("acqs", rows, columns)
        acqs = acqs.T
        if save_path is None:
            save_path = Path(self.file_path).with_suffix(".bin")
        else:
            save_path = (Path(save_path) / Path(self.file_path).stem).with_suffix(
                ".bin"
            )
        acqs.tofile(save_path)

    def close(self):
        if self.file is not None:
            self.file.close()
        self.file_open = False

    def set_start(self, start: int = 0):
        """Set the start of the recording in samples.

        Args:
            start (int, optional): Sample to start the recording. Defaults to 0.
        """
        self.set_file_attr("start", start)

    def set_end(self, end: int):
        """Set the end of the recording in samples. Defaults to the length of the recording

        Args:
            end (int): Sample to end the recording.

        Raises:
            ValueError: _description_
        """
        self.open()
        len_of_rec = self.file["acqs"].shape[1]
        if end > len_of_rec:
            self.close()
            raise ValueError(
                f"{end} is longer than the length of the recording ({len_of_rec})"
            )
        self.close()
        self.set_file_attr("end", end)

    def set_probe(self, probe: str, array: np.array):
        self.set_grp_dataset("probes", probe, array)

    def set_callback(self, callback: callable):
        self.callback = callback