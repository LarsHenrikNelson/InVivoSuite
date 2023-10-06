import os
from pathlib import Path
from typing import Literal, Union

import h5py
import numpy as np
from scipy import signal

from .filtering_functions import Filters, Windows, filter_array, iirnotch_zero
from .lfp_manager import LFPManager
from .spike_manager import SpkManager


class AcqManager(SpkManager, LFPManager):
    filters = Filters
    windows = Windows

    def __init__(self):
        self.file = None
        self.file_open = False
        self.spike_data = {}

    def create_hdf5_file(
        self,
        acqs,
        sample_rates,
        coeffs,
        units,
        enabled,
        identifier="test",
        save_path="",
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
        self.set_file_attr("id", identifier)
        self.set_file_attr("start", 0)
        self.set_file_attr("end", acqs.shape[1])
        self.close()

    def open_hdf5_file(self, file_path):
        self.file_path = file_path

    def open(self):
        if not self.file_open:
            self.file = h5py.File(self.file_path, "r+")
            self.file_open = True

    def downsample(self, array, sample_rate, resample_freq, up_sample):
        ratio = int(sample_rate / resample_freq)
        resampled = signal.resample_poly(array, up_sample, up_sample * ratio)
        return resampled

    @property
    def num_channels(self):
        self.open()
        channels = self.file["acqs"].shape[0]
        self.close()
        return channels

    @property
    def shape(self):
        self.open()
        shape = self.file["acqs"].shape
        self.close()
        return shape

    @property
    def id(self):
        self.open()
        id = self.file.attrs["id"]
        self.close()
        return id

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

    def compute_cmr(self):
        start = self.get_file_attr("start")
        end = self.get_file_attr("end")
        nchans = self.num_channels
        cmr = np.zeros((nchans, end - start))
        means = self.get_file_dataset("means")
        means = means.reshape((nchans, 1))
        for i in range((start - end) // 1000):
            start = int(start + i * 1000)
            stop = int(start + i * 1000) + 1000
            array = self.get_file_dataset(
                "acqs", columns=(start, stop)
            ) * self.get_file_dataset("coeffs")
            array -= means
            cmr[start:stop] = array.median(axis=0)
        cmr[end // 1000 :] = array[end // 1000 :].median(axis=1)
        return cmr

    def compute_channel_means(self):
        start = self.get_file_attr("start")
        end = self.get_file_attr("end")
        nchans = self.num_channels
        means = np.zeros(nchans)
        for i in range(nchans):
            array = self.get_file_dataset(
                "acqs", rows=i, columns=(start, end)
            ) * self.get_file_dataset("coeffs", rows=i)
            means[i] = array.mean()
        self.set_file_dataset("chan_means", data=means)

    def acq(
        self,
        acq_num: int,
        acq_type: Literal["spike", "lfp", "raw"],
        map_channel: bool = False,
        electrode: str = "None",
    ):
        start = self.get_file_attr("start")
        end = self.get_file_attr("end")
        if map_channel:
            acq_num = self.get_mapped_channel(electrode, acq_num)
        if electrode != "None":
            data = self.get_grp_dataset("electrode", electrode)
            acq_num += data[0]
        array = self.get_file_dataset(
            "acqs", rows=acq_num, columns=(start, end)
        ) * self.get_file_dataset("coeffs", rows=acq_num)
        array -= array.mean()
        if acq_type == "raw":
            return array
        filter_dict = self.get_filter(acq_type)
        sample_rate = self.get_file_dataset("sample_rate", rows=acq_num)
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
        self.close()
        return acq

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

    def get_grp_dataset(self, grp, data):
        self.open()
        if grp in self.file:
            if data in self.file[grp]:
                grp_data = self.file[grp][data][()]
                self.close()
            else:
                self.close()
                raise AttributeError(f"{data} does not exist.")
        else:
            self.close()
            raise AttributeError(f"{grp} does not exist.")
        return grp_data

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
            if rows is None and columns is None:
                file_dataset = self.file[dataset][()]
                self.close()
            elif columns is None:
                if isinstance(rows, int):
                    file_dataset = self.file[dataset][rows]
                    self.close()
                else:
                    file_dataset = self.file[dataset][rows[0] : rows[1]]
                    self.close()
            elif rows is None:
                if isinstance(columns, int):
                    file_dataset = self.file[dataset][:, columns]
                    self.close()
                else:
                    file_dataset = self.file[dataset][:, columns[0] : columns[1]]
                    self.close()
            else:
                if isinstance(columns, int) and isinstance(rows, int):
                    file_dataset = self.file[dataset][rows, columns]
                    self.close()
                elif isinstance(columns, int) and not isinstance(rows, int):
                    file_dataset = self.file[dataset][rows[0] : rows[1], columns]
                    self.close()
                elif not isinstance(columns, int) and isinstance(rows, int):
                    file_dataset = self.file[dataset][rows, columns[0] : columns[1]]
                    self.close()
                else:
                    file_dataset = self.file[dataset][
                        rows[0] : rows[1], columns[0] : columns[1]
                    ]
                    self.close()
        else:
            self.close()
            raise KeyError(f"{dataset} does not exist.")
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
        self, map_path: Union[str, Path], electrode: str = "None"
    ):
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
        self.set_grp_dataset("channel_maps", electrode, data=chan_map)

    def set_channel_map_from_array(self, chan_map, electrode: str = "None"):
        if chan_map.ndim > 1 and chan_map.shape[1] > 1 and chan_map.shape[0] > 1:
            raise ValueError("File can only inlude one column or row.")
        if chan_map.ndim > 1:
            chan_map = chan_map.flatten()
        self.set_grp_dataset("channel_maps", electrode, data=chan_map)

    def set_channel_map(
        self, chan_map: Union[str, Path, np.ndarray], electrode: str = "None"
    ):
        if isinstance(chan_map, str) or isinstance(chan_map, Path):
            self.set_channel_map_from_file(chan_map, electrode)
        elif isinstance(chan_map, np.ndarray):
            self.set_channel_map_from_array(chan_map, electrode)
        else:
            raise ValueError(f"{chan_map} chan_map must be str, Path or np.ndarray")

    def get_mapped_channel(self, channel: int, electrode: str):
        channel_map = self.get_grp_dataset(electrode, "channel_map")
        mapped_channel = channel_map[channel]
        return mapped_channel

    def set_spike_data(self, dir, id):
        if self.file.get(id):
            self.spike_data[id] = dir
        else:
            raise AttributeError("Must set name for spike first.")

    def save_to_bin(
        self,
        rows: Union[int, tuple[int, int], list[int, int], None] = None,
        columns: Union[int, tuple[int, int], list[int, int], None] = None,
        electrode: Union[str, None] = None,
        save_path=None,
    ):
        if electrode is None:
            if rows is None:
                rows = (0, self.shape[0])
            if columns is None:
                try:
                    start = self.get_file_attr("start")
                    end = self.get_file_attr("end")
                    columns = (start, end)
                except Exception:
                    columns = (0, self.shape[1])
        else:
            if columns is None:
                try:
                    start = self.get_file_attr("start")
                    end = self.get_file_attr("end")
                    columns = (start, end)
                except Exception:
                    columns = (0, self.shape[1])
            if rows is None:
                rows = (0, self.shape[0])
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
        self.set_file_attr("start", start)

    def set_end(self, end):
        self.set_file_attr("end", end)

    def set_electrode(self, electrode: str, array: np.array):
        self.set_grp_dataset("electrodes", electrode, array)
