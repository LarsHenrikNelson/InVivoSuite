import os
from pathlib import Path
from typing import Literal, Union

import h5py
import numpy as np
from scipy import signal

from .filtering_functions import Filters, Windows, filter_array
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
        identifier="",
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
        self.set_file_attr("id", data=identifier)
        self.file.create_dataset("units", data=units)
        self.file.create_dataset("enabled", data=enabled)
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
    def id(self):
        self.open()
        id = self.file.attrs["id"]
        self.close()
        return id

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
        }
        self.open()
        for key, value in input_dict.items():
            if value is None:
                value = "None"
            self.set_grp_attr(acq_type, key, value)
        self.close()

    def get_filter(self, acq_type: Literal["spike", "lfp"], close=True):
        self.open()
        try:
            grp = self.file[acq_type]
        except KeyError:
            raise KeyError(
                f"{acq_type} does not exist. Use set_filter to create {acq_type}."
            )
        input_dict = {
            "filter_type": grp.attrs["filter_type"],
            "order": grp.attrs["order"],
            "highpass": grp.attrs["highpass"],
            "high_width": grp.attrs["high_width"],
            "lowpass": grp.attrs["lowpass"],
            "low_width": grp.attrs["low_width"],
            "window": grp.attrs["window"],
            "sample_rate": grp.attrs["sample_rate"],
            "polyorder": grp.attrs["polyorder"],
            "up_sample": grp.attrs["up_sample"],
        }
        for key, value in input_dict.items():
            if value == "None":
                input_dict[key] = None
        if close:
            self.close()
        return input_dict

    def acq(
        self,
        acq_type: Literal["spike", "lfp", "raw"],
        acq_num: int,
        electrode: Union[None, str] = None,
        map_channel: bool = False,
    ):
        self.open()
        start = self.file.attrs["start"]
        end = self.file.attrs["end"]
        if map_channel and self.file.get("channel_map"):
            acq_num = self.file["channel_map"][acq_num]
        if electrode is not None:
            data = self._get_grp_dataset("electrode", electrode)
            acq_num += data[0]
        array = self.file["acqs"][acq_num, start:end] * self.file["coeffs"][acq_num]
        if acq_type == "raw":
            return array
        if not self.file.get(acq_type):
            raise KeyError(
                f"{acq_type} does not exist. Use set_filter to create {acq_type}."
            )
        else:
            grp = self.file[acq_type]
        input_dict = self.get_filter(acq_type, close=False)
        sample_rate = self.file["sample_rate"][acq_num]
        acq = filter_array(
            array,
            sample_rate=sample_rate,
            filter_type=input_dict["filter_type"],
            order=input_dict["order"],
            highpass=input_dict["highpass"],
            high_width=input_dict["high_width"],
            lowpass=input_dict["lowpass"],
            low_width=input_dict["low_width"],
            window=input_dict["window"],
            polyorder=input_dict["polyorder"],
        )
        if grp.attrs["sample_rate"] != sample_rate:
            acq = self.downsample(
                acq, sample_rate, grp.attrs["sample_rate"], grp.attrs["up_sample"]
            )
        self.close()
        return acq

    def set_file_attr(self, attr, data):
        self.open()
        if not self.file.attrs.get(attr):
            self.file.attrs.create(attr, data=data)
        else:
            self.file.attrs[attr] = data
        self.close()

    def set_file_dataset(self, name, data):
        self.open()
        if self.file.get(name):
            del self.file[name]
            self.file.create_dataset(name, data=data)
        else:
            self.file.create_dataset(name, data=data)
        self.close()

    def set_grp_attr(self, grp_name, attr, data):
        self.open()
        if self.file.get(grp_name):
            grp = self.file[grp_name]
        else:
            grp = self.file.create_group(grp_name)
        if not self.file.attrs.get(attr):
            grp.attrs.create(attr, data)
        else:
            grp.attrs[attr] = data
        self.close()

    def set_grp_dataset(self, grp_name, name, data):
        self.open()
        if self.file.get(grp_name):
            grp = self.file[grp_name]
        else:
            grp = self.file.create_group(grp_name)
        if grp.get(name):
            del grp[name]
            grp.create_dataset(name, data=data)
        else:
            grp.create_dataset(name, data=data)
        self.close()

    def _get_grp_dataset(self, grp, data):
        if self.file.get(grp):
            if self.file[grp].get(data):
                grp_data = self.file[grp][data][()]
            else:
                raise AttributeError(f"{data} does not exist.")
        else:
            raise AttributeError(f"{grp} does not exist.")
        return grp_data

    def get_grp_attr(self, grp_name, name):
        self.open()
        if self.file.get(grp_name):
            grp = self.file[grp_name]
            if grp.attrs.get(name):
                value = grp.attrs[name]
                self.close()
                return value
            else:
                self.close()
                raise KeyError(f"{name} is not an attribute of {grp_name}.")
        else:
            self.close()
            raise KeyError(f"{grp_name} does not exist.")

    def set_channel_map(self, map_path):
        path = Path(map_path)
        if path.suffix == ".xlsx":
            raise NotImplementedError("xlsx files are not supported")
        elif path.suffix == ".csv":
            chan_map = np.loadtxt(path, delimiter=",", dtype=np.int16)
        elif path.suffix == ".txt":
            chan_map = np.loadtxt(path, dtype=np.int16)
        else:
            raise NotImplementedError("File type not recognized.")
        self.set_file_dataset("channel_map", data=chan_map)

    def set_spike_data(self, dir, id):
        if self.file.get(id):
            self.spike_data[id] = dir
        else:
            raise AttributeError("Must set name for spike first.")

    def save_to_bin(
        self, acqs=None, electrode=None, save_path=None, map_channels=False
    ):
        self.open()
        start = self.file.attrs["start"]
        end = self.file.attrs["end"]
        if acqs is None:
            shape = self.file["acqs"].shape
            acqs = [0]
            acqs.append(shape[0])
        if electrode is not None and acqs is not None:
            data = self._get_grp_dataset("electrode", electrode)
            acqs[0] += data[0]
            acqs[1] += data[0]
        if map_channels and self.file.get("channel_map"):
            acqs = np.zeros((shape[0], end - start))
            channel_map = self.file["channel_map"][()]
            for i in range(shape[1]):
                acqs[i] = self.file["acqs"][channel_map[i], start:end]
        else:
            acqs = np.asarray(
                self.file["acqs"][acqs[0] : acqs[1], start:end],
                dtype=np.int16,
                order="C",
            )
        self.close()

        # Need to transpose to (n_samples, n_channels) fo kilosort
        acqs = acqs.T
        if save_path is None:
            save_path = Path(self.save_path).parents[0] / Path(self.save_path).stem
            save_path = save_path / ".bin"
        else:
            save_path = Path(save_path) / ".bin"
        acqs.tofile(save_path)

    def close(self):
        if self.file is not None:
            self.file.close()
        self.file_open = False

    def set_start(self, start: int = 0):
        self.set_file_attr("start", start)

    def set_end(self, end):
        self.set_file_attr("end", end)
