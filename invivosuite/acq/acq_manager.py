import os
from pathlib import Path
from typing import Literal, Union

import h5py
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
        self.file.create_dataset("id", data=identifier)
        self.file.create_dataset("units", data=units)
        self.file.create_dataset("enabled", data=enabled)
        self.file.set_file_attr("start", 0)
        self.file.set_file_attr("end", acqs.shape[1])
        self.close()

    def set_hdf5_file(self, file_path):
        self.file_path = file_path

    def open(self):
        if not self.file_open:
            self.file = h5py.File(self.file_path, "r+")
            self.file_open = True

    def downsample(self, array, sample_rate, resample_freq, up_sample):
        ratio = int(sample_rate / resample_freq)
        resampled = signal.resample_poly(array, up_sample, up_sample * ratio)
        return resampled

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
        resample_freq: Union[float, None] = None,
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
            "resample_freq": resample_freq,
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
            "resample_freq": grp.attrs["resample_freq"],
            "polyorder": grp.attrs["polyorder"],
            "up_sample": grp.attrs["up_sample"],
        }
        if close:
            self.close()
        return input_dict

    def acq(
        self,
        acq_type: Literal["spike", "lfp", "raw"],
        acq_num: int,
    ):
        self.open()
        start = self.file.attrs["start"]
        end = self.file.attrs["end"]
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
        for key, value in input_dict.items():
            if value == "None":
                input_dict[key] = None
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
        if grp.attrs["resample_freq"] != "None":
            acq = self.downsample(
                acq, sample_rate, grp.attrs["resample_freq"], grp.attrs["up_sample"]
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

    def close(self):
        if self.file is not None:
            self.file.close()
        self.file_open = False

    def set_start(self, start: int = 0):
        self.set_file_attr("start", start)

    def set_end(self, end):
        self.set_file_attr("end", end)
