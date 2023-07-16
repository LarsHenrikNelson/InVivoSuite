import os
from pathlib import Path
from typing import Literal, Union

import h5py
from scipy import signal

from .filtering_functions import (
    Filters,
    Windows,
    bessel,
    bessel_zero,
    butterworth,
    butterworth_zero,
    ewma_afilt,
    ewma_filt,
    fir_zero_1,
    fir_zero_2,
    median_filter,
    remez_1,
    remez_2,
    savgol_filt,
)
from .lfp_manager import LFPManager
from .spike_manager import SpkManager


class AcqManager(SpkManager, LFPManager):
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
        self.file.close()
        self.file_open = False

    def load_hdf5_file(self, file_path):
        self.file_path = file_path
        self.file = h5py.File(self.file_path, "r+")
        self.file_open = True

    def load_hdf5_acq(self):
        self.file = h5py.File(self.file_path, "r+")
        self.file_open = True

    def downsample(self, array, resample_freq, up_sample):
        ratio = int(self.sample_rate / resample_freq)
        resampled = signal.resample_poly(array, up_sample, up_sample * ratio)
        return resampled

    def filter_array(
        self,
        array,
        sample_rate,
        filter_type: Filters = "butterworth_zero",
        order: Union[None, int] = 301,
        highpass: Union[int, float, None] = None,
        high_width: Union[int, float, None] = None,
        lowpass: Union[int, float, None] = None,
        low_width: Union[int, float, None] = None,
        window: Windows = "hann",
        polyorder: Union[int, None] = None,
    ):
        if filter_type == "median":
            filtered_array = median_filter(array=array, order=order)
        elif filter_type == "bessel":
            filtered_array = bessel(
                array=array,
                order=order,
                sample_rate=sample_rate,
                highpass=highpass,
                lowpass=lowpass,
            )
        elif filter_type == "bessel_zero":
            filtered_array = bessel_zero(
                array=array,
                order=order,
                sample_rate=sample_rate,
                highpass=highpass,
                lowpass=lowpass,
            )
        elif filter_type == "butterworth":
            filtered_array = butterworth(
                array=array,
                order=order,
                sample_rate=sample_rate,
                highpass=highpass,
                lowpass=lowpass,
            )
        elif filter_type == "butterworth_zero":
            filtered_array = butterworth_zero(
                array=array,
                order=order,
                sample_rate=sample_rate,
                highpass=highpass,
                lowpass=lowpass,
            )
        elif filter_type == "fir_zero_1":
            filtered_array = fir_zero_1(
                array=array,
                sample_rate=sample_rate,
                order=order,
                highpass=highpass,
                high_width=high_width,
                lowpass=lowpass,
                low_width=low_width,
                window=window,
            )
        elif filter_type == "fir_zero_2":
            filtered_array = fir_zero_2(
                array=array,
                sample_rate=sample_rate,
                order=order,
                highpass=highpass,
                high_width=high_width,
                lowpass=lowpass,
                low_width=low_width,
                window=window,
            )
        elif filter_type == "remez_1":
            filtered_array = remez_1(
                array=array,
                sample_rate=sample_rate,
                order=order,
                highpass=highpass,
                high_width=high_width,
                lowpass=lowpass,
                low_width=low_width,
            )
        elif filter_type == "remez_2":
            filtered_array = remez_2(
                array=array,
                sample_rate=sample_rate,
                order=order,
                highpass=highpass,
                high_width=high_width,
                lowpass=lowpass,
                low_width=low_width,
            )
        elif filter_type == "savgol":
            filtered_array = savgol_filt(array=array, order=order, polyorder=polyorder)

        elif filter_type == "None":
            filtered_array = array.copy()

        elif filter_type == "subtractive":
            array = fir_zero_2(
                array,
                order=order,
                sample_rate=sample_rate,
                highpass=highpass,
                high_width=high_width,
                lowpass=lowpass,
                low_width=low_width,
                window=window,
            )
            filtered_array = array - array

        elif filter_type == "ewma":
            filtered_array = ewma_filt(
                array=array, window=order, sum_proportion=polyorder
            )
        elif filter_type == "ewma_a":
            filtered_array = ewma_afilt(
                array=array, window=order, sum_proportion=polyorder
            )
        return filtered_array

    def set_filter(
        self,
        acq_type: Literal["spike", "lfp"],
        filter_type: Filters = "fir_zero_2",
        order: Union[None, int] = None,
        highpass: Union[int, float, None] = None,
        high_width: Union[int, float, None] = None,
        lowpass: Union[int, float, None] = None,
        low_width: Union[int, float, None] = None,
        window: Windows = "hann",
        polyorder: Union[int, None] = 0,
        resample_freq=None,
        up_sample=3,
    ):
        if not self.file_open:
            self.load_hdf5_acq()
        if resample_freq is not None:
            sample_rate = resample_freq
        else:
            sample_rate = self.sample_rate
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
            "resample_freq": resample_freq,
            "up_sample": up_sample,
        }
        if not self.file_open:
            self.load_hdf5_acq()
        if self.file.get(acq_type):
            grp = self.file[acq_type]
        else:
            grp = self.file.create_group(acq_type)
        for key, value in input_dict.items():
            if value is None:
                value = "None"
            self.set_grp_attr(grp, key, value)

    def acq(
        self,
        acq_type: Literal["spike", "lfp", "raw"],
        acq_num: int,
        start: int = 0,
        end: int = 24000000,
    ):
        if not self.file_open:
            self.load_hdf5_acq()
        if acq_type == "raw":
            return (
                self.file["acqs"][acq_num - 1, start:end]
                * self.file["coeffs"][acq_num - 1]
            )
        grp = self.file[acq_type]
        input_dict = {
            "sample_rate": grp.attrs["sample_rate"],
            "filter_type": grp.attrs["filter_type"],
            "order": grp.attrs["order"],
            "highpass": grp.attrs["highpass"],
            "high_width": grp.attrs["high_width"],
            "lowpass": grp.attrs["lowpass"],
            "low_width": grp.attrs["low_width"],
            "window": grp.attrs["window"],
            "polyorder": grp.attrs["polyorder"],
        }
        for key, value in input_dict.items():
            if value == "None":
                input_dict[key] = None
        array = (
            self.file["acqs"][acq_num - 1, start:end] * self.file["coeffs"][acq_num - 1]
        )
        acq = self.filter_array(
            array,
            sample_rate=self.sample_rate,
            filter_type=input_dict["filter_type"],
            order=input_dict["order"],
            highpass=input_dict["highpass"],
            high_width=input_dict["high_width"],
            lowpass=input_dict["lowpass"],
            low_width=input_dict["low_width"],
            window=input_dict["window"],
            polyorder=input_dict["polyorder"],
        )
        if grp.attrs["sample_rate"] != self.sample_rate:
            acq = self.downsample(
                acq, input_dict["sample_rate"], grp.attrs["up_sample"]
            )
        return acq

    def set_file_attr(self, attr, data):
        if not self.file_open:
            self.load_hdf5_acq()
        if not self.file.attrs.get(attr):
            self.file.attrs.create(attr, data=data)
        else:
            self.file.attrs[attr] = data

    def set_grp_attr(self, grp, attr, data):
        if not self.file_open:
            self.load_hdf5_acq()
        if not self.file.attrs.get(attr):
            grp.attrs.create(attr, data)
        else:
            grp.attrs[attr] = data

    def close(self):
        if self.file is not None:
            self.file.close()
        self.file_open = False
