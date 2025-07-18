from typing import Literal, Union

import numpy as np
import pywt
from scipy import signal

Filters = Literal[
    "remez_2",
    "remez_1",
    "fir_zero_2",
    "fir_zero_1",
    "savgol",
    "ewma",
    "ewma_a",
    "median",
    "bessel",
    "butterworth",
    "bessel_zero",
    "butterworth_zero",
]

Windows = Literal[
    "hann",
    "hamming",
    "blackmanharris",
    "barthann",
    "nuttall",
    "blackman",
    "tukey",
    "kaiser",
    "gaussian",
    "parzen",
]


def median_filter(array: Union[np.ndarray, list], order: int):
    if isinstance(order, float):
        order = int(order)
    filt_array = signal.medfilt(array, order)
    return filt_array


def bessel(
    array: Union[np.ndarray, list],
    order: int,
    sample_rate: int,
    highpass: Union[int, None] = None,
    lowpass: Union[int, None] = None,
):
    if len(array.shape) > 1:
        axis = 1
    else:
        axis = -1
    if highpass is not None and lowpass is not None:
        sos = signal.bessel(
            N=order,
            Wn=[highpass, lowpass],
            btype="bandpass",
            output="sos",
            fs=sample_rate,
        )
        filt_array = signal.sosfilt(sos, array, axis=axis)
        return filt_array
    elif highpass is not None and lowpass is None:
        sos = signal.bessel(
            order, Wn=highpass, btype="highpass", output="sos", fs=sample_rate
        )
        filt_array = signal.sosfilt(sos, array, axis=axis)
    elif highpass is None and lowpass is not None:
        sos = signal.bessel(
            order, Wn=lowpass, btype="lowpass", output="sos", fs=sample_rate
        )
        filt_array = signal.sosfilt(sos, array, axis=axis)
    return filt_array


def bessel_zero(
    array: Union[np.ndarray, list],
    order: int,
    sample_rate: int,
    highpass: Union[int, None] = None,
    lowpass: Union[int, None] = None,
):
    if len(array.shape) > 1:
        axis = 1
    else:
        axis = -1
    if highpass is not None and lowpass is not None:
        sos = signal.bessel(
            N=order,
            Wn=[highpass, lowpass],
            btype="bandpass",
            output="sos",
            fs=sample_rate,
        )
        filt_array = signal.sosfiltfilt(sos, array, axis=axis)
        return filt_array
    elif highpass is not None and lowpass is None:
        sos = signal.bessel(
            order, Wn=highpass, btype="highpass", output="sos", fs=sample_rate
        )
        filt_array = signal.sosfiltfilt(sos, array, axis=axis)
    elif highpass is None and lowpass is not None:
        sos = signal.bessel(
            order, Wn=lowpass, btype="lowpass", output="sos", fs=sample_rate
        )
        filt_array = signal.sosfiltfilt(sos, array, axis=axis)
    return filt_array


def butterworth(
    array: Union[np.ndarray, list],
    order: int,
    sample_rate: int,
    highpass: Union[int, None] = None,
    lowpass: Union[int, None] = None,
):
    if len(array.shape) > 1:
        axis = 1
    else:
        axis = -1
    if highpass is not None and lowpass is not None:
        sos = signal.butter(
            N=order,
            Wn=[highpass, lowpass],
            btype="bandpass",
            output="sos",
            fs=sample_rate,
        )
        filt_array = signal.sosfilt(sos, array, axis=axis)
        return filt_array
    elif highpass is not None and lowpass is None:
        sos = signal.butter(
            order, Wn=highpass, btype="highpass", output="sos", fs=sample_rate
        )
        filt_array = signal.sosfilt(sos, array, axis=axis)
    elif highpass is None and lowpass is not None:
        sos = signal.butter(
            order, Wn=lowpass, btype="lowpass", output="sos", fs=sample_rate
        )
        filt_array = signal.sosfilt(sos, array, axis=axis)
    return filt_array


def butterworth_zero(
    array: Union[np.ndarray, list],
    order: int,
    sample_rate: int,
    highpass: Union[int, None] = None,
    lowpass: Union[int, None] = None,
):
    if len(array.shape) > 1:
        axis = 1
    else:
        axis = -1
    if highpass is not None and lowpass is not None:
        sos = signal.butter(
            N=order,
            Wn=[highpass, lowpass],
            btype="bandpass",
            output="sos",
            fs=sample_rate,
        )
        filt_array = signal.sosfiltfilt(sos, array, axis=axis)
        return filt_array
    elif highpass is not None and lowpass is None:
        sos = signal.butter(
            order, Wn=highpass, btype="highpass", output="sos", fs=sample_rate
        )
        filt_array = signal.sosfiltfilt(sos, array, axis=axis)
    elif highpass is None and lowpass is not None:
        sos = signal.butter(
            order, Wn=lowpass, btype="lowpass", output="sos", fs=sample_rate
        )
        filt_array = signal.sosfiltfilt(sos, array, axis=axis)
    return filt_array


def fir_zero_1(
    array: Union[np.ndarray, list],
    order: int,
    sample_rate: int,
    highpass: Union[int, None] = None,
    high_width: Union[int, None] = None,
    lowpass: Union[int, None] = None,
    low_width: Union[int, None] = None,
    window: str = "hann",
):
    if len(array.shape) > 1:
        axis = 1
    else:
        axis = -1
    if highpass is not None and lowpass is not None:
        filt = signal.firwin2(
            order,
            freq=[
                0,
                highpass - high_width,
                highpass,
                lowpass,
                lowpass + low_width,
                sample_rate / 2,
            ],
            gain=[0, 0, 1, 1, 0, 0],
            window=window,
            fs=sample_rate,
        )
        filt_array = signal.filtfilt(filt, 1.0, array, axis=axis)
    elif highpass is not None and lowpass is None:
        filt = signal.firwin2(
            order,
            freq=[0, highpass - high_width, highpass, sample_rate / 2],
            gain=[0, 0, 1, 1],
            window=window,
            fs=sample_rate,
        )
        filt_array = signal.filtfilt(filt, 1.0, array, axis=axis)
    elif highpass is None and lowpass is not None:
        filt = signal.firwin2(
            order,
            freq=[0, lowpass, lowpass + low_width, sample_rate / 2],
            gain=[1, 1, 0, 0],
            window=window,
            fs=sample_rate,
        )
        filt_array = signal.filtfilt(filt, 1.0, array, axis=axis)
    return filt_array


def fir_zero_2(
    array: Union[np.ndarray, list],
    order: int,
    sample_rate: int,
    highpass: Union[int, None] = None,
    high_width: Union[int, None] = None,
    lowpass: Union[int, None] = None,
    low_width: Union[int, None] = None,
    window: str = "hann",
):
    if len(array.shape) > 1:
        axis = 1
    else:
        axis = -1
    grp_delay = int(0.5 * (order - 1))
    if highpass is not None and lowpass is not None:
        filt = signal.firwin2(
            order,
            freq=[
                0,
                highpass - high_width,
                highpass,
                lowpass,
                lowpass + low_width,
                sample_rate / 2,
            ],
            gain=[0, 0, 1, 1, 0, 0],
            window=window,
            fs=sample_rate,
        )
        acq1 = np.hstack((array, np.zeros(grp_delay)))
        filt_acq = signal.lfilter(filt, 1.0, acq1, axis=axis)
        filt_array = filt_acq[grp_delay:]
    elif highpass is not None and lowpass is None:
        hi = signal.firwin2(
            order,
            [0, highpass - high_width, highpass, sample_rate / 2],
            gain=[0, 0, 1, 1],
            window=window,
            fs=sample_rate,
        )
        acq1 = np.hstack((array, np.zeros(grp_delay)))
        filt_acq = signal.lfilter(hi, 1.0, acq1, axis=axis)
        filt_array = filt_acq[grp_delay:]
    elif highpass is None and lowpass is not None:
        lo = signal.firwin2(
            order,
            [0, lowpass, lowpass + low_width, sample_rate / 2],
            gain=[1, 1, 0, 0],
            window=window,
            fs=sample_rate,
        )
        acq1 = np.hstack((array, np.zeros(grp_delay)))
        filt_acq = signal.lfilter(lo, 1.0, acq1, axis=axis)
        filt_array = filt_acq[grp_delay:]
    return filt_array


def remez_1(
    array: Union[np.ndarray, list],
    order: int,
    sample_rate: int,
    highpass: Union[int, None] = None,
    high_width: Union[int, None] = None,
    lowpass: Union[int, None] = None,
    low_width: Union[int, None] = None,
):
    if len(array.shape) > 1:
        axis = 1
    else:
        axis = -1
    if highpass is not None and lowpass is not None:
        filt = signal.remez(
            order,
            [
                0,
                highpass - high_width,
                highpass,
                lowpass,
                lowpass + low_width,
                sample_rate / 2,
            ],
            [0, 1, 0],
            fs=sample_rate,
        )
        filt_acq = signal.filtfilt(filt, 1.0, array, axis=axis)
    elif highpass is not None and lowpass is None:
        hi = signal.remez(
            order,
            [0, highpass - high_width, highpass, sample_rate / 2],
            [0, 1],
            fs=sample_rate,
        )
        filt_acq = signal.filtfilt(hi, 1.0, array, axis=axis)
    elif highpass is None and lowpass is not None:
        lo = signal.remez(
            order,
            [0, lowpass, lowpass + low_width, sample_rate / 2],
            [1, 0],
            fs=sample_rate,
        )
        filt_acq = signal.filtfilt(lo, 1.0, array, axis=axis)
    return filt_acq


def remez_2(
    array: Union[np.ndarray, list],
    order: int,
    sample_rate: int,
    highpass: Union[int, None] = None,
    high_width: Union[int, None] = None,
    lowpass: Union[int, None] = None,
    low_width: Union[int, None] = None,
):
    if len(array.shape) > 1:
        axis = 1
    else:
        axis = -1
    grp_delay = int(0.5 * (order - 1))
    if highpass is not None and lowpass is not None:
        filt = signal.remez(
            numtaps=order,
            bands=[
                0,
                highpass - high_width,
                highpass,
                lowpass,
                lowpass + low_width,
                sample_rate / 2,
            ],
            desired=[0, 1, 0],
            fs=sample_rate,
        )
        acq1 = np.hstack((array, np.zeros(grp_delay)))
        filt_acq = signal.lfilter(filt, 1.0, acq1, axis=axis)
        filt_array = filt_acq[grp_delay:]
    elif highpass is not None and lowpass is None:
        hi = signal.remez(
            order,
            [0, highpass - high_width, highpass, sample_rate / 2],
            [0, 1],
            fs=sample_rate,
        )
        acq1 = np.hstack((array, np.zeros(grp_delay)))
        filt_acq = signal.lfilter(hi, 1.0, acq1, axis=axis)
        filt_array = filt_acq[grp_delay:]
    elif highpass is None and lowpass is not None:
        lo = signal.remez(
            order,
            [0, lowpass, lowpass + low_width, sample_rate / 2],
            [1, 0],
            fs=sample_rate,
        )
        acq1 = np.hstack((array, np.zeros(grp_delay)))
        filt_acq = signal.lfilter(lo, 1.0, acq1, axis=axis)
        filt_array = filt_acq[grp_delay:]
    return filt_array


def filter_array(
    array,
    sample_rate,
    filter_type: Filters = "butterworth_zero",
    order: Union[None, int] = 301,
    highpass: Union[int, float, None] = None,
    high_width: Union[int, float, None] = None,
    lowpass: Union[int, float, None] = None,
    low_width: Union[int, float, None] = None,
    window: Windows = "hann",
    **kwargs,
):
    if filter_type == "bessel":
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
    return filtered_array


def iirnotch_zero(array: np.ndarray, freq: float, q: float, fs: float):
    b, a = signal.iirnotch(freq, q, fs)
    filtered_array = signal.filtfilt(b, a, array)
    return filtered_array


def iirnotch(array: np.ndarray, freq: float, q: float, fs: float):
    a, b = signal.iirnotch(freq, q, fs)
    filtered_array = signal.lfilter(b, a, array)
    return filtered_array


def dwavelet_filter(data, order, filter_type, wavelet="db4"):
    DWTcoeffs = pywt.wavedec(data, wavelet)
    if filter_type == "highpass":
        r = range(0, order)
    else:
        r = range(-1, -order + 1)
    for i in r:
        DWTcoeffs[i] = np.zeros_like(DWTcoeffs[i])

    filtered_data_dwt = pywt.waverec(DWTcoeffs, "db4")
    return filtered_data_dwt
