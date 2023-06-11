from typing import Literal, Union

import numpy as np
from scipy import signal

from . import lfp
from .filtering_functions import Filters, Windows


class LFPManager:
    def cwt_settings(self, start_freq=1, stop_freq=110, steps=200, scaling="log"):
        self.pxx_settings(
            "cwt",
            start_freq=start_freq,
            stop_freq=stop_freq,
            steps=steps,
            scaling=scaling,
        )

    def welch_settings(
        self, nperseg, noverlap, nfft, window=("tukey", 0.25), scaling="density"
    ):
        self.pxx_settings(
            "windowed",
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            window=window,
            scaling=scaling,
        )

    def periodgram_settings(
        self,
        nperseg=2048,
        noverlap=0,
        nfft=2048,
        window=("tukey", 0.25),
        scaling="density",
    ):
        self.pxx_settings(
            "windowed",
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            window=window,
            scaling=scaling,
        )

    def pxx_settings(
        self,
        pxx_type: Literal["cwt", "windowed", "multitaper", "welch", "spectrogram"],
        **kwargs,
    ):
        if not self.file_open:
            self.load_hdf5_acq()
        if self.file.get(pxx_type):
            grp = self.file[pxx_type]
        else:
            grp = self.file.create_group(pxx_type)
        for key, value in kwargs.items():
            self.set_grp_attr(grp, key, value)

    def corr_freqs(self, freq_band_1, freq_band_2, window):
        band_1 = self.file[freq_band_1][()]
        band_2 = self.file[freq_band_2][()]
        corr = lfp.corr_freqs(band_1, band_2, window)
        return corr

    def pxx(self, pxx_type, nthreads=-1, array=None):
        if not self.file_open:
            self.load_hdf5_acq()
        if array is None:
            array = self.acq("lfp")
        grp = self.file["lfp"]
        if pxx_type == "cwt":
            pxx_grp = self.file.get("cwt")
            freqs, pxx = lfp.create_cwt(
                array, grp.attrs["sample_rate"], **pxx_grp.attrs, nthreads=nthreads
            )
        elif pxx_type == "multitaper":
            freqs, pxx = lfp.create_multitaper_ps(array, **pxx_grp.attrs)
        elif pxx_type == "windowed":
            freqs, pxx = lfp.create_window_ps(array, **pxx_grp.attrs)
        elif pxx_type == "welch":
            freqs, pxx = lfp.create_window_ps(array, **pxx_grp.attrs)
        else:
            AttributeError("Pxx_type must be cwt, multitaper or windowed")
            return None
        return freqs, pxx

    def create_all_freq_windows(self, freq_dict, pxx_type, nthreads):
        if not self.file_open:
            self.load_hdf5_acq()
        if not self.file.get(pxx_type):
            AttributeError("Pxx settings do not exist.")
            return None
        grp = self.file["lfp"]
        array = self.acq("lfp")
        if pxx_type == "cwt":
            pxx_grp = self.file.get("cwt")
            freqs, pxx = lfp.create_cwt(
                array, grp.attrs["sample_rate"], **pxx_grp.attrs, nthreads=nthreads
            )
        elif pxx_type == "multitaper":
            freqs, pxx = lfp.create_multitaper_ps(array, **pxx_grp.attrs)
        elif pxx_type == "windowed":
            freqs, pxx = lfp.create_window_ps(array, **pxx_grp.attrs)
        else:
            AttributeError("Pxx_type must be cwt, multitaper or windowed")
            return None
        bands = lfp.create_all_freq_windows(freq_dict, freqs, pxx)
        for key, value in bands.items():
            if self.file.get(key):
                del self.file[key]
                self.file.create_dataset(key, data=value)
            else:
                self.file.create_dataset(key, shape=value.shape, maxshape=array.shape)
                self.file[key][...] = value

    def get_hilbert(
        self,
        filter_type: Filters = "butterworth_zero",
        order: Union[None, int] = 4,
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
        array = self.file["array"][()]
        acq = self.filter_array(
            array,
            filter_type=filter_type,
            order=order,
            highpass=highpass,
            high_width=high_width,
            lowpass=lowpass,
            low_width=low_width,
            window=window,
            polyorder=polyorder,
            sample_rate=self.sample_rate,
        )
        if resample_freq is not None:
            acq = self.downsample(acq, resample_freq, up_sample)

        hil_acq = signal.hilbert(acq)
        return hil_acq

    def get_freq_band(self, band):
        if not self.file_open:
            self.load_hdf5_acq()
        if self.file.get(band):
            return self.file[band][()]
        else:
            AttributeError(f"The {band} frequency band does not exist.")
            return None

    def psd(self, acq_type, nperseg, noverlap):
        array = self.acq(acq_type)
        freq, pxx = signal.welch(
            array,
            fs=self.file[acq_type].attrs["sample_rate"],
            nperseg=nperseg,
            noverlap=noverlap,
        )
        return freq, pxx

    def calc_all_pdi(self, freq_dict, nthreads=4, size=5000):
        if not self.file_open:
            self.load_hdf5_acq()
        lfp_grp = self.file["lfp"]
        array = self.acq("lfp")
        pxx_grp = self.file.get("cwt")
        if not self.file.get("pdi"):
            pdi_grp = self.file.create_group("pdi")
        else:
            pdi_grp = self.file["pdi"]
        freqs, cwt = lfp.create_cwt(
            array, lfp_grp.attrs["sample_rate"], **pxx_grp.attrs, nthreads=nthreads
        )
        cohy = lfp.binned_cohy(cwt, size)
        for key, value in freq_dict.items():
            pdi = lfp.phase_discontinuity_index(
                cohy,
                freqs,
                value[0],
                value[1],
            )
            self.set_grp_attr(pdi_grp, key, pdi)

    def get_short_time_energy(self, window="hamming", wlen=501):
        acq = self.acq("lfp")
        se_array = lfp.short_time_energy(acq, window=window, wlen=wlen)
        return se_array

    def find_lfp_bursts(
        self,
        window="hamming",
        min_len=0.2,
        max_len=20,
        min_burst_int=0.2,
        wlen=200,
        threshold=10,
        pre=3,
        post=3,
        order=100,
        method="spline",
        tol=0.001,
        deg=90,
    ):
        if not self.file_open:
            self.load_hdf5_acq()
        acq = self.acq("lfp")
        bursts = lfp.find_bursts(
            acq,
            window=window,
            min_len=min_len,
            max_len=max_len,
            min_burst_int=min_burst_int,
            wlen=wlen,
            threshold=threshold,
            fs=self.file["lfp"].attrs["sample_rate"],
            pre=pre,
            post=post,
            order=order,
            method=method,
            tol=tol,
            deg=deg,
        )
        input_dict = {
            "window": window,
            "min_len": min_len,
            "max_len": max_len,
            "min_burst_int": min_burst_int,
            "wlen": wlen,
            "threshold": threshold,
            "pre": pre,
            "post": post,
            "order": order,
            "method": method,
            "tol": tol,
            "deg": deg,
        }
        if self.file.get("burst_settings"):
            grp = self.file["burst_settings"]
        else:
            grp = self.file.create_group("burst_settings")
        for key, value in input_dict.items():
            if value is None:
                value = "None"
            self.set_grp_attr(grp, key, value)
        if self.file.get("lfp_bursts"):
            del self.file["lfp_bursts"]
            self.file.create_dataset("lfp_bursts", data=bursts)
        else:
            self.file.create_dataset(
                "lfp_bursts",
                dtype=bursts.dtype,
                shape=bursts.shape,
                maxshape=(acq.size, 2),
            )
            self.file["lfp_bursts"].resize(bursts.shape)
            self.file["lfp_bursts"][...] = bursts

    def get_lfp_burst_indexes(self):
        if not self.file_open:
            self.load_hdf5_acq()
        return np.asarray(self.file["lfp_bursts"][()], dtype=np.int64)

    def get_burst_baseline(self):
        if not self.file_open:
            self.load_hdf5_acq()
        bursts = self.file["lfp_bursts"][()]
        size = int(self.file.attrs["rec_len"] * self.file["lfp"].attrs["sample_rate"])
        bursts_baseline = lfp.burst_baseline_periods(bursts, size)
        return bursts_baseline

    def lfp_burst_stats(self):
        if not self.file_open:
            self.load_hdf5_acq()
        acq = self.acq("lfp")
        bursts = self.file["lfp_bursts"][()]
        fs = self.file["lfp"].attrs["sample_rate"]
        baseline = self.get_burst_baseline()
        ave_len, iei, rms = lfp.burst_stats(acq, bursts, baseline, fs)
        return ave_len, iei, rms
