import os
from typing import Literal, Union

import fcwt

# import numpy as np
from scipy import signal

from . import lfp
from .filtering_functions import Filters, Windows, filter_array
from .tapered_spectra import multitaper


class LFPManager:
    def set_cwt(
        self,
        f0: float = 1.0,
        f1: float = 110,
        fn: int = 400,
        scaling: Literal["log", "lin"] = "log",
        norm: bool = True,
        nthreads: int = -1,
    ):
        self.pxx_settings(
            "cwt",
            f0=f0,
            f1=f1,
            fn=fn,
            scaling=scaling,
            nthreads=nthreads,
        )

    def set_periodgram(
        self,
        nfft: int = 2048,
        window=("tukey", 0.25),
        scaling: Literal["density", "spectrum"] = "density",
    ):
        self.pxx_settings(
            "periodogram",
            nfft=nfft,
            window=window,
            scaling=scaling,
        )

    def set_multitaper(
        self,
        NW: float = 2.5,
        BW: Union[float, None] = None,
        adaptive: bool = False,
        jackknife: bool = True,
        low_bias: bool = True,
        sides: str = "default",
        NFFT: Union[int, None] = None,
    ):
        self.pxx_settings(
            "multitaper",
            NW=NW,
            BW=BW,
            adaptive=adaptive,
            jackknife=jackknife,
            low_bias=low_bias,
            sides=sides,
            NFFT=NFFT,
        )

    def set_welch(
        self,
        nperseg: int = 2048,
        noverlap: int = 0,
        nfft: int = 2048,
        window=("tukey", 0.25),
        scaling: str = "density",
    ):
        self.pxx_settings(
            "welch",
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            window=window,
            scaling=scaling,
        )

    def pxx_settings(
        self,
        pxx_type: Literal["cwt", "periodogram", "multitaper", "welch"],
        **kwargs,
    ):
        self.open()
        if self.file.get(pxx_type):
            grp = self.file[pxx_type]
        else:
            grp = self.file.create_group(pxx_type)
        for key, value in kwargs.items():
            self.set_grp_attr(grp, key, value)

    def pxx(
        self,
        acq_num: int,
        pxx_type: Literal["cwt", "periodogram", "multitaper", "welch"],
        nthreads: int = -1,
    ):
        self.open()
        if not self.file.get(pxx_type):
            raise KeyError(f"{pxx_type} settings do not exist in file. Use set_pxx")
        else:
            pxx_grp = self.file[pxx_type]
        if acq_num <= self.file["acqs"].shape[0]:
            if self.file["lfp"].attrs["resample_freq"] != "None":
                fs = self.file["lfp"].attrs["resample_freq"]
            else:
                fs = self.file["sample_rate"][acq_num]
            array = self.acq("lfp", acq_num)
        else:
            raise AttributeError(f"{acq_num} does not exist.")
        if pxx_type == "multitaper":
            freqs, pxx = multitaper(array, fs=fs, **pxx_grp.attrs)
        elif pxx_type == "periodogram":
            freqs, pxx = signal.periodogram(array, fs=fs, **pxx_grp.attrs)
        elif pxx_type == "welch":
            freqs, pxx = signal.welch(array, fs=fs, **pxx_grp.attrs)
        elif pxx_type == "cwt":
            cpuc = os.cpu_count()
            if nthreads == -1 or nthreads > cpuc:
                nthreads = cpuc
            freqs, sxx = fcwt.cwt(signal=array, fs=fs, **pxx_grp.attrs)
            pxx = sxx.mean(axis=1)
        else:
            AttributeError("Pxx_type must be cwt, multitaper, periodogram, or welch")
            return None
        return freqs, pxx

    def hilbert(
        self,
        acq_num: int,
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
        self.open()
        start = self.file.attr["start"]
        end = self.file.attr["end"]
        array = self.file["array"][acq_num, start:end]
        acq = filter_array(
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
        self.close()
        return hil_acq

    def calc_all_pdi(self, freq_dict, nthreads=4, size=5000):
        self.open()
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

    def get_short_time_energy(self, acq_num, window="hamming", wlen=501):
        acq = self.acq("lfp", acq_num)
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
        for key, value in input_dict.items():
            if value is None:
                value = "None"
            self.set_grp_attr("lfp_bursts", key, value)

        self.open()
        shape = self.file["acqs"].shape[0]
        fs = self.file["lfp"].attrs["sample_rate"]
        self.close()
        for i in range(shape):
            acq_i = self.acq(
                "lfp",
                i,
            )
            bursts = lfp.find_bursts(
                acq_i,
                window=window,
                min_len=min_len,
                max_len=max_len,
                min_burst_int=min_burst_int,
                wlen=wlen,
                threshold=threshold,
                fs=fs,
                pre=pre,
                post=post,
                order=order,
                method=method,
                tol=tol,
                deg=deg,
            )
            self.set_grp_dataset("lfp_bursts", str(i), bursts)

    # def get_lfp_burst_indexes(self):
    #     self.open()
    #     return np.asarray(self.file["lfp_bursts"][()], dtype=np.int64)

    def get_burst_baseline(self, acq_num):
        self.open()
        bursts = self.file["lfp_bursts"][acq_num]
        size = int(self.file.attrs["stop"] - self.file.attrs["start"])
        burst_baseline = lfp.burst_baseline_periods(bursts, size)
        self.close()
        return burst_baseline

    def lfp_burst_stats(self):
        self.open()
        acq = self.acq("lfp")
        bursts = self.file["lfp_bursts"][()]
        fs = self.file["lfp"].attrs["sample_rate"]
        baseline = self.get_burst_baseline()
        ave_len, iei, rms = lfp.burst_stats(acq, bursts, baseline, fs)
        return ave_len, iei, rms
