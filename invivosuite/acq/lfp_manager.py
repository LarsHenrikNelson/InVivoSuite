import os
from typing import Literal, Union

import fcwt
import numpy as np
from scipy import signal

from . import lfp
from .filtering_functions import Filters, Windows, filter_array
from .tapered_spectra import multitaper


class LFPManager:
    def set_cwt(
        self,
        f0: int = 1,
        f1: int = 110,
        fn: int = 400,
        scaling: Literal["log", "lin"] = "log",
        norm: bool = True,
        nthreads: int = -1,
    ):
        self.set_spectral_settings(
            "cwt",
            f0=f0,
            f1=f1,
            fn=fn,
            scaling=scaling,
            nthreads=nthreads,
            norm=norm,
        )

    def set_spectrogram(
        self,
        nperseg: int = 2048,
        noverlap: int = 0,
        nfft: int = 2048,
        window=("tukey", 0.25),
        scaling: str = "density",
    ):
        self.set_spectral_settings(
            "spectrogram",
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            window=window,
            scaling=scaling,
        )

    def set_periodgram(
        self,
        nfft: int = 2048,
        window=("tukey", 0.25),
        scaling: Literal["density", "spectrum"] = "density",
    ):
        self.set_spectral_settings(
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
        self.set_spectral_settings(
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
        window: Union[str, tuple[str, float]] = ("tukey", 0.25),
        scaling: str = "density",
    ):
        self.set_spectral_settings(
            "welch",
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            window=window,
            scaling=scaling,
        )

    def set_spectral_settings(
        self,
        pxx_type: Literal["cwt", "periodogram", "multitaper", "welch", "spectrogram"],
        **kwargs,
    ):
        for key, value in kwargs.items():
            if type(value) is tuple:
                for index, i in enumerate(value):
                    self.set_grp_attr(pxx_type, f"{key}_{index}", i)
            else:
                self.set_grp_attr(pxx_type, key, value)

    def get_spectral_settings(self, pxx_type: str):
        pxx_dict = self.get_grp_attrs(pxx_type)
        window_keys = [
            i.split("_")[1] for i in pxx_dict.keys() if i.split("_")[0] == "window"
        ]
        if len(window_keys) > 0:
            pxx_dict["window"] = tuple(
                pxx_dict[f"window_{i}"] for i in range(len(window_keys))
            )
            for i in window_keys:
                del pxx_dict["window_" + i]
        return pxx_dict

    def pxx(
        self,
        acq_num: int,
        pxx_type: Literal["cwt", "periodogram", "multitaper", "welch"],
        cmr_probe: str,
        cmr: bool = True,
        map_channel: bool = False,
        probe: str = "none",
    ):
        pxx_attrs = self.get_spectral_settings(pxx_type)
        if acq_num > self.n_chans:
            raise ValueError(f"{acq_num} does not exist.")
        fs = self.get_grp_attr("lfp", "sample_rate")
        array = self.acq(
            acq_num,
            "lfp",
            cmr=cmr,
            cmr_probe=cmr_probe,
            probe=probe,
            map_channel=map_channel,
        )
        if pxx_type == "multitaper":
            freqs, pxx = multitaper(array, fs=fs, **pxx_attrs)
        elif pxx_type == "periodogram":
            freqs, pxx = signal.periodogram(array, fs=fs, **pxx_attrs)
        elif pxx_type == "welch":
            freqs, pxx = signal.welch(array, fs=fs, **pxx_attrs)
        elif pxx_type == "cwt":
            cpuc = os.cpu_count()
            if pxx_attrs["nthreads"] == -1 or pxx_attrs["nthreads"] > cpuc:
                pxx_attrs["nthreads"] = cpuc
            freqs, sxx = fcwt.cwt(
                array,
                int(fs),
                int(pxx_attrs["f0"]),
                int(pxx_attrs["f1"]),
                int(pxx_attrs["fn"]),
                int(pxx_attrs["nthreads"]),
                pxx_attrs["scaling"],
                False,
                bool(pxx_attrs["norm"]),
            )
            pxx = np.abs(sxx).mean(axis=1)
        else:
            AttributeError("pxx_type must be cwt, multitaper, periodogram, or welch")
            return None
        return freqs, pxx

    def sxx(
        self,
        acq_num: int,
        sxx_type: Literal["cwt", "spectrogram"],
        cmr_probe: str,
        cmr: bool = True,
        map_channel: bool = False,
        probe: str = "none",
    ):
        sxx_attrs = self.get_grp_attrs(sxx_type)
        if acq_num > self.n_chans:
            raise ValueError(f"{acq_num} does not exist.")
        fs = self.get_grp_attr("lfp", "sample_rate")
        array = self.acq(
            acq_num,
            "lfp",
            probe=probe,
            cmr_probe=cmr_probe,
            cmr=cmr,
            map_channel=map_channel,
        )
        if sxx_type == "cwt":
            cpuc = os.cpu_count()
            if sxx_attrs["nthreads"] == -1 or sxx_attrs["nthreads"] > cpuc:
                sxx_attrs["nthreads"] = cpuc
            freqs, sxx = fcwt.cwt(
                array,
                int(fs),
                int(sxx_attrs["f0"]),
                int(sxx_attrs["f1"]),
                int(sxx_attrs["fn"]),
                sxx_attrs["nthreads"],
                str(sxx_attrs["scaling"]),
                False,
                bool(sxx_attrs["norm"]),
            )
        return freqs, sxx

    def hilbert(
        self,
        acq_num: int,
        map_channel: bool = False,
        probe: str = "none",
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
        start = self.get_file_attr("start")
        end = self.get_file_attr("end")
        sample_rate = self.get_file_dataset("sample_rate", rows=acq_num)
        if map_channel:
            acq_num = self.get_mapped_channel(probe, acq_num)
        if probe != "none":
            data = self.get_grp_dataset("probes", probe)
            acq_num += data[0]
        array = self.get_file_dataset(
            "acqs", rows=acq_num, columns=(start, end)
        ) * self.get_file_dataset("coeffs", rows=acq_num)
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
            sample_rate=sample_rate,
        )
        if resample_freq is not None:
            acq = self.downsample(acq, sample_rate, resample_freq, up_sample)

        hil_acq = signal.hilbert(acq)
        return hil_acq

    def calc_all_pdi(
        self,
        freq_dict: dict[str, Union[tuple[int, int], tuple[float, float]]],
        cmr: bool = False,
        map_channel: bool = False,
        size: int = 2000,
    ):
        pdi_dict = {key: np.zeros(self.n_chans) for key in freq_dict.keys()}
        probes = self.probes
        for region in probes:
            start = self.get_grp_dataset("probes", region)[0]
            for i in range(0, 64):
                freqs, cwt = self.sxx(
                    i,
                    "cwt",
                    cmr=cmr,
                    map_channel=map_channel,
                    probe=region,
                    cmr_probe=region,
                )
                pdi_temp = lfp.phase_discontinuity_index(
                    cwt,
                    freqs,
                    freq_dict,
                    size,
                )
                for key, value in pdi_temp.items():
                    pdi_dict[key][start + i] = value
        for key, value in pdi_dict.items():
            self.set_grp_dataset("pdi", key, value)

    def get_short_time_energy(
        self,
        acq: Union[None, np.ndarray] = None,
        acq_num: Union[None, int] = None,
        window: str = "hamming",
        wlen: float = 0.2,
        fs: Union[float, int, None] = None,
        map_channel: bool = False,
    ):
        """This is a convience function to test out different short time energy settings
        or get the short time energy of an specific acquisition.

        Args:
            acq (Union[None, np.ndarray], optional): Numpy array containing the acq.
            Defaults to None.
            acq_num (Union[None, int], optional): Acquistion number must be supplied as
            a zero-indexed (e.g. 1 is 0, 2 is 1, etc). Defaults to None.
            window (str, optional): _description_. Defaults to "hamming".
            wlen (float, optional): _description_. Defaults to 0.2.
            fs (Union[float, int, None], optional): _description_. Defaults to None.
            map_channel (bool, optional): _description_. Defaults to None.

        Raises:
            AttributeError: Raises error is acq or acq_num is not supplied.
            AttributeError: Raises error if fs is not supplied if an acq is supplied.

        Returns:
            np.ndarray: The short time energy
        """
        if acq is None and acq_num is None:
            raise AttributeError(
                "An acquisition or acquisition number must be supplied"
            )
        if acq is not None:
            if fs is None:
                raise AttributeError("fs must be supplied if using acq argument.")
        else:
            self.open()
            if map_channel and self.file.get("channel_map"):
                acq_num = self.file["channel_map"][acq_num]
            acq = self.acq(acq_num, "lfp")
            fs = self.get_grp_attr("lfp", "sample_rate")
        se_array = lfp.short_time_energy(acq, window=window, wlen=wlen, fs=fs)
        return se_array

    def get_ste_baseline(
        self,
        ste: np.ndarray,
        tol: Union[float, None] = None,
        method: Union[Literal["spline", "fixed", "polynomial"], None] = None,
        deg: Union[int, None] = None,
        threshold: Union[float, None] = None,
    ):
        if tol is None:
            tol = self.get_grp_attr("lfp_bursts", "tol")
        if method is None:
            method = self.get_grp_attr("lfp_bursts", "method")
        if deg is None:
            deg = self.get_grp_attr("lfp_bursts", "deg")
        if threshold is None:
            threshold = self.get_grp_attr("lfp_bursts", "threshold")
        baseline = lfp.kde_baseline(
            ste, method=method, tol=tol, deg=deg, threshold=threshold
        )
        return baseline

    def find_lfp_bursts(
        self,
        window: lfp.Windows = "hamming",
        min_len: float = 0.2,
        max_len: float = 20.0,
        min_burst_int: float = 0.2,
        minimum_peaks: int = 5,
        wlen: float = 0.2,
        threshold: Union[float, int] = 10,
        pre: Union[float, int] = 3.0,
        post: Union[float, int] = 3.0,
        order: Union[float, int] = 0.1,
        method: Literal["spline", "fixed", "polynomial"] = "spline",
        tol: float = 0.001,
        deg: int = 90,
        cmr: bool = False,
    ):
        input_dict = {
            "window": window,
            "min_len": min_len,
            "max_len": max_len,
            "min_burst_int": min_burst_int,
            "mininum_peaks": minimum_peaks,
            "wlen": wlen,
            "threshold": threshold,
            "pre": pre,
            "post": post,
            "order": order,
            "method": method,
            "tol": tol,
            "deg": deg,
            "cmr": cmr,
        }
        for key, value in input_dict.items():
            if value is None:
                value = "none"
            self.set_grp_attr("lfp_bursts", key, value)

        fs = self.get_grp_attr("lfp", "sample_rate")
        probes = self.probes
        for region in probes:
            start = self.get_grp_dataset("probes", region)[0]
            for i in range(0, 64):
                acq_i = self.acq(
                    i, "lfp", cmr=cmr, cmr_probe=region, map_channel=False, probe=region
                )
                bursts = lfp.find_bursts(
                    acq_i,
                    window=window,
                    min_len=min_len,
                    max_len=max_len,
                    min_burst_int=min_burst_int,
                    minimum_peaks=minimum_peaks,
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
                self.set_grp_dataset("lfp_bursts", str(int(i + start)), bursts)

    def get_lfp_burst_indexes(
        self, acq_num: int, map_channel: bool = False, probe: str = "none"
    ):
        if map_channel:
            acq_num = self.get_mapped_channel(acq_num, probe)
        indexes = self.get_grp_dataset("lfp_bursts", str(acq_num))
        return indexes

    def get_burst_baseline(
        self, acq_num: int, map_channel: bool = False, probe: str = "none"
    ):
        if map_channel:
            acq_num = self.get_mapped_channel(acq_num, probe)
        bursts = self.get_grp_dataset("lfp_bursts", str(acq_num))
        start = self.get_file_attr("start")
        end = self.get_file_attr("end")
        size = int(end - start)
        fs_lfp = self.get_grp_attr("lfp", "sample_rate")
        fs_raw = self.get_file_dataset("sample_rate", rows=acq_num)
        size = int(size / (fs_raw / fs_lfp))
        burst_baseline = lfp.burst_baseline_periods(bursts, size)
        self.close()
        return burst_baseline

    def lfp_burst_stats_channel(
        self,
        acq_num: int,
        bands: dict[str, Union[tuple[int, int], tuple[float, float]]],
        calc_average: bool = True,
        map_channel: bool = False,
        probe: str = "none",
    ):
        b_stats = {"channel": acq_num}
        if map_channel:
            acq_num = self.get_mapped_channel(probe, acq_num)
        b_stats["mapped_channel"] = acq_num
        acq = self.acq(acq_num, "lfp")
        bursts = self.get_grp_dataset("lfp_bursts", str(acq_num))
        fs = self.get_grp_attr("lfp", "sample_rate")
        baseline = self.get_burst_baseline(
            acq_num, map_channel=map_channel, probe=probe
        )
        temp = lfp.burst_stats(acq, bursts, baseline, bands, fs)
        if calc_average:
            mean_data = {
                f"{key}_mean": value.mean()
                for key, value in temp.items()
                if not isinstance(value, int)
            }
            std_data = {
                f"{key}_std": value.std()
                for key, value in temp.items()
                if not isinstance(value, int)
            }
            b_stats.update(mean_data)
            b_stats.update(std_data)
        else:
            b_stats.update(temp)
        return b_stats

    def lfp_burst_stats(
        self,
        bands: dict[str, Union[tuple[int, int], tuple[float, float]]],
        calc_average: bool = True,
        map_channel: bool = False,
        probe: str = "none",
    ):
        total_data = []
        for i in range(self.n_chans):
            data = self.lfp_burst_stats_channel(
                i,
                bands=bands,
                calc_average=calc_average,
                map_channel=map_channel,
                probe=probe,
            )
            total_data.append(data)
        return total_data

    def set_freq_bands(self, freq_dict):
        for key, value in freq_dict.items():
            self.set_grp_dataset("freq_bands", key, np.array(value))
