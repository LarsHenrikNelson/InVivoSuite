import os
from typing import Literal, Union, Iterable, Optional

import fcwt
import numpy as np
from scipy import signal

from ..functions.filter_functions import Filters, Windows, filter_array, downsample
from ..spectral import multitaper, get_freq_window
from ..functions import lfp_functions, signal_functions


class LFPManager:
    def set_cwt(
        self,
        f0: int = 1,
        f1: int = 110,
        fn: int = 200,
        scaling: Literal["log", "lin"] = "log",
        sigma: float = 2.0,
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
            sigma=sigma,
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
            if isinstance(value, tuple):
                for index, i in enumerate(value):
                    self.set_grp_attr(pxx_type, f"{key}_{index}", i)
            elif value is None:
                self.set_grp_attr(pxx_type, key, "none")
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
        for key, value in pxx_dict.items():
            if value == "none":
                pxx_dict[key] = None
        return pxx_dict

    def pxx(
        self,
        channel: int,
        pxx_type: Literal["cwt", "periodogram", "multitaper", "welch"],
        ref_type: Literal["none", "cmr", "car"] = "cmr",
        ref_probe: str = "all",
        map_channel: bool = False,
        probe: str = "all",
        start: int = 0,
        end: int = 0,
    ):
        pxx_attrs = self.get_spectral_settings(pxx_type)
        if channel > self.n_chans:
            raise ValueError(f"{channel} does not exist.")
        fs = self.get_grp_attr("lfp", "sample_rate")
        array = self.acq(
            channel,
            "lfp",
            ref_type=ref_type,
            ref_probe=ref_probe,
            probe=probe,
            map_channel=map_channel,
            start=start,
            end=end,
        )
        if pxx_type == "multitaper":
            print(pxx_attrs)
            freqs, pxx, _ = multitaper(array, fs=fs, **pxx_attrs)
        elif pxx_type == "periodogram":
            freqs, pxx = signal.periodogram(array, fs=fs, **pxx_attrs)
        elif pxx_type == "welch":
            freqs, pxx = signal.welch(array, fs=fs, **pxx_attrs)
        elif pxx_type == "cwt":
            cpuc = os.cpu_count()
            if pxx_attrs["nthreads"] == -1 or pxx_attrs["nthreads"] > cpuc:
                pxx_attrs["nthreads"] = (cpuc // 2) - 1
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

    def _create_cwt_object(self, attrs, size):
        self.cwt_morl = fcwt.Morlet(attrs["sigma"])

        _scales = (
            fcwt.FCWT_LINFREQS if attrs["scaling"] == "lin" else fcwt.FCWT_LOGSCALES
        )

        self.cwt_scales = fcwt.Scales(
            self.cwt_morl,
            _scales,
            int(attrs["fs"]),
            float(attrs["f0"]),
            float(attrs["f1"]),
            int(attrs["fn"]),
        )
        self.cwt_obj = fcwt.FCWT(
            self.cwt_morl, int(attrs["nthreads"]), False, bool(attrs["norm"])
        )
        self.cwt_output = np.zeros((attrs["fn"], size), dtype=np.complex64)
        self.cwt_frequencies = np.zeros((attrs["fn"]), dtype=np.float32)
        self.cwt_scales.getFrequencies(self.cwt_frequencies)

    def sxx(
        self,
        channel: int,
        sxx_type: Literal["cwt", "spectrogram"],
        ref_type: Literal["none", "cmr", "car"] = "cmr",
        ref_probe: str = "all",
        map_channel: bool = False,
        probe: str = "all",
        start: int = 0,
        end: int = 0,
    ):
        sxx_attrs = self.get_grp_attrs(sxx_type)
        if channel > self.n_chans:
            raise ValueError(f"{channel} does not exist.")
        sxx_attrs["fs"] = self.get_grp_attr("lfp", "sample_rate")
        array = self.acq(
            channel,
            "lfp",
            ref_type=ref_type,
            ref_probe=ref_probe,
            probe=probe,
            map_channel=map_channel,
            start=start,
            end=end,
        )
        array = array.astype(np.float32)
        if sxx_type == "cwt":
            cpuc = os.cpu_count()
            if sxx_attrs["nthreads"] == -1 or sxx_attrs["nthreads"] > (cpuc // 2):
                sxx_attrs["nthreads"] = (cpuc // 2) - 1
            if not self._set_cwt:
                self._create_cwt_object(sxx_attrs, array.size)
            self.cwt_obj.cwt(
                array,
                self.cwt_scales,
                self.cwt_output,
            )
            freqs = self.cwt_frequencies
            sxx = self.cwt_output
        return freqs, sxx

    def hilbert(
        self,
        channel: int,
        ref_type: Literal["none", "cmr", "car"] = "cmr",
        ref_probe: str = "all",
        map_channel: bool = False,
        probe: str = "all",
        filter_type: Filters = "butterworth_zero",
        order: Union[None, int] = 4,
        highpass: Union[int, float, None] = None,
        high_width: Union[int, float, None] = None,
        lowpass: Union[int, float, None] = None,
        low_width: Union[int, float, None] = None,
        window: Windows = "hann",
        polyorder: Union[int, None] = 0,
        resample_freq: float = 1000.0,
        up_sample=3,
        start: int = 0,
        end: int = 0,
    ):
        start = self.start + start
        if end > 0:
            end = self.start + end
        else:
            end = self.end
        channel = self.get_mapped_channel(channel, probe=probe, map_channel=map_channel)
        sample_rate = self.get_file_dataset("sample_rate", rows=channel)
        array = self.get_file_dataset(
            "acqs", rows=channel, columns=(start, end)
        ) * self.get_file_dataset("coeffs", rows=channel)
        array -= array.mean()
        if ref_type != "none":
            ref_data = self.get_grp_dataset(ref_type, ref_probe)
            array -= ref_data
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
            acq = downsample(acq, sample_rate, resample_freq, up_sample)

        hil_acq = signal.hilbert(acq)
        return hil_acq

    def all_pxx(
        self,
        freq_dict: dict[str, Union[tuple[int, int], tuple[float, float]]],
        pxx_type: Literal["cwt", "periodogram", "multitaper", "welch"],
        ref_type: Literal["none", "cmr", "car"] = "cmr",
        ref_probe: str = "all",
        map_channel: bool = False,
        probe: str = "all",
        start: int = 0,
        end: int = 0,
        log_transform=True,
        window_type: Literal["sum", "mean"] = "sum",
    ):
        output = {}
        chans = self.get_grp_dataset("probes", probe)
        start_chan = chans[0] - chans[0]
        end_chan = chans[1] - chans[0]
        output = {key: np.zeros(end_chan) for key in freq_dict.keys()}
        output["channels"] = np.arange(start_chan, end_chan)
        for index, channel in enumerate(output["channels"]):
            self.callback(
                f"Extracting frequency data for channel {channel} on probe {probe}."
            )
            f, p = self.pxx(
                channel=channel,
                pxx_type=pxx_type,

                ref_type=ref_type,
                ref_probe=ref_probe,
                map_channel=map_channel,
                probe=probe,
                start=start,
                end=end,
            )
            for fr, val in freq_dict.items():
                output[fr][index] = get_freq_window(
                    pxx=p,
                    freqs=f,
                    lower_limit=val[0],
                    upper_limit=val[1],
                    log_transform=log_transform,
                    window_type=window_type,
                )
        output["channels"] = np.arange(start_chan, end_chan)
        return output

    def calc_all_pdi(
        self,
        freq_dict: dict[str, Union[tuple[int, int], tuple[float, float]]],
        ref_type: Literal["none", "cmr", "car"] = "cmr",
        ref_probe: str = "all",
        map_channel: bool = False,
        size: int = 2000,
        probe: str = "all",
        start: int = 0,
        end: int = 0,
    ):
        output = {}
        chans = self.get_grp_dataset("probes", probe)
        start_chan = chans[0] - chans[0]
        end_chan = chans[1] - chans[0]
        output = {key: np.zeros(end_chan) for key in freq_dict.keys()}
        output["channels"] = np.arange(start_chan, end_chan)
        for index, channel in enumerate(output["channels"]):
            self.callback(
                f"Extracting phase data for channel {channel} on probe {probe}."
            )
            freqs, cwt = self.sxx(
                channel,
                "cwt",

                ref_type=ref_type,
                ref_probe=ref_probe,
                map_channel=map_channel,
                probe=probe,
                start=start,
                end=end
            )
            pdi_temp = lfp_functions.phase_discontinuity_index(
                cwt,
                freqs,
                freq_dict,
                size,
            )
            for key, value in pdi_temp.items():
                output[key][index] = value
        return output

    def get_short_time_energy(
        self,
        channel: Union[None, int] = None,
        ref_type: Literal["none", "cmr", "car"] = "cmr",
        ref_probe: str = "all",
        map_channel: bool = False,
        probe: str = "all",
        window: str = "hamming",
        wlen: float = 0.2,
        start: int = 0,
        end: int = 0,
    ):
        """This is a convience function to test out different short time energy settings
        or get the short time energy of an specific acquisition.

        Args:
            acq (Union[None, np.ndarray], optional): Numpy array containing the acq.
            Defaults to None.
            channel (Union[None, int], optional): Acquistion number must be supplied as
            a zero-indexed (e.g. 1 is 0, 2 is 1, etc). Defaults to None.
            window (str, optional): _description_. Defaults to "hamming".
            wlen (float, optional): _description_. Defaults to 0.2.
            fs (Union[float, int, None], optional): _description_. Defaults to None.
            map_channel (bool, optional): _description_. Defaults to None.

        Raises:
            AttributeError: Raises error is acq or channel is not supplied.
            AttributeError: Raises error if fs is not supplied if an acq is supplied.

        Returns:
            np.ndarray: The short time energy
        """
        acq = self.acq(
            channel,
            "lfp",
            ref_type=ref_type,
            ref_probe=ref_probe,
            probe=probe,
            map_channel=map_channel,
            start=start,
            end=end,
        )
        fs = self.get_grp_attr("lfp", "sample_rate")
        se_array = signal_functions.short_time_energy(
            acq, window=window, wlen=wlen, fs=fs
        )
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
        baseline = signal_functions.kde_baseline(
            ste, method=method, tol=tol, deg=deg, threshold=threshold
        )
        return baseline

    def find_lfp_bursts(
        self,
        window: signal_functions.Windows = "hamming",
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
        start: int = 0,
        end: int = 0,
    ):
        start = self.start + start
        if end > 0:
            end = self.start + end
        else:
            end = self.end
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
            for i in range(0, 64):
                acq_i = self.acq(
                    i,
                    "lfp",
                    cmr=cmr,
                    cmr_probe=region,
                    map_channel=False,
                    probe=region,
                    start=start,
                    end=end,
                )
                bursts = lfp_functions.find_bursts(
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
        self, channel: int, map_channel: bool = False, probe: str = "none"
    ):
        if map_channel:
            channel = self.get_mapped_channel(channel, probe)
        indexes = self.get_grp_dataset("lfp_bursts", str(channel))
        return indexes

    def get_burst_baseline(
        self,
        channel: int,
        map_channel: bool = False,
        probe: str = "none",
        start: int = 0,
        end: int = 0,
    ):
        if map_channel:
            channel = self.get_mapped_channel(channel, probe)
        bursts = self.get_grp_dataset("lfp_bursts", str(channel))
        start = self.start + start
        if end > 0:
            end = self.start + end
        else:
            end = self.end
        size = int(end - start)
        fs_lfp = self.get_grp_attr("lfp", "sample_rate")
        fs_raw = self.get_file_dataset("sample_rate", rows=channel)
        size = int(size / (fs_raw / fs_lfp))
        burst_baseline = lfp_functions.burst_baseline_periods(bursts, size)
        self.close()
        return burst_baseline

    def lfp_burst_stats_channel(
        self,
        channel: int,
        bands: dict[str, Union[tuple[int, int], tuple[float, float]]],
        calc_average: bool = True,
        ref_type: Literal["none", "cmr", "car"] = "cmr",
        ref_probe: str = "all",
        map_channel=True,
        probe: str = "all",
        start: int = 0,
        end: int = 0,
    ):
        b_stats = {"channel": channel}
        if map_channel:
            channel = self.get_mapped_channel(probe, channel)
        b_stats["mapped_channel"] = channel
        acq = self.acq(
            channel,
            "lfp",
            ref_type=ref_type,
            ref_probe=ref_probe,
            probe=probe,
            map_channel=map_channel,
            start=start,
            end=end,
        )
        bursts = self.get_grp_dataset("lfp_bursts", str(channel))
        fs = self.get_grp_attr("lfp", "sample_rate")
        baseline = self.get_burst_baseline(
            channel, map_channel=map_channel, probe=probe
        )
        temp = lfp_functions.burst_stats(acq, bursts, baseline, bands, fs)
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

    def get_cwt_freq_bands(
        self,
        freq_bands: dict[str, Iterable],
        channel: int,
        ref_type: Literal["none", "cmr", "car"] = "cmr",
        ref_probe: str = "all",
        map_channel=True,
        probe: str = "all",
        start: int = 0,
        end: int = 0,
    ) -> dict[str, np.ndarray]:
        freqs, cwt = self.sxx(
            channel=channel,
            sxx_type="cwt",
            ref_type=ref_type,
            ref_probe=ref_probe,
            map_channel=map_channel,
            probe=probe,
            start=start,
            end=end,
        )

        band_dict = {}
        for b_name, fr in freq_bands.items():
            band_ind = np.where((freqs > fr[0]) & (freqs < fr[1]))[0]
            cwt_band = cwt[band_ind]
            band_dict[b_name] = cwt_band.mean(axis=0)
        return band_dict

    def get_hilbert_freq_bands(
        self,
        freq_bands: dict[str, Iterable],
        channel: int,
        ref_type: Literal["none", "cmr", "car"] = "cmr",
        ref_probe: str = "all",
        map_channel=True,
        probe: str = "all",
        start: int = 0,
        end: int = 0,
    ) -> dict[str, np.ndarray]:
        band_dict = {}
        for b_name, fr in freq_bands.items():
            band_dict[b_name] = self.hilbert(
                channel=channel,

                ref_type=ref_type,
                ref_probe=ref_probe,
                map_channel=map_channel,
                probe=probe,
                highpass=fr[0],
                lowpass=fr[1],
                filter_type="butterworth_zero",
                order=4,
                start=start,
                end=end,
            )
        return band_dict

    def get_sxx_freq_bands(
        self,
        output_type: Literal["phase", "power", "frequency"],
        freq_bands: dict[str, Iterable],
        sxx_type: Literal["cwt", "hilbert"],
        channel: int,
        ref_type: Literal["none", "cmr", "car"] = "cmr",
        ref_probe: str = "all",
        map_channel=True,
        probe: str = "all",
        start: int = 0,
        end: int = 0,
    ) -> dict[str, np.ndarray]:
        if sxx_type == "cwt":
            band_dict = self.get_cwt_freq_bands(
                freq_bands=freq_bands,
                channel=channel,

                ref_type=ref_type,
                ref_probe=ref_probe,
                map_channel=map_channel,
                probe=probe,
                start=start,
                end=end,
            )
        else:
            band_dict = self.get_hilbert_freq_bands(
                freq_bands=freq_bands,
                channel=channel,

                ref_type=ref_type,
                ref_probe=ref_probe,
                map_channel=map_channel,
                probe=probe,
                start=start,
                end=end,
            )
        for key, value in band_dict.items():
            if output_type == "phase":
                band_dict[key] = np.angle(value)
            elif output_type == "power":
                band_dict[key] = np.abs(value)
            elif output_type == "frequency":
                band_dict[key] = value.real
            else:
                raise AttributeError(
                    "output_type not recognized. Must be phase or power."
                )
        return band_dict

    def phase_changes(
        self,
        freq_bands: dict[str, Iterable],
        sxx_type: Literal["cwt", "hilbert"],
        channel: int,
        ref_type: Literal["none", "cmr", "car"] = "cmr",
        ref_probe: str = "all",
        map_channel=True,
        probe: str = "all",
        start: int = 0,
        end: int = 0,
    ):
        output = {}
        start = self.start + start
        if end > 0:
            end = self.start + end
        else:
            end = self.end
        chans = self.get_grp_dataset("probes", probe)
        start_chan = chans[0] - chans[0]
        end_chan = chans[1] - chans[0]
        output = {key: np.zeros(end_chan) for key in freq_bands.keys()}
        output["channels"] = np.arange(start_chan, end_chan)
        for index, channel in enumerate(output["channels"]):
            self.callback(
                f"Extracting phase data for channel {channel} on probe {probe}."
            )

            phase_dict = self.get_sxx_freq_bands(
                output_type="phase",
                freq_bands=freq_bands,
                sxx_type=sxx_type,
                channel=channel,

                ref_type=ref_type,
                ref_probe=ref_probe,
                map_channel=map_channel,
                probe=probe,
                start=start,
                end=end,
            )
            # for b_name, phase in phase_dict.items():
            #     output_dict[b_name] = b_phases
            #     stats = self.analyze_spike_phase(b_phases)
            #     output_stats.update(
            #         {f"{b_name}_{key}": value for key, value in stats.items()}
            #     )

    def cross_freq_coupling(
        self,
        phase: tuple[int | float, int | float],
        power: tuple[int | float, int | float],
        steps: int = 50,
        ref_type: Literal["none", "cmr", "car"] = "cmr",
        ref_probe: str = "all",
        map_channel=True,
        probe: str = "all",
        start: int = 0,
        end: int = 0,
    ):
        chans = self.get_grp_dataset("probes", probe)
        start_chan = chans[0] - chans[0]
        end_chan = chans[1] - chans[0]
        output = {
            "modulation_index": np.zeros(end_chan),
            "channels": np.arange(start_chan, end_chan),
        }
        for index, channel in enumerate(output["channels"][:5]):
            self.callback(
                f"Extracting phase and amplitude data for channel {channel} on probe {probe}."
            )
            phi = self.hilbert(
                channel=channel,

                ref_probe=ref_probe,
                ref_type=ref_type,
                map_channel=map_channel,
                probe=probe,
                highpass=phase[0],
                lowpass=phase[1],
                start=start,
                end=end
            )
            amp = self.hilbert(
                channel=channel,

                ref_probe=ref_probe,
                ref_type=ref_type,
                map_channel=map_channel,
                probe=probe,
                highpass=power[0],
                lowpass=power[1],
                start=start,
                end=end
            )
            phi = np.angle(phi)
            amp = np.abs(amp)
            mi = lfp_functions.modulation_index(
                phi=phi,
                amp=amp,
                steps=steps,
            )
            output["modulation_index"][index] = mi
        return output
