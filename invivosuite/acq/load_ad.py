from collections import namedtuple
from pathlib import Path, PurePath
from typing import Union

import numpy as np

from .acq_manager import AcqManager
from .pypl2 import PL2DigitalChannelInfo  # noqa: F401
from .pypl2 import PL2SpikeChannelInfo  # noqa: F401
from .pypl2 import pl2_comments  # noqa: F401
from .pypl2 import pl2_events  # noqa: F401
from .pypl2 import pl2_info  # noqa: F401
from .pypl2 import pl2_spikes  # noqa: F401
from .pypl2 import PL2AnalogChannelInfo, PL2FileInfo, PyPL2FileReader, pl2_ad

PL2Ad = namedtuple("PL2Ad", "adfrequency n timestamps fragmentcounts ad")

__all__ = [
    "load_acq",
    "load_hdf5_acqs",
    "load_pl2_acqs",
]


def load_pl2_acqs(
    pl2_path: str,
    save_path: str = "",
    start: int = 0,
    end: Union[None, int] = None,
):
    name = PurePath(pl2_path).stem
    if not _path_checker(pl2_path, save_path):
        return None
    reader = PyPL2FileReader()
    handle = reader.pl2_open_file(pl2_path)
    file_info = PL2FileInfo()
    res = reader.pl2_get_file_info(handle, file_info)
    if res == 0:
        return None

    # Use spike channels because this ignores other analog channels
    nchannels = file_info.m_TotalNumberOfAnalogChannels
    wb_channels = []
    ai_channels = []
    plx_channel = []
    for i in range(nchannels):
        ad_info = PL2AnalogChannelInfo()
        ad_res = reader.pl2_get_analog_channel_info(handle, i, ad_info)
        if (
            ad_info.m_ChannelEnabled
            and ad_res != 0
            and ad_info.m_Name.decode("ascii")[:2] == "WB"
        ):
            wb_channels.append(i)
            plx_channel.append(ad_info.m_Channel)
        if (
            ad_info.m_ChannelEnabled
            and ad_res != 0
            and ad_info.m_Name.decode("ascii")[:2] == "AI"
        ):
            ai_channels.append(i)

    acqs = None
    fs = np.zeros(len(wb_channels))
    coeffs = np.zeros(len(wb_channels))
    timestamps = np.zeros(len(wb_channels))
    units = []
    enabled = np.zeros(len(wb_channels), np.int16)
    for index, i in enumerate(wb_channels):
        ad_info = PL2AnalogChannelInfo()
        ad_res = reader.pl2_get_analog_channel_info(handle, i, ad_info)
        data = pl2_ad(pl2_path, i)
        enabled[index] = 1
        if acqs is None:
            if end is None or end > data.ad.size:
                end = data.ad.size
            acqs = np.zeros((len(wb_channels), end - start), np.int16)
        acqs[index] = data.ad[start:end]
        fs[index] = data.adfrequency
        coeffs[index] = data.coeff
        timestamps[index] = data.timestamps[0]
        units.append(ad_info.m_Units)

    ais = None
    ai_data = None
    ai_fs = np.zeros(len(ai_channels))
    ai_coeffs = np.zeros(len(ai_channels))
    ai_timestamp = np.zeros(len(ai_channels))
    ai_units = []
    for index, i in enumerate(ai_channels):
        ad_info = PL2AnalogChannelInfo()
        ad_res = reader.pl2_get_analog_channel_info(handle, i, ad_info)
        data = pl2_ad(pl2_path, i)
        enabled[index] = 1
        if ais is None:
            if end is None or end > data.ad.size:
                end = data.ad.size
            ais = np.zeros((len(ai_channels), end - start), np.int16)
        ais[index] = data.ad[start:end]
        ai_fs[index] = data.adfrequency
        ai_timestamp[index] = data.timestamps[0]
        ai_coeffs[index] = data.coeff
        ai_units.append(ad_info.m_Units)
    ai_data = (ais, ai_fs, ai_coeffs, np.asarray(ai_units), ai_timestamp)
    acq_man = AcqManager()
    if save_path == "":
        save_path = Path(pl2_path).parent
    acq_man.create_hdf5_file(
        acqs, wb_channels, fs, coeffs, timestamps, np.asarray(units), enabled, name, save_path, ai=ai_data
    )
    # acq_data = (acqs, fs, coeffs, units, enabled, name, save_path)
    reader.pl2_close_file(handle)
    return acq_man


def load_hdf5_acqs(directory: str):
    path = Path(directory)
    acqs = []
    for i in path.rglob("*.hdf5"):
        acq = AcqManager()
        acq.load_hdf5(i)
        acqs.append(acq)
    return acqs


def load_acq(
    file_path: str,
    channel: int,
    save_path: Union[None, str] = None,
    start: int = 0,
    end: int = 24000000,
):
    acq = AcqManager()
    if not _path_checker(file_path, save_path):
        return None
    elif PurePath(file_path).suffix == ".pl2":
        data = pl2_ad(file_path, channel)
        acq.load_pl2_acq(
            data.ad,
            int(data.adfrequency),
            channel,
            start=start,
            end=end,
            save_path=save_path,
        )
        return acq
    elif PurePath(file_path).suffix == ".hdf5":
        acq.load_hdf5_file(file_path)
        return acq
    else:
        RuntimeError("File type not recognized.")
        return None


def _path_checker(file_path: str, save_path: str):
    if save_path is None:
        if Path(file_path).exists():
            return True
        else:
            FileNotFoundError("The file path does not exist.")
            return False
    elif save_path is not None:
        if Path(save_path).exists() and Path(file_path).exists():
            return True
        elif not Path(save_path).exists() and Path(file_path).exists():
            NotADirectoryError("The save path is not a directory.")
            return False
        elif Path(save_path).exists() and not Path(file_path).exists():
            FileNotFoundError("The file path does not exist.")
            return False
        else:
            return False
    else:
        return False
