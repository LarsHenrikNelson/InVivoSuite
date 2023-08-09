from collections import namedtuple
from typing import Union
from pathlib import Path, PurePath

import numpy as np

from .pypl2 import (
    PL2FileInfo,
    PL2AnalogChannelInfo,
    PL2SpikeChannelInfo,
    PL2DigitalChannelInfo,
    PyPL2FileReader,
    pl2_ad,
    pl2_spikes,
    pl2_events,
    pl2_info,
    pl2_comments,
)
from .acq_manager import AcqManager

PL2Ad = namedtuple("PL2Ad", "adfrequency n timestamps fragmentcounts ad")


def load_pl2_acqs(
    pl2_path: str,
    save_path: str = "",
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
    channels = file_info.m_TotalNumberOfSpikeChannels
    acqs = None
    fs = np.zeros(channels)
    coeffs = np.zeros(channels)
    units = []
    enabled = np.zeros(channels, np.int16)
    for i in range(0, channels):
        ad_info = PL2AnalogChannelInfo()
        ad_res = reader.pl2_get_analog_channel_info(handle, i, ad_info)
        if (
            ad_info.m_ChannelEnabled
            and ad_res != 0
            and ad_info.m_Name.decode("ascii")[:2] == "WB"
        ):
            data = pl2_ad(pl2_path, i)
            enabled[i] = 1
            if acqs is None:
                acqs = np.zeros((channels, data.ad.size), np.int16)
            acqs[i] = data.ad[: data.ad.size]
            fs[i] = data.adfrequency
            coeffs[i] = data.coeff
            units.append(ad_info.m_Units)
    acq_man = AcqManager()
    if save_path == "":
        save_path = Path(pl2_path).parent
    acq_man.create_hdf5_file(acqs, fs, coeffs, units, enabled, name, save_path)
    reader.pl2_close_file(handle)
    return acq_man


def load_hdf5_acqs(directory: str):
    path = Path(directory)
    acqs = []
    for i in path.rglob("*.hdf5"):
        acq = AcqManager()
        acq.open_hdf5_file(i)
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


def process_acqs(acqs):
    for i in acqs:
        i.create_lfp()
        i.create_spike()


if __name__ == "__main__":
    load_pl2_acqs()
    load_hdf5_acqs()
    load_acq()
