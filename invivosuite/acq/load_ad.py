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
    acqs = None
    channels = []
    for i in range(nchannels):
        ad_info = PL2AnalogChannelInfo()
        ad_res = reader.pl2_get_analog_channel_info(handle, i, ad_info)
        if (
            ad_info.m_ChannelEnabled
            and ad_res != 0
            and ad_info.m_Name.decode("ascii")[:2] == "WB"
        ):
            channels.append(ad_info.m_Channel-1)
    fs = np.zeros(len(channels))
    coeffs = np.zeros(len(channels))
    units = []
    enabled = np.zeros(len(channels), np.int16)
    for i in channels:
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
                if end is None or end > data.ad.size:
                    end = data.ad.size
                acqs = np.zeros((channels, end - start), np.int16)
            acqs[i] = data.ad[start:end]
            fs[i] = data.adfrequency
            coeffs[i] = data.coeff
            units.append(ad_info.m_Units)
    acq_man = AcqManager()
    if save_path == "":
        save_path = Path(pl2_path).parent
    acq_man.create_hdf5_file(
        acqs, fs, coeffs, np.asarray(units), enabled, name, save_path
    )
    # acq_data = (acqs, fs, coeffs, units, enabled, name, save_path)
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
