from collections import namedtuple
from typing import Union
from pathlib import Path, PurePath

from .pypl2 import PL2FileInfo, PL2AnalogChannelInfo, PyPL2FileReader, pl2_ad
from .acq_manager import AcqManager

PL2Ad = namedtuple("PL2Ad", "adfrequency n timestamps fragmentcounts ad")


def load_pl2_acqs(
    file_path: str,
    save_path: str,
    start: int = 0,
    end: int = 24000000,
    identifier: str = "",
):
    if not _path_checker(file_path, save_path):
        return None
    reader = PyPL2FileReader()
    handle = reader.pl2_open_file(file_path)
    file_info = PL2FileInfo()
    res = reader.pl2_get_file_info(handle, file_info)
    if res == 0:
        return None
    channels = file_info.m_TotalNumberOfAnalogChannels
    ad_channels = []
    for i in range(0, channels):
        ad_info = PL2AnalogChannelInfo()
        ad_res = reader.pl2_get_analog_channel_info(handle, i, ad_info)
        if (
            ad_info.m_ChannelEnabled
            and ad_res != 0
            and ad_info.m_Name.decode("ascii")[:2] == "WB"
        ):
            data = pl2_ad(file_path, i)
            acq = AcqManager()
            acq.load_pl2_acq(
                data.ad, int(data.adfrequency), i, start, end, identifier, save_path
            )
            ad_channels.append(acq)

    reader.pl2_close_file(handle)
    return ad_channels


def load_hdf5_acqs(directory: str):
    path = Path(directory)
    acqs = []
    for i in path.glob("*.hdf5"):
        acq = AcqManager()
        acq.load_hdf5_file(i)
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
