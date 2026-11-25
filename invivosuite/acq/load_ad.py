from collections import namedtuple
from pathlib import Path

import numpy as np

from .acq_manager import AcqManager

PL2Ad = namedtuple("PL2Ad", "adfrequency n timestamps fragmentcounts ad")

__all__ = [
    "load_hdf5_acqs",
    "load_pl2_acqs",
]


def _import_pypl2():
    """Lazily import the Plexon PL2 wrappers (requires the Plexon DLL, Windows only).

    Importing at module load time would make the whole package unimportable on
    machines without the DLL (e.g. CI runners); keep these imports inside the
    functions that actually read PL2 files.
    """
    from .pypl2 import PL2AnalogChannelInfo, PL2FileInfo, PyPL2FileReader, pl2_ad

    return PL2AnalogChannelInfo, PL2FileInfo, PyPL2FileReader, pl2_ad


def load_pl2_acqs(
    pl2_path: str | Path,
    save_path: str | Path = "",
    start: int = 0,
    end: int | None = None,
):
    name = Path(pl2_path).stem
    PL2AnalogChannelInfo, PL2FileInfo, PyPL2FileReader, pl2_ad = _import_pypl2()
    if save_path == "":
        save_path = Path(pl2_path).with_suffix(".hdf5")
    else:
        save_path = Path(save_path)
        if save_path.suffix != ".hdf5":
            save_path = save_path.with_suffix(".hdf5")
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
    acq_man = AcqManager(save_path)
    acq_man.create_hdf5_file(
        acqs, wb_channels, fs, coeffs, timestamps, np.asarray(units), enabled, name, ai=ai_data
    )
    reader.pl2_close_file(handle)
    return acq_man


def load_hdf5_acqs(directory: str):
    """Load every HDF5 file under ``directory`` into a list of ``AcqManager``
    instances (lazy — the files stay on disk and data is only read on access).

    Returns base ``AcqManager`` instances (the data-access layer). Wrap them
    in ``SpkLFPManager`` (via ``invivosuite.acq.SpkLFPManager``) before
    running spike/LFP analyses.
    """
    path = Path(directory)
    acqs = []
    for i in path.rglob("*.hdf5"):
        acqs.append(AcqManager(i))
    return acqs
