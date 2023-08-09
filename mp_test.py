# %%
import multiprocessing as mp
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from invivosuite.acq import AcqManager, lfp, load_hdf5_acqs, load_pl2_acqs

"""This is a template for multiprocessing files
"""


# %%
def multi_func(file_path):
    acq_manager = AcqManager()
    acq_manager.set_hdf5_file(file_path)
    acq_manager.find_lfp_bursts(
        window="hamming",
        min_len=0.2,
        max_len=20,
        min_burst_int=0.2,
        wlen=0.2,
        threshold=10,
        pre=3,
        post=3,
        order=100,
        method="spline",
        tol=0.001,
        deg=90,
    )
    return True


# %%
if __name__ == "__main__":
    file_paths = list(Path(r"D:\in_vivo_ephys\acqs").rglob("*.hdf5"))
    pool = mp.Pool(mp.cpu_count())
    with mp.Pool() as pool:
        result = pool.map(multi_func, file_paths)
        print(result)
