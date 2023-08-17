# %%
import multiprocessing as mp
from pathlib import Path

from invivosuite.acq import AcqManager

"""This is a template for multiprocessing files
"""


def multi_func(file_path):
    acq_manager = AcqManager()
    acq_manager.open_hdf5_file(file_path)
    acq_manager.find_lfp_bursts(
        window="hamming",
        min_len=0.2,
        max_len=20.0,
        min_burst_int=0.2,
        wlen=0.2,
        threshold=5,
        pre=3.0,
        post=3.0,
        order=0.1,
        method="spline",
        tol=0.001,
        deg=90,
    )
    return True


if __name__ == "__main__":
    file_paths = list(Path(r"D:\in_vivo_ephys\acqs").rglob("*.hdf5"))
    pool = mp.Pool(mp.cpu_count())
    with mp.Pool() as pool:
        result = pool.map(multi_func, file_paths)
        print(result)
