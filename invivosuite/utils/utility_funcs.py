import math
import os
from tempfile import mkdtemp
from typing import Any, Tuple, Union

import numpy as np
from numba import njit

__all__ = [
    "concatenate_dicts",
    "expand_data",
    "round_sig",
    "save_tsv",
    "split_at_zeros",
    "TempMemmap",
    "tsv_row",
]


@njit()
def split_at_zeros(array):
    array_list = []
    temp_list = []
    for index, i in enumerate(array):
        if i > 0:
            temp_list.append(index)
        else:
            if temp_list:
                array_list.append(temp_list)
            temp_list = []
    return array_list


@njit(cache=True)
def expand_data(data: np.ndarray, counts: np.ndarray) -> np.ndarray:
    total = np.sum(counts)
    output = np.zeros(total)
    j = 0
    for k in range(data.size):
        for _ in range(counts[k]):
            output[j] = data[k]
            j += 1
    return output


def concatenate_dicts(data_list: dict) -> dict:
    if len(data_list) > 1:
        output = {k: [] for k in data_list[0].keys()}
        for data in data_list:
            for key in data.keys():
                output[key].append(data[key])
        output = {
            k: (
                np.concatenate(j)
                if isinstance(j[0], (list, np.ndarray))
                else np.array(j)
            )
            for k, j in output.items()
        }
    else:
        output = data
    return output


def round_sig(x: float, sig=4):
    if np.isnan(x):
        return np.nan
    elif x == 0:
        return 0
    elif x != 0 or not np.isnan(x):
        temp = math.floor(math.log10(abs(x)))
        if np.isnan(temp):
            return round(x, 0)
        else:
            return round(x, sig - int(temp) - 1)


def tsv_row(data: Union[list, tuple]):
    num_values = len(data)
    temp_str = ""
    for i in range(num_values):
        if i < (num_values - 1):
            temp_str = temp_str + f"{data[i]}\t"
        else:
            temp_str = temp_str + f"{data[i]}\n"
    return temp_str


def save_tsv(name: str, data: dict, mode: str = "w", encoding: str = "utf-8"):
    with open(f"{name}.tsv", mode, encoding=encoding) as record_file:
        keys = list(data.keys())
        record_file.write("\t".join(keys) + "\n")
        size = len(data[keys[0]])
        for i in range(size):
            t = []
            for j in keys:
                if isinstance(data[j][i], (str, int)):
                    t.append(data[j][i])
                else:
                    t.append(round_sig(data[j][i]))
            # row = tsv_row(t)
            record_file.write("\t".join(str(x) for x in t) + "\n")

class TempMemmap(np.memmap):
    def __new__(cls, shape: Tuple[int, ...], dtype: Any = 'float32'):
        temp_dir = mkdtemp()
        temp_path = os.path.join(temp_dir, 'temp_array.dat')
        
        obj = super(TempMemmap, cls).__new__(
            cls,
            temp_path,
            dtype=dtype,
            mode='w+',
            shape=shape
        )
        
        obj.temp_dir = temp_dir
        obj.temp_path = temp_path
        return obj

    def __del__(self):
        try:
            super().__del__()  # Call parent's __del__ first
        except Exception:
            pass

        # Then delete the temporary files
        try:
            if hasattr(self, 'temp_path') and os.path.exists(self.temp_path):
                os.unlink(self.temp_path)
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                os.rmdir(self.temp_dir)
        except Exception:
            pass
