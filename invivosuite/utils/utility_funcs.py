import math
from typing import Union

import numpy as np
from numba import njit

__all__ = [
    "expand_data",
    "concatenate_dicts",
    "round_sig",
    "save_tsv",
    "tsv_row",
]


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
