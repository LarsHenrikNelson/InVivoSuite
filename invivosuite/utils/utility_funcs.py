import math
from typing import Union

import numpy as np

__all__ = [
    "round_sig",
    "save_tsv",
    "tsv_row",
]


def round_sig(x, sig=4):
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
