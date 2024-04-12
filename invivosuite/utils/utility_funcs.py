from typing import Union

__all__ = ["save_tsv", "tsv_row"]


def tsv_row(data: Union[list, tuple]):
    num_values = len(data)
    temp_str = ""
    for i in range(num_values):
        if i < (num_values - 1):
            temp_str = temp_str + f"{data[i]}  "
        else:
            temp_str = temp_str + f"{data[i]}\n"
    return temp_str


def save_tsv(
    name: str,
    data: dict,
):
    with open(f"{name}.tsv", "w") as record_file:
        keys = data.keys()
        record_file.write(tsv_row(keys))
        size = len(data[keys[0]])
        for i in range(size):
            for j in keys:
                t = []
                t.append(data[j][i])
            row = tsv_row(t)
            record_file.write(row)
