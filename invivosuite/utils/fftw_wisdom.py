from pathlib import Path
from typing import Literal, NamedTuple
import pyfftw

from ..utils import config


__all__ = [
    "import_wisdom",
    "save_wisdom",
    "load_wisdom",
    "FFTDATA",
]


class FFTDATA(NamedTuple):
    length: int
    dtype: Literal["complex128", "float64"]


def import_wisdom(wisdom):
    pyfftw.import_wisdom(wisdom)


def create_wisdom(inputs: list[FFTDATA], outputs: list[str]):
    for i, o in zip(inputs, outputs):
        if i.dtype == "float64" and o.dtype == "complex128":
            a = pyfftw.empty_aligned(i.length, dtype="float64")
            b = pyfftw.empty_aligned(i.length // 2 + 1, dtype="complex128")

            fft_object = pyfftw.FFTW(a, b)  # noqa
        elif i.dtype == "complex128" and o.dtype == "complex128":
            a = pyfftw.empty_aligned(i.length, dtype="complex128")
            b = pyfftw.empty_aligned(i.length, dtype="complex128")

            fft_object = pyfftw.FFTW(a, b)  # noqa
    return pyfftw.export_wisdom()


def save_wisdom(wisdom):
    save_path = Path(f"{config.PROG_DIR}")
    for index, i in enumerate(wisdom):
        with open(save_path / f"{index}_wisdom", "wb") as f:
            f.write(i)


def load_wisdom():
    temp_path = list(Path(f"{config.PROG_DIR}").glob("*_wisdom"))
    if len(temp_path) > 0:
        data = []
        for i in temp_path:
            with open(i, "r") as rf:
                data.append(rf.read())
        ex = tuple(str.encode(i) for i in data)
        pyfftw.import_wisdom(ex)
