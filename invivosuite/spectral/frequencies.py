from typing import Literal

import numpy as np

from .wavelets import Wavelet
from .wavelet_utils import f_to_s


class Frequencies:
    def __init__(
        self,
        wavelet: Wavelet,
        f0: float,
        f1: float,
        fn: float,
        fs: float,
        scaling: Literal["linear", "log"] = "log",
    ):
        self.wavelet = wavelet
        self.f0 = f0
        self.f1 = f1
        self.fn = fn
        self.fs = fs
        self.scaling = scaling

        if scaling == "linear":
            self.f = np.linspace(start=f0, stop=f1, num=fn)[::-1]
        else:
            self.f = np.logspace(start=np.log10(f0), stop=np.log10(f1), num=fn)[::-1]

        self.s = f_to_s(self.f, self.fs, self.wavelet.n_cycles)

