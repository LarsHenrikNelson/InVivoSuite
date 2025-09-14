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
        steps: float,
        fs: float,
        scaling: Literal["linear", "log"] = "log",
    ):
        self.wavelet = wavelet
        self.f0 = f0
        self.f1 = f1
        self.steps = steps
        self.fs = fs
        self.scaling = scaling

        if scaling == "linear":
            self.freqs = np.linspace(start=f0, stop=f1, num=steps)
        else:
            self.freqs = np.logspace(start=np.log10(f0), stop=np.log10(f1), num=steps)

        self.scales = f_to_s(self.freqs, self.fs, self.wavelet.n_cycles)

