from typing import Literal

import numpy as np


__all__ = ["get_freq_window"]


def get_freq_window(
    pxx: np.ndarray,
    freqs: np.ndarray,
    lower_limit: float,
    upper_limit: float,
    log_transform: bool = True,
    window_type: Literal["sum", "mean"] = "sum",
):
    f_lim = np.where((freqs <= upper_limit) & (freqs >= lower_limit))
    pxx_freq = pxx[f_lim]
    if log_transform:
        pxx_freq = np.log10(pxx_freq)
    if window_type == "sum":
        ave = np.sum(pxx_freq, axis=0)
    elif window_type == "mean":
        ave = np.mean(pxx_freq, axis=0)
    return ave
