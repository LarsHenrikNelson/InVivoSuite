from typing import Literal, Union

import numpy as np

from ..signal_functions import kde


def stepped_cwt_cohy(cwt: np.ndarray, size: int):
    noverlap = 0
    nperseg = size
    step = nperseg - noverlap
    shape = cwt.shape[:-1] + ((cwt.shape[-1] - noverlap) // step, nperseg)
    strides = cwt.strides[:-1] + (step * cwt.strides[-1], cwt.strides[-1])
    temp = np.lib.stride_tricks.as_strided(cwt, shape=shape, strides=strides)
    coh = []
    for i in range(1, temp.shape[1], 1):
        sxy = np.mean(temp[:, i - 1, :], axis=1) * np.conjugate(
            np.mean(temp[:, i, :], axis=1)
        )
        sxx = np.mean(temp[:, i - 1, :], axis=1) * np.conjugate(
            np.mean(temp[:, i - 1, :], axis=1)
        )
        syy = np.mean(temp[:, i, :], axis=1) * np.conjugate(
            np.mean(temp[:, i, :], axis=1)
        )
        coh_temp = sxy / np.sqrt(sxx * syy + 1e-8)
        coh.append(coh_temp)
    coh = np.array(coh).T
    return coh


def phase_discontinuity_index(
    cwt: np.ndarray,
    freqs: np.ndarray,
    freq_dict: dict[str, Union[tuple[int, int], tuple[float, float]]],
    size: int = 5000,
    tol: float = 0.01,
    bw_method: Literal["ISJ", "silverman", "scott"] = "ISJ",
) -> dict[str, float]:
    pdi = {}
    coh = stepped_cwt_cohy(cwt, size)
    coh = np.diff(np.angle(coh))
    for key, value in freq_dict.items():
        f_lim = np.where((freqs <= value[1]) & (freqs >= value[0]))[0]
        g = coh[f_lim]
        g = g.flatten()
        x, y = kde(g, tol=tol, bw_metho=bw_method)
        # power2 = int(np.ceil(np.log2(g.size)))
        # width = np.cov(g)
        # min_g = g.min() - width * tol
        # max_g = g.max() + width * tol
        # x = np.linspace(min_g, max_g, num=1 << power2)
        # y = KDEpy.FFTKDE(bw="ISJ").fit(g).evaluate(x)
        args1 = np.where((x > np.pi / 5) | (x < -np.pi / 5))[0]
        args2 = np.where((x <= np.pi / 5) & (x >= -np.pi / 5))[0]
        pdi[key] = np.sum(y[args1]) / np.sum(y[args2])
    return pdi
