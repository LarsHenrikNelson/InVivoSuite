from typing import Literal, Union

import numpy as np
import statsmodels as sm
from scipy import signal
from tapered_spectra import multitaper


def find_logpx_baseline(
    acq,
    fs=1000,
    freqs=(0, 100),
    nperseg=10000,
    noverlap=5000,
    nfft=10000,
    method: Literal[
        "AndrewWave",
        "TrimmedMean",
        "RamsayE",
        "HuberT",
        "TukeyBiweight",
        "Hampel",
        "LeastSquares",
    ] = "AndrewWave",
    window: str = "hann",
    NW: float = 2.5,
    BW: Union[float, None] = None,
    adaptive=False,
    jackknife=True,
    low_bias=True,
    sides="default",
    NFFT=None,
):
    if window != "dpss":
        f, px = signal.welch(acq, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    elif window == "dpss":
        f, px, _ = multitaper(
            acq,
            fs=fs,
            NW=NW,
            BW=BW,
            adaptive=adaptive,
            jackknife=jackknife,
            low_bias=low_bias,
            sides=sides,
            NFFT=NFFT,
        )
    else:
        raise ValueError("Window must be specified.")
    ind = np.where((f <= freqs[1]) & (f >= freqs[0]))[0]
    x = np.ones((ind.size, 2))
    x[:, 1] = f[ind]
    y = np.log10(px[ind]).reshape((ind.size, 1))
    match method:
        case "AndrewWave":
            M = sm.robust.norms.AndrewWave()
        case "TrimmedMean":
            M = sm.robust.norms.TrimmedMean()
        case "HuberT":
            M = sm.robust.norms.HuberT()
        case "RamsayE":
            M = sm.robust.norms.RamsayE()
        case "TukeyBiweight":
            M = sm.robust.norms.TukeyBiweight()
        case "Hampel":
            M = sm.robust.norms.Hampel()
        case "LeastSquares":
            M = sm.robust.norms.LeastSquares()
    rlm_m = sm.RLM(y, x, M=M)
    rlm_r = rlm_m.fit(cov="H2")
    rline = rlm_r.params[1] * f[ind] + rlm_r.params[0]
    return f[ind], px[ind], rline
