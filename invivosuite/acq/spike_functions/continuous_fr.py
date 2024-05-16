import numpy as np
from scipy import ndimage
from scipy.signal import windows


def getSDF(
    spikes, startTimes, windowDur, sampInt=0.001, filt="gaussian", sigma=0.02, avg=True
):
    t = np.arange(0, windowDur + sampInt, sampInt)
    counts = np.zeros((startTimes.size, t.size - 1))
    for i, start in enumerate(startTimes):
        counts[i] = np.histogram(
            spikes[(spikes >= start) & (spikes <= start + windowDur)] - start, t
        )[0]

    if filt in ("exp", "exponential"):
        filtPts = int(5 * sigma / sampInt)
        expFilt = np.zeros(filtPts * 2)
        expFilt[-filtPts:] = windows.exponential(
            filtPts, center=0, tau=sigma / sampInt, sym=False
        )
        expFilt /= expFilt.sum()
        sdf = ndimage.filters.convolve1d(counts, expFilt, axis=1)
    else:
        sdf = ndimage.filters.gaussian_filter1d(counts, sigma / sampInt, axis=1)
    if avg:
        sdf = sdf.mean(axis=0)
    sdf /= sampInt
    return sdf, t[:-1]
