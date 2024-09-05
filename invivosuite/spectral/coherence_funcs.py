from typing import Literal, Union

import numpy as np
from scipy import fft, signal


def coherence(
    acq1: np.ndarray,
    acq2: np.ndarray,
    fs: Union[float, int] = 1000,
    noverlap: int = 1000,
    nperseg: int = 10000,
    nfft: Union[int, None] = None,
    window: Union[str, tuple] = "hamming",
    ret_type: Literal[
        "icohere",
        "mscohere1",
        "mscohere2",
        "lcohere2019",
        "icohere2019",
        "cohy",
    ] = "icohere",
    scaling: Literal["density", "spectrum"] = "density",
):
    # Modified version of scipy to work with the imaginary part of coherence
    acq1 = np.asarray(acq1)
    acq2 = np.asarray(acq2)
    step = nperseg - noverlap
    shape = acq1.shape[:-1] + ((acq1.shape[-1] - noverlap) // step, nperseg)
    strides = acq1.strides[:-1] + (step * acq1.strides[-1], acq1.strides[-1])
    win = signal.get_window(window, nperseg)
    if scaling == "density":
        scale = 1.0 / (fs * (win * win).sum())
    else:
        scale = 1.0
    temp1 = np.lib.stride_tricks.as_strided(acq1, shape=shape, strides=strides)
    temp2 = np.lib.stride_tricks.as_strided(acq2, shape=shape, strides=strides)
    temp1 = temp1 - temp1.mean(axis=1, keepdims=True)
    temp2 = temp2 - temp2.mean(axis=1, keepdims=True)
    temp1 *= win
    temp2 *= win
    if nfft is None:
        nfft = nperseg
    freqs = fft.rfftfreq(nfft, 1 / fs)
    fft1 = fft.rfft(temp1, n=nfft)
    fft2 = fft.rfft(temp2, n=nfft)
    sxx = fft1 * np.conjugate(fft1)
    syy = fft2 * np.conjugate(fft2)
    sxy = fft1 * np.conjugate(fft2)
    sxx *= scale
    sxx = sxx.mean(axis=0)
    syy *= scale
    syy = syy.mean(axis=0)
    sxy *= scale
    sxy = sxy.mean(axis=0)
    if ret_type == "icohere2019":
        # See Nolte et. al. 2004
        output = np.abs((sxy / (np.sqrt(sxx.real * sxy.real) + 1e-18)).imag)
    if ret_type == "icohere":
        # This is from Brainstorm
        cohy = sxy / np.sqrt(sxx * syy)
        output = (cohy.imag**2) / ((1 - cohy.real**2) + 1e-18)
    elif ret_type == "lcohere2019":
        # This is from Brainstorm
        cohy = sxy / np.sqrt(sxx * syy)
        output = np.abs(cohy.imag) / (np.sqrt(1 - cohy.real**2) + 1e-18)
    elif ret_type == "mscohere1":
        output = (np.abs(sxy) ** 2) / (sxx.real * syy.real)
    elif ret_type == "mscohere2":
        output = np.abs(sxy) / np.sqrt(sxx.real * syy.real)
    elif ret_type == "cohy":
        output = sxy / np.sqrt((sxx * syy) + 1e-18)
    # elif ret_type == "plv":
    #     plv = np.abs(sxy / np.sqrt(sxy))
    #     output = plv
    # elif ret_type == "ciplv":
    #     acc = sxy / np.sqrt((sxx * syy) + 1e-18)
    #     iplv = np.abs(acc.imag)
    #     rplv = acc.real
    #     rplv = np.clip(plv, -1, 1)
    #     mask = np.abs(rplv) == 1
    #     rplv[mask] = 0
    #     ciplv = iplv / np.sqrt(1 - rplv**2)
    #     output = ciplv
    # elif ret_type == "pli":
    #     pli = np.abs(np.sign(sxy.imag))
    #     output = cohy
    # elif ret_type == "upli":
    #     pli = np.abs(np.sign(sxy.imag))
    #     upli = pli**2 - 1
    #     output = upli
    # elif ret_type == "dpli":
    #     dpli = np.heaviside(np.imag(sxy), 0.5)
    #     output = dpli
    # elif ret_type == "wpli":
    #     num = np.abs(sxy.imag)
    #     denom = np.abs(sxy.imag)
    #     z_denom = np.where(denom == 0.0)[0]
    #     denom[z_denom] = 1.0
    #     con = num / denom
    #     con[z_denom] = 0.0
    #     output = con
    # elif ret_type == "dwpli":
    #     sum_abs_im_csd = np.abs(sxy.imag)
    #     sum_sq_im_csd = (sxy.imag) ** 2
    #     denom = sum_abs_im_csd**2 - sum_sq_im_csd
    #     z_denom = np.where(denom == 0.0)
    #     denom[z_denom] = 1.0
    #     con = (sxy**2 - sum_sq_im_csd) / denom
    #     con[z_denom] = 0.0
    #     output = con
    # elif ret_type == "ppc":
    #     denom = np.abs(sxy)
    #     z_denom = np.where(denom == 0.0)
    #     denom[z_denom] = 1.0
    #     this_acc = sxy / denom
    #     this_acc[z_denom] = 0.0
    #     (this_acc * np.conj(this_acc) - 1)  # / (1 * (1 - 1))
    else:
        AttributeError(
            "Return type must be icohere, icohere2019, lcohere2019, mscohere or cohy"
        )
    return freqs, output


def phase_lag_value(acqs, plv_type: Literal["plv, iplv, ciplv"] = "ciplv"):
    acqs = acqs / np.abs(acqs)
    sxy = acqs @ acqs.T
    if plv_type == "ciplv":
        top = sxy.imag / acqs.shape[1]
        bottom = np.sqrt((sxy.real / acqs.shape[1]) ** 2)
        return top / bottom
    elif plv_type == "iplv":
        return sxy.imag / acqs.shape[1]
    elif plv_type == "plv":
        return np.abs(sxy / acqs.shape[1])
    else:
        raise AttributeError("plv_type not recogized.")


def phase_slope_index(
    cohy: np.ndarray[np.dtype[np.complex_]],
    freqs: Union[np.ndarray[np.dtype[float], np.dtype[int]], list[Union[float, int]]],
    f_band: Union[tuple[float, float], tuple[int, int]],
) -> float:
    """Calculates the phase slope index
    See: Nolte, G. et al. Robustly Estimating the Flow Direction of Information in Complex Physical Systems. Phys. Rev. Lett. 100, 234101 (2008).

    Args:
        cohy (np.ndarray of complex values): A 1D numpy array.
        freqs (list or np.ndarray of floats or ints): A 1D list or numpy array of floats or ints.
        f_band (tuple): A tuple containing the lower and upper frequency limit.

    Returns:
        float: Phase slope index
    """
    if f_band[0] > f_band[1]:
        f_band = (f_band[1], f_band[0])
    if len(cohy.shape) > 1:
        cohy = cohy.flatten()
    f_ind = np.where((freqs > f_band[0]) & (freqs < f_band[1]))[0]
    psi = np.sum(np.conj(cohy[f_ind[0] : f_ind[-2]] * cohy[f_ind[1] : f_ind[-1]])).imag
    return psi
