from typing import Literal

import numpy as np
from scipy.signal import welch, resample, detrend as sp_detrend
from scipy.interpolate import interp1d

def _default_hset(mode='standard'):
    """
    Recommended resampling factor sets.
    """
    if mode == 'fast':
        # 7 values, well-spaced — good for exploration
        return np.array([1.1, 1.2, 1.3, 1.5, 1.6, 1.75, 1.9])

    elif mode == 'standard':
        # 17 values — the Wen & Liu (2016) original
        return np.arange(1.1, 1.95, 0.05)

    elif mode == 'conservative':
        # Skip values close to 1 (weak smearing) and near-integer
        # ratios (potential harmonic preservation)
        return np.array([1.2, 1.3, 1.4, 1.55, 1.65, 1.75, 1.85, 1.9])
    elif model == "robust":
        return np.array([ 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9
        ])

def irasa(data, fs, band=(1, 100), win_sec=4, overlap_frac=0.5,
          hset: int | Literal["fast", "standard", "conservative", "robust"]="fast", detrend=True):
    """
    Irregular-Resampling Auto-Spectral Analysis (IRASA).

    Separates the aperiodic (fractal / 1-over-f) and periodic (oscillatory)
    components of a power spectrum without assuming a parametric form for
    either component.

    Reference
    ---------
    Wen H & Liu Z (2016). Separating Fractal and Oscillatory Components in
    the Power Spectrum of Neurophysiological Signal. Brain Topography, 29(1).

    Parameters
    ----------
    data : array_like, 1-D
        Raw time series (single channel).
    fs : float
        Sampling frequency in Hz.
    band : tuple of (float, float)
        Frequency range of interest in Hz. The lower bound should be >= 1/win_sec
        and the upper bound should be well below fs/2.
    win_sec : float
        Window length in seconds for internal Welch PSD estimation. Controls
        frequency resolution (1 / win_sec Hz). Must be long enough to resolve
        the lowest frequency in `band`.
    overlap_frac : float
        Fractional overlap between Welch segments, 0-1. Default 0.5.
    hset : array_like or None
        Set of resampling factors > 1. Each h produces an up-sampled (×h)
        and down-sampled (÷h) version of the signal. Default is
        np.arange(1.1, 1.95, 0.05), i.e. 17 factors.
    detrend : bool
        If True, linear-detrend the data before analysis.

    Returns
    -------
    freqs : 1-D np.ndarray
        Frequency vector (within `band`).
    psd_aperiodic : 1-D np.ndarray
        Aperiodic (fractal) power spectral density.
    psd_periodic : 1-D np.ndarray
        Periodic (oscillatory) power spectral density (original − aperiodic).
        Can contain small negative values at frequencies with no oscillatory
        power; these are within estimation noise.
    """
    data = np.asarray(data, dtype=np.float64).ravel()
    if detrend:
        data = sp_detrend(data, type='linear')

    if isinstance(hset, str):
        hset = _default_hset(mode=hset)

    n_samples = len(data)
    nperseg = int(win_sec * fs)
    noverlap = int(nperseg * overlap_frac)

    # --- Validate inputs ---
    freq_resolution = 1.0 / win_sec
    if band[0] < freq_resolution:
        raise ValueError(
            f"Lower band edge ({band[0]} Hz) is below the frequency "
            f"resolution ({freq_resolution:.3f} Hz). Increase win_sec to "
            f"at least {1.0 / band[0]:.1f} s."
        )
    min_nyquist = fs / hset.max() / 2.0
    if band[1] > min_nyquist:
        raise ValueError(
            f"Upper band edge ({band[1]} Hz) exceeds the Nyquist of the "
            f"most down-sampled signal ({min_nyquist:.1f} Hz). Lower the "
            f"upper band edge or reduce max(hset)."
        )
    min_samples_down = int(np.round(n_samples / hset.max()))
    nperseg_down = int(np.round(nperseg / hset.max()))
    if nperseg_down < 8:
        raise ValueError(
            f"The most aggressive down-sampling (h={hset.max():.2f}) "
            f"produces a window of only {nperseg_down} samples. "
            f"Decrease max(hset) or increase win_sec."
        )

    # --- Original PSD (what we decompose) ---
    freqs_orig, psd_orig = welch(
        data, fs=fs, nperseg=nperseg, noverlap=noverlap, window='hamming'
    )
    mask = (freqs_orig >= band[0]) & (freqs_orig <= band[1])
    freqs = freqs_orig[mask]
    psd_orig = psd_orig[mask]

    # --- Resampled PSDs ---
    aperiodic_estimates = np.zeros((len(hset), len(freqs)))

    for i, h in enumerate(hset):
        # Up-sample by h
        n_up = int(np.round(n_samples * h))
        nperseg_up = int(np.round(nperseg * h))
        noverlap_up = int(nperseg_up * overlap_frac)
        data_up = resample(data, n_up)
        f_up, p_up = welch(
            data_up, fs=fs * h,
            nperseg=nperseg_up, noverlap=noverlap_up, window='hamming'
        )

        # Down-sample by 1/h
        n_down = int(np.round(n_samples / h))
        nperseg_down = int(np.round(nperseg / h))
        noverlap_down = int(nperseg_down * overlap_frac)
        data_down = resample(data, n_down)
        f_down, p_down = welch(
            data_down, fs=fs / h,
            nperseg=nperseg_down, noverlap=noverlap_down, window='hamming'
        )

        # Interpolate both onto the common frequency grid
        interp_up = interp1d(f_up, p_up, kind='linear',
                             bounds_error=False, fill_value=np.nan)
        interp_down = interp1d(f_down, p_down, kind='linear',
                               bounds_error=False, fill_value=np.nan)

        p_up_i = interp_up(freqs)
        p_down_i = interp_down(freqs)

        # Geometric mean preserves fractal component, smears oscillatory
        aperiodic_estimates[i] = np.sqrt(p_up_i * p_down_i)

    # Median across resampling factors (robust to residual peak leakage)
    psd_aperiodic = np.median(aperiodic_estimates, axis=0)
    psd_periodic = psd_orig - psd_aperiodic

    return freqs, psd_aperiodic, psd_periodic


def fit_aperiodic_params(freqs, psd_aperiodic):
    """
    Fit a linear model in log-log space to the aperiodic PSD to extract
    the offset and exponent: log10(P) = offset - exponent * log10(f).

    Parameters
    ----------
    freqs : 1-D array
        Frequency vector (output of `irasa`).
    psd_aperiodic : 1-D array
        Aperiodic PSD (output of `irasa`).

    Returns
    -------
    params : dict
        'offset' : float — intercept in log10 space (overall power level).
        'exponent' : float — slope magnitude (positive; steeper = higher).
        'r_squared' : float — goodness of fit.
        'fit_line' : 1-D array — the fitted PSD (linear space), same length
            as `freqs`, for plotting.
    """
    valid = (freqs > 0) & (psd_aperiodic > 0) & np.isfinite(psd_aperiodic)
    log_f = np.log10(freqs[valid])
    log_p = np.log10(psd_aperiodic[valid])

    # Linear regression in log-log space
    slope, intercept = np.polyfit(log_f, log_p, 1)

    # Goodness of fit
    predicted = intercept + slope * log_f
    ss_res = np.sum((log_p - predicted) ** 2)
    ss_tot = np.sum((log_p - log_p.mean()) ** 2)
    r_squared = 1.0 - ss_res / ss_tot

    # Full fit line (including any NaN positions)
    fit_line = np.full_like(psd_aperiodic, np.nan)
    fit_line[valid] = 10 ** (intercept + slope * np.log10(freqs[valid]))

    return {
        'offset': intercept,
        'exponent': -slope,     # convention: positive exponent
        'r_squared': r_squared,
        'fit_line': fit_line,
    }


def irasa_band_power(freqs, psd, bands=None):
    """
    Integrate power within canonical frequency bands via trapezoidal rule.

    Parameters
    ----------
    freqs : 1-D array
        Frequency vector.
    psd : 1-D array
        Power spectrum (aperiodic, periodic, or original).
    bands : dict or None
        {name: (low, high)} in Hz. Default: standard neuro bands.

    Returns
    -------
    dict : {band_name: power}
    """
    if bands is None:
        bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'low_gamma': (30, 60),
            'high_gamma': (60, 100),
        }

    result = {}
    for name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs <= hi)
        if mask.sum() < 2:
            result[name] = np.nan
        else:
            result[name] = np.trapz(psd[mask], freqs[mask])
    return result