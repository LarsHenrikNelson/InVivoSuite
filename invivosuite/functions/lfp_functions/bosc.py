from typing import Literal
from dataclasses import dataclass

from ...spectral import PyFCWT

import numpy as np
from scipy import stats

@dataclass
class Threshold:
    aperiodic_fit: np.ndarray
    percentile: float | int = 95.0
    correct_multiple_comparisons: bool = False

@dataclass
class Aperiodic:
    aperiodic_fit: np.ndarray
    percentile: float | int = 95.0
    n_simulations: int = 1000

@dataclass
class NullFit:
    frequencies: np.ndarray
    sample_rate: float | int
    cwt_class: PyFCWT
    percentile: float | int = 95.0
    n_simulations: int = 100

def compute_aperiodic_threshold(
    aperiodic_fit: np.ndarray,
    n_times: int,
    percentile: float | int = 95,
    n_simulations: int = 1000,
):
    """Compute threshold empirically from null distribution (pure 1/f noise).

    Parameters
    ----------
    aperiodic_fit : np.ndarray
        1/f fit from IRASA or similar (same units as cwt_power)
    n_times : int
        Length of CWT or recording in samples.
    percentile : float | int, optional
        Percentile to use for drawing from chi-square distribution, by default 95.0
    n_simulations : int, optional
        Number of iterations to use for bootstraping the threshold, by default 1000

    Returns
    -------
    _type_
        _description_
    """
    # Simulate chi-squared(2) scaled by aperiodic
    # CWT power of Gaussian noise ~ χ²(2) * variance
    null_ratios = []

    for _ in range(n_simulations):
        # Chi-squared with df=2 has mean=2, so divide by 2 to normalize to mean=1
        null_power = np.random.chisquare(df=2, size=(len(aperiodic_fit), n_times)) / 2
        null_ratios.append(null_power.flatten())

    null_ratios = np.concatenate(null_ratios)
    threshold = np.percentile(null_ratios, percentile)
    return threshold

def compute_null_threshold(
    cwt_power: np.ndarray,
    freqs: np.ndarray,
    sample_rate: float | int,
    cwt_function,
    percentile: int | float = 95,
    n_simulations: int = 100,
):
    """Bootstrap threshold using the actual CWT-averaged spectrum
    as the noise model. No parametric aperiodic assumption.

    Parameters
    ----------
    cwt_power : np.ndarray (n_freq, n_times)
        _description_
    freqs : np.ndarray
        _description_
    sample_rate : float | int
        _description_
    cwt_function : _type_
        _description_
    percentile : int | float, optional
        _description_, by default 95
    n_simulations : int, optional
        _description_, by default 100

    Returns
    -------
    _type_
        _description_
    """

    cwt_avg = cwt_power.mean(axis=1)
    n_times = cwt_power.shape[1]

    # Build noise with the exact observed spectrum shape
    # including theta, gamma, everything
    freqs_fft = np.fft.rfftfreq(n_times, d=1 / sample_rate)
    target_psd = np.interp(freqs_fft, freqs, cwt_avg)
    target_amplitude = np.sqrt(target_psd)

    thresholds = np.empty((n_simulations, len(freqs)))

    for i in range(n_simulations):
        # Surrogate with matched spectrum, random phases
        random_phases = np.exp(2j * np.pi * np.random.uniform(size=len(freqs_fft)))
        surrogate_fft = target_amplitude * random_phases
        surrogate = np.fft.irfft(surrogate_fft, n=n_times)

        # Through actual CWT pipeline
        null_power = cwt_function(surrogate, freqs, sample_rate)

        # Ratio against observed average
        null_ratio = null_power / cwt_avg[:, np.newaxis]
        thresholds[i] = np.percentile(null_ratio, percentile, axis=1)

    return np.mean(thresholds, axis=0)


def bosc_oscillations(
    cwt_power: np.ndarray,
    threshold_type: Threshold | Aperiodic | NullFit = Threshold
) -> np.ndarray:
    """Detect oscillatory episodes above aperiodic background using the BOSC method.

    Parameters
    ----------
    cwt_power : np.ndarray (n_freqs, n_times)
        Power from CWT (amplitude squared)

    Returns
    -------
    np.ndarray
        _description_
    """

    if isinstance(threshold_type, Threshold):
        if threshold_type.correct_multiple_comparisons:
            # Bonferroni-style: adjust percentile for n_pixels
            n_pixels = threshold_type.cwt_power.shape[0] * threshold_type.cwt_power.shape[1]
            adjusted_p = (1 - threshold_type.percentile / 100) / n_pixels
            threshold = stats.chi2.ppf(1 - adjusted_p, df=2) / 2
        else:
            threshold = stats.chi2.ppf(threshold_type.percentile / 100, df=2) / 2
        aperiodic_fit= threshold_type.aperiodic_fit
    elif isinstance(threshold_type, Aperiodic):
        threshold = compute_aperiodic_null_threshold(
            threshold_type.aperiodic_fit, cwt_power.shape[1],threshold_type.percentile, threshold_type.n_iters
        )
        aperiodic_fit= threshold_type.aperiodic_fit
    elif isinstance(threshold_type, NullFit):
        threshold = compute_aperiodic_threshold(
            cwt_power, threshold_type.frequencies, threshold_type.percentile, threshold_type.n_iters
        )
        aperiodic_fit= cwt_power.mean(axis=1)
    aperiodic_2d = aperiodic_fit[:, np.newaxis]
    normalized = cwt_power / aperiodic_2d
    is_oscillation = normalized > threshold
    return is_oscillation