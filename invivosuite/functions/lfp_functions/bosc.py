from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class Ratio:
    percentile: float | int = 95.0
    correct_multiple_comparisons: bool = False


@dataclass
class Threshold:
    threshold: float | int = 2.0


def bosc_oscillations(
    cwt_power: np.ndarray,
    aperiodic_power: np.ndarray | None = None,
    threshold_type: Ratio | Threshold | None = None,
) -> np.ndarray:
    """_summary_

    Parameters
    ----------
    cwt_power : np.ndarray (n_freqs, n_times)
        Power spectrum density from CWT (amplitude squared)
    aperiodic_power : np.ndarray | None, optional
        Aperiodic power spectrum density. If None then aperiodic_power is the time average.
        of the cwt_power. Default is None.
    threshold_type : Ratio | Threshold | None, optional
        Threshold type to use for finding oscillations. Accepts Ratio or Threshold.
        By default a Ratio instance will be created. Default is None.

    Returns
    -------
    np.ndarray
        Binary array of the same shape as cwt_power where 1 is an oscillation.
    """

    if threshold_type is None:
        threshold_type = Ratio()

    if aperiodic_power is None:
        aperiodic_2d = cwt_power.mean(axis=1, keepdims=True)
    else:
        aperiodic_2d = aperiodic_power[:, np.newaxis]

    if isinstance(threshold_type, Ratio):
        if threshold_type.correct_multiple_comparisons:
            # Bonferroni-style: adjust percentile for n_pixels
            n_pixels = cwt_power.shape[0] * cwt_power.shape[1]
            adjusted_p = (1 - threshold_type.percentile / 100) / n_pixels
            threshold = stats.chi2.ppf(1 - adjusted_p, df=2) / 2
        else:
            threshold = stats.chi2.ppf(threshold_type.percentile / 100, df=2) / 2
        normalized = cwt_power / aperiodic_2d
        is_oscillation = normalized > threshold
    elif isinstance(threshold_type, Threshold):
        is_oscillation = cwt_power > (aperiodic_2d * threshold_type.threshold)
    return is_oscillation
