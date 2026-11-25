"""Basis source functions for the kCSD method (Jan et al. 2012).

Ported from Elephant's ``current_source_density_src/basis_functions.py``
(BSD-3-Clause), originally based on Grzegorz Parka's GSoC 2014 INFC/pykCSD,
by Michal Czerwinski and Chaitanya Chintaluri (Nencki Institute of
Experimental Biology, Warsaw).

Two source types are provided ('gauss' and 'step'); 'gauss_lim' is the
truncated gaussian. `basis_1D`/`basis_2D` map source-type names to functions.
"""

import numpy as np

__all__ = [
    "basis_1D",
    "basis_2D",
    "gauss",
    "gauss_1D",
    "gauss_2D",
    "gauss_lim_1D",
    "gauss_lim_2D",
    "step_1D",
    "step_2D",
]


def gauss(d, stdev, dim):
    """Gaussian function normalized in `dim` dimensions.

    Parameters
    ----------
    d : float or np.ndarray
        Distance (or array of distances) to the point of evaluation.
    stdev : float
        Standard deviation of the gaussian.
    dim : int
        Dimensionality of the normalization.
    """
    return np.exp(-(d**2) / (2 * stdev**2)) / (np.sqrt(2 * np.pi) * stdev) ** dim


def step_1D(d, R):
    """Normalized 1D step function: (d <= R) / R."""
    return (d <= R) / R


def gauss_1D(d, three_stdev):
    """Normalized gaussian 1D scale function, parameterized by 3 * stdev."""
    stdev = three_stdev / 3.0
    return gauss(d, stdev, 1)


def gauss_lim_1D(d, three_stdev):
    """Gaussian 1D function cut off after 3 standard deviations."""
    return gauss_1D(d, three_stdev) * (d < three_stdev)


def step_2D(d, R):
    """Normalized 2D step function: (d <= R) / (pi R^2)."""
    return (d <= R) / (np.pi * (R**2))


def gauss_2D(d, three_stdev):
    """Normalized gaussian 2D scale function, parameterized by 3 * stdev."""
    stdev = three_stdev / 3.0
    return gauss(d, stdev, 2)


def gauss_lim_2D(d, three_stdev):
    """Gaussian 2D function cut off after 3 standard deviations."""
    return (d <= three_stdev) * gauss_2D(d, three_stdev)


basis_1D = {
    "step": step_1D,
    "gauss": gauss_1D,
    "gauss_lim": gauss_lim_1D,
}

basis_2D = {
    "step": step_2D,
    "gauss": gauss_2D,
    "gauss_lim": gauss_lim_2D,
}
