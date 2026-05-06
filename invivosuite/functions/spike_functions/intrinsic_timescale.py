from typing import NamedTuple, TypeVar

import numpy as np
from scipy import optimize

from .sttc_methods import sttc


class ModelMetrics(NamedTuple):
    reduced_chisq: float = np.nan
    r2: float = np.nan
    residuals: np.ndarray = np.array([])
    parameter_uncertainties: np.ndarray = np.array([])


class SExpDecayFit(NamedTuple):
    amplitude: float = np.nan
    tau: float = np.nan
    offset: float = np.nan


class SExpDecay:
    _params: SExpDecayFit
    _model_metrics: ModelMetrics
    _fit_success: bool

    def __init__(self):
        self._params = self._create_nan_result()
        self._model_metrics = ModelMetrics()
        self._fit_success = False

    @property
    def params(self) -> SExpDecayFit:
        return self._params

    @staticmethod
    def _fit_function(
        x: np.ndarray, amplitude: float, tau: float, offset: float = 0.0
    ) -> np.ndarray:
        y = amplitude * np.exp(-x / tau) + offset
        return y

    def _get_bounds(self, x: np.ndarray, y: np.ndarray) -> tuple:
        upper_bounds = [y.max() * 2, x.max() * 5, np.inf]
        lower_bounds = [0.0, 0.0, -np.inf]
        return lower_bounds, upper_bounds

    def _get_initial_params(self, x: np.ndarray, y: np.ndarray) -> list:
        return [y.max(), x.max() / 5, 0.0]

    def _create_nan_result(self):
        return SExpDecayFit()

    def _create_result(self, popt: tuple):
        return SExpDecayFit(*popt)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self._params is None:
            raise ValueError("Must call fit() before predict()")
        return self._fit_function(x, *self._params)

    def _reduced_chisq(self, popt: tuple, x: np.ndarray, y: np.ndarray) -> float:
        y_fit = self.predict(x)
        resid = y - y_fit
        chi_sq = np.sum(resid**2)
        dof = len(y) - len(popt)
        return chi_sq / dof

    def _parameter_uncertainties(self, popt, pcov) -> np.ndarray:
        param_errors = np.sqrt(np.diag(pcov))
        relative_errors = param_errors / np.abs(popt) * 100
        return relative_errors

    def _r_squared(self, popt: tuple, x: np.ndarray, y: np.ndarray) -> float:
        y_fit = self.predict(x)
        ss_res = np.sum((y - y_fit) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        return r_squared

    def _residuals(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        y_fit = self.predict(x)
        return y - y_fit

    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs):
        y = np.asarray(y)
        x = np.asarray(x)
        try:
            p0 = self._get_initial_params(x, y)
            bounds = self._get_bounds(x, y)

            popt, pcov = optimize.curve_fit(
                self._fit_function, x, y, p0=p0, bounds=bounds, **kwargs
            )

            self._params = self._create_result(popt)
            chisq = self._reduced_chisq(popt, x, y)
            uncertain = self._parameter_uncertainties(popt, pcov)
            residuals = self._residuals(x, y)
            r2: int | float = self._r_squared(popt, x, y)
            self._model_metrics = ModelMetrics(chisq, r2, residuals, uncertain)

            self._fit_success = True

        except Exception:
            self._params = self._create_nan_result()
            self._fit_success = False

        return self._params


def _sttc_positive_lags(
    timestamps1: np.ndarray,
    timestamps2: np.ndarray,
    dt: float,
    nlags: int,
    max_lag: float,
    start: float,
    stop: float,
) -> tuple[np.ndarray, np.ndarray]:
    rec_len = stop - start
    lags = np.linspace(0, max_lag, endpoint=True, num=nlags)
    output = np.zeros(nlags)
    for index, lag in enumerate(lags):
        win_len = rec_len - lag
        if win_len <= 0:
            output[index] = np.nan
            continue

        # timestamps1 shifted by +lag: filter to [start+lag, stop]
        idx1_lo = np.searchsorted(timestamps1, start + lag, side="left")
        idx1_hi = np.searchsorted(timestamps1, stop, side="right")
        a = timestamps1[idx1_lo:idx1_hi] - lag - start

        # timestamps2: filter to [start, stop - lag]
        idx2_lo = np.searchsorted(timestamps2, start, side="left")
        idx2_hi = np.searchsorted(timestamps2, stop - lag, side="right")
        b = timestamps2[idx2_lo:idx2_hi] - start

        output[index] = sttc(a, b, dt=dt, start=0, stop=win_len)[0]
    return lags, output


def sttc_crosscorr(
    timestamps1: np.ndarray,
    timestamps2: np.ndarray,
    dt: float,
    nlags: int,
    max_lag: float,
    start: float,
    stop: float,
) -> tuple[np.ndarray, np.ndarray]:
    pos_lags, pos_vals = _sttc_positive_lags(
        timestamps1, timestamps2, dt, nlags, max_lag, start, stop
    )
    neg_lags, neg_vals = _sttc_positive_lags(
        timestamps2, timestamps1, dt, nlags, max_lag, start, stop
    )
    # Combine: negative (reversed, excluding 0) + positive
    lags = np.concatenate((-neg_lags[::-1][:-1], pos_lags))  # drop duplicate 0
    values = np.concatenate((neg_vals[::-1][:-1], pos_vals))
    return lags, values
