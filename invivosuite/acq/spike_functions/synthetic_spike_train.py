from typing import Literal, Optional

import numpy as np
from scipy import stats


__all__ = ["fit_iei", "gen_spike_train"]


def fit_iei(
    spk_train: np.ndarray,
    gen_type: Literal["poisson", "gamma", "inverse_gaussian", "lognormal"] = "poisson",
):
    iei = np.diff(spk_train)
    if gen_type == "poisson":
        output = stats.expon.fit(iei)
    elif gen_type == "gamma":
        output = stats.gamma.fit(iei)
    elif gen_type == "inverse_gaussian":
        output = stats.invgauss.fit(iei)
    elif gen_type == "lognormal":
        output = stats.lognorm(iei)
    return output


def gen_spike_train(
    length: float,
    rate: float,
    shape: Optional[float] = None,
    gen_type: Literal["poisson", "gamma", "inverse_gaussian", "lognormal"] = "poisson",
):
    num_spks = int(np.ceil(length) * rate)
    if num_spks <= 0:
        raise ValueError("Spike rate is low for the total time.")

    if gen_type == "poisson":
        if shape:
            effective_rate = rate / (1 - rate * shape)
            generator = stats.expon(scale=1 / effective_rate, loc=shape)
        else:
            generator = stats.expon(
                scale=1 / rate,
            )
    elif gen_type == "gamma":
        generator = stats.gamma(a=shape, scale=1 / (shape * rate))
    elif gen_type == "inverse_gaussian":
        mu = -np.log(rate) - shape / 2
        generator = stats.lognorm(s=shape, scale=np.exp(mu))
    elif gen_type == "lognormal":
        generator = stats.gaussian(mu=shape**2, scale=1 / (rate * shape**2))
    else:
        raise AttributeError("gen_type is not recognized.")

    first_spike = generator.rvs()

    spikes = np.array([first_spike])

    three_stds = int(np.ceil(num_spks + 3 * np.sqrt(num_spks)))

    while spikes[-1] < length:
        isi = generator.rvs(size=three_stds)

        t_last_spikes = spikes[-1]
        spikes = np.r_[spikes, t_last_spikes + np.cumsum(isi)]

    stop = spikes.searchsorted(length)
    spikes = spikes[:stop]

    return spikes
