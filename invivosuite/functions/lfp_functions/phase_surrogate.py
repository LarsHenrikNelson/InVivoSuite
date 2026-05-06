import numpy as np

from ...spectral import fft


def phase_surrogate(signal: np.ndarray, seed: int = 42):
    rng = np.random.default_rng(seed)
    n_times = len(signal)
    signal_fft = fft.r2c_rfft(signal)
    amplitudes = np.abs(signal_fft)

    random_phases = np.exp(1j * rng.uniform(0, 2 * np.pi, size=len(signal_fft)))
    random_phases[0] = 0
    if n_times % 2 == 0:
        random_phases[-1] = 0
    surrogate_fft = amplitudes * random_phases
    surrogate = fft.c2r_rifft(surrogate_fft)
    return surrogate
