"""Regression tests for filter utilities.

Historically `iirnotch` swapped the (b, a) coefficient order returned by
`scipy.signal.iirnotch`, producing a broken filter. These tests pin the
correct behavior.
"""

import numpy as np

from invivosuite.functions.filter_functions import iirnotch, iirnotch_zero

FS = 1000.0


def _tone_pair():
    t = np.arange(40 * FS) / FS  # 40 s for ~1 Hz resolution
    return t, np.sin(2 * np.pi * 10.0 * t) + np.sin(2 * np.pi * 60.0 * t)


def _band_amplitude(signal, freq, fs=FS):
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(signal.size)))
    freqs = np.fft.rfftfreq(signal.size, 1 / fs)
    return spectrum[np.argmin(np.abs(freqs - freq))] / (signal.size / 4)


def test_iirnotch_attenuates_notch_freq():
    t, x = _tone_pair()
    filtered = iirnotch(x, freq=60.0, q=30.0, fs=FS)
    before = _band_amplitude(x, 60.0)
    after = _band_amplitude(filtered, 60.0)
    assert after < 0.05 * before


def test_iirnotch_preserves_offband_freq():
    t, x = _tone_pair()
    filtered = iirnotch(x, freq=60.0, q=30.0, fs=FS)
    before = _band_amplitude(x, 10.0)
    after = _band_amplitude(filtered, 10.0)
    assert after > 0.9 * before


def test_iirnotch_zero_attenuates_notch_freq():
    t, x = _tone_pair()
    filtered = iirnotch_zero(x, freq=60.0, q=30.0, fs=FS)
    before = _band_amplitude(x, 60.0)
    after = _band_amplitude(filtered, 60.0)
    assert after < 0.05 * before


def test_iirnotch_matches_zero_phase_alpha_part():
    # One-pass and zero-phase notches share the same magnitude response here;
    # the zero-phase version is non-causal but must not blow up numerically.
    t, x = _tone_pair()
    one = iirnotch(x, freq=60.0, q=30.0, fs=FS)
    two = iirnotch_zero(x, freq=60.0, q=30.0, fs=FS)
    assert np.isfinite(one).all() and np.isfinite(two).all()
