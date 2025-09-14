import os
from typing import Literal

import numpy as np
import pyfftw
from numba import njit, prange

from .wavelets import Wavelet
from .frequencies import Frequencies


@njit(cache=True, parallel=True)
def daughter_wavelet_multiplication(
    input_fft: np.ndarray,
    output: np.ndarray,
    mother: np.ndarray,
    scale: float,
    mscale: float,
    imaginary: bool = False,
    doublesided: bool = False,
):
    """FFT wavelet convolution from fCWT using numpy. That utilizes
    numpy pass-by assignment as pass-by references/pointer.

    Parameters
    ----------
    input_fft : np.ndarray
        FFT of the signal of interest, must be a 1D signal
    output : np.ndarray
        Pre allocated output array
    mother : np.ndarray
        Mother wavelet
    scale : float
        Scale to convolve input with
    threads : int, optional
        Number of threads to use, by default 1
    imaginary : bool, optional
        Whether the wavelet contains imaginary numbers, by default False
    doublesided : bool, optional
        Whether the wavelet is has a doubleside FFT (not in use), by default False
    """
    isize = input_fft.size
    isizef = float(isize)
    endpointf = min(isizef / mscale, (isizef * mscale) / scale)
    step = scale / mscale
    endpoint = int(endpointf)
    batchsize = endpoint
    maximum = isizef - 1.0
    multiplier = -1j if imaginary else 1j

    for q1 in prange(0, int(batchsize)):
        tmp = min(maximum, step * q1)

        output[q1] = input_fft[q1].real * mother[int(tmp)] + (
            input_fft[q1].imag * mother[int(tmp)] * multiplier
        )
    # if doublesided:
    #     for q1 in prange(0, int(batchsize)):
    #         tmp = min(mm, step * q1)

    #         output[s1 - q1] = input_fft[s1 - q1].real * mother[int(tmp)] + input_fft[
    #             s1 - q1
    #         ].imag * mother[int(tmp)] * (1j - 2 * imaginary)

class PyFCWT:
    def __init__(
        self,
        wavelet: Wavelet,
        frequencies: Frequencies,
        threads: int = -1,
        norm: bool = True,
        dtype: Literal["complex128, complex64"] = "complex128",
        zero_pad: bool = False,
    ):
        self.frequencies = frequencies
        self.wavelet = wavelet
        self.mother = None
        if threads == -1:
            self.threads = os.cpu_count() // 2
        else:
            self.threads = threads
        self.norm = norm
        if dtype == "complex128":
            self.fftw_fdtype = "float64"
            self.fftw_cdtype = "complex128"
            self.n_fdtype = np.float64
            self.n_cdtype = np.complex128
        else:
            self.fftw_fdtype = "float32"
            self.fftw_cdtype = "complex64"
            self.n_fdtype = np.float32
            self.n_cdtype = np.complex64
        self.zero_pad = zero_pad

    def cwt(
        self,
        input_data: np.ndarray,
    ):
        size = input_data.size
        newsize = pyfftw.next_fast_len(size)

        if self.mother is None:
            self.mother = self.wavelet.frequency(newsize)
        if self.mother.dtype != self.n_cdtype:
            self.mother.astype(self.n_cdtype)

        if input_data.dtype != self.n_fdtype:
            input_data = input_data.astype(self.n_fdtype)

        # Only need for rfft or if using fftw
        a = pyfftw.zeros_aligned(newsize, dtype=self.fftw_fdtype)
        b = pyfftw.empty_aligned(newsize // 2 + 1, dtype=self.fftw_cdtype)
        Ihat = np.zeros(newsize, dtype=complex)
        a[:size] = input_data

        forward_fft = pyfftw.FFTW(a, b, threads=self.threads)
        forward_fft()

        Ihat[: newsize // 2 + 1] = b

        Ihat[newsize // 2 :] = np.conjugate(b[1:][::-1])

        c = pyfftw.empty_aligned(newsize, dtype=self.fftw_cdtype)
        d = pyfftw.empty_aligned(newsize, dtype=self.fftw_cdtype)
        backward_fft = pyfftw.FFTW(
            c, d, direction="FFTW_BACKWARD", threads=self.threads
        )

        cwt = np.zeros((self.frequencies.scales.size, size), dtype=self.n_cdtype)
        for index, s in enumerate(self.frequencies.scales[::-1]):
            # if s == self.scales[-1]:
            #     last_scale = True
            daughter_wavelet_multiplication(
                input_fft=Ihat,
                output=c,
                mother=self.mother,
                scale=s,
                mscale=self.wavelet.scale,
                imaginary=self.wavelet.imaginary,
            )
            backward_fft()
            if not self.wavelet.imaginary:
                cwt[index, :] = d[:size]
            else:
                startind = ((self.wavelet.daughter_length(self.frequencies.freqs[::-1][index]) + size - 1) - size) // 2
                endind = startind + size

                cwt[index, :] = d[startind:endind]

        if self.norm:
            cwt /= newsize

        return cwt
