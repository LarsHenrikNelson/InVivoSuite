import os

import numpy as np

# from scipy import fft
import pyfftw
from numba import njit, prange

from .wavelet_utils import get_amp_scale
from .wavelets import fcwt_wavelet, mne_wavelet, scipy_wavelet


# Reimplemented fcwt in "pure" python as a learning resource


@njit(cache=True, parallel=True)
def daughter_wavelet_multiplication(
    input_fft: np.ndarray,
    output: np.ndarray,
    mother: np.ndarray,
    scale: float,
    threads: int = 1,
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
    endpointf = min(isizef / 2.0, (isizef * 2.0) / scale)
    step = scale / 2.0
    endpoint = int(endpointf)
    # endpoint4 = endpoint >> 2  # bit shifting finds max n for 2**n <= endpoint
    athreads = min(threads, max(1, endpoint / 16))
    batchsize = endpoint / athreads
    mm = isizef - 1
    s1 = isize - 1

    # get_amp_scale

    # output = np.zeros(input_fft.size, dtype=np.complex128)

    for q1 in prange(0, int(batchsize)):
        tmp = min(mm, step * q1)

        output[q1] = input_fft[q1].real * mother[int(tmp)] + (
            input_fft[q1].imag * mother[int(tmp)] * (1j - 2 * imaginary)
        )
    if doublesided:
        for q1 in prange(0, int(batchsize)):
            tmp = min(mm, step * q1)

            output[s1 - q1] = input_fft[s1 - q1].real * mother[int(tmp)] + input_fft[
                s1 - q1
            ].imag * mother[int(tmp)] * (1j - 2 * imaginary)
    # return output


@njit(cache=True, parallel=True)
def daughter_wavelet(
    input_fft: np.ndarray,
    mother: np.ndarray,
    scale: float,
    threads: int = 1,
    imaginary: bool = False,
    doublesided: bool = False,
):
    """Daughter wavelet that is derived from the mother wavelet.

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
    endpointf = min(
        isizef / 2.0, (isizef * 2.0) / scale
    )  # 2.0 might be mother wavelet scale
    step = scale / 2.0  # 2.0 might be mother wavelet scale
    endpoint = int(endpointf)
    # endpoint4 = endpoint >> 2  # bit shifting finds max n for 2**n <= endpoint
    athreads = min(threads, max(1, endpoint / 16))
    batchsize = endpoint / athreads
    mm = isizef - 1
    s1 = isize - 1

    output = np.zeros(input_fft.size, dtype=np.complex128)

    for q1 in prange(0, int(batchsize)):
        tmp = min(mm, step * q1)

        output[q1] = mother[int(tmp)] + (mother[int(tmp)] * (1j - 2 * imaginary))
    if doublesided:
        for q1 in prange(0, int(batchsize)):
            tmp = min(mm, step * q1)

            output[s1 - q1] = mother[int(tmp)] + mother[int(tmp)] * (1j - 2 * imaginary)
    return output


@njit(cache=True, parallel=True)
def fft_normalize(transform, size):
    """Simple FFT normalization that is parallelized
    by Numba.

    Parameters
    ----------
    transform :
        FFT transform
    size : _type_
        _description_
    """
    transform / size


class PyFCWT:
    def __init__(self, fs: float):
        self.fs = fs
        self.mother = None
        self.frequencies = None
        self.scales = None

    def mother_wavelet(self):
        pass

    def cwt(
        self,
        input_data: np.ndarray,
        threads: int = -1,
        norm: bool = True,
    ):
        if threads == -1:
            threads = os.cpu_count() // 2
        size = input_data.size
        newsize = 1 << int(np.ceil(np.log2(size)))

        # Ihat = fft.fft(input, n=newsize)

        # Only need for rfft or if using fftw
        a = pyfftw.empty_aligned(newsize, dtype="float64")
        b = pyfftw.empty_aligned(newsize // 2 + 1, dtype="complex128")
        Ihat = np.zeros(newsize, dtype=complex)
        a[:size] = input_data

        forward_fft = pyfftw.FFTW(a, b, threads=threads)
        forward_fft()

        Ihat[: newsize // 2 + 1] = b
        # for i in range(1, newsize >> 1):
        #     Ihat[newsize - i] = Ihat[i].real + Ihat[i].imag * -1j

        Ihat[newsize // 2 :] = np.conjugate(b[1:][::-1])

        # last_scale = True

        c = pyfftw.empty_aligned(newsize, dtype="complex128")
        d = pyfftw.empty_aligned(newsize, dtype="complex128")
        backward_fft = pyfftw.FFTW(c, d, direction="FFTW_BACKWARD", threads=threads)

        cwt = np.zeros((self.scales.size, size), dtype=np.complex128)
        for index, s in enumerate(self.scales):
            # if s == self.scales[-1]:
            #     last_scale = True
            daughter_wavelet_multiplication(Ihat, c, self.mother, s)
            backward_fft()
            cwt[index, :] = d[:size]

        # cwt = pyfftw.zeros_aligned((self.scales.size, size), dtype=np.complex128)
        # for index, s in enumerate(self.scales):
        #     # if s == self.scales[-1]:
        #     #     last_scale = True
        #     output = daughter_wavelet_multiplication(Ihat, mother, s)
        #     c[:] = output
        #     backward_fft()
        #     cwt[index] = d[:size]

        # cwt = fft.ifft(temp)[:, :size]

        if norm:
            fft_normalize(cwt, newsize)

        return cwt
