from typing import Literal
import os

import numpy as np
import pyfftw


__all__ = ["r2c_rfft", "r2c_fft", "c2c_fft", "ifft"]


def next_power_two(data: np.ndarray):
    return 1 << int(np.ceil(np.log2(data.size)))


def norm_fft(array: np.ndarray, norm: Literal["none", "sqrt", "n"] = "none"):
    if norm == "sqrt":
        array /= np.sqrt(array.size)
    elif norm == "n":
        array /= array.size


def r2c_rfft(
    data: np.ndarray,
    nfft: int = -1,
    norm: Literal["none", "sqrt", "n"] = "none",
    threads: int = -1,
):
    if nfft == -1:
        nfft = 1 << int(np.ceil(np.log2(data.size)))
    else:
        nfft = data.size
    if threads == -1:
        threads = os.cpu_count() // 2
    input_array = pyfftw.empty_aligned(nfft, dtype="float64")
    output_array = pyfftw.empty_aligned(nfft // 2 + 1, dtype="complex128")
    input_array[: data.size] = data
    forward_fft = pyfftw.FFTW(input_array, output_array, threads=threads)
    forward_fft()

    norm_fft(output_array, norm=norm)

    return output_array


def r2c_fft(
    data: np.ndarray,
    nfft: int = -1,
    norm: Literal["none", "sqrt", "n"] = "none",
    threads: int = -1,
):
    if nfft == -1:
        nfft = 1 << int(np.ceil(np.log2(data.size)))
    if threads == -1:
        threads = os.cpu_count() // 2
    input_array = pyfftw.empty_aligned(nfft, dtype="float64")
    temp = pyfftw.empty_aligned(nfft // 2 + 1, dtype="complex128")
    forward_fft = pyfftw.FFTW(input_array, temp, threads=threads)
    input_array[: data.size] = data
    forward_fft()
    output_array = np.zeros(nfft, dtype=complex)
    output_array[: nfft // 2 + 1] = temp

    # for i in range(1, nfft >> 1):
    #     output_array[nfft - i] = output_array[i].real + output_array[i].imag * -1j

    output_array[(nfft + 1) // 2 :] = np.conjugate(temp[1:][::-1])

    norm_fft(output_array, norm=norm)

    return output_array


def c2c_fft(
    data: np.ndarray,
    nfft: int = -1,
    norm: Literal["none", "sqrt", "n"] = "none",
    threads: int = -1,
):
    if nfft == -1:
        nfft = 1 << int(np.ceil(np.log2(data.size)))
    if threads == -1:
        threads = os.cpu_count() // 2
    input_array = pyfftw.empty_aligned(nfft, dtype="complex128")
    output_array = pyfftw.empty_aligned(nfft, dtype="complex128")
    forward_fft = pyfftw.FFTW(input_array, output_array, threads=threads)
    input_array[: data.size] = data
    forward_fft()

    norm_fft(output_array, norm=norm)

    return output_array


def ifft(data: np.ndarray, norm: Literal["none", "sqrt", "n"] = "none", threads=-1):
    norm_fft(data, norm=norm)
    if threads == -1:
        threads = os.cpu_count() // 2
    c = pyfftw.empty_aligned(data.size, dtype="complex128")
    d = pyfftw.empty_aligned(data.size, dtype="complex128")
    c[:] = data
    backward_fft = pyfftw.FFTW(c, d, direction="FFTW_BACKWARD", threads=threads)
    backward_fft()
    return d
