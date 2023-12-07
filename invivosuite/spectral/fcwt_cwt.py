import numpy as np
from numba import njit, prange
from scipy import fft

# Reimplemented fcwt in "pure" python as a learning resource


@njit(cache=True, parallel=True)
def gen_wavelet(mu, size, scale=2.0):
    toradians = (2 * np.pi) / size
    norm = np.sqrt(2 * np.pi) * np.power(np.pi, -1 / 4)

    mother = np.zeros(size)
    for i in range(size):
        temp1 = (scale * i * toradians) * mu - 2.0 * np.pi * mu
        temp1 = -(temp1**2) / 2
        mother[i] = norm * np.exp(temp1)
    return mother


@njit(cache=True, parallel=True)
def daughter_wavelet_multiplication(
    input_fft: np.ndarray,
    output: np.ndarray,
    output_index: int,
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
    output_index : int
        Index for the current wavelet, input convolution
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

    # for q1 in prange(0, int(batchsize)):
    #     q = float(q1)
    #     tmp = min(mm, step * q)

    #     output[output_index, q1] = input_fft[q1].real * mother[int(tmp)] + (
    #         input_fft[q1].imag * mother[int(tmp)] * (1j - 2 * imaginary)
    #     )
    if not imaginary:
        output[output_index, 0:batchsize] = (
            input_fft[0:batchsize] * mother[0:batchsize:step]
        )
    else:
        output[output_index, 0:batchsize] = input_fft[0:batchsize] * np.conjugate(
            mother[0:batchsize:step]
        )
    if doublesided:
        for q1 in prange(0, int(batchsize)):
            q = float(q1)
            tmp = min(mm, step * q)

            output[output_index, s1 - q1] = input_fft[s1 - q1].real * mother[
                int(tmp)
            ] + input_fft[s1 - q1].imag * mother[int(tmp)] * (1j - 2 * imaginary)


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


def fcwt_cwt(
    input: np.ndarray,
    scales: np.ndarray,
    sigma,
    norm: bool = True,
):
    size = input.size
    newsize = 1 << int(np.ceil(np.log2(size)))

    Ihat = fft.fft(input, n=newsize)

    # Only need for rfft or if using fftw
    # Ihat = np.zeros(newsize, dtype=np.complex128)
    # temp = fft.fft(input, n=newsize)
    # Ihat[:temp.size] = temp
    # for i in range(1, newsize>>1):
    #     Ihat[newsize-i] = (Ihat[i].real, Ihat[i].imag)

    mother = gen_wavelet(sigma, newsize)

    temp = np.zeros((scales.size, newsize), dtype=np.complex128)
    # last_scale = True
    for index, s in enumerate(scales):
        # if s == scales[-1]:
        #     last_scale = True
        daughter_wavelet_multiplication(Ihat, temp, index, mother, s)
    transform = fft.ifft(temp)[:, :size]

    if norm:
        fft_normalize(transform, newsize)

    return transform
