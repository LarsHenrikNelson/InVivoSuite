from scipy import fft
import numpy as np


def xcorr(array_1, array_2):
    array_2 = np.ascontiguousarray(array_2[::-1])
    shape = array_1.size - array_2.size
    fshape = fft.next_fast_len(shape)
    sp1 = fft.fft(array_1, fshape)
    sp2 = fft.fft(array_2, fshape)

    ret = fft.ifft(sp1 * sp2, fshape)

    ret = ret[shape]
    return ret
