from math import log

import numpy as np
from numba import jit

"""
This code is from Antropy: https://github.com/raphaelvallat/antropy/tree/master
"""


@jit("uint32(uint32[:])", nopython=True)
def _lz_complexity(binary_string):
    """Internal Numba implementation of the Lempel-Ziv (LZ) complexity.
    https://github.com/Naereen/Lempel-Ziv_Complexity/blob/master/src/lziv_complexity.py
    - Updated with strict integer typing instead of strings
    - Slight restructuring based on Yacine Mahdid's notebook:
    https://github.com/BIAPT/Notebooks/blob/master/features/Lempel-Ziv%20Complexity.ipynb
    """
    # Initialize variables
    complexity = 1
    prefix_len = 1
    len_substring = 1
    max_len_substring = 1
    pointer = 0

    # Iterate until the entire string has not been parsed
    while prefix_len + len_substring <= len(binary_string):
        # Given a prefix length, find the largest substring
        if (
            binary_string[pointer + len_substring - 1]
            == binary_string[prefix_len + len_substring - 1]  # noqa: W503
        ):
            len_substring += 1  # increase the length of the substring
        else:
            max_len_substring = max(len_substring, max_len_substring)
            pointer += 1
            # Since all pointers have been scanned, pick largest as the jump
            # size
            if pointer == prefix_len:
                # Increment complexity
                complexity += 1
                # Set prefix length to the max substring size found so far
                # (jump size)
                prefix_len += max_len_substring
                # Reset pointer and max substring size
                pointer = 0
                max_len_substring = 1
            # Reset length of current substring
            len_substring = 1

    # Check if final iteration occurred in the middle of a substring
    if len_substring != 1:
        complexity += 1

    return complexity


def lziv_complexity(sequence, normalize=False):
    """
    Lempel-Ziv (LZ) complexity of (binary) sequence.

    .. versionadded:: 0.1.1

    Parameters
    ----------
    sequence : str or array
        A sequence of character, e.g. ``'1001111011000010'``,
        ``[0, 1, 0, 1, 1]``, or ``'Hello World!'``.
    normalize : bool
        If ``True``, returns the normalized LZ (see Notes).

    Returns
    -------
    lz : int or float
        LZ complexity, which corresponds to the number of different
        substrings encountered as the stream is viewed from the
        beginning to the end. If ``normalize=False``, the output is an
        integer (counts), otherwise the output is a float.

    Notes
    -----
    LZ complexity is defined as the number of different substrings encountered
    as the sequence is viewed from begining to the end.

    Although the raw LZ is an important complexity indicator, it is heavily
    influenced by sequence length (longer sequence will result in higher LZ).
    Zhang and colleagues (2009) have therefore proposed the normalized LZ,
    which is defined by

    .. math:: \\text{LZn} = \\frac{\\text{LZ}}{(n / \\log_b{n})}

    where :math:`n` is the length of the sequence and :math:`b` the number of
    unique characters in the sequence.

    References
    ----------
    * Lempel, A., & Ziv, J. (1976). On the Complexity of Finite Sequences.
      IEEE Transactions on Information Theory / Professional Technical
      Group on Information Theory, 22(1), 75–81.
      https://doi.org/10.1109/TIT.1976.1055501
    * Zhang, Y., Hao, J., Zhou, C., & Chang, K. (2009). Normalized
      Lempel-Ziv complexity and its application in bio-sequence analysis.
      Journal of Mathematical Chemistry, 46(4), 1203–1212.
      https://doi.org/10.1007/s10910-008-9512-2
    * https://en.wikipedia.org/wiki/Lempel-Ziv_complexity
    * https://github.com/Naereen/Lempel-Ziv_Complexity

    Examples
    --------
    >>> from antropy import lziv_complexity
    >>> # Substrings = 1 / 0 / 01 / 1110 / 1100 / 0010
    >>> s = '1001111011000010'
    >>> lziv_complexity(s)
    6

    Using a list of integer / boolean instead of a string

    >>> # 1 / 0 / 10
    >>> lziv_complexity([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    3

    With normalization

    >>> lziv_complexity(s, normalize=True)
    1.5

    This function also works with characters and words

    >>> s = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    >>> lziv_complexity(s), lziv_complexity(s, normalize=True)
    (26, 1.0)

    >>> s = 'HELLO WORLD! HELLO WORLD! HELLO WORLD! HELLO WORLD!'
    >>> lziv_complexity(s), lziv_complexity(s, normalize=True)
    (11, 0.38596001132145313)
    """
    assert isinstance(sequence, (str, list, np.ndarray))
    assert isinstance(normalize, bool)
    if isinstance(sequence, (list, np.ndarray)):
        sequence = np.asarray(sequence)
        if sequence.dtype.kind in "bfi":
            # Convert [True, False] or [1., 0.] to [1, 0]
            s = sequence.astype("uint32")
        else:
            # Treat as numpy array of strings
            # Map string characters to utf-8 integer representation
            s = np.fromiter(map(ord, "".join(sequence.astype(str))), dtype="uint32")
            # Can't preallocate length (by specifying count) due to string
            # concatenation
    else:
        s = np.fromiter(map(ord, sequence), dtype="uint32")

    if normalize:
        # 1) Timmermann et al. 2019
        # The sequence is randomly shuffled, and the normalized LZ
        # is calculated as the ratio of the LZ of the original sequence
        # divided by the LZ of the randomly shuffled LZ. However, the final
        # output is dependent on the random seed.
        # sl_shuffled = list(s)
        # rng = np.random.RandomState(None)
        # rng.shuffle(sl_shuffled)
        # s_shuffled = ''.join(sl_shuffled)
        # return _lz_complexity(s) / _lz_complexity(s_shuffled)
        # 2) Zhang et al. 2009
        n = len(s)
        base = sum(np.bincount(s) > 0)  # Number of unique characters
        base = 2 if base < 2 else base
        return _lz_complexity(s) / (n / log(n, base))
    else:
        return _lz_complexity(s)
