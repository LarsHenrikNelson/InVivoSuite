import numpy as np

__all__ = [
    "bin_spikes",
    "create_binary_spikes",
]


def create_binary_spikes(spikes, size):
    if len(spikes) > 0:
        u, uc = np.unique(spikes, return_counts=True)
        binary_spikes = np.zeros(shape=(size,), dtype=np.int16)
        binary_spikes[u] = uc
        return binary_spikes
    else:
        return np.zeros(shape=(size,), dtype=np.int16)


def _bin_spikes(binary_spks, bin_size):
    noverlap = 0
    nperseg = bin_size
    step = nperseg - noverlap
    shape = binary_spks.shape[:-1] + (
        (binary_spks.shape[-1] - noverlap) // step,
        nperseg,
    )
    strides = binary_spks.strides[:-1] + (
        step * binary_spks.strides[-1],
        binary_spks.strides[-1],
    )
    temp = np.lib.stride_tricks.as_strided(binary_spks, shape=shape, strides=strides)
    temp = temp.sum(axis=1)
    left_over = binary_spks.size % bin_size
    if left_over > 0:
        output = np.zeros(temp.size + 1)
        output[: temp.size] = temp
        start = binary_spks.size - left_over
        output[-1] = bin_spikes[start:].sum()
    else:
        output = temp
    return output


def bin_spikes(spikes, binary_size, nperseg):
    binary_spikes = create_binary_spikes(spikes, binary_size)
    binned_spikes = _bin_spikes(binary_spikes, nperseg)
    return binned_spikes
