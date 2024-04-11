import numpy as np


def create_binary_spikes(spikes, size):
    if len(spikes) > 0:
        binary_spikes = np.zeros(shape=(size,))
        binary_spikes[spikes] = 1
        return binary_spikes
    else:
        AttributeError("There are no spikes in the acquisition.")


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
    output = temp.sum(axis=1)
    return output


def bin_spikes(spikes, binary_size, nperseg):
    binary_spikes = create_binary_spikes(spikes, binary_size)
    binned_spikes = _bin_spikes(binary_spikes, nperseg)
    return binned_spikes
