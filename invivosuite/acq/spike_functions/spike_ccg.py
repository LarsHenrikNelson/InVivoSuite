import numpy as np
from numba import njit


@njit()
def get_ccg(spikes1, spikes2, width=0.1, num_jitter=5, jitter_win=0.02):

    d = []
    djit = []  # Distance between any two spike times
    n_sp = len(spikes2)  # Number of spikes in the input spike train

    jitter = (
        np.random.random((num_jitter + 1, spikes1.size)) * (2 * jitter_win) - jitter_win
    )
    jitter[0] = np.zeros(spikes1.size)

    for jit in range(num_jitter):
        spikes1_j = spikes1 + jitter[jit]
        i, j = 0, 0
        for t in spikes1_j:
            # For each spike we only consider those spikes times that are at most
            # at a 'width' time lag. This requires finding the indices
            # associated with the limiting spikes.
            while i < n_sp and spikes2[i] < t - width:
                i += 1
            while j < n_sp and spikes2[j] < t + width:
                j += 1

            # Once the relevant spikes are found, add the time differences
            # to the list
            if jit == 0:
                d.extend(spikes2[i:j] - t)
            else:
                djit.extend(spikes2[i:j] - t)

    return d, djit


@njit()
def get_ccg_corr(s1, s2, width=1, bin_width=0.001):
    num_steps = np.int(width / bin_width)
    shifts = np.linspace(-num_steps, num_steps, 2 * num_steps + 1)

    corr = np.zeros(shifts.size)
    for i, shift in enumerate(shifts):
        #        corr[i] = np.dot(s1, np.roll(s2,np.int(shift)))
        corr[i] = (s1 * np.roll(s2, np.int(shift))).sum()

    return corr
