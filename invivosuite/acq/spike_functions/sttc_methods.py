from typing import Union


import numpy as np
from numba import njit

__all__ = ["sttc", "sttc_ele", "sttc_python"]


@njit()
def run_P(
    spk_times_1: np.ndarray, spk_times_2: np.ndarray, dt: Union[int, float]
) -> int:
    Nab = 0
    j = 0
    k = 0
    add_spk = 0
    for i in range(0, spk_times_1.size):
        while j < spk_times_2.size:
            # Need this for unsigned ints to work with numba
            if spk_times_1[i] < spk_times_2[j]:
                temp = spk_times_2[j] - spk_times_1[i]
                add_spk = 1
            else:
                temp = spk_times_1[i] - spk_times_2[j]
                add_spk = 0
            if np.abs(temp) <= dt:
                Nab = Nab + 1
                k += add_spk
                break
            elif spk_times_2[j] > spk_times_1[i]:
                break
            else:
                j += 1
    return Nab, k


@njit()
def run_T(
    spk_train: np.ndarray,
    dt: Union[float, int],
    start: Union[float, int],
    stop: Union[float, int],
) -> float:
    i = 0
    time_A = 2 * spk_train.size * dt
    if spk_train.size == 1:
        if spk_train[0] - start < dt:
            time_A = time_A - start + spk_train[0] - dt
        elif (spk_train[0] + dt) > stop:
            time_A = time_A - spk_train[0] - dt + stop
    else:
        for i in range(0, spk_train.size - 1):
            diff = spk_train[i + 1] - spk_train[i]
            if diff < (2 * dt):
                time_A = time_A - 2 * dt + diff
        if (spk_train[0] - start) < dt:
            time_A = time_A - start + spk_train[0] - dt
        if (stop - spk_train[-1]) < dt:
            time_A = time_A - spk_train[-1] - dt + stop
    return time_A


@njit()
def sttc(
    spk_times_1: np.ndarray,
    spk_times_2: np.ndarray,
    dt: Union[float, int],
    start: Union[float, int],
    stop: Union[float, int],
) -> tuple[float, int, int, int, int]:
    """This is a Numba accelerated version of spike timing tiling coefficient.
    It is faster than Elephants version by about 50 times. This adds up when
    there are 10000+ comparisons to make. This function can run using unsigned ints
    which is the most numerically precise.

    Args:
        spk_times_1 (np.ndarray): np.array of spike times
        spk_times_2 (np.ndarray): np.array of spike times
        dt (int, float): largest time difference at which two spikes can be considered
        to co-occur
        start (int, float): start of time to assess
        stop (int, float): stop of time to assess, usually the length of the recording

    Returns:
        float: spike time tiling coefficient
        int: number of spikes in spk_times_1 that occur within dt of spikes spk_times_2
        int: number of spikes in spk_times_1 that occur before spikes spk_times_2
        int: number of spikes in spk_times_2 that occur within dt of spikes spk_times_1
        int: number of spikes in spk_times_2 that occur before spikes spk_times_1
    """
    if spk_times_1.size == 0 or spk_times_2.size == 0:
        return 0.0, 0, 0, 0, 0
    else:
        dt = float(dt)
        t = stop - start
        tA = run_T(spk_times_1, dt, start, stop)
        tA = float(tA) / float(t)
        tB = run_T(spk_times_2, dt, start, stop)
        tB = float(tB) / float(t)
        pA_a, kA = run_P(spk_times_1, spk_times_2, dt)
        pA = pA_a / float(spk_times_1.size)
        pB_b, kB = run_P(spk_times_2, spk_times_1, dt)
        pB = pB_b / float(spk_times_2.size)
        # if (pA * tB) == 1 and (pB * tA) == 1:
        #     index = 1.0
        # elif (pA * tB) == 1:
        #     index = 0.5 + 0.5 * (pB - tA) / (1 - pB * tA)
        # elif (pB * tA) == 1:
        #     index = 0.5 + 0.5 * (pA - tB) / (1 - pA * tB)
        # else:
        #     index = 0.5 * (pA - tB) / (1 - tB * pA) + 0.5 * (pB - tA) / (1 - tA * pB)
        index = 0.5 * (pA - tB) / (1 - tB * pA) + 0.5 * (pB - tA) / (1 - tA * pB)
        return index, pA_a, kA, pB_b, kB


def run_p(
    spiketrain_j: np.ndarray,
    spiketrain_i: np.ndarray,
    dt: Union[int, float],
) -> float:
    # Create a boolean array where each element represents whether a spike
    # in spiketrain_j lies within +- dt of any spike in spiketrain_i.
    tiled_spikes_j = np.isclose(
        spiketrain_j[:, np.newaxis],
        spiketrain_i,
        atol=dt,
        rtol=0,
    )
    # Determine which spikes in spiketrain_j satisfy the time window
    # condition.
    tiled_spike_indices = np.any(tiled_spikes_j, axis=1)
    # Extract the spike times in spiketrain_j that satisfy the condition.
    tiled_spikes_j = spiketrain_j[tiled_spike_indices]
    # Calculate the ratio of matching spikes in j to the total spikes in j.
    return len(tiled_spikes_j) / len(spiketrain_j)


def run_t(
    spiketrain: np.ndarray,
    dt: Union[int, float],
    t_start: Union[int, float],
    t_stop: Union[int, float],
) -> float:
    dt = dt
    sorted_spikes = spiketrain

    diff_spikes = np.diff(sorted_spikes)

    overlap_durations = diff_spikes[diff_spikes <= 2 * dt]
    covered_time_overlap = np.sum(overlap_durations)

    non_overlap_durations = diff_spikes[diff_spikes > 2 * dt]
    covered_time_non_overlap = len(non_overlap_durations) * 2 * dt

    if sorted_spikes[0] - t_start < dt:
        covered_time_overlap += sorted_spikes[0] - t_start
    else:
        covered_time_non_overlap += dt
    if t_stop - sorted_spikes[-1] < dt:
        covered_time_overlap += t_stop - sorted_spikes[-1]
    else:
        covered_time_non_overlap += dt

    total_time_covered = covered_time_overlap + covered_time_non_overlap
    total_time = t_stop - t_start

    return total_time_covered / total_time


def sttc_ele(spiketrain_i, spiketrain_j, dt, start, stop):
    if len(spiketrain_i) == 0 or len(spiketrain_j) == 0:
        index = np.nan
    else:
        if spiketrain_i.dtype != float:
            spiketrain_i = spiketrain_i.astype(float)
        if spiketrain_j.dtype != float:
            spiketrain_j = spiketrain_j.astype(float)
        TA = run_t(spiketrain_j, dt, start, stop)
        TB = run_t(spiketrain_i, dt, start, stop)
        PA = run_p(spiketrain_j, spiketrain_i, dt)
        PB = run_p(spiketrain_i, spiketrain_j, dt)

        # check if the P and T values are 1 to avoid division by zero
        # This only happens for TA = PB = 1 and/or TB = PA = 1,
        # which leads to 0/0 in the calculation of the index.
        # In those cases, every spike in the train with P = 1
        # is within dt of a spike in the other train,
        # so we set the respective (partial) index to 1.
        if PA * TB == 1 and PB * TA == 1:
            index = 1.0
        elif PA * TB == 1:
            index = 0.5 + 0.5 * (PB - TA) / (1 - PB * TA)
        elif PB * TA == 1:
            index = 0.5 + 0.5 * (PA - TB) / (1 - PA * TB)
        else:
            index = 0.5 * (PA - TB) / (1 - PA * TB) + 0.5 * (PB - TA) / (1 - PB * TA)
    return index

@njit()
def run_T_python(spiketrain, N, dt, start, stop):
    """
    Calculate the proportion of the total recording time 'tiled' by spikes.
    """
    time_A = 2 * N * dt  # maxium possible time

    if N == 1:  # for just one spike in train
        if spiketrain[0] - start < dt:
            time_A = time_A - dt + spiketrain[0] - start
        elif spiketrain[0] + dt > stop:
            time_A = time_A - dt - spiketrain[0] + stop

    else:  # if more than one spike in train
        '''
            This part of code speeds up calculation with respect to the original version
        '''
        diff = np.diff(spiketrain)
        idx = np.where(diff<(2*dt))[0]
        Lidx = len(idx)
        time_A = time_A - 2 * Lidx * dt + diff[idx].sum()

        if (spiketrain[0] - start) < dt:
            time_A = time_A + spiketrain[0] - dt - start

        if (stop - spiketrain[N - 1]) < dt:
            time_A = time_A - spiketrain[-1] - dt + stop

    T = (time_A / (stop - start)) #.item()
    return T

@njit()
def run_P_python(spiketrain_1, spiketrain_2, N1, N2, dt):
    """
    Check every spike in train 1 to see if there's a spike in train 2
    within dt
    """
    Nab = 0
    j = 0
    for i in range(N1):
        L=0
        while j < N2:  # don't need to search all j each iteration
            if np.abs(spiketrain_1[i] - spiketrain_2[j]) <= dt:
                Nab = Nab + 1
                L+=1
                break
            elif spiketrain_2[j] > spiketrain_1[i]:
                break
            else:
                j = j + 1
    return Nab

def sttc_python(spiketrain_1,spiketrain_2,N1,N2,dt,start=0,stop=3e5):
    '''
    '''
    TA = run_T_python(spiketrain_1, N1, dt, start=start, stop=stop)
    TB = run_T_python(spiketrain_2, N2, dt, start=start, stop=stop)
    PA = run_P_python(spiketrain_1, spiketrain_2, N1, N2, dt)
    PA = PA / float(N1)
    PB = run_P_python(spiketrain_2, spiketrain_1, N2, N1, dt)
    PB = PB / float(N2)
    index = 0.5 * (PA - TB) / (1 - PA * TB) + 0.5 * (PB - TA) / (1 - PB * TA)
    return index

 
