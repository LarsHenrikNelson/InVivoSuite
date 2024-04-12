import numpy as np
from numba import njit


__all__ = ["get_template_parts", "find_minmax_exponent_3"]


@njit()
def get_template_parts(template: np.ndarray) -> tuple[int, int, int]:
    index = np.argmin(template)
    start_index = index
    end_index = index
    current = template[index]
    previous = current - 1
    while previous < current and start_index > 0:
        previous = current
        start_index -= 1
        current = template[start_index]
    start_index += 1
    current = template[index]
    previous = current - 1
    while previous < current and end_index < template.size - 1:
        previous = current
        end_index += 1
        current = template[end_index]
    end_index -= 1
    amp = template[index]
    return start_index, end_index, index, amp


@njit(cache=True)
def find_minmax_exponent_3(spike_waveforms, min_val, max_val):
    # min_val = np.finfo("d").max
    # max_val = np.finfo("d").min
    for one in range(spike_waveforms.shape[0]):
        for two in range(spike_waveforms.shape[1]):
            for three in range(spike_waveforms.shape[2]):
                if spike_waveforms[one, two, three] != 0:
                    temp_value = np.log10(np.abs(spike_waveforms[one, two, three]))
                    if temp_value < min_val:
                        min_val = temp_value
                    if temp_value > max_val:
                        max_val = temp_value
    return np.floor(min_val), np.floor(max_val)
