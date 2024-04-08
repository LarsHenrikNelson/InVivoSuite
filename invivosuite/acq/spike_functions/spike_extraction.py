import numpy as np
from numba import njit


__all__ = ["_extract_waveforms_chunk", "find_minmax_exponent_3"]


@njit()
def _extract_waveforms_chunk(
    spike_times,
    spike_templates,
    sparse_templates,
    output,
    recording_chunk,
    nchans,
    start,
    end,
    waveform_length,
    peaks,
    channels,
    template_amplitudes,
):
    # Currently not working, but potential speed up with numba.
    current_spikes = np.where((spike_times < end) & (spike_times > start))[0]

    # Sort the spikes by amplitude
    extract_indexes = np.argsort(template_amplitudes[current_spikes])

    spk_chans = nchans * 2
    width = waveform_length / 2
    cutoff_end = end - width
    for i in extract_indexes:
        # Get the spike info in loop
        curr_spk_index = current_spikes[i]
        cur_spk_time = spike_times[curr_spk_index]
        if cur_spk_time < cutoff_end and (cur_spk_time - width) > start:
            cur_template_index = spike_templates[curr_spk_index]
            cur_template = sparse_templates[cur_template_index]
            chans = channels[cur_template_index, 1:3]
            peak_chan = channels[cur_template_index, 0]

            # Trying this to see if I can align spikes to templates a little bit better.
            tt = recording_chunk[
                int(cur_spk_time - start - width) : int(cur_spk_time - start + width),
                peak_chan,
            ]
            # Get min on best channel
            min_val = np.argmin(tt)

            # Adjust cur_spk_time forward or backward
            cur_spk_time = cur_spk_time + (min_val - cur_spk_time)

            output[curr_spk_index, :, :spk_chans] = recording_chunk[
                int(cur_spk_time - start - width) : int(cur_spk_time - start + width),
                chans[0] : chans[1],
            ]
            tj = peaks[cur_template_index, 0] / (tt[min_val] + 1e-10)
            recording_chunk[
                int(cur_spk_time - start - width) : int(cur_spk_time - start + width),
                chans[0] : chans[1],
            ] -= (
                cur_template[:, chans[0] : chans[1]] / tj
            )
        elif cur_spk_time < cutoff_end:
            cur_template_index = spike_templates[curr_spk_index]
            cur_template = sparse_templates[cur_template_index]
            chans = channels[cur_template_index, 1:3]
            peak_chan = channels[cur_template_index, 0]

            output[curr_spk_index, : int(cutoff_end - start), :spk_chans] = (
                recording_chunk[
                    int(cur_spk_time - start - width) : int(cutoff_end - start),
                    chans[0] : chans[1],
                ]
            )


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
