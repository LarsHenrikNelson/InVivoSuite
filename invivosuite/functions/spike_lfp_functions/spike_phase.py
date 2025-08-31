from typing import TypedDict

import numpy as np

from ..circular_stats import (
    h_test,
    mean_vector_length,
    periodic_mean_std,
    rayleightest,
    ppc_dot_product,
)


class CircStats(TypedDict):
    rayleigh_pval: float
    circ_mean: float
    circ_std: float
    h: float
    m: float
    fpp: float
    vector_length: float
    vector_pval: float
    ppc: float

def analyze_spike_phase(phases: np.ndarray) -> CircStats:
    cm, stdev = periodic_mean_std(phases)
    h, m, fpp = h_test(phases)
    p = rayleightest(phases)
    vlen, vp = mean_vector_length(phases)
    ppc = ppc_dot_product(phases)
    
    stats = CircStats(
        rayleigh_pval=p,
        circ_mean=cm,
        circ_std=stdev,
        h=h,
        m=m,
        fpp=fpp,
        vector_length=vlen,
        vector_pval=vp,
        ppc=ppc
    )
    return stats

def extract_spike_phase_data(
    phase_dict: dict[str, np.ndarray],
    spike_times: np.ndarray,
) -> tuple[dict[str, np.ndarray]]:
    output_dict = {}
    output_stats = {}
    for b_name, phase in phase_dict.items():
        b_phases = phase[spike_times]
        output_dict[b_name] = b_phases
        stats = analyze_spike_phase(b_phases)
        output_stats.update(
            {f"{b_name}_{key}": value for key, value in stats.items()}
        )
    return output_stats, output_dict