import os

import numpy as np
from joblib import Parallel, delayed

from . import spike


def sttc_parallel(models, threads=-1):
    if threads == -1:
        threads = os.cpu_count() // 2
    out = Parallel(n_jobs=threads, prefer="threads")(
        delayed(sttc_per_model)(i) for i in models
    )
    return out


def sttc_per_model(model):
    data = {}
    stem = model.directory.stem.split("_")
    data["date"] = f"{stem[0]}_{stem[1]}_{stem[2]}"
    data["sex"] = stem[3]
    data["id"] = stem[4]
    data["genotype"] = stem[5]
    output_index = 0
    output = np.zeros(model.cluster_ids.size * (model.cluster_ids.size - 1) // 2)
    acq_index = np.zeros(
        (model.cluster_ids.size * (model.cluster_ids.size - 1) // 2, 4), dtype=int
    )
    for index1 in range(model.cluster_ids.size - 1):
        clust_id1 = model.cluster_ids[index1]
        indexes1 = model.get_cluster_spike_indexes(clust_id1)
        for index2 in range(index1 + 1, model.cluster_ids.size):
            clust_id2 = model.cluster_ids[index2]
            indexes2 = model.get_cluster_spike_indexes(clust_id2)
            m = spike.sttc(indexes1, indexes2, dt=200, start=0, stop=40000 * 600)
            output[output_index] = m
            acq_index[output_index, 0] = clust_id1
            acq_index[output_index, 1] = indexes1.size
            acq_index[output_index, 2] = clust_id2
            acq_index[output_index, 3] = indexes2.size
            output_index += 1
    data["sttc"] = output
    data["clust_id1"] = acq_index[:, 0]
    data["clust_id1_num_spks"] = acq_index[:, 1]
    data["clust_id2"] = acq_index[:, 2]
    data["clust_id2_num_spks"] = acq_index[:, 3]
    return data
