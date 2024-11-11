from typing import Literal

import numpy as np


def synchronous_periods(
    raster_continuous,
    raster_binary,
    cluster_ids,
    threshold: float,
    threshold_type: Literal["relative", "absolute"],
    min_length: float | int,
    channels: np.ndarray,
):
    if threshold_type == "relative":
        threshold = threshold * cluster_ids.size
    sdata = _find_synchronous_periods(
        threshold, raster=raster_continuous, min_length=min_length
    )
    group_data, cluster_data, groups, channels = _analyze_synchronous_periods(
        cluster_ids=cluster_ids,
        raster_continuous=raster_continuous,
        raster_binary=raster_binary,
        sdata=sdata,
        channels=channels,
    )
    return group_data, cluster_data, groups, channels


def _find_synchronous_periods(
    threshold: int | float, min_length: int | float, raster: np.ndarray
):
    summed_raster = raster.sum(axis=0)
    cutoff = np.where(summed_raster > threshold)[0]
    indices = np.where(np.diff(cutoff) > 1)[0]
    sdata = np.split(cutoff, indices + 1)
    sdata = [i for i in sdata if i.size > min_length]
    sdata = np.array([[i[0], i[-1]] for i in sdata])
    return sdata


def _analyze_synchronous_periods(
    cluster_ids: np.ndarray,
    raster_continuous: np.ndarray,
    raster_binary: np.ndarray,
    sdata: np.ndarray,
    channels: np.ndarray,
):
    groups = [
        np.apply_along_axis(np.any, 1, raster_continuous[:, i[0] : i[-1]])
        for i in sdata
    ]
    probs = [
        np.apply_along_axis(np.sum, 1, raster_continuous[:, i[0] : i[-1]])
        for i, g in zip(sdata, groups)
    ]
    spikes = [
        np.apply_along_axis(
            lambda x: np.sum(np.where(x == 0, 1, x)[0]),
            1,
            raster_binary[g, i[0] : i[-1]],
        )
        for i, g in zip(sdata, groups)
    ]

    group_dict = {}
    group_dict["unit_count"] = [i.sum() for i in groups]
    group_dict["length"] = [i[-1] - i[0] for i in sdata]
    group_dict["prob"] = [i.sum() for i in probs]
    group_dict["nspikes"] = [i.sum() for i in spikes]
    group_dict["total_units"] = cluster_ids.size

    cluster_dict = {}
    groups_set = [set(cluster_ids[i]) for i in groups]
    cluster_ids, cid_counts = np.unique(
        np.array([item for subset in groups_set for item in subset]), return_counts=True
    )
    cluster_dict["cluster_id"] = cluster_ids
    cluster_dict["counts"] = cid_counts
    cluster_dict["ngroups"] = len(groups)

    cluster_dict["prob"] = np.array(probs).sum(axis=0)

    channels = [channels[i] for i in groups]
    groups = [cluster_ids[i] for i in groups]

    return group_dict, cluster_dict, groups, channels
