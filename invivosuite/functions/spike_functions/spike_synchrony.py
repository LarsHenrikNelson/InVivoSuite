from typing import Literal

import numpy as np

from typing import TypedDict


class SyncData(TypedDict):
    group_data: dict[str, np.ndarray]
    cluster_data: dict[str, np.ndarray]
    groups: list[np.ndarray[int]]
    channels: np.ndarray
    sdata: np.ndarray[int]
    cluster_ids: np.ndarray[int]
    group_connectivity: dict[str, np.ndarray]


def synchronous_periods(
    raster_continuous,
    raster_binary,
    cluster_ids,
    threshold: float,
    threshold_type: Literal["relative", "absolute"],
    min_length: float | int,
    channels: np.ndarray,
) -> SyncData:
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

    conn_matrix = np.zeros((cluster_ids.size, cluster_ids.size))
    cid_index = {cid: value for value, cid in enumerate(cluster_ids)}
    for grp in groups:
        for j in range(grp.size - 1):
            for k in range(j + 1, grp.size):
                index1 = cid_index[grp[j]]
                index2 = cid_index[grp[k]]
                conn_matrix[index1, index2] += 1
                conn_matrix[index2, index1] += 1
    conn_matrix /= float(len(groups))
    indices = np.triu_indices(conn_matrix.shape[0], k=1)
    cluster1_id = np.array([cluster_ids[i] for i in indices[0]])
    cluster2_id = np.array([cluster_ids[i] for i in indices[1]])
    values = conn_matrix[indices]
    group_conn = {
        "cluster1_id": cluster1_id,
        "cluster2_id": cluster2_id,
        "connectivity_value": values,
    }

    output = SyncData(
        group_data=group_data,
        cluster_data=cluster_data,
        groups=groups,
        channels=channels,
        sdata=sdata,
        cluster_ids=cluster_ids,
        group_connectivity=group_conn,
    )

    return output


def _find_synchronous_periods(
    threshold: int | float, min_length: int | float, raster: np.ndarray
) -> np.ndarray:
    raster = raster.sum(axis=0)
    cutoff = np.where(raster > threshold)[0]
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
    grouped_cluster_ids = [cluster_ids[i] for i in groups]
    cluster_ids, cid_counts = np.unique(
        np.array([item for subset in grouped_cluster_ids for item in subset]), return_counts=True
    )
    cluster_dict["cluster_id"] = cluster_ids
    cluster_dict["counts"] = cid_counts
    cluster_dict["ngroups"] = len(grouped_cluster_ids)

    cluster_dict["prob"] = np.array(probs).sum(axis=0)

    channels = [channels[i] for i in groups]

    return group_dict, cluster_dict, grouped_cluster_ids, channels
