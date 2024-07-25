from typing import Union, Optional

import networkx as nx
import numpy as np

Iterable = Union[np.ndarray, list]

__all__ = ["create_sttc_graph"]


def create_sttc_graph(
    cluster1_ids: np.ndarray,
    cluster2_ids: np.ndarray,
    sttc_values: np.ndarray,
    connections: np.ndarray,
    edge_args: Optional[dict[np.ndarray]] = None,
    node_args: Optional[dict[np.ndarray]] = None,
):
    sttc_graph = nx.Graph(n_units=cluster1_ids.size)
    for i in range(cluster1_ids.size):
        if connections[i]:
            sttc_graph.add_edge(
                cluster1_ids[i],
                cluster2_ids[i],
                weight=sttc_values[i],
            )
            if edge_args is not None:
                for key in edge_args.keys():
                    sttc_graph.edges[cluster1_ids[i]][cluster2_ids[i]][key] = edge_args[
                        key
                    ][i]
    if node_args is not None:
        cluster_ids = node_args["cluster_id"]
        for i in range(len(cluster_ids)):
            for key, value in node_args.items():
                sttc_graph.nodes[cluster_ids[i]][key] = value[i]
    return sttc_graph
