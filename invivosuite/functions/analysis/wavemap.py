import umap
import networkx as nx
import numpy as np

__all__ = ["find_best_resolution"]


def find_best_resolution(
    templates: np.ndarray,
    resolution_min: float = 0.3,
    resolution_max: float = 1.5,
    iterations: int = 20,
    n_neighbors: int = 20,
    min_dist: float = 0.0,
    subset: float = 0.15,
    n_jobs: int = 1,
) -> tuple[np.ndarray, dict, dict]:
    modularity_dict = {}
    n_clusts_dict = {}
    subsets = [int(subset * templates.shape[0])]
    resolution_list = np.linspace(resolution_min, resolution_max, iterations)

    for res in resolution_list:
        print(f"Resolution: {res}")
        for frac in subsets:
            rand_list = []
            n_clusts = []
            for _ in list(range(1, 12)):
                reducer_rand_test = umap.UMAP(
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    n_jobs=n_jobs,
                )
                rand_data = np.random.permutation(templates)[
                    0 : (int(len(templates) * frac)), :
                ]
                mapper = reducer_rand_test.fit(rand_data)

                G = nx.from_scipy_sparse_array(mapper.graph_)
                clustering = nx.community.louvain_communities(G, seed=0, resolution=res)
                modularity = nx.community.modularity(G, clustering)
                rand_list.append(modularity)
                n_clusts.append(len(clustering))
        modularity_dict[str(res)] = rand_list
        n_clusts_dict[str(res)] = n_clusts
    return resolution_list, modularity_dict, n_clusts_dict
