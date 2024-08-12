import umap
import networkx as nx
import numpy as np

import spike_functions as spkf

__all__ = ["best_resolution_bootstrap", "wavemap"]


def best_resolution_bootstrap(
    templates: np.ndarray,
    resolution_min: float = 0.3,
    resolution_max: float = 1.5,
    resolution_iterations: int = 20,
    n_neighbors: int = 20,
    min_dist: float = 0.0,
    subset: float = 0.15,
    subset_iterations: int = 10,
    n_jobs: int = 1,
) -> tuple[np.ndarray, dict, dict]:
    modularity_dict = {}
    n_clusts_dict = {}
    subsets = [int(subset * templates.shape[0])]
    resolution_list = np.linspace(resolution_min, resolution_max, resolution_iterations)

    for res in resolution_list:
        print(f"Resolution: {res}")
        for frac in subsets:
            rand_list = []
            n_clusts = []
            for _ in range(subset_iterations):
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


def wavemap(
    templates,
    n_neighbors=20,
    min_dist=0.0,
    n_components=2,
    random_state=0,
    n_jobs=1,
    resolution=1.1,
    rescale_templates=True,
    multiplier=1e6,
    center_templates=True,
):

    if rescale_templates:
        templates = spkf.rescale_templates(templates, multiplier=multiplier)

    if center_templates:
        templates = spkf.center_templates(templates=templates, center=41)
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    mapper = reducer.fit(templates)
    standard_embedding = mapper.transform(templates)

    G = nx.from_scipy_sparse_array(mapper.graph_)
    comms = nx.community.louvain_communities(
        G, seed=random_state, resolution=resolution
    )
    modularity = nx.community.modularity(G, comms)

    if rescale_templates or center_templates:
        return comms, modularity, standard_embedding, templates
    else:
        return comms, modularity, standard_embedding
