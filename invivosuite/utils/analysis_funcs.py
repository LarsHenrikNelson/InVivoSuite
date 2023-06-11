import numpy as np

__all__ = ["compile_pairwise_data", "plexon64a", "comp_chan_dist"]


def sqr_indices(shape):
    p = np.full(shape, True)
    return tuple(
        np.broadcast_to(inds, p.shape)[p] for inds in np.indices(p.shape, sparse=True)
    )


def compile_pairwise_data(data, v_ind, h_ind, offset=1, ret_type="upper"):
    pw = {}
    for key, value in data.items():
        if ret_type == "upper":
            gd_index = np.triu_indices(v_ind[1] - v_ind[0], offset)
            pw[key] = value[v_ind[0] : v_ind[1], h_ind[0] : h_ind[1]][gd_index]
        else:
            gd_index = sqr_indices((v_ind[1] - v_ind[0], h_ind[1] - h_ind[0]))
            pw[key] = value[v_ind[0] : v_ind[1], h_ind[0] : h_ind[1]].flatten()
    return pw, gd_index


def plexon64a():
    r1 = np.arange(32) * 60
    r2 = r1 + 30
    r1 = [np.array([0, i]) for i in r1]
    r2 = [np.array([43.3, i]) for i in r2]
    p_contacts = r1 + r2
    p_contacts.sort(key=lambda i: i[1])
    return p_contacts


def comp_chan_dist(probe_map):
    distances = np.zeros((64, 64))
    for i in range(64):
        for j in range(64):
            z = probe_map[i] - probe_map[j]
            distances[i, j] = np.sqrt(z[0] ** 2 + z[1] ** 2)
    # distances += distances.T
    return distances
