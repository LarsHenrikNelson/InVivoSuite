# %%
from pathlib import Path

import numpy as np
import pandas as pd

from invivosuite.acq import lfp, load_hdf5_acqs, SpikeModel, spike


# %%
p = list(Path(r"E:\Lars\in_vivo_ephys\ACC_kilosort").glob("*"))[:-2]

# %%
spk_models = []
for i in p:
    try:
        spk_models.append(SpikeModel(i))
    except Exception:
        print("no can do")

# %%
final_data = []
for model in spk_models:
    for clust_id in model.cluster_ids:
        indexes = model.get_cluster_spike_indexes(clust_id)
        data = {}
        if indexes.size > 1:
            data["hertz"] = 1 / np.mean(np.diff(indexes / 40000))
        data["num_spikes"] = indexes.size
        if indexes.size > 100:
            b_data = spike.max_int_bursts(
                indexes,
                40000,
                min_dur=0.01,
                max_start=0.170,
                max_int=0.3,
                max_end=0.34,
                output_type="time",
            )
            bursts_dict = spike.get_burst_data(b_data)
            stem = model.directory.stem.split("_")
            data["date"] = f"{stem[0]}_{stem[1]}_{stem[2]}"
            data["sex"] = stem[3]
            data["id"] = stem[4]
            data["genotype"] = stem[5]
            data.update(bursts_dict)
            final_data.append(data)
df = pd.DataFrame(final_data)

# %%
df.to_excel(
    (
        r"C:\Users\LarsNelson\OneDrive - University of Pittsburgh\exp_data"
        r"\Shank3B\Shank3B_in_vivo\acc_spike_data_prelim.xlsx"
    ),
    index=False,
)

# %%
# Use convenience function to load the files
parent_path = r"D:\in_vivo_ephys\acqs"
acqs = load_hdf5_acqs(parent_path)

# %%
burst_data = []
bands = {
    "theta": [4, 10],
    "low_gamma": [30, 50],
    "high_gamma": [70, 80],
    "beta": [12, 28],
}
for acq in acqs:
    print(acq.file_path)
    # stats = acq.lfp_burst_stats(bands)
    temp = acq.lfp_burst_stats_channel(0, bands)
    temp["file_path"] = acq.file_path
    burst_data.append(temp)

burst_df = pd.DataFrame(burst_data)
burst_df.to_excel(r"burst_data.xlsx")


# %%
def whitening_matrix(data, distances, um=200):
    nchans = data.shape[0]
    nt = data.shape[1]
    data = data.T
    M = np.mean(data, axis=0)
    data -= M
    cc = data.T @ data
    cc /= nt
    W = np.zeros((nchans, nchans))
    for i in range(nchans):
        inds = np.where(distances[0] < um)[0]
        print(inds)
        u, s, vh = np.linalg.svd(cc[inds, :][:, inds], hermitian=True)
        W_local = (u @ np.diag(1 / np.sqrt(s + 1e-8))) @ vh
        W[inds, i] = W_local[:, i]
    return W, M


# %%
# This is how to run current source density in elephant
# sig = AnalogSignal(temp[:, :10000].T, units="uV", sampling_rate=1000 * pq.Hz)
# coords = pq.Quantity(elec_map[["y"]].to_numpy() / 1000, pq.mm)
# est = current_source_density.estimate_csd(sig, coords, "KCSD1D")

# %%
freq_dict = {"theta": (4, 10), "gamma": (30, 80), "beta": (12, 30)}
tot_data = []
for key, acq in acqs.items():
    print(key)
    for i in range(120):
        print(f"Acq {i+1} out of 120")
        f, pxx = acqs[0].pxx(0, "welch")
        min_value = np.sum(pxx)
        th_w = lfp.get_ave_freq_window(pxx, f, lower_limit=4, upper_limit=10)
        th_b = lfp.get_ave_freq_window(pxx, f, lower_limit=12, upper_limit=30)
        th_g = lfp.get_ave_freq_window(pxx, f, lower_limit=30, upper_limit=80)
        # data["pdi_gamma"] = i.file["pdi"].attrs["gamma"]
        # data["pdi_theta"] = i.file["pdi"].attrs["theta"]
        # data["pdi_beta"] = i.file["pdi"].attrs["beta"]
        # data["date_rec"] = i.file.attrs["date_rec"]
        # data["genotype"] = i.file.attrs["genotype"]
        # data["acq_number"] = i.file.attrs["acq_number"]
        # data["sex"] = i.file.attrs["sex"]
        # data["id"] = i.file.attrs["id"]
        # data["theta"] = np.mean(i.file["theta"][()])
        # data["gamma"] = np.mean(i.file["gamma"][()])
        # data["beta"] = np.mean(i.file["beta"][()])
        # data["theta_welch"] = th_w - min_value
        # data["gamma_welch"] = th_g - min_value
        # data["beta_welch"] = th_b - min_value
        data = {}
        ave_len, iei, rms = i.lfp_burst_stats()
        data["lfp_burst_len"] = ave_len
        data["lfp_bursts_iei"] = iei
        data["lfp_burst_rms"] = rms
        tot_data.append(data)
        i.close()
df = pd.DataFrame(tot_data)
df.to_excel("D:/in_vivo_ephys/2023_05_15.xlsx")
# df.to_csv("/Volumes/Backup/in_vivo_ephys/2023_03_27.csv", index=False)
