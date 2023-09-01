# %%
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# import quantities as pq
# from elephant import current_source_density
import KDEpy

# from neo import AnalogSignal
from scipy import fft

#
from phylib.io.model import load_model

from invivosuite.acq import lfp, load_hdf5_acqs


# %%
# Use convenience function to load the files
parent_path = r"D:\in_vivo_ephys\acqs"
acqs = load_hdf5_acqs(parent_path)


# %%
acq = acqs[2].acq(0, "lfp")
bursts = acqs[2].get_lfp_burst_indexes(0)
baselines = acqs[2].get_burst_baseline(0)

# %%
bands={"theta": [4, 10], "low_gamma": [30, 50], "high_gamma": [70, 80], "beta": [12, 28]}
b_stats = acqs[-1].lfp_burst_stats(0, bands)

#%%
n_features = len(b_stats.p_dict)+5
n_samples = 122
output_dict = 
for i in b_stats:
    if isinstance(i, dict):
        for key, value in i.items():


# %%
f, cwt = acqs[-1].sxx(0, "cwt")
bands = lfp.get_cwt_bands(
    cwt,
    f,
    bands={"theta": [4, 10], "gamma": [30, 80], "beta": [12, 28]},
    ret_type="raw",
)
hil_gamma = acqs[-1].hilbert(0, highpass=50, lowpass=70, resample_freq=1000)
low_gamma = acqs[-1].hilbert(0, highpass=30, lowpass=50, resample_freq=1000)
high_gamma = acqs[-1].hilbert(0, highpass=70, lowpass=80, resample_freq=1000)
theta = acqs[-1].hilbert(0, highpass=4, lowpass=10, resample_freq=1000)

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
burst_psdm = []
for index, i in enumerate(bursts):
    f, p = lfp.multitaper(
        acq[int(i[0]) : int(i[1])],
        k=5,
        nw=3,
        fs=1000,
        nperseg=acq[int(i[0]) : int(i[1])].size,
        noverlap=0,
        nfft=2**13,
        ret_type="pxx",
    )
    burst_psdm.append(p)
    plt.plot(p, alpha=0.5, c="black")


# %%
save_path = "C:/Users/LarsNelson/OneDrive - University of Pittsburgh/exp_data/Shank3B/Shank3B_in_vivo/Plots/lfp_bursts/bursts_fft_log.svg"
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams["font.size"] = 20
fig, ax = plt.subplots()
ax.spines["left"].set_linewidth(2)
ax.spines["bottom"].set_linewidth(2)
ax.tick_params(width=2)
ax.semilogy(f, baseline_psd.mean(axis=0), c="black", linewidth=2, alpha=0.5)
ax.semilogy(f, burst_psdw.mean(axis=0), c="red", linewidth=2, alpha=0.5)
ax.set_xlim(0, 100)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ticks = ax.get_yticks()
ax.set_ylim(bottom=10**-13, top=10**-9)
plt.savefig(
    save_path,
    format="svg",
    bbox_inches="tight",
)


# %%
# This is how to run current source density in elephant
sig = AnalogSignal(temp[:, :10000].T, units="uV", sampling_rate=1000 * pq.Hz)
coords = pq.Quantity(elec_map[["y"]].to_numpy() / 1000, pq.mm)
est = current_source_density.estimate_csd(sig, coords, "KCSD1D")


# %%
g = np.zeros((128, 128))
t = np.zeros((128, 128))
b = np.zeros((128, 128))

for key, value in acqs.items():
    print(key)
    for i in range(len(value)):
        acq1 = acqs[key][acq_pos[i]].acq("lfp")
        for j in range(i, len(value)):
            print(i, j)
            acq2 = acqs[key][acq_pos[j]].acq("lfp")
            f1, cwt1 = lfp.create_cwt(
                acq1,
                fs=1000,
                f0=1,
                f1=200,
                steps=400,
                scaling="log",
                nthreads=-1,
            )
            f2, cwt2 = lfp.create_cwt(
                acq2,
                fs=1000,
                f0=1,
                f1=200,
                steps=400,
                scaling="log",
                nthreads=-1,
            )
            t[i, j] = np.mean(lfp.synchrony_cwt(cwt1, cwt2, f2, f0=4, f1=10))
            g[i, j] = np.mean(lfp.synchrony_cwt(cwt1, cwt2, f2, f0=30, f1=80))
            b[i, j] = np.mean(lfp.synchrony_cwt(cwt1, cwt2, f2, f0=12, f1=30))
    file = h5py.File(f"{mac_paths[key]}_phase_synchrony.hdf5", "a")
    file.create_dataset("gamma", data=g)
    file.create_dataset("theta", data=t)
    file.create_dataset("beta", data=b)
    file.close()


# %%
freq_dict = {"theta": (4, 10), "gamma": (30, 80), "beta": (12, 30)}
tot_data = []
for key, acq in acqs.items():
    print(key)
    for ind, i in enumerate(acq):
        print(f"Acq {ind+1} out of {len(acq)}")
        # f, pxx = signal.welch(i.acq("lfp"), fs=1000, nperseg=2048)
        # min_value = np.sum(pxx)
        # th_w = lfp.get_ave_freq_window(pxx, f, lower_limit=4, upper_limit=10)
        # th_b = lfp.get_ave_freq_window(pxx, f, lower_limit=12, upper_limit=30)
        # th_g = lfp.get_ave_freq_window(pxx, f, lower_limit=30, upper_limit=80)
        # data = i.acq_data()
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

# %%
elec_map = pd.read_excel(
    "C:/Users/LarsNelson/OneDrive - University of Pittsburgh/mapping.xlsx"
)
# elec_map = pd.read_excel(
#     "/Users/larsnelson/OneDrive - University of Pittsburgh/mapping.xlsx"
# )
elec_map.sort_values("depth", inplace=True)
sorted_acqs = elec_map["Acq"].to_numpy().flatten()
dms_acqs = elec_map[(elec_map["Depth"] > 32)]["Acq"].to_numpy()
cortex_acqs = elec_map[(elec_map["Depth"] <= 32)]["Acq"].to_numpy()


# %%
# Load model using phylib
model = load_model("E:/in_vivo_ephys/DMS_kilosort/FL5_KO_2023_02_17/params.py")

# %%
# Data for each spike
model.amplitudes
model.spike_times
model.spike_samples
model.spike_clusters
model.spike_templates

# %%
waveforms = model.get_cluster_spike_waveforms(cluster_id)
n_spikes, n_samples, n_channels_loc = waveforms.shape

# We get the channel ids where the waveforms are located.
channel_ids = model.get_cluster_channels(cluster_id)

# %%
# Shows the best channel for each cluster listed in the cluster_ids
model.clusters_channels

# Gets the best channel for the cluster
model.cluster_ids
model.clusters_channels
model.clusters_channels[model.cluster_ids[-1]]

# %%
acq = acqs["FKO"][64].acq("lfp")
spk = acqs["FKO"][64].acq("spike")

# %%
temps = np.where(model.clusters_channels == 4)[0]

# %%
# Extract the spikes
clus = np.where(model.spike_clusters == temps[-1])[0]
channel = model.clusters_channels[model.cluster_ids[5]]
indexes = model.spike_samples[clus]


# %%
spike_clusters = np.load(
    r"E:\in_vivo_ephys\DMS_kilosort\FL5_KO_2023_02_17\spike_clusters.npy"
)
spike_templates = np.load(
    r"E:\in_vivo_ephys\DMS_kilosort\FL5_KO_2023_02_17\spike_templates.npy"
)
spike_ids = np.where(spike_clusters == cluster_id)[0]
st = spike_templates[spike_ids]
template_ids, counts = np.unique(st, return_counts=True)
ind = np.argmax(counts)
template_id = template_ids[ind]
