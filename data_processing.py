# %%
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from invivosuite.acq import lfp
from invivosuite.acq import load_hdf5_acqs, load_pl2_acqs, load_ad
from invivosuite.acq.pypl2 import pl2_ad

# %%
file_paths = list(Path("D:\in_vivo_ephys").rglob("*.hdf5"))
for file_path in file_paths:
    f = h5py.File(file_path, "r+")
    # add code to modify files here
    f.close()

# %%
for index, i in enumerate(acqs):
    print(index)
    i.analyze_bursts(
        method="max_int",
        min_count=5,
        min_dur=0,
        max_start=0.170,
        max_end=0.300,
        max_int=0.200,
    )

# %%
for index, i in enumerate(acqs):
    print(index)
    i.find_lfp_bursts(threshold=10)

# %%
for i in acqs:
    i.close()

# %%
pc_paths = {
    # "L4": "D:/in_vivo_ephys/2022_12_16/L4",
    # "R4": "D:/in_vivo_ephys/2022_12_14/R4",
    # "R5": "D:/in_vivo_ephys/2022_12_14/R5",
    # "L5": "D:/in_vivo_ephys/2022_12_14/L5",
    "FN_WT": "D:/in_vivo_ephys/2023_02_17/FN_WT",
    "FKO": "D:/in_vivo_ephys/2023_02_17/FL5_KO",
    "MWT": "D:/in_vivo_ephys/2023_02_17/ML5_WT",
    "ML4": "D:/in_vivo_ephys/2023_02_24/ML4",
    "ML5": "D:/in_vivo_ephys/2023_02_24/ML5",
    "ML1_WT": "D:/in_vivo_ephys/2023_03_03/ML1_WT",
    "ML4_KO": "D:/in_vivo_ephys/2023_03_03/ML4_KO",
    "MN_KO": "D:/in_vivo_ephys/2023_03_08/MN_KO_P16",
    "FL_WT": "D:/in_vivo_ephys/2023_03_09/FL",
    "ML_WT": "D:/in_vivo_ephys/2023_03_09/ML",
    "MLL_WT": "D:/in_vivo_ephys/2023_03_09/MLL",
    "FL5_KO": "D:/in_vivo_ephys/2023_03_17/FL5_KO",
    "FWT": "D:/in_vivo_ephys/2023_03_20/FWT",
}
mac_paths = {
    # "L4": "/Volumes/Backup/in_vivo_ephys/2022_12_16/L4",
    # "R4": "/Volumes/Backup/in_vivo_ephys/2022_12_14/R4",
    # "R5": "/Volumes/Backup/in_vivo_ephys/2022_12_14/R5",
    # "L5": "/Volumes/Backup/in_vivo_ephys/2022_12_14/L5",
    "FN_WT": "/Volumes/Backup/in_vivo_ephys/2023_02_17/FN_WT",
    "FKO": "/Volumes/Backup/in_vivo_ephys/2023_02_17/FL5_KO",
    "MWT": "/Volumes/Backup/in_vivo_ephys/2023_02_17/ML5_WT",
    "ML4": "/Volumes/Backup/in_vivo_ephys/2023_02_24/ML4",
    "ML5": "/Volumes/Backup/in_vivo_ephys/2023_02_24/ML5",
    "ML1_WT": "/Volumes/Backup/in_vivo_ephys/2023_03_03/ML1_WT",
    "ML4_KO": "/Volumes/Backup/in_vivo_ephys/2023_03_03/ML4_KO",
    "MN_KO": "/Volumes/Backup/in_vivo_ephys/2023_03_08/MN_KO_P16",
    "FL_WT": "/Volumes/Backup/in_vivo_ephys/2023_03_09/FL",
    "ML_WT": "/Volumes/Backup/in_vivo_ephys/2023_03_09/ML",
    "MLL_WT": "/Volumes/Backup/in_vivo_ephys/2023_03_09/MLL",
    "FL5_KO": "/Volumes/Backup/in_vivo_ephys/2023_03_17/FL5_KO",
    "FWT": "/Volumes/Backup/in_vivo_ephys/2023_03_20/FWT",
}
acqs = {}
for key, value in pc_paths.items():
    temp = load_hdf5_acqs(value)
    acqs[key] = temp

# %%
elec_map = pd.read_excel(
    "C:/Users/LarsNelson/OneDrive - University of Pittsburgh/mapping.xlsx"
)
# elec_map = pd.read_excel(
#     "/Users/larsnelson/OneDrive - University of Pittsburgh/mapping.xlsx"
# )
elec_map.sort_values("Depth", inplace=True)
acq_pos = elec_map["Acq"].to_numpy().flatten()
acq_pos = np.concatenate((acq_pos, acq_pos + 64))


# %%
g = np.zeros((128, 128))
t = np.zeros((128, 128))
b = np.zeros((128, 128))
for key, value in acqs.items():
    print(key)
    lfps = np.zeros((128, 600000))
    for p in range(128):
        lfps[p] = acqs[key][acq_pos[p]].acq("lfp")
        acqs[key][acq_pos[p]].close()
    for i in range(64, 128):
        for j in range(0, 64):
            print(i, j)
            cohy = lfp.coherence(lfps[i], lfps[j], ret_type="cohy")
            t[i, j] = lfp.phase_slope_index(cohy, f_band=[4, 10])
            g[i, j] = lfp.phase_slope_index(cohy, f_band=[30, 80])
            b[i, j] = lfp.phase_slope_index(cohy, f_band=[12, 30])
    file = h5py.File(f"{pc_paths[key]}_acc_to_dms_psi.hdf5", "a")
    file.create_dataset("gamma", data=g)
    file.create_dataset("theta", data=t)
    file.create_dataset("beta", data=b)
    file.close()

# %%
for key, value in acqs.items():
    print(key)
    for index, i in enumerate(value):
        print(index)
        i.find_lfp_bursts(method="spline", deg=90, threshold=10, wlen=200)
        i.close()

# %%
for key, value in acqs.items():
    print(key)
    t_acqs = np.zeros((128, 600000), dtype="complex128")
    for i in range(len(acq_pos)):
        t_acqs[i] = acqs[key][acq_pos[i]].get_hilbert(
            filter_type="butterworth_zero",
            order=4,
            highpass=30,
            lowpass=80,
            resample_freq=1000,
        )
    t = np.zeros((128, 128))
    for i in range(0, 128):
        for j in range(i, 128):
            t[i, j] = np.mean(lfp.phase_synchrony(t_acqs[i], t_acqs[j]))
    file = h5py.File(f"{pc_paths[key]}_h_phase_sync.hdf5", "r+")
    del file["gamma"]
    file.create_dataset("gamma", data=t)
    file.close()

# %%
pdi = {}
p_syn = {}
for key, acq in acqs.items():
    # Create the average cwts
    p_syn = {}
    acq1 = np.zeros(600000)
    acq2 = np.zeros(600000)
    acq3 = np.zeros(600000)
    for i in acq[64:]:
        t = i.acq("lfp")
        i.close()
        acq1 += t
    acq1 /= 64
    f, cwt1 = lfp.create_cwt(
        acq1, 1000, start_freq=1, stop_freq=110, steps=400, nthreads=10, scaling="log"
    )

    for j in dms_acqs:
        t = acq[j].acq("lfp")
        acq[j].close()
        acq2 += t
    acq2 /= 32
    f, cwt2 = lfp.create_cwt(
        acq2, 1000, start_freq=1, stop_freq=110, steps=400, nthreads=10, scaling="log"
    )

    for m in cortex_acqs:
        t = acq[m].acq("lfp")
        acq[m].close()
        acq3 += t
    acq3 /= 32
    f, cwt3 = lfp.create_cwt(
        acq3, 1000, start_freq=1, stop_freq=110, steps=400, nthreads=10, scaling="log"
    )

    # Analyze the synchrony
    p_syn["acc_dms_theta"] = lfp.synchrony(
        cwt1, cwt2, method="cwt", freqs=f, f0=4, f1=10
    )
    p_syn["acc_cortex_theta"] = lfp.synchrony(cwt1, cwt3, freqs=f, f0=4, f1=10)
    p_syn["cortex_dms_theta"] = lfp.synchrony(cwt3, cwt2, freqs=f, f0=4, f1=10)
    p_syn["acc_dms_gamma"] = lfp.synchrony(cwt1, cwt2, freqs=f, f0=30, f1=80)
    p_syn["acc_cortex_gamma"] = lfp.synchrony(cwt1, cwt3, freqs=f, f0=30, f1=80)
    p_syn["cortex_dms_gamma"] = lfp.synchrony(cwt3, cwt2, freqs=f, f0=30, f1=80)
    p_syn["acc_dms_beta"] = lfp.synchrony(cwt1, cwt2, freqs=f, f0=12, f1=30)
    p_syn["acc_cortex_beta"] = lfp.synchrony(cwt1, cwt3, freqs=f, f0=12, f1=30)
    p_syn["cortex_dms_beta"] = lfp.synchrony(cwt3, cwt2, freqs=f, f0=12, f1=30)

    # Analyze the phase discontinuity
    acc_pdi = {}
    acc_coh = lfp.binned_coh(cwt1, 1000)
    acc_pdi["theta"] = lfp.phase_discontinuity_index(acc_coh, f, 4, 10)
    acc_pdi["beta"] = lfp.phase_discontinuity_index(acc_coh, f, 12, 30)
    acc_pdi["gamma"] = lfp.phase_discontinuity_index(acc_coh, f, 30, 80)
    dms_pdi = {}
    dms_coh = lfp.binned_coh(cwt2, 1000)
    dms_pdi["theta"] = lfp.phase_discontinuity_index(dms_coh, f, 4, 10)
    dms_pdi["beta"] = lfp.phase_discontinuity_index(dms_coh, f, 12, 30)
    dms_pdi["gamma"] = lfp.phase_discontinuity_index(dms_coh, f, 30, 80)
    coxtex_pdi = {}
    coxtex_coh = lfp.binned_coh(cwt3, 1000)
    coxtex_pdi["theta"] = lfp.phase_discontinuity_index(coxtex_coh, f, 4, 10)
    coxtex_pdi["beta"] = lfp.phase_discontinuity_index(coxtex_coh, f, 12, 30)
    coxtex_pdi["gamma"] = lfp.phase_discontinuity_index(coxtex_coh, f, 30, 80)
    pdi["acc"] = acc_pdi
    pdi["dms"] = dms_pdi
    pdi["cortex"] = coxtex_pdi

    # Save data
    # path = f"/Volumes/Backup/in_vivo_ephys/bulk_data/{key}.hdf5"
    path = f"C:/Users/LarsNelson/OneDrive - University of Pittsburgh/exp_data/Shank3B/ \
        Shank3B_in_vivo/Recordings/bulk_data/{key}.hdf5"
    file = h5py.File(path, "a")
    file.create_dataset("acc_cwt", data=cwt1)
    file.create_dataset("dms_cwt", data=cwt2)
    file.create_dataset("cortex_cwt", data=cwt3)
    grp = file.create_group("phase_syn")
    for k, i in p_syn.items():
        grp.create_dataset(k, data=i)
    for region, d in pdi.items():
        grp = file.create_group(region)
        for band, f in d.items():
            grp.create_dataset(band, data=f)
    file.close()


# %%
for i in fls:
    if not i.get("coh"):
        grp = i.create_group("coh")
    else:
        grp = i["coh"]
    acc = i["acc_cwt"][()]
    dms = i["dms_cwt"][()]
    cortex = i["cortex_cwt"][()]
    coh = lfp.cwt_coh(acc, dms)
    acc_dms_theta = lfp.get_ave_freq_window(coh, f, 4, 10)
    acc_dms_beta = lfp.get_ave_freq_window(coh, f, 12, 30)
    acc_dms_gamma = lfp.get_ave_freq_window(coh, f, 30, 80)
    if grp.get("acc_dms_theta"):
        del grp["acc_dms_theta"]
        grp.create_dataset("acc_dms_theta", data=acc_dms_theta)
    else:
        grp.create_dataset(
            "acc_dms_theta",
            shape=acc_dms_theta.shape,
            maxshape=acc_dms_theta.shape,
        )
        grp["acc_dms_theta"][...] = np.abs(acc_dms_theta)
    if grp.get("acc_dms_beta"):
        del grp["acc_dms_beta"]
        grp.create_dataset("acc_dms_beta", data=np.abs(acc_dms_theta))
    else:
        grp.create_dataset(
            "acc_dms_beta",
            shape=acc_dms_beta.shape,
            maxshape=acc_dms_beta.shape,
        )
        grp["acc_dms_beta"][...] = np.abs(acc_dms_beta)
    if grp.get("acc_dms_gamma"):
        del grp["acc_dms_gamma"]
        grp.create_dataset("acc_dms_gamma", data=acc_dms_gamma)
    else:
        grp.create_dataset(
            "acc_dms_gamma",
            shape=acc_dms_gamma.shape,
            maxshape=acc_dms_gamma.shape,
        )
        grp["acc_dms_gamma"][...] = np.abs(acc_dms_gamma)
    coh = lfp.cwt_coh(acc, cortex)
    acc_cortex_theta = lfp.get_ave_freq_window(coh, f, 4, 10)
    acc_cortex_beta = lfp.get_ave_freq_window(coh, f, 12, 30)
    acc_cortex_gamma = lfp.get_ave_freq_window(coh, f, 30, 80)
    if grp.get("acc_cortex_theta"):
        del grp["acc_cortex_theta"]
        grp.create_dataset("acc_cortex_theta", data=acc_cortex_theta)
    else:
        grp.create_dataset(
            "acc_cortex_theta",
            shape=acc_cortex_theta.shape,
            maxshape=acc_cortex_theta.shape,
        )
        grp["acc_cortex_theta"][...] = np.abs(acc_cortex_theta)
    if grp.get("acc_cortex_beta"):
        del grp["acc_cortex_beta"]
        grp.create_dataset("acc_cortex_beta", data=acc_cortex_beta)
    else:
        grp.create_dataset(
            "acc_cortex_beta",
            shape=acc_cortex_beta.shape,
            maxshape=acc_cortex_beta.shape,
        )
        grp["acc_cortex_beta"][...] = np.abs(acc_cortex_beta)
    if grp.get("acc_cortex_gamma"):
        del grp["acc_cortex_gamma"]
        grp.create_dataset("acc_cortex_gamma", data=acc_cortex_gamma)
    else:
        grp.create_dataset(
            "acc_cortex_gamma",
            shape=acc_cortex_gamma.shape,
            maxshape=acc_cortex_gamma.shape,
        )
        grp["acc_cortex_gamma"][...] = np.abs(acc_cortex_gamma)
    coh = lfp.cwt_coh(cortex, dms)
    cortex_dms_theta = lfp.get_ave_freq_window(coh, f, 4, 10)
    cortex_dms_beta = lfp.get_ave_freq_window(coh, f, 12, 30)
    cortex_dms_gamma = lfp.get_ave_freq_window(coh, f, 30, 80)
    if grp.get("cortex_dms_theta"):
        del grp["cortex_dms_theta"]
        grp.create_dataset("cortex_dms_theta", data=cortex_dms_theta)
    else:
        grp.create_dataset(
            "cortex_dms_theta",
            shape=cortex_dms_theta.shape,
            maxshape=cortex_dms_theta.shape,
        )
        grp["cortex_dms_theta"][...] = np.abs(cortex_dms_theta)
    if grp.get("cortex_dms_beta"):
        del grp["cortex_dms_beta"]
        grp.create_dataset("cortex_dms_beta", data=cortex_dms_beta)
    else:
        grp.create_dataset(
            "cortex_dms_beta",
            shape=cortex_dms_beta.shape,
            maxshape=cortex_dms_beta.shape,
        )
        grp["cortex_dms_beta"][...] = np.abs(cortex_dms_beta)
    if grp.get("cortex_dms_gamma"):
        del grp["cortex_dms_gamma"]
        grp.create_dataset("cortex_dms_gamma", data=cortex_dms_gamma)
    else:
        grp.create_dataset(
            "cortex_dms_gamma",
            shape=cortex_dms_gamma.shape,
            maxshape=cortex_dms_gamma.shape,
        )
        grp["cortex_dms_gamma"][...] = np.abs(cortex_dms_gamma)
    i.close()
