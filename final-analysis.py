# %%
"""
This analysis makes some assumptions about the layout of the data.
Each recording should have its own folder containing data exported from the spike
and spike-lfp analysis. The data is globbed and concatenated then joined (SQL-style outer join)
After initial cleaning data is 
"""

# %%
from datetime import datetime
from pathlib import Path
import functools as ft

import numpy as np
import pandas as pd

# Everything that is saved gets labeled with a date
current_date = str(datetime.today().date())


# %%
def get_dataframes(path, match_path):
    csv_paths = np.sort(list(path.glob(match_path)))
    csv_list = []
    for i in csv_paths:
        temp_df = pd.read_csv(i)
        temp_df["id"] = i.parent.stem
        # temp_df["uid"] = temp_df["id"] + "_" + temp_df["cluster_id"].astype(str)
        csv_list.append(temp_df)
    df = pd.concat(csv_list)
    assert df.shape[0] == sum(map(lambda x: x.shape[0], csv_list))
    return df


# %%
data_path = Path(r"path/to/data")
spk_props = get_dataframes(data_path, "**/spike-properties-*.csv")
burst_props = get_dataframes(data_path, "**/bursts-properties-*.csv")
phase_props = get_dataframes(data_path, "**/phase-properties-hilbert-*.csv")
burst_props.rename(
    columns={
        "local_sfa": "burst_local_sfa",
        "divisor_sfa": "burst_divisor_sfa",
        "abi_sfa": "burst_abi_sfa",
        "rlocal_sfa": "burst_rlocal_sfa",
    },
    inplace=True,
)
template_props = get_dataframes(data_path, "**/template-properties-*.csv")

df_merged = pd.merge(spk_props, burst_props, how="outer", on=["cluster_id", "id"])
df_merged = pd.merge(df_merged, template_props, how="outer", on=["cluster_id", "id"])
df_merged = pd.merge(df_merged, phase_props, how="outer", on=["cluster_id", "id"])
df = df_merged.sort_values(by=["id", "cluster_id"])

# %%
df["intra_burst_hz"] = (1 / df["intra_burst_iei"]).replace(np.inf, 0.0)
df["inter_burst_hz"] = (1 / df["inter_burst_iei"]).replace(np.inf, 0.0)
df["bursting_cell"] = np.where(df["num_bursts"] == 0, 0, 1)
df["ce_ratio"] = (
    df["center_y"] / df["end_y"]
)  # some labs use this for cell type identification
df["ce_time"] = (
    df["end"] - df["center"]
)  # This is a very useful metric, more useful than halfwidth
df["cs_time"] = df["center"] - df["start"]
df["me_time"] = (
    df["end"] - df["min_x"]
)  # This is a very useful metric, more useful than halfwidth adjusts for spikes that may not be centered around Kilosort spike time
df["me_ratio"] = df["min_y"] / df["end_y"]
df["ms_time"] = df["min_x"] - df["start"]
df["ms_ratio"] = df["min_y"] / df["start_y"]
df["amplitudeL"] = np.abs(df["min_y"] - df["end_y"])
df["amplitudeR"] = np.abs(df["min_y"] - df["start_y"])
df["violation_rate"] = df["num_violations"] / df["n_spikes"]

# %%
"""
The next couple sections below does some data cleaning primarily based on low spike count
and template properties. This mostly gets rid of units that contain noise.
"""

# %%
df.drop(df[df["min_x"] < 30].index, inplace=True)
df.drop(df[df["min_x"] > 50].index, inplace=True)
df.drop(df[df["burst_len"] > 20].index, inplace=True)
df.drop(df[df["n_spikes"] < 10].index, inplace=True)
df.drop(df[df["ms_time"] < 0].index, inplace=True)


# %%
def get_npy_templates(path, npy_path, df_path):
    npy_paths = sorted(list(path.glob(npy_path)))
    df_paths = sorted(list(path.glob(df_path)))
    npy_list = []
    pd_list = []
    for npyp, dfp in zip(npy_paths, df_paths):
        temp_npy = np.load(npyp)
        pd_data = pd.read_csv(dfp)
        npy_list.append(temp_npy)
        pd_list.append(pd_data)
    templates = np.concatenate(npy_list)
    pd_temp = pd.concat(pd_list)
    pd_temp.reset_index(drop=True, inplace=True)
    pd_temp["index"] = pd_temp.index.to_numpy()
    return templates, pd_temp


# %%
"""
Pandas shuffles data when doing SQL-style joins so it the template_df is used to identify the
index of each template that is associated with the specific spike cluster within a unique id
"""
npy_path = Path(r"path/to/data")
templates, template_df = get_npy_templates(
    npy_path, "**/spike-templates-*.npy", "**/template-ids-*.csv"
)
df = pd.merge(df, template_df, how="inner", on=["cluster_id", "id"])
templates = templates[df["index"].to_numpy().flatten(), :]


# %%
# Removes templates with noise, generally low spike number templates and very few (1% of templates)
temps_to_keep_r = [
    i for i in range(templates.shape[0]) if not np.any(templates[i, 70:] < -2e-5)
]
temps_to_keep_l = [
    i for i in range(templates.shape[0]) if not np.any(templates[i, :20] < -2e-5)
]
temps_to_keep = np.intersect1d(temps_to_keep_r, temps_to_keep_l)
templates = templates[temps_to_keep, :]
df = df.iloc[temps_to_keep]
df["index"] = np.arange(templates.shape[0])


# %%
"""
After the initial data cleaning and concatenation. I assign cell types to the waveforms.
Much of the literature uses spike frequency, template halfwidth (preferably full-width at 
half-maximum) and the trough to peak ratio. I do not think this is a good approach for several
reasons.
1. Frequency assumes that interneurons are continuously firing at a higher frequency than excitatory cells.
However during development PV cells cannot fire very quickly. This will likely only be true in the adult cortex.
Additionally, depending on whether you calculate firing rate from n-spikes/time or 1/iei you will can get very
different firing rate measures especially if a cell is bursty but mostly quiet.
2. Frequency is a non-linear/non-normal measure. Typically a frequency histogram of all the cells will
have a long tail.
3. Using frequency assumes you are not including multiunit activity and that you are finding almost every spike for every unit. 
4. Halfwidth leaves out a lot of information on how the waveform is centered around the center of the spike.
I have found the that the start and end of a spike is a very useful measure since waveform shape tends to remain
the same at different amplitudes.

I prefer to use the start and end of the spike waveform. These measures seem to be consistent across amplitudes and frequency
however, theses template measures are best calculated with asymmetric waveforms.
Interestingly, the end measure alone is usually enough to pick out short waveform cells that tend to have a higher spike
frequency (i.e. interneurons) and on average have a shorter trough (i.e. interneurons like PV cells).
I suggest a preliminary data exploration where you use KDE and histograms. For skewed data you can use the log10 transform
or the negative inverse (preserves direction compared to standard inverse) to get the data normally distributed.
For decomposition of multiple attributes I suggest using UMAP over PCA since UMAP preserves local clustering and can deal with non-linear data.
For clustering you can use HDBSCAN or for partitioning GaussianMixtureModel (seems to respect natural boundaries more that KMeans, but
assumes underlying groups have a gaussian distribution) or KMeans.
"""


# %%
"""
This section is use to clean the sttc data and average it per unit.
"""


# %%
def get_cleaned_sttc(sttc_df, output_df, column):
    sttc_df["pairs"] = sttc_df["uid1"] + "_" + sttc_df["uid2"]
    if isinstance(column, str):
        column = [column]
    first_merge = column + ["pairs", "id"]
    sttc_total = pd.merge(
        output_df[["cluster_id", "id"]],
        sttc_df[first_merge].assign(cluster_id=sttc_df["cluster1_id"]),
        how="inner",
        on=["cluster_id", "id"],
    ).rename(columns={"cluster_id": "cluster1_id"})
    sttc_total1 = pd.merge(
        output_df[["cluster_id", "id"]],
        sttc_df[first_merge].assign(cluster_id=sttc_df["cluster2_id"]),
        how="inner",
        on=["cluster_id", "id"],
    ).rename(columns={"cluster_id": "cluster2_id"})
    return pd.merge(sttc_total, sttc_total1, on=first_merge, how="inner")


def get_mean_per_cell(sttc_df, output_df, column):
    if isinstance(column, str):
        column = [column]
    df_list = []
    for i in column:
        temp1 = sttc_df[[i, "id", "cluster1_id"]].copy()
        temp2 = sttc_df[[i, "id", "cluster2_id"]].copy()
        temp1.rename(columns={"cluster1_id": "cluster_id"}, inplace=True)
        temp2.rename(columns={"cluster2_id": "cluster_id"}, inplace=True)
        sttc_vals = pd.concat([temp1, temp2])
        good_units = output_df[["cluster_id", "id"]]
        sttc_new = pd.merge(sttc_vals, good_units, how="inner", on=["cluster_id", "id"])
        df_list.append(sttc_new.groupby(["id", "cluster_id"]).mean().reset_index())
    sttc_per_cell_df = ft.reduce(
        lambda left, right: pd.merge(left, right, how="outer", on=["cluster_id", "id"]),
        df_list,
    )
    return sttc_per_cell_df


def assign_sttc_cell_types(sttc_df, df2):
    rep_dict = dict(
        zip(
            df2["id"] + "_" + df2["cluster_id"].astype(str),
            df2["cell_type"],
        )
    )
    sttc_df["cell_type1"] = sttc_df["uid1"].map(rep_dict)
    sttc_df["cell_type2"] = sttc_df["uid2"].map(rep_dict)


# %%
# Load and clean data. I typically run the STTC with multiple window sizes.
data_path = Path(r"path/to/data")
sttc_df_5ms = get_dataframes(data_path, "**/sttc-5ms-*.csv")
sttc_df_25ms = get_dataframes(data_path, "**/sttc-25ms-*.csv")

# STTC of 1.0 means spike times are identical
sttc_df_25ms.drop(sttc_df_25ms[sttc_df_25ms["sttc"] == 1.0].index, inplace=True)
sttc_df_5ms.drop(sttc_df_5ms[sttc_df_5ms["sttc"] == 1.0].index, inplace=True)

sttc_df_25ms.rename(columns={"sttc": "sttc_25ms"}, inplace=True)
sttc_df_5ms.rename(columns={"sttc": "sttc_5ms"}, inplace=True)

sttc_df = pd.merge(
    sttc_df_25ms[["sttc_25ms", "cluster1_id", "cluster2_id", "id"]],
    sttc_df_5ms[["sttc_5ms", "cluster1_id", "cluster2_id", "id"]],
    on=["cluster1_id", "cluster2_id", "id"],
    how="outer",
)

sttc_df["uid1"] = sttc_df["id"] + "_" + sttc_df["cluster1_id"].astype(str)
sttc_df["uid2"] = sttc_df["id"] + "_" + sttc_df["cluster2_id"].astype(str)

assign_sttc_cell_types(sttc_df, df)

# %%
# Final data cleaning based on some Allen Institute metrics and my own preferred metrics
# The fano factor is highly correlated with the violation rate so it could be used too.
df_cleaned = df.copy()
n_start = df_cleaned.shape[0]
df_cleaned.drop(df_cleaned[df_cleaned["violation_rate"] > 0.005].index, inplace=True)
df_cleaned.drop(df_cleaned[df_cleaned["amp_cutoff"] > 0.1].index, inplace=True)
df_cleaned.drop(df_cleaned[df_cleaned["presence_ratio"] < 0.8].index, inplace=True)
n_end = df_cleaned.shape[0]
print(f"Removed {n_start-n_end} of {n_start} units, left with {n_end}")


# %%
# It is important to calculate the STTC data after cleaning the bad spikes out.
# Save the data, always include the current date
sttc_cell_ave = get_mean_per_cell(sttc_df, df_cleaned, column=["sttc_25ms", "sttc_5ms"])
cleaned = get_cleaned_sttc(sttc_df, df_cleaned, column=["sttc_5ms", "sttc_25ms"])
final_df = pd.merge(
    df_cleaned,
    sttc_cell_ave[["sttc_25ms", "sttc_5ms", "cluster_id", "id"]],
    on=["cluster_id", "id"],
    how="outer",
)
df_output = pd.merge(
    df_cleaned,
    sttc_cell_ave[["sttc_25ms", "sttc_5ms", "cluster_id", "id"]],
    on=["cluster_id", "id"],
    how="outer",
)
cleaned.to_csv(
    rf"path/to/data\sttc-strict-cleaning-hilbert-{current_date}.csv",
    index=False,
)
df_output.to_csv(
    rf"path/to/data\acc-assigned-cell-type-strict-cleaning-hilbert-{current_date}.csv",
    index=False,
)
