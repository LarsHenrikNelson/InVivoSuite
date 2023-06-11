# %%
import sys
from pathlib import Path

# sys.path.append(r"D:/working_python_files")
sys.path.append(r"/Volumes/Backup/working_python_files")

import h5py
import matplotlib.gridspec as grid_spec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import seaborn.objects as so
import statsmodels.api as sm
import statsmodels.formula.api as smf
from KDEpy import FFTKDE
from new_plot_funcs import plot_two_way
from seaborn import axes_style
from stats_functions import two_way_anova

from invivosuite import utils

# %%
# df = pd.read_excel("D:/in_vivo_ephys/final_data.xlsx")
# elec_map = pd.read_excel(
#     "C:/Users/LarsNelson/OneDrive - University of Pittsburgh/mapping.xlsx"
# )
df = pd.read_excel("/Volumes/Backup/in_vivo_ephys/final_data.xlsx")
elec_map = pd.read_excel(
    "/Users/larsnelson/OneDrive - University of Pittsburgh/mapping.xlsx"
)

# %%
df["genotype"].replace({"ko": "Shank3B-/-", "wt": "Shank3B+/+"}, inplace=True)
df["sex"] = df["sex"].str.capitalize()
df_acc = df[df["Plexon"] > 64].copy(deep=True)
df_dms = df[(df["Plexon"] < 65) & (df["Depth"] > 31)].copy(deep=True)
df_cortex = df[(df["Plexon"] < 65) & (df["Depth"] < 32)].copy(deep=True)


# %%
acc = df_acc.groupby(["genotype", "sex"])["lfp_burst_rms"].agg(
    ["count", "mean", "median", "std", "sem"]
)
dms = df_dms.groupby(["genotype", "sex"])["lfp_burst_rms"].agg(
    ["count", "mean", "median", "std", "sem"]
)

cortex = df_cortex.groupby(["genotype", "sex"])["lfp_burst_rms"].agg(
    ["count", "mean", "median", "std", "sem"]
)
print(acc)
print(dms)
print(cortex)

# %%
ds_acc, acc_aov, acc_posthoc = two_way_anova(
    df_acc, "genotype", "sex", "lfp_burst_len", "bonferroni"
)
ds_dms, dms_aov, dms_posthoc = two_way_anova(
    df_dms, "genotype", "sex", "lfp_burst_len", "bonferroni"
)
ds_cortex, cortex_aov, cortex_posthoc = two_way_anova(
    df_cortex, "genotype", "sex", "lfp_burst_len", "bonferroni"
)
print(acc_aov["PR(>F)"])
print(dms_aov["PR(>F)"])
print(cortex_aov["PR(>F)"])


# %%
save_path = "/Users/larsnelson/OneDrive - University of Pittsburgh/exp_data/Shank3B/Shank3B_in_vivo/Plots/acc/"
fig, ax = plot_two_way(
    df=df_acc,
    group="genotype",
    subgroup="sex",
    y="lfp_bursts_iei",
    order=[r"Shank3B+/+", r"Shank3B-/-"],
    hue_order=["Male", "Female"],
    y_label="Burst  IEI (sec)",
    title="",
    x_pval=0.8,
    color=None,  # {"Male": "darkorange", "Female": "slateblue"},
    alpha=0.5,
    y_lim=[None, None],
    y_scale="linear",
    steps=5,
    aspect=0.8 / 1,
    color_pval=0.05551,
    path=save_path,
    filetype="png",
)


# %%
plt.rcParams["axes.formatter.use_mathtext"] = True
# save_path = "C:/Users/LarsNelson/OneDrive - University of Pittsburgh/exp_data/Shank3B/Shank3B_in_vivo/Plots/acc"
save_path = "/Users/larsnelson/OneDrive - University of Pittsburgh/exp_data/Shank3B/Shank3B_in_vivo/Plots/acc"
fig, ax = plot_two_way2(
    df=df_acc,
    group="genotype",
    subgroup="sex",
    y="beta",
    order=[r"Shank3B+/+", r"Shank3B-/-"],
    hue_order=["Male", "Female"],
    y_label="Bursts len (sec)",
    title="",
    x_pval=0.8,
    color=None,  # {"Male": "darkorange", "Female": "slateblue"},
    alpha=0.5,
    y_lim=[None, None],
    y_scale="log",
    steps=5,
    aspect=0.8 / 1,
    color_pval=0.05551,
    # path=save_path,
    filetype="png",
)

# %%
pp = list(Path("D:/in_vivo_ephys").glob("**/*_icohere.hdf5"))
mscoh = {}
for i in pp:
    mscoh[i.stem.split("_p")[0]] = h5py.File(i, "r+")
fwt = ["FN_WT", "FWT", "FL_WT"]
mwt = ["ML5_WT", "ML1_WT", "ML", "MLL"]
fko = ["FL5_KO", "FKO"]
mko = ["ML5", "ML4", "ML4_KO", "MN_KO_P16"]
gts = [fwt, fko, mwt, mko]
gt = {
    "FN_WT": "Shank3B+/+",
    "FWT": "Shank3B+/+",
    "FL_WT": "Shank3B+/+",
    "ML5_WT": "Shank3B+/+",
    "ML1_WT": "Shank3B+/+",
    "ML": "Shank3B+/+",
    "MLL": "Shank3B+/+",
    "FL5_KO": "Shank3B-/-",
    "FKO": "Shank3B-/-",
    "ML5": "Shank3B-/-",
    "ML4": "Shank3B-/-",
    "ML4_KO": "Shank3B-/-",
    "MN_KO_P16": "Shank3B-/-",
}

sex = {
    "FN_WT": "Female",
    "FWT": "Female",
    "FL_WT": "Female",
    "ML5_WT": "Male",
    "ML1_WT": "Male",
    "ML": "Male",
    "MLL": "Male",
    "FL5_KO": "Female",
    "FKO": "Female",
    "ML5": "Male",
    "ML4": "Male",
    "ML4_KO": "Male",
    "MN_KO_P16": "Male",
}

dates = {}
for i in pp:
    dates[i.stem.split("_p")[0]] = str(i.parent).split("\\")[-1]

# %%
theta = {}
beta = {}
gamma = {}
for key, val in mscoh.items():
    theta[key] = val["theta"][()]
    gamma[key] = val["gamma"][()]
    beta[key] = val["beta"][()]

# %%
dat, ind = utils.compile_pairwise_data(
    beta, v_ind=(32, 64), h_ind=(64, 128), offset=1, ret_type="square"
)
dms_acc_df = (
    pd.DataFrame(data=dat)
    .stack()
    .reset_index()
    .sort_values(["level_1"])
    .drop(columns="level_0")
    .rename(columns={"level_1": "id", 0: "beta"})
)
dms_acc_df["gt"] = dms_acc_df["id"].replace(gt)
dms_acc_df["date"] = dms_acc_df["id"].replace(dates)
dms_acc_df["sex"] = dms_acc_df["id"].replace(sex)
dms_acc_df["group"] = dms_acc_df["gt"] + "_" + dms_acc_df["sex"]
dms_acc_df["id2"] = dms_acc_df["date"] + "_" + dms_acc_df["id"]

dat_g = utils.compile_pairwise_data(
    gamma, v_ind=(32, 64), h_ind=(64, 128), offset=1, ret_type="square"
)
gamma_dms_acc_df = (
    pd.DataFrame(data=dat_g)
    .stack()
    .reset_index()
    .sort_values(["level_1"])
    .drop(columns="level_0")
    .rename(columns={"level_1": "id", 0: "gamma"})
)
dat_t = utils.compile_pairwise_data(
    theta, v_ind=(32, 64), h_ind=(64, 128), offset=1, ret_type="square"
)
theta_dms_acc_df = (
    pd.DataFrame(data=dat_t)
    .stack()
    .reset_index()
    .sort_values(["level_1"])
    .drop(columns="level_0")
    .rename(columns={"level_1": "id", 0: "theta"})
)
dms_acc_df["theta"] = theta_dms_acc_df["theta"]
dms_acc_df["gamma"] = gamma_dms_acc_df["gamma"]

# %%
mdf = smf.mixedlm(
    "theta~sex*gt", data=dms_acc_df, groups=dms_acc_df["id2"], re_formula="~"
).fit()
mdf.summary()

# %%
ds_acc, acc_aov, acc_posthoc = two_way_anova(
    dms_acc_df, "gt", "sex", "theta", "bonferroni"
)

# %%
save_path = "C:/Users/LarsNelson/OneDrive - University of Pittsburgh/exp_data/Shank3B/Shank3B_in_vivo/Plots/coherence_maps/icohere/"
fig, ax = plot_two_way(
    df=dms_acc_df,
    group="gt",
    subgroup="sex",
    y="beta",
    order=[r"Shank3B+/+", r"Shank3B-/-"],
    hue_order=["Male", "Female"],
    y_label="Beta iCohere",
    title="",
    x_pval=0.8,
    color=None,  # {"Male": "darkorange", "Female": "slateblue"},
    alpha=0.05,
    y_lim=[10**-3, 10**0],
    y_scale="log",
    steps=5,
    decimals=3,
    aspect=0.8 / 1,
    color_pval=0.05551,
    path=save_path,
    filetype="png",
)


# %%
name = "acc_dms_icohere_beta"
plot_data = dat
labels = ["FWT", "FKO", "MWT", "MKO"]
gs = grid_spec.GridSpec(len(gts), 1)
fig = plt.figure(figsize=(12, 6))
ax_objs = []
tol = 0.1
for i in range(len(gts)):
    s = plot_data[gts[i][0]].size
    y = np.zeros(s * len(gts[i]))
    for ind, j in enumerate(gts[i]):
        ll = ind * s
        y[ll : ll + s] = plot_data[j]
    min_v = y.min()
    max_v = y.max()
    power2 = int(np.ceil(np.log2(y.size)))
    x = np.linspace(min_v - tol, max_v + tol, num=2**power2)
    kde_y = FFTKDE(kernel="biweight", bw="silverman").fit(y, weights=None).evaluate(x)
    y_min = np.zeros(2**power2)

    ax_objs.append(fig.add_subplot(gs[i : i + 1, 0:]))
    ax_objs[-1].plot(x, kde_y, lw=3, c="white")
    ax_objs[-1].fill_between(x, kde_y, y_min, alpha=0.5, color="black")
    ax_objs[-1].axvline(np.mean(y), color="magenta", alpha=0.5)
    ax_objs[-1].axvline(np.median(y), color="tomato", alpha=0.5)
    ax_objs[-1].axvline(x[np.argmax(kde_y)], color="green", alpha=0.5)
    ax_objs[-1].set_xlim(0, 0.04)
    rect = ax_objs[-1].patch
    rect.set_alpha(0)
    ax_objs[-1].set_yticklabels([])
    ax_objs[-1].set_ylabel(
        labels[i],
        rotation="horizontal",
        ha="right",
        y=0,
        fontsize=14,
        fontweight="bold",
    )
    # ax_objs[-1].yaxis.set_label_coords(0,0)
    ax_objs[-1].set_yticks([])
    if i != len(gts) - 1:
        ax_objs[-1].set_xticklabels([])

    spines = ["top", "right", "left", "bottom"]
    for s in spines:
        ax_objs[-1].spines[s].set_visible(False)
    # ax_objs[-1].text(0, 0, labels[i], fontweight="bold", fontsize=14, ha="right")
gs.update(hspace=-0.2)
# plt.savefig(
#     f"C:/Users/LarsNelson/OneDrive - University of Pittsburgh/exp_data/Shank3B/Shank3B_in_vivo/Plots/coherence_maps/{name}.png",
#     format="png",
#     bbox_inches="tight",
# )

# %%
