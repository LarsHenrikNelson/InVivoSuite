import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "plot_bursts_baseline",
    "plot_clusters",
    "plot_lfp_bursts_with_freq",
    "plot_spks",
]


def plot_clusters(embedding, components, colors, alpha=0.7, marker="."):
    fig, ax = plt.subplots()
    n_colors = np.unique(colors)
    for i in n_colors:
        temp_color = np.where(i == colors)[0]
        ax.scatter(
            embedding[temp_color, components[0]],
            embedding[temp_color, components[1]],
            marker=marker,
            alpha=alpha,
        )


def plot_spks(spks):
    fig, ax = plt.subplots()
    for i in range(spks.shape[0]):
        ax.plot(spks[i], c="black", linewidth=1, alpha=0.6)


def plot_lfp_bursts_with_freq(acq, length, save_path):
    a = acq.acq("lfp")
    bursts = np.asarray(acq.get_lfp_burst_indexes(), dtype=np.int64)
    f, pxx = acq.create_pxx("cwt")
    theta = np.where(np.logical_and(f >= 4, f < 10))[0]
    beta = np.where(np.logical_and(f >= 12, f < 30))[0]
    gamma = np.where(np.logical_and(f >= 30, f < 80))[0]
    tpxx = np.abs(pxx[theta].mean(axis=0))
    bpxx = np.abs(pxx[beta].mean(axis=0))
    gpxx = np.abs(pxx[gamma].mean(axis=0))
    ind = np.where(bursts[:, 1] < length)[0]
    plt.rcParams["xtick.labelsize"] = 20
    plt.rcParams["ytick.labelsize"] = 20
    plt.rcParams["font.size"] = 20
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(20, 10))
    ax[0, 0].plot(a[:length], c="black", alpha=0.8, linewidth=0.3)
    ax[0, 1].plot(tpxx[:length], c="black", alpha=0.8, linewidth=0.3)
    ax[1, 0].plot(bpxx[:length], c="black", alpha=0.8, linewidth=0.3)
    ax[1, 1].plot(gpxx[:length], c="black", alpha=0.8, linewidth=0.3)
    ax[0, 0].set_title("LFP")
    ax[0, 1].set_title("Theta")
    ax[1, 0].set_title("Beta")
    ax[1, 1].set_title("Gamma")
    for i in ind:
        x = np.arange(bursts[i][0], bursts[i][1])
        ax[0, 0].plot(
            x, a[bursts[i][0] : bursts[i][1]], c="magenta", alpha=0.5, linewidth=0.3
        )
        ax[0, 1].plot(
            x, tpxx[bursts[i][0] : bursts[i][1]], c="magenta", alpha=0.5, linewidth=0.3
        )
        ax[1, 0].plot(
            x, bpxx[bursts[i][0] : bursts[i][1]], c="magenta", alpha=0.5, linewidth=0.3
        )
        ax[1, 1].plot(
            x, gpxx[bursts[i][0] : bursts[i][1]], c="magenta", alpha=0.5, linewidth=0.3
        )
    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            ax[i, j].spines["top"].set_visible(False)
            ax[i, j].spines["right"].set_visible(False)
    plt.subplots_adjust(hspace=0.5)
    if save_path is not None:
        plt.savefig(
            save_path,
            format="svg",
            bbox_inches="tight",
        )


def plot_bursts_baseline(
    burst, baseline, freqs, bottom=None, top=None, num=5, save_path=None, format="svg"
):
    plt.rcParams["xtick.labelsize"] = 20
    plt.rcParams["ytick.labelsize"] = 20
    plt.rcParams["font.size"] = 20
    fig, ax = plt.subplots()
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    ax.tick_params(width=2)
    ax.semilogy(freqs, baseline.mean(axis=0), c="black", linewidth=2, alpha=0.75)
    ax.semilogy(freqs, burst.mean(axis=0), c="red", linewidth=2, alpha=0.75)
    ax.set_xlim(0, 100)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if bottom is not None and top is not None:
        ticks = np.linspace(bottom, top, num=num)
        ax.set_yticks(ticks)
        ax.set_ylim(bottom=ticks[0], top=ticks[-1])
    elif bottom is not None:
        gticks = ax.get_yticks()
        ticks = np.linspace(bottom, gticks[-1], num=num)
        ax.set_yticks(ticks)
        ax.set_ylim(bottom=ticks[0], top=ticks[-1])
    elif top is not None:
        gticks = ax.get_yticks()
        ticks = np.linspace(gticks[0], top, num=num)
        ax.set_yticks(ticks)
        ax.set_ylim(bottom=ticks[0], top=ticks[-1])
    if save_path is not None:
        plt.savefig(
            save_path,
            format=format,
            bbox_inches="tight",
        )
