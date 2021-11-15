"""Plotting utilities"""

import warnings
from os import path

import ase.io
import matplotlib.pyplot as plt
import numpy as np

try:
    from quippy.descriptors import Descriptor
except ModuleNotFoundError:
    pass


def plot_one_page_figure(data, key_list):
    """Plot committee-std estimate on trajectories, 8 on one fig

    Parameters
    ----------
    data: dict
        data dictionary with filename keys
    key_list: list, __len__<=8
        keys in data to use for this plot


    Returns
    -------
    figure

    """
    fig = plt.figure()
    fig.set_figheight(26)
    fig.set_figwidth(20)
    gs_main = plt.GridSpec(nrows=4, ncols=2, figure=fig)

    colors = dict(H="tab:green", C="tab:gray", O="tab:red", energy="tab:blue")

    for i, key in enumerate(key_list):
        d = data[key]

        gs_local = gs_main[i].subgridspec(nrows=2, ncols=1, hspace=0., height_ratios=(3, 1))
        ax = fig.add_subplot(gs_local[0])
        ax_similarity = fig.add_subplot(gs_local[1])

        for sym in ["H", "C", "O"]:
            try:
                ax.plot(d["x"], 1000 * d["fvar_max_{}".format(sym)], c=colors[sym], label=sym, marker="o")
                ax.plot(d["x"], 1000 * d["fvar_mean_{}".format(sym)], c=colors[sym], linestyle="--",
                        alpha=0.3,
                        marker="o")
            except KeyError:
                pass

        ax_ener = ax.twinx()
        ax_ener.plot(d["x"], 1000 * d["evar"], c="tab:blue", label="energy", marker="o")

        # limits
        ax.set_ylim(0, 600)
        ax_ener.set_ylim(1e-2, 1e4)

        ax.set_ylabel("force sqrt(var) / meV/Å", fontsize=20)
        ax_ener.set_ylabel("energy sqrt(var) / meV/atom", fontsize=20)
        ax.set_title(f"{d['formula']} - {path.basename(path.dirname(key))}", fontsize=20)
        ax_ener.grid(which='major', color='#CCCCCC', linestyle='--')

        # ax.set_yscale("log")
        ax_ener.set_yscale("log")
        ax_ener.axhline(10, c="tab:blue", linestyle="-.", alpha=0.5)

        # legend magic
        lines = ax.get_lines() + ax_ener.get_lines()
        labels = [lbl.get_label() for lbl in lines]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ax.legend(lines, labels, loc=0, fontsize=20)

        # ----------------------------------------------------------------------------------------
        for k in d.keys():
            if "max_similarity" not in k:
                continue
            c = colors[k.split("_")[2]]
            linestyle = ("-" if k.split("_")[3] == "3.0" else ":")
            ax_similarity.plot(d["x"], 1 - d[k], label=k, color=c, linestyle=linestyle)

        ax_similarity.set_ylim(1e-3, 0.5)
        ax_similarity.set_yscale("log")
        ax_similarity.legend()

    plt.tight_layout()
    return fig


def calc_and_plot_kernel_vs_first(fn, suptitle=None, index=":", threshold=0.85):
    import seaborn as sns

    # settings:
    c = {"H": "tab:green",
         "C": "tab:gray",
         "O": "tab:red", }
    cmap = sns.color_palette("tab20c")
    vmin = 0.5
    text_kw = dict(fontsize=20)

    soap_3 = Descriptor("soap cutoff=3.0 n_max=8 l_max=6 atom_sigma=0.3 n_species=3 species_Z={1 6 8}")

    # read file and perform descriptor calc
    frames = ase.io.read(fn, index)

    arr = np.array(soap_3.calc_descriptor(frames))
    sim_first = np.sum(arr[0, :, :] * arr, axis=2) ** 2

    fig = plt.figure(figsize=(22, 25))
    gs = plt.GridSpec(nrows=3, ncols=1)
    ax = fig.add_subplot(gs[0])
    other_axes = [fig.add_subplot(gs[i]) for i in range(1, 3)]

    first_prod = np.exp(np.mean(np.log(sim_first), axis=1))
    kw_plot = dict(marker='+', linestyle="-.")

    # the values
    ax.plot(first_prod, label="first", **kw_plot)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(which='major', color='#CCCCCC', linestyle='--')
    ax.set_title("exp(mean(log(k), over atoms)", **text_kw)
    _ = ax.legend(**text_kw)

    # line to show where threshold would be met
    try:
        vline_x = np.argwhere(first_prod < threshold).min()
    except ValueError:
        vline_x = len(first_prod)

    ax.axvline(vline_x, linestyle="-.", c="k", alpha=0.7)

    # min distance
    min_distance = []
    for at in frames:
        dis = at.get_all_distances()
        min_distance.append(np.min(dis[dis > 0.]))

    other_axes[0].plot(min_distance, **kw_plot)
    other_axes[0].set_title("Minimal interatomic distance", **text_kw)
    other_axes[0].set_ylim(0, other_axes[0].get_ylim()[-1])
    other_axes[0].set_ylabel("distance / Å", **text_kw)
    other_axes[0].grid(which='major', color='#CCCCCC', linestyle='--')
    if vline_x is not None:
        other_axes[0].axvline(vline_x, linestyle="-.", c="k", alpha=0.7)

    # first heatmap
    sns.heatmap(sim_first.T, ax=other_axes[1], cmap=cmap, vmin=vmin, annot=True)
    _ = other_axes[1].set_yticklabels(frames[1].get_chemical_symbols())
    other_axes[1].set_title("Similarity to first frame, cutoff=3.0, zeta=2", **text_kw)

    if suptitle is not None:
        fig.suptitle(suptitle, **text_kw)
    #     fig.tight_layout()
    return fig
