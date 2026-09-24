"""GRAPE loss curves and final-result distributions.

(a) Single optimisations on the nominal device, from docs/figures/rl_grape_robust.npz: J_T at each
    iteration (thin) and best so far (thick) for the kept restart of each method.
(b) Across the drifted devices of docs/figures/sample_efficiency.npz: best-so-far J_T of GRAPE from a
    random guess and of GRAPE started from the domain-randomised RL pulse, as median (solid),
    geometric mean (dashed) and inter-quartile band.
(c) Zoom on the last 20 % of iterations of (b), on a normalised iteration axis.
(d) Final J_T on every drifted device for every method of the sample-efficiency experiment.

Writes docs/figures/grape_losses.png.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "docs", "figures")
SINGLE = {"rl+grape": "RL → GRAPE", "grape": "GRAPE from random", "robust": "robust GRAPE (mean over recool)",
          "rl+robust": "RL → robust GRAPE"}
COL = {"grape": "#4c72b0", "robust grape": "#8172b2", "rl": "#c44e52", "rl-dr": "#dd8452", "rl-dr+grape": "#55a868"}


def band(ax, x, H, colour, label):
    B = np.minimum.accumulate(H, axis=1)
    ax.semilogy(x, np.median(B, 0), c=colour, lw=2, label=f"{label}: median")
    ax.semilogy(x, 10 ** np.log10(B).mean(0), c=colour, lw=1.3, ls="--", label=f"{label}: geometric mean")
    ax.fill_between(x, *np.quantile(B, [0.25, 0.75], 0), color=colour, alpha=0.2, lw=0)


def main():
    fig, axs = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)

    ax = axs[0, 0]
    d = np.load(os.path.join(FIG, "rl_grape_robust.npz"))
    for i, (k, lab) in enumerate(SINGLE.items()):
        if f"hist_{k}" in d:
            h = d[f"hist_{k}"]
            c = plt.rcParams["axes.prop_cycle"].by_key()["color"][i]
            it = np.arange(1, len(h) + 1)
            ax.semilogy(it, h, c=c, lw=0.5, alpha=0.3)
            ax.semilogy(it, np.minimum.accumulate(h), c=c, lw=1.8, label=lab)
    ax.set_xscale("log")
    ax.set_title("(a) Single optimisations, nominal device (thin: each iteration)", fontsize=10)
    ax.set_xlabel("GRAPE iteration")
    ax.set_ylabel("$J_T$")
    ax.legend(fontsize=8)

    s = np.load(os.path.join(FIG, "sample_efficiency.npz"))
    se = json.load(open(os.path.join(FIG, "sample_efficiency.json")))
    Hg, Hr = s["hist_grape"][1:], s["hist_refine"][1:]           # index 0 is the nominal device
    ax = axs[0, 1]
    band(ax, np.arange(1, Hg.shape[1] + 1), Hg, COL["grape"], "GRAPE from random")
    band(ax, np.arange(0, Hr.shape[1]), Hr, COL["rl-dr+grape"], "GRAPE from RL-DR pulse")
    ax.axhline(1e-3, c="k", ls=":", lw=0.8)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(0, Hg.shape[1])
    ax.set_title(f"(b) Best $J_T$ so far on {len(Hg)} recool-drifted devices (band: quartiles)", fontsize=10)
    ax.set_xlabel("GRAPE iteration (0 = RL pulse)")
    ax.set_ylabel("$J_T$")
    ax.legend(fontsize=7)

    ax = axs[1, 0]
    for H, lab, key in ((Hg, "GRAPE from random", "grape"), (Hr[:, 1:], "GRAPE from RL-DR pulse", "rl-dr+grape")):
        n = H.shape[1]
        k0 = int(0.8 * n)
        x = np.arange(k0, n) / n
        B = np.minimum.accumulate(H, axis=1)[:, k0:]
        for row in B:
            ax.semilogy(x, row, c=COL[key], lw=0.5, alpha=0.25)
        band(ax, x, H[:, k0:] * 0 + B, COL[key], lab)
    ax.set_title("(c) Zoom: last 20 % of the iterations (thin: each device)", fontsize=10)
    ax.set_xlabel("fraction of the iteration budget (GRAPE 1000, refinement 200)")
    ax.set_ylabel("best $J_T$ so far")
    ax.legend(fontsize=7)

    ax = axs[1, 1]
    names = ["grape", "robust grape", "rl", "rl-dr", "rl-dr+grape"]
    data = [np.asarray(se["per_device"][n][1:]) for n in names]
    bp = ax.boxplot(data, positions=range(len(names)), widths=0.5, showfliers=False, patch_artist=True)
    for patch, n in zip(bp["boxes"], names):
        patch.set_facecolor(COL[n])
        patch.set_alpha(0.35)
    for j, (v, n) in enumerate(zip(data, names)):
        ax.scatter(j + np.random.default_rng(j).uniform(-0.12, 0.12, len(v)), v, s=10, color=COL[n], zorder=3)
        ax.scatter([j], [np.exp(np.log(v).mean())], marker="D", s=30, color="k", zorder=4)
    ax.set_yscale("log")
    ax.axhline(1e-3, c="k", ls=":", lw=0.8)
    ax.set_xticks(range(len(names)), names)
    ax.set_title("(d) Final $J_T$ per drifted device (box: quartiles, ◆: geometric mean)", fontsize=10)
    ax.set_ylabel("$J_T$")
    for a in axs.ravel():
        a.grid(alpha=0.3)
    out = os.path.join(FIG, "grape_losses.png")
    fig.savefig(out, dpi=115)
    print("wrote", out)


if __name__ == "__main__":
    main()
