"""Replicate paper Figs. 5-8: evolution of the deterministic pulse over training.

Rows: pulse spectrum (Fig. 5), reward (Fig. 6), concurrence error 1-C (Fig. 7) and
unitarity error 1-U with a 1.05 ns minimum filter (Fig. 8), each as a heatmap over
training steps. Left column: the paper's v1 run (rlquantopt/paper_plots/training_evals_v1.npz,
first 15 ns of each episode); right column: a JAX run (runs/<name>/evals.npz).

    python scripts/replicate_training_figs.py runs/trpo_..._paper
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from scipy.ndimage import minimum_filter1d

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DT_STEP = 0.15      # ns per env step
DT = 0.05           # ns per sample


def load_v1():
    d = np.load(os.path.join(ROOT, "rlquantopt", "paper_plots", "training_evals_v1.npz"))
    return dict(step=d["step"], pulse=d["pulse"][:, 1:], reward=d["reward"],
                C=d["concurrence"], U=d["unitarity"])


def load_jx(run, t_max):
    d = np.load(os.path.join(run, "evals.npz"))
    n = int(round(t_max / DT_STEP))
    alive = d["alive"][:, :n]
    mask = lambda x: np.where(alive, x[:, :n], np.nan)
    return dict(step=d["step"], pulse=d["amps"][:, :n].reshape(len(d["step"]), -1),
                reward=mask(d["reward"]), C=mask(d["concurrence"]), U=mask(d["unitarity"]))


def spectrum(pulse):
    f = np.fft.rfftfreq(pulse.shape[1], DT)
    return f, np.abs(np.fft.rfft(pulse, axis=1))


def panel(ax, steps, y, Z, title, ylabel, norm=None, cmap="viridis"):
    im = ax.pcolormesh(steps, y, Z.T, shading="nearest", norm=norm, cmap=cmap)
    ax.set_xscale("log")
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel)
    plt.colorbar(im, ax=ax, pad=0.01)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", nargs="?", help="JAX run directory with evals.npz")
    p.add_argument("--t-max", type=float, default=15.0, help="pulse window shown, ns (v1 data covers 15 ns)")
    p.add_argument("--out", default=os.path.join(ROOT, "docs", "figures", "training_evolution.png"))
    args = p.parse_args()

    cols = [("paper run (v1, SB3 TRPO)", load_v1())]
    if args.run:
        cols.append(("this work (JAX TRPO)", load_jx(args.run, args.t_max)))
    fig, axs = plt.subplots(4, len(cols), figsize=(6.5 * len(cols), 13), squeeze=False, constrained_layout=True)
    for j, (name, d) in enumerate(cols):
        steps = d["step"]
        t = (np.arange(d["C"].shape[1]) + 1) * DT_STEP
        f, S = spectrum(d["pulse"])
        keep = f <= 8
        panel(axs[0, j], steps, f[keep], S[:, keep], f"{name}: pulse spectrum (Fig. 5)", "frequency [GHz]",
              LogNorm(vmin=max(S[:, keep].max() * 1e-4, 1e-3), vmax=S[:, keep].max()), "magma")
        axs[0, j].axhline(0.8588, c="c", lw=0.8, ls="--")
        panel(axs[1, j], steps, t, d["reward"], "reward (Fig. 6)", "pulse time [ns]", Normalize(0, 4), "viridis")
        err_c = np.clip(1 - d["C"], 1e-6, 1)
        panel(axs[2, j], steps, t, err_c, "1 - C (Fig. 7)", "pulse time [ns]", LogNorm(1e-6, 1), "magma_r")
        err_u = minimum_filter1d(np.nan_to_num(np.clip(1 - d["U"], 1e-6, 1), nan=1.0), size=7, axis=1)
        panel(axs[3, j], steps, t, err_u, "1 - U, 1.05 ns min filter (Fig. 8)", "pulse time [ns]",
              LogNorm(1e-4, 1), "magma_r")
        axs[3, j].set_xlabel("training step")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
