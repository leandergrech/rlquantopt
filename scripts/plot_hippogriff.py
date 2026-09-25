"""Figures for idea i08 hippogriff (docs/ideas/hippogriff.md), from runs/i08_hippogriff/seed*/results.json.

    python scripts/plot_hippogriff.py            # writes docs/figures/hippogriff_{returns,belief}.png and .json

(a) Return in the first deployment episode on 12 held-out devices, per mode, noise-free and noisy
    measurements; dots are seeds, bars their mean, diamonds the mean in episode 5.
(b) The belief during the first episode: its actual error in the drift z against the error it claims
    (rho x prior std x sqrt(2)); the gap is the filter's overconfidence.
"""
import glob
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODES = ["blind", "robust", "filter_A", "hippogriff_A", "filter_B", "hippogriff_B", "oracle"]
LABELS = ["blind", "robust", "filter\nA", "hippo-\ngriff A", "filter\nB", "hippo-\ngriff B", "oracle"]
BLUE, ORANGE, GREY, DARK = "#2a78d6", "#eb6834", "#8a8984", "#3b3a37"     # reference palette slots 1-2, neutrals
COLOUR = dict(blind=GREY, robust=GREY, oracle=DARK, filter_A=BLUE, hippogriff_A=BLUE, filter_B=ORANGE, hippogriff_B=ORANGE)
PRIOR_STD = 1 / np.sqrt(3)


def load(tag):
    out = []
    for f in sorted(glob.glob("runs/i08_hippogriff/seed*/results.json")):
        if f.split("/")[-2].endswith("noisy") == (tag == "noisy"):
            out.append(json.load(open(f))["modes"])
    return out


def main():
    data = {tag: load(tag) for tag in ("quiet", "noisy")}
    summary = {}
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9})

    # (a) returns
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, (tag, runs) in zip(axes, data.items()):
        summary[tag] = {}
        for i, mode in enumerate(MODES):
            ep1 = np.array([np.mean(r[mode]["returns"][0]) for r in runs])
            ep5 = np.array([np.mean(r[mode]["returns"][-1]) for r in runs])
            summary[tag][mode] = dict(ep1_per_seed=ep1.tolist(), ep5_per_seed=ep5.tolist())
            filled = not mode.startswith("filter")
            c = COLOUR[mode]
            ax.plot([i - 0.28, i + 0.28], [ep1.mean()] * 2, color=c, lw=2, solid_capstyle="round")
            ax.scatter(i + np.linspace(-0.12, 0.12, len(ep1)), ep1, s=36, zorder=3,
                       facecolor=c if filled else "white", edgecolor=c, linewidth=1.5)
            ax.scatter([i + 0.36], [ep5.mean()], marker="D", s=22, color=c, zorder=3)
        ax.set_xticks(range(len(MODES)), LABELS)
        ax.set_title({"quiet": "(a) Noise-free measurements", "noisy": "(b) Noisy measurements (score ± 0.5, Δu ± 0.02)"}[tag],
                     loc="left", fontsize=10)
        ax.grid(axis="y", color="#e4e3df", lw=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Return, first episode on the device")
    handles = [plt.Line2D([], [], marker="o", ls="", mfc="white", mec=GREY, mew=1.5, label="filter: passive identification (dots: seeds)"),
               plt.Line2D([], [], marker="o", ls="", color=GREY, label="hippogriff: + probing, trust region, safety"),
               plt.Line2D([], [], marker="D", ls="", color=GREY, ms=5, label="mean in episode 5"),
               plt.Line2D([], [], color=BLUE, lw=2, label="A: exact (grey-box) model"),
               plt.Line2D([], [], color=ORANGE, lw=2, label="B: learned model")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig("docs/figures/hippogriff_returns.png", dpi=130)

    # (b) belief: actual vs claimed error during episode 1
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, (tag, runs) in zip(axes, data.items()):
        for mode, c in (("hippogriff_A", BLUE), ("hippogriff_B", ORANGE)):
            z = np.mean([np.mean(r[mode]["zerr"][0], 0) for r in runs], 0)
            rho = np.mean([np.mean(r[mode]["rho"][0], 0) for r in runs], 0)
            claimed = rho * PRIOR_STD * np.sqrt(2)
            t = np.arange(1, len(z) + 1)
            ax.plot(t, z, color=c, lw=2)
            ax.plot(t, claimed, color=c, lw=2, ls=(0, (4, 3)))
            name = "A" if mode.endswith("A") else "B"
            ax.annotate(f"{name} actual", (t[-1], z[-1]), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8)
            ax.annotate(f"{name} claimed", (t[-1], claimed[-1]), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8)
            summary[tag][mode + "_belief"] = dict(actual=z.tolist(), claimed=claimed.tolist())
        ax.set_yscale("log")
        ax.set_xlim(0, 118)
        ax.set_xlabel("Step in the first episode")
        ax.set_title({"quiet": "(c) Noise-free", "noisy": "(d) Noisy"}[tag], loc="left", fontsize=10)
        ax.grid(axis="y", color="#e4e3df", lw=0.8, which="major")
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Error in the drift z (norm)")
    fig.legend(handles=[plt.Line2D([], [], color=GREY, lw=2, label="actual error |m − z|"),
                        plt.Line2D([], [], color=GREY, lw=2, ls=(0, (4, 3)), label="error the belief claims (ρ × prior std × √2)")],
               loc="lower center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig("docs/figures/hippogriff_belief.png", dpi=130)
    with open("docs/figures/hippogriff.json", "w") as f:
        json.dump(summary, f)
    print("wrote docs/figures/hippogriff_{returns,belief}.png, hippogriff.json")


if __name__ == "__main__":
    main()
