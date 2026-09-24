"""Compare training runs: best evaluation J_T against environment steps and against wall-clock time.

    python scripts/compare_training.py results/jax_trpo_paper_s123 runs/ppo_*_s1 ... --labels TRPO PPO ...

Runs with the same label are grouped (seeds): the plot shows each seed thinly and the median
thickly; the table reports per-run numbers and the per-label median. Writes
docs/figures/training_comparison.{png,json}.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(run):
    d = pd.read_csv(os.path.join(run, "progress.csv"))
    e = d.dropna(subset=["eval_JT_min"])
    cfg = json.load(open(os.path.join(run, "config.json")))
    best = np.minimum.accumulate(e["eval_JT_min"].to_numpy())
    return dict(step=e["step"].to_numpy(), wall=e["wall"].to_numpy(), JT=e["eval_JT_min"].to_numpy(),
                best=best, first_pe=e["eval_t_first_PE"].to_numpy(), seed=cfg["args"]["seed"],
                algo=cfg["args"]["algo"], total_wall=float(d["wall"].iloc[-1]), total_steps=int(d["step"].iloc[-1]),
                one_minus_U=e["eval_1mU_at_best"].to_numpy())


def summary(r):
    i = int(np.argmin(r["JT"]))
    pe = r["first_pe"][np.isfinite(r["first_pe"])]
    brk = r["step"][np.argmax(r["JT"] < 1e-2)] if (r["JT"] < 1e-2).any() else np.nan
    return dict(seed=r["seed"], best_JT=float(r["JT"][i]), step_of_best=int(r["step"][i]),
                one_minus_U_at_best=float(r["one_minus_U"][i]),
                steps_to_JT_below_1e2=float(brk), final_first_PE_ns=float(pe[-1]) if len(pe) else float("nan"),
                wall_minutes=r["total_wall"] / 60, steps_per_second=r["total_steps"] / r["total_wall"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--out", default=os.path.join(ROOT, "docs", "figures"))
    args = p.parse_args()
    assert len(args.labels) == len(args.runs)

    groups = {}
    for run, lab in zip(args.runs, args.labels):
        groups.setdefault(lab, []).append(load(run))

    colours = dict(zip(groups, plt.rcParams["axes.prop_cycle"].by_key()["color"]))
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    table = {}
    for lab, runs in groups.items():
        for r in runs:
            axs[0].semilogy(r["step"] / 1e6, r["best"], c=colours[lab], lw=0.8, alpha=0.5)
            axs[1].semilogy(r["wall"] / 60, r["best"], c=colours[lab], lw=0.8, alpha=0.5)
        grid = np.linspace(0, min(r["step"][-1] for r in runs), 200)
        med = np.median([np.interp(grid, r["step"], r["best"]) for r in runs], axis=0)
        axs[0].semilogy(grid / 1e6, med, c=colours[lab], lw=2.2, label=f"{lab} (n={len(runs)})")
        rows = [summary(r) for r in runs]
        table[lab] = dict(runs=rows, median={k: float(np.nanmedian([row[k] for row in rows]))
                                             for k in rows[0] if k != "seed"})
    for ax, xl in zip(axs, ("environment steps [M]", "wall-clock time [min], laptop CPU")):
        ax.axhline(9.5e-5, c="k", ls="--", lw=0.8)
        ax.set_xlabel(xl)
        ax.set_ylabel("best evaluation $J_T$ so far")
        ax.grid(alpha=0.3)
    axs[0].text(0.2, 1.2e-4, "paper agent (9.5e-5)", fontsize=8)
    axs[0].legend()
    os.makedirs(args.out, exist_ok=True)
    fig.savefig(os.path.join(args.out, "training_comparison.png"), dpi=120)
    with open(os.path.join(args.out, "training_comparison.json"), "w") as f:
        json.dump(table, f, indent=2)
    for lab, t in table.items():
        print(lab, json.dumps(t["median"], indent=None))
        for row in t["runs"]:
            print("   ", row)


if __name__ == "__main__":
    main()
