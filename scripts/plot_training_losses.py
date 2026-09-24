"""Training losses and diagnostics of RL runs, from their progress.csv.

    python scripts/plot_training_losses.py RUN [RUN ...] --labels L [L ...]

Panels: best evaluation J_T, mean rollout reward, value loss, policy objective (TRPO) or clipped
surrogate loss (PPO), KL between successive policies, and the update health (TRPO line-search
success rate, PPO clip fraction). Curves are smoothed over 20 updates. Writes
docs/figures/training_losses.png.
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANELS = [
    ("best evaluation $J_T$", lambda d: np.minimum.accumulate(d["eval_JT_min"].ffill().bfill().to_numpy()), True),
    ("mean rollout reward", lambda d: d["mean_reward"], False),
    ("value loss", lambda d: d["value_loss"] if "value_loss" in d else d["v_loss"], True),
    ("policy objective (TRPO) / clipped loss (PPO)", lambda d: d["policy_objective"] if "policy_objective" in d else d["pg_loss"], False),
    ("KL between successive policies", lambda d: d["kl"] if "kl" in d else d["approx_kl"], True),
    ("line-search success (TRPO) / clip fraction (PPO)", lambda d: d["line_search_success"] if "line_search_success" in d else d["clip_frac"], False),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--out", default=os.path.join(ROOT, "docs", "figures", "training_losses.png"))
    args = p.parse_args()
    colours = {}
    fig, axs = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    for run, lab in zip(args.runs, args.labels):
        d = pd.read_csv(os.path.join(run, "progress.csv"))
        c = colours.setdefault(lab, plt.rcParams["axes.prop_cycle"].by_key()["color"][len(colours)])
        x = d["step"] / 1e6
        first = lab not in [l.get_label() for l in axs[0, 0].get_lines()]
        for ax, (title, f, logy) in zip(axs.ravel(), PANELS):
            y = pd.Series(np.asarray(f(d), float))
            if "best" not in title:
                y = y.rolling(20, min_periods=1).mean()
            ax.plot(x, y, c=c, lw=1.0, alpha=0.85, label=lab if first else None)
            ax.set_title(title, fontsize=10)
            if logy:
                ax.set_yscale("log")
    for ax in axs[1]:
        ax.set_xlabel("environment steps [M]")
    axs[0, 1].set_ylim(-1, 3)      # early out-of-bounds penalties (-20) would squash the rest
    axs[0, 0].axhline(9.5e-5, c="k", ls="--", lw=0.8)
    axs[0, 0].legend(fontsize=8)
    for ax in axs.ravel():
        ax.grid(alpha=0.3)
    fig.savefig(args.out, dpi=110)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
