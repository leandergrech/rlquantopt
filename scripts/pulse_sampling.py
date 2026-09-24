"""Why v1 samples the pulse every 50 ps: the paper's RL pulse re-sampled as coarser piecewise-constant pulses.

Each coarse version holds the block average of the original for 0.25, 1 or 10 ns (10 ns is the
sampling of earlier RL pulse-design work, ~50 MHz bandwidth). J_T is evaluated at 17.25 ns with the
JAX simulator. Writes docs/figures/pulse_sampling.png.
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rlquantopt.jx import env as jenv, robustness as rb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N, DT = 345, 0.05

u = pd.read_csv(os.path.join(ROOT, "rlquantopt", "paper_plots", "final_pulses", "RL_pulse.csv"))["amplist"].to_numpy()[1:N + 1]
cfg = jenv.EnvConfig(pulse_length=N, T=N * DT)
t = (np.arange(N) + 1) * DT
fig, axs = plt.subplots(2, 2, figsize=(11, 5.5), sharex=True, sharey=True, constrained_layout=True)
for ax, hold in zip(axs.ravel(), (0.05, 0.25, 1.0, 10.0)):
    k = int(round(hold / DT))
    v = np.concatenate([np.full(len(b), b.mean()) for b in np.array_split(u, np.arange(k, N, k))])
    JT = float(rb.final_JT(v, cfg, np.asarray(cfg.model.omega_s)))
    ax.plot(t, u / (2 * np.pi), c="0.8", lw=0.6)
    ax.plot(t, v / (2 * np.pi), lw=1.0)
    ax.set_title(f"{hold * 1000:.0f} ps samples: J_T = {JT:.1e}" if hold < 1 else f"{hold:.0f} ns samples: J_T = {JT:.1e}", fontsize=10)
    print(hold, JT)
for ax in axs[1]:
    ax.set_xlabel("t [ns]")
for ax in axs[:, 0]:
    ax.set_ylabel("u / 2π [GHz]")
fig.savefig(os.path.join(ROOT, "docs", "figures", "pulse_sampling.png"), dpi=110)
