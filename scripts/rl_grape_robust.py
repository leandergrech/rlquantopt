"""RL vs GRAPE vs robust GRAPE at equal gate time and amplitude bound (roadmap items 1 and 2).

All pulses have the RL pulse's duration (17.25 ns, 345 samples of 50 ps) and the RL
amplitude bound (|u| <= 20 rad/ns). Compared:

  rl               the paper's RL pulse, stopped at its best time
  rl+grape         RL pulse refined by GRAPE on the nominal Hamiltonian
  grape            GRAPE from random smooth guesses (nominal Hamiltonian)
  robust           robust GRAPE from random guesses: mean J_T over a 5x5 grid of ±10 MHz detunings
  rl+robust        robust GRAPE started from the RL pulse
  rl_jax+grape     (if present) the pulse of our own JAX-trained agent, refined by GRAPE at its own length

For each pulse: nominal J_T, the robustness map on the paper's ±50 MHz grid (the ensemble
only covers ±10 MHz, so most of the map is held out), the area with J_T <= 1e-3 and 1e-2,
and smoothness (total variation, max |u|). Writes docs/figures/rl_grape_robust.{png,npz,json}.

    JAX_PLATFORMS=cpu python scripts/rl_grape_robust.py
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rlquantopt.jx import grape, physics, robustness as rb
from rlquantopt.jx import env as jenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DT = 0.05
N_SAMPLES = 345             # 17.25 ns: the paper RL pulse's best time
U_MAX = 20.0                # rad/ns, the RL action bound (10/pi GHz)


def load_rl_pulse():
    u = pd.read_csv(os.path.join(ROOT, "rlquantopt", "paper_plots", "final_pulses", "RL_pulse.csv"))["amplist"]
    return jnp.asarray(u.to_numpy()[1:N_SAMPLES + 1])


def load_jax_agent_pulse():
    """Best evaluation episode of our JAX TRPO run, cut at its best step."""
    path = os.path.join(ROOT, "results", "jax_trpo_paper_s123", "evals.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path)
    JT = np.where(d["alive"], d["JT"], np.inf)
    i, k = np.unravel_index(np.argmin(JT), JT.shape)
    return jnp.asarray(d["amps"][i, :k + 1].ravel()), float(JT[i, k])


def summarise(name, u, ham, cost_evals, d_map):
    cfg = jenv.EnvConfig(pulse_length=len(u), T=DT * len(u))
    JT0 = float(grape.final_cost(u, ham, grape.GrapeConfig())[0])
    M = rb.detuning_map(u, cfg, d_map, d_map)
    cell = (d_map[1] - d_map[0]) ** 2
    r = np.hypot(*np.meshgrid(d_map, d_map, indexing="ij"))
    row = dict(name=name, JT_nominal=JT0,
               area_JT_le_1e3_MHz2=float((M <= 1e-3).sum() * cell),
               area_JT_le_1e2_MHz2=float((M <= 1e-2).sum() * cell),
               mean_log10_JT_r10=float(np.log10(M[r <= 10]).mean()),
               mean_log10_JT_r25=float(np.log10(M[r <= 25]).mean()),
               total_variation=float(np.abs(np.diff(np.asarray(u))).sum()),
               max_abs_u=float(np.abs(np.asarray(u)).max()),
               simulator_pulse_evals=int(cost_evals))
    print(f"{name:13s} J_T={JT0:.2e}  area(J_T<=1e-3)={row['area_JT_le_1e3_MHz2']:6.0f} MHz²  "
          f"area(<=1e-2)={row['area_JT_le_1e2_MHz2']:6.0f} MHz²  <log10 J_T>_r10={row['mean_log10_JT_r10']:.2f}  "
          f"TV={row['total_variation']:.0f}  evals={cost_evals:.1e}", flush=True)
    return row, M


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--restarts", type=int, default=4)
    p.add_argument("--ens-half-width", type=float, default=10.0, help="MHz")
    p.add_argument("--ens-n", type=int, default=5, help="ensemble grid points per axis")
    p.add_argument("--map-step", type=float, default=1.0, help="MHz")
    p.add_argument("--out", default=os.path.join(ROOT, "docs", "figures"))
    args = p.parse_args()

    model = physics.ModelParams()
    ham = physics.sector_hamiltonian(model)
    hams = grape.ensemble_hamiltonian(model, grape.detuning_grid(model, args.ens_half_width, args.ens_n))
    n_ens = args.ens_n ** 2
    refine = grape.GrapeConfig(n_iter=args.iters, lr=0.01)      # small steps: start is already good
    scratch = grape.GrapeConfig(n_iter=args.iters, lr=0.05)
    d_map = np.arange(-50, 50 + 1e-9, args.map_step)
    key = jax.random.PRNGKey(0)
    guesses = grape.random_guesses(key, args.restarts, N_SAMPLES, U_MAX, DT)
    rl = load_rl_pulse()

    pulses, t_opt = {}, {}

    def best(out):
        u, JT, *_ = out
        return u[int(jnp.argmin(JT))]

    t0 = time.perf_counter(); pulses["rl"] = rl; t_opt["rl"] = 0.0
    runs = [("rl+grape", rl[None], ham, refine, 1),
            ("grape", guesses, ham, scratch, args.restarts),
            ("robust", guesses, hams, scratch, args.restarts * n_ens),
            ("rl+robust", rl[None], hams, refine, n_ens)]
    evals = {"rl": 13_300_000}      # RL: env steps of the paper's training (each a 3-sample segment)
    for name, u0, h, cfg, n_traj in runs:
        t0 = time.perf_counter()
        pulses[name] = best(grape.optimise(u0, h, U_MAX, cfg))
        t_opt[name] = time.perf_counter() - t0
        evals[name] = 2 * args.iters * n_traj           # forward + backward pulse simulations
        print(f"{name}: optimised in {t_opt[name]:.0f} s", flush=True)

    rows, maps = [], {}
    for name, u in pulses.items():
        row, maps[name] = summarise(name, u, ham, evals[name], d_map)
        row["optimisation_seconds"] = t_opt[name]
        rows.append(row)

    extra = load_jax_agent_pulse()
    if extra is not None:
        u_jax, JT_jax = extra
        t0 = time.perf_counter()
        u_ref = best(grape.optimise(u_jax[None], ham, U_MAX, refine))
        dt = time.perf_counter() - t0
        JT_ref = float(grape.final_cost(u_ref, ham, grape.GrapeConfig())[0])
        print(f"rl_jax+grape: our JAX agent's pulse ({len(u_jax) * DT:.2f} ns) J_T {JT_jax:.2e} -> {JT_ref:.2e} "
              f"in {dt:.0f} s", flush=True)
        rows.append(dict(name="rl_jax", JT_nominal=JT_jax, gate_time_ns=len(u_jax) * DT))
        rows.append(dict(name="rl_jax+grape", JT_nominal=JT_ref, gate_time_ns=len(u_jax) * DT,
                         optimisation_seconds=dt, simulator_pulse_evals=2 * args.iters))

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "rl_grape_robust.json"), "w") as f:
        settings = {k: v for k, v in vars(args).items() if k != "out"} | dict(n_samples=N_SAMPLES, dt=DT, u_max=U_MAX)
        json.dump(dict(settings=settings, results=rows), f, indent=2)
    np.savez_compressed(os.path.join(args.out, "rl_grape_robust.npz"), d_mhz=d_map,
                        **{f"map_{k}": v for k, v in maps.items()}, **{f"pulse_{k}": np.asarray(v) for k, v in pulses.items()})

    names = list(pulses)
    fig = plt.figure(figsize=(3.6 * len(names), 7.2), constrained_layout=True)
    gs = fig.add_gridspec(2, len(names))
    t = (np.arange(N_SAMPLES) + 1) * DT
    for j, name in enumerate(names):
        ax = fig.add_subplot(gs[0, j])
        ax.plot(t, np.asarray(pulses[name]) / (2 * np.pi), lw=0.8)
        ax.set_ylim(-3.3, 3.3)
        ax.set_title(f"{name}: J_T={rows[j]['JT_nominal']:.1e}", fontsize=10)
        ax.set_xlabel("t [ns]")
        if j == 0:
            ax.set_ylabel("u / 2π [GHz]")
        ax = fig.add_subplot(gs[1, j])
        im = ax.imshow(-np.log10(maps[name].T), origin="lower", extent=[-50, 50, -50, 50], cmap="terrain",
                       vmin=0.5, vmax=5)
        ax.contour(d_map, d_map, maps[name].T, levels=[1e-3], colors="r", linewidths=0.8)
        ax.add_patch(plt.Rectangle((-args.ens_half_width,) * 2, 2 * args.ens_half_width, 2 * args.ens_half_width,
                                   fill=False, ec="w", ls="--", lw=0.8))
        ax.set_xlabel("Δω qubit 0 [MHz]")
        if j == 0:
            ax.set_ylabel("Δω qubit 1 [MHz]")
    fig.colorbar(im, ax=fig.axes[1::2], shrink=0.8, label="$-\\log_{10} J_T$ (red: $J_T = 10^{-3}$)")
    path = os.path.join(args.out, "rl_grape_robust.png")
    fig.savefig(path, dpi=110)
    print("wrote", path)


if __name__ == "__main__":
    main()
