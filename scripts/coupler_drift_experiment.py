"""Large coupler drift: does a single robust pulse still suffice, and what does a drift-aware policy add?

Drift model (per device): both qubit frequencies within ±5.7 MHz (recool, Zhang et al. 2022) and
the coupler frequency within ±140 MHz, the worst-loop flux drift after 17 days of Dai et al., PRX
Quantum 2, 040313 (2021), converted with an assumed f_max = 8 GHz (scripts/coupler_drift_scan.py).

    python scripts/coupler_drift_experiment.py robust                       # robust GRAPE, 3D ensemble
    python scripts/coupler_drift_experiment.py evaluate --dr-run RUN --static-run RUN

Methods on 24 held-out devices: the 3D robust pulse, the fabrication-range robust pulse (which never
saw coupler drift), the nominal and the drift-trained PPO policies, the drift-trained policy's pulse
refined by GRAPE, GRAPE from random at the policy's gate time (the matched control), and GRAPE from
random at 17.25 ns. Writes docs/figures/coupler_drift.{png,json} (and coupler_robust.{npz,json}).
"""
import argparse
import importlib.util
import json
import os
import time

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rlquantopt.jx import env as jenv, grape, physics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "docs", "figures")
U_MAX, DT, N17 = 20.0, 0.05, 345
W_Q, W_C = 5.7, 140.0            # MHz half-widths: qubits (recool), coupler (worst-loop flux drift)
M0 = physics.ModelParams()


def model_at(dq0, dq1, dc):
    return M0._replace(omega_s=(M0.omega_s[0] + dq0 * 1e-3, M0.omega_s[1] + dq1 * 1e-3), omega_c_0=M0.omega_c_0 + dc * 1e-3)


def stack(models):
    hams = [physics.sector_hamiltonian(m) for m in models]
    return jax.tree_util.tree_map(lambda *x: jnp.stack(x), *hams)


def robust(args):
    members = [model_at(a * W_Q, b * W_Q, c) for c in np.linspace(-W_C, W_C, 7)
               for a, b in ((0, 0), (-1, -1), (-1, 1), (1, -1), (1, 1))]
    cfg = grape.GrapeConfig(n_iter=args.iters, lr=0.05)
    u0 = grape.random_guesses(jax.random.PRNGKey(7), args.restarts, N17, U_MAX, DT)
    t0 = time.perf_counter()
    u, JT, C, U, h = grape.optimise(u0, stack(members), U_MAX, cfg)
    wall = time.perf_counter() - t0
    i = int(jnp.argmin(JT))
    np.savez_compressed(os.path.join(FIG, "coupler_robust.npz"), pulse=np.asarray(u[i]), hist=np.asarray(h[i]))
    json.dump(dict(members=len(members), restarts=args.restarts, iters=args.iters, mean_JT_ensemble=float(JT[i]),
                   wall_s=wall, simulator_samples=float(2 * args.iters * args.restarts * len(members) * N17 * 1.5)),
              open(os.path.join(FIG, "coupler_robust.json"), "w"), indent=2)
    print(f"robust 3D: mean J_T over {len(members)} members {float(JT[i]):.2e} in {wall:.0f} s")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def final_JT(u, m):
    return float(grape.final_cost(jnp.asarray(u), physics.sector_hamiltonian(m), grape.GrapeConfig())[0])


def evaluate(args):
    se, t1 = _load("sample_efficiency"), _load("drift_dimension_policies")
    cfg = jenv.EnvConfig()
    ev = jax.jit(t1.evaluate_params, static_argnums=(0, 2))
    ms, ps, *_ = se.best_params(args.static_run)
    md, pd_, *_ = se.best_params(args.dr_run)
    u_rob = np.load(os.path.join(FIG, "coupler_robust.npz"))["pulse"]
    u_fab = np.load(os.path.join(FIG, "rl_grape_robust_fab.npz"))["pulse_robust"]
    rng = np.random.default_rng(11)
    devices = [model_at(*rng.uniform(-1, 1, 2) * W_Q, rng.uniform(-1, 1) * W_C) for _ in range(args.n_devices)]
    ref_cfg = grape.GrapeConfig(n_iter=args.refine_iters, lr=3e-4)
    ctl_cfg = grape.GrapeConfig(n_iter=args.control_iters, lr=0.05)
    rec = {k: [] for k in ("robust GRAPE, 3D", "robust GRAPE, fab. (no coupler)", "RL, no drift", "RL, coupler drift",
                           "RL + GRAPE", "GRAPE, random, matched T", "GRAPE, random, 17.25 ns")}
    H = {"RL + GRAPE": [], "GRAPE, random, matched T": [], "GRAPE, random, 17.25 ns": []}
    gate = []
    for i, m in enumerate(devices):
        ham = physics.sector_hamiltonian(m)
        rec["robust GRAPE, 3D"].append(final_JT(u_rob, m))
        rec["robust GRAPE, fab. (no coupler)"].append(final_JT(u_fab, m))
        for key, (mod, par) in (("RL, no drift", (ms, ps)), ("RL, coupler drift", (md, pd_))):
            JT, amps, alive = jax.device_get(ev(mod, par, cfg, m))
            JT = np.where(alive, JT, np.inf)
            k = int(np.argmin(JT))
            rec[key].append(float(JT[k]))
            if key == "RL, coupler drift":
                u_rl = np.asarray(amps[:k + 1]).ravel()
        gate.append(len(u_rl) * DT)
        _, JT_r, _, _, h = grape.optimise(jnp.asarray(u_rl)[None], ham, U_MAX, ref_cfg)
        H["RL + GRAPE"].append(np.minimum.accumulate(np.concatenate([[rec["RL, coupler drift"][-1]], np.asarray(h[0])])))
        rec["RL + GRAPE"].append(float(H["RL + GRAPE"][-1][-1]))
        for key, n in (("GRAPE, random, matched T", len(u_rl)), ("GRAPE, random, 17.25 ns", N17)):
            u0 = grape.random_guesses(jax.random.PRNGKey(1000 + i), 1, n, U_MAX, DT)
            _, JT_c, _, _, h = grape.optimise(u0, ham, U_MAX, ctl_cfg)
            H[key].append(np.minimum.accumulate(np.asarray(h[0])))
            rec[key].append(float(JT_c[0]))
        print(f"device {i:2d} T={gate[-1]:5.2f} ns  " + "  ".join(f"{k}={v[-1]:.1e}" for k, v in rec.items()), flush=True)

    thr = args.threshold
    summary = {k: dict(median_JT=float(np.median(v)), q25=float(np.quantile(v, 0.25)), q75=float(np.quantile(v, 0.75)),
                       fraction_reaching=float(np.mean(np.asarray(v) <= thr))) for k, v in rec.items()}
    at = {k: {int(n): dict(median_JT=float(np.median([h[min(n, len(h) - 1)] for h in hs])),
                           fraction_reaching=float(np.mean([h[min(n, len(h) - 1)] <= thr for h in hs])))
              for n in (50, 100, 200, 500, 1000) if n <= max(len(h) for h in hs)} for k, hs in H.items()}
    # Cost to reach the threshold: one-off (RL pretraining, robust optimisation) and per device (GRAPE steps)
    def iters_to(h):
        idx = np.flatnonzero(np.asarray(h) <= thr)
        return int(idx[0]) if len(idx) else np.inf
    import pandas as pd
    train_steps = int(pd.read_csv(os.path.join(args.dr_run, "progress.csv"))["step"].iloc[-1])
    rob = json.load(open(os.path.join(FIG, "coupler_robust.json")))
    rl_per = [cfg.n_steps * 3 + iters_to(h) * len_ * 3 for h, len_ in zip(H["RL + GRAPE"], [int(round(t / DT)) for t in gate])]
    costs = {
        "RL + GRAPE": dict(one_off=float(train_steps * 3), per_device=float(np.median(rl_per))),
        "GRAPE, random, matched T": dict(one_off=0.0, per_device=float(np.median(
            [iters_to(h) * int(round(t / DT)) * 3 for h, t in zip(H["GRAPE, random, matched T"], gate)]))),
        "GRAPE, random, 17.25 ns": dict(one_off=0.0, per_device=float(np.median(
            [iters_to(h) * N17 * 3 for h in H["GRAPE, random, 17.25 ns"]]))),
        "robust GRAPE, 3D": dict(one_off=float(rob["simulator_samples"]),
                                 per_device=0.0 if summary["robust GRAPE, 3D"]["fraction_reaching"] >= 0.5 else np.inf),
    }
    for k, v in summary.items():
        print(k, v)
    print("costs", costs)
    print("at iterations", at)
    json.dump(dict(settings=dict(qubit_half_width_mhz=W_Q, coupler_half_width_mhz=W_C, n_devices=args.n_devices,
                                 refine_iters=args.refine_iters, control_iters=args.control_iters, threshold=thr),
                   gate_time_ns=dict(median=float(np.median(gate)), q25=float(np.quantile(gate, 0.25)),
                                     q75=float(np.quantile(gate, 0.75))),
                   summary=summary, by_iterations=at, costs=costs, per_device=rec),
              open(os.path.join(FIG, "coupler_drift.json"), "w"), indent=2)

    cols = ["#8172b2", "#b8a9d9", "#c44e52", "#dd8452", "#55a868", "#4c72b0", "#9fb7d9"]
    fig, axs = plt.subplots(1, 3, figsize=(20, 5), constrained_layout=True, gridspec_kw=dict(width_ratios=[1.3, 1, 1]))
    ax = axs[0]
    for j, (k, v) in enumerate(rec.items()):
        ax.scatter(j + np.random.default_rng(j).uniform(-0.15, 0.15, len(v)), v, s=12, color=cols[j], alpha=0.8)
        ax.scatter([j], [np.median(v)], marker="_", s=500, color="k")
    ax.set_yscale("log")
    ax.axhline(thr, c="k", ls="--", lw=0.8)
    ax.set_xticks(range(len(rec)), [k.replace(", ", ",\n") for k in rec], fontsize=8)
    ax.set_ylabel("$J_T$ per device (bar: median)")
    ax.set_title(f"{args.n_devices} devices: qubits ±{W_Q} MHz, coupler ±{W_C:.0f} MHz", fontsize=10)
    ax = axs[1]
    for k, c in (("RL + GRAPE", cols[4]), ("GRAPE, random, matched T", cols[5]), ("GRAPE, random, 17.25 ns", cols[6])):
        L = min(len(h) for h in H[k])
        A = np.stack([h[:L] for h in H[k]])
        x = np.arange(L)
        ax.semilogy(x, np.median(A, 0), c=c, lw=2, label=k)
        ax.fill_between(x, *np.quantile(A, [0.25, 0.75], 0), color=c, alpha=0.2, lw=0)
    ax.axhline(thr, c="k", ls="--", lw=0.8)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(0, None)
    ax.set_xlabel("GRAPE iteration (0 = starting pulse)")
    ax.set_ylabel("best $J_T$ so far (median, quartiles)")
    ax.set_title("Per-device GRAPE: warm start vs random start", fontsize=10)
    ax.legend(fontsize=8)
    ax = axs[2]
    N = np.logspace(0, 4, 200)
    cc = {"RL + GRAPE": cols[4], "GRAPE, random, matched T": cols[5], "GRAPE, random, 17.25 ns": cols[6],
          "robust GRAPE, 3D": cols[0]}
    for k, c in cc.items():
        one, per = costs[k]["one_off"], costs[k]["per_device"]
        if np.isfinite(per):
            ax.loglog(N, one + N * per + 1, c=c, lw=2, label=k)
    one, per = costs["RL + GRAPE"]["one_off"], costs["RL + GRAPE"]["per_device"]
    ax.loglog(N, np.full_like(N, one), c=cc["RL + GRAPE"], ls=":", lw=1.2, label="…of which RL pretraining")
    if np.isfinite(per):
        ax.loglog(N, N * per, c=cc["RL + GRAPE"], ls="--", lw=1.2, label="…of which per-device GRAPE")
    ax.set_xlabel("number of devices or re-calibrations")
    ax.set_ylabel("total simulator cost to reach $J_T \\leq 10^{-3}$, 50 ps samples")
    ax.set_title("Cost against number of devices (∞ per device: not drawn)", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    for a in axs[:2]:
        a.grid(alpha=0.3, axis="y")
    fig.savefig(os.path.join(FIG, "coupler_drift.png"), dpi=120)
    print("wrote coupler_drift.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["robust", "evaluate"])
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--restarts", type=int, default=2)
    ap.add_argument("--dr-run")
    ap.add_argument("--static-run")
    ap.add_argument("--n-devices", type=int, default=24)
    ap.add_argument("--refine-iters", type=int, default=200)
    ap.add_argument("--control-iters", type=int, default=1000)
    ap.add_argument("--threshold", type=float, default=1e-3)
    args = ap.parse_args()
    robust(args) if args.mode == "robust" else evaluate(args)


if __name__ == "__main__":
    main()
