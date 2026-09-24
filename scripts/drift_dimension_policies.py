"""Test T1 of the hypothesis: drift-aware policies trained with more drifting parameters.

Policies (PPO, seed 123, 20M steps, recool-width drift) trained with
  D=2: omega_0, omega_1 (±0.11 %),  D=3: + omega_c (±5.7 MHz),  D=5: + g_0, g_1 (±5.7 MHz)
are evaluated on held-out devices whose parameters drift (a) in the policy's own training
dimensions and (b) in all five. Per device: the policy's best J_T, and J_T after a short GRAPE
refinement started from the policy's pulse (the warm start). Writes
docs/figures/drift_dimension_policies.{png,json}.

    JAX_PLATFORMS=cpu python scripts/drift_dimension_policies.py --runs D2=runs/..recool D3=runs/..d3 D5=runs/..d5
"""
import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rlquantopt.jx import env as jenv, grape, physics
from rlquantopt.jx.agents.common import ActorCritic

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
U_MAX = 20.0
DIMS = {"D2": 2, "D3": 3, "D5": 5}


def load_best(run):
    d = pd.read_csv(os.path.join(run, "progress.csv")).dropna(subset=["eval_JT_min"])
    cfg = json.load(open(os.path.join(run, "config.json")))
    for s in d.sort_values("eval_JT_min")["step"].astype(int):
        path = os.path.join(run, f"params_{s}.pkl")
        if os.path.exists(path):
            return ActorCritic.make(3, tuple(cfg["algo"]["hidden"]), cfg["algo"]["activation"]), pickle.load(open(path, "rb"))
    raise FileNotFoundError(run)


def draw_devices(n, D, rng, m=physics.ModelParams()):
    """Drifted parameter sets: qubits ±0.11 %, coupler and couplings ±5.7 MHz, first D of (w0, w1, wc, g0, g1)."""
    out = []
    for _ in range(n):
        e = np.zeros(5)
        e[:D] = rng.uniform(-1, 1, D)
        out.append(m._replace(omega_s=(m.omega_s[0] * (1 + 1.1e-3 * e[0]), m.omega_s[1] * (1 + 1.1e-3 * e[1])),
                              omega_c_0=m.omega_c_0 + 5.7e-3 * e[2],
                              g=(m.g[0] + 5.7e-3 * e[3], m.g[1] + 5.7e-3 * e[4])))
    return out


def evaluate_params(model, params, cfg, mp):
    obs, state = jenv.reset_params(mp, cfg)

    def body(carry, _):
        obs, state, alive = carry
        mean, _ = model.dist(params["actor"], obs)
        obs, state, r, term, trunc, info = jenv.step(state, mean, cfg)
        return (obs, state, alive & ~(term | trunc)), (info["JT"], state.amps_cur, alive)

    _, (JT, amps, alive) = jax.lax.scan(body, (obs, state, jnp.array(True)), None, cfg.n_steps)
    return JT, amps, alive


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="LABEL=RUN_DIR with LABEL in D2, D3, D5")
    ap.add_argument("--n-devices", type=int, default=24)
    ap.add_argument("--refine-iters", type=int, default=200)
    ap.add_argument("--refine-lr", type=float, default=3e-4)
    ap.add_argument("--threshold", type=float, default=1e-3)
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "figures"))
    args = ap.parse_args()
    runs = dict(r.split("=", 1) for r in args.runs)
    cfg = jenv.EnvConfig()
    ev = jax.jit(evaluate_params, static_argnums=(0, 2))
    rcfg = grape.GrapeConfig(n_iter=args.refine_iters, lr=args.refine_lr)
    results = {}
    for lab, run in runs.items():
        model, params = load_best(run)
        for test_D in sorted({DIMS[lab], 5}):
            rng = np.random.default_rng(100 + test_D)          # same test devices for every policy
            rl, ref = [], []
            for mp in draw_devices(args.n_devices, test_D, rng):
                JT, amps, alive = jax.device_get(ev(model, params, cfg, mp))
                JT = np.where(alive, JT, np.inf)
                k = int(np.argmin(JT))
                u = jnp.asarray(np.asarray(amps[:k + 1]).ravel())
                _, JT_r, *_ = grape.optimise(u[None], physics.sector_hamiltonian(mp), U_MAX, rcfg)
                rl.append(float(JT[k]))
                ref.append(float(min(JT[k], JT_r[0])))
            key = f"{lab} policy on D={test_D} devices"
            results[key] = dict(train_D=DIMS[lab], test_D=test_D, rl=rl, rl_grape=ref,
                                rl_median=float(np.median(rl)), rl_grape_median=float(np.median(ref)),
                                rl_grape_fraction_reaching=float(np.mean(np.asarray(ref) <= args.threshold)))
            print(key, f"RL median {np.median(rl):.1e}  RL+GRAPE median {np.median(ref):.1e}  "
                  f"reach {np.mean(np.asarray(ref) <= args.threshold):.2f}", flush=True)
    os.makedirs(args.out, exist_ok=True)
    json.dump(dict(settings={k: v for k, v in vars(args).items() if k not in ("out", "runs")}, results=results),
              open(os.path.join(args.out, "drift_dimension_policies.json"), "w"), indent=2)

    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    keys = list(results)
    for j, k in enumerate(keys):
        r = results[k]
        for off, vals, c in ((-0.15, r["rl"], "#dd8452"), (0.15, r["rl_grape"], "#55a868")):
            ax.scatter(j + off + np.random.default_rng(j).uniform(-0.06, 0.06, len(vals)), vals, s=10, color=c, alpha=0.8)
            ax.scatter([j + off], [np.median(vals)], marker="_", s=400, color="k")
    ax.scatter([], [], color="#dd8452", label="policy alone")
    ax.scatter([], [], color="#55a868", label=f"policy + {args.refine_iters} GRAPE steps")
    ax.set_yscale("log")
    ax.axhline(args.threshold, c="k", ls="--", lw=0.8)
    ax.set_xticks(range(len(keys)), [k.replace(" policy on ", " policy\non ") for k in keys], fontsize=8)
    ax.set_ylabel("$J_T$ per held-out device (bar: median)")
    ax.set_title("Drift-aware policies trained with D = 2, 3, 5 drifting parameters", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.savefig(os.path.join(args.out, "drift_dimension_policies.png"), dpi=120)
    print("wrote", os.path.join(args.out, "drift_dimension_policies.png"))


if __name__ == "__main__":
    main()
