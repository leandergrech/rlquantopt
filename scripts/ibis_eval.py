"""Idea i09 (ibis): how much measurement data does a policy need, and how small can it be?

Each trained policy is rolled out on the 24 held-out large-coupler-drift devices of
coupler_drift_experiment.py (qubits ±5.7 MHz, coupler ±140 MHz), with its observations estimated
from N shots per measurement setting, N in --eval-shots (0 = exact). Reported per policy and N:
1 - F to sqrt(iSWAP) of the full pulse (the final step: no model needed to choose it) and of the
pulse cut at its best step (needs the exact fidelity, for reference only), and the shots per pulse,
n_steps x 30 settings x N. Noisy rollouts are repeated --noise-seeds times per device.

    python scripts/ibis_eval.py --runs LABEL=RUN_DIR ... [--eval-shots 0 1000 100 30]

Writes docs/figures/ibis_eval.{json,png} (and panels ibis_eval_a/b.png).
"""
import argparse
import dataclasses
import importlib.util
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

from rlquantopt.jx import config  # noqa: F401  (float64)
from rlquantopt.jx import env as jenv
from rlquantopt.jx.agents.common import ActorCritic

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "docs", "figures")
SETTINGS_PER_STEP = 30


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_run(run):
    """Env config, model and best parameters (lowest nominal eval 1 - F) of a training run."""
    cfg = json.load(open(os.path.join(run, "config.json")))
    fields = {f.name for f in dataclasses.fields(jenv.EnvConfig)} - {"model", "max_drift", "coupler_drift_mhz", "g_drift_mhz"}
    env_cfg = jenv.EnvConfig(**{k: tuple(v) if isinstance(v, list) else v
                                for k, v in cfg["env"].items() if k in fields})
    model = ActorCritic.make(env_cfg.act_dim, tuple(cfg["algo"]["hidden"]), cfg["algo"]["activation"])
    prog = pd.read_csv(os.path.join(run, "progress.csv"))
    for s in prog.dropna(subset=["eval_JT_min"]).sort_values("eval_JT_min")["step"].astype(int):
        path = os.path.join(run, f"params_{s}.pkl")
        if os.path.exists(path):
            return env_cfg, model, pickle.load(open(path, "rb")), s, prog
    raise FileNotFoundError(run)


def rollout(model, params, cfg, mp, key):
    """Deterministic policy (mean action) for a whole episode; per-step exact 1 - F."""
    obs, state = jenv.reset_params(mp, cfg, key)

    def body(carry, _):
        obs, state = carry
        mean, _ = model.dist(params["actor"], obs)
        obs, state, r, term, trunc, info = jenv.step(state, mean, cfg)
        return (obs, state), (info["JT"], info["oob"])

    _, (JT, oob) = jax.lax.scan(body, (obs, state), None, cfg.n_steps)
    return JT, oob


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="LABEL=RUN_DIR")
    ap.add_argument("--eval-shots", type=int, nargs="+", default=[0, 1000, 100, 30])
    ap.add_argument("--noise-seeds", type=int, default=3)
    ap.add_argument("--n-devices", type=int, default=24)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    c = _load("coupler_drift_experiment")
    rng = np.random.default_rng(11)             # the devices of coupler_drift_experiment.py
    devices = [c.model_at(*rng.uniform(-1, 1, 2) * c.W_Q, rng.uniform(-1, 1) * c.W_C) for _ in range(args.n_devices)]

    out = {}
    for label, run in (r.split("=", 1) for r in args.runs):
        cfg, model, params, step, prog = load_run(run)
        n_params = sum(int(np.prod(x.shape)) for x in jax.tree_util.tree_leaves(params["actor"]["net"]))
        ev = jax.jit(lambda mp, key, cfg: rollout(model, params, cfg, mp, key), static_argnums=2)
        rec = dict(run=run, best_step=int(step), n_steps=cfg.n_steps, K=cfg.n_time_steps,
                   train_shots_per_setting=cfg.shots, actor_params=n_params, by_shots={})
        for N in args.eval_shots:
            ecfg = dataclasses.replace(cfg, shots=N)
            final, best, oob = [], [], []
            for i, mp in enumerate(devices):
                for sd in range(args.noise_seeds if N else 1):
                    JT, ob = jax.device_get(ev(mp, jax.random.PRNGKey(1000 * i + sd), ecfg))
                    final.append(float(JT[-1]))
                    best.append(float(JT.min()))
                    oob.append(float(ob.mean()))
            q = lambda v: dict(median=float(np.median(v)), q25=float(np.quantile(v, 0.25)),
                               q75=float(np.quantile(v, 0.75)), below_1e2=float(np.mean(np.asarray(v) <= 1e-2)))
            rec["by_shots"][str(N)] = dict(shots_per_pulse=cfg.n_steps * SETTINGS_PER_STEP * N, final=q(final),
                                           best_step=q(best), oob_step_fraction=float(np.mean(oob)))
            print(f"{label:22s} N={N:5d}  shots/pulse {cfg.n_steps * SETTINGS_PER_STEP * N:9.2e}  "
                  f"final 1-F {np.median(final):.2e}  best-step {np.median(best):.2e}  "
                  f"oob steps {np.mean(oob):.1%}", flush=True)
        # training: env steps (and measurement settings) to first reach 1 - F <= 1e-2 on the nominal device
        p = prog.dropna(subset=["eval_JT_min"])
        hit = p[p["eval_JT_min"] <= 1e-2]
        rec["steps_to_1e2"] = int(hit["step"].iloc[0]) if len(hit) else None
        rec["curve"] = dict(step=p["step"].astype(int).tolist(), jt=p["eval_JT_min"].tolist())
        out[label] = rec
    json.dump(out, open(os.path.join(FIG, f"ibis_eval{args.tag}.json"), "w"), indent=2)
    plot(out, os.path.join(FIG, f"ibis_eval{args.tag}.png"))


def plot(out, path):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    ax = axs[0]
    for label, r in out.items():
        pts = sorted((v["shots_per_pulse"], v["final"]["median"], v["final"]["q25"], v["final"]["q75"])
                     for N, v in r["by_shots"].items() if int(N) > 0)
        exact = r["by_shots"].get("0")
        x, y, lo, hi = map(np.asarray, zip(*pts))
        l, = ax.loglog(x, y, "o-", label=f"{label} ({r['actor_params']} actor params)")
        ax.fill_between(x, lo, hi, color=l.get_color(), alpha=0.12, lw=0)
        if exact:
            ax.axhline(exact["final"]["median"], color=l.get_color(), ls=":", lw=1)
    ax.axhline(1e-2, c="k", ls="--", lw=0.8)
    ax.set_xlabel("measurement shots per pulse (steps × 30 settings × N)")
    ax.set_ylabel("1 − F of the full pulse, median over devices (dotted: exact observations)")
    ax.set_title("(a) Policy quality against measurement data at deployment", fontsize=10)
    ax.legend(fontsize=7)
    ax = axs[1]
    for label, r in out.items():
        ax.loglog(r["curve"]["step"], np.minimum.accumulate(r["curve"]["jt"]), label=label)
    ax.axhline(1e-2, c="k", ls="--", lw=0.8)
    ax.set_xlabel("training env steps (on hardware each is one experiment: 30 settings × N shots)")
    ax.set_ylabel("best nominal 1 − F so far")
    ax.set_title("(b) Training: data to reach a given gate quality", fontsize=10)
    ax.legend(fontsize=7)
    for a in axs:
        a.grid(alpha=0.3, which="both")
    fig.savefig(path, dpi=120)
    print("wrote", path, _load("figtools").save_panels(fig, axs, path))


if __name__ == "__main__":
    main()
