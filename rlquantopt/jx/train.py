"""Train PPO or TRPO on the JAX ZCQPEE.

    python -m rlquantopt.jx.train --algo trpo --total-steps 20000000 --seed 123
    python -m rlquantopt.jx.train --algo ppo --n-envs 256 --n-steps 128 --max-drift 1e-3

Writes to runs/<algo>_<timestamp>_s<seed>/: config.json, progress.csv (one row
per update), evals.npz (deterministic episode at every eval: per-step reward,
J_T, C, U and the pulse) and params_<step>.pkl checkpoints.
"""
import argparse
import dataclasses
import json
import os
import pickle
import time
from datetime import datetime

import numpy as np
import jax
import jax.numpy as jnp

from rlquantopt.jx import env as jenv
from rlquantopt.jx.agents import ppo, trpo
from rlquantopt.jx.agents.common import evaluate


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algo", choices=["ppo", "trpo"], default="trpo")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--total-steps", type=int, default=20_000_000)
    p.add_argument("--n-envs", type=int, default=None, help="default: 4 (TRPO, as the paper) / 64 (PPO)")
    p.add_argument("--n-steps", type=int, default=None, help="default: 2048 (TRPO) / 128 (PPO)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--activation", choices=["relu", "tanh"], default=None,
                   help="default: relu (TRPO, as the paper run) / tanh (PPO)")
    p.add_argument("--max-drift", type=float, default=0.0, help="domain randomisation of omega_s, fraction")
    p.add_argument("--fixed-drift", action="store_true",
                   help="draw the drift once per env (v1 ZCQPEEWRD) instead of every episode")
    p.add_argument("--eval-every", type=int, default=2_000_000, help="env steps between evals/checkpoints")
    p.add_argument("--out", default="runs")
    p.add_argument("--tag", default="")
    return p.parse_args(argv)


def build(args):
    env_cfg = jenv.EnvConfig(max_drift=args.max_drift)
    common = dict(total_steps=args.total_steps, lr=args.lr, resample_drift=not args.fixed_drift)
    if args.n_envs:
        common["n_envs"] = args.n_envs
    if args.n_steps:
        common["n_steps"] = args.n_steps
    if args.activation:
        common["activation"] = args.activation
    if args.algo == "trpo":
        algo_cfg, algo = trpo.TRPOConfig(**common), trpo
    else:
        algo_cfg, algo = ppo.PPOConfig(**common), ppo
    return env_cfg, algo_cfg, algo


def eval_summary(ep, cfg):
    alive = np.asarray(ep["alive"])
    JT = np.where(alive, np.asarray(ep["JT"]), np.nan)
    C = np.asarray(ep["concurrence"])
    t = np.asarray(ep["t"])
    pe = np.flatnonzero(alive & (C >= 1.0))
    best = np.nanargmin(JT)
    return dict(eval_return=float(np.sum(np.asarray(ep["reward"]) * alive)),
                eval_len=int(alive.sum()),
                eval_JT_min=float(JT[best]), eval_t_JT_min=float(t[best]),
                eval_JT_end=float(JT[alive.sum() - 1]),
                eval_1mU_at_best=float(1 - np.asarray(ep["unitarity"])[best]),
                eval_t_first_PE=float(t[pe[0]]) if len(pe) else float("nan"))


def main(argv=None):
    args = parse_args(argv)
    env_cfg, algo_cfg, algo = build(args)
    name = f"{args.algo}_{datetime.now():%Y%m%d-%H%M%S}_s{args.seed}{'_' + args.tag if args.tag else ''}"
    out = os.path.join(args.out, name)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(dict(args=vars(args), env=dataclasses.asdict(env_cfg), algo=dataclasses.asdict(algo_cfg),
                       backend=jax.default_backend(), x64=bool(jax.config.jax_enable_x64)), f, indent=2, default=str)

# --8<-- [start:train_loop]
    model, runner, opt_state = algo.init(jax.random.PRNGKey(args.seed), env_cfg, algo_cfg)
    update = algo.make_update(model, env_cfg, algo_cfg)
    eval_fn = jax.jit(lambda params: evaluate(model, params, env_cfg))
    steps_per_update = algo_cfg.n_envs * algo_cfg.n_steps
    eval_every = max(1, args.eval_every // steps_per_update)

    rows, evals = [], {k: [] for k in ("step", "reward", "JT", "concurrence", "unitarity", "amps", "alive")}
    t0 = time.perf_counter()
    for i in range(algo_cfg.n_updates):
        runner, opt_state, stats = update(runner, opt_state)
        step = (i + 1) * steps_per_update
        row = dict(step=step, wall=time.perf_counter() - t0, **{k: float(v) for k, v in stats.items()})
        if (i + 1) % eval_every == 0 or i == algo_cfg.n_updates - 1:
            ep = jax.device_get(eval_fn(runner.params))
            row.update(eval_summary(ep, env_cfg))
            evals["step"].append(step)
            for k in ("reward", "JT", "concurrence", "unitarity", "amps", "alive"):
                evals[k].append(np.asarray(ep[k]))
            np.savez_compressed(os.path.join(out, "evals.npz"), **{k: np.stack(v) for k, v in evals.items()})
            with open(os.path.join(out, f"params_{step}.pkl"), "wb") as f:
                pickle.dump(jax.device_get(runner.params), f)
            print(f"[{step:>10d}] {row['wall']:7.0f}s  R={row['eval_return']:8.1f}  "
                  f"minJT={row['eval_JT_min']:.2e}@{row['eval_t_JT_min']:.1f}ns  "
                  f"firstPE={row['eval_t_first_PE']:.2f}ns  1-U={row['eval_1mU_at_best']:.1e}", flush=True)
        rows.append(row)
        if (i + 1) % 10 == 0 or i == algo_cfg.n_updates - 1:
            _write_csv(os.path.join(out, "progress.csv"), rows)
# --8<-- [end:train_loop]
    return out


def _write_csv(path, rows):
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "step", k != "wall", k))
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")


if __name__ == "__main__":
    main()
