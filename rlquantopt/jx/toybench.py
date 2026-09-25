"""Smoke-test PPO ingredients in isolation on the toy envs (idea i05 echidna), seeds batched with vmap.

    python -m rlquantopt.jx.toybench --env tracker --variant ou --seeds 5
    python -m rlquantopt.jx.toybench --env mountaincar --variant all --total-steps 2000000

Writes runs/i05_echidna/toybench/<env>_<variant>_<timestamp>/curves.npz (per-seed eval return and score
against env steps) and summary.json (final and area-under-curve, mean and std over seeds).
"""
import argparse
import dataclasses
import json
import os
import time
from datetime import datetime

import numpy as np
import jax
import jax.numpy as jnp

from rlquantopt.jx.agents import ppo_plus
from rlquantopt.jx.ideas import run_root
from rlquantopt.jx.toy_envs import ENVS

VARIANTS = {
    "ppo": {},
    "ou": dict(ou_rho=0.9),
    "veto": dict(veto=True),
    "sil": dict(sil_coef=0.1),
    "all": dict(ou_rho=0.9, veto=True, sil_coef=0.1),
}

# Per-env PPO settings: the same for every variant, so only the ingredient differs
ENV_PPO = {
    "pendulum": dict(n_envs=16, n_steps=256, gamma=0.95, lr=1e-3, total_steps=1_000_000),
    "mountaincar": dict(n_envs=16, n_steps=512, gamma=0.99, lr=3e-4, total_steps=2_000_000),
    "tracker": dict(n_envs=16, n_steps=256, gamma=0.95, lr=3e-4, total_steps=1_000_000),
    "ham": dict(n_envs=64, n_steps=128, gamma=0.99, lr=3e-4, total_steps=20_000_000, activation="relu"),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", choices=list(ENVS), required=True)
    p.add_argument("--variant", choices=list(VARIANTS), required=True)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--total-steps", type=int, default=None)
    p.add_argument("--evals", type=int, default=40, help="evaluation points over training")
    p.add_argument("--set", nargs="*", default=[], help="config overrides, key=value (python literal)")
    p.add_argument("--out", default="runs")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    env = ENVS[args.env]()
    kw = dict(ENV_PPO[args.env], **VARIANTS[args.variant])
    if args.total_steps:
        kw["total_steps"] = args.total_steps
    for s in args.set:
        k, v = s.split("=", 1)
        kw[k] = eval(v)
    cfg = ppo_plus.PlusConfig(**kw)

    out = os.path.join(run_root(args.out, "echidna"), "toybench",
                       f"{args.env}_{args.variant}_{datetime.now():%Y%m%d-%H%M%S}")
    os.makedirs(out)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(dict(args=vars(args), algo=dataclasses.asdict(cfg)), f, indent=2, default=str)

    keys = jax.random.split(jax.random.PRNGKey(0), args.seeds)
    # init returns a static model: build it once, and vmap only the arrays over seeds
    model, _, _ = ppo_plus.init(keys[0], env, cfg)
    runner, opt = jax.vmap(lambda k: ppo_plus.init(k, env, cfg)[1:])(keys)
    update = jax.jit(jax.vmap(ppo_plus.make_update(model, cfg)))
    evaluate = jax.jit(jax.vmap(lambda p, k: ppo_plus.evaluate_return(model, p, k)))

    n_updates = cfg.n_updates
    eval_at = set(np.unique(np.linspace(0, n_updates - 1, args.evals).astype(int)).tolist())
    curves = dict(step=[], ret=[], score=[], veto_rate=[], mean_reward=[])
    t0 = time.perf_counter()
    for i in range(n_updates):
        runner, opt, stats = update(runner, opt)
        if i in eval_at:
            ret, score = evaluate(runner.params, jax.random.split(jax.random.PRNGKey(1), args.seeds))
            step = (i + 1) * cfg.batch_size
            curves["step"].append(step)
            curves["ret"].append(np.asarray(ret))
            curves["score"].append(np.asarray(score))
            curves["veto_rate"].append(np.asarray(stats["veto_rate"]))
            curves["mean_reward"].append(np.asarray(stats["mean_reward"]))
            print(f"[{step:>9d}] {time.perf_counter() - t0:6.0f}s  eval return {np.mean(ret):9.2f} ± {np.std(ret):7.2f}"
                  f"  score {np.mean(score):8.2f}  veto {np.mean(stats['veto_rate']):.3f}", flush=True)
    c = {k: np.stack(v) if k != "step" else np.asarray(v) for k, v in curves.items()}
    np.savez_compressed(os.path.join(out, "curves.npz"), **c)
    auc = c["ret"].mean(0)                      # per seed: mean eval return over training
    summary = dict(env=args.env, variant=args.variant, seeds=args.seeds, wall=time.perf_counter() - t0,
                   final_mean=float(c["ret"][-1].mean()), final_std=float(c["ret"][-1].std()),
                   auc_mean=float(auc.mean()), auc_std=float(auc.std()),
                   final_per_seed=c["ret"][-1].tolist())
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary))
    return out


if __name__ == "__main__":
    main()
