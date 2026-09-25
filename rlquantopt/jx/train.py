"""Train PPO or TRPO on the JAX ZCQPEE.

    python -m rlquantopt.jx.train --algo trpo --total-steps 20000000 --seed 123
    python -m rlquantopt.jx.train --algo ppo --n-envs 256 --n-steps 128 --max-drift 1e-3
    python -m rlquantopt.jx.train --idea axolotl --algo ppo_judge --activation relu

Writes to runs/<algo>_<timestamp>_s<seed>/ (runs/iNN_<idea>/... with --idea, see ideas.py): config.json, progress.csv (one row
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
from rlquantopt.jx.agents import chameleon, dragonfly, ppo, ppo_judge, ppo_plus, trpo
from rlquantopt.jx.ideas import run_root
from rlquantopt.jx.agents.common import evaluate


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algo", choices=["ppo", "trpo", "ppo_judge", "chameleon", "dragonfly", "ppo_plus"], default="trpo")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--total-steps", type=int, default=20_000_000)
    p.add_argument("--n-envs", type=int, default=None, help="default: 4 (TRPO, as the paper) / 64 (PPO)")
    p.add_argument("--n-steps", type=int, default=None, help="default: 2048 (TRPO) / 128 (PPO)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--activation", choices=["relu", "tanh", "leaky_relu", "gelu"], default=None,
                   help="default: relu (TRPO, as the paper run) / tanh (PPO)")
    p.add_argument("--max-drift", type=float, default=0.0, help="domain randomisation of omega_s, fraction")
    p.add_argument("--coupler-drift-mhz", type=float, default=0.0, help="domain randomisation of the coupler frequency, ±MHz")
    p.add_argument("--g-drift-mhz", type=float, default=0.0, help="domain randomisation of both couplings, ±MHz")
    p.add_argument("--fixed-drift", action="store_true",
                   help="draw the drift once per env (v1 ZCQPEEWRD) instead of every episode")
    p.add_argument("--objective", choices=["pe", "sqrt_iswap"], default="pe",
                   help="pe: the paper's perfect-entangler J_T; sqrt_iswap: 1 - gate fidelity to sqrt(iSWAP), free Z")
    p.add_argument("--obs-mode", choices=["amplitudes", "measured", "context", "measured+context"], default="amplitudes",
                   help="amplitudes: v1 state amplitudes; measured: readout populations and Pauli expectations; "
                        "context: measured qubit/coupler frequency offsets only (open loop); measured+context: both")
    p.add_argument("--oob-mode", choices=["terminate", "clip"], default="terminate",
                   help="terminate (v1): leaving the amplitude bound ends the episode; clip: clip and penalise")
    p.add_argument("--oob-penalty", type=float, default=1.0, help="clip mode: reward lost per unit of normalised excess")
    p.add_argument("--shots", type=int, default=0, help="measured observations from N shots per setting (0 = exact)")
    p.add_argument("--n-time-steps", type=int, default=3, help="pulse samples per env step (K)")
    p.add_argument("--delta-scale", type=float, default=0.0,
                   help="rad/ns per sample for a unit action (default: a_scale = 20, as v1; use 20 * 3 / K for long steps)")
    p.add_argument("--hidden", type=int, nargs="+", default=None, help="hidden layer widths (default 128 128)")
    p.add_argument("--eval-every", type=int, default=2_000_000, help="env steps between evals/checkpoints")
    p.add_argument("--out", default="runs")
    p.add_argument("--idea", default=None, help="idea codename (rlquantopt/jx/ideas.py): runs go to <out>/iNN_<idea>/")
    g = p.add_argument_group("ppo_judge (idea axolotl)")
    g.add_argument("--no-ar", action="store_true", help="Gaussian MLP policy instead of the autoregressive one")
    g.add_argument("--judge-coef", type=float, default=0.5, help="guide-loss weight; 0 turns the judge off")
    g.add_argument("--judge-z", type=float, default=1.0, help="guide only where |mu|/sigma exceeds this")
    g.add_argument("--judge-probe", type=int, default=512, help="samples per update with a measured target")
    g.add_argument("--judge-warmup", type=int, default=20, help="updates before the guide loss starts")
    g = p.add_argument_group("ppo_judge additions (idea badger)")
    g.add_argument("--judge-return", action="store_true", help="the judge also sees the lambda-return")
    g.add_argument("--judge-every", type=int, default=1, help="policy updates per judge update")
    g.add_argument("--judge-half-life", type=float, default=float("inf"),
                   help="guide weight halves every this many policy updates since the judge was trained")
    g.add_argument("--target-kl", type=float, default=float("inf"),
                   help="stop an update's epochs once a minibatch starts with approx KL > 1.5 * this")
    g.add_argument("--n-epochs", type=int, default=None, help="PPO epochs per update (default 10)")
    g.add_argument("--clip-eps", type=float, default=None, help="PPO clip range (default 0.2)")
    g = p.add_argument_group("chameleon (idea chameleon); activation defaults to leaky_relu")
    g.add_argument("--n-candidates", type=int, default=8, help="policy proposals re-ranked by the jumpy model; 1 = no search")
    g.add_argument("--task-reward-coef", type=float, default=1.0, help="weight of cos(z' - z, g) in the policy reward")
    g.add_argument("--env-reward-coef", type=float, default=1.0, help="weight of the env reward in the policy reward")
    g.add_argument("--no-gate", action="store_true", help="fixed exploration share --fixed-eps instead of the learned gate")
    g.add_argument("--fixed-eps", type=float, default=0.0)
    g.add_argument("--goal-every", type=int, default=8, help="env steps per latent task")
    g.add_argument("--latent-dim", type=int, default=16)
    g.add_argument("--model-warmup", type=int, default=5, help="updates that only train the world model")
    g = p.add_argument_group("dragonfly (idea dragonfly); also uses the chameleon flags except --no-gate/--fixed-eps")
    g.add_argument("--sil-coef", type=float, default=None,
                   help="self-imitation of golden-cache actions (default 0.1 for dragonfly, 0 = off for ppo_plus)")
    g.add_argument("--search-value-coef", type=float, default=1.0, help="weight of the lower-bound Q in the search")
    g.add_argument("--lcb-kappa", type=float, default=1.0, help="lower bound = ensemble mean - kappa * std")
    g.add_argument("--cache-size", type=int, default=64)
    g.add_argument("--tau-max", type=float, default=64.0, help="longest exploration heading, in env steps")
    g = p.add_argument_group("ppo_plus (idea echidna); also --sil-coef")
    g.add_argument("--ou-rho", type=float, default=0.0, help="OU exploration-noise persistence; 0 = plain PPO")
    g.add_argument("--veto", action="store_true", help="pessimistic Q-ensemble veto of candidate actions")
    p.add_argument("--tag", default="")
    return p.parse_args(argv)


def build(args):
    env_cfg = jenv.EnvConfig(max_drift=args.max_drift, coupler_drift_mhz=args.coupler_drift_mhz,
                             g_drift_mhz=args.g_drift_mhz, objective=args.objective, obs_mode=args.obs_mode,
                             oob_mode=args.oob_mode, oob_penalty=args.oob_penalty, shots=args.shots,
                             n_time_steps=args.n_time_steps, delta_scale=args.delta_scale)
    common = dict(total_steps=args.total_steps, lr=args.lr, resample_drift=not args.fixed_drift)
    if args.n_envs:
        common["n_envs"] = args.n_envs
    if args.n_steps:
        common["n_steps"] = args.n_steps
    if args.activation:
        common["activation"] = args.activation
    if args.hidden:
        common["hidden"] = tuple(args.hidden)
    if args.algo == "trpo":
        algo_cfg, algo = trpo.TRPOConfig(**common), trpo
    elif args.algo == "ppo_judge":
        extra = {k: v for k, v in dict(n_epochs=args.n_epochs, clip_eps=args.clip_eps).items() if v is not None}
        algo_cfg, algo = ppo_judge.JudgePPOConfig(
            **common, **extra, autoregressive=not args.no_ar, judge_coef=args.judge_coef, judge_z=args.judge_z,
            judge_probe=args.judge_probe, judge_warmup=args.judge_warmup, judge_return=args.judge_return,
            judge_every=args.judge_every, judge_half_life=args.judge_half_life, target_kl=args.target_kl), ppo_judge
    elif args.algo == "chameleon":
        extra = {k: v for k, v in dict(n_epochs=args.n_epochs, clip_eps=args.clip_eps).items() if v is not None}
        algo_cfg, algo = chameleon.ChameleonConfig(
            **common, **extra, n_candidates=args.n_candidates, task_reward_coef=args.task_reward_coef,
            env_reward_coef=args.env_reward_coef, gate=not args.no_gate, fixed_eps=args.fixed_eps,
            goal_every=args.goal_every, latent_dim=args.latent_dim, model_warmup=args.model_warmup), chameleon
    elif args.algo == "dragonfly":
        extra = {k: v for k, v in dict(n_epochs=args.n_epochs, clip_eps=args.clip_eps).items() if v is not None}
        algo_cfg, algo = dragonfly.DragonflyConfig(
            **common, **extra, n_candidates=args.n_candidates, task_reward_coef=args.task_reward_coef,
            env_reward_coef=args.env_reward_coef, goal_every=args.goal_every, latent_dim=args.latent_dim,
            model_warmup=args.model_warmup, sil_coef=0.1 if args.sil_coef is None else args.sil_coef, search_value_coef=args.search_value_coef,
            lcb_kappa=args.lcb_kappa, cache_size=args.cache_size, tau_max=args.tau_max), dragonfly
    elif args.algo == "ppo_plus":
        from rlquantopt.jx.toy_envs import Ham
        algo_cfg = ppo_plus.PlusConfig(**common, ou_rho=args.ou_rho, veto=args.veto, sil_coef=args.sil_coef or 0.0)
        # same entry points as the other algos: the gate env behind the toy interface, a jitted update
        algo = type("PPOPlusOnHam", (), dict(
            init=staticmethod(lambda key, e, c: ppo_plus.init(key, Ham(e), c)),
            make_update=staticmethod(lambda m, e, c: jax.jit(ppo_plus.make_update(m, c)))))
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
    out = os.path.join(run_root(args.out, args.idea) if args.idea else args.out, name)
    os.makedirs(out)        # fails rather than overwrite an existing run
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(dict(args=vars(args), env=dataclasses.asdict(env_cfg), algo=dataclasses.asdict(algo_cfg),
                       backend=jax.default_backend(), x64=bool(jax.config.jax_enable_x64)), f, indent=2, default=str)

# --8<-- [start:train_loop]
    model, runner, opt_state = algo.init(jax.random.PRNGKey(args.seed), env_cfg, algo_cfg)
    update = algo.make_update(model, env_cfg, algo_cfg)
    eval_fn = jax.jit(lambda params: getattr(algo, "evaluate", evaluate)(model, params, env_cfg))
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
    if hasattr(opt_state, "judge"):
        with open(os.path.join(out, "judge_final.pkl"), "wb") as f:
            pickle.dump(jax.device_get(opt_state.judge), f)
    return out


def _write_csv(path, rows):
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "step", k != "wall", k))
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")


if __name__ == "__main__":
    main()
