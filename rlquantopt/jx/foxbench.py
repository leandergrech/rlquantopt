"""Per-device adaptation benchmark for idea i06 fox (black-box: only samples, no simulator gradients).

1. Pretrain plain PPO on the nominal Tracker (one policy per seed).
2. Draw held-out devices: an unknown action gain and output offset (toy_envs.Tracker). Overshooting is a
   true terminal here (``oob_terminal``): as a truncation, the bootstrap on a drift-overestimating critic
   makes overshooting pay and PPO fine-tuning collapses into it. The reward is kept positive
   (``reward_shift=1``) so that staying alive is always worth more than ending the episode.
3. On every (seed, device), adapt from the pretrained policy for ``--budget`` updates with
   ``ppo`` (PPO fine-tuning) and ``fox`` (PPO + improvement-equivalent model + stopping rule). Both start
   from the same parameters and random keys; fox with ``--no-exploit`` reproduces ppo exactly.
4. Evaluate after every update (deterministic episodes; for reporting only, not seen by the learners).

Costs: one update = one batch of samples = ``cost_per_update`` return units. Reported per method:
samples-to-target, and utility ``eval return - cost * updates`` at fox's online stopping point, at the full
budget, and (for ppo) at the best fixed budget in hindsight, which fox has to match without hindsight.

    python -m rlquantopt.jx.foxbench --devices 8 --seeds 3 --budget 60
"""
import argparse
import dataclasses
import json
import os
import time
from datetime import datetime

import numpy as np
import jax

from rlquantopt.jx.agents import ppo_plus
from rlquantopt.jx.agents.fox import FoxAdapter, FoxConfig
from rlquantopt.jx.ideas import run_root
from rlquantopt.jx.toy_envs import Tracker
from rlquantopt.jx.toybench import ENV_PPO


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--seed-start", type=int, default=0, help="first pretraining seed (to split seeds over processes)")
    p.add_argument("--summarise", nargs="*", default=None, help="only aggregate the results.json of these run dirs")
    p.add_argument("--devices", type=int, default=8)
    p.add_argument("--device-start", type=int, default=0, help="skip the first devices of the (fixed) device list")
    p.add_argument("--budget", type=int, default=60, help="adaptation updates per device")
    p.add_argument("--pretrain-steps", type=int, default=1_000_000)
    p.add_argument("--gain-range", type=float, nargs=2, default=(0.6, 1.4))
    p.add_argument("--offset-range", type=float, nargs=2, default=(-0.15, 0.15))
    p.add_argument("--target-frac", type=float, default=0.9, help="target = this x pretrained return on the nominal device")
    p.add_argument("--cost-per-update", type=float, default=1.0)
    p.add_argument("--no-exploit", action="store_true")
    p.add_argument("--eval-episodes", type=int, default=16)
    p.add_argument("--ft-lr", type=float, default=1e-4, help="fine-tuning learning rate")
    p.add_argument("--critic-warmup", type=int, default=5, help="first updates train only the critic")
    p.add_argument("--out", default="runs")
    return p.parse_args(argv)


def pretrain(seed, steps):
    env = Tracker(oob_terminal=True, reward_shift=1.0)
    cfg = ppo_plus.PlusConfig(**dict(ENV_PPO["tracker"], total_steps=steps))
    model, runner, opt = ppo_plus.init(jax.random.PRNGKey(seed), env, cfg)
    update = jax.jit(ppo_plus.make_update(model, cfg))
    for _ in range(cfg.n_updates):
        runner, opt, _ = update(runner, opt)
    return model, runner.params


def adapt(model, params, device, seed, method, args):
    cfg = ppo_plus.PlusConfig(**dict(ENV_PPO["tracker"], total_steps=args.budget * 16 * 256, lr=args.ft_lr,
                                     critic_warmup=args.critic_warmup))
    model = model._replace(env=device, cfg=cfg)
    _, runner, opt = ppo_plus.init(jax.random.PRNGKey(1000 + seed), device, cfg)
    runner = runner._replace(params=params)
    opt = opt._replace(policy=ppo_plus.ppo.make_optimizer(cfg).init({k: params[k] for k in ("actor", "critic")}))
    update = jax.jit(ppo_plus.make_update(model, cfg))
    evaluate = jax.jit(lambda p: ppo_plus.evaluate_return(model, p, jax.random.PRNGKey(7), args.eval_episodes)[0])
    fcfg = FoxConfig(cost_per_update=args.cost_per_update, exploit=not args.no_exploit)
    fox = FoxAdapter(update, runner, opt, fcfg, device.max_steps) if method == "fox" else None
    evals, J = [float(evaluate(params))], []
    for _ in range(args.budget):
        if fox:
            stats, _ = fox.step()
            p = fox.runner.params
        else:
            runner, opt, stats = update(runner, opt)
            p = runner.params
        J.append(float(stats["mean_reward"]) * device.max_steps)
        evals.append(float(evaluate(p)))
    out = dict(evals=evals, J=J)
    if fox:
        out.update(stopped_at=fox.stopped_at, stops={str(k): v for k, v in fox.stops.items()}, log=fox.log)
    return out


def main(argv=None):
    args = parse_args(argv)
    if args.summarise is not None:
        results = [r for d in args.summarise for r in json.load(open(os.path.join(d, "results.json")))]
        summary = summarise(results, args)
        print(json.dumps(summary, indent=2))
        return summary
    out = os.path.join(run_root(args.out, "fox"), "foxbench", f"tracker_{datetime.now():%Y%m%d-%H%M%S}")
    os.makedirs(out)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(dict(args=vars(args), fox=dataclasses.asdict(FoxConfig(cost_per_update=args.cost_per_update,
                                                                          exploit=not args.no_exploit))), f, indent=2)
    rng = np.random.default_rng(0)
    devices = [Tracker(gain=float(rng.uniform(*args.gain_range)), offset=float(rng.uniform(*args.offset_range)),
                       oob_terminal=True, reward_shift=1.0)
               for _ in range(args.device_start + args.devices)][args.device_start:]
    t0 = time.perf_counter()
    results = []
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        model, params = pretrain(seed, args.pretrain_steps)
        nominal = float(jax.jit(lambda p: ppo_plus.evaluate_return(model, p, jax.random.PRNGKey(7), args.eval_episodes)[0])(params))
        target = args.target_frac * nominal
        print(f"seed {seed}: pretrained, nominal return {nominal:.1f}, target {target:.1f} ({time.perf_counter() - t0:.0f}s)", flush=True)
        for d, dev in enumerate(devices):
            row = dict(seed=seed, device=args.device_start + d, gain=dev.gain, offset=dev.offset, nominal=nominal, target=target)
            for method in ("ppo", "fox"):
                row[method] = adapt(model, params, dev, seed, method, args)
            e_p, e_f = row["ppo"]["evals"], row["fox"]["evals"]
            print(f"  device {d} (gain {dev.gain:.2f}, offset {dev.offset:+.3f}): start {e_p[0]:7.1f}  "
                  f"ppo end {e_p[-1]:7.1f}  fox end {e_f[-1]:7.1f}  fox stop {row['fox']['stopped_at']}  "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
            results.append(row)
            with open(os.path.join(out, "results.json"), "w") as f:
                json.dump(results, f, default=float)
    summary = summarise(results, args)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return out


def summarise(results, args):
    c = args.cost_per_update
    B = args.budget

    def to_target(ev, target):
        hit = np.flatnonzero(np.asarray(ev) >= target)
        return int(hit[0]) if len(hit) else np.inf       # updates spent (0 = already there)

    util = lambda ev, k: ev[k] - c * k
    rows = []
    for r in results:
        ep, ef = np.asarray(r["ppo"]["evals"]), np.asarray(r["fox"]["evals"])
        stop = r["fox"]["stopped_at"] or B
        stop3 = (r["fox"].get("stops") or {}).get("3") or B
        rows.append(dict(start=ep[0], ppo_end=ep[-1], fox_end=ef[-1], u_no_adapt=ep[0],
                         u_fox_stop3=util(ef, stop3), fox_stop3=stop3,
                         ppo_auc=ep.mean(), fox_auc=ef.mean(),
                         ppo_to_target=to_target(ep, r["target"]), fox_to_target=to_target(ef, r["target"]),
                         u_ppo_full=util(ep, B), u_fox_full=util(ef, B),
                         u_ppo_oracle=max(util(ep, k) for k in range(B + 1)),
                         u_fox_stop=util(ef, stop), fox_stop=stop))
    keys = rows[0].keys()
    mean = {k: float(np.mean([x[k] for x in rows if np.isfinite(x[k])])) for k in keys}
    n_reached = {m: int(sum(np.isfinite(x[f"{m}_to_target"]) for x in rows)) for m in ("ppo", "fox")}
    med_to_target = {m: float(np.median([x[f"{m}_to_target"] for x in rows])) for m in ("ppo", "fox")}
    diff = lambda a, b: [x[a] - x[b] for x in rows]
    paired = {name: dict(mean=float(np.mean(v)), sem=float(np.std(v) / np.sqrt(len(v))), fox_better=int(np.sum(np.asarray(v) > 0)))
              for name, v in [("end", diff("fox_end", "ppo_end")), ("auc", diff("fox_auc", "ppo_auc")),
                              ("utility_stop_vs_oracle", diff("u_fox_stop", "u_ppo_oracle")),
                              ("utility_stop_vs_ppo_full", diff("u_fox_stop", "u_ppo_full")),
                              ("utility_stop_vs_no_adapt", diff("u_fox_stop", "u_no_adapt")),
                              ("utility_stop3_vs_no_adapt", diff("u_fox_stop3", "u_no_adapt")),
                              ("utility_stop3_vs_stop10", diff("u_fox_stop3", "u_fox_stop"))]}
    return dict(n=len(rows), mean=mean, reached_target=n_reached, median_updates_to_target=med_to_target,
                paired_fox_minus_ppo=paired)


if __name__ == "__main__":
    main()
