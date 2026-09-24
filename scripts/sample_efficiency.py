"""Sample efficiency of RL and GRAPE, with and without hardware drift.

Question: what does it cost, in simulator work, to get a good gate for (a) the nominal device,
(b) one new device drawn from the recool drift range, and (c) many such devices (a device that
is recalibrated after every cooldown, or a fleet of devices)?

Methods, all with the RL amplitude bound |u| <= 20 rad/ns:
  grape          GRAPE from a random guess at T = 17.25 ns, per device
  robust grape   one robust-GRAPE pulse (recool ensemble), reused on every device
  rl             PPO trained on the nominal device only; one rollout per device
  rl-dr          PPO trained with per-episode domain randomisation over the recool range; one rollout per device
  rl-dr+grape    the rl-dr pulse (cut at its best step) refined by a short GRAPE run, per device

Cost unit: simulated 50 ps samples. A GRAPE iteration counts 3 (forward + backward, the backward
pass costing about twice the forward); an RL environment step counts 3 (three samples).
Training costs are what the runs actually used (20M environment steps). Wall-clock times on the
laptop CPU are reported next to them.

    JAX_PLATFORMS=cpu python scripts/sample_efficiency.py --static-run runs/ppo_..._s123 --dr-run runs/ppo_..._dr
    python scripts/sample_efficiency.py --replot      # redraw the figure from the saved JSON
"""
import argparse
import json
import os
import pickle
import time

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rlquantopt.jx import env as jenv, grape, physics
from rlquantopt.jx.agents.common import ActorCritic, evaluate
from rlquantopt.jx.drift import DRIFT_RANGES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
U_MAX, DT, N_GRAPE = 20.0, 0.05, 345
SAMPLES_PER_ITER = 3        # forward + backward ~ 3 forward passes


def best_params(run):
    d = pd.read_csv(os.path.join(run, "progress.csv")).dropna(subset=["eval_JT_min"])
    cfg = json.load(open(os.path.join(run, "config.json")))
    steps = d.sort_values("eval_JT_min")["step"].astype(int)
    for s in steps:
        path = os.path.join(run, f"params_{s}.pkl")
        if os.path.exists(path):
            params = pickle.load(open(path, "rb"))
            model = ActorCritic.make(3, tuple(cfg["algo"]["hidden"]), cfg["algo"]["activation"])
            train_steps = int(d["step"].iloc[-1])
            return model, params, s, train_steps, float(pd.read_csv(os.path.join(run, "progress.csv"))["wall"].iloc[-1])
    raise FileNotFoundError(run)


def params_at(run, step):
    """Parameters of the checkpoint closest to ``step`` (for the training-budget study)."""
    cands = [int(f[7:-4]) for f in os.listdir(run) if f.startswith("params_") and f[7:-4].isdigit()]
    s = min(cands, key=lambda c: abs(c - step))
    return pickle.load(open(os.path.join(run, f"params_{s}.pkl"), "rb")), s


def rollout(model, params, cfg, omega):
    ep = jax.device_get(evaluate(model, params, cfg, jnp.asarray(omega)))
    JT = np.where(ep["alive"], ep["JT"], np.inf)
    k = int(np.argmin(JT))
    return float(JT[k]), np.asarray(ep["amps"][:k + 1]).ravel()


def iters_to(hist, thr):
    below = np.flatnonzero(np.minimum.accumulate(hist) <= thr)
    return int(below[0]) + 1 if len(below) else np.inf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--static-run", required=True)
    p.add_argument("--dr-run", required=True)
    p.add_argument("--robust-npz", default=os.path.join(ROOT, "docs", "figures", "rl_grape_robust.npz"))
    p.add_argument("--n-devices", type=int, default=24)
    p.add_argument("--grape-iters", type=int, default=1000)
    p.add_argument("--refine-iters", type=int, default=500)
    p.add_argument("--refine-report", type=int, nargs="+", default=[10, 50, 100, 200, 500],
                   help="refinement budgets reported (J_T after this many GRAPE steps)")
    p.add_argument("--cost-refine-iters", type=int, default=200, help="refinement budget used in the cost comparison")
    p.add_argument("--budget-steps", type=float, nargs="*", default=[2e6, 5e6, 10e6, 20e6],
                   help="RL training budgets to evaluate (nearest checkpoints of the drift-trained run)")
    p.add_argument("--range", default="recool", choices=sorted(DRIFT_RANGES), help="drift range of the test devices")
    p.add_argument("--tag", default="", help="suffix for the output files, e.g. _fab")
    p.add_argument("--refine-lr", type=float, default=3e-4, help="small steps keep GRAPE in the RL pulse's basin")
    p.add_argument("--threshold", type=float, default=1e-3)
    p.add_argument("--fleet", type=int, default=100, help="number of devices/recalibrations in scenario (c)")
    p.add_argument("--out", default=os.path.join(ROOT, "docs", "figures"))
    args = p.parse_args()

    model_s, params_s, step_s, train_s, wall_s = best_params(args.static_run)
    model_d, params_d, step_d, train_d, wall_d = best_params(args.dr_run)
    cfg = jenv.EnvConfig()
    w0 = np.asarray(cfg.model.omega_s)
    half = DRIFT_RANGES[args.range].half_width_mhz
    rng = np.random.default_rng(2026)
    devices = [w0] + [w0 + rng.uniform(-half, half, 2) * 1e-3 for _ in range(args.n_devices)]   # index 0 = nominal

    robust_u = jnp.asarray(np.load(args.robust_npz)["pulse_robust"])
    robust_cost = None
    rj = args.robust_npz[:-4] + ".json"        # the JSON written next to the robust pulse
    if os.path.exists(rj):
        rows = {r["name"]: r for r in json.load(open(rj))["results"]}
        robust_cost = rows["robust"]["simulator_pulse_evals"] / 2 * N_GRAPE * SAMPLES_PER_ITER
        robust_wall = rows["robust"].get("optimisation_seconds", np.nan)

    gcfg = grape.GrapeConfig(n_iter=args.grape_iters, lr=0.05)
    rcfg = grape.GrapeConfig(n_iter=args.refine_iters, lr=args.refine_lr)
    rec = {k: [] for k in ("grape", "robust grape", "rl", "rl-dr", "rl-dr+grape")}
    hist_grape, hist_ref, cost, wall = [], [], {k: [] for k in rec}, {k: [] for k in rec}
    pulse_len = []
    for i, om in enumerate(devices):
        ham = physics.sector_hamiltonian(cfg.model._replace(omega_s=jnp.asarray(om)))
        t0 = time.perf_counter()
        JT_s, _ = rollout(model_s, params_s, cfg, om)
        wall["rl"].append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        JT_d, u_d = rollout(model_d, params_d, cfg, om)
        pulse_len.append(len(u_d))
        wall["rl-dr"].append(time.perf_counter() - t0)
        rec["rl"].append(JT_s)
        rec["rl-dr"].append(JT_d)
        # a method that does not reach the threshold on this device has infinite cost there
        cost["rl"].append(cfg.n_steps * 3 if JT_s <= args.threshold else np.inf)
        cost["rl-dr"].append(cfg.n_steps * 3 if JT_d <= args.threshold else np.inf)

        rec["robust grape"].append(float(grape.final_cost(robust_u, ham, gcfg)[0]))
        cost["robust grape"].append(0.0 if rec["robust grape"][-1] <= args.threshold else np.inf)
        wall["robust grape"].append(0.0)

        t0 = time.perf_counter()
        u0 = grape.random_guesses(jax.random.PRNGKey(i), 1, N_GRAPE, U_MAX, DT)
        _, JT_g, _, _, h = grape.optimise(u0, ham, U_MAX, gcfg)
        h = np.asarray(h[0])
        wall["grape"].append(time.perf_counter() - t0)
        rec["grape"].append(float(JT_g[0]))
        hist_grape.append(h)
        cost["grape"].append(iters_to(h, args.threshold) * N_GRAPE * SAMPLES_PER_ITER)

        t0 = time.perf_counter()
        _, JT_r, _, _, h = grape.optimise(jnp.asarray(u_d)[None], ham, U_MAX, rcfg)
        h = np.concatenate([[JT_d], np.asarray(h[0])])
        wall["rl-dr+grape"].append((time.perf_counter() - t0) * args.cost_refine_iters / args.refine_iters
                                   + wall["rl-dr"][-1])
        h_cost = h[:args.cost_refine_iters + 1]
        rec["rl-dr+grape"].append(float(np.min(h_cost)))
        hist_ref.append(h)
        cost["rl-dr+grape"].append(cfg.n_steps * 3 + (iters_to(h_cost, args.threshold) - 1) * len(u_d) * SAMPLES_PER_ITER)
        print(f"device {i:2d} Δω=({(om[0] - w0[0]) * 1e3:+5.2f},{(om[1] - w0[1]) * 1e3:+5.2f}) MHz  "
              + "  ".join(f"{k}={v[-1]:.1e}" for k, v in rec.items()), flush=True)

    train = {"rl": train_s * 3, "rl-dr": train_d * 3, "rl-dr+grape": train_d * 3,
             "robust grape": robust_cost or np.nan, "grape": 0.0}
    train_wall = {"rl": wall_s, "rl-dr": wall_d, "rl-dr+grape": wall_d,
                  "robust grape": robust_wall if robust_cost else np.nan, "grape": 0.0}

    def med(x):
        return float(np.median(np.asarray(x, float)))

    summary = {}
    for k in rec:
        per_dev = np.asarray(cost[k][1:], float)
        per_dev_cost = med(per_dev)                  # inf if the method fails on most devices
        per_dev_wall = med(wall[k][1:]) if np.isfinite(per_dev_cost) else np.inf
        reach = float(np.mean(np.asarray(rec[k][1:]) <= args.threshold))
        summary[k] = dict(JT_nominal=rec[k][0], JT_drift_median=med(rec[k][1:]),
                          JT_drift_q25=float(np.quantile(rec[k][1:], 0.25)),
                          JT_drift_q75=float(np.quantile(rec[k][1:], 0.75)),
                          fraction_reaching_threshold=reach,
                          one_off_cost_samples=float(train[k]), per_device_cost_samples=per_dev_cost,
                          one_off_wall_s=float(train_wall[k]), per_device_wall_s=per_dev_wall,
                          cost_1_device=float(train[k] + per_dev_cost),
                          cost_fleet=float(train[k] + args.fleet * per_dev_cost),
                          wall_1_device_s=float(train_wall[k] + per_dev_wall),
                          wall_fleet_s=float(train_wall[k] + args.fleet * per_dev_wall))
        print(k, {a: f"{b:.3g}" for a, b in summary[k].items()})

    # Refinement budget: J_T after k GRAPE steps from the drift-trained RL pulse
    Hr = np.minimum.accumulate(np.stack(hist_ref)[1:], axis=1)
    refine_budget = {int(k): dict(JT_median=float(np.median(Hr[:, min(k, Hr.shape[1] - 1)])),
                                  fraction_reaching=float(np.mean(Hr[:, min(k, Hr.shape[1] - 1)] <= args.threshold)),
                                  per_device_cost_samples=float(cfg.n_steps * 3 + k * np.median(pulse_len[1:]) * SAMPLES_PER_ITER))
                     for k in args.refine_report}
    print("refinement budget", refine_budget)

    # Training budget: the drift-trained policy at earlier checkpoints
    bcfg = grape.GrapeConfig(n_iter=args.cost_refine_iters, lr=args.refine_lr)
    training_budget = {}
    for target in args.budget_steps:
        params_b, s_b = params_at(args.dr_run, target)
        JTs, JTr = [], []
        for i, om in enumerate(devices[1:]):
            ham = physics.sector_hamiltonian(cfg.model._replace(omega_s=jnp.asarray(om)))
            JT_b, u_b = rollout(model_d, params_b, cfg, om)
            _, JT_rb, *_ = grape.optimise(jnp.asarray(u_b)[None], ham, U_MAX, bcfg)
            JTs.append(JT_b)
            JTr.append(float(min(JT_rb[0], JT_b)))
        training_budget[int(s_b)] = dict(training_samples=float(s_b * 3), rl_JT_median=med(JTs),
                                         rl_grape_JT_median=med(JTr),
                                         rl_grape_fraction_reaching=float(np.mean(np.asarray(JTr) <= args.threshold)))
        print("training budget", s_b, training_budget[int(s_b)], flush=True)

    os.makedirs(args.out, exist_ok=True)
    json.dump(dict(settings={k: v for k, v in vars(args).items() if k not in ("out", "robust_npz", "tag")}
                   | dict(static_best_step=int(step_s), dr_best_step=int(step_d), range_half_width_mhz=half),
                   summary=summary, per_device={k: v for k, v in rec.items()},
                   refinement_budget=refine_budget, training_budget=training_budget),
              open(os.path.join(args.out, f"sample_efficiency{args.tag}.json"), "w"), indent=2)
    np.savez_compressed(os.path.join(args.out, f"sample_efficiency{args.tag}.npz"),
                        hist_grape=np.stack(hist_grape), hist_refine=np.stack(hist_ref))
    plot(summary, rec, args.threshold, args.fleet, os.path.join(args.out, f"sample_efficiency{args.tag}.png"),
         DRIFT_RANGES[args.range])


def plot(summary, rec, threshold, fleet, path, rng=DRIFT_RANGES["recool"]):
    """(a) Cost to reach the threshold, split into one-off (RL pretraining, robust optimisation) and per-device
    GRAPE; (b) total cost against the number of devices; (c) gate quality per device."""
    names = ["grape", "robust grape", "rl", "rl-dr", "rl-dr+grape"]
    labels = {"grape": "GRAPE per device", "robust grape": "robust GRAPE (one pulse)", "rl": "RL, trained without drift",
              "rl-dr": "RL, trained with drift", "rl-dr+grape": "RL with drift → short GRAPE"}
    colours = dict(zip(names, ["#4c72b0", "#8172b2", "#c44e52", "#dd8452", "#55a868"]))
    what = {"recool": "recool", "fab_targeting": "fabrication"}.get(rng.name, rng.name)
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.2), constrained_layout=True, gridspec_kw=dict(width_ratios=[1.2, 1, 1]))

    def parts(n):
        s = summary[n]
        per = s["per_device_cost_samples"]
        return s["one_off_cost_samples"], per

    ax = axs[0]
    scen = [1, fleet]
    x = np.arange(len(scen))
    wb = 0.15
    finite = []
    for j, n in enumerate(names):
        one, per = parts(n)
        for xi, N in zip(x, scen):
            xb = xi + (j - 2) * wb
            if not np.isfinite(per):
                continue
            total = one + N * per
            finite += [v for v in (one, total) if v > 0]
    top = max(finite) * 4
    bottom = min(finite) / 3
    for j, n in enumerate(names):
        one, per = parts(n)
        for xi, N in zip(x, scen):
            xb = xi + (j - 2) * wb
            if not np.isfinite(per):
                ax.bar(xb, top - bottom, wb, bottom=bottom, color="none", edgecolor=colours[n], hatch="///", lw=0.8,
                       label=labels[n] if xi == 0 else None)
                ax.text(xb, top * 1.1, "✗", ha="center", va="bottom", fontsize=11, color=colours[n])
                continue
            if one > 0:
                ax.bar(xb, one - bottom, wb, bottom=bottom, color=colours[n], label=labels[n] if xi == 0 else None)
            if N * per > 0:
                ax.bar(xb, N * per, wb, bottom=max(one, bottom), color=colours[n], alpha=0.35, hatch="..",
                       edgecolor=colours[n], lw=0.5, label=None if one > 0 or xi else labels[n])
    from matplotlib.patches import Patch
    handles, labs = ax.get_legend_handles_labels()
    handles += [Patch(facecolor="0.35", label="one-off: RL pretraining / robust optimisation"),
                Patch(facecolor="0.8", hatch="..", edgecolor="0.35", label="per device: GRAPE (or one rollout)")]
    ax.set_yscale("log")
    ax.set_ylim(bottom, top * 3)
    ax.set_xticks(x, [f"1 new device\n({what} drift)", f"{fleet} devices\n({what} drift)"])
    ax.set_ylabel("simulator cost, 50 ps samples")
    ax.set_title("(a) Cost to reach $J_T \\leq 10^{-3}$, one-off vs per device (✗: never reaches it)", fontsize=10)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    one, per = parts("rl-dr+grape")
    ax.text(x[1] + 2 * wb, (one + fleet * per) * 1.3, f"pretrain {one:.1e}\n+ {fleet} × GRAPE {per:.1e}",
            ha="center", fontsize=7, color=colours["rl-dr+grape"])

    ax = axs[1]
    N = np.logspace(0, 4, 200)
    for n in ("grape", "robust grape", "rl-dr+grape"):
        one, per = parts(n)
        if np.isfinite(per):
            ax.loglog(N, one + N * per, c=colours[n], lw=2, label=labels[n])
    one, per = parts("rl-dr+grape")
    ax.loglog(N, np.full_like(N, one), c=colours["rl-dr+grape"], ls=":", lw=1.2, label="…of which RL pretraining")
    ax.loglog(N, N * per, c=colours["rl-dr+grape"], ls="--", lw=1.2, label="…of which per-device GRAPE")
    g1, gper = parts("grape")
    if np.isfinite(per) and gper > per:
        nstar = one / (gper - per)
        ax.axvline(nstar, c="k", lw=0.8, ls="--")
        ax.text(nstar * 1.1, ax.get_ylim()[0] * 3 if ax.get_ylim()[0] > 0 else 1e6, f"break-even\n≈ {nstar:.0f} devices",
                fontsize=8)
    ax.set_xlabel("number of devices or re-calibrations")
    ax.set_ylabel("total simulator cost, 50 ps samples")
    ax.set_title("(b) Total cost against number of devices", fontsize=10)
    ax.legend(fontsize=7)

    ax = axs[2]
    for j, n in enumerate(names):
        v = np.asarray(rec[n][1:])
        ax.scatter(np.full(len(v), j) + np.random.default_rng(j).uniform(-0.15, 0.15, len(v)), v, s=12,
                   color=colours[n], alpha=0.75)
        ax.scatter([j], [rec[n][0]], marker="*", s=150, color="k", zorder=3)
    ax.set_yscale("log")
    ax.set_xticks(range(len(names)), ["GRAPE\nper device", "robust\nGRAPE", "RL\nno drift", "RL\nwith drift",
                                      "RL with drift\n→ GRAPE"], fontsize=8)
    ax.axhline(threshold, c="k", ls="--", lw=0.8)
    ax.set_ylabel("$J_T$")
    ax.set_title(f"(c) Quality: nominal (★), {len(rec['rl']) - 1} devices ±{rng.half_width_mhz} MHz (dots)", fontsize=10)
    for a in axs:
        a.grid(alpha=0.3, which="both" if a is axs[1] else "major", axis="y" if a is not axs[1] else "both")
    fig.savefig(path, dpi=120)
    print("wrote", path)


def replot(tag=""):
    d = json.load(open(os.path.join(ROOT, "docs", "figures", f"sample_efficiency{tag}.json")))
    rng = DRIFT_RANGES[d["settings"].get("range", "recool")]
    plot(d["summary"], d["per_device"], d["settings"]["threshold"], d["settings"]["fleet"],
         os.path.join(ROOT, "docs", "figures", f"sample_efficiency{tag}.png"), rng)


if __name__ == "__main__":
    import sys
    if "--replot" in sys.argv:
        replot(sys.argv[sys.argv.index("--replot") + 1] if len(sys.argv) > sys.argv.index("--replot") + 1 else "")
    else:
        main()
