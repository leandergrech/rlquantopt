"""Measure the logical-core time of each building block on this machine, so costs can be quoted in core-hours.

Logical-core time is the CPU time the process consumes, summed over all its threads (time.process_time),
so it counts the work done and not how long the machine was shared: a job spread over 16 threads for one
hour is 16 core-hours. Wall time is recorded alongside. Measures, with JAX float64 on the CPU:
  - GRAPE: seconds per iteration for one system, for several pulse lengths (fit: a + b * n_samples);
  - robust GRAPE: seconds per iteration for ensembles of several sizes (fit per member);
  - PPO training: seconds per environment step (paper setup: 64 envs x 128 steps, [128, 128] ReLU);
  - one deterministic policy rollout (333 steps).
Writes docs/figures/compute_costs.json, used by rlquantopt.jx-based analysis scripts via
scripts/compute_cost.py.

    JAX_PLATFORMS=cpu XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
        taskset -c 2 python scripts/bench_costs.py

Pinned to one logical core, core time is the time on that core, without the spinning of idle pool
threads. A laptop CPU's clock varies tenfold with load and temperature (400 MHz to 4.7 GHz here), so
the core's clock is sampled during every measurement and the result is stated in cycles and in
core-seconds at the base clock (REF_GHZ): the same work gives the same numbers on a busy or idle machine.
"""
import json
import os
import platform
import threading
import time

import numpy as np
import jax
import jax.numpy as jnp

from rlquantopt.jx import env as jenv, grape, physics
from rlquantopt.jx.agents import ppo
from rlquantopt.jx.agents.common import evaluate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
U_MAX, DT = 20.0, 0.05
REF_GHZ = 2.1        # i7-1260P performance-core base clock


class Clock:
    """Mean clock (GHz) of the pinned logical core while the block runs, sampled every 20 ms."""
    def __init__(self):
        cpu = sorted(os.sched_getaffinity(0))[0]
        self.path = f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq"

    def __enter__(self):
        self.samples, self.stop = [], threading.Event()
        def run():
            while not self.stop.is_set():
                self.samples.append(int(open(self.path).read()) / 1e6)
                time.sleep(0.02)
        self.t = threading.Thread(target=run, daemon=True)
        self.t.start()
        return self

    def __exit__(self, *a):
        self.stop.set()
        self.t.join()
        self.ghz = float(np.mean(self.samples))


def timed(f, *a, repeats=3):
    """Median (core seconds at REF_GHZ, measured core seconds) of a compiled call."""
    jax.block_until_ready(f(*a))                          # compile
    ref, cs = [], []
    for _ in range(repeats):
        with Clock() as clk:
            c0 = time.process_time()
            jax.block_until_ready(f(*a))
            c = time.process_time() - c0
        cs.append(c)
        ref.append(c * clk.ghz / REF_GHZ)
    return float(np.median(ref)), float(np.median(cs))


def main():
    m = physics.ModelParams()
    ham = physics.sector_hamiltonian(m)
    iters = 20
    cfg = grape.GrapeConfig(n_iter=iters, lr=0.05)
    out = dict(machine=dict(cpu=platform.processor() or platform.machine(), threads=os.cpu_count(),
                            jax=jax.__version__, backend=jax.default_backend(), x64=bool(jax.config.jax_enable_x64),
                            load_average_1min=os.getloadavg()[0], logical_cores_used=len(os.sched_getaffinity(0)),
                            reference_clock_ghz=REF_GHZ, units="core-seconds at the reference clock"))
    for line in open("/proc/cpuinfo"):
        if line.startswith("model name"):
            out["machine"]["cpu"] = line.split(":", 1)[1].strip()
            break

    run = lambda h: jax.jit(lambda u0: grape.optimise(u0, h, U_MAX, cfg)[1])   # outer jit: compile once

    ns, ts, ws = [345, 700, 1000], [], []    # ts at the reference clock, ws measured
    for n in ns:
        u0 = grape.random_guesses(jax.random.PRNGKey(0), 1, n, U_MAX, DT)
        c, w = timed(run(ham), u0)
        ts.append(c / iters)
        ws.append(w / iters)
        print(f"GRAPE, {n} samples: {c / iters * 1e3:.2f} core-ms at {REF_GHZ} GHz ({w / iters * 1e3:.2f} measured)", flush=True)
    b, a = np.polyfit(ns, ts, 1)
    out["grape"] = dict(core_s_per_iter_by_samples=dict(zip(map(str, ns), ts)),
                        measured_core_s_per_iter_by_samples=dict(zip(map(str, ns), ws)),
                        fit_a_s=float(a), fit_b_s_per_sample=float(b))

    Ms, tm, wm = [5, 15, 35], [], []    # tm at the reference clock, wm measured
    u0 = grape.random_guesses(jax.random.PRNGKey(1), 1, 345, U_MAX, DT)
    for M in Ms:
        hams = grape.ensemble_hamiltonian(m, np.tile(np.asarray(m.omega_s), (M, 1)))
        c, w = timed(run(hams), u0)
        tm.append(c / iters)
        wm.append(w / iters)
        print(f"robust GRAPE, {M} members, 345 samples: {c / iters * 1e3:.2f} core-ms at {REF_GHZ} GHz "
              f"({w / iters * 1e3:.2f} measured) per iteration", flush=True)
    bm, am = np.polyfit(Ms, tm, 1)
    out["robust_grape_345"] = dict(core_s_per_iter_by_members=dict(zip(map(str, Ms), tm)),
                                   measured_core_s_per_iter_by_members=dict(zip(map(str, Ms), wm)),
                                   fit_a_s=float(am), fit_b_s_per_member=float(bm))

    env_cfg = jenv.EnvConfig()
    pcfg = ppo.PPOConfig(activation="relu", total_steps=64 * 128 * 12)
    model, runner, opt = ppo.init(jax.random.PRNGKey(0), env_cfg, pcfg)
    update = ppo.make_update(model, env_cfg, pcfg)
    runner, opt, _ = update(runner, opt)
    jax.block_until_ready(runner.params)
    with Clock() as clk:
        c0 = time.process_time()
        for _ in range(3):
            runner, opt, _ = update(runner, opt)
        jax.block_until_ready(runner.params)
        c = time.process_time() - c0
    w_step = c / (3 * pcfg.batch_size)
    s_step = w_step * clk.ghz / REF_GHZ
    out["ppo"] = dict(s_per_env_step=float(s_step), measured_core_s_per_env_step=float(w_step),
                      core_hours_per_20M_steps=float(s_step * 20e6 / 3600))
    print(f"PPO: {s_step * 1e6:.1f} core-µs at {REF_GHZ} GHz ({w_step * 1e6:.1f} measured) per env step -> "
          f"{s_step * 20e6 / 3600:.2f} core-hours for 20M steps", flush=True)

    ev = jax.jit(lambda p: evaluate(model, p, env_cfg)["JT"])
    c, w = timed(ev, runner.params)
    out["rollout"] = dict(s=c, measured_core_s=w)
    print(f"one policy rollout: {c * 1e3:.1f} core-ms at {REF_GHZ} GHz ({w * 1e3:.1f} measured)", flush=True)
    json.dump(out, open(os.path.join(ROOT, "docs", "figures", "compute_costs.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
