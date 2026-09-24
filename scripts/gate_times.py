"""Gate times chosen by the policies in the drift experiments.

Policies' pulses are cut at their best step, so the pulse that a warm-started GRAPE refines can
be much longer than the 17.25 ns used for GRAPE from scratch. This script re-runs the
(deterministic) rollouts on the same held-out devices as scripts/sample_efficiency.py (fabrication
range) and scripts/drift_dimension_policies.py, and records the gate time of each best step.
Writes docs/figures/gate_times.json.

    JAX_PLATFORMS=cpu python scripts/gate_times.py --fab-dr RUN --t1 D2=RUN D3=RUN D5=RUN
"""
import argparse
import json
import os

import numpy as np
import jax
import jax.numpy as jnp

from rlquantopt.jx import env as jenv
from rlquantopt.jx.drift import DRIFT_RANGES

import importlib.util

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def stats(t):
    t = np.asarray(t)
    return dict(median_ns=float(np.median(t)), q25_ns=float(np.quantile(t, 0.25)), q75_ns=float(np.quantile(t, 0.75)),
                min_ns=float(t.min()), max_ns=float(t.max()), times_ns=[float(x) for x in t])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fab-dr", required=True)
    ap.add_argument("--t1", nargs="+", required=True)
    ap.add_argument("--n-devices", type=int, default=24)
    args = ap.parse_args()
    se, t1 = _load("sample_efficiency"), _load("drift_dimension_policies")
    cfg = jenv.EnvConfig()
    ev = jax.jit(t1.evaluate_params, static_argnums=(0, 2))
    out = {}

    model, params, *_ = se.best_params(args.fab_dr)
    w0 = np.asarray(cfg.model.omega_s)
    half = DRIFT_RANGES["fab_targeting"].half_width_mhz
    rng = np.random.default_rng(2026)
    devs = [w0 + rng.uniform(-half, half, 2) * 1e-3 for _ in range(args.n_devices)]   # as in sample_efficiency
    times = []
    for om in devs:
        JT, _, alive = jax.device_get(ev(model, params, cfg, cfg.model._replace(omega_s=tuple(om))))
        times.append((int(np.argmin(np.where(alive, JT, np.inf))) + 1) * cfg.n_time_steps * cfg.dt)
    out["fab-range policy on fab devices"] = stats(times)
    print("fab", out["fab-range policy on fab devices"]["median_ns"], flush=True)

    for item in args.t1:
        lab, run = item.split("=", 1)
        model, params = t1.load_best(run)
        for test_D in sorted({t1.DIMS[lab], 5}):
            rng = np.random.default_rng(100 + test_D)
            times = []
            for mp in t1.draw_devices(args.n_devices, test_D, rng):
                JT, _, alive = jax.device_get(ev(model, params, cfg, mp))
                times.append((int(np.argmin(np.where(alive, JT, np.inf))) + 1) * cfg.n_time_steps * cfg.dt)
            key = f"{lab} policy on D={test_D} devices"
            out[key] = stats(times)
            print(key, out[key]["median_ns"], flush=True)
    json.dump(out, open(os.path.join(ROOT, "docs", "figures", "gate_times.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
