"""Control for the warm-start comparisons: GRAPE from a random guess at the policy's own gate time.

The drift-aware policies' pulses are cut at their best step, typically 36-45 ns
(docs/figures/gate_times.json), while GRAPE from scratch in scripts/sample_efficiency.py runs at
17.25 ns. A longer pulse gives GRAPE more freedom, so the fair control for "policy pulse + GRAPE"
is GRAPE from a random guess at the same length on the same device. This script runs that control
for the fabrication-range devices and for the D=5 devices of test T1, and compares it with the
warm-started runs at equal numbers of GRAPE steps. Writes docs/figures/matched_gate_time.json.

    JAX_PLATFORMS=cpu python scripts/matched_gate_time.py
"""
import argparse
import importlib.util
import json
import os

import numpy as np
import jax
import jax.numpy as jnp

from rlquantopt.jx import env as jenv, grape, physics
from rlquantopt.jx.drift import DRIFT_RANGES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "docs", "figures")
U_MAX, DT = 20.0, 0.05


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(devices, times_ns, iters, thr, seed):
    """GRAPE from random at each device's gate time; returns final J_T and best-so-far J_T at checkpoints."""
    cfg = grape.GrapeConfig(n_iter=iters, lr=0.05)
    final, at = [], {k: [] for k in (50, 100, 200, 500, 1000) if k <= iters}
    for i, (mp, t) in enumerate(zip(devices, times_ns)):
        n = int(round(t / DT))
        u0 = grape.random_guesses(jax.random.PRNGKey(seed + i), 1, n, U_MAX, DT)
        _, JT, _, _, h = grape.optimise(u0, physics.sector_hamiltonian(mp), U_MAX, cfg)
        h = np.minimum.accumulate(np.asarray(h[0]))
        final.append(float(JT[0]))
        for k in at:
            at[k].append(float(h[k - 1]))
        print(f"  device {i:2d}  T={t:5.2f} ns  GRAPE-from-random at matched T: J_T={float(JT[0]):.1e}", flush=True)
    return dict(final=final, median_final=float(np.median(final)),
                fraction_reaching=float(np.mean(np.asarray(final) <= thr)),
                median_at={int(k): float(np.median(v)) for k, v in at.items()},
                fraction_reaching_at={int(k): float(np.mean(np.asarray(v) <= thr)) for k, v in at.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--threshold", type=float, default=1e-3)
    args = ap.parse_args()
    gt = json.load(open(os.path.join(FIG, "gate_times.json")))
    t1 = _load("drift_dimension_policies")
    m = physics.ModelParams()
    out = {}

    w0 = np.asarray(m.omega_s)
    half = DRIFT_RANGES["fab_targeting"].half_width_mhz
    rng = np.random.default_rng(2026)
    fab = [m._replace(omega_s=tuple(w0 + rng.uniform(-half, half, 2) * 1e-3)) for _ in range(24)]
    print("fabrication range", flush=True)
    out["fabrication range"] = run(fab, gt["fab-range policy on fab devices"]["times_ns"], args.iters, args.threshold, 0)
    se = np.load(os.path.join(FIG, "sample_efficiency_fab.npz"))
    hr = np.minimum.accumulate(se["hist_refine"][1:], axis=1)
    out["fabrication range"]["warm_start_median_at"] = {int(k): float(np.median(hr[:, min(k, hr.shape[1] - 1)]))
                                                          for k in out["fabrication range"]["median_at"]}

    rng = np.random.default_rng(105)
    d5 = t1.draw_devices(24, 5, rng)
    print("T1, D=5 devices", flush=True)
    out["T1 D=5"] = run(d5, gt["D5 policy on D=5 devices"]["times_ns"], args.iters, args.threshold, 1000)
    json.dump(out, open(os.path.join(FIG, "matched_gate_time.json"), "w"), indent=2)
    for k, v in out.items():
        print(k, {a: b for a, b in v.items() if a != "final"})


if __name__ == "__main__":
    main()
