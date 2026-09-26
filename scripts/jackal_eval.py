"""Idea i10 (jackal): deploy calibration policies on the full device model.

Each policy (trained on the simplified or the full device, see env_device.py) is deployed on the full
model: no RWA, direct coupling, SQUID flux curve, 1 ns AWG, 0.5 ns flux-line filter. Test devices:
24 draws with qubits +-5.7 MHz and coupler flux +-20 mPhi0 ("measured": everything the calibration sees),
and the same with couplings +-5.7 MHz and anharmonicities +-5 MHz as well ("unmeasured"). Calibration
error at deployment x1 and x3 the training error. Reported: 1 - F of the full pulse (the last step).

    python scripts/jackal_eval.py --runs LABEL=RUN_DIR ...

Writes docs/figures/jackal_eval{tag}.json.
"""
import argparse
import dataclasses
import importlib.util
import json
import os

import numpy as np
import jax

from rlquantopt.jx import config  # noqa: F401  (float64)
from rlquantopt.jx import device as dv, env as jenv, env_device

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "docs", "figures")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_run(run):
    """Env config (device parameters restored from their JSON list), model, best parameters, progress."""
    cfg = json.load(open(os.path.join(run, "config.json")))
    e = cfg["env"]
    fields = {f.name for f in dataclasses.fields(jenv.EnvConfig)} - {"model", "device"}
    env_cfg = jenv.EnvConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in e.items() if k in fields},
                             device=dv.DeviceParams(*e["device"]) if isinstance(e.get("device"), list) else dv.DeviceParams())
    return (env_cfg, *_policy(run, env_cfg))


def _policy(run, env_cfg):
    import pickle
    import pandas as pd
    from rlquantopt.jx.agents.common import ActorCritic
    cfg = json.load(open(os.path.join(run, "config.json")))
    model = ActorCritic.make(env_cfg.act_dim, tuple(cfg["algo"]["hidden"]), cfg["algo"]["activation"])
    prog = pd.read_csv(os.path.join(run, "progress.csv"))
    for s in prog.dropna(subset=["eval_JT_min"]).sort_values("eval_JT_min")["step"].astype(int):
        path = os.path.join(run, f"params_{s}.pkl")
        if os.path.exists(path):
            return model, pickle.load(open(path, "rb")), s, prog
    raise FileNotFoundError(run)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="LABEL=RUN_DIR")
    ap.add_argument("--n-devices", type=int, default=24)
    ap.add_argument("--noise-seeds", type=int, default=3)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    ie = _load("ibis_eval")
    out = {}
    for label, run in (r.split("=", 1) for r in args.runs):
        cfg, model, params, step, prog = load_run(run)
        full = dataclasses.replace(cfg, device_simplified=False, awg_dt=1.0, filter_tau=0.5)
        ev = jax.jit(lambda mp, key, c: ie.rollout(model, params, c, mp, key), static_argnums=2)
        rec = dict(run=run, trained_on="simplified" if cfg.device_simplified else "full", best_step=int(step), by_case={})
        for drift in ("measured", "unmeasured"):
            dcfg = dataclasses.replace(full, qubit_drift_mhz=5.7, flux_drift_mphi0=20.0,
                                       g_drift_mhz=5.7 if drift == "unmeasured" else 0.0,
                                       eta_drift_mhz=5.0 if drift == "unmeasured" else 0.0)
            devices = [env_device.sample(jax.random.PRNGKey(10_000 + i), dcfg) for i in range(args.n_devices)]
            for scale in (1, 3):
                ecfg = dataclasses.replace(full, context_noise_mhz=tuple(scale * np.asarray(full.context_noise_mhz)))
                final = [float(jax.device_get(ev(mp, jax.random.PRNGKey(1000 * i + sd), ecfg)[0])[-1])
                         for i, mp in enumerate(devices) for sd in range(args.noise_seeds)]
                key = f"{drift} drift | calibration error x{scale}"
                rec["by_case"][key] = dict(median=float(np.median(final)), q25=float(np.quantile(final, 0.25)),
                                           q75=float(np.quantile(final, 0.75)),
                                           below_1e2=float(np.mean(np.asarray(final) <= 1e-2)),
                                           below_1e3=float(np.mean(np.asarray(final) <= 1e-3)))
                print(f"{label:22s} {key:42s} median {np.median(final):.2e}  <=1e-2 {np.mean(np.asarray(final) <= 1e-2):.0%}"
                      f"  <=1e-3 {np.mean(np.asarray(final) <= 1e-3):.0%}", flush=True)
        p = prog.dropna(subset=["eval_JT_min"])
        hit = p[p["eval_JT_min"] <= 1e-2]
        rec["steps_to_1e2"] = int(hit["step"].iloc[0]) if len(hit) else None
        out[label] = rec
    json.dump(out, open(os.path.join(FIG, f"jackal_eval{args.tag}.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
