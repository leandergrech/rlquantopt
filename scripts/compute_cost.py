"""Convert the experiments' simulator-sample counts into logical-core hours on the benchmarked machine.

The experiments record cost as simulated 50 ps samples: an RL environment step is 3 samples, a
GRAPE iteration on an n-sample pulse is 3 n (forward + backward), a robust-GRAPE iteration is 3 n per
ensemble member. docs/figures/compute_costs.json (scripts/bench_costs.py) holds the logical-core time of
each building block, measured on one pinned logical core with JAX; these helpers turn counts into
core-hours. RL training is priced per environment step, which includes the policy-network updates.
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_C = None


def rates():
    global _C
    if _C is None:
        _C = json.load(open(os.path.join(ROOT, "docs", "figures", "compute_costs.json")))
    return _C


# --8<-- [start:core_hours]
def rl_training_hours(env_steps):
    return env_steps * rates()["ppo"]["s_per_env_step"] / 3600


def grape_iteration_seconds(n_samples):
    """Core seconds of one GRAPE iteration, interpolated between the measured pulse lengths
    (the cost grows faster than linearly with length, so a straight-line fit would misprice short pulses)."""
    g = rates()["grape"]["core_s_per_iter_by_samples"]
    ns = sorted(int(k) for k in g)
    return float(np.interp(n_samples, ns, [g[str(n)] for n in ns]))


def grape_hours_from_samples(samples, n_samples=345):
    """GRAPE work given as 3 * iterations * n_samples."""
    return samples / 3 / n_samples * grape_iteration_seconds(n_samples) / 3600


def robust_hours_from_samples(samples, n_samples=345):
    """Robust-GRAPE work given as 3 * iterations * members * pulse samples."""
    per_member_iter = rates()["robust_grape_345"]["fit_b_s_per_member"] * n_samples / 345
    return samples / 3 / n_samples * per_member_iter / 3600
# --8<-- [end:core_hours]


def rollout_hours(n=1):
    return n * rates()["rollout"]["s"] / 3600


def machine():
    m = rates()["machine"]
    return f"{m['cpu'].replace('12th Gen ', '').replace('(R)', '').replace('(TM)', '')}, JAX {m['jax']}, float64"
