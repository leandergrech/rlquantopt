"""Model-free pulse refinement from measured data only (idea i09, ibis).

The optimisers see nothing but noisy fidelity estimates: the device (here the simulator, standing in
for hardware) plays a pulse, the 30 measurement settings are read out with N shots each, and the
fidelity to sqrt(iSWAP) with free Z is reconstructed from those counts (metrics.fidelity_from_observables).
Every evaluation costs 30 N shots and is counted. The pulse is a start pulse (a policy's, or a random
guess) plus a smooth correction in a sine basis, so both optimisers work in n_basis dimensions. Each basis
function is normalised to 1 rad of integrated control area, the natural unit here: the gate responds to
the accumulated phase, so a warm start needs steps of ~0.05-0.1 rad and a random start steps of ~1 rad.

- SPSA (Spall 1992): two evaluations per iteration at x +- c_k Delta (Delta random +-1 in every coordinate)
  give a gradient estimate along all coordinates at once; gains a_k = a / (k + 1 + A)^0.602,
  c_k = c / (k + 1)^0.101, a calibrated from the first gradient estimates so that the first steps move
  the coefficients by about `first_step`.
- CMA-ES (Hansen), pycma: a Gaussian search distribution whose mean moves toward the best of each
  generation and whose covariance adapts to the landscape.
"""
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp

from rlquantopt.jx import env as jenv, metrics, physics

SETTINGS = 30


class Trace(NamedTuple):
    shots: np.ndarray       # cumulative shots after each evaluation
    true_JT: np.ndarray     # exact 1 - F of the current iterate (for scoring only; the optimiser never sees it)


def sine_basis(n_samples, n_basis, dt=0.05):
    """Sine modes, zero at both ends, each scaled to 1 rad of integrated area (sum |phi| dt = 1)."""
    t = (np.arange(n_samples) + 0.5) / n_samples
    B = np.stack([np.sin(np.pi * (k + 1) * t) for k in range(n_basis)])
    return B / (np.abs(B).sum(1, keepdims=True) * dt)


def knob_basis(u0, n_sine, dt=0.05):
    """Physical knobs first: a constant offset (a coupler-frequency drift is exactly this), the scale of the
    pulse's oscillating part, a linear ramp; then n_sine sine modes. All normalised to 1 rad of area."""
    n = len(u0)
    t = (np.arange(n) + 0.5) / n
    ac = u0 - u0.mean()
    B = np.stack([np.ones(n), ac, t - 0.5] + list(sine_basis(n, n_sine, dt)))
    return B / (np.abs(B).sum(1, keepdims=True) * dt)


def make_device(model: physics.ModelParams, u0, n_basis, shots, dt=0.05, u_max=20.0, basis="sine"):
    """Returns measure(x, key) -> estimated 1 - F and exact(x) -> true 1 - F for pulse u0 + x @ basis.
    basis "sine": n_basis sine modes; "knobs": 3 physical knobs + (n_basis - 3) sine modes."""
    ham = physics.sector_hamiltonian(model)
    u0 = np.asarray(u0)
    B = jnp.asarray(sine_basis(len(u0), n_basis, dt) if basis == "sine" else knob_basis(u0, n_basis - 3, dt))
    u0 = jnp.asarray(u0)
    pulse = lambda x: jnp.clip(u0 + x @ B, -u_max, u_max)
    final = lambda x: physics.propagate(ham, physics.initial_state(), pulse(x), dt)

    @jax.jit
    def measure(x, key):
        pops, paulis = physics.measured_observables(final(x))
        if shots:
            pops, paulis = jenv.shot_estimates(key, pops, paulis, shots)
        return 1 - metrics.fidelity_from_observables(pops, paulis)

    @jax.jit
    def exact(x):
        return 1 - metrics.fidelity_free_z(physics.realised_gate(final(x)))[0]

    return measure, exact, pulse


# --8<-- [start:spsa]
def spsa(measure, exact, n_basis, shots, max_shots, seed=0, c=0.05, first_step=0.05, A=20, calib=10):
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    x = np.zeros(n_basis)
    used, tr_s, tr_j = 0, [], []

    def f(x):
        nonlocal key, used
        key, k = jax.random.split(key)
        used += SETTINGS * shots
        return float(measure(jnp.asarray(x), k))

    g = [abs(f(x + c * d) - f(x - c * d)) / (2 * c) for d in rng.choice([-1.0, 1.0], (calib, n_basis))]
    a = first_step * (A + 1) ** 0.602 / max(np.mean(g), 1e-9)
    k = 0
    while used + 2 * SETTINGS * shots <= max_shots:
        ak, ck = a / (k + 1 + A) ** 0.602, c / (k + 1) ** 0.101
        d = rng.choice([-1.0, 1.0], n_basis)
        grad = (f(x + ck * d) - f(x - ck * d)) / (2 * ck) * d
        x = x - ak * grad
        tr_s.append(used)
        tr_j.append(float(exact(jnp.asarray(x))))
        k += 1
    return x, Trace(np.asarray(tr_s), np.asarray(tr_j))
# --8<-- [end:spsa]


def cmaes(measure, exact, n_basis, shots, max_shots, seed=0, sigma0=0.1):
    import cma
    key = jax.random.PRNGKey(seed)
    es = cma.CMAEvolutionStrategy(np.zeros(n_basis), sigma0, {"seed": seed + 1, "verbose": -9})
    used, tr_s, tr_j = 0, [], []
    while used + es.popsize * SETTINGS * shots <= max_shots:
        X = es.ask()
        keys = jax.random.split(key, len(X) + 1)
        key = keys[0]
        es.tell(X, [float(measure(jnp.asarray(xx), kk)) for xx, kk in zip(X, keys[1:])])
        used += len(X) * SETTINGS * shots
        tr_s.append(used)
        tr_j.append(float(exact(jnp.asarray(es.mean))))
    return es.mean, Trace(np.asarray(tr_s), np.asarray(tr_j))
