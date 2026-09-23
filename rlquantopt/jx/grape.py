"""GRAPE on the JAX simulator: gradient descent on J_T through the exact propagators.

The amplitude bound is enforced by construction, u = u_max * tanh(theta), so every
iterate is a valid pulse. We minimise log10(J_T) with Adam (cosine-decayed step); restarts are vmapped.
Weyl coordinates are used without rounding (see metrics.c1c2c3).

Used for the quantum-speed-limit curve of paper Fig. 3: the smallest J_T reachable
at each gate time T for several amplitude limits.
"""
from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp
import optax

from rlquantopt.jx import physics, metrics


@dataclass(frozen=True)
class GrapeConfig:
    dt: float = 0.05
    n_iter: int = 1500
    lr: float = 0.05            # cosine-decayed to lr * lr_final_frac
    lr_final_frac: float = 0.02
    concurrence_weight: float = 1.0
    unitarity_weight: float = 3.0


def final_cost(u, ham, cfg: GrapeConfig):
    sector = physics.propagate(ham, physics.initial_state(), u, cfg.dt)
    JT, C, U = metrics.cost_JT(physics.realised_gate(sector), cfg.concurrence_weight, cfg.unitarity_weight,
                               digits=None)
    return JT, (C, U)


def random_guesses(key, n, n_samples, u_max, dt, detuning_ghz=0.8588):
    """Smooth random guesses: a few low-frequency components plus one near the qubit-qubit detuning."""
    t = jnp.arange(n_samples) * dt
    T = n_samples * dt
    k1, k2, k3 = jax.random.split(key, 3)
    n_modes = 4
    amps = jax.random.normal(k1, (n, n_modes)) * 0.3
    phases = jax.random.uniform(k2, (n, n_modes + 1)) * 2 * jnp.pi
    freqs = jnp.arange(1, n_modes + 1) / (2 * T)
    low = jnp.sum(amps[:, :, None] * jnp.sin(2 * jnp.pi * freqs[None, :, None] * t + phases[:, :n_modes, None]), 1)
    drive = jax.random.uniform(k3, (n, 1), minval=0.2, maxval=0.8) * jnp.sin(
        2 * jnp.pi * detuning_ghz * t + phases[:, n_modes:])
    envelope = jnp.sin(jnp.pi * (t + dt / 2) / T) ** 0.5
    return jnp.clip((low + drive) * envelope, -0.95, 0.95) * u_max


def optimise(u0, ham, u_max, cfg: GrapeConfig):
    """Optimise a batch of initial pulses u0 (n, n_samples). Returns final pulses, J_T, C, U and the J_T history."""
    theta0 = jnp.arctanh(jnp.clip(u0 / u_max, -0.99, 0.99))
    tx = optax.adam(optax.cosine_decay_schedule(cfg.lr, cfg.n_iter, cfg.lr_final_frac))

    def loss(theta):
        JT, aux = final_cost(u_max * jnp.tanh(theta), ham, cfg)
        return jnp.log10(jnp.maximum(JT, 1e-12)), (JT, aux)

    grad_fn = jax.value_and_grad(loss, has_aux=True)

    def run_one(theta):
        def step(carry, _):
            theta, opt_state, best = carry
            (l, (JT, _)), g = grad_fn(theta)
            g = jnp.nan_to_num(g)
            updates, opt_state = tx.update(g, opt_state)
            best_theta, best_JT = best
            better = JT < best_JT
            best = (jnp.where(better, theta, best_theta), jnp.where(better, JT, best_JT))
            return (optax.apply_updates(theta, updates), opt_state, best), JT

        init = (theta, tx.init(theta), (theta, jnp.inf))
        (_, _, (best_theta, _)), hist = jax.lax.scan(step, init, None, cfg.n_iter)
        u = u_max * jnp.tanh(best_theta)
        JT, (C, U) = final_cost(u, ham, cfg)
        return u, JT, C, U, hist

    return jax.jit(jax.vmap(run_one))(theta0)


def qsl_scan(Ts_ns, u_max_ghz, n_restarts=8, cfg: GrapeConfig = GrapeConfig(), model=physics.ModelParams(), seed=0):
    """Best J_T per (u_max, T). Returns dict with arrays of shape (len(u_max_ghz), len(Ts_ns))."""
    ham = physics.sector_hamiltonian(model)
    best = np.full((len(u_max_ghz), len(Ts_ns)), np.nan)
    key = jax.random.PRNGKey(seed)
    for i, a in enumerate(u_max_ghz):
        u_max = 2 * np.pi * a
        for j, T in enumerate(Ts_ns):
            n = int(round(T / cfg.dt))
            key, k = jax.random.split(key)
            u0 = random_guesses(k, n_restarts, n, u_max, cfg.dt)
            _, JT, *_ = optimise(u0, ham, u_max, cfg)
            best[i, j] = float(jnp.min(JT))
            print(f"u_max={a:.2f} GHz  T={T:5.1f} ns  best J_T={best[i, j]:.2e}", flush=True)
    return dict(T=np.asarray(Ts_ns), u_max_ghz=np.asarray(u_max_ghz), JT=best)
