"""Robustness of a fixed pulse to static qubit-frequency detuning (paper Figs. 10-12)."""
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp

from rlquantopt.jx import env as jenv, physics, metrics


def load_pulse_csv(path):
    """Read a v1/paper pulse CSV (columns tlist, amplist; amplitudes in rad/ns).

    Returns (amps, cfg) where amps[k] drives the k-th sample, following v1:
    amplist[0] is the value at t=0 and is not used.
    """
    df = pd.read_csv(path)
    t, u = df["tlist"].to_numpy(), df["amplist"].to_numpy()
    dt = float(t[1] - t[0])
    cfg = jenv.EnvConfig(pulse_length=len(t) - 1, T=dt * (len(t) - 1))
    return jnp.asarray(u[1:]), cfg


def final_JT(amps, cfg: jenv.EnvConfig, omega_s):
    ham = physics.sector_hamiltonian(cfg.model._replace(omega_s=omega_s))
    sector = physics.propagate(ham, physics.initial_state(), amps, cfg.dt)
    JT, C, U = metrics.cost_JT(physics.realised_gate(sector), cfg.concurrence_weight, cfg.unitarity_weight)
    return JT


def detuning_map(amps, cfg: jenv.EnvConfig, d_omega0_mhz, d_omega1_mhz, chunk=1024):
    """J_T at the end of the pulse on a grid of absolute detunings (MHz) of qubit 0 and qubit 1.

    Returns an array of shape (len(d_omega0_mhz), len(d_omega1_mhz)).
    """
    w0 = np.asarray(cfg.model.omega_s)
    g0, g1 = np.meshgrid(np.asarray(d_omega0_mhz), np.asarray(d_omega1_mhz), indexing="ij")
    omegas = np.stack([w0[0] + g0.ravel() * 1e-3, w0[1] + g1.ravel() * 1e-3], axis=-1)
    f = jax.jit(jax.vmap(lambda om: final_JT(amps, cfg, om)))
    out = np.concatenate([np.asarray(f(jnp.asarray(omegas[i:i + chunk]))) for i in range(0, len(omegas), chunk)])
    return out.reshape(g0.shape)
