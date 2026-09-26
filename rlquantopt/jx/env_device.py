"""The gate environment on the realistic device (idea i10), used by env.py when cfg.physics_model == "device".

Same interface and reward as env.py; what changes:
- the physics is rlquantopt/jx/device.py (no RWA, direct coupling, flux-tunable coupler), parameters from
  Sung et al. 2021;
- the control is the coupler flux offset from its idle point, in flux quanta, driven by the carrier knobs
  (action_mode "carrier"): x(t) = offset + A cos(2 pi f_d t + phi), held for awg_dt ns per AWG sample and
  low-pass filtered with time constant filter_tau ns (a first-order model of the flux line);
- drift is physical: qubit frequencies (qubit_drift_mhz), the coupler's flux offset (flux_drift_mphi0; Dai
  et al. 2021 measured up to 20 mPhi0 after 17 days), and optionally the couplings and anharmonicities,
  which the calibration does not measure (g_drift_mhz, eta_drift_mhz);
- device_simplified: the model one would write down first (RWA, no direct coupling, linear flux curve, ideal
  AWG and flux line), for the sim-to-sim transfer test: train on it, deploy on the full model;
- the calibration (context) is what spectroscopy measures: the dressed qubit frequencies and the coupler's
  idle frequency, as offsets from nominal in MHz; the carrier frequency is the measured dressed detuning.
"""
import functools
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp

from rlquantopt.jx import device as dv, metrics
from rlquantopt.jx.config import fdtype

TWO_PI = 2 * np.pi


class DeviceModel(NamedTuple):
    params: dv.DeviceParams
    phi_offset: jnp.ndarray         # flux drift, flux quanta


@functools.lru_cache(maxsize=16)
def nominal(p: dv.DeviceParams):
    """Idle flux and the dressed qubit frequencies (GHz) of the nominal device (constants, even inside jit)."""
    with jax.ensure_compile_time_eval():
        phi = dv.idle_flux(p)
        (_, ee), (_, eo) = dv.dressed_basis(dv.block_hamiltonian(p), phi)
        ee, eo = np.asarray(ee), np.asarray(eo)
    return phi, float((eo[1] - ee[0]) / TWO_PI), float((eo[0] - ee[0]) / TWO_PI)


@functools.lru_cache(maxsize=16)
def _lin(p: dv.DeviceParams):
    with jax.ensure_compile_time_eval():
        return dv.linearisation(p)


def carrier_ghz(cfg):
    _, w1, w2 = nominal(cfg.device)
    return abs(w1 - w2)


def sample(key, cfg):
    p = cfg.device
    u = lambda i, n=None: jax.random.uniform(jax.random.fold_in(key, i), () if n is None else (n,), fdtype(), -1, 1)
    q = cfg.qubit_drift_mhz * 1e-3
    p = p._replace(omega_1=p.omega_1 + q * u(0), omega_2=p.omega_2 + q * u(1))
    if cfg.g_drift_mhz:
        g = cfg.g_drift_mhz * 1e-3 * u(2, 2)
        p = p._replace(g_1c=p.g_1c + g[0], g_2c=p.g_2c + g[1])
    if cfg.eta_drift_mhz:
        e = cfg.eta_drift_mhz * 1e-3 * u(3, 3)
        p = p._replace(eta_1=p.eta_1 + e[0], eta_2=p.eta_2 + e[1], eta_c=p.eta_c + e[2])
    return DeviceModel(p, cfg.flux_drift_mphi0 * 1e-3 * u(4))


def nominal_model(cfg, omega=None):
    p = cfg.device if omega is None else cfg.device._replace(omega_1=omega[0], omega_2=omega[1])
    return DeviceModel(p, jnp.zeros((), fdtype()))


def reset(model: DeviceModel, cfg, key=None):
    from rlquantopt.jx import env as jenv
    assert cfg.obs_mode == "context" and cfg.action_mode == "carrier", "device physics: context obs, carrier actions"
    phi0, w1n, w2n = nominal(cfg.device)
    k_obs, k_state, k_ctx = jax.random.split(jax.random.PRNGKey(0) if key is None else key, 3)
    h = dv.block_hamiltonian(model.params, cfg.device_simplified, _lin(cfg.device) if cfg.device_simplified else None)
    phi_idle = phi0 + model.phi_offset
    basis = dv.dressed_basis(h, phi_idle)
    (_, ee), (_, eo) = basis
    w1, w2 = (eo[1] - ee[0]) / TWO_PI, (eo[0] - ee[0]) / TWO_PI        # dressed, as spectroscopy sees them
    wc = dv.coupler_frequency(phi_idle, model.params)
    offsets = jnp.stack([w1 - w1n, w2 - w2n, wc - cfg.device.omega_c_idle]) * 1e3
    context = (offsets + jax.random.normal(k_ctx, (3,), fdtype()) * jnp.asarray(cfg.context_noise_mhz, fdtype())
               + jnp.asarray(cfg.context_bias_mhz, fdtype()))
    state = jenv.EnvState(sector=dv.initial_state(basis), ham=(h, basis, phi_idle),
                          amps_cur=jnp.zeros(cfg.n_time_steps, fdtype()), cur_idx=jnp.zeros((), jnp.int32),
                          omega_s=jnp.stack([model.params.omega_1, model.params.omega_2]).astype(fdtype()),
                          key=k_state, context=context, knobs=jnp.zeros(3, fdtype()),
                          filt=jnp.zeros((), fdtype()))
    return jenv.observation(state, state.cur_idx, cfg, k_obs), state


def step(state, action, cfg):
    from rlquantopt.jx import env as jenv
    K = cfg.n_time_steps
    h, basis, phi_idle = state.ham
    action = jnp.clip(jnp.reshape(action, (3,)), -1, 1).astype(fdtype())
    knobs = state.knobs + action * jnp.asarray(cfg.carrier_steps, fdtype())
    knobs = knobs.at[0].set(jnp.clip(knobs[0], 0.0, cfg.a_scale)).at[2].set(jnp.clip(knobs[2], -cfg.a_scale, cfg.a_scale))
    f = carrier_ghz(cfg) + (state.context[0] - state.context[1]) * 1e-3       # the measured dressed detuning
    t = (state.cur_idx + jnp.arange(K) + 0.5) * cfg.dt
    if cfg.awg_dt and not cfg.device_simplified:
        t = (jnp.floor(t / cfg.awg_dt) + 0.5) * cfg.awg_dt                  # AWG sample-and-hold
    x = knobs[2] + knobs[0] * jnp.cos(TWO_PI * f * t + knobs[1])
    x_norm = x / cfg.a_scale
    oob = jnp.max(jnp.abs(x_norm)) > cfg.a_norm_max
    excess = jnp.sum(jnp.maximum(jnp.abs(x_norm) - cfg.a_norm_max, 0.0))
    x = jnp.clip(x_norm, -cfg.a_norm_max, cfg.a_norm_max) * cfg.a_scale
    if cfg.filter_tau and not cfg.device_simplified:                        # first-order flux-line filter
        alpha = 1 - np.exp(-cfg.dt / cfg.filter_tau)
        _, amps = jax.lax.scan(lambda y, xk: ((y + alpha * (xk - y),) * 2), state.filt, x)
    else:
        amps = x

    sector = dv.propagate(h, state.sector, phi_idle + amps, cfg.dt)
    G = dv.gate(sector, basis, (state.cur_idx + K) * cfg.dt)
    JT, C, U = metrics.cost(G.conj().T, cfg.objective, cfg.concurrence_weight, cfg.unitarity_weight)
    reward = -jnp.log10(jnp.maximum(JT, cfg.jt_floor)) * cfg.rew_scale - cfg.rew_thresh
    tv = jnp.sum(jnp.abs(jnp.diff(amps))) / cfg.a_scale * cfg.tv_penalty_scale
    reward = (reward - tv - cfg.oob_penalty * excess).astype(fdtype())

    k_obs, k_next = jax.random.split(state.key)
    new_state = state._replace(sector=sector, amps_cur=amps, key=k_next, knobs=knobs, filt=amps[-1])
    obs = jenv.observation(new_state, state.cur_idx, cfg, k_obs)
    cur_idx = state.cur_idx + K
    new_state = new_state._replace(cur_idx=cur_idx)
    terminated = cur_idx + K - 1 >= cfg.pulse_length
    info = dict(JT=JT, concurrence=C, unitarity=U, tv_penalty=tv, oob=oob, t=cur_idx * cfg.dt)
    return obs, new_state, reward, terminated, jnp.array(False), info
