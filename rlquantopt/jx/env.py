"""Functional, jit/vmap-friendly re-implementation of the v1 ZCQPEE environment.

Semantics follow rlquantopt.rl_envs.zc_qpee.ZCQPEE as used for the paper run
(05-12-24_201634): delta-mode actions, K samples per step, the concurrence and
unitarity reward with a TV penalty, and the out-of-bounds truncation penalty.
See docs/code/environment.md for the exact list.

    cfg = EnvConfig()
    obs, state = reset(key, cfg)
    obs, state, reward, terminated, truncated, info = step(state, action, cfg)
"""
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from rlquantopt.jx import physics, metrics
from rlquantopt.jx import device as device_mod
from rlquantopt.jx.config import fdtype


@dataclass(frozen=True)
class EnvConfig:
    pulse_length: int = 1000            # samples per episode
    T: float = 50.0                     # ns
    n_time_steps: int = 3               # samples per env step (K)
    a_scale: float = 20.0               # rad/ns per unit action; also the amplitude normaliser
    a_norm_max: float = 1.0             # |u| <= a_norm_max * a_scale
    tv_penalty_scale: float = 1e-3
    concurrence_weight: float = 1.0
    unitarity_weight: float = 3.0
    terminal_penalty: float = -20.0
    log_lim: float = 1e-2               # v1 subtracts -log10(1 - log_lim) from every reward
    rew_scale: float = 1.0
    obs_scale: float = 0.9
    jt_floor: float = 1e-12             # keeps -log10(J_T) finite for an exactly perfect gate
    # Domain randomisation of the qubit frequencies: omega_s *= 1 + U(-max_drift, max_drift)
    max_drift: float = 0.0
    # Further drifting parameters (absolute, MHz): coupler frequency omega_c_0 and couplings g_0, g_1
    coupler_drift_mhz: float = 0.0
    g_drift_mhz: float = 0.0
    # Idea i07: "pe" = the paper's perfect-entangler J_T; "sqrt_iswap" = 1 - average gate fidelity to
    # sqrt(iSWAP) after free virtual-Z corrections. info["JT"] always holds the active objective's cost.
    objective: str = "pe"
    # "amplitudes" = v1's 12 complex state amplitudes; "measured" = readout populations and Pauli
    # expectations a lab can measure (physics.measured_observables)
    obs_mode: str = "amplitudes"
    # Idea i09. "terminate" (v1): an amplitude beyond the bound ends the episode with terminal_penalty;
    # "clip": the amplitude is clipped, the episode goes on, and the reward loses oob_penalty per unit of
    # normalised excess, so the policy learns to stay inside the bound without losing the pulse.
    oob_mode: str = "terminate"
    oob_penalty: float = 1.0
    # Largest amplitude change per sample for a unit action, rad/ns; 0 = a_scale (v1). Scale it down with
    # longer steps (e.g. a_scale * 3 / K) or one step can sweep the whole range several times.
    delta_scale: float = 0.0
    # Measured observations as finite-shot estimates: N shots per measurement setting (0 = exact)
    shots: int = 0
    # obs_mode "context" / "measured+context": the policy also sees a calibration of the device, measured
    # once before the pulse (spectroscopy of both qubits and the coupler), as MHz offsets from nominal with
    # Gaussian measurement error context_noise_mhz, divided by context_scale_mhz
    context_noise_mhz: tuple = (0.1, 0.1, 1.0)
    context_scale_mhz: tuple = (5.7, 5.7, 140.0)
    context_bias_mhz: tuple = (0.0, 0.0, 0.0)     # a stale calibration: fixed error added to the context
    # "delta" (v1): each action is K per-sample amplitude increments. "carrier": the pulse is
    # u(t) = offset + A cos(2 pi f_d t + phi) and each action nudges the three slow knobs (A, phi, offset) by
    # up to carrier_steps = (rad/ns, rad, rad/ns); f_d is the nominal qubit-qubit detuning, corrected by the
    # measured calibration when the policy has one (context modes). act_dim is then 3 for any K.
    action_mode: str = "delta"
    carrier_steps: tuple = (1.0, 0.2, 1.0)
    # Idea i10: "rwa" = the paper's model (physics.py); "device" = the realistic tunable-coupler device
    # (device.py, env_device.py), where the control is the coupler flux (a_scale, carrier knobs in flux
    # quanta), drift is physical (qubit MHz, coupler flux offset in mPhi0, unmeasured couplings and
    # anharmonicities), and the waveform goes through an AWG hold (awg_dt ns) and a flux-line filter (filter_tau ns)
    physics_model: str = "rwa"
    device: device_mod.DeviceParams = device_mod.DeviceParams()
    qubit_drift_mhz: float = 0.0
    flux_drift_mphi0: float = 0.0
    eta_drift_mhz: float = 0.0
    awg_dt: float = 0.0
    filter_tau: float = 0.0
    device_simplified: bool = False
    model: physics.ModelParams = physics.ModelParams()

    @property
    def dt(self):
        return self.T / self.pulse_length

    @property
    def n_steps(self):
        """Episode length in env steps (333 for the paper configuration)."""
        K = self.n_time_steps
        return next(i for i in range(1, self.pulse_length + 1) if i * K + K - 1 >= self.pulse_length)

    @property
    def obs_dim(self):
        quantum = {"amplitudes": 24, "measured": physics.N_MEASURED, "context": 0,
                   "measured+context": physics.N_MEASURED}[self.obs_mode]
        control = 4 if self.action_mode == "carrier" else self.n_time_steps
        return quantum + (3 if "context" in self.obs_mode else 0) + control + 1

    @property
    def act_dim(self):
        return 3 if self.action_mode == "carrier" else self.n_time_steps

    @property
    def carrier_ghz(self):
        if self.physics_model == "device":
            from rlquantopt.jx import env_device
            return env_device.carrier_ghz(self)
        return abs(self.model.omega_s[1] - self.model.omega_s[0])

    @property
    def rew_thresh(self):
        return -jnp.log10(1 - self.log_lim) * self.rew_scale

    def __hash__(self):
        return hash(tuple((k, str(v)) for k, v in self.__dict__.items()))


class EnvState(NamedTuple):
    sector: physics.SectorState
    ham: physics.SectorHamiltonian
    amps_cur: jnp.ndarray       # (K,) last segment amplitudes, rad/ns
    cur_idx: jnp.ndarray        # sample index at the start of the next step
    omega_s: jnp.ndarray        # (2,) qubit frequencies of this episode, GHz
    key: jnp.ndarray            # PRNG key for measurement noise (used only with shots > 0)
    context: jnp.ndarray        # (3,) measured calibration: qubit and coupler frequency offsets, MHz
    knobs: jnp.ndarray          # (3,) carrier mode: amplitude A (rad/ns), phase phi (rad), offset (rad/ns)
    filt: jnp.ndarray           # device physics: flux-line filter state (last filtered sample)


def sample_omega_s(key, cfg: EnvConfig):
    omega = jnp.asarray(cfg.model.omega_s, fdtype())
    if cfg.max_drift == 0:
        return omega
    return omega * (1 + jax.random.uniform(key, omega.shape, fdtype(), -cfg.max_drift, cfg.max_drift))


def sample_model(key, cfg: EnvConfig):
    """Draw the episode's physical parameters: qubit frequencies, and optionally coupler frequency and couplings."""
    model = cfg.model._replace(omega_s=sample_omega_s(key, cfg))      # same key as before: runs stay reproducible
    if cfg.coupler_drift_mhz:
        d = jax.random.uniform(jax.random.fold_in(key, 1), (), fdtype(), -1, 1) * cfg.coupler_drift_mhz * 1e-3
        model = model._replace(omega_c_0=cfg.model.omega_c_0 + d)
    if cfg.g_drift_mhz:
        d = jax.random.uniform(jax.random.fold_in(key, 2), (2,), fdtype(), -1, 1) * cfg.g_drift_mhz * 1e-3
        model = model._replace(g=jnp.asarray(cfg.model.g, fdtype()) + d)
    return model


def reset_params(model: physics.ModelParams, cfg: EnvConfig, key=None):
    """Reset with fully specified physical parameters; ``key`` seeds the measurement noise."""
    if cfg.physics_model == "device":
        from rlquantopt.jx import env_device
        return env_device.reset(model, cfg, key)
    omega_s = jnp.asarray(model.omega_s, fdtype())
    k_obs, k_state, k_ctx = jax.random.split(jax.random.PRNGKey(0) if key is None else key, 3)
    offsets = jnp.concatenate([omega_s - jnp.asarray(cfg.model.omega_s, fdtype()),
                               jnp.atleast_1d(jnp.asarray(model.omega_c_0, fdtype()) - cfg.model.omega_c_0)]) * 1e3
    context = (offsets + jax.random.normal(k_ctx, (3,), fdtype()) * jnp.asarray(cfg.context_noise_mhz, fdtype())
               + jnp.asarray(cfg.context_bias_mhz, fdtype()))
    state = EnvState(sector=physics.initial_state(), ham=physics.sector_hamiltonian(model),
                     amps_cur=jnp.zeros(cfg.n_time_steps, fdtype()),
                     cur_idx=jnp.zeros((), jnp.int32), omega_s=omega_s, key=k_state, context=context,
                     knobs=jnp.zeros(3, fdtype()), filt=jnp.zeros((), fdtype()))
    return observation(state, state.cur_idx, cfg, k_obs), state


def reset_to(omega_s, cfg: EnvConfig):
    """Reset with given qubit frequencies, other parameters nominal (sweeps, fixed-per-env randomisation)."""
    if cfg.physics_model == "device":
        from rlquantopt.jx import env_device
        return env_device.reset(env_device.nominal_model(cfg, omega_s), cfg)
    return reset_params(cfg.model._replace(omega_s=omega_s), cfg)


def reset(key, cfg: EnvConfig):
    if cfg.physics_model == "device":
        from rlquantopt.jx import env_device
        return env_device.reset(env_device.sample(key, cfg), cfg, jax.random.fold_in(key, 3))
    return reset_params(sample_model(key, cfg), cfg, jax.random.fold_in(key, 3))


# --8<-- [start:shots]
def shot_estimates(key, pops, paulis, shots):
    """Finite-shot estimates of the measured observables, ``shots`` per measurement setting.

    Populations: a multinomial over the 5 readout outcomes of each input. Pauli expectations: each
    value from a multinomial over +1, -1 and "a transmon left the computational levels" (probability
    1 - <II>), so the estimate is (n+ - n-)/N. Values sharing a setting are sampled independently (their
    correlations are ignored). One step costs 3 + 3 x 9 = 30 settings, i.e. 30 N shots.
    """
    k1, k2 = jax.random.split(key)
    p = jnp.clip(pops.reshape(3, 5), 0, 1)
    n_pop = jax.random.multinomial(k1, shots, p / p.sum(-1, keepdims=True))
    ii = jnp.repeat(paulis.reshape(3, 16)[:, :1], 16, axis=1).reshape(-1)
    q = jnp.clip(jnp.stack([(ii + paulis) / 2, (ii - paulis) / 2, 1 - ii], -1), 0, 1)
    n_pauli = jax.random.multinomial(k2, shots, q / q.sum(-1, keepdims=True))
    return (n_pop / shots).reshape(-1), (n_pauli[:, 0] - n_pauli[:, 1]) / shots
# --8<-- [end:shots]


# --8<-- [start:observation]
def observation(state: EnvState, idx_for_time, cfg: EnvConfig, key=None):
    parts = []
    if cfg.obs_mode.startswith("measured"):
        pops, paulis = physics.measured_observables(state.sector)
        if cfg.shots:
            pops, paulis = shot_estimates(key, pops, paulis, cfg.shots)
        parts.append(jnp.concatenate([2 * pops - 1, paulis]))          # all in [-1, 1]
    elif cfg.obs_mode == "amplitudes":
        z = physics.sector_amplitudes(state.sector)
        parts.append(jnp.stack([2 * jnp.abs(z) - 1, jnp.angle(z) / jnp.pi], axis=-1).reshape(-1))
    if "context" in cfg.obs_mode:
        parts.append(state.context / jnp.asarray(cfg.context_scale_mhz, fdtype()))
    if cfg.action_mode == "carrier":
        A, phi, off = state.knobs
        amps = jnp.stack([A / cfg.a_scale, off / cfg.a_scale, jnp.cos(phi), jnp.sin(phi)])
    else:
        amps = state.amps_cur / cfg.a_scale / cfg.a_norm_max
    t = idx_for_time * 2 / cfg.pulse_length - 1
    obs = jnp.concatenate(parts + [amps, jnp.atleast_1d(t)]) * cfg.obs_scale
    return obs.astype(fdtype())
# --8<-- [end:observation]


# --8<-- [start:step]
def step(state: EnvState, action, cfg: EnvConfig):
    """One env step. ``action`` in [-1, 1]^K (clipped, as SB3 does) are amplitude deltas."""
    if cfg.physics_model == "device":
        from rlquantopt.jx import env_device
        return env_device.step(state, action, cfg)
    K = cfg.n_time_steps
    knobs = state.knobs
    if cfg.action_mode == "carrier":
        action = jnp.clip(jnp.reshape(action, (3,)), -1, 1).astype(fdtype())
        knobs = knobs + action * jnp.asarray(cfg.carrier_steps, fdtype())
        knobs = knobs.at[0].set(jnp.clip(knobs[0], 0.0, cfg.a_scale)).at[2].set(jnp.clip(knobs[2], -cfg.a_scale, cfg.a_scale))
        f = cfg.carrier_ghz
        if "context" in cfg.obs_mode:       # the measured detuning, not the true one
            f = f + jnp.sign(cfg.model.omega_s[1] - cfg.model.omega_s[0]) * (state.context[1] - state.context[0]) * 1e-3
        t = (state.cur_idx + jnp.arange(K) + 0.5) * cfg.dt
        amps = knobs[2] + knobs[0] * jnp.cos(2 * jnp.pi * f * t + knobs[1])
    else:
        action = jnp.clip(jnp.reshape(action, (K,)), -1, 1).astype(fdtype())
        deltas = action * (cfg.delta_scale or cfg.a_scale)
        prev = jnp.where(state.cur_idx == 0, 0.0, state.amps_cur[-1])
        amps = prev + jnp.cumsum(deltas)
    amps_norm = amps / cfg.a_scale
    oob = jnp.max(jnp.abs(amps_norm)) > cfg.a_norm_max
    excess = jnp.sum(jnp.maximum(jnp.abs(amps_norm) - cfg.a_norm_max, 0.0))
    amps = jnp.clip(amps_norm, -cfg.a_norm_max, cfg.a_norm_max) * cfg.a_scale

    sector = physics.propagate(state.ham, state.sector, amps, cfg.dt)
    G = physics.realised_gate(sector)
    JT, C, U = metrics.cost(G, cfg.objective, cfg.concurrence_weight, cfg.unitarity_weight)
    reward = -jnp.log10(jnp.maximum(JT, cfg.jt_floor)) * cfg.rew_scale - cfg.rew_thresh
    tv = jnp.sum(jnp.abs(jnp.diff(amps))) * cfg.tv_penalty_scale
    reward = reward - tv
    if cfg.oob_mode == "clip":          # keep the pulse, penalise pushing against the bound
        reward = (reward - cfg.oob_penalty * excess).astype(fdtype())
    else:                               # v1: leaving the bound ends the episode
        oob_reward = cfg.terminal_penalty * (1 - state.cur_idx / cfg.pulse_length)
        reward = jnp.where(oob, oob_reward, reward).astype(fdtype())

    k_obs, k_next = jax.random.split(state.key)
    new_state = state._replace(sector=sector, amps_cur=amps, key=k_next, knobs=knobs)
    obs = observation(new_state, state.cur_idx, cfg, k_obs)
    cur_idx = state.cur_idx + K
    new_state = new_state._replace(cur_idx=cur_idx)
    terminated = cur_idx + K - 1 >= cfg.pulse_length
    truncated = oob if cfg.oob_mode == "terminate" else jnp.array(False)
    info = dict(JT=JT, concurrence=C, unitarity=U, tv_penalty=tv, oob=oob,
                t=cur_idx * cfg.dt)
    return obs, new_state, reward, terminated, truncated, info
# --8<-- [end:step]


# --8<-- [start:step_autoreset]
def step_autoreset(key, state: EnvState, action, cfg: EnvConfig, resample=True):
    """Step, and reset when the episode ends. Returns the pre-reset obs as ``info['final_obs']``.

    With ``resample=False`` the episode restarts on the same frequencies, which
    mimics v1 ZCQPEEWRD (drift drawn once per env instance).
    """
    obs, new_state, reward, terminated, truncated, info = step(state, action, cfg)
    done = terminated | truncated
    obs_reset, state_reset = reset(key, cfg) if resample else reset_to(state.omega_s, cfg)
    state_out = jax.tree_util.tree_map(lambda r, s: jnp.where(done, r, s), state_reset, new_state)
    obs_out = jnp.where(done, obs_reset, obs)
    info = dict(info, final_obs=obs)
    return obs_out, state_out, reward, terminated, truncated, info
# --8<-- [end:step_autoreset]


def rollout_pulse(amps, cfg: EnvConfig, omega_s=None):
    """Evaluate a fixed pulse (one amplitude per sample, rad/ns) and return per-step metrics.

    ``amps[k]`` is held on (t_k, t_{k+1}], matching v1 where the pulse CSV's
    ``amplist[k + 1]`` drives the k-th sample.
    """
    omega_s = cfg.model.omega_s if omega_s is None else omega_s
    ham = physics.sector_hamiltonian(cfg.model._replace(omega_s=jnp.asarray(omega_s)))

    def body(sector, u):
        sector = physics.propagate(ham, sector, u[None], cfg.dt)
        JT, C, U = metrics.cost(physics.realised_gate(sector), cfg.objective, cfg.concurrence_weight,
                                cfg.unitarity_weight)
        return sector, (JT, C, U)

    _, (JT, C, U) = jax.lax.scan(body, physics.initial_state(), jnp.asarray(amps))
    return dict(JT=JT, concurrence=C, unitarity=U)
