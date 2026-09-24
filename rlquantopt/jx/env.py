"""Functional, jit/vmap-friendly re-implementation of the v1 ZCQPEE environment.

Semantics follow rlquantopt.rl_envs.zc_qpee.ZCQPEE as used for the paper run
(05-12-24_201634): delta-mode actions, K samples per step, the concurrence and
unitarity reward with a TV penalty, and the out-of-bounds truncation penalty.
See docs/v2_jax_plan.md for the exact list.

    cfg = EnvConfig()
    obs, state = reset(key, cfg)
    obs, state, reward, terminated, truncated, info = step(state, action, cfg)
"""
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from rlquantopt.jx import physics, metrics
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
        return 24 + self.n_time_steps + 1

    @property
    def act_dim(self):
        return self.n_time_steps

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


def sample_omega_s(key, cfg: EnvConfig):
    omega = jnp.asarray(cfg.model.omega_s, fdtype())
    if cfg.max_drift == 0:
        return omega
    return omega * (1 + jax.random.uniform(key, omega.shape, fdtype(), -cfg.max_drift, cfg.max_drift))


def reset_to(omega_s, cfg: EnvConfig):
    """Reset with given qubit frequencies (used for sweeps and fixed-per-env randomisation)."""
    ham = physics.sector_hamiltonian(cfg.model._replace(omega_s=omega_s))
    state = EnvState(sector=physics.initial_state(), ham=ham,
                     amps_cur=jnp.zeros(cfg.n_time_steps, fdtype()),
                     cur_idx=jnp.zeros((), jnp.int32), omega_s=omega_s)
    return observation(state, state.cur_idx, cfg), state


def reset(key, cfg: EnvConfig):
    return reset_to(sample_omega_s(key, cfg), cfg)


# --8<-- [start:observation]
def observation(state: EnvState, idx_for_time, cfg: EnvConfig):
    z = physics.sector_amplitudes(state.sector)
    polar = jnp.stack([2 * jnp.abs(z) - 1, jnp.angle(z) / jnp.pi], axis=-1).reshape(-1)
    amps = state.amps_cur / cfg.a_scale / cfg.a_norm_max
    t = idx_for_time * 2 / cfg.pulse_length - 1
    obs = jnp.concatenate([polar, amps, jnp.atleast_1d(t)]) * cfg.obs_scale
    return obs.astype(fdtype())
# --8<-- [end:observation]


# --8<-- [start:step]
def step(state: EnvState, action, cfg: EnvConfig):
    """One env step. ``action`` in [-1, 1]^K (clipped, as SB3 does) are amplitude deltas."""
    K = cfg.n_time_steps
    action = jnp.clip(jnp.reshape(action, (K,)), -1, 1).astype(fdtype())
    deltas = action * cfg.a_scale
    prev = jnp.where(state.cur_idx == 0, 0.0, state.amps_cur[-1])
    amps = prev + jnp.cumsum(deltas)
    amps_norm = amps / cfg.a_scale
    oob = jnp.max(jnp.abs(amps_norm)) > cfg.a_norm_max
    amps = jnp.clip(amps_norm, -cfg.a_norm_max, cfg.a_norm_max) * cfg.a_scale

    sector = physics.propagate(state.ham, state.sector, amps, cfg.dt)
    G = physics.realised_gate(sector)
    JT, C, U = metrics.cost_JT(G, cfg.concurrence_weight, cfg.unitarity_weight)
    reward = -jnp.log10(jnp.maximum(JT, cfg.jt_floor)) * cfg.rew_scale - cfg.rew_thresh
    tv = jnp.sum(jnp.abs(jnp.diff(amps))) * cfg.tv_penalty_scale
    reward = reward - tv
    oob_reward = cfg.terminal_penalty * (1 - state.cur_idx / cfg.pulse_length)
    reward = jnp.where(oob, oob_reward, reward).astype(fdtype())

    new_state = state._replace(sector=sector, amps_cur=amps)
    obs = observation(new_state, state.cur_idx, cfg)
    cur_idx = state.cur_idx + K
    new_state = new_state._replace(cur_idx=cur_idx)
    terminated = cur_idx + K - 1 >= cfg.pulse_length
    truncated = oob
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
    omega = jnp.where(resample, sample_omega_s(key, cfg), state.omega_s)
    obs_reset, state_reset = reset_to(omega, cfg)
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
        JT, C, U = metrics.cost_JT(physics.realised_gate(sector), cfg.concurrence_weight, cfg.unitarity_weight)
        return sector, (JT, C, U)

    _, (JT, C, U) = jax.lax.scan(body, physics.initial_state(), jnp.asarray(amps))
    return dict(JT=JT, concurrence=C, unitarity=U)
