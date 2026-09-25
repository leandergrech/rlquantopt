"""Lightweight pure-JAX environments for smoke-testing RL ideas in isolation (idea i05 echidna), plus an adapter
that gives the gate environment (env.py) the same interface.

Each env is a frozen (hashable, so jit-static) dataclass with ``obs_dim``, ``act_dim``, ``max_steps``,
``reset(key) -> (obs, state)`` and ``step(state, action) -> (obs, state, reward, terminated, truncated, info)``.
Actions are in [-1, 1]^act_dim (clipped by the env).

- ``Pendulum``: gym Pendulum-v1 (dense reward; a sanity check).
- ``MountainCar``: gym MountainCarContinuous-v0 (sparse +100 behind a hill; the classic exploration trap).
- ``Tracker``: a miniature of the gate problem: K delta actions per step on an amplitude bounded to [-1, 1]
  (overshooting ends the episode with a penalty that shrinks over time), rewarded with
  -log10(tracking error) against a sinusoidal target pulse of random phase.
- ``Ham``: the ZCQPEE gate environment of env.py behind the same interface.
"""
from dataclasses import dataclass, field
from typing import NamedTuple

import jax
import jax.numpy as jnp

from rlquantopt.jx import env as jenv


class TState(NamedTuple):
    x: jnp.ndarray
    t: jnp.ndarray


@dataclass(frozen=True)
class Pendulum:
    max_steps: int = 200
    obs_dim: int = 3
    act_dim: int = 1

    def reset(self, key):
        k1, k2 = jax.random.split(key)
        x = jnp.stack([jax.random.uniform(k1, (), jnp.float32, -jnp.pi, jnp.pi),
                       jax.random.uniform(k2, (), jnp.float32, -1, 1)])
        return self._obs(x), TState(x, jnp.zeros((), jnp.int32))

    def _obs(self, x):
        return jnp.stack([jnp.cos(x[0]), jnp.sin(x[0]), x[1] / 8.0])

    def step(self, s: TState, action):
        th, thdot = s.x
        u = 2.0 * jnp.clip(action[0], -1, 1)
        ang = ((th + jnp.pi) % (2 * jnp.pi)) - jnp.pi
        cost = ang ** 2 + 0.1 * thdot ** 2 + 0.001 * u ** 2
        thdot = jnp.clip(thdot + (3 * 10.0 / 2 * jnp.sin(th) + 3.0 * u) * 0.05, -8, 8)
        x = jnp.stack([th + thdot * 0.05, thdot]).astype(jnp.float32)
        t = s.t + 1
        return self._obs(x), TState(x, t), -cost, jnp.array(False), t >= self.max_steps, dict(score=-cost)


@dataclass(frozen=True)
class MountainCar:
    max_steps: int = 999
    obs_dim: int = 2
    act_dim: int = 1

    def reset(self, key):
        x = jnp.stack([jax.random.uniform(key, (), jnp.float32, -0.6, -0.4), jnp.zeros((), jnp.float32)])
        return self._obs(x), TState(x, jnp.zeros((), jnp.int32))

    def _obs(self, x):
        return jnp.stack([(x[0] + 0.3) / 0.9, x[1] / 0.07])

    def step(self, s: TState, action):
        pos, vel = s.x
        f = jnp.clip(action[0], -1, 1)
        vel = jnp.clip(vel + f * 0.0015 - 0.0025 * jnp.cos(3 * pos), -0.07, 0.07)
        pos = jnp.clip(pos + vel, -1.2, 0.6)
        vel = jnp.where((pos <= -1.2) & (vel < 0), 0.0, vel)
        goal = (pos >= 0.45) & (vel >= 0)
        reward = -0.1 * f ** 2 + 100.0 * goal
        x = jnp.stack([pos, vel]).astype(jnp.float32)
        t = s.t + 1
        return self._obs(x), TState(x, t), reward, goal, (t >= self.max_steps) & ~goal, dict(score=goal.astype(jnp.float32))


class TrackerState(NamedTuple):
    u: jnp.ndarray          # amplitude after the last sample
    phase: jnp.ndarray
    t: jnp.ndarray          # env step


@dataclass(frozen=True)
class Tracker:
    """K samples per step: u_k = u + cumsum(delta_scale * a); target f(i) = 0.8 sin(4 pi i / N + phase)."""
    max_steps: int = 100
    K: int = 3
    delta_scale: float = 0.1
    terminal_penalty: float = -20.0
    # Device drift (idea i06 fox): an unknown gain on the actions and an unknown offset on the output,
    # which the observation does not show (it reports the commanded amplitude)
    gain: float = 1.0
    offset: float = 0.0
    # Overshooting as a true terminal. False (i05) mirrors env.py/v1: it is a truncation, so PPO bootstraps
    # gamma * V(final) onto the penalty; with a critic that overestimates V (e.g. after device drift) that
    # can make overshooting pay, and fine-tuning collapses into it.
    oob_terminal: bool = False
    # Reward per step = -log10(error) + reward_shift. The i05 value (-2) makes the reward negative once the
    # error exceeds 1%, so on a badly drifted device ending the episode early (the -20 penalty) beats staying
    # alive. +1 keeps it positive for any error below 10, as in the gate env (-log10 J_T is > 0 for J_T < 1).
    reward_shift: float = -2.0

    @property
    def obs_dim(self):
        return 2 * self.K + 3

    @property
    def act_dim(self):
        return self.K

    def target(self, idx, phase):
        return 0.8 * jnp.sin(4 * jnp.pi * idx / (self.max_steps * self.K) + phase)

    def _obs(self, s: TrackerState):
        idx = s.t * self.K + jnp.arange(1, 2 * self.K + 1)          # the next two steps' targets
        return jnp.concatenate([self.target(idx, s.phase), jnp.stack([s.u, 2 * s.t / self.max_steps - 1,
                                                                       jnp.sin(s.phase)])]).astype(jnp.float32)

    def reset(self, key):
        s = TrackerState(jnp.zeros((), jnp.float32), jax.random.uniform(key, (), jnp.float32, 0, 2 * jnp.pi),
                         jnp.zeros((), jnp.int32))
        return self._obs(s), s

    def step(self, s: TrackerState, action):
        u = s.u + jnp.cumsum(self.gain * self.delta_scale * jnp.clip(action, -1, 1))
        out = u + self.offset                                          # what the device actually produces
        oob = jnp.max(jnp.abs(out)) > 1
        err = jnp.mean((out - self.target(s.t * self.K + jnp.arange(1, self.K + 1), s.phase)) ** 2)
        score = -jnp.log10(err + 1e-8)
        reward = jnp.where(oob, self.terminal_penalty * (1 - s.t / self.max_steps), score + self.reward_shift)
        t = s.t + 1
        ns = TrackerState(jnp.clip(u[-1], -1, 1).astype(jnp.float32), s.phase, t)
        term = oob & self.oob_terminal
        return self._obs(ns), ns, reward, term, (oob | (t >= self.max_steps)) & ~term, dict(score=score)


class DriftState(NamedTuple):
    u: jnp.ndarray
    phase: jnp.ndarray
    t: jnp.ndarray
    gain: jnp.ndarray       # this episode's device
    offset: jnp.ndarray
    zhat: jnp.ndarray       # (2,) the drift estimate shown to the policy


@dataclass(frozen=True)
class DriftTracker:
    """Tracker whose device (gain, offset) is drawn per episode (idea i08 hippogriff).

    The observation is the Tracker's plus ``zhat``, a 2-vector in [-1, 1]^2 that stands for the device:
    during training it is the true normalised (gain, offset) (``show_z``) or zeros (a blind, robust policy);
    at deployment the caller writes its belief into ``state.zhat``. Overshooting is terminal and the reward
    positive, as in the fox benchmark. ``outcome`` is shared by ``step`` and the measurement functions, so a
    filter that uses ``measure`` has the environment's exact model.
    """
    max_steps: int = 100
    K: int = 3
    delta_scale: float = 0.1
    terminal_penalty: float = -20.0
    reward_shift: float = 1.0
    gain_range: tuple = (0.6, 1.4)
    offset_range: tuple = (-0.15, 0.15)
    show_z: bool = True

    @property
    def obs_dim(self):
        return 2 * self.K + 5

    @property
    def act_dim(self):
        return self.K

    def z_of(self, gain, offset):
        (g0, g1), (o0, o1) = self.gain_range, self.offset_range
        return jnp.stack([2 * (gain - g0) / (g1 - g0) - 1, 2 * (offset - o0) / (o1 - o0) - 1])

    def params_of(self, z):
        (g0, g1), (o0, o1) = self.gain_range, self.offset_range
        return g0 + (z[0] + 1) * (g1 - g0) / 2, o0 + (z[1] + 1) * (o1 - o0) / 2

    def target(self, idx, phase):
        return 0.8 * jnp.sin(4 * jnp.pi * idx / (self.max_steps * self.K) + phase)

    def obs_base(self, s: DriftState):
        idx = s.t * self.K + jnp.arange(1, 2 * self.K + 1)
        return jnp.concatenate([self.target(idx, s.phase), jnp.stack([s.u, 2 * s.t / self.max_steps - 1,
                                                                       jnp.sin(s.phase)])]).astype(jnp.float32)

    def obs(self, s: DriftState):
        z = s.zhat if self.show_z else jnp.zeros_like(s.zhat)
        return jnp.concatenate([self.obs_base(s), z.astype(jnp.float32)])

    def reset_with(self, key, gain, offset, zhat=None):
        gain, offset = jnp.float32(gain), jnp.float32(offset)
        s = DriftState(jnp.zeros((), jnp.float32), jax.random.uniform(key, (), jnp.float32, 0, 2 * jnp.pi),
                       jnp.zeros((), jnp.int32), gain, offset,
                       (self.z_of(gain, offset) if zhat is None else zhat).astype(jnp.float32))
        return self.obs(s), s

    def reset(self, key):
        k1, k2, k3 = jax.random.split(key, 3)
        return self.reset_with(k3, jax.random.uniform(k1, (), jnp.float32, *self.gain_range),
                               jax.random.uniform(k2, (), jnp.float32, *self.offset_range))

    def outcome(self, s: DriftState, action, gain, offset):
        """Commanded amplitudes u_k, device output, tracking score, and the amplitude kept for the next step."""
        u = s.u + jnp.cumsum(gain * self.delta_scale * jnp.clip(action, -1, 1))
        out = u + offset
        err = jnp.mean((out - self.target(s.t * self.K + jnp.arange(1, self.K + 1), s.phase)) ** 2)
        return u, out, -jnp.log10(err + 1e-8), jnp.clip(u[-1], -1, 1)

    def step(self, s: DriftState, action):
        u, out, score, u_next = self.outcome(s, action, s.gain, s.offset)
        oob = jnp.max(jnp.abs(out)) > 1
        reward = jnp.where(oob, self.terminal_penalty * (1 - s.t / self.max_steps), score + self.reward_shift)
        ns = s._replace(u=u_next.astype(jnp.float32), t=s.t + 1)
        return self.obs(ns), ns, reward, oob, (ns.t >= self.max_steps) & ~oob, dict(score=score)

    # --- what a deployed agent can measure after a step, as a differentiable function of the drift z
    def measure(self, s: DriftState, action, z):
        """y = (change of the commanded amplitude, tracking score); both observable after the step."""
        gain, offset = self.params_of(z)
        _, _, score, u_next = self.outcome(s, action, gain, offset)
        return jnp.stack([u_next - s.u, score])

    def max_out(self, s: DriftState, action, z):
        gain, offset = self.params_of(z)
        return jnp.max(jnp.abs(self.outcome(s, action, gain, offset)[1]))


@dataclass(frozen=True)
class Ham:
    """The gate environment of env.py behind the toy interface."""
    cfg: jenv.EnvConfig = field(default_factory=jenv.EnvConfig)

    @property
    def obs_dim(self):
        return self.cfg.obs_dim

    @property
    def act_dim(self):
        return self.cfg.act_dim

    @property
    def max_steps(self):
        return self.cfg.n_steps

    def reset(self, key):
        return jenv.reset(key, self.cfg)

    def step(self, state, action):
        obs, state, r, term, trunc, info = jenv.step(state, action, self.cfg)
        return obs, state, r, term, trunc, dict(score=-jnp.log10(info["JT"]))


ENVS = {"pendulum": Pendulum, "mountaincar": MountainCar, "tracker": Tracker, "ham": Ham, "drift_tracker": DriftTracker}


def step_autoreset(env, key, state, action):
    """Step, and reset when the episode ends; the pre-reset obs is returned as info['final_obs']."""
    obs, new_state, reward, term, trunc, info = env.step(state, action)
    done = term | trunc
    obs_r, state_r = env.reset(key)
    state_out = jax.tree_util.tree_map(lambda r, s: jnp.where(done, r, s), state_r, new_state)
    return jnp.where(done, obs_r, obs), state_out, reward, term, trunc, dict(info, final_obs=obs)
