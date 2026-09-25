"""Autoregressive delta policy: one head emits one amplitude delta and is rolled out K times per env step.

The trunk embeds the observation once. The head sees ``[embedding, previous delta, running amplitude]``
and returns the mean of a single delta; the sampled (clipped) delta is fed back as the next "previous
delta" and added to the running amplitude, K times. The first call is seeded from the observation
itself (last amplitude, and the last delta of the previous step), so the head is the same function at
every sample of the pulse, across step boundaries too. One state-independent log-std is shared by all
K deltas.

Same interface as ``common.ActorCritic`` (sample / mode / logp / entropy / value), so ``collect``,
``evaluate`` and the PPO losses use it unchanged. ``logp`` teacher-forces the stored actions, so the
K head calls run in parallel during the update.
"""
from typing import NamedTuple, Sequence

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn

from rlquantopt.jx.agents.common import ACTIVATIONS, MLP, gaussian_logp


class Trunk(nn.Module):
    hidden: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        act = ACTIVATIONS[self.activation]
        for h in self.hidden:
            x = act(nn.Dense(h, kernel_init=nn.initializers.orthogonal(np.sqrt(2)))(x))
        return x


class ARActorCritic(NamedTuple):
    trunk: Trunk
    head: MLP
    critic: MLP
    K: int                  # deltas per env step
    amp_start: int          # index of the K amplitudes in the observation
    obs_scale: float
    a_norm_max: float

    @classmethod
    def make(cls, env_cfg, hidden=(128, 128), head_hidden=64, activation="tanh"):
        K = env_cfg.n_time_steps
        return cls(Trunk(tuple(hidden), activation), MLP((head_hidden,), 1, 0.01, activation),
                   MLP(tuple(hidden), 1, 1.0, activation), K, env_cfg.obs_dim - K - 1,
                   env_cfg.obs_scale, env_cfg.a_norm_max)

    def init(self, key, obs_dim, log_std_init=0.0):
        k1, k2, k3 = jax.random.split(key, 3)
        x = jnp.zeros((1, obs_dim), jnp.float32)
        actor = dict(trunk=self.trunk.init(k1, x),
                     head=self.head.init(k2, jnp.zeros((1, self.trunk.hidden[-1] + 2), jnp.float32)),
                     log_std=jnp.asarray(log_std_init, jnp.float32))
        return dict(actor=actor, critic=self.critic.init(k3, x))

    def value(self, critic_params, obs):
        return self.critic.apply(critic_params, obs.astype(jnp.float32))[..., 0]

    def _seed(self, obs):
        """Running amplitude and previous delta at the start of the step, in action units."""
        amps = obs[..., self.amp_start:self.amp_start + self.K] / self.obs_scale * self.a_norm_max
        return amps[..., -1], amps[..., -1] - amps[..., -2]

    def _mean(self, actor_params, h, prev, amp):
        x = jnp.concatenate([h, prev[..., None], amp[..., None]], axis=-1)
        return self.head.apply(actor_params["head"], x)[..., 0]

    def _rollout(self, actor_params, obs, noise):
        """Feed the head its own (clipped) output K times. ``noise`` (..., K); zeros give the mode."""
        obs = obs.astype(jnp.float32)
        h = self.trunk.apply(actor_params["trunk"], obs)
        amp, prev = self._seed(obs)
        std = jnp.exp(actor_params["log_std"])

        def body(carry, eps):
            prev, amp = carry
            mean = self._mean(actor_params, h, prev, amp)
            a = mean + std * eps
            a_env = jnp.clip(a, -1, 1)          # what the env applies
            return (a_env, amp + a_env / self.a_norm_max), (a, mean)

        _, (a, mean) = jax.lax.scan(body, (prev, amp), jnp.moveaxis(noise.astype(jnp.float32), -1, 0))
        return jnp.moveaxis(a, 0, -1), jnp.moveaxis(mean, 0, -1)

    def sample(self, actor_params, obs, key):
        noise = jax.random.normal(key, obs.shape[:-1] + (self.K,))
        action, mean = self._rollout(actor_params, obs, noise)
        return action, gaussian_logp(mean, actor_params["log_std"], action)

    def mode(self, actor_params, obs):
        return self._rollout(actor_params, obs, jnp.zeros(obs.shape[:-1] + (self.K,)))[0]

    def logp(self, actor_params, obs, action):
        """Teacher-forced: the head's inputs are rebuilt from the stored actions, all K in parallel."""
        obs = obs.astype(jnp.float32)
        h = self.trunk.apply(actor_params["trunk"], obs)
        amp0, prev0 = self._seed(obs)
        a_env = jnp.clip(action, -1, 1)
        prev = jnp.concatenate([prev0[..., None], a_env[..., :-1]], axis=-1)
        amp = amp0[..., None] + jnp.concatenate(
            [jnp.zeros_like(amp0)[..., None], jnp.cumsum(a_env[..., :-1] / self.a_norm_max, axis=-1)], axis=-1)
        h = jnp.broadcast_to(h[..., None, :], h.shape[:-1] + (self.K, h.shape[-1]))
        mean = self._mean(actor_params, h, prev, amp)
        return gaussian_logp(mean, actor_params["log_std"], action)

    def entropy(self, actor_params):
        return self.K * (actor_params["log_std"] + 0.5 * jnp.log(2 * jnp.pi * jnp.e))
