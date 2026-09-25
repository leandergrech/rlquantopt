"""Networks, rollouts and GAE shared by the JAX agents.

Mirrors the Stable-Baselines3 ActorCriticPolicy used in the paper: separate
policy and value MLPs ([128, 128]; ReLU in the paper run, tanh is the SB3 default), orthogonal init (gain sqrt(2) hidden,
0.01 policy head, 1 value head), a state-independent Gaussian log-std (init 0),
actions clipped to [-1, 1] only when sent to the env, and SB3's bootstrap of
truncated episodes (reward += gamma * V(final_obs)).
"""
from typing import NamedTuple, Sequence

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn

from rlquantopt.jx import env as jenv


ACTIVATIONS = {"tanh": nn.tanh, "relu": nn.relu, "leaky_relu": nn.leaky_relu,   # leaky slope 0.01
               "gelu": nn.gelu}


class MLP(nn.Module):
    hidden: Sequence[int]
    out_dim: int
    out_gain: float
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        act = ACTIVATIONS[self.activation]
        for h in self.hidden:
            x = act(nn.Dense(h, kernel_init=nn.initializers.orthogonal(np.sqrt(2)))(x))
        return nn.Dense(self.out_dim, kernel_init=nn.initializers.orthogonal(self.out_gain))(x)


class ActorCritic(NamedTuple):
    actor: MLP
    critic: MLP

    @classmethod
    def make(cls, act_dim, hidden=(128, 128), activation="tanh"):
        return cls(MLP(tuple(hidden), act_dim, 0.01, activation), MLP(tuple(hidden), 1, 1.0, activation))

    def init(self, key, obs_dim, log_std_init=0.0):
        k1, k2 = jax.random.split(key)
        x = jnp.zeros((1, obs_dim), jnp.float32)
        actor = dict(net=self.actor.init(k1, x),
                     log_std=jnp.full((self.actor.out_dim,), log_std_init, jnp.float32))
        return dict(actor=actor, critic=self.critic.init(k2, x))

    def dist(self, actor_params, obs):
        return self.actor.apply(actor_params["net"], obs.astype(jnp.float32)), actor_params["log_std"]

    def value(self, critic_params, obs):
        return self.critic.apply(critic_params, obs.astype(jnp.float32))[..., 0]

    # The interface the rollout, PPO and evaluation use, so other policies (e.g. autoregressive.py) can plug in.
    def sample(self, actor_params, obs, key):
        mean, log_std = self.dist(actor_params, obs)
        action = mean + jnp.exp(log_std) * jax.random.normal(key, mean.shape)
        return action, gaussian_logp(mean, log_std, action)

    def mode(self, actor_params, obs):
        return self.dist(actor_params, obs)[0]

    def logp(self, actor_params, obs, action):
        return gaussian_logp(*self.dist(actor_params, obs), action)

    def entropy(self, actor_params):
        log_std = actor_params["log_std"]
        return jnp.sum(log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e))


def gaussian_logp(mean, log_std, a):
    return jnp.sum(-0.5 * ((a - mean) / jnp.exp(log_std)) ** 2 - log_std - 0.5 * jnp.log(2 * jnp.pi), axis=-1)


def gaussian_kl(mean_p, log_std_p, mean_q, log_std_q):
    """KL(p || q) for diagonal Gaussians, summed over action dims."""
    var_p, var_q = jnp.exp(2 * log_std_p), jnp.exp(2 * log_std_q)
    return jnp.sum(log_std_q - log_std_p + (var_p + (mean_p - mean_q) ** 2) / (2 * var_q) - 0.5, axis=-1)


class Transition(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    logp: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray         # includes the truncation bootstrap
    done: jnp.ndarray
    JT: jnp.ndarray
    concurrence: jnp.ndarray
    unitarity: jnp.ndarray
    t: jnp.ndarray


class RunnerState(NamedTuple):
    params: dict
    env_state: jenv.EnvState
    obs: jnp.ndarray
    key: jnp.ndarray


# --8<-- [start:collect]
def collect(model, runner: RunnerState, cfg: jenv.EnvConfig, n_steps: int, gamma: float,
            resample_drift: bool = True):
    """Roll out ``n_steps`` in every env. Arrays in the returned Transition are (n_steps, n_envs, ...)."""
    v_step = jax.vmap(jenv.step_autoreset, in_axes=(0, 0, 0, None, None))

    def body(runner, _):
        params, env_state, obs, key = runner
        key, k_act, k_env = jax.random.split(key, 3)
        action, logp = model.sample(params["actor"], obs, k_act)
        value = model.value(params["critic"], obs)
        n = obs.shape[0]
        next_obs, env_state, reward, term, trunc, info = v_step(
            jax.random.split(k_env, n), env_state, action, cfg, resample_drift)
        bootstrap = model.value(params["critic"], info["final_obs"])
        reward = reward + gamma * bootstrap * (trunc & ~term)
        tr = Transition(obs, action, logp, value, reward.astype(jnp.float32), term | trunc,
                        info["JT"], info["concurrence"], info["unitarity"], info["t"])
        return RunnerState(params, env_state, next_obs, key), tr

    return jax.lax.scan(body, runner, None, n_steps)
# --8<-- [end:collect]


# --8<-- [start:gae]
def gae(traj: Transition, last_value, gamma, lam):
    def body(carry, tr):
        next_adv, next_value = carry
        not_done = 1.0 - tr.done
        delta = tr.reward + gamma * next_value * not_done - tr.value
        adv = delta + gamma * lam * not_done * next_adv
        return (adv, tr.value), adv

    _, adv = jax.lax.scan(body, (jnp.zeros_like(last_value), last_value), traj, reverse=True)
    return adv, adv + traj.value
# --8<-- [end:gae]


def evaluate(model, params, cfg: jenv.EnvConfig, omega_s=None, stop_on_truncation=True):
    """Deterministic episode (mean actions). Returns per-step reward, J_T, C, U, t and the pulse.

    With ``stop_on_truncation=False`` the episode keeps going after an out-of-bounds step (amplitudes
    stay clipped), as the v1 analysis notebooks did; ``alive`` then only ends at the time limit.
    """
    omega_s = jnp.asarray(cfg.model.omega_s if omega_s is None else omega_s)
    obs, state = jenv.reset_to(omega_s, cfg)

    def body(carry, _):
        obs, state, alive = carry
        obs, state, r, term, trunc, info = jenv.step(state, model.mode(params["actor"], obs), cfg)
        out = dict(reward=r, JT=info["JT"], concurrence=info["concurrence"], unitarity=info["unitarity"],
                   t=info["t"], amps=state.amps_cur, alive=alive)
        ended = term | trunc if stop_on_truncation else term
        return (obs, state, alive & ~ended), out

    _, out = jax.lax.scan(body, (obs, state, jnp.array(True)), None, cfg.n_steps)
    return out
