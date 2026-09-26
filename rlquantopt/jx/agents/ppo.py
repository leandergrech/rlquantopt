"""PPO (clipped surrogate), fully jitted: rollout, GAE and minibatch epochs in one ``update`` call."""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax

from rlquantopt.jx import env as jenv
from rlquantopt.jx.agents.common import ActorCritic, RunnerState, collect, gae, gaussian_logp


@dataclass(frozen=True)
class PPOConfig:
    n_envs: int = 64
    n_steps: int = 128
    n_epochs: int = 10
    n_minibatches: int = 32
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    lr: float = 3e-4
    lr_harmonic_k: float = 10 ** 0.4    # lr / (1 + k * progress), as the paper's harmonic schedule
    total_steps: int = 20_000_000
    hidden: tuple = (128, 128)
    activation: str = "tanh"  # SB3 default
    resample_drift: bool = True

    @property
    def batch_size(self):
        return self.n_envs * self.n_steps

    @property
    def n_updates(self):
        return self.total_steps // self.batch_size


def make_optimizer(cfg: PPOConfig):
    n_grad_steps = cfg.n_updates * cfg.n_epochs * cfg.n_minibatches
    schedule = lambda i: cfg.lr / (1 + cfg.lr_harmonic_k * i / n_grad_steps)
    return optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(schedule, eps=1e-5))


def init(key, env_cfg: jenv.EnvConfig, cfg: PPOConfig):
    model = ActorCritic.make(env_cfg.act_dim, cfg.hidden, cfg.activation)
    k_net, k_env, key = jax.random.split(key, 3)
    params = model.init(k_net, env_cfg.obs_dim)
    obs, env_state = jax.vmap(jenv.reset, in_axes=(0, None))(jax.random.split(k_env, cfg.n_envs), env_cfg)
    opt_state = make_optimizer(cfg).init(params)
    return model, RunnerState(params, env_state, obs, key), opt_state


def make_update(model: ActorCritic, env_cfg: jenv.EnvConfig, cfg: PPOConfig):
    tx = make_optimizer(cfg)

    # --8<-- [start:ppo_loss]
    def loss_fn(params, batch):
        obs, action, logp_old, adv, ret = batch
        mean, log_std = model.dist(params["actor"], obs)
        logp = gaussian_logp(mean, log_std, action)
        ratio = jnp.exp(logp - logp_old)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        pg = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv).mean()
        v = model.value(params["critic"], obs)
        v_loss = ((ret - v) ** 2).mean()
        entropy = jnp.sum(log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e))
        loss = pg + cfg.vf_coef * v_loss - cfg.ent_coef * entropy
        return loss, dict(pg_loss=pg, v_loss=v_loss, approx_kl=((ratio - 1) - jnp.log(ratio)).mean(),
                          clip_frac=(jnp.abs(ratio - 1) > cfg.clip_eps).mean())
    # --8<-- [end:ppo_loss]

    # --8<-- [start:ppo_update]
    @jax.jit
    def update(runner: RunnerState, opt_state):
        runner, traj = collect(model, runner, env_cfg, cfg.n_steps, cfg.gamma, cfg.resample_drift)
        last_value = model.value(runner.params["critic"], runner.obs)
        adv, ret = gae(traj, last_value, cfg.gamma, cfg.gae_lambda)
        flat = lambda x: x.reshape((cfg.batch_size,) + x.shape[2:])
        data = (flat(traj.obs), flat(traj.action), flat(traj.logp), flat(adv), flat(ret))

        def epoch(carry, key):
            params, opt_state = carry
            perm = jax.random.permutation(key, cfg.batch_size)
            mbs = jax.tree_util.tree_map(
                lambda x: x[perm].reshape((cfg.n_minibatches, -1) + x.shape[1:]), data)

            def minibatch(carry, mb):
                params, opt_state = carry
                (loss, stats), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, mb)
                updates, opt_state = tx.update(grads, opt_state, params)
                return (optax.apply_updates(params, updates), opt_state), stats

            return jax.lax.scan(minibatch, (params, opt_state), mbs)

        key, k_ep = jax.random.split(runner.key)
        (params, opt_state), stats = jax.lax.scan(
            epoch, (runner.params, opt_state), jax.random.split(k_ep, cfg.n_epochs))
        runner = runner._replace(params=params, key=key)
        stats = jax.tree_util.tree_map(jnp.mean, stats)
        stats.update(rollout_stats(traj))
        return runner, opt_state, stats
    # --8<-- [end:ppo_update]

    return update


def rollout_stats(traj):
    return dict(mean_reward=traj.reward.mean(), min_JT=traj.JT.min(),
                frac_done=traj.done.mean())
