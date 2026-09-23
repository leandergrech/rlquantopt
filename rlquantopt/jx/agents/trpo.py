"""TRPO following sb3_contrib.TRPO (the paper's agent), fully jitted.

Per update: collect n_envs x n_steps transitions, GAE, one natural-gradient
step on the actor (conjugate gradient on the KL Hessian with damping, then a
backtracking line search that must keep KL < target_kl and improve the
surrogate), then n_critic_updates epochs of Adam on the value loss.
"""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax
from jax.flatten_util import ravel_pytree

from rlquantopt.jx import env as jenv
from rlquantopt.jx.agents.common import ActorCritic, RunnerState, collect, gae, gaussian_logp, gaussian_kl


@dataclass(frozen=True)
class TRPOConfig:
    n_envs: int = 4
    n_steps: int = 2048
    batch_size: int = 128               # critic minibatch
    n_critic_updates: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    target_kl: float = 0.01
    cg_max_steps: int = 15
    cg_damping: float = 0.1
    line_search_shrinking_factor: float = 0.8
    line_search_max_iter: int = 10
    lr: float = 3e-4
    lr_harmonic_k: float = 10 ** 0.4    # paper: lr / (1 + k (1 - progress_remaining))
    total_steps: int = 20_000_000
    hidden: tuple = (128, 128)
    activation: str = "relu"  # the paper run (SB3, Dec 2024) used ReLU
    resample_drift: bool = True

    @property
    def rollout_size(self):
        return self.n_envs * self.n_steps

    @property
    def n_updates(self):
        return self.total_steps // self.rollout_size

    @property
    def n_minibatches(self):
        return self.rollout_size // self.batch_size


def make_optimizer(cfg: TRPOConfig):
    steps_per_update = cfg.n_critic_updates * cfg.n_minibatches
    # SB3 sets the lr once per update from the fraction of total timesteps done.
    schedule = lambda i: cfg.lr / (1 + cfg.lr_harmonic_k * (i // steps_per_update) / cfg.n_updates)
    return optax.adam(schedule, eps=1e-5)


def init(key, env_cfg: jenv.EnvConfig, cfg: TRPOConfig):
    model = ActorCritic.make(env_cfg.act_dim, cfg.hidden, cfg.activation)
    k_net, k_env, key = jax.random.split(key, 3)
    params = model.init(k_net, env_cfg.obs_dim)
    obs, env_state = jax.vmap(jenv.reset, in_axes=(0, None))(jax.random.split(k_env, cfg.n_envs), env_cfg)
    opt_state = make_optimizer(cfg).init(params["critic"])
    return model, RunnerState(params, env_state, obs, key), opt_state


def conjugate_gradient(Avp, b, n_iter, tol=1e-10):
    def body(carry, _):
        x, r, p, rr = carry
        Ap = Avp(p)
        alpha = rr / (p @ Ap)
        x_new, r_new = x + alpha * p, r - alpha * Ap
        rr_new = r_new @ r_new
        p_new = r_new + rr_new / rr * p
        done = rr < tol
        keep = lambda a, b: jnp.where(done, a, b)
        return (keep(x, x_new), keep(r, r_new), keep(p, p_new), keep(rr, rr_new)), None

    x0 = jnp.zeros_like(b)
    (x, *_), _ = jax.lax.scan(body, (x0, b, b, b @ b), None, n_iter)
    return x


def make_update(model: ActorCritic, env_cfg: jenv.EnvConfig, cfg: TRPOConfig):
    tx = make_optimizer(cfg)

    def actor_step(actor_params, obs, action, logp_old, adv):
        flat0, unravel = ravel_pytree(actor_params)
        mean_old, log_std_old = jax.lax.stop_gradient(model.dist(actor_params, obs))

        def surrogate(flat):
            mean, log_std = model.dist(unravel(flat), obs)
            return jnp.mean(adv * jnp.exp(gaussian_logp(mean, log_std, action) - logp_old))

        def kl(flat):
            mean, log_std = model.dist(unravel(flat), obs)
            return jnp.mean(gaussian_kl(mean, log_std, mean_old, log_std_old))

        obj0, g = jax.value_and_grad(surrogate)(flat0)
        grad_kl = jax.grad(kl)

        def fvp(v):
            return jax.grad(lambda f: grad_kl(f) @ v)(flat0) + cfg.cg_damping * v

        step_dir = conjugate_gradient(fvp, g, cfg.cg_max_steps)
        max_step = jnp.sqrt(2 * cfg.target_kl / (step_dir @ fvp(step_dir)))
        coeffs = cfg.line_search_shrinking_factor ** jnp.arange(cfg.line_search_max_iter)
        cands = flat0[None] + (coeffs * max_step)[:, None] * step_dir[None]
        objs = jax.vmap(surrogate)(cands)
        kls = jax.vmap(kl)(cands)
        ok = (kls < cfg.target_kl) & (objs > obj0)
        first = jnp.argmax(ok)
        success = ok.any()
        new_flat = jnp.where(success, cands[first], flat0)
        stats = dict(policy_objective=jnp.where(success, objs[first], obj0),
                     kl=jnp.where(success, kls[first], 0.0), line_search_success=success.astype(jnp.float32))
        return unravel(new_flat), stats

    def value_loss(critic_params, obs, ret):
        return jnp.mean((ret - model.value(critic_params, obs)) ** 2)

    @jax.jit
    def update(runner: RunnerState, opt_state):
        runner, traj = collect(model, runner, env_cfg, cfg.n_steps, cfg.gamma, cfg.resample_drift)
        params = runner.params
        last_value = model.value(params["critic"], runner.obs)
        adv, ret = gae(traj, last_value, cfg.gamma, cfg.gae_lambda)
        flat = lambda x: x.reshape((cfg.rollout_size,) + x.shape[2:])
        obs, action, logp_old, adv, ret = map(flat, (traj.obs, traj.action, traj.logp, adv, ret))
        adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)

        actor, stats = actor_step(params["actor"], obs, action, logp_old, adv_n)

        def epoch(carry, key):
            critic, opt_state = carry
            perm = jax.random.permutation(key, cfg.rollout_size)
            mbs = jax.tree_util.tree_map(
                lambda x: x[perm].reshape((cfg.n_minibatches, -1) + x.shape[1:]), (obs, ret))

            def mb_step(carry, mb):
                critic, opt_state = carry
                loss, grads = jax.value_and_grad(value_loss)(critic, *mb)
                updates, opt_state = tx.update(grads, opt_state, critic)
                return (optax.apply_updates(critic, updates), opt_state), loss

            return jax.lax.scan(mb_step, (critic, opt_state), mbs)

        key, k_ep = jax.random.split(runner.key)
        (critic, opt_state), v_losses = jax.lax.scan(
            epoch, (params["critic"], opt_state), jax.random.split(k_ep, cfg.n_critic_updates))
        runner = runner._replace(params=dict(actor=actor, critic=critic), key=key)
        stats.update(value_loss=v_losses.mean(), mean_reward=traj.reward.mean(), min_JT=traj.JT.min(),
                     frac_done=traj.done.mean(), std=jnp.exp(actor["log_std"]).mean())
        return runner, opt_state, stats

    return update
