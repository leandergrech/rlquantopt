"""Idea i04 "dragonfly": chameleon made faster, exploring by a guided Gaussian random walk.

Differences from chameleon (i03):

- **History as leaky integrators.** Instead of a GRU, the history is a set of exponential traces of
  ``[z_t, z_t - z_{t-1}, a_{t-1}, r_{t-1}]`` at the decays ``trace_decays`` (reset at episode start),
  mapped to a context c by an MLP. The traces of the latent increments are momenta. A linear recurrence
  trains with ``associative_scan`` (parallel over time) instead of backpropagating through a sequential GRU.
- **Worst case until proven otherwise.** Q and V are ensembles (``n_heads``, each on its own bootstrap
  mask). Their lower bound ``mean - lcb_kappa * std`` is what the search and the guide see: an action or
  state is assumed as bad as the pessimistic head says until the heads agree.
- **Search** scores candidates by ``cos(jump, g) + search_value_coef * (Q_lcb - max_candidates Q_lcb)``,
  so a candidate predicted to end the episode is vetoed (chameleon ranked by alignment alone).
- **Exploration is an Ornstein-Uhlenbeck walk**, per latent dimension d and every env step:
  ``xi_d <- rho_d xi_d + sqrt(1 - rho_d^2) sigma_d eta``, ``rho_d = exp(-1 / tau_d)``, eta ~ N(0, 1).
  The stationary spread is sigma_d whatever rho_d is, so the walk's entropy is constant while it carries
  ``-1/2 log(1 - rho_d^2)`` nats of momentum from step to step; tau_d is how many steps a heading persists.
  The task is ``g = sqrt(1-eps) u + sqrt(eps) xi_perp / |xi_perp|``, with u the value-gradient
  direction (set at goal steps) and xi_perp the walk projected orthogonally to u.
- **The guide** (a small meta-learner) picks ``tau_d, sigma_d`` (per dimension) and ``eps`` at goal steps,
  from the context, the measured entropy of each latent dimension, the pessimistic value, and the
  performance ``p`` relative to the golden cache. It is trained with a clipped PPO surrogate on the
  GAE advantage of the env reward (model V baseline). How tau relates to p and to the entropy is learned,
  and logged (``tau_corr_p``, ``tau_corr_entropy``).
- **Golden cache.** Up to ``cache_size`` experiences with the best reward or lambda-return seen so far,
  canonical (at most one per latent ball of radius ``cache_radius``), refreshed every rollout.
  ``p = mean of min-max position of (last reward, predicted return) within the cache``; 0 while the cache
  is empty. The policy imitates cached actions whose return beat the value estimated when they were
  taken (self-imitation, weight ``sil_coef``).
"""
from dataclasses import dataclass
from typing import Any, NamedTuple, Sequence

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

from rlquantopt.jx import env as jenv
from rlquantopt.jx.agents import ppo
from rlquantopt.jx.agents.chameleon import _cos, _f32, gae_arrays, gamma_feature, gamma_targets, lambda_returns
from rlquantopt.jx.agents.common import MLP, gaussian_logp


@dataclass(frozen=True)
class DragonflyConfig(ppo.PPOConfig):
    activation: str = "leaky_relu"
    latent_dim: int = 16
    enc_hidden: tuple = (128,)
    trace_decays: tuple = (0.5, 0.9, 0.99)
    ctx_dim: int = 64
    model_hidden: tuple = (128, 128)
    n_heads: int = 4                  # Q/V ensemble for the lower bound
    head_keep: float = 0.8            # bootstrap mask probability per head
    lcb_kappa: float = 1.0
    gammas: tuple = (0.5, 0.8, 0.9, 0.95, 0.99)
    plan_gamma: float = 0.9
    model_lambda: float = 0.95
    goal_every: int = 8
    n_candidates: int = 8
    search_value_coef: float = 1.0
    task_reward_coef: float = 1.0
    env_reward_coef: float = 1.0
    tau_max: float = 64.0
    guide_hidden: tuple = (64,)
    guide_ent_coef: float = 1e-3
    cache_size: int = 64
    cache_radius: float = 0.5
    cache_candidates: int = 64        # per ranking (reward, return), per rollout
    sil_coef: float = 0.1
    entropy_ema: float = 0.9
    model_lr: float = 3e-4
    model_epochs: int = 4
    model_envs: int = 8
    model_warmup: int = 5
    model_reward_scale: float = 0.1
    vicreg_coef: float = 1.0

    @property
    def guide_dim(self):
        return 2 * self.latent_dim + 1


class DFModel(nn.Module):
    latent_dim: int
    enc_hidden: Sequence[int]
    ctx_dim: int
    model_hidden: Sequence[int]
    n_heads: int
    activation: str

    def setup(self):
        act = self.activation
        self.enc = MLP(tuple(self.enc_hidden), self.latent_dim, 1.0, act)
        self.ctx = MLP((self.ctx_dim,), self.ctx_dim, 1.0, act)
        self.jump_net = MLP(tuple(self.model_hidden), self.latent_dim, 0.1, act)
        self.q_nets = [MLP(tuple(self.model_hidden), 1, 1.0, act) for _ in range(self.n_heads)]
        self.v_nets = [MLP(tuple(self.model_hidden), 1, 1.0, act) for _ in range(self.n_heads)]

    def encode(self, obs):
        return self.enc(_f32(obs))

    def context(self, traces):
        return self.ctx(_f32(traces.reshape(traces.shape[:-2] + (-1,))))

    def jump(self, z, c, a, gamma):
        return self.jump_net(_f32(jnp.concatenate([z, c, a, gamma_feature(gamma)[..., None]], -1)))

    def q(self, z, c, a, gamma):
        x = _f32(jnp.concatenate([z, c, a, gamma_feature(gamma)[..., None]], -1))
        return jnp.stack([n(x)[..., 0] for n in self.q_nets], -1)

    def v(self, z, c, gamma):
        x = _f32(jnp.concatenate([z, c, gamma_feature(gamma)[..., None]], -1))
        return jnp.stack([n(x)[..., 0] for n in self.v_nets], -1)

    def __call__(self, obs, traces, a, gamma):
        z, c = self.encode(obs), self.context(traces)
        return self.jump(z, c, a, gamma), self.q(z, c, a, gamma), self.v(z, c, gamma)


class Walk(NamedTuple):
    """Per-env agent state carried between env steps."""
    traces: jnp.ndarray     # (n, n_decays, trace_in)
    z_prev: jnp.ndarray
    a_prev: jnp.ndarray
    r_prev: jnp.ndarray
    first: jnp.ndarray
    u: jnp.ndarray          # exploit direction of the current goal
    xi: jnp.ndarray         # exploration walk
    tau: jnp.ndarray        # (n, d)
    sig: jnp.ndarray        # (n, d)
    eps: jnp.ndarray
    graw: jnp.ndarray       # guide's raw (Gaussian) action at the last goal step
    glogp: jnp.ndarray
    gin: jnp.ndarray        # guide's input at the last goal step
    since: jnp.ndarray


class Cache(NamedTuple):
    obs: jnp.ndarray
    z: jnp.ndarray
    g: jnp.ndarray
    action: jnp.ndarray
    r: jnp.ndarray          # scaled env reward
    G: jnp.ndarray          # scaled env lambda-return
    v: jnp.ndarray          # model value when it was collected
    valid: jnp.ndarray


class Dragonfly(NamedTuple):
    wm: DFModel
    actor: MLP
    critic: MLP
    guide: MLP
    act_dim: int
    obs_dim: int
    cfg: DragonflyConfig

    @classmethod
    def make(cls, env_cfg: jenv.EnvConfig, cfg: DragonflyConfig):
        a = cfg.activation
        return cls(DFModel(cfg.latent_dim, cfg.enc_hidden, cfg.ctx_dim, cfg.model_hidden, cfg.n_heads, a),
                   MLP(tuple(cfg.hidden), env_cfg.act_dim, 0.01, a), MLP(tuple(cfg.hidden), 1, 1.0, a),
                   MLP(tuple(cfg.guide_hidden), cfg.guide_dim, 0.01, a), env_cfg.act_dim, env_cfg.obs_dim, cfg)

    @property
    def trace_in(self):
        return 2 * self.cfg.latent_dim + self.act_dim + 1

    @property
    def gin_dim(self):
        return self.cfg.ctx_dim + 1 + self.cfg.latent_dim + 2

    def init(self, key):
        k = jax.random.split(key, 4)
        d, A = self.cfg.latent_dim, self.act_dim
        wm = self.wm.init(k[0], jnp.zeros((1, self.obs_dim)), jnp.zeros((1, len(self.cfg.trace_decays), self.trace_in)),
                          jnp.zeros((1, A)), jnp.full((1,), 0.9))
        pi_in = jnp.zeros((1, self.obs_dim + 2 * d))
        return dict(actor=dict(net=self.actor.init(k[1], pi_in), log_std=jnp.zeros(A, jnp.float32)),
                    critic=self.critic.init(k[2], pi_in),
                    guide=dict(net=self.guide.init(k[3], jnp.zeros((1, self.gin_dim))),
                               log_std=jnp.zeros(self.cfg.guide_dim, jnp.float32)),
                    model=wm)

    def m(self, params, method, *args):
        return self.wm.apply(params["model"], *args, method=method)

    def pi_input(self, obs, z, g):
        return _f32(jnp.concatenate([_f32(obs), jax.lax.stop_gradient(_f32(z)), _f32(g)], -1))

    def pi_mean(self, params, obs, z, g):
        return self.actor.apply(params["actor"]["net"], self.pi_input(obs, z, g))

    def value(self, params, obs, z, g):
        return self.critic.apply(params["critic"], self.pi_input(obs, z, g))[..., 0]

    def guide_mean(self, params, gin):
        return self.guide.apply(params["guide"]["net"], jax.lax.stop_gradient(_f32(gin)))

    def decode_guide(self, raw):
        d = self.cfg.latent_dim
        s = jax.nn.sigmoid(raw)
        return 1.0 + (self.cfg.tau_max - 1.0) * s[..., :d], s[..., d:2 * d], s[..., -1]    # tau, sigma, eps

    def observe(self, params, walk: Walk, obs):
        """Encode the obs and advance the traces (reset at episode start). Returns z, traces, c."""
        z = self.m(params, DFModel.encode, obs)
        keep = (~walk.first)[..., None]
        z_prev = jnp.where(keep, walk.z_prev, z)
        x = jnp.concatenate([z, z - z_prev, walk.a_prev * keep,
                             (walk.r_prev * self.cfg.model_reward_scale)[..., None] * keep], -1)
        dec = jnp.asarray(self.cfg.trace_decays, jnp.float32)[:, None]
        traces = dec * walk.traces * keep[..., None] + (1 - dec) * x[..., None, :]
        return z, traces, self.m(params, DFModel.context, traces)


def lower_bound(heads, kappa):
    return heads.mean(-1) - kappa * heads.std(-1)


def exploit_direction(model: Dragonfly, params, z, c):
    gam = jnp.full(z.shape[:-1], model.cfg.plan_gamma)
    u = jax.grad(lambda z_: model.m(params, DFModel.v, z_, c, gam).mean(-1).sum())(z)
    return u / (jnp.linalg.norm(u, axis=-1, keepdims=True) + 1e-8)


def cache_position(cache: Cache, r, v):
    """p in [0, 1]: where (last reward, predicted return) sit between the cache's worst and best; 0 if empty."""
    big = 1e9
    rb, rw = jnp.max(jnp.where(cache.valid, cache.r, -big)), jnp.min(jnp.where(cache.valid, cache.r, big))
    Gb, Gw = jnp.max(jnp.where(cache.valid, cache.G, -big)), jnp.min(jnp.where(cache.valid, cache.G, big))
    pr = jnp.clip((r - rw) / (rb - rw + 1e-6), 0, 1)
    pG = jnp.clip((v - Gw) / (Gb - Gw + 1e-6), 0, 1)
    return jnp.where(cache.valid.any(), 0.5 * (pr + pG), 0.0)


# --8<-- [start:act]
def act(model: Dragonfly, params, walk: Walk, obs, entropy, cache: Cache, key, deterministic=False):
    cfg = model.cfg
    k_guide, k_walk, k_cand = jax.random.split(key, 3)
    z, traces, c = model.observe(params, walk, obs)
    new_goal = walk.first | (walk.since >= cfg.goal_every)

    # Goal step: exploit direction, pessimistic value, cache position, guide decision
    u = jnp.where(new_goal[..., None], exploit_direction(model, params, z, c), walk.u)
    v_heads = model.m(params, DFModel.v, z, c, jnp.full(z.shape[:-1], cfg.gamma))
    v_mean, v_std = v_heads.mean(-1), v_heads.std(-1)
    p = cache_position(cache, walk.r_prev * cfg.model_reward_scale * ~walk.first, v_mean)
    gin = jnp.concatenate([c, p[..., None], jnp.broadcast_to(entropy, z.shape),
                           (v_mean - cfg.lcb_kappa * v_std)[..., None], v_std[..., None]], -1)
    gmean = model.guide_mean(params, gin)
    glog_std = params["guide"]["log_std"]
    raw = gmean if deterministic else gmean + jnp.exp(glog_std) * jax.random.normal(k_guide, gmean.shape, jnp.float32)
    tau_n, sig_n, eps_n = model.decode_guide(raw)
    if deterministic:
        eps_n = jnp.zeros_like(eps_n)
    sel = lambda new, old: jnp.where(new_goal.reshape(new_goal.shape + (1,) * (new.ndim - new_goal.ndim)), new, old)
    tau, sig, eps = sel(tau_n, walk.tau), sel(sig_n, walk.sig), sel(eps_n, walk.eps)
    graw, glogp, gin = sel(raw, walk.graw), sel(gaussian_logp(gmean, glog_std, raw), walk.glogp), sel(gin, walk.gin)

    # Exploration walk (every step): Ornstein-Uhlenbeck with per-dimension persistence and spread
    eta = jax.random.normal(k_walk, z.shape, jnp.float32)
    rho = jnp.exp(-1.0 / tau)
    xi = jnp.where(walk.first[..., None], sig * eta, rho * walk.xi + jnp.sqrt(1 - rho ** 2) * sig * eta)
    xi_perp = xi - jnp.sum(xi * u, -1, keepdims=True) * u
    xi_perp = xi_perp / (jnp.linalg.norm(xi_perp, axis=-1, keepdims=True) + 1e-8)
    g = jnp.sqrt(1 - eps)[..., None] * u + jnp.sqrt(eps)[..., None] * xi_perp

    # Candidate search: alignment with the task plus the pessimistic Q
    mean = model.pi_mean(params, obs, z, g)
    log_std = params["actor"]["log_std"]
    noise = jax.random.normal(k_cand, (cfg.n_candidates,) + mean.shape, jnp.float32)
    if deterministic:
        noise = noise.at[0].set(0.0)
    cands = mean + jnp.exp(log_std) * noise
    N = cfg.n_candidates
    zb, cb = jnp.broadcast_to(z, (N,) + z.shape), jnp.broadcast_to(c, (N,) + c.shape)
    gam = jnp.full((N,) + z.shape[:-1], cfg.plan_gamma)
    align = _cos(model.m(params, DFModel.jump, zb, cb, cands, gam), g)
    q_lcb = lower_bound(model.m(params, DFModel.q, zb, cb, cands, gam), cfg.lcb_kappa)
    score = align + cfg.search_value_coef * (q_lcb - q_lcb.max(0))
    best = jnp.argmax(score, 0)
    action = jnp.take_along_axis(cands, best[None, :, None], 0)[0]
    logp = gaussian_logp(mean, log_std, action)

    walk = walk._replace(traces=traces, z_prev=z, u=u, xi=xi, tau=tau, sig=sig, eps=eps, graw=graw, glogp=glogp,
                         gin=gin, since=jnp.where(new_goal, 1, walk.since + 1))
    info = dict(z=z, c=c, g=g, new_goal=new_goal, p=p, v_mean=v_mean, v_std=v_std,
                search_gain=score.max(0) - score.mean(0), vetoed=(q_lcb - q_lcb.max(0) < -0.5).mean(0))
    return action, logp, walk, info
# --8<-- [end:act]


class DFRunner(NamedTuple):
    params: dict
    env_state: jenv.EnvState
    obs: jnp.ndarray
    key: jnp.ndarray
    walk: Walk
    cache: Cache
    lat_mean: jnp.ndarray
    lat_var: jnp.ndarray
    n_updates: jnp.ndarray


class DFTransition(NamedTuple):
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    z: jnp.ndarray
    g: jnp.ndarray
    a_prev: jnp.ndarray
    r_prev: jnp.ndarray
    first: jnp.ndarray
    action: jnp.ndarray
    logp: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    r_env: jnp.ndarray
    task: jnp.ndarray
    v_model: jnp.ndarray
    gin: jnp.ndarray
    graw: jnp.ndarray
    glogp: jnp.ndarray
    new_goal: jnp.ndarray
    done: jnp.ndarray
    tau: jnp.ndarray
    sig: jnp.ndarray
    eps: jnp.ndarray
    p: jnp.ndarray
    v_std: jnp.ndarray
    search_gain: jnp.ndarray
    vetoed: jnp.ndarray
    JT: jnp.ndarray


def measured_entropy(lat_var):
    return 0.5 * jnp.log(2 * jnp.pi * jnp.e * (lat_var + 1e-6))


def collect(model: Dragonfly, runner: DFRunner, env_cfg, n_steps):
    cfg = model.cfg
    v_step = jax.vmap(jenv.step_autoreset, in_axes=(0, 0, 0, None, None))
    entropy = measured_entropy(runner.lat_var)

    def body(r: DFRunner, _):
        key, k_act, k_env = jax.random.split(r.key, 3)
        p_ = r.params
        w0 = r.walk
        action, logp, walk, info = act(model, p_, w0, r.obs, entropy, r.cache, k_act)
        z, g = info["z"], info["g"]
        next_obs, env_state, r_env, term, trunc, einfo = v_step(
            jax.random.split(k_env, r.obs.shape[0]), r.env_state, action, env_cfg, cfg.resample_drift)
        z_next = model.m(p_, DFModel.encode, einfo["final_obs"])
        task = _cos(z_next - z, g)
        value = model.value(p_, r.obs, z, g)
        reward = cfg.env_reward_coef * r_env + cfg.task_reward_coef * task
        reward = reward + cfg.gamma * model.value(p_, einfo["final_obs"], z_next, g) * (trunc & ~term)
        done = term | trunc
        tr = DFTransition(r.obs, einfo["final_obs"], z, g, w0.a_prev, w0.r_prev, w0.first, action, logp, value,
                          _f32(reward), _f32(r_env), task, info["v_mean"], walk.gin, walk.graw, walk.glogp,
                          info["new_goal"], done, walk.tau, walk.sig, walk.eps, info["p"], info["v_std"],
                          info["search_gain"], info["vetoed"], einfo["JT"])
        walk = walk._replace(a_prev=_f32(jnp.clip(action, -1, 1)), r_prev=_f32(r_env), first=done)
        return r._replace(env_state=env_state, obs=next_obs, key=key, walk=walk), tr

    return jax.lax.scan(body, runner, None, n_steps)


# --8<-- [start:model_loss]
def model_loss(model: Dragonfly, mparams, seq: DFTransition, traces0, z_prev0, key):
    """seq: (T, B, ...) env sequences; traces0 / z_prev0: their trace state before t = 0."""
    cfg = model.cfg
    params = dict(model=mparams)
    z = model.m(params, DFModel.encode, seq.obs)
    z_next = jax.lax.stop_gradient(model.m(params, DFModel.encode, seq.next_obs))
    keep = (~seq.first)[..., None]
    z_prev = jnp.concatenate([z_prev0[None], z[:-1]], 0)
    z_prev = jnp.where(keep, z_prev, z)
    x = jnp.concatenate([z, z - z_prev, seq.a_prev * keep, (seq.r_prev * cfg.model_reward_scale)[..., None] * keep], -1)
    dec = jnp.asarray(cfg.trace_decays, jnp.float32)
    a = dec[None, None, :, None] * keep[..., None]                      # (T, B, K, 1)
    b = (1 - dec)[None, None, :, None] * x[:, :, None, :]               # (T, B, K, D)
    A, Bc = jax.lax.associative_scan(lambda e1, e2: (e1[0] * e2[0], e2[0] * e1[1] + e2[1]), (a, b), axis=0)
    traces = A * jax.lax.stop_gradient(traces0)[None] + Bc
    c = model.m(params, DFModel.context, traces)

    T, B = seq.done.shape
    k_gam, k_mask = jax.random.split(key)
    gamma = jax.random.choice(k_gam, jnp.asarray(cfg.gammas, jnp.float32), (B,))
    gam_tb = jnp.broadcast_to(gamma, (T, B))
    y = gamma_targets(z_next, seq.done, gamma)
    jump = model.m(params, DFModel.jump, z, c, seq.action, gam_tb)
    jump_loss = jnp.mean(jnp.sum((jump - jax.lax.stop_gradient(y - z)) ** 2, -1))
    jump_rel = jump_loss / (jnp.mean(jnp.sum(jax.lax.stop_gradient(y - z) ** 2, -1)) + 1e-8)

    q = model.m(params, DFModel.q, z, c, seq.action, gam_tb)            # (T, B, H)
    v = model.m(params, DFModel.v, z, c, gam_tb)
    G = jax.lax.stop_gradient(lambda_returns(seq.r_env * cfg.model_reward_scale, v.mean(-1), seq.done, gamma,
                                             cfg.model_lambda))[..., None]
    mask = jax.random.bernoulli(k_mask, cfg.head_keep, q.shape).astype(jnp.float32)
    value_loss = jnp.sum(mask * ((q - G) ** 2 + (v - G) ** 2)) / jnp.maximum(mask.sum(), 1.0)

    zf = z.reshape(-1, z.shape[-1])
    zc = zf - zf.mean(0)
    var_loss = jnp.mean(jax.nn.relu(1 - jnp.sqrt(zc.var(0) + 1e-4)))
    cov = zc.T @ zc / (zf.shape[0] - 1)
    cov_loss = (jnp.sum(cov ** 2) - jnp.sum(jnp.diag(cov) ** 2)) / zf.shape[-1]
    loss = jump_loss + value_loss + cfg.vicreg_coef * (var_loss + cov_loss)
    stats = dict(m_jump=jump_loss, m_jump_rel=jump_rel, m_value=value_loss, m_var=var_loss, m_cov=cov_loss,
                 m_head_std=jnp.mean(q.std(-1)))
    return loss, {k: v_.astype(jnp.float32) for k, v_ in stats.items()}
# --8<-- [end:model_loss]


MODEL_STATS = ("m_jump", "m_jump_rel", "m_value", "m_var", "m_cov", "m_head_std")


# --8<-- [start:cache]
def update_cache(model: Dragonfly, params, cache: Cache, obs, g, action, r, G, v):
    """Merge this rollout's best (by reward and by return) into the canonical golden cache."""
    cfg = model.cfg
    M, K = cfg.cache_candidates, cfg.cache_size
    idx = jnp.concatenate([jnp.argsort(-G)[:M], jnp.argsort(-r)[:M]])
    pool = Cache(*(jnp.concatenate([a, b[idx]]) for a, b in zip(
        cache, (obs, jnp.zeros((obs.shape[0], cfg.latent_dim), jnp.float32), g, action, r, G, v,
                jnp.ones(obs.shape[0], bool)))))
    pool = pool._replace(z=model.m(params, DFModel.encode, pool.obs))
    big = 1e9
    norm = lambda x: (x - jnp.min(jnp.where(pool.valid, x, big))) / (
        jnp.max(jnp.where(pool.valid, x, -big)) - jnp.min(jnp.where(pool.valid, x, big)) + 1e-6)
    score = jnp.where(pool.valid, 0.5 * (norm(pool.r) + norm(pool.G)), -jnp.inf)
    order = jnp.argsort(-score)

    def body(i, carry):
        accepted, count = carry
        j = order[i]
        dist = jnp.linalg.norm(pool.z - pool.z[j], axis=-1)
        ok = pool.valid[j] & ~jnp.any(accepted & (dist < cfg.cache_radius)) & (count < K)
        return accepted.at[j].set(ok), count + ok

    accepted, count = jax.lax.fori_loop(0, score.shape[0], body, (jnp.zeros(score.shape[0], bool), 0))
    sel = jnp.nonzero(accepted, size=K, fill_value=0)[0]
    new = Cache(*(a[sel] for a in pool))
    return new._replace(valid=jnp.arange(K) < count)
# --8<-- [end:cache]


class DFOpt(NamedTuple):
    policy: Any
    model: Any


def _policy_part(params):
    return {k: params[k] for k in ("actor", "critic", "guide")}


def init(key, env_cfg: jenv.EnvConfig, cfg: DragonflyConfig):
    model = Dragonfly.make(env_cfg, cfg)
    k_net, k_env, key = jax.random.split(key, 3)
    params = model.init(k_net)
    obs, env_state = jax.vmap(jenv.reset, in_axes=(0, None))(jax.random.split(k_env, cfg.n_envs), env_cfg)
    n, A, d, K = cfg.n_envs, env_cfg.act_dim, cfg.latent_dim, cfg.cache_size
    f = lambda *s: jnp.zeros(s, jnp.float32)
    walk = Walk(f(n, len(cfg.trace_decays), model.trace_in), f(n, d), f(n, A), f(n), jnp.ones(n, bool), f(n, d),
                f(n, d), jnp.ones((n, d), jnp.float32), f(n, d), f(n), f(n, cfg.guide_dim), f(n), f(n, model.gin_dim),
                jnp.full(n, cfg.goal_every, jnp.int32))
    cache = Cache(f(K, env_cfg.obs_dim), f(K, d), f(K, d), f(K, A), f(K), f(K), f(K), jnp.zeros(K, bool))
    runner = DFRunner(params, env_state, obs, key, walk, cache, f(d), jnp.ones(d, jnp.float32), jnp.zeros((), jnp.int32))
    opt = DFOpt(ppo.make_optimizer(cfg).init(_policy_part(params)),
                optax.chain(optax.clip_by_global_norm(1.0), optax.adam(cfg.model_lr)).init(params["model"]))
    return model, runner, opt


def _masked_corr(x, y, m):
    n = jnp.maximum(m.sum(), 1.0)
    mx, my = (m * x).sum() / n, (m * y).sum() / n
    cxy = (m * (x - mx) * (y - my)).sum() / n
    return cxy / jnp.sqrt((m * (x - mx) ** 2).sum() / n * (m * (y - my) ** 2).sum() / n + 1e-12)


def make_update(model: Dragonfly, env_cfg: jenv.EnvConfig, cfg: DragonflyConfig):
    tx = ppo.make_optimizer(cfg)
    mtx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(cfg.model_lr))
    n_groups = cfg.n_envs // cfg.model_envs

    def policy_loss(pp, mb, cache: Cache, sil_w):
        params = dict(pp)
        obs, z, g, action, logp_old, adv, ret, gin, graw, glogp_old, gadv, goal = mb
        mean = model.pi_mean(params, obs, z, g)
        log_std = params["actor"]["log_std"]
        ratio = jnp.exp(gaussian_logp(mean, log_std, action) - logp_old)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        pg = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv).mean()
        v_loss = ((ret - model.value(params, obs, z, g)) ** 2).mean()

        gratio = jnp.exp(gaussian_logp(model.guide_mean(params, gin), params["guide"]["log_std"], graw) - glogp_old)
        gsur = jnp.minimum(gratio * gadv, jnp.clip(gratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * gadv)
        guide_loss = -(goal * gsur).sum() / jnp.maximum(goal.sum(), 1.0)
        guide_ent = jnp.sum(params["guide"]["log_std"] + 0.5 * jnp.log(2 * jnp.pi * jnp.e))

        sil_lp = gaussian_logp(model.pi_mean(params, cache.obs, cache.z, cache.g), log_std, cache.action)
        sil = -(sil_w * sil_lp).sum() / jnp.maximum((sil_w > 0).sum(), 1.0)

        loss = pg + cfg.vf_coef * v_loss + guide_loss - cfg.guide_ent_coef * guide_ent + cfg.sil_coef * sil
        return loss, dict(pg_loss=pg, v_loss=v_loss, guide_loss=guide_loss, sil_loss=sil,
                          approx_kl=((ratio - 1) - jnp.log(ratio)).mean(),
                          clip_frac=(jnp.abs(ratio - 1) > cfg.clip_eps).mean())

    @jax.jit
    def update(runner: DFRunner, opt: DFOpt):
        traces0, z_prev0 = runner.walk.traces, runner.walk.z_prev
        runner, traj = collect(model, runner, env_cfg, cfg.n_steps)
        p = runner.params
        z_last, _, c_last = model.observe(p, runner.walk, runner.obs)
        # (the walk may be about to set a new goal there; the current task is the best available)
        g_last = jnp.sqrt(1 - runner.walk.eps)[..., None] * runner.walk.u
        adv, ret = gae_arrays(traj.reward, traj.value, traj.done, model.value(p, runner.obs, z_last, g_last),
                              cfg.gamma, cfg.gae_lambda)
        last_vm = model.m(p, DFModel.v, z_last, c_last, jnp.full(z_last.shape[:-1], cfg.gamma)).mean(-1)
        r_s = traj.r_env * cfg.model_reward_scale
        gadv_raw, G_env = gae_arrays(r_s, traj.v_model, traj.done, last_vm, cfg.gamma, cfg.gae_lambda)
        goal = traj.new_goal.astype(jnp.float32)
        gm = (gadv_raw * goal).sum() / goal.sum()
        gadv = (gadv_raw - gm) / (jnp.sqrt((((gadv_raw - gm) * goal) ** 2).sum() / goal.sum()) + 1e-8)

        flat = lambda x: x.reshape((cfg.batch_size,) + x.shape[2:])
        cache = update_cache(model, p, runner.cache, _f32(flat(traj.obs)), flat(traj.g), flat(traj.action), flat(r_s),
                             flat(G_env), flat(traj.v_model))
        sil_w = jnp.where(cache.valid, jax.nn.relu(cache.G - cache.v), 0.0)
        sil_w = sil_w / (sil_w.sum() / jnp.maximum((sil_w > 0).sum(), 1.0) + 1e-8)

        data = tuple(flat(x) for x in (traj.obs, traj.z, traj.g, traj.action, traj.logp, adv, ret, traj.gin, traj.graw,
                                       traj.glogp, gadv, goal))
        warm = runner.n_updates < cfg.model_warmup

        def epoch(carry, x):
            key, ep = x
            k_perm, k_env, k_m = jax.random.split(key, 3)
            perm = jax.random.permutation(k_perm, cfg.batch_size)
            mbs = jax.tree_util.tree_map(lambda a: a[perm].reshape((cfg.n_minibatches, -1) + a.shape[1:]), data)
            envs = jax.random.permutation(k_env, cfg.n_envs)
            groups = jnp.tile(envs.reshape(n_groups, cfg.model_envs), (cfg.n_minibatches // n_groups + 1, 1))
            groups = groups[:cfg.n_minibatches]

            def minibatch(carry, x):
                params, popt, mopt = carry
                mb, grp, k = x
                pparams = _policy_part(params)
                (_, pstats), grads = jax.value_and_grad(policy_loss, has_aux=True)(pparams, mb, cache, sil_w)
                upd, new_popt = tx.update(grads, popt, pparams)
                keep = lambda new, old: jax.tree_util.tree_map(lambda n, o: jnp.where(warm, o, n), new, old)
                pparams, popt = keep(optax.apply_updates(pparams, upd), pparams), keep(new_popt, popt)

                def model_step(args):
                    mparams, mopt = args
                    seq = jax.tree_util.tree_map(lambda a: a[:, grp], traj)
                    (_, mstats), mgrads = jax.value_and_grad(
                        lambda mp: model_loss(model, mp, seq, traces0[grp], z_prev0[grp], k), has_aux=True)(mparams)
                    mupd, mopt = mtx.update(mgrads, mopt, mparams)
                    return optax.apply_updates(mparams, mupd), mopt, mstats

                def skip(args):
                    return args[0], args[1], {k_: jnp.full((), jnp.nan, jnp.float32) for k_ in MODEL_STATS}

                mparams, mopt, mstats = jax.lax.cond(ep < cfg.model_epochs, model_step, skip, (params["model"], mopt))
                return (dict(pparams, model=mparams), popt, mopt), dict(pstats, **mstats)

            return jax.lax.scan(minibatch, carry, (mbs, groups, jax.random.split(k_m, cfg.n_minibatches)))

        key, k_ep = jax.random.split(runner.key)
        (params, popt, mopt), stats = jax.lax.scan(
            epoch, (runner.params, opt.policy, opt.model), (jax.random.split(k_ep, cfg.n_epochs), jnp.arange(cfg.n_epochs)))
        stats = jax.tree_util.tree_map(jnp.nanmean, stats)

        zf = flat(traj.z)
        a = cfg.entropy_ema
        lat_mean = a * runner.lat_mean + (1 - a) * zf.mean(0)
        lat_var = a * runner.lat_var + (1 - a) * zf.var(0)
        runner = runner._replace(params=params, key=key, cache=cache, lat_mean=lat_mean, lat_var=lat_var,
                                 n_updates=runner.n_updates + 1)

        gsum = goal.sum()
        tau_bar = traj.tau.mean(-1)
        tau_dim = (traj.tau * goal[..., None]).sum((0, 1)) / gsum
        ent = measured_entropy(runner.lat_var)
        stats.update(mean_reward=traj.r_env.mean(), min_JT=traj.JT.min(), frac_done=traj.done.mean(),
                     task_reward=traj.task.mean(), policy_reward=traj.reward.mean(),
                     tau_mean=(tau_bar * goal).sum() / gsum, sigma_mean=(traj.sig.mean(-1) * goal).sum() / gsum,
                     eps_mean=(traj.eps * goal).sum() / gsum, p_mean=(traj.p * goal).sum() / gsum,
                     tau_corr_p=_masked_corr(tau_bar, traj.p, goal),
                     tau_corr_entropy=jnp.corrcoef(tau_dim, ent)[0, 1], entropy_mean=ent.mean(),
                     v_std=traj.v_std.mean(), search_gain=traj.search_gain.mean(), vetoed=traj.vetoed.mean(),
                     cache_n=cache.valid.sum().astype(jnp.float32),
                     cache_G_best=jnp.max(jnp.where(cache.valid, cache.G, -jnp.inf)),
                     cache_r_best=jnp.max(jnp.where(cache.valid, cache.r, -jnp.inf)),
                     log_std=params["actor"]["log_std"].mean(), warmup=warm.astype(jnp.float32))
        return runner, DFOpt(popt, mopt), stats

    return update


def evaluate(model: Dragonfly, params, cfg: jenv.EnvConfig, omega_s=None, stop_on_truncation=True):
    """Deterministic episode: guide means, eps = 0 (pure exploit), empty cache (p = 0), search over the mean
    action and fixed-key samples, measured entropy of a unit-variance latent."""
    mc = model.cfg
    omega_s = jnp.asarray(cfg.model.omega_s if omega_s is None else omega_s)
    obs, state = jenv.reset_to(omega_s, cfg)
    f = lambda *s: jnp.zeros(s, jnp.float32)
    d, A, K = mc.latent_dim, model.act_dim, mc.cache_size
    walk = Walk(f(1, len(mc.trace_decays), model.trace_in), f(1, d), f(1, A), f(1), jnp.ones(1, bool), f(1, d), f(1, d),
                jnp.ones((1, d), jnp.float32), f(1, d), f(1), f(1, mc.guide_dim), f(1), f(1, model.gin_dim),
                jnp.full(1, mc.goal_every, jnp.int32))
    cache = Cache(f(K, model.obs_dim), f(K, d), f(K, d), f(K, A), f(K), f(K), f(K), jnp.zeros(K, bool))
    entropy = measured_entropy(jnp.ones(d, jnp.float32))

    def body(carry, _):
        obs, state, alive, walk, key = carry
        key, k = jax.random.split(key)
        action, _, walk, _ = act(model, params, walk, obs[None], entropy, cache, k, deterministic=True)
        obs, state, r, term, trunc, info = jenv.step(state, action[0], cfg)
        walk = walk._replace(a_prev=_f32(jnp.clip(action, -1, 1)), r_prev=jnp.atleast_1d(r).astype(jnp.float32),
                             first=jnp.zeros(1, bool))
        out = dict(reward=r, JT=info["JT"], concurrence=info["concurrence"], unitarity=info["unitarity"],
                   t=info["t"], amps=state.amps_cur, alive=alive)
        ended = term | trunc if stop_on_truncation else term
        return (obs, state, alive & ~ended, walk, key), out

    _, out = jax.lax.scan(body, (obs, state, jnp.array(True), walk, jax.random.PRNGKey(0)), None, cfg.n_steps)
    return out
