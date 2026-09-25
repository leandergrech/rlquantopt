"""PPO plus three switchable ingredients (idea i05 echidna), for any env with the toy_envs interface.

- **OU exploration noise** (``ou_rho > 0``): the action noise is an Ornstein-Uhlenbeck process with a unit
  marginal, ``eps_t = rho eps_{t-1} + sqrt(1 - rho^2) eta_t``, ``a_t = mu(s_t) + sigma eps_t``, restarted at
  every episode. The policy is conditioned on the previous noise: ``pi(a | s, eps_prev) =
  N(mu(s) + sigma rho eps_prev, sigma^2 (1 - rho^2))``, with eps_prev stored in the rollout, so the PPO
  ratio stays an exact likelihood ratio under the new mu and sigma. ``rho = 0`` is plain PPO.
- **Pessimistic veto** (``veto``): an ensemble of ``n_heads`` Q networks (own optimiser, each head on a
  bootstrap mask) regresses the lambda-return of (s, a). ``n_candidates`` actions are drawn from pi; the
  first is taken unless its lower bound ``mean - lcb_kappa * std`` is more than ``veto_delta`` (in units of
  the batch return spread) below the best candidate's, in which case the first candidate that is not is
  taken. Off only after ``veto_warmup`` updates. When it fires, the executed action is no longer a plain
  sample of pi (the PPO ratio still uses pi's likelihood); ``veto_rate`` logs how often.
- **Golden-cache self-imitation** (``sil_coef > 0``): a canonical archive of the ``cache_size`` best
  transitions by return and by reward (at most one per ball of radius ``cache_radius`` in standardised obs
  space), refreshed every rollout. The loss ``-log pi(a|s) w - ...`` imitates cached actions with weight
  ``w = relu(G - V(s)) / std(returns)`` and pulls V up towards G where it is below (Oh et al. 2018).
"""
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
import optax

from rlquantopt.jx.agents import ppo
from rlquantopt.jx.agents.common import MLP, ActorCritic, gae
from rlquantopt.jx.toy_envs import step_autoreset


@dataclass(frozen=True)
class PlusConfig(ppo.PPOConfig):
    ou_rho: float = 0.0
    veto: bool = False
    n_candidates: int = 8
    n_heads: int = 4
    head_keep: float = 0.8
    lcb_kappa: float = 1.0
    veto_delta: float = 0.5
    veto_warmup: int = 5
    q_hidden: tuple = (64, 64)
    q_lr: float = 1e-3
    sil_coef: float = 0.0
    sil_value_coef: float = 0.01
    cache_size: int = 64
    cache_radius: float = 0.1
    cache_candidates: int = 64
    critic_warmup: int = 0          # first updates train only the critic (fine-tuning on a new device)


def _f32(x):
    return jnp.asarray(x).astype(jnp.float32)


def cond_dist(mean, log_std, eps_prev, first, rho):
    """Mean and log-std of pi(a | s, eps_prev); the first step of an episode has no memory."""
    r = rho * (1.0 - first.astype(jnp.float32))[..., None]
    std = jnp.exp(log_std)
    return mean + std * r * eps_prev, log_std + 0.5 * jnp.log(1 - r ** 2)


def cond_logp(mean, log_std, a):
    return jnp.sum(-0.5 * ((a - mean) / jnp.exp(log_std)) ** 2 - log_std - 0.5 * jnp.log(2 * jnp.pi), axis=-1)


class QEnsemble(NamedTuple):
    heads: tuple

    @classmethod
    def make(cls, n, hidden):
        return cls(tuple(MLP(tuple(hidden), 1, 1.0, "tanh") for _ in range(n)))

    def init(self, key, in_dim):
        return [h.init(k, jnp.zeros((1, in_dim), jnp.float32)) for h, k in zip(self.heads, jax.random.split(key, len(self.heads)))]

    def __call__(self, qparams, obs, action):
        x = jnp.concatenate([_f32(obs), _f32(jnp.clip(action, -1, 1))], -1)
        return jnp.stack([h.apply(p, x)[..., 0] for h, p in zip(self.heads, qparams)], -1)


class Cache(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    eps_prev: jnp.ndarray
    first: jnp.ndarray
    G: jnp.ndarray
    r: jnp.ndarray
    valid: jnp.ndarray


class PlusRunner(NamedTuple):
    params: dict            # actor, critic (the ActorCritic) and q (the ensemble)
    env_state: Any
    obs: jnp.ndarray
    key: jnp.ndarray
    eps_prev: jnp.ndarray
    first: jnp.ndarray
    cache: Cache
    ret_std: jnp.ndarray    # spread of the last batch's returns (veto and SIL scale)
    n_updates: jnp.ndarray


class PlusOpt(NamedTuple):
    policy: Any
    q: Any


class Traj(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    logp: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    eps_prev: jnp.ndarray
    first: jnp.ndarray
    vetoed: jnp.ndarray
    score: jnp.ndarray


class Plus(NamedTuple):
    ac: ActorCritic
    q: QEnsemble
    env: Any
    cfg: PlusConfig

    def mode(self, actor_params, obs):
        """Deterministic action (the mean), used by common.evaluate on the gate env."""
        return self.ac.mode(actor_params, obs)


def init(key, env, cfg: PlusConfig):
    ac = ActorCritic.make(env.act_dim, cfg.hidden, cfg.activation)
    q = QEnsemble.make(cfg.n_heads, cfg.q_hidden)
    model = Plus(ac, q, env, cfg)
    k_net, k_q, k_env, key = jax.random.split(key, 4)
    params = ac.init(k_net, env.obs_dim)
    params["q"] = q.init(k_q, env.obs_dim + env.act_dim)
    obs, env_state = jax.vmap(env.reset)(jax.random.split(k_env, cfg.n_envs))
    n, A, K = cfg.n_envs, env.act_dim, cfg.cache_size
    f = lambda *s: jnp.zeros(s, jnp.float32)
    cache = Cache(f(K, env.obs_dim), f(K, A), f(K, A), jnp.ones(K, bool), f(K), f(K), jnp.zeros(K, bool))
    runner = PlusRunner(params, env_state, obs, key, f(n, A), jnp.ones(n, bool), cache, jnp.ones((), jnp.float32),
                        jnp.zeros((), jnp.int32))
    pol = {k: params[k] for k in ("actor", "critic")}
    opt = PlusOpt(ppo.make_optimizer(cfg).init(pol), optax.adam(cfg.q_lr).init(params["q"]))
    return model, runner, opt


# --8<-- [start:act]
def act(model: Plus, params, obs, eps_prev, first, key, veto_on, ret_std):
    cfg = model.cfg
    mean, log_std = model.ac.dist(params["actor"], obs)
    cmean, clog_std = cond_dist(mean, log_std, eps_prev, first, cfg.ou_rho)
    N = cfg.n_candidates if cfg.veto else 1
    cands = cmean + jnp.exp(clog_std) * jax.random.normal(key, (N,) + mean.shape, jnp.float32)
    choice = jnp.zeros(mean.shape[:-1], jnp.int32)
    vetoed = jnp.zeros(mean.shape[:-1], bool)
    if cfg.veto:
        qs = model.q(params["q"], jnp.broadcast_to(obs, (N,) + obs.shape), cands)
        lcb = qs.mean(-1) - cfg.lcb_kappa * qs.std(-1)                     # (N, B)
        ok = lcb >= lcb.max(0) - cfg.veto_delta * ret_std
        vetoed = veto_on & ~ok[0]
        choice = jnp.where(vetoed, jnp.argmax(ok, 0), 0)
    action = jnp.take_along_axis(cands, choice[None, :, None], 0)[0]
    logp = cond_logp(cmean, clog_std, action)
    eps = (action - mean) / jnp.exp(log_std)                              # the noise actually used
    return action, logp, eps, vetoed
# --8<-- [end:act]


def collect(model: Plus, runner: PlusRunner, n_steps):
    cfg, env = model.cfg, model.env
    v_step = jax.vmap(lambda k, s, a: step_autoreset(env, k, s, a))
    veto_on = runner.n_updates >= cfg.veto_warmup

    def body(r: PlusRunner, _):
        key, k_act, k_env = jax.random.split(r.key, 3)
        action, logp, eps, vetoed = act(model, r.params, r.obs, r.eps_prev, r.first, k_act, veto_on, r.ret_std)
        value = model.ac.value(r.params["critic"], r.obs)
        next_obs, env_state, reward, term, trunc, info = v_step(jax.random.split(k_env, r.obs.shape[0]),
                                                               r.env_state, action)
        reward = reward + cfg.gamma * model.ac.value(r.params["critic"], info["final_obs"]) * (trunc & ~term)
        done = term | trunc
        tr = Traj(_f32(r.obs), action, logp, value, _f32(reward), done, r.eps_prev, r.first, vetoed, _f32(info["score"]))
        return r._replace(env_state=env_state, obs=next_obs, key=key, eps_prev=eps, first=done), tr

    return jax.lax.scan(body, runner, None, n_steps)


# --8<-- [start:cache]
def update_cache(cfg: PlusConfig, cache: Cache, obs, action, eps_prev, first, G, r):
    M, K = cfg.cache_candidates, cfg.cache_size
    idx = jnp.concatenate([jnp.argsort(-G)[:M], jnp.argsort(-r)[:M]])
    pool = Cache(*(jnp.concatenate([a, b[idx]]) for a, b in zip(
        cache, (obs, action, eps_prev, first, G, r, jnp.ones(obs.shape[0], bool)))))
    scale = obs.std(0) + 1e-6
    big = 1e9
    norm = lambda x: (x - jnp.min(jnp.where(pool.valid, x, big))) / (
        jnp.max(jnp.where(pool.valid, x, -big)) - jnp.min(jnp.where(pool.valid, x, big)) + 1e-6)
    score = jnp.where(pool.valid, 0.5 * (norm(pool.G) + norm(pool.r)), -jnp.inf)
    order = jnp.argsort(-score)
    zs = pool.obs / scale

    def body(i, carry):
        acc, count = carry
        j = order[i]
        dist = jnp.linalg.norm(zs - zs[j], axis=-1) / jnp.sqrt(zs.shape[-1])
        ok = pool.valid[j] & ~jnp.any(acc & (dist < cfg.cache_radius)) & (count < K)
        return acc.at[j].set(ok), count + ok

    acc, count = jax.lax.fori_loop(0, score.shape[0], body, (jnp.zeros(score.shape[0], bool), 0))
    new = Cache(*(a[jnp.nonzero(acc, size=K, fill_value=0)[0]] for a in pool))
    return new._replace(valid=jnp.arange(K) < count)
# --8<-- [end:cache]


def make_update(model: Plus, cfg: PlusConfig):
    tx = ppo.make_optimizer(cfg)
    qtx = optax.adam(cfg.q_lr)
    use_sil = cfg.sil_coef > 0

    def policy_loss(pp, mb, cache: Cache, ret_std):
        obs, action, logp_old, adv, ret, eps_prev, first = mb
        mean, log_std = model.ac.dist(pp["actor"], obs)
        cm, cls = cond_dist(mean, log_std, eps_prev, first, cfg.ou_rho)
        ratio = jnp.exp(cond_logp(cm, cls, action) - logp_old)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        pg = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv).mean()
        v_loss = ((ret - model.ac.value(pp["critic"], obs)) ** 2).mean()
        loss = pg + cfg.vf_coef * v_loss
        stats = dict(pg_loss=pg, v_loss=v_loss, approx_kl=((ratio - 1) - jnp.log(ratio)).mean(),
                     clip_frac=(jnp.abs(ratio - 1) > cfg.clip_eps).mean())
        if use_sil:
            m, ls = model.ac.dist(pp["actor"], cache.obs)
            cm, cls = cond_dist(m, ls, cache.eps_prev, cache.first, cfg.ou_rho)
            v = model.ac.value(pp["critic"], cache.obs)
            gap = jnp.where(cache.valid, jax.nn.relu(cache.G - v), 0.0)
            w = jax.lax.stop_gradient(gap) / ret_std
            n = jnp.maximum((w > 0).sum(), 1.0)
            sil = -(w * cond_logp(cm, cls, cache.action)).sum() / n
            sil_v = 0.5 * (gap ** 2).sum() / n
            loss = loss + cfg.sil_coef * (sil + cfg.sil_value_coef * sil_v)
            stats.update(sil_loss=sil, sil_active=(w > 0).mean())
        return loss, stats

    def q_loss(qp, mb, key):
        obs, action, _, _, ret, _, _ = mb
        qs = model.q(qp, obs, action)
        mask = jax.random.bernoulli(key, cfg.head_keep, qs.shape).astype(jnp.float32)
        return jnp.sum(mask * (qs - ret[:, None]) ** 2) / jnp.maximum(mask.sum(), 1.0)

    def update(runner: PlusRunner, opt: PlusOpt):
        runner, traj = collect(model, runner, cfg.n_steps)
        p = runner.params
        last_value = model.ac.value(p["critic"], runner.obs)
        # gae() reads reward, value and done from the trajectory
        adv, ret = gae(traj, last_value, cfg.gamma, cfg.gae_lambda)
        flat = lambda x: x.reshape((cfg.batch_size,) + x.shape[2:])
        data = tuple(flat(x) for x in (traj.obs, traj.action, traj.logp, adv, ret, traj.eps_prev, traj.first))
        ret_std = jnp.std(ret) + 1e-6
        cache = runner.cache
        if use_sil:
            cache = update_cache(cfg, cache, *(flat(x) for x in (traj.obs, traj.action, traj.eps_prev, traj.first)),
                                 flat(ret), flat(traj.reward))

        def epoch(carry, key):
            k_perm, k_q = jax.random.split(key)
            perm = jax.random.permutation(k_perm, cfg.batch_size)
            mbs = jax.tree_util.tree_map(lambda x: x[perm].reshape((cfg.n_minibatches, -1) + x.shape[1:]), data)

            def minibatch(carry, x):
                params, popt, qopt = carry
                mb, k = x
                pol = {k_: params[k_] for k_ in ("actor", "critic")}
                (_, stats), grads = jax.value_and_grad(policy_loss, has_aux=True)(pol, mb, cache, ret_std)
                if cfg.critic_warmup:
                    frozen = runner.n_updates < cfg.critic_warmup
                    grads = dict(grads, actor=jax.tree_util.tree_map(lambda g: jnp.where(frozen, 0.0 * g, g), grads["actor"]))
                upd, popt = tx.update(grads, popt, pol)
                pol = optax.apply_updates(pol, upd)
                qp = params["q"]
                if cfg.veto:
                    ql, qg = jax.value_and_grad(q_loss)(qp, mb, k)
                    qu, qopt = qtx.update(qg, qopt, qp)
                    qp = optax.apply_updates(qp, qu)
                    stats = dict(stats, q_loss=ql)
                return (dict(pol, q=qp), popt, qopt), stats

            return jax.lax.scan(minibatch, carry, (mbs, jax.random.split(k_q, cfg.n_minibatches)))

        key, k_ep = jax.random.split(runner.key)
        (params, popt, qopt), stats = jax.lax.scan(epoch, (p, opt.policy, opt.q), jax.random.split(k_ep, cfg.n_epochs))
        stats = jax.tree_util.tree_map(jnp.mean, stats)
        stats.update(mean_reward=traj.reward.mean(), mean_score=traj.score.mean(), frac_done=traj.done.mean(),
                     veto_rate=traj.vetoed.mean(), log_std=params["actor"]["log_std"].mean(),
                     cache_n=cache.valid.sum().astype(jnp.float32))
        runner = runner._replace(params=params, key=key, cache=cache, ret_std=ret_std, n_updates=runner.n_updates + 1)
        return runner, PlusOpt(popt, qopt), stats

    return update


def evaluate_return(model: Plus, params, key, n_episodes=8):
    """Mean undiscounted return of deterministic (mean-action) episodes from ``n_episodes`` random starts."""
    env = model.env

    def episode(k):
        obs, s = env.reset(k)

        def body(carry, _):
            obs, s, alive, ret, score = carry
            obs, s, r, term, trunc, info = env.step(s, model.ac.mode(params["actor"], obs))
            return (obs, s, alive & ~(term | trunc), ret + _f32(r) * alive, score + _f32(info["score"]) * alive), None

        zero = jnp.zeros((), jnp.float32)
        (_, _, _, ret, score), _ = jax.lax.scan(body, (obs, s, jnp.array(True), zero, zero), None, env.max_steps)
        return ret, score

    ret, score = jax.vmap(episode)(jax.random.split(key, n_episodes))
    return ret.mean(), score.mean()
