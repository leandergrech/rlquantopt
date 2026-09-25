"""Idea i03 "chameleon": a policy that follows latent tasks set by a jumpy, gamma-conditioned world model.

**World model** (its own optimiser, trained on every PPO minibatch):

- encoder ``z = E(s)``; VICReg variance/covariance terms keep the latent axes decorrelated and unit-scale,
  so nearby experiences map to nearby latents and the axes carry the task's main variations;
- history ``c_t = GRU(c_{t-1}, [z_t, a_{t-1}, r_{t-1}])``, reset at episode starts, so it summarises a
  history of any length;
- jumpy model ``F(z, c, a, gamma)``: in one call, the displacement to the gamma-discounted future latent
  ``y_t = sum_{j>=1} (1-gamma) gamma^(j-1) z_{t+j}``, for any gamma (trained on ``gammas``);
- values ``Q(z, c, a, gamma)`` and ``V(z, c, gamma)``, regressed on the TD(lambda) return of the env
  reward at that gamma (bootstrapped with V), i.e. the model's own GAE-style estimates. Q - V is its
  estimate of how much an action choice moves training.

**Planner**, every ``goal_every`` steps (and at episode start):

- exploit direction ``u = grad_z V(z, c, plan_gamma) / |.|``: where in latent space value rises fastest;
- explore direction ``xi``: a uniformly random unit vector in the subspace orthogonal to u (maximum
  entropy there, and exploring never undoes exploiting);
- a learned gate ``eps ~ Beta(G(c, z))`` sets the mix: ``g = sqrt(1-eps) u + sqrt(eps) xi`` (unit length,
  eps is exactly the share of exploration). The gate is trained with a clipped PPO surrogate on the GAE
  advantage of the env reward at the goal step, with the model's V as baseline.

**Policy** ``pi(a | s, z, g)`` proposes ``n_candidates`` actions; the jumpy model re-ranks them by
``cos(F(z, c, a, plan_gamma), g)`` (does the action move the latent where the task asks) and the best one
is executed. The policy is trained with PPO on
``env_reward_coef * r_env + task_reward_coef * cos(z_{t+1} - z_t, g)``. Its PPO ratio uses pi's own
likelihood of the executed action, so with search the update also imitates the planner's picks
(weighted by advantage); ``n_candidates=1`` is plain sampling.

The first ``model_warmup`` updates train only the world model (the policy and gate stay at their
initialisation) to calibrate the latent space before the policy is asked to follow it.
"""
from dataclasses import dataclass
from typing import Any, NamedTuple, Sequence

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from jax.scipy.special import betaln, digamma
from jax.scipy.stats import beta as beta_dist

from rlquantopt.jx import env as jenv
from rlquantopt.jx.agents import ppo
from rlquantopt.jx.agents.common import MLP, gaussian_logp


@dataclass(frozen=True)
class ChameleonConfig(ppo.PPOConfig):
    activation: str = "leaky_relu"
    latent_dim: int = 16
    enc_hidden: tuple = (128,)
    hist_dim: int = 64
    model_hidden: tuple = (128, 128)
    gammas: tuple = (0.5, 0.8, 0.9, 0.95, 0.99)     # horizons the jumpy model is trained on
    plan_gamma: float = 0.9
    model_lambda: float = 0.95
    goal_every: int = 8
    n_candidates: int = 8
    task_reward_coef: float = 1.0
    env_reward_coef: float = 1.0
    gate: bool = True               # False: fixed_eps instead of the learned gate
    fixed_eps: float = 0.0
    gate_ent_coef: float = 1e-3
    model_lr: float = 3e-4
    model_epochs: int = 4           # PPO epochs during which every minibatch also updates the model
    model_envs: int = 8             # env sequences per model minibatch
    model_warmup: int = 5           # updates that train only the model
    model_reward_scale: float = 0.1
    vicreg_coef: float = 1.0


def _f32(x):
    return jnp.asarray(x).astype(jnp.float32)


def gamma_feature(gamma):
    """Horizon on a log scale: 0.15 for gamma = 0.5, 1 for gamma = 0.99."""
    return jnp.log(1.0 / (1.0 - gamma)) / jnp.log(100.0)


class WorldModel(nn.Module):
    latent_dim: int
    enc_hidden: Sequence[int]
    hist_dim: int
    model_hidden: Sequence[int]
    activation: str

    def setup(self):
        self.enc = MLP(tuple(self.enc_hidden), self.latent_dim, 1.0, self.activation)
        self.gru = nn.GRUCell(features=self.hist_dim)
        self.jump_net = MLP(tuple(self.model_hidden), self.latent_dim, 0.1, self.activation)
        self.q_net = MLP(tuple(self.model_hidden), 1, 1.0, self.activation)
        self.v_net = MLP(tuple(self.model_hidden), 1, 1.0, self.activation)

    def encode(self, obs):
        return self.enc(_f32(obs))

    def history(self, h, x):
        return self.gru(_f32(h), _f32(x))[0]

    def jump(self, z, c, a, gamma):
        return self.jump_net(_f32(jnp.concatenate([z, c, a, gamma_feature(gamma)[..., None]], -1)))

    def q(self, z, c, a, gamma):
        return self.q_net(_f32(jnp.concatenate([z, c, a, gamma_feature(gamma)[..., None]], -1)))[..., 0]

    def v(self, z, c, gamma):
        return self.v_net(_f32(jnp.concatenate([z, c, gamma_feature(gamma)[..., None]], -1)))[..., 0]

    def __call__(self, obs, h, a_prev, r_prev, a, gamma):     # touches every submodule, for init
        z = self.encode(obs)
        c = self.history(h, jnp.concatenate([z, a_prev, r_prev[..., None]], -1))
        return self.jump(z, c, a, gamma), self.q(z, c, a, gamma), self.v(z, c, gamma)


class Chameleon(NamedTuple):
    wm: WorldModel
    actor: MLP
    critic: MLP
    gate: MLP
    act_dim: int
    cfg: ChameleonConfig

    @classmethod
    def make(cls, env_cfg: jenv.EnvConfig, cfg: ChameleonConfig):
        act = cfg.activation
        return cls(WorldModel(cfg.latent_dim, cfg.enc_hidden, cfg.hist_dim, cfg.model_hidden, act),
                   MLP(tuple(cfg.hidden), env_cfg.act_dim, 0.01, act), MLP(tuple(cfg.hidden), 1, 1.0, act),
                   MLP((64,), 2, 0.01, act), env_cfg.act_dim, cfg)

    def init(self, key, obs_dim):
        k = jax.random.split(key, 4)
        d, H, A = self.cfg.latent_dim, self.cfg.hist_dim, self.act_dim
        wm = self.wm.init(k[0], jnp.zeros((1, obs_dim)), jnp.zeros((1, H)), jnp.zeros((1, A)), jnp.zeros(1),
                          jnp.zeros((1, A)), jnp.full((1,), 0.9))
        pi_in = jnp.zeros((1, obs_dim + 2 * d))
        return dict(actor=dict(net=self.actor.init(k[1], pi_in), log_std=jnp.zeros(A, jnp.float32)),
                    critic=self.critic.init(k[2], pi_in), gate=self.gate.init(k[3], jnp.zeros((1, H + d))),
                    model=wm)

    # --- world model calls
    def _wm(self, params, method, *args):
        return self.wm.apply(params["model"], *args, method=method)

    def context(self, params, obs, h, a_prev, r_prev, first):
        """Encode the obs and advance the history (reset where an episode starts)."""
        z = self._wm(params, WorldModel.encode, obs)
        keep = (~first)[..., None]
        x = jnp.concatenate([z, a_prev * keep, (r_prev * ~first * self.cfg.model_reward_scale)[..., None]], -1)
        return z, self._wm(params, WorldModel.history, h * keep, x)

    # --- policy side
    def pi_input(self, obs, z, g):
        return _f32(jnp.concatenate([_f32(obs), jax.lax.stop_gradient(_f32(z)), _f32(g)], -1))

    def pi_mean(self, params, obs, z, g):
        return self.actor.apply(params["actor"]["net"], self.pi_input(obs, z, g))

    def value(self, params, obs, z, g):
        return self.critic.apply(params["critic"], self.pi_input(obs, z, g))[..., 0]

    def gate_ab(self, params, c, z):
        out = self.gate.apply(params["gate"], jax.lax.stop_gradient(jnp.concatenate([c, z], -1)))
        return 1.0 + jax.nn.softplus(out[..., 0]), 1.0 + jax.nn.softplus(out[..., 1])


# --8<-- [start:goal]
def exploit_direction(model: Chameleon, params, z, c):
    """Unit latent direction of steepest rise of the model's value at plan_gamma."""
    gam = jnp.full(z.shape[:-1], model.cfg.plan_gamma)
    u = jax.grad(lambda z_: model._wm(params, WorldModel.v, z_, c, gam).sum())(z)
    return u / (jnp.linalg.norm(u, axis=-1, keepdims=True) + 1e-8)


def orthogonal_explore(key, u):
    """Uniformly random unit vector orthogonal to each (unit) u."""
    xi = jax.random.normal(key, u.shape, u.dtype)
    xi = xi - jnp.sum(xi * u, -1, keepdims=True) * u
    return xi / (jnp.linalg.norm(xi, axis=-1, keepdims=True) + 1e-8)


def mix_goal(u, xi, eps):
    return jnp.sqrt(1 - eps)[..., None] * u + jnp.sqrt(eps)[..., None] * xi
# --8<-- [end:goal]


def _cos(a, b):
    return jnp.sum(a * b, -1) / (jnp.linalg.norm(a, axis=-1) * jnp.linalg.norm(b, axis=-1) + 1e-8)


def beta_entropy(a, b):
    return betaln(a, b) - (a - 1) * digamma(a) - (b - 1) * digamma(b) + (a + b - 2) * digamma(a + b)


class ChamRunner(NamedTuple):
    params: dict
    env_state: jenv.EnvState
    obs: jnp.ndarray
    key: jnp.ndarray
    h: jnp.ndarray          # (n_envs, hist_dim) history state
    a_prev: jnp.ndarray
    r_prev: jnp.ndarray
    first: jnp.ndarray      # True on an episode's first step
    g: jnp.ndarray          # current latent task
    eps: jnp.ndarray
    gate_logp: jnp.ndarray
    since_goal: jnp.ndarray
    n_updates: jnp.ndarray


class ChamTransition(NamedTuple):
    obs: jnp.ndarray
    next_obs: jnp.ndarray   # pre-reset obs after the step
    z: jnp.ndarray
    c: jnp.ndarray
    g: jnp.ndarray
    a_prev: jnp.ndarray
    r_prev: jnp.ndarray
    first: jnp.ndarray
    action: jnp.ndarray
    logp: jnp.ndarray
    value: jnp.ndarray      # policy critic
    reward: jnp.ndarray     # policy reward (env + task, with the truncation bootstrap)
    r_env: jnp.ndarray
    task: jnp.ndarray
    v_model: jnp.ndarray    # model V at gamma = cfg.gamma (the gate's baseline)
    eps: jnp.ndarray
    gate_logp: jnp.ndarray
    new_goal: jnp.ndarray
    done: jnp.ndarray
    search_gain: jnp.ndarray
    JT: jnp.ndarray


def act(model: Chameleon, params, obs, h, a_prev, r_prev, first, g, eps, gate_logp, since_goal, key,
        deterministic=False):
    """History update, planner (on goal steps) and candidate search, batched over envs."""
    cfg = model.cfg
    k_xi, k_gate, k_cand = jax.random.split(key, 3)
    z, c = model.context(params, obs, h, a_prev, r_prev, first)
    new_goal = first | (since_goal >= cfg.goal_every)

    u = exploit_direction(model, params, z, c)
    xi = orthogonal_explore(k_xi, u)
    ga, gb = model.gate_ab(params, c, z)
    if not cfg.gate:
        eps_new = jnp.full(z.shape[:-1], cfg.fixed_eps, jnp.float32)
    elif deterministic:
        eps_new = jnp.zeros(z.shape[:-1], jnp.float32)    # evaluate the pure exploit plan
    else:
        eps_new = jnp.clip(jax.random.beta(k_gate, ga, gb), 1e-4, 1 - 1e-4).astype(jnp.float32)
    g = jnp.where(new_goal[..., None], mix_goal(u, xi, eps_new), g)
    gate_logp = jnp.where(new_goal, beta_dist.logpdf(eps_new, ga, gb).astype(jnp.float32), gate_logp)
    eps = jnp.where(new_goal, eps_new, eps)
    since_goal = jnp.where(new_goal, 1, since_goal + 1)

    mean = model.pi_mean(params, obs, z, g)
    log_std = params["actor"]["log_std"]
    noise = jax.random.normal(k_cand, (cfg.n_candidates,) + mean.shape, jnp.float32)
    if deterministic:
        noise = noise.at[0].set(0.0)                      # the mean action is always a candidate
    cands = mean + jnp.exp(log_std) * noise               # (N, B, A)
    gam = jnp.full((cfg.n_candidates,) + z.shape[:-1], cfg.plan_gamma)
    jumps = model._wm(params, WorldModel.jump, jnp.broadcast_to(z, (cfg.n_candidates,) + z.shape),
                      jnp.broadcast_to(c, (cfg.n_candidates,) + c.shape), cands, gam)
    score = _cos(jumps, g)                                # (N, B)
    best = jnp.argmax(score, 0)
    action = jnp.take_along_axis(cands, best[None, :, None], 0)[0]
    logp = gaussian_logp(mean, log_std, action)
    info = dict(z=z, c=c, new_goal=new_goal, search_gain=score.max(0) - score.mean(0))
    return action, logp, c, g, eps, gate_logp, since_goal, info


def collect(model: Chameleon, runner: ChamRunner, env_cfg, n_steps):
    cfg = model.cfg
    v_step = jax.vmap(jenv.step_autoreset, in_axes=(0, 0, 0, None, None))

    def body(r: ChamRunner, _):
        key, k_act, k_env = jax.random.split(r.key, 3)
        p = r.params
        action, logp, c, g, eps, gate_logp, since_goal, info = act(
            model, p, r.obs, r.h, r.a_prev, r.r_prev, r.first, r.g, r.eps, r.gate_logp, r.since_goal, k_act)
        z = info["z"]
        next_obs, env_state, r_env, term, trunc, einfo = v_step(
            jax.random.split(k_env, r.obs.shape[0]), r.env_state, action, env_cfg, cfg.resample_drift)
        z_next = model._wm(p, WorldModel.encode, einfo["final_obs"])
        task = _cos(z_next - z, g)
        value = model.value(p, r.obs, z, g)
        reward = cfg.env_reward_coef * r_env + cfg.task_reward_coef * task
        bootstrap = model.value(p, einfo["final_obs"], z_next, g)
        reward = reward + cfg.gamma * bootstrap * (trunc & ~term)
        v_model = model._wm(p, WorldModel.v, z, c, jnp.full(z.shape[:-1], cfg.gamma))
        done = term | trunc
        tr = ChamTransition(r.obs, einfo["final_obs"], z, c, g, r.a_prev, r.r_prev, r.first, action, logp,
                            value, reward.astype(jnp.float32), r_env.astype(jnp.float32), task, v_model, eps,
                            gate_logp, info["new_goal"], done, info["search_gain"], einfo["JT"])
        r = r._replace(env_state=env_state, obs=next_obs, key=key, h=c, a_prev=_f32(jnp.clip(action, -1, 1)),
                       r_prev=r_env.astype(jnp.float32), first=done, g=g, eps=eps, gate_logp=gate_logp,
                       since_goal=since_goal)
        return r, tr

    return jax.lax.scan(body, runner, None, n_steps)


def gae_arrays(reward, value, done, last_value, gamma, lam):
    def body(carry, x):
        next_adv, next_value = carry
        r, v, d = x
        nd = 1.0 - d
        delta = r + gamma * next_value * nd - v
        adv = delta + gamma * lam * nd * next_adv
        return (adv, v), adv

    _, adv = jax.lax.scan(body, (jnp.zeros_like(last_value), last_value), (reward, value, done.astype(jnp.float32)),
                          reverse=True)
    return adv, adv + value


# --8<-- [start:model_loss]
def gamma_targets(z_next, done, gamma):
    """y_t = (1-gamma) z_{t+1} + gamma y_{t+1}, with the latent held after an episode ends. (T, B, d)."""
    def body(y_next, x):
        zn, d = x
        y = (1 - gamma)[:, None] * zn + gamma[:, None] * jnp.where(d[:, None], zn, y_next)
        return y, y

    _, y = jax.lax.scan(body, z_next[-1], (z_next, done), reverse=True)
    return y


def lambda_returns(r, v, done, gamma, lam):
    """TD(lambda) return bootstrapped with v (the rollout's last step bootstraps on its own v)."""
    v_next = jnp.concatenate([v[1:], v[-1:]], 0)

    def body(g_next, x):
        r_t, vn, d = x
        g = r_t + gamma * (1 - d) * ((1 - lam) * vn + lam * g_next)
        return g, g

    _, G = jax.lax.scan(body, v[-1], (r, v_next, done.astype(jnp.float32)), reverse=True)
    return G


def model_loss(model: Chameleon, mparams, seq, h0, key):
    """seq: ChamTransition fields, (T, B, ...) for B env sequences; h0: their history state at t = 0."""
    cfg = model.cfg
    params = dict(model=mparams)
    z = model._wm(params, WorldModel.encode, seq.obs)
    z_next = jax.lax.stop_gradient(model._wm(params, WorldModel.encode, seq.next_obs))

    def hist(h, x):
        z_t, a_prev, r_prev, first = x
        keep = (~first)[:, None]
        inp = jnp.concatenate([z_t, a_prev * keep, (r_prev * ~first * cfg.model_reward_scale)[:, None]], -1)
        h = model._wm(params, WorldModel.history, h * keep, inp)
        return h, h

    _, c = jax.lax.scan(hist, h0, (z, seq.a_prev, seq.r_prev, seq.first))
    T, B = seq.done.shape
    gamma = jax.random.choice(key, jnp.asarray(cfg.gammas, jnp.float32), (B,))
    gam_tb = jnp.broadcast_to(gamma, (T, B))

    y = gamma_targets(z_next, seq.done, gamma)
    jump = model._wm(params, WorldModel.jump, z, c, seq.action, gam_tb)
    jump_loss = jnp.mean(jnp.sum((jump - jax.lax.stop_gradient(y - z)) ** 2, -1))

    q = model._wm(params, WorldModel.q, z, c, seq.action, gam_tb)
    v = model._wm(params, WorldModel.v, z, c, gam_tb)
    G = jax.lax.stop_gradient(lambda_returns(seq.r_env * cfg.model_reward_scale, v, seq.done, gamma, cfg.model_lambda))
    value_loss = jnp.mean((q - G) ** 2 + (v - G) ** 2)

    zf = z.reshape(-1, z.shape[-1])
    zc = zf - zf.mean(0)
    std = jnp.sqrt(zc.var(0) + 1e-4)
    var_loss = jnp.mean(jax.nn.relu(1 - std))
    cov = zc.T @ zc / (zf.shape[0] - 1)
    cov_loss = (jnp.sum(cov ** 2) - jnp.sum(jnp.diag(cov) ** 2)) / zf.shape[-1]
    loss = jump_loss + value_loss + cfg.vicreg_coef * (var_loss + cov_loss)
    stats = dict(m_jump=jump_loss, m_value=value_loss, m_var=var_loss, m_cov=cov_loss,
                 m_q_minus_v=jnp.mean(jnp.abs(q - v)))
    return loss, {k: v.astype(jnp.float32) for k, v in stats.items()}


MODEL_STATS = ("m_jump", "m_value", "m_var", "m_cov", "m_q_minus_v")
# --8<-- [end:model_loss]


class ChamOpt(NamedTuple):
    policy: Any
    model: Any


def _policy_part(params):
    return {k: params[k] for k in ("actor", "critic", "gate")}


def init(key, env_cfg: jenv.EnvConfig, cfg: ChameleonConfig):
    model = Chameleon.make(env_cfg, cfg)
    k_net, k_env, key = jax.random.split(key, 3)
    params = model.init(k_net, env_cfg.obs_dim)
    obs, env_state = jax.vmap(jenv.reset, in_axes=(0, None))(jax.random.split(k_env, cfg.n_envs), env_cfg)
    n, A, d = cfg.n_envs, env_cfg.act_dim, cfg.latent_dim
    z32 = lambda *shape: jnp.zeros(shape, jnp.float32)
    runner = ChamRunner(params, env_state, obs, key, z32(n, cfg.hist_dim), z32(n, A),
                        z32(n), jnp.ones(n, bool), z32(n, d), z32(n), z32(n),
                        jnp.full(n, cfg.goal_every, jnp.int32), jnp.zeros((), jnp.int32))
    opt = ChamOpt(ppo.make_optimizer(cfg).init(_policy_part(params)),
                  optax.chain(optax.clip_by_global_norm(1.0), optax.adam(cfg.model_lr)).init(params["model"]))
    return model, runner, opt


def make_update(model: Chameleon, env_cfg: jenv.EnvConfig, cfg: ChameleonConfig):
    tx = ppo.make_optimizer(cfg)
    mtx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(cfg.model_lr))
    n_groups = cfg.n_envs // cfg.model_envs

    def policy_loss(pparams, mb):
        params = dict(pparams)
        obs, z, g, c, action, logp_old, adv, ret, eps, glogp_old, gadv, goal = mb
        mean = model.pi_mean(params, obs, z, g)
        logp = gaussian_logp(mean, params["actor"]["log_std"], action)
        ratio = jnp.exp(logp - logp_old)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        pg = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv).mean()
        v_loss = ((ret - model.value(params, obs, z, g)) ** 2).mean()

        loss = pg + cfg.vf_coef * v_loss
        gate_loss = gate_ent = jnp.zeros(())
        if cfg.gate:
            ga, gb = model.gate_ab(params, c, z)
            gratio = jnp.exp(beta_dist.logpdf(eps, ga, gb) - glogp_old)
            gsur = jnp.minimum(gratio * gadv, jnp.clip(gratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * gadv)
            n_goal = jnp.maximum(goal.sum(), 1.0)
            gate_loss = -(goal * gsur).sum() / n_goal
            gate_ent = (goal * beta_entropy(ga, gb)).sum() / n_goal
            loss = loss + gate_loss - cfg.gate_ent_coef * gate_ent
        return loss, dict(pg_loss=pg, v_loss=v_loss, gate_loss=gate_loss, gate_entropy=gate_ent,
                          approx_kl=((ratio - 1) - jnp.log(ratio)).mean(),
                          clip_frac=(jnp.abs(ratio - 1) > cfg.clip_eps).mean())

    @jax.jit
    def update(runner: ChamRunner, opt: ChamOpt):
        h0 = runner.h
        runner, traj = collect(model, runner, env_cfg, cfg.n_steps)
        p = runner.params
        # Bootstrap values at the state after the rollout (history advanced one step, as act would)
        z_last, c_last = model.context(p, runner.obs, runner.h, runner.a_prev, runner.r_prev, runner.first)
        # (the goal for that state may be about to change; the current one is the best available)
        adv, ret = gae_arrays(traj.reward, traj.value, traj.done, model.value(p, runner.obs, z_last, runner.g),
                              cfg.gamma, cfg.gae_lambda)
        last_vm = model._wm(p, WorldModel.v, z_last, c_last, jnp.full(z_last.shape[:-1], cfg.gamma))
        gadv, _ = gae_arrays(traj.r_env * cfg.model_reward_scale, traj.v_model, traj.done, last_vm,
                             cfg.gamma, cfg.gae_lambda)
        goal = traj.new_goal.astype(jnp.float32)
        gm = (gadv * goal).sum() / goal.sum()
        gs = jnp.sqrt((((gadv - gm) * goal) ** 2).sum() / goal.sum())
        gadv = (gadv - gm) / (gs + 1e-8)

        flat = lambda x: x.reshape((cfg.batch_size,) + x.shape[2:])
        data = tuple(flat(x) for x in (traj.obs, traj.z, traj.g, traj.c, traj.action, traj.logp, adv, ret,
                                       traj.eps, traj.gate_logp, gadv, goal))
        warm = runner.n_updates < cfg.model_warmup

        def epoch(carry, x):
            key, ep = x
            k_perm, k_env, k_gam = jax.random.split(key, 3)
            perm = jax.random.permutation(k_perm, cfg.batch_size)
            mbs = jax.tree_util.tree_map(lambda a: a[perm].reshape((cfg.n_minibatches, -1) + a.shape[1:]), data)
            envs = jax.random.permutation(k_env, cfg.n_envs)
            groups = jnp.tile(envs.reshape(n_groups, cfg.model_envs), (cfg.n_minibatches // n_groups + 1, 1))
            groups = groups[:cfg.n_minibatches]

            def minibatch(carry, x):
                params, popt, mopt = carry
                mb, grp, k = x
                pparams = _policy_part(params)
                (_, pstats), pgrads = jax.value_and_grad(policy_loss, has_aux=True)(pparams, mb)
                upd, new_popt = tx.update(pgrads, popt, pparams)
                new_pparams = optax.apply_updates(pparams, upd)
                keep = lambda new, old: jax.tree_util.tree_map(lambda n, o: jnp.where(warm, o, n), new, old)
                pparams, popt = keep(new_pparams, pparams), keep(new_popt, popt)

                def model_step(args):
                    mparams, mopt = args
                    seq = jax.tree_util.tree_map(lambda a: a[:, grp], traj)
                    (_, mstats), mgrads = jax.value_and_grad(
                        lambda mp: model_loss(model, mp, seq, h0[grp], k), has_aux=True)(mparams)
                    mupd, mopt = mtx.update(mgrads, mopt, mparams)
                    return optax.apply_updates(mparams, mupd), mopt, mstats

                def no_model_step(args):            # skipped, not computed-and-discarded
                    nan = jnp.full((), jnp.nan, jnp.float32)
                    return args[0], args[1], {k_: nan for k_ in MODEL_STATS}

                mparams, mopt, mstats = jax.lax.cond(ep < cfg.model_epochs, model_step, no_model_step,
                                                     (params["model"], mopt))
                return (dict(pparams, model=mparams), popt, mopt), dict(pstats, **mstats)

            ks = jax.random.split(k_gam, cfg.n_minibatches)
            return jax.lax.scan(minibatch, carry, (mbs, groups, ks))

        key, k_ep = jax.random.split(runner.key)
        (params, popt, mopt), stats = jax.lax.scan(
            epoch, (runner.params, opt.policy, opt.model),
            (jax.random.split(k_ep, cfg.n_epochs), jnp.arange(cfg.n_epochs)))
        stats = jax.tree_util.tree_map(jnp.nanmean, stats)       # model stats: over the steps that ran
        runner = runner._replace(params=params, key=key, n_updates=runner.n_updates + 1)
        stats.update(mean_reward=traj.r_env.mean(), min_JT=traj.JT.min(), frac_done=traj.done.mean(),
                     task_reward=traj.task.mean(), policy_reward=traj.reward.mean(),
                     gate_eps=(traj.eps * goal).sum() / goal.sum(), search_gain=traj.search_gain.mean(),
                     log_std=params["actor"]["log_std"].mean(), warmup=warm.astype(jnp.float32))
        return runner, ChamOpt(popt, mopt), stats

    return update


def evaluate(model: Chameleon, params, cfg: jenv.EnvConfig, omega_s=None, stop_on_truncation=True):
    """Deterministic episode: pure exploit goals (eps = 0), search over the mean action and fixed-key samples."""
    mc = model.cfg
    omega_s = jnp.asarray(cfg.model.omega_s if omega_s is None else omega_s)
    obs, state = jenv.reset_to(omega_s, cfg)
    A = model.act_dim
    z32 = lambda *shape: jnp.zeros(shape, jnp.float32)
    carry0 = (obs, state, jnp.array(True), z32(1, mc.hist_dim), z32(1, A), z32(1),
              jnp.ones(1, bool), z32(1, mc.latent_dim), z32(1), z32(1),
              jnp.full(1, mc.goal_every, jnp.int32), jax.random.PRNGKey(0))

    def body(carry, _):
        obs, state, alive, h, a_prev, r_prev, first, g, eps, glp, since, key = carry
        key, k = jax.random.split(key)
        action, _, h, g, eps, glp, since, _ = act(model, params, obs[None], h, a_prev, r_prev, first, g, eps,
                                                  glp, since, k, deterministic=True)
        obs, state, r, term, trunc, info = jenv.step(state, action[0], cfg)
        out = dict(reward=r, JT=info["JT"], concurrence=info["concurrence"], unitarity=info["unitarity"],
                   t=info["t"], amps=state.amps_cur, alive=alive)
        ended = term | trunc if stop_on_truncation else term
        carry = (obs, state, alive & ~ended, h, _f32(jnp.clip(action, -1, 1)), jnp.atleast_1d(r).astype(jnp.float32),
                 jnp.zeros(1, bool), g, eps, glp, since, key)
        return carry, out

    _, out = jax.lax.scan(body, carry0, None, cfg.n_steps)
    return out
