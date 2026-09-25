"""PPO with a sample judge (idea i01 "axolotl"): a small model predicts whether a sample helps the update.

Per update, on top of plain PPO:

1. **Target, from the policy gradient.** At the rollout parameters, for ``judge_probe`` random samples,
   the per-sample PPO gradient g_i is compared with the mean PPO gradient G of the *other* samples in the
   batch: ``help_i = cos(g_i, G)``. Positive: a step on sample i alone lowers the loss of the rest of the
   batch to first order (it agrees with the update); negative: it pulls against it.
2. **Judge.** An MLP of (obs, action, reward) outputs a mean and a std of ``help`` and is trained with the
   Gaussian NLL on a ring buffer of the last ``judge_buffer`` updates' probe targets.
3. **Guide loss.** Before the PPO epochs the judge scores every sample of the new batch (it has not seen
   their targets). Only confident verdicts count: m_i = sign(mu_i) if |mu_i| / sigma_i > judge_z, else 0.
   The PPO surrogate gets a separate term ``judge_coef * mean(m_i * l_i)``, and the sum is divided by
   ``1 + judge_coef * mean(m_i)`` so the step size is unchanged: confidently helpful samples weigh more,
   confidently harmful ones less, the rest are untouched. An untrained judge has a wide sigma, so the
   guide stays quiet until it has learnt something (and for ``judge_warmup`` updates regardless).

``judge_coef=0`` switches the judge off completely (no targets computed), which with ``autoregressive``
gives the AR-only ablation.

Idea i02 "badger" adds, each off by default so the i01 runs reproduce:

- ``judge_return``: the judge also sees the sample's lambda-return (the GAE target of the critic).
- ``judge_every``: targets are measured and the judge trained only every n-th policy update.
- ``judge_half_life``: the guide weight decays with the judge's age, the number of policy updates since
  it was last trained: ``judge_coef * 0.5 ** (age / judge_half_life)``. The judge scores a batch before it
  trains on it, so the age is 1 right after training.
- ``target_kl``: SB3-style early stop. Once a minibatch starts with approx KL above ``1.5 * target_kl``,
  the remaining minibatches and epochs of that update are skipped. Loss statistics are averaged over the
  steps actually taken. The optimiser's step count only advances on taken steps, so the learning-rate
  schedule decays more slowly than without the stop.
"""
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax

from rlquantopt.jx import env as jenv
from rlquantopt.jx.agents import ppo
from rlquantopt.jx.agents.autoregressive import ARActorCritic
from rlquantopt.jx.agents.common import MLP, ActorCritic, RunnerState, collect, gae


@dataclass(frozen=True)
class JudgePPOConfig(ppo.PPOConfig):
    autoregressive: bool = True
    head_hidden: int = 64
    judge_coef: float = 0.5         # weight of the guide loss; 0 disables the judge
    judge_z: float = 1.0            # guide only where |mu| / sigma exceeds this
    judge_hidden: tuple = (64, 64)
    judge_lr: float = 1e-3
    judge_probe: int = 512          # samples per update whose gradient alignment is measured
    judge_buffer: int = 8           # updates of probe targets the judge trains on
    judge_steps: int = 16           # judge minibatch steps per update
    judge_minibatch: int = 256
    judge_warmup: int = 20          # updates before the guide loss is switched on
    judge_reward_scale: float = 0.1
    # i02 badger
    judge_return: bool = False      # add the lambda-return to the judge's input
    judge_return_scale: float = 0.01
    judge_every: int = 1            # policy updates per judge update
    judge_half_life: float = float("inf")   # in policy updates since the judge was last trained
    target_kl: float = float("inf") # early-stop the epochs once approx KL > 1.5 * target_kl

    @property
    def use_judge(self):
        return self.judge_coef > 0


class JudgeState(NamedTuple):
    """Carried through ``update`` in place of the optimiser state."""
    opt: Any                # policy optimiser state
    judge: dict
    judge_opt: Any
    buf_x: jnp.ndarray      # (judge_buffer * judge_probe, in_dim)
    buf_y: jnp.ndarray
    n_updates: jnp.ndarray
    n_judge: jnp.ndarray    # judge updates so far
    judge_trained_at: jnp.ndarray   # policy update index of the last judge update


def make_judge(cfg: JudgePPOConfig):
    return MLP(tuple(cfg.judge_hidden), 2, 0.01, "tanh")


def judge_in_dim(env_cfg: jenv.EnvConfig, cfg: JudgePPOConfig):
    return env_cfg.obs_dim + env_cfg.act_dim + 1 + cfg.judge_return


def judge_input(obs, action, reward, ret, cfg: JudgePPOConfig):
    cols = [obs.astype(jnp.float32), jnp.clip(action, -1, 1).astype(jnp.float32),
            (reward * cfg.judge_reward_scale)[..., None].astype(jnp.float32)]
    if cfg.judge_return:
        cols.append((ret * cfg.judge_return_scale)[..., None].astype(jnp.float32))
    return jnp.concatenate(cols, axis=-1)


def judge_predict(judge, judge_params, x):
    out = judge.apply(judge_params, x)
    return out[..., 0], jnp.clip(out[..., 1], -8.0, 2.0)          # mu, log sigma


def init(key, env_cfg: jenv.EnvConfig, cfg: JudgePPOConfig):
    if cfg.autoregressive:
        model = ARActorCritic.make(env_cfg, cfg.hidden, cfg.head_hidden, cfg.activation)
    else:
        model = ActorCritic.make(env_cfg.act_dim, cfg.hidden, cfg.activation)
    k_net, k_env, k_judge, key = jax.random.split(key, 4)
    params = model.init(k_net, env_cfg.obs_dim)
    obs, env_state = jax.vmap(jenv.reset, in_axes=(0, None))(jax.random.split(k_env, cfg.n_envs), env_cfg)
    in_dim = judge_in_dim(env_cfg, cfg)
    judge_params = make_judge(cfg).init(k_judge, jnp.zeros((1, in_dim), jnp.float32))
    n_buf = cfg.judge_buffer * cfg.judge_probe if cfg.use_judge else 1
    state = JudgeState(ppo.make_optimizer(cfg).init(params), judge_params,
                       optax.adam(cfg.judge_lr).init(judge_params),
                       jnp.zeros((n_buf, in_dim), jnp.float32), jnp.zeros((n_buf,), jnp.float32),
                       jnp.zeros((), jnp.int32), jnp.zeros((), jnp.int32), jnp.full((), -1, jnp.int32))
    return model, RunnerState(params, env_state, obs, key), state


def _tree_dot(a, b):
    return sum(jnp.sum(x * y) for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)))


# --8<-- [start:help_target]
def help_targets(model, actor_params, obs, action, logp_old, adv, probe, ref):
    """cos(per-sample PPO gradient of each probe sample, mean PPO gradient of the ref samples)."""
    def sample_loss(p, o, a, lo, ad):
        ratio = jnp.exp(model.logp(p, o[None], a[None])[0] - lo)
        return -ratio * ad          # the clipped surrogate at ratio = 1, where the update starts

    per_sample = jax.vmap(sample_loss, in_axes=(None, 0, 0, 0, 0))
    G = jax.grad(lambda p: per_sample(p, obs[ref], action[ref], logp_old[ref], adv[ref]).mean())(actor_params)
    g = jax.vmap(jax.grad(sample_loss), in_axes=(None, 0, 0, 0, 0))(
        actor_params, obs[probe], action[probe], logp_old[probe], adv[probe])
    dots = jax.vmap(lambda gi: _tree_dot(gi, G))(g)
    norms = jnp.sqrt(jax.vmap(lambda gi: _tree_dot(gi, gi))(g))
    return dots / (norms * jnp.sqrt(_tree_dot(G, G)) + 1e-12)
# --8<-- [end:help_target]


def make_update(model, env_cfg: jenv.EnvConfig, cfg: JudgePPOConfig):
    tx = ppo.make_optimizer(cfg)
    judge = make_judge(cfg)
    jtx = optax.adam(cfg.judge_lr)

    def judge_nll(judge_params, x, y):
        mu, log_sigma = judge_predict(judge, judge_params, x)
        return (0.5 * ((y - mu) / jnp.exp(log_sigma)) ** 2 + log_sigma).mean()

    # --8<-- [start:guided_loss]
    def loss_fn(params, batch, coef):
        obs, action, logp_old, adv, ret, mark = batch
        logp = model.logp(params["actor"], obs, action)
        ratio = jnp.exp(logp - logp_old)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        l = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv)
        pg = l.mean()
        guide = coef * (mark * l).mean()
        pg_total = (pg + guide) / (1 + coef * mark.mean())
        v = model.value(params["critic"], obs)
        v_loss = ((ret - v) ** 2).mean()
        loss = pg_total + cfg.vf_coef * v_loss - cfg.ent_coef * model.entropy(params["actor"])
        return loss, dict(pg_loss=pg, guide_loss=guide, v_loss=v_loss,
                          approx_kl=((ratio - 1) - jnp.log(ratio)).mean(),
                          clip_frac=(jnp.abs(ratio - 1) > cfg.clip_eps).mean())
    # --8<-- [end:guided_loss]

    def judge_phase(state: JudgeState, actor_params, data, reward, key):
        """Score the batch with the current judge; every ``judge_every`` updates also measure the probe
        targets and train the judge on them (after scoring, so the scores never saw their targets)."""
        obs, action, logp_old, adv, ret = data
        x = judge_input(obs, action, reward, ret, cfg)
        mu, log_sigma = judge_predict(judge, state.judge, x)
        confident = jnp.abs(mu) > cfg.judge_z * jnp.exp(log_sigma)
        mark = jnp.where(confident, jnp.sign(mu), 0.0) * (state.n_updates >= cfg.judge_warmup)
        age = state.n_updates - state.judge_trained_at
        coef = cfg.judge_coef * 0.5 ** (age / cfg.judge_half_life)

        def train(state):
            k_perm, k_train = jax.random.split(key)
            perm = jax.random.permutation(k_perm, cfg.batch_size)
            probe, ref = perm[:cfg.judge_probe], perm[cfg.judge_probe:]
            adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
            y = help_targets(model, actor_params, obs, action, logp_old, adv_n, probe, ref)

            # Skill on unseen samples, before the judge trains on them
            mu_p, ls_p = mu[probe], log_sigma[probe]
            corr = jnp.corrcoef(mu_p, y)[0, 1]
            z_true = (y - mu_p) / jnp.exp(ls_p)

            slot = (state.n_judge % cfg.judge_buffer) * cfg.judge_probe
            buf_x = jax.lax.dynamic_update_slice(state.buf_x, x[probe], (slot, jnp.zeros_like(slot)))
            buf_y = jax.lax.dynamic_update_slice(state.buf_y, y.astype(jnp.float32), (slot,))
            n_filled = jnp.minimum(state.n_judge + 1, cfg.judge_buffer) * cfg.judge_probe

            def train_step(carry, k):
                jp, jo = carry
                idx = jax.random.randint(k, (cfg.judge_minibatch,), 0, n_filled)
                nll, grads = jax.value_and_grad(judge_nll)(jp, buf_x[idx], buf_y[idx])
                upd, jo = jtx.update(grads, jo, jp)
                return (optax.apply_updates(jp, upd), jo), nll

            (jp, jo), nll = jax.lax.scan(train_step, (state.judge, state.judge_opt),
                                         jax.random.split(k_train, cfg.judge_steps))
            stats = dict(judge_nll=nll.mean(), judge_corr=corr, judge_calib=jnp.std(z_true),
                         help_mean=y.mean(), help_std=y.std())
            state = state._replace(judge=jp, judge_opt=jo, buf_x=buf_x, buf_y=buf_y, n_judge=state.n_judge + 1,
                                   judge_trained_at=state.n_updates)
            return state, jax.tree_util.tree_map(lambda v: v.astype(jnp.float32), stats)

        def skip(state):
            nan = jnp.full((), jnp.nan, jnp.float32)
            return state, dict(judge_nll=nan, judge_corr=nan, judge_calib=nan, help_mean=nan, help_std=nan)

        state, stats = jax.lax.cond(state.n_updates % cfg.judge_every == 0, train, skip, state)
        stats.update(judge_sigma=jnp.exp(log_sigma).mean(), judge_frac_help=(mark > 0).mean(),
                     judge_frac_hurt=(mark < 0).mean(), judge_age=age, judge_coef_eff=coef)
        return state, mark, coef, stats

    @jax.jit
    def update(runner: RunnerState, state: JudgeState):
        runner, traj = collect(model, runner, env_cfg, cfg.n_steps, cfg.gamma, cfg.resample_drift)
        last_value = model.value(runner.params["critic"], runner.obs)
        adv, ret = gae(traj, last_value, cfg.gamma, cfg.gae_lambda)
        flat = lambda x: x.reshape((cfg.batch_size,) + x.shape[2:])
        data = (flat(traj.obs), flat(traj.action), flat(traj.logp), flat(adv), flat(ret))

        key, k_ep, k_judge = jax.random.split(runner.key, 3)
        if cfg.use_judge:
            state, mark, coef, judge_stats = judge_phase(state, runner.params["actor"], data, flat(traj.reward), k_judge)
        else:
            mark, coef, judge_stats = jnp.zeros(cfg.batch_size), 0.0, {}
        data = data + (mark,)

        # --8<-- [start:kl_stop]
        def epoch(carry, key):
            perm = jax.random.permutation(key, cfg.batch_size)
            mbs = jax.tree_util.tree_map(
                lambda x: x[perm].reshape((cfg.n_minibatches, -1) + x.shape[1:]), data)

            def minibatch(carry, mb):
                params, opt_state, stop = carry
                (loss, stats), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, mb, coef)
                stop = stop | (stats["approx_kl"] > 1.5 * cfg.target_kl)     # SB3: checked before the step
                updates, new_opt = tx.update(grads, opt_state, params)
                keep = lambda new, old: jax.tree_util.tree_map(lambda n, o: jnp.where(stop, o, n), new, old)
                carry = (keep(optax.apply_updates(params, updates), params), keep(new_opt, opt_state), stop)
                return carry, dict(stats, taken=~stop)

            return jax.lax.scan(minibatch, carry, mbs)

        (params, opt, _), stats = jax.lax.scan(
            epoch, (runner.params, state.opt, jnp.array(False)), jax.random.split(k_ep, cfg.n_epochs))
        taken = stats.pop("taken").astype(jnp.float32)
        stats = jax.tree_util.tree_map(lambda v: (v * taken).sum() / jnp.maximum(taken.sum(), 1), stats)
        stats["frac_steps_taken"] = taken.mean()
        # --8<-- [end:kl_stop]
        runner = runner._replace(params=params, key=key)
        state = state._replace(opt=opt, n_updates=state.n_updates + 1)
        stats.update(judge_stats)
        stats.update(ppo.rollout_stats(traj))
        stats["log_std"] = jnp.mean(params["actor"]["log_std"])
        return runner, state, stats

    return update
