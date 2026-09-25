"""Idea i01 axolotl: the autoregressive policy and PPO with the sample judge."""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from rlquantopt.jx import env as jenv
from rlquantopt.jx.agents import ppo_judge
from rlquantopt.jx.agents.autoregressive import ARActorCritic


def _model_and_obs(n=5):
    env_cfg = jenv.EnvConfig()
    model = ARActorCritic.make(env_cfg, (32, 32), 16)
    params = model.init(jax.random.PRNGKey(0), env_cfg.obs_dim, log_std_init=-1.0)
    # Non-trivial amplitudes in the obs, so the seeds (last amplitude, last delta) matter
    obs = jax.random.uniform(jax.random.PRNGKey(1), (n, env_cfg.obs_dim), minval=-0.9, maxval=0.9)
    return env_cfg, model, params, obs


def test_ar_teacher_forced_logp_matches_sampling():
    _, model, params, obs = _model_and_obs()
    action, logp = model.sample(params["actor"], obs, jax.random.PRNGKey(2))
    assert action.shape == (5, 3)
    np.testing.assert_allclose(model.logp(params["actor"], obs, action), logp, rtol=1e-5)


def test_ar_head_is_fed_its_own_output():
    env_cfg, model, params, obs = _model_and_obs()
    a = model.mode(params["actor"], obs)
    h = model.trunk.apply(params["actor"]["trunk"], obs)
    amp, prev = model._seed(obs)
    np.testing.assert_allclose(amp, obs[:, model.amp_start + 2] / env_cfg.obs_scale, rtol=1e-6)
    for k in range(3):
        np.testing.assert_allclose(model._mean(params["actor"], h, prev, amp), a[:, k], rtol=1e-5, atol=1e-7)
        prev, amp = jnp.clip(a[:, k], -1, 1), amp + jnp.clip(a[:, k], -1, 1)


def test_help_target_is_cosine_to_the_rest_of_the_batch():
    _, model, params, obs = _model_and_obs(8)
    action, logp = model.sample(params["actor"], obs, jax.random.PRNGKey(3))
    adv = jax.random.normal(jax.random.PRNGKey(4), (8,))
    probe, ref = jnp.arange(3), jnp.arange(3, 8)
    y = ppo_judge.help_targets(model, params["actor"], obs, action, logp, adv, probe, ref)
    flat = lambda t: jnp.concatenate([x.ravel() for x in jax.tree_util.tree_leaves(t)])
    grad_i = lambda i: flat(jax.grad(lambda p: -model.logp(p, obs[i:i + 1], action[i:i + 1])[0] * adv[i])(params["actor"]))
    G = sum(grad_i(i) for i in range(3, 8)) / 5
    for i in range(3):
        g = grad_i(i)
        np.testing.assert_allclose(y[i], g @ G / jnp.linalg.norm(g) / jnp.linalg.norm(G), rtol=1e-4)
    assert np.all(np.abs(y) <= 1 + 1e-6)


@pytest.mark.parametrize("ar,coef", [(True, 0.5), (True, 0.0), (False, 0.5)])
def test_judge_update_runs(ar, coef):
    env_cfg = jenv.EnvConfig()
    cfg = ppo_judge.JudgePPOConfig(n_envs=4, n_steps=16, n_minibatches=4, n_epochs=2, total_steps=640,
                                   hidden=(32, 32), autoregressive=ar, judge_coef=coef, judge_probe=16,
                                   judge_buffer=2, judge_minibatch=8, judge_steps=4, judge_warmup=0,
                                   judge_z=0.0)
    model, runner, state = ppo_judge.init(jax.random.PRNGKey(0), env_cfg, cfg)
    update = ppo_judge.make_update(model, env_cfg, cfg)
    for _ in range(3):      # wraps the ring buffer
        new_runner, new_state, stats = update(runner, state)
        runner, state = new_runner, new_state
    assert all(np.isfinite(float(v)) for v in stats.values()), stats
    if coef:
        assert float(stats["judge_frac_help"] + stats["judge_frac_hurt"]) > 0     # z=0: every sample is marked


def test_badger_rare_judge_age_decay_and_kl_stop():
    """Idea i02: judge trained every 2nd update with the return as input, guide weight halving per update of
    age, and a KL stop so tight that updates end early."""
    env_cfg = jenv.EnvConfig()
    cfg = ppo_judge.JudgePPOConfig(n_envs=4, n_steps=16, n_minibatches=4, n_epochs=3, total_steps=640,
                                   hidden=(32, 32), judge_probe=16, judge_buffer=2, judge_minibatch=8,
                                   judge_steps=4, judge_warmup=0, judge_z=0.0, judge_return=True,
                                   judge_every=2, judge_half_life=1.0, target_kl=1e-6)
    model, runner, state = ppo_judge.init(jax.random.PRNGKey(0), env_cfg, cfg)
    assert state.buf_x.shape[1] == env_cfg.obs_dim + env_cfg.act_dim + 2
    update = ppo_judge.make_update(model, env_cfg, cfg)
    ages, coefs, trained = [], [], []
    for _ in range(4):
        runner, state, stats = update(runner, state)
        ages.append(int(stats["judge_age"]))
        coefs.append(float(stats["judge_coef_eff"]))
        trained.append(bool(np.isfinite(float(stats["judge_nll"]))))
        assert 0 < float(stats["frac_steps_taken"]) < 1
    assert trained == [True, False, True, False]
    assert ages == [1, 1, 2, 1]                     # trained at updates 0 and 2; scored before training
    np.testing.assert_allclose(coefs, [cfg.judge_coef * 0.5 ** a for a in ages], rtol=1e-6)
    assert int(state.n_judge) == 2
