"""Idea i03 chameleon: planner geometry, model targets, history reset, and the full update."""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from rlquantopt.jx import env as jenv
from rlquantopt.jx.agents import chameleon as ch


def test_explore_direction_is_orthogonal_and_goal_is_unit():
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
    u = jax.random.normal(k1, (100, 16))
    u = u / jnp.linalg.norm(u, axis=-1, keepdims=True)
    xi = ch.orthogonal_explore(k2, u)
    eps = jax.random.uniform(k3, (100,))
    g = ch.mix_goal(u, xi, eps)
    np.testing.assert_allclose(jnp.sum(xi * u, -1), 0, atol=1e-5)
    np.testing.assert_allclose(jnp.linalg.norm(xi, axis=-1), 1, rtol=1e-5)
    np.testing.assert_allclose(jnp.linalg.norm(g, axis=-1), 1, rtol=1e-5)
    np.testing.assert_allclose(jnp.sum(g * u, -1), jnp.sqrt(1 - eps), rtol=1e-4)   # eps = explored share


def test_gamma_targets_and_lambda_returns_match_loops():
    rng = np.random.default_rng(0)
    T, B, d = 9, 3, 2
    zn = rng.normal(size=(T, B, d))
    done = rng.random((T, B)) < 0.25
    gam = np.array([0.5, 0.9, 0.99])
    y = np.asarray(ch.gamma_targets(jnp.asarray(zn), jnp.asarray(done), jnp.asarray(gam)))
    ref, nxt = np.zeros_like(zn), zn[-1]
    for t in reversed(range(T)):
        ref[t] = (1 - gam)[:, None] * zn[t] + gam[:, None] * np.where(done[t][:, None], zn[t], nxt)
        nxt = ref[t]
    np.testing.assert_allclose(y, ref, rtol=1e-6)

    r, v, lam = rng.normal(size=(T, B)), rng.normal(size=(T, B)), 0.9
    G = np.asarray(ch.lambda_returns(jnp.asarray(r), jnp.asarray(v), jnp.asarray(done), jnp.asarray(gam), lam))
    ref, g_next = np.zeros((T, B)), v[-1]
    for t in reversed(range(T)):
        vn = v[t + 1] if t + 1 < T else v[-1]
        ref[t] = r[t] + gam * (1 - done[t]) * ((1 - lam) * vn + lam * g_next)
        g_next = ref[t]
    np.testing.assert_allclose(G, ref, rtol=1e-6)


def _small_cfg(**kw):
    return ch.ChameleonConfig(**dict(dict(n_envs=4, n_steps=16, n_minibatches=4, n_epochs=2, total_steps=640,
                                          hidden=(32, 32), enc_hidden=(32,), hist_dim=16, model_hidden=(32,),
                                          latent_dim=4, n_candidates=3, goal_every=4, model_envs=2,
                                          model_epochs=1, model_warmup=1), **kw))


def test_history_resets_at_episode_start():
    env_cfg, cfg = jenv.EnvConfig(), _small_cfg()
    model = ch.Chameleon.make(env_cfg, cfg)
    params = model.init(jax.random.PRNGKey(0), env_cfg.obs_dim)
    obs = jax.random.uniform(jax.random.PRNGKey(1), (2, env_cfg.obs_dim))
    h = jax.random.normal(jax.random.PRNGKey(2), (2, cfg.hist_dim))
    a, r = jnp.ones((2, 3)), jnp.ones(2)
    z1, c1 = model.context(params, obs, h, a, r, jnp.ones(2, bool))
    z0, c0 = model.context(params, obs, jnp.zeros_like(h), 0 * a, 0 * r, jnp.zeros(2, bool))
    np.testing.assert_allclose(c1, c0, rtol=1e-6)
    _, c2 = model.context(params, obs, h, a, r, jnp.zeros(2, bool))
    assert not np.allclose(c2, c0)


@pytest.mark.parametrize("kw", [{}, dict(gate=False, fixed_eps=0.0, n_candidates=1)])
def test_update_warmup_then_trains_and_evaluates(kw):
    env_cfg, cfg = jenv.EnvConfig(), _small_cfg(**kw)
    model, runner, opt = ch.init(jax.random.PRNGKey(0), env_cfg, cfg)
    update = ch.make_update(model, env_cfg, cfg)
    p0 = runner.params
    runner, opt, stats = update(runner, opt)                  # warm-up: only the model moves
    assert float(stats["warmup"]) == 1
    same = jax.tree_util.tree_map(lambda a, b: bool(jnp.all(a == b)), ch._policy_part(p0), ch._policy_part(runner.params))
    assert all(jax.tree_util.tree_leaves(same))
    assert not all(jax.tree_util.tree_leaves(jax.tree_util.tree_map(
        lambda a, b: bool(jnp.all(a == b)), p0["model"], runner.params["model"])))
    p1 = runner.params
    runner, opt, stats = update(runner, opt)
    assert float(stats["warmup"]) == 0
    assert all(np.isfinite(float(v)) for v in stats.values()), stats
    moved = jax.tree_util.tree_map(lambda a, b: bool(jnp.any(a != b)), p1["actor"], runner.params["actor"])
    assert any(jax.tree_util.tree_leaves(moved))
    ep = ch.evaluate(model, runner.params, env_cfg)
    assert ep["JT"].shape == (env_cfg.n_steps,) and np.all(np.isfinite(np.asarray(ep["JT"])))
