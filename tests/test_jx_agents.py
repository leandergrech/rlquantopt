import numpy as np
import jax
import jax.numpy as jnp
import pytest
from scipy.stats import norm

from rlquantopt.jx import env as jenv
from rlquantopt.jx.agents import common, ppo, trpo


def test_gaussian_logp_and_kl():
    rng = np.random.default_rng(0)
    m, a = rng.normal(size=3), rng.normal(size=3)
    ls = rng.normal(size=3) * 0.3
    np.testing.assert_allclose(common.gaussian_logp(m, ls, a), norm.logpdf(a, m, np.exp(ls)).sum(), rtol=1e-6)
    m2, ls2 = rng.normal(size=3), rng.normal(size=3) * 0.3
    s, s2 = np.exp(ls), np.exp(ls2)
    kl_ref = np.sum(np.log(s2 / s) + (s ** 2 + (m - m2) ** 2) / (2 * s2 ** 2) - 0.5)
    np.testing.assert_allclose(common.gaussian_kl(m, ls, m2, ls2), kl_ref, rtol=1e-6)


def test_conjugate_gradient_solves_spd_system():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(20, 20))
    A = A @ A.T + 20 * np.eye(20)
    b = rng.normal(size=20)
    x = trpo.conjugate_gradient(lambda v: jnp.asarray(A) @ v, jnp.asarray(b), 20, tol=1e-24)
    np.testing.assert_allclose(np.asarray(x), np.linalg.solve(A, b), rtol=1e-6, atol=1e-8)


def test_gae_matches_reference_loop():
    rng = np.random.default_rng(2)
    T, N = 7, 3
    r, v, d = rng.normal(size=(T, N)), rng.normal(size=(T, N)), rng.random((T, N)) < 0.3
    last = rng.normal(size=N)
    tr = common.Transition(*(jnp.zeros((T, N)),) * 3, jnp.asarray(v), jnp.asarray(r), jnp.asarray(d, float),
                           *(jnp.zeros((T, N)),) * 4)
    adv, ret = common.gae(tr, jnp.asarray(last), 0.99, 0.95)
    ref, nxt_adv, nxt_v = np.zeros((T, N)), np.zeros(N), last
    for t in reversed(range(T)):
        nd = 1 - d[t]
        delta = r[t] + 0.99 * nxt_v * nd - v[t]
        nxt_adv = delta + 0.99 * 0.95 * nd * nxt_adv
        ref[t], nxt_v = nxt_adv, v[t]
    np.testing.assert_allclose(np.asarray(adv), ref, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(ret), ref + v, rtol=1e-6)


@pytest.mark.parametrize("algo,cfg", [
    (ppo, ppo.PPOConfig(n_envs=4, n_steps=16, n_minibatches=4, n_epochs=2, total_steps=640)),
    (trpo, trpo.TRPOConfig(n_envs=4, n_steps=16, batch_size=16, n_critic_updates=2, total_steps=640)),
])
def test_one_update_runs_and_changes_params(algo, cfg):
    env_cfg = jenv.EnvConfig()
    model, runner, opt_state = algo.init(jax.random.PRNGKey(0), env_cfg, cfg)
    update = algo.make_update(model, env_cfg, cfg)
    new_runner, opt_state, stats = update(runner, opt_state)
    assert all(np.isfinite(float(v)) for v in stats.values())
    diff = jax.tree_util.tree_map(lambda a, b: float(jnp.abs(a - b).max()), runner.params, new_runner.params)
    assert max(jax.tree_util.tree_leaves(diff["critic"])) > 0
    if algo is trpo:
        assert float(stats["kl"]) < cfg.target_kl
