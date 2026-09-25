"""Idea i05 echidna: toy envs and the PPO ingredients (OU noise, pessimistic veto, self-imitation)."""
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from scipy.stats import multivariate_normal

from rlquantopt.jx import toy_envs as te
from rlquantopt.jx.agents import ppo_plus as pp


def test_pendulum_matches_gym_update():
    env = te.Pendulum()
    s = te.TState(jnp.array([0.3, -0.5], jnp.float32), jnp.zeros((), jnp.int32))
    _, s2, r, *_ = env.step(s, jnp.array([0.25]))
    th, thdot, u = 0.3, -0.5, 0.5
    new_thdot = thdot + (3 * 10 / 2 * np.sin(th) + 3.0 * u) * 0.05
    np.testing.assert_allclose(s2.x, [th + new_thdot * 0.05, new_thdot], rtol=1e-5)
    np.testing.assert_allclose(r, -(th ** 2 + 0.1 * thdot ** 2 + 0.001 * u ** 2), rtol=1e-5)


def test_tracker_overshoot_ends_the_episode_with_the_penalty():
    env = te.Tracker()
    obs, s = env.reset(jax.random.PRNGKey(0))
    assert obs.shape == (env.obs_dim,)
    s = s._replace(u=jnp.float32(0.95), t=jnp.int32(10))
    _, _, r, term, trunc, _ = env.step(s, jnp.ones(3))
    assert bool(trunc) and not bool(term)
    np.testing.assert_allclose(r, -20 * (1 - 10 / 100), rtol=1e-6)


def test_mountaincar_reaches_goal_with_bang_bang():
    env = te.MountainCar()
    obs, s = env.reset(jax.random.PRNGKey(0))
    got = False
    for _ in range(300):                       # push with the velocity: the textbook solution
        a = jnp.sign(s.x[1] + 1e-6)[None]
        obs, s, r, term, trunc, _ = env.step(s, a)
        if bool(term):
            got = r > 90
            break
    assert got


def test_ou_conditional_likelihood_is_the_exact_ar1_joint_density():
    """Sum of pi(a_t | s_t, eps_{t-1}) = joint Gaussian density of an AR(1) noise sequence (rho, sigma)."""
    rho, sig, T = 0.8, 0.5, 6
    mu = np.linspace(-0.3, 0.4, T)[:, None]
    a = mu + sig * np.random.default_rng(0).normal(size=(T, 1))
    log_std = jnp.log(jnp.array([sig]))
    total, eps_prev = 0.0, jnp.zeros((1,))
    for t in range(T):
        m, ls = pp.cond_dist(jnp.asarray(mu[t]), log_std, eps_prev, jnp.asarray(t == 0), rho)
        total += float(pp.cond_logp(m, ls, jnp.asarray(a[t])))
        eps_prev = (jnp.asarray(a[t]) - mu[t]) / sig
    cov = sig ** 2 * rho ** np.abs(np.subtract.outer(np.arange(T), np.arange(T)))
    np.testing.assert_allclose(total, multivariate_normal(mu[:, 0], cov).logpdf(a[:, 0]), rtol=1e-5)


def test_rho_zero_is_plain_gaussian():
    m, ls = jnp.array([0.2, -0.1]), jnp.array([-0.5, 0.1])
    cm, cls = pp.cond_dist(m, ls, jnp.array([1.0, -2.0]), jnp.array(False), 0.0)
    np.testing.assert_allclose(cm, m)
    np.testing.assert_allclose(cls, ls)


def test_veto_replaces_a_candidate_the_ensemble_rates_poorly():
    env = te.Tracker()
    cfg = pp.PlusConfig(n_envs=2, veto=True, n_candidates=4, veto_delta=0.0, lcb_kappa=0.0)
    model, runner, _ = pp.init(jax.random.PRNGKey(0), env, cfg)
    params = dict(runner.params)
    # A Q ensemble that prefers a large first action component: Q = 10 * a_0 (every head)
    q = lambda qp, obs, a: jnp.stack([10 * jnp.clip(a, -1, 1)[..., 0]] * 4, -1)
    model = model._replace(q=q)
    obs = jnp.zeros((2, env.obs_dim))
    action, _, _, vetoed = pp.act(model, params, obs, jnp.zeros((2, 3)), jnp.ones(2, bool),
                                  jax.random.PRNGKey(3), jnp.array(True), jnp.float32(1.0))
    # delta = 0 keeps only the best-rated candidate (the same draws act() makes)
    mean, log_std = model.ac.dist(params["actor"], obs)
    cands = mean + jnp.exp(log_std) * jax.random.normal(jax.random.PRNGKey(3), (4, 2, 3), jnp.float32)
    rating = jnp.clip(cands[..., 0], -1, 1)
    np.testing.assert_allclose(jnp.clip(action[:, 0], -1, 1), rating.max(0), rtol=1e-6)
    np.testing.assert_array_equal(vetoed, rating[0] < rating.max(0))


@pytest.mark.parametrize("env_name", ["tracker", "ham"])
@pytest.mark.parametrize("variant", ["ppo", "ou", "veto", "sil", "all"])
def test_every_variant_updates(env_name, variant):
    from rlquantopt.jx.toybench import VARIANTS
    env = te.ENVS[env_name]()
    kw = dict(VARIANTS[variant], n_envs=4, n_steps=16, n_minibatches=4, n_epochs=2, total_steps=640,
              hidden=(32, 32), veto_warmup=1, cache_size=8, cache_candidates=4)
    cfg = pp.PlusConfig(**kw)
    model, runner, opt = pp.init(jax.random.PRNGKey(0), env, cfg)
    update = jax.jit(pp.make_update(model, cfg))
    for _ in range(2):
        runner, opt, stats = update(runner, opt)
    assert all(np.isfinite(float(v)) for v in stats.values()), stats
    if cfg.sil_coef:
        assert float(stats["cache_n"]) > 0
    ret, score = pp.evaluate_return(model, runner.params, jax.random.PRNGKey(1), n_episodes=2)
    assert np.isfinite(float(ret))


def test_tracker_oob_terminal_option():
    env = te.Tracker(oob_terminal=True)
    _, s = env.reset(jax.random.PRNGKey(0))
    _, _, r, term, trunc, _ = env.step(s._replace(u=jnp.float32(0.95)), jnp.ones(3))
    assert bool(term) and not bool(trunc)
