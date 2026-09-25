"""Idea i04 dragonfly: parallel traces, the exploration walk, the golden cache, and the full update."""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from rlquantopt.jx import env as jenv
from rlquantopt.jx.agents import dragonfly as df


def _small_cfg(**kw):
    return df.DragonflyConfig(**dict(dict(n_envs=4, n_steps=16, n_minibatches=4, n_epochs=2, total_steps=640,
                                          hidden=(32, 32), enc_hidden=(32,), ctx_dim=16, model_hidden=(32,),
                                          latent_dim=4, n_heads=3, n_candidates=3, goal_every=4, model_envs=2,
                                          model_epochs=1, model_warmup=1, cache_size=8, cache_candidates=6,
                                          guide_hidden=(16,)), **kw))


def _setup(**kw):
    env_cfg, cfg = jenv.EnvConfig(), _small_cfg(**kw)
    model, runner, opt = df.init(jax.random.PRNGKey(0), env_cfg, cfg)
    return env_cfg, cfg, model, runner, opt


def test_parallel_traces_match_the_step_by_step_recurrence():
    """The associative scan used in training reproduces observe() applied one step at a time."""
    env_cfg, cfg, model, runner, _ = _setup()
    params = runner.params
    T, B = 7, 2
    rng = np.random.default_rng(0)
    obs = jnp.asarray(rng.uniform(-0.9, 0.9, (T, B, env_cfg.obs_dim)), jnp.float32)
    a_prev = jnp.asarray(rng.normal(size=(T, B, 3)), jnp.float32)
    r_prev = jnp.asarray(rng.normal(size=(T, B)), jnp.float32)
    first = jnp.asarray([[True, False]] + [[False, False]] * 2 + [[False, True]] + [[False, False]] * 3)
    walk = runner.walk._replace(traces=jnp.asarray(rng.normal(size=(B,) + runner.walk.traces.shape[1:]), jnp.float32),
                                z_prev=jnp.asarray(rng.normal(size=(B, cfg.latent_dim)), jnp.float32))
    traces0, z_prev0 = walk.traces, walk.z_prev
    cs = []
    for t in range(T):
        walk = walk._replace(a_prev=a_prev[t], r_prev=r_prev[t], first=first[t])
        z, traces, c = model.observe(params, walk, obs[t])
        walk = walk._replace(traces=traces, z_prev=z)
        cs.append(c)

    # Rebuild the model's context through the same code path as model_loss
    z = model.m(params, df.DFModel.encode, obs)
    keep = (~first)[..., None]
    zp = jnp.where(keep, jnp.concatenate([z_prev0[None], z[:-1]], 0), z)
    x = jnp.concatenate([z, z - zp, a_prev * keep, (r_prev * cfg.model_reward_scale)[..., None] * keep], -1)
    dec = jnp.asarray(cfg.trace_decays, jnp.float32)
    a = dec[None, None, :, None] * keep[..., None]
    b = (1 - dec)[None, None, :, None] * x[:, :, None, :]
    A, Bc = jax.lax.associative_scan(lambda e1, e2: (e1[0] * e2[0], e2[0] * e1[1] + e2[1]), (a, b), axis=0)
    c_par = model.m(params, df.DFModel.context, A * traces0[None] + Bc)
    np.testing.assert_allclose(c_par, jnp.stack(cs), rtol=1e-4, atol=1e-5)


def test_ou_walk_keeps_its_spread_and_momentum():
    """rho = exp(-1/tau): stationary std sigma for any tau, lag-1 autocorrelation rho."""
    key = jax.random.PRNGKey(1)
    n, T = 4000, 200
    tau, sig = jnp.array([1.0, 10.0]), jnp.array([0.3, 0.7])
    rho = jnp.exp(-1 / tau)
    xi = sig * jax.random.normal(key, (n, 2))
    xs = []
    for t in range(T):
        key, k = jax.random.split(key)
        xi = rho * xi + jnp.sqrt(1 - rho ** 2) * sig * jax.random.normal(k, (n, 2))
        xs.append(xi)
    xs = np.asarray(jnp.stack(xs))
    np.testing.assert_allclose(xs.std((0, 1)), sig, rtol=0.03)
    lag1 = [np.corrcoef(xs[:-1, :, d].ravel(), xs[1:, :, d].ravel())[0, 1] for d in range(2)]
    np.testing.assert_allclose(lag1, rho, atol=0.02)


def test_golden_cache_is_canonical_and_keeps_the_best():
    env_cfg, cfg, model, runner, _ = _setup(cache_radius=0.5)
    params = runner.params
    rng = np.random.default_rng(2)
    N = 40
    obs = jnp.asarray(rng.uniform(-0.9, 0.9, (N, env_cfg.obs_dim)), jnp.float32)
    r = jnp.asarray(rng.normal(size=N), jnp.float32)
    G = jnp.asarray(rng.normal(size=N), jnp.float32)
    obs = obs.at[1].set(obs[0])                                   # a duplicate latent of sample 0 ...
    r, G = r.at[0].set(5.0).at[1].set(4.0), G.at[0].set(5.0).at[1].set(4.0)   # ... slightly worse
    g = jnp.zeros((N, cfg.latent_dim), jnp.float32)
    a = jnp.zeros((N, 3), jnp.float32)
    cache = df.update_cache(model, params, runner.cache, obs, g, a, r, G, jnp.zeros(N, jnp.float32))
    n = int(cache.valid.sum())
    assert 0 < n <= cfg.cache_size
    zc = np.asarray(cache.z)[:n]
    d = np.linalg.norm(zc[:, None] - zc[None], axis=-1) + np.eye(n) * 9
    assert d.min() >= cfg.cache_radius                            # canonical: no two entries in one ball
    assert float(cache.r[:n].max()) == 5.0                         # the best is kept, its duplicate is not
    assert int((np.asarray(cache.r)[:n] == 4.0).sum()) == 0


def test_update_warmup_then_trains_and_evaluates():
    env_cfg, cfg, model, runner, opt = _setup()
    update = df.make_update(model, env_cfg, cfg)
    p0 = runner.params
    runner, opt, stats = update(runner, opt)
    assert float(stats["warmup"]) == 1
    same = jax.tree_util.tree_map(lambda a, b: bool(jnp.all(a == b)), df._policy_part(p0), df._policy_part(runner.params))
    assert all(jax.tree_util.tree_leaves(same))
    assert float(stats["cache_n"]) > 0
    for _ in range(2):
        runner, opt, stats = update(runner, opt)
    assert float(stats["warmup"]) == 0
    bad = {k: float(v) for k, v in stats.items() if not np.isfinite(float(v))}
    assert not bad, bad
    assert 1 <= float(stats["tau_mean"]) <= cfg.tau_max
    ep = df.evaluate(model, runner.params, env_cfg)
    assert np.all(np.isfinite(np.asarray(ep["JT"])))
