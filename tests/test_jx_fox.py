"""Idea i06 fox: the improvement-equivalent model, its outlier cache, and the adapter."""
import numpy as np
import jax
import pytest

from rlquantopt.jx.agents import fox, ppo_plus
from rlquantopt.jx.toy_envs import Tracker


def _model_with_basis(P=50, m=6, seed=0, **kw):
    rng = np.random.default_rng(seed)
    M = fox.ImprovementModel(fox.FoxConfig(basis=m, **kw))
    for _ in range(m):
        M.add_proposal(rng.normal(size=P))
    return M, rng


def test_recovers_the_improvement_direction_within_the_basis():
    M, rng = _model_with_basis(window=200)
    g_true = M.D.T @ rng.normal(size=M.D.shape[0])                     # a direction inside the basis
    for k in range(120):
        step = M.D.T @ rng.normal(size=M.D.shape[0]) * rng.uniform(0.5, 2)
        M.observe(fox.Obs(step, float(step @ g_true + 0.01 * rng.normal()), k))
    g, gain = M.direction()
    assert g @ g_true / np.linalg.norm(g_true) > 0.99
    np.testing.assert_allclose(gain, np.linalg.norm(g_true), rtol=0.05)
    pred, sd = M.predict(g_true)
    np.testing.assert_allclose(pred, np.linalg.norm(g_true), rtol=0.05)


def test_unexplained_observations_are_cached_then_released():
    M, rng = _model_with_basis(window=10, out_z=2.0, in_z=1.0)
    g_true = M.D.T @ rng.normal(size=M.D.shape[0])
    obs = lambda k, bump=0.0: (lambda s: fox.Obs(s, float(s @ g_true + bump), k))(M.D.T @ rng.normal(size=M.D.shape[0]))
    M.observe(obs(0, bump=50.0))                                      # a surprise
    for k in range(1, 12):
        M.observe(obs(k))
    assert len(M.cache) == 1                                          # left the window unexplained -> kept
    # the world changes so that the surprise becomes the rule: the model now explains it and drops it
    surprise = M.cache[0]
    for k in range(12, 40):
        s = surprise.step * rng.uniform(0.5, 2)
        M.observe(fox.Obs(s, surprise.dJ * np.linalg.norm(s) / np.linalg.norm(surprise.step), k))
    assert len(M.cache) == 0


@pytest.fixture(scope="module")
def setup():
    dev = Tracker(gain=1.2, offset=0.05)
    cfg = ppo_plus.PlusConfig(n_envs=4, n_steps=32, n_minibatches=4, n_epochs=2, total_steps=4 * 32 * 12, hidden=(32, 32))
    model, runner, opt = ppo_plus.init(jax.random.PRNGKey(0), dev, cfg)
    return dev, cfg, model, runner, opt, jax.jit(ppo_plus.make_update(model, cfg))


def test_without_exploit_fox_is_exactly_ppo(setup):
    dev, cfg, model, runner, opt, update = setup
    r1, o1 = runner, opt
    for _ in range(4):
        r1, o1, _ = update(r1, o1)
    F = fox.FoxAdapter(update, runner, opt, fox.FoxConfig(exploit=False, min_updates=0), dev.max_steps)
    for _ in range(4):
        F.step()
    same = jax.tree_util.tree_map(lambda a, b: bool(np.allclose(a, b, atol=1e-6)), r1.params["actor"], F.runner.params["actor"])
    assert all(jax.tree_util.tree_leaves(same))


def test_fox_steps_and_logs(setup):
    dev, cfg, model, runner, opt, update = setup
    F = fox.FoxAdapter(update, runner, opt, fox.FoxConfig(basis=3, min_updates=2, patience=1, cost_per_update=1e9), dev.max_steps)
    for _ in range(8):
        stats, rec = F.step()
    assert len(F.model.window) == 7 and F.model.post is not None
    assert F.stopped_at is not None                                   # an absurd cost stops it immediately
    assert all(np.isfinite(r["J"]) for r in F.log)
