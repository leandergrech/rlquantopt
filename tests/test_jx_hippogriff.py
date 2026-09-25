"""Idea i08 hippogriff: the drift env, the belief (EKF), information gain, and the deployment modes."""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from rlquantopt.jx import hippogriff as gr
from rlquantopt.jx.agents import ppo_plus
from rlquantopt.jx.toy_envs import DriftTracker, Tracker


def test_drift_tracker_matches_tracker_at_fixed_params():
    dt = DriftTracker()
    tr = Tracker(gain=1.23, offset=-0.07, oob_terminal=True, reward_shift=1.0)
    _, s = dt.reset_with(jax.random.PRNGKey(0), 1.23, -0.07)
    _, ts = tr.reset(jax.random.PRNGKey(5))
    ts = ts._replace(phase=s.phase)
    for a in (jnp.array([0.3, -0.2, 0.5]), jnp.array([1.0, 1.0, -0.4]), jnp.array([-0.8, 0.1, 0.0])):
        o1, s, r1, te1, tr1, i1 = dt.step(s, a)
        o2, ts, r2, te2, tr2, i2 = tr.step(ts, a)
        np.testing.assert_allclose(o1[:-2], o2, rtol=1e-6)
        np.testing.assert_allclose([r1, i1["score"]], [r2, i2["score"]], rtol=1e-6)
        assert bool(te1) == bool(te2) and bool(tr1) == bool(tr2)


def test_measure_predicts_what_the_env_returns():
    env = DriftTracker()
    _, s = env.reset_with(jax.random.PRNGKey(1), 0.8, 0.1)
    a = jnp.array([0.4, 0.2, -0.1])
    _, s2, _, _, _, info = env.step(s, a)
    y = env.measure(s, a, env.z_of(jnp.float32(0.8), jnp.float32(0.1)))
    np.testing.assert_allclose(y, [s2.u - s.u, info["score"]], rtol=1e-5, atol=1e-6)
    g, o = env.params_of(env.z_of(jnp.float32(0.8), jnp.float32(0.1)))
    np.testing.assert_allclose([g, o], [0.8, 0.1], rtol=1e-6)


def test_ekf_update_never_increases_uncertainty():
    rng = np.random.default_rng(0)
    m, P = gr.prior()
    R = jnp.diag(jnp.array([1e-2, 5e-2]) ** 2)
    for _ in range(20):
        J = jnp.asarray(rng.normal(size=(2, 2)), jnp.float32)
        m2, P2 = gr.ekf_update(m, P, jnp.asarray(rng.normal(size=2), jnp.float32), jnp.zeros(2), J, R)
        assert np.linalg.eigvalsh(np.asarray(P - P2)).min() > -1e-6       # P2 <= P in the Loewner order
        m, P = m2, P2
    assert float(gr.rho_of(P, gr.prior()[1])) < 0.05


def test_information_gain():
    P = gr.prior()[1]
    R = jnp.eye(2) * 0.01
    J = jnp.eye(2)
    assert float(gr.info_gain(0 * J, P, R)) == pytest.approx(0.0, abs=1e-6)
    assert float(gr.info_gain(2 * J, P, R)) > float(gr.info_gain(J, P, R)) > 0


def test_filter_identifies_the_device_with_the_exact_model():
    env = DriftTracker()
    meas = gr.SimModel(env, gr.GriffinConfig())
    z_true = env.z_of(jnp.float32(1.3), jnp.float32(-0.1))
    _, s = env.reset_with(jax.random.PRNGKey(3), 1.3, -0.1)
    m, P = gr.prior()
    rng = np.random.default_rng(1)
    for t in range(30):
        a = jnp.asarray(rng.uniform(-0.3, 0.3, 3), jnp.float32) - 0.5 * s.u     # gentle, stays in bounds
        _, s2, _, term, trunc, info = env.step(s, a)
        y = jnp.stack([s2.u - s.u, info["score"]])
        J = jax.jacfwd(lambda z: meas.measure(s, a, z))(m)
        m, P = gr.ekf_update(m, P, y, meas.measure(s, a, m), J, meas.R)
        s = s2
    assert float(jnp.linalg.norm(m - z_true)) < 0.05
    assert float(gr.rho_of(P, gr.prior()[1])) < 0.1


@pytest.mark.parametrize("filt", ["ekf", "ekf_nis", "grid"])
@pytest.mark.parametrize("mode", gr.MODES)
def test_every_mode_runs(mode, filt):
    if filt != "ekf" and not mode.startswith(("filter", "hippogriff")):
        pytest.skip("the filter only matters where there is a belief")
    env = DriftTracker(show_z=mode != "blind")
    cfg = ppo_plus.PlusConfig(n_envs=4, n_steps=16, hidden=(16, 16))
    model, runner, _ = ppo_plus.init(jax.random.PRNGKey(0), env, cfg)
    gcfg = gr.GriffinConfig(n_candidates=4, filter=filt, grid_n=11)
    if mode.endswith("_B"):
        meas, fit = gr.fit_learned_model(env, model, runner.params, jax.random.PRNGKey(1), n_envs=8, episodes=1,
                                         train_steps=50)
    else:
        meas = gr.SimModel(env, gcfg) if mode.endswith("_A") else None
    run = gr.make_episode(env, model, runner.params["actor"], meas, gcfg, mode)
    m, P, logw, P0 = gr.initial_belief(gcfg, 2)
    ret, m, P, logw, out = run(jax.random.split(jax.random.PRNGKey(2), 2), jnp.array([0.8, 1.2]),
                               jnp.array([0.05, -0.1]), m, P, logw, P0)
    assert ret.shape == (2,) and np.all(np.isfinite(np.asarray(ret)))
    assert np.all(np.asarray(out["rho"]) <= 1 + 1e-4)
    if mode.startswith(("filter", "hippogriff")) and filt == "ekf":
        assert np.all(np.diff(np.asarray(out["rho"]), axis=1) <= 1e-4)   # the plain EKF's uncertainty never grows


def test_measurement_noise_enters_R_and_the_measurements():
    env = DriftTracker()
    quiet, noisy = gr.GriffinConfig(), gr.GriffinConfig(noise_du=0.02, noise_score=0.5)
    assert float(gr.SimModel(env, noisy).R[1, 1]) == pytest.approx(0.05 ** 2 + 0.5 ** 2, rel=1e-5)
    cfg = ppo_plus.PlusConfig(n_envs=4, n_steps=16, hidden=(16, 16))
    model, runner, _ = ppo_plus.init(jax.random.PRNGKey(0), env, cfg)
    m, P, logw, P0 = gr.initial_belief(quiet, 2)
    args = (jax.random.split(jax.random.PRNGKey(2), 2), jnp.array([0.8, 1.2]), jnp.array([0.05, -0.1]), m, P, logw, P0)
    rho = {}
    for name, g in (("quiet", quiet), ("noisy", noisy)):
        run = gr.make_episode(env, model, runner.params["actor"], gr.SimModel(env, g), g, "filter_A")
        rho[name] = float(np.asarray(run(*args)[4]["rho"])[:, 10].mean())
    assert rho["noisy"] > rho["quiet"]                    # noise slows the shrinking of the uncertainty


def test_nis_check_inflates_only_on_surprise():
    m, P0 = gr.prior()
    P = P0 * 1e-4                                            # a very sure belief
    R = jnp.eye(2) * 1e-2
    J = jnp.eye(2, dtype=jnp.float32)
    _, P_ok, nis_ok = gr.ekf_nis_update(m, P, jnp.zeros(2), jnp.zeros(2), J, R, P0, 9.21)
    _, P_bad, nis_bad = gr.ekf_nis_update(m, P, jnp.array([1.0, -1.0]), jnp.zeros(2), J, R, P0, 9.21)
    assert float(nis_ok) < 9.21 < float(nis_bad)
    assert np.linalg.eigvalsh(np.asarray(P_bad - P_ok)).min() > 0       # the surprise left it less sure
    np.testing.assert_allclose(P_ok, gr.ekf_update(m, P, jnp.zeros(2), jnp.zeros(2), J, R)[1], rtol=1e-6)


def test_grid_filter_identifies_the_device_and_stays_honest():
    env = DriftTracker()
    gcfg = gr.GriffinConfig(filter="grid", grid_n=41)
    meas = gr.SimModel(env, gcfg)
    Z = gr.make_grid(41)
    z_true = env.z_of(jnp.float32(1.3), jnp.float32(-0.1))
    _, s = env.reset_with(jax.random.PRNGKey(3), 1.3, -0.1)
    logw = jnp.full((Z.shape[0],), -jnp.log(Z.shape[0]), jnp.float32)
    rng = np.random.default_rng(1)
    for t in range(30):
        a = jnp.asarray(rng.uniform(-0.3, 0.3, 3), jnp.float32) - 0.5 * s.u
        _, s2, _, _, _, info = env.step(s, a)
        y = jnp.stack([s2.u - s.u, info["score"]])
        logw = gr.grid_update(logw, y, jax.vmap(lambda z: meas.measure(s, a, z))(Z), meas.R)
        s = s2
    m, P = gr.grid_moments(logw, Z)
    err = float(jnp.linalg.norm(m - z_true))
    claimed = float(jnp.sqrt(jnp.trace(P)))
    assert err < 0.05                                        # within one grid spacing
    assert err < 3 * claimed + 0.05                          # and not wildly overconfident


def test_predictive_info_gain_matches_linear_case():
    """For a linear model the predictive-spread information gain equals the Jacobian formula."""
    Z = gr.make_grid(21)
    w = jax.nn.softmax(jnp.zeros(Z.shape[0]))
    A = jnp.array([[2.0, 0.5], [0.0, 1.0]], jnp.float32)
    R = jnp.eye(2) * 0.1
    m, P = gr.grid_moments(jnp.log(w), Z)
    np.testing.assert_allclose(gr.predictive_info_gain(Z @ A.T, w, R), gr.info_gain(A, P, R), rtol=1e-4)
