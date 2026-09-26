"""Idea i09 (ibis): amplitude clipping with a penalty, and finite-shot observations."""
import numpy as np

from rlquantopt.jx import config  # noqa: F401  (float64)
import jax
import jax.numpy as jnp

from rlquantopt.jx import env as jenv, physics


def test_clip_mode_keeps_going_and_penalises():
    base = dict(objective="sqrt_iswap", obs_mode="measured")
    push = jnp.ones(3)          # +20 rad/ns per sample: out of bounds within the first step
    for mode in ("terminate", "clip"):
        cfg = jenv.EnvConfig(oob_mode=mode, **base)
        obs, s = jenv.reset(jax.random.PRNGKey(0), cfg)
        _, s, r, term, trunc, info = jenv.step(s, push, cfg)
        assert bool(info["oob"])
        if mode == "terminate":
            assert bool(trunc) and float(r) < -10
        else:
            assert not bool(trunc)
            assert float(jnp.max(jnp.abs(s.amps_cur))) <= cfg.a_scale + 1e-12
            ref = jenv.EnvConfig(oob_mode="clip", oob_penalty=0.0, **base)
            _, s0 = jenv.reset(jax.random.PRNGKey(0), ref)
            _, _, r0, *_ = jenv.step(s0, push, ref)
            assert abs((float(r0) - float(r)) - 3.0) < 1e-9     # excess 0 + 1 + 2 in units of the bound


def test_shot_estimates_are_unbiased_with_binomial_spread():
    ham = physics.sector_hamiltonian(physics.ModelParams())
    u = jnp.full(120, 6.0)
    pops, paulis = physics.measured_observables(physics.propagate(ham, physics.initial_state(), u, 0.05))
    N, reps = 100, 4000
    keys = jax.random.split(jax.random.PRNGKey(1), reps)
    ep, eq = jax.vmap(lambda k: jenv.shot_estimates(k, pops, paulis, N))(keys)
    ep, eq = np.asarray(ep), np.asarray(eq)
    np.testing.assert_allclose(ep.mean(0), np.asarray(pops), atol=5 * 0.5 / np.sqrt(N * reps))
    np.testing.assert_allclose(eq.mean(0), np.asarray(paulis), atol=5 * 1.0 / np.sqrt(N * reps))
    p = np.clip(np.asarray(pops), 0, 1)
    np.testing.assert_allclose(ep.std(0), np.sqrt(p * (1 - p) / N), atol=0.01)


def test_noisy_observations_differ_and_default_is_exact():
    cfg = jenv.EnvConfig(obs_mode="measured", shots=50)
    exact = jenv.EnvConfig(obs_mode="measured")
    o1, _ = jenv.reset(jax.random.PRNGKey(0), cfg)
    o2, _ = jenv.reset(jax.random.PRNGKey(1), cfg)
    oe, _ = jenv.reset(jax.random.PRNGKey(0), exact)
    assert not np.allclose(o1, o2) and not np.allclose(o1, oe)
    assert np.all(np.abs(np.asarray(o1)) <= 1.0)
    assert jenv.EnvConfig(n_time_steps=15).obs_dim == 24 + 16 and jenv.EnvConfig(n_time_steps=15).n_steps == 66


def test_fidelity_from_observables_matches_exact():
    """With exact data the measurement-only estimate equals the true free-Z fidelity (coupler terms negligible)."""
    from rlquantopt.jx import grape, metrics
    ham = physics.sector_hamiltonian(physics.ModelParams())
    for seed in range(3):
        u = grape.random_guesses(jax.random.PRNGKey(seed), 1, 400, 20.0, 0.05)[0]
        s = physics.propagate(ham, physics.initial_state(), u, 0.05)
        F = float(metrics.fidelity_free_z(physics.realised_gate(s))[0])
        Fe = float(metrics.fidelity_from_observables(*physics.measured_observables(s)))
        assert abs(F - Fe) < 1e-3 * (1 - F) + 1e-9


def test_refine_counts_shots_and_improves():
    from rlquantopt.jx import grape, refine
    u0 = np.asarray(grape.random_guesses(jax.random.PRNGKey(3), 1, 345, 20.0, 0.05)[0])
    meas, exact, _ = refine.make_device(physics.ModelParams(), u0, 8, 200)
    x, tr = refine.spsa(meas, exact, 8, 200, max_shots=30 * 200 * 200, c=0.5, first_step=0.5)   # random-start steps
    assert tr.shots[-1] <= 30 * 200 * 200 and np.all(np.diff(tr.shots) > 0)
    assert tr.true_JT.min() < float(exact(jnp.zeros(8)))


def test_context_observation():
    """The calibration context is the drift in MHz plus measurement error, fixed over the episode."""
    for mode, dim in (("context", 3 + 4), ("measured+context", 63 + 3 + 4)):
        cfg = jenv.EnvConfig(obs_mode=mode, coupler_drift_mhz=140.0, max_drift=0.0011, context_noise_mhz=(0.0, 0.0, 0.0))
        obs, s = jenv.reset(jax.random.PRNGKey(4), cfg)
        assert obs.shape == (dim,) == (cfg.obs_dim,)
        m = cfg.model
        true = np.array([s.omega_s[0] - m.omega_s[0], s.omega_s[1] - m.omega_s[1], 0.0]) * 1e3
        np.testing.assert_allclose(np.asarray(s.context)[:2], true[:2], atol=1e-9)
        assert abs(float(s.context[2])) <= 140.0
        _, s2, *_ = jenv.step(s, jnp.zeros(3), cfg)
        np.testing.assert_array_equal(np.asarray(s2.context), np.asarray(s.context))


def test_carrier_mode():
    """Carrier actions steer (A, phi, offset); the samples oscillate at the qubit detuning."""
    cfg = jenv.EnvConfig(action_mode="carrier", obs_mode="measured", oob_mode="clip", n_time_steps=15)
    assert cfg.act_dim == 3 and cfg.obs_dim == 63 + 4 + 1
    obs, s = jenv.reset(jax.random.PRNGKey(0), cfg)
    amps = []
    for _ in range(20):
        obs, s, *_ = jenv.step(s, jnp.array([1.0, 0.0, 0.0]), cfg)
        amps.append(np.asarray(s.amps_cur))
    assert float(s.knobs[0]) == 20.0                       # A ramps by 1 rad/ns per step, capped at the bound
    u = np.concatenate(amps)[-200:]
    f = np.fft.rfftfreq(len(u), cfg.dt)[np.argmax(np.abs(np.fft.rfft(u))[1:]) + 1]
    assert abs(f - cfg.carrier_ghz) < 0.11                 # FFT resolution 1 / (200 * 50 ps) = 0.1 GHz
    ctx = jenv.EnvConfig(action_mode="carrier", obs_mode="context")
    assert ctx.obs_dim == 3 + 4 + 1
