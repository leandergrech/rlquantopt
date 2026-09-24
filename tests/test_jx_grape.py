import numpy as np
import jax
import jax.numpy as jnp

from rlquantopt.jx import grape, physics

MODEL = physics.ModelParams()


def test_single_member_ensemble_equals_nominal():
    u = grape.random_guesses(jax.random.PRNGKey(0), 1, 120, 20.0, 0.05)[0]
    ham = physics.sector_hamiltonian(MODEL)
    hams = grape.ensemble_hamiltonian(MODEL, np.asarray(MODEL.omega_s)[None])
    cfg = grape.GrapeConfig()
    np.testing.assert_allclose(float(grape.final_cost(u, ham, cfg)[0]),
                               float(jax.vmap(lambda h: grape.final_cost(u, h, cfg)[0])(hams)[0]), rtol=1e-12)


def test_detuning_grid_is_centred():
    om = grape.detuning_grid(MODEL, 10.0, 3)
    assert om.shape == (9, 2)
    np.testing.assert_allclose(om[4], MODEL.omega_s)
    np.testing.assert_allclose(om[0] - np.asarray(MODEL.omega_s), [-0.01, -0.01])


def test_optimise_improves_and_respects_bound():
    cfg = grape.GrapeConfig(n_iter=60)
    ham = physics.sector_hamiltonian(MODEL)
    u0 = grape.random_guesses(jax.random.PRNGKey(1), 2, 240, 2 * np.pi * 1.5, cfg.dt)
    u, JT, C, U, hist = grape.optimise(u0, ham, 2 * np.pi * 1.5, cfg)
    assert np.all(np.abs(np.asarray(u)) <= 2 * np.pi * 1.5 + 1e-9)
    assert np.all(np.asarray(JT) <= np.asarray(hist)[:, 0])

    hams = grape.ensemble_hamiltonian(MODEL, grape.detuning_grid(MODEL, 5.0, 2))
    u, JT, *_ = grape.optimise(u0[:1], hams, 2 * np.pi * 1.5, cfg)
    assert np.isfinite(float(JT[0]))
