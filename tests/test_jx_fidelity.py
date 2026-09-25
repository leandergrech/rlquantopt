"""Idea i06: named-gate fidelity with free virtual-Z corrections, and measurable observations."""
import numpy as np
import pytest
from scipy.optimize import minimize

from rlquantopt.jx import config  # noqa: F401  (float64)
import jax
import jax.numpy as jnp

from rlquantopt.jx import env as jenv, grape, metrics, physics

DT = 0.05


def gate_of(u, model=physics.ModelParams()):
    ham = physics.sector_hamiltonian(model)
    return physics.realised_gate(physics.propagate(ham, physics.initial_state(), jnp.asarray(u), DT))


def smooth_pulse(seed, n=345):
    return np.asarray(grape.random_guesses(jax.random.PRNGKey(seed), 1, n, 20.0, DT)[0])


def z(phi0, phi1):
    """Z(phi0) on q0 and Z(phi1) on q1 in the basis |00>, |01>, |10>, |11>."""
    return np.diag([1, np.exp(1j * phi1), np.exp(1j * phi0), np.exp(1j * (phi0 + phi1))])


def brute_force_fidelity(P, V, restarts=40, seed=0):
    """max over four independent Z phases (before and after), with scipy, from many starts."""
    rng = np.random.default_rng(seed)
    M0 = V.conj().T

    def neg(ph):
        M = M0 @ z(ph[0], ph[1]) @ P @ z(ph[2], ph[3])
        return -(np.trace(M @ M.conj().T).real + abs(np.trace(M)) ** 2) / 20

    return -min(minimize(neg, rng.uniform(-np.pi, np.pi, 4), method="BFGS").fun for _ in range(restarts))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fidelity_matches_brute_force(seed):
    G = gate_of(smooth_pulse(seed))
    F, _ = metrics.fidelity_free_z(G)
    P = np.asarray(metrics.physical_gate(G))
    assert abs(float(F) - brute_force_fidelity(P, np.asarray(metrics.SQRT_ISWAP))) < 1e-9


def test_target_up_to_z_has_unit_fidelity():
    V = np.asarray(metrics.SQRT_ISWAP)
    rng = np.random.default_rng(3)
    for _ in range(5):
        P = z(*rng.uniform(-np.pi, np.pi, 2)) @ V @ z(*rng.uniform(-np.pi, np.pi, 2)) * np.exp(1j * rng.uniform())
        F, U = metrics.fidelity_free_z(jnp.asarray(P).conj().T)       # realised_gate convention
        assert abs(float(F) - 1) < 1e-12 and abs(float(U) - 1) < 1e-12
    F, _ = metrics.fidelity_free_z(jnp.asarray(V))                    # sqrt(iSWAP)^dagger: Z-equivalent
    assert abs(float(F) - 1) < 1e-12


def test_gradient_matches_finite_differences():
    u = jnp.asarray(smooth_pulse(4, 120))
    f = lambda u: metrics.fidelity_free_z(gate_of(u))[0]
    g = jax.grad(f)(u)
    e = jnp.zeros_like(u).at[37].set(1e-6)
    fd = (f(u + e) - f(u - e)) / 2e-6
    assert abs(float(g[37]) - float(fd)) < 1e-6 * max(1.0, abs(float(fd)))


def test_measured_populations_match_amplitudes():
    ham = physics.sector_hamiltonian(physics.ModelParams())
    s = physics.propagate(ham, physics.initial_state(), jnp.asarray(smooth_pulse(5, 200)), DT)
    pops, paulis = physics.measured_observables(s)
    pops = np.asarray(pops).reshape(3, 5)
    U1, psi2 = np.asarray(s.U1), np.asarray(s.psi2)
    i1, i2 = physics.S1_POS_010, physics.S1_POS_100
    # input |01>: the coupler-excited |001> reads as qubits 00; |010> as 01; |100> as 10
    np.testing.assert_allclose(pops[0, :3], np.abs(U1[[0, 1, 2], i1]) ** 2, atol=1e-12)
    np.testing.assert_allclose(pops[1, :3], np.abs(U1[[0, 1, 2], i2]) ** 2, atol=1e-12)
    # input |11>: qubits read 11 only from |110>
    assert abs(pops[2, 3] - abs(psi2[physics.S2_POS_110]) ** 2) < 1e-12
    assert np.all(pops >= -1e-12) and np.allclose(pops.sum(1), 1)
    assert np.all(np.abs(np.asarray(paulis)) <= 1 + 1e-12)


@pytest.mark.parametrize("obs_mode", ["amplitudes", "measured"])
def test_env_modes_run(obs_mode):
    cfg = jenv.EnvConfig(objective="sqrt_iswap", obs_mode=obs_mode, coupler_drift_mhz=140.0)
    obs, state = jenv.reset(jax.random.PRNGKey(0), cfg)
    assert obs.shape == (cfg.obs_dim,)
    obs, state, r, term, trunc, info = jenv.step(state, jnp.full(3, 0.1), cfg)
    assert obs.shape == (cfg.obs_dim,) and np.isfinite(float(r)) and 0 <= float(info["JT"]) <= 1


def test_default_env_unchanged():
    """The defaults still give the paper's J_T and amplitude observations."""
    cfg = jenv.EnvConfig()
    obs, state = jenv.reset(jax.random.PRNGKey(0), cfg)
    _, state, _, _, _, info = jenv.step(state, jnp.full(3, 0.2), cfg)
    JT, _, _ = metrics.cost_JT(physics.realised_gate(state.sector))
    assert cfg.obs_dim == 28 and float(info["JT"]) == float(JT)
