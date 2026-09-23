"""Cross-check rlquantopt.jx physics and metrics against the v1 QuTiP/weylchamber implementation."""
import numpy as np
import pytest
import qutip
import scipy.linalg
import weylchamber

from rlquantopt.jx import physics, metrics
from rlquantopt.rl_envs.zcqubits import ZCQubits, setup_ZCQubits4MKrauss_params

DT = 0.05


@pytest.fixture(scope="module")
def v1_model():
    return ZCQubits(**setup_ZCQubits4MKrauss_params())


def test_full_hamiltonian_matches_v1(v1_model):
    drift, control = physics.full_hamiltonian(physics.ModelParams())
    np.testing.assert_allclose(np.asarray(drift), v1_model.drift.full(), atol=1e-12)
    np.testing.assert_allclose(np.asarray(control), v1_model.control.full(), atol=1e-12)


def test_hamiltonian_conserves_excitations(v1_model):
    H = v1_model.drift.full()
    n_exc = np.array([i // 9 + (i // 3) % 3 + i % 3 for i in range(27)])
    mixes = np.abs(H[n_exc[:, None] != n_exc[None, :]])
    assert mixes.max() == 0.0


def _full_propagation(v1_model, amps):
    H0, C = v1_model.drift.full(), v1_model.control.full()
    U = np.eye(27, dtype=complex)
    for u in amps:
        U = scipy.linalg.expm(-1j * (H0 + u * C) * DT) @ U
    return U


def test_sector_propagation_matches_full_expm(v1_model):
    rng = np.random.default_rng(0)
    amps = rng.uniform(-20, 20, size=60)
    state = physics.propagate(physics.sector_hamiltonian(physics.ModelParams()),
                              physics.initial_state(), amps, DT)
    U = _full_propagation(v1_model, amps)
    np.testing.assert_allclose(np.asarray(state.U1),
                               U[np.ix_(physics.SECTOR1_IDX, physics.SECTOR1_IDX)], atol=1e-10)
    np.testing.assert_allclose(np.asarray(state.psi2), U[physics.SECTOR2_IDX, 12], atol=1e-10)


def test_sector_propagation_matches_qutip_solver(v1_model):
    """Same stepping as v1 ZCQPEE.forward_dynamics (piecewise-constant QobjEvo, SESolver.step)."""
    rng = np.random.default_rng(1)
    amps = rng.uniform(-10, 10, size=30)
    solver = qutip.SESolver(v1_model.H, options=dict(method='adams', atol=1e-12, rtol=1e-10))
    psi0 = qutip.ket((1, 1, 0), dim=[3, 3, 3])
    solver.start(psi0, 0.)
    for k, u in enumerate(amps):
        psi = solver.step((k + 1) * DT, args=dict(A=u))
    state = physics.propagate(physics.sector_hamiltonian(physics.ModelParams()),
                              physics.initial_state(), amps, DT)
    np.testing.assert_allclose(np.asarray(state.psi2), psi.full().ravel()[physics.SECTOR2_IDX], atol=1e-7)


def _v1_gate(U):
    basis = [0, 3, 9, 12]
    # G[i, j] = <psi_i|b_j> with psi_i = U|b_i>
    return np.array([[np.conj(U[bj, bi]) for bj in basis] for bi in basis])


def test_realised_gate_and_metrics_match_weylchamber(v1_model):
    rng = np.random.default_rng(2)
    for trial in range(20):
        amps = rng.uniform(-20, 20, size=rng.integers(20, 300))
        state = physics.propagate(physics.sector_hamiltonian(physics.ModelParams()),
                                  physics.initial_state(), amps, DT)
        G = np.asarray(physics.realised_gate(state))
        G_ref = _v1_gate(_full_propagation(v1_model, amps))
        np.testing.assert_allclose(G, G_ref, atol=1e-10)

        c_ref = weylchamber.c1c2c3(qutip.Qobj(G_ref, dims=[[2, 2], [2, 2]]))
        np.testing.assert_allclose(np.asarray(metrics.c1c2c3(G)), c_ref, atol=2e-8)
        C_ref = weylchamber.concurrence(*c_ref)
        U_ref = (qutip.Qobj(G_ref).dag() * qutip.Qobj(G_ref)).tr().real / 4
        JT, C, U = metrics.cost_JT(G)
        np.testing.assert_allclose(float(C), C_ref, atol=1e-7)
        np.testing.assert_allclose(float(U), U_ref, atol=1e-12)
        np.testing.assert_allclose(float(JT), 1 - (C_ref + 3 * U_ref) / 4, atol=1e-7)


_S = 1 / np.sqrt(2)
STANDARD_GATES = {
    "identity": np.eye(4),
    "iswap": np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]]),
    "sqrt_iswap": np.array([[1, 0, 0, 0], [0, _S, 1j * _S, 0], [0, 1j * _S, _S, 0], [0, 0, 0, 1]]),
    "swap": np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]),
    "cz": np.diag([1, 1, 1, -1]),
    "cphase_pi3": np.diag([1, 1, 1, np.exp(1j * np.pi / 3)]),
    "leaky": np.diag([1, 0.9, 0.95, 0.8 * np.exp(0.3j)]),
}


@pytest.mark.parametrize("name", sorted(STANDARD_GATES))
def test_c1c2c3_on_standard_gates(name):
    G = STANDARD_GATES[name].astype(complex)
    c_ref = weylchamber.c1c2c3(qutip.Qobj(G, dims=[[2, 2], [2, 2]]))
    np.testing.assert_allclose(np.asarray(metrics.c1c2c3(G)), c_ref, atol=1e-8)
    np.testing.assert_allclose(float(metrics.concurrence(metrics.c1c2c3(G))), weylchamber.concurrence(*c_ref), atol=1e-8)
