"""Idea i10: the realistic tunable-coupler device model (rlquantopt/jx/device.py)."""
import numpy as np
import pytest
from scipy.linalg import expm

from rlquantopt.jx import config  # noqa: F401  (float64)
import jax.numpy as jnp

from rlquantopt.jx import device as dv, metrics

P = dv.DeviceParams()
PHI0 = dv.idle_flux(P)


def full_hamiltonian(phi, p=P):
    """The 27-level Hamiltonian without truncation, rad/ns, kets |q1 q2 c>."""
    a = np.diag(np.sqrt(np.arange(1, 3)), 1)
    I = np.eye(3)
    k = lambda x, y, z: np.kron(np.kron(x, y), z)
    A1, A2, C = k(a, I, I), k(I, a, I), k(I, I, a)
    n = lambda o: o.T @ o
    wc = float(dv.coupler_frequency(phi, p))
    s = np.sqrt(wc / p.omega_c_idle)
    X = lambda o: o + o.T
    H = (p.omega_1 * n(A1) + p.omega_2 * n(A2) + wc * n(C)
         + p.eta_1 / 2 * n(A1) @ (n(A1) - np.eye(27)) + p.eta_2 / 2 * n(A2) @ (n(A2) - np.eye(27))
         + p.eta_c / 2 * n(C) @ (n(C) - np.eye(27))
         + s * (p.g_1c * X(A1) @ X(C) + p.g_2c * X(A2) @ X(C)) + p.g_12 * X(A1) @ X(A2))
    return 2 * np.pi * H


def test_flux_curve():
    assert abs(float(dv.coupler_frequency(0.0, P)) - 6.7) < 1e-12
    assert abs(float(dv.coupler_frequency(0.5, P)) - 3.7) < 1e-9
    assert abs(float(dv.coupler_frequency(PHI0, P)) - 5.45) < 1e-9


def test_truncation_against_full_space():
    """Dropping states with more than 3 excitations: gate elements within 2e-3, fidelity within 1e-5."""
    h = dv.block_hamiltonian(P)
    basis = dv.dressed_basis(h, PHI0)
    dt, n = 0.1, 300
    t = (np.arange(n) + 0.5) * dt
    phis = PHI0 + 0.05 * np.sin(2 * np.pi * 0.16 * t) * np.sin(np.pi * t / (n * dt))
    G = np.asarray(dv.gate(dv.propagate(h, dv.initial_state(basis), phis, dt), basis, n * dt))
    # full space: dressed idle states from the full Hamiltonian, same frame convention
    H0 = full_hamiltonian(PHI0)
    w, v = np.linalg.eigh(H0)
    idx = [0, 3, 9, 12]         # |000>, |010>, |100>, |110>
    cols = []
    for i in idx:
        j = int(np.argmax(np.abs(v[i, :])))
        cols.append((v[:, j] * np.exp(-1j * np.angle(v[i, j])), w[j]))
    U = np.eye(27, dtype=complex)
    for ph in phis:
        U = expm(-1j * full_hamiltonian(ph) * dt) @ U
    Gf = np.array([[np.exp(1j * ei * n * dt) * (vi.conj() @ U @ vj) for vj, _ in cols] for vi, ei in cols])
    # dropping N = 4 shifts only the |11> phase slightly (~1e-3 rad over 30 ns); fidelities agree to < 1e-5
    assert np.max(np.abs(G - Gf)) < 2e-3
    F = lambda g: float(metrics.fidelity_free_z(jnp.asarray(g).conj().T)[0])
    assert abs(F(G) - F(Gf)) < 1e-5


def test_idle_is_close_to_identity():
    h = dv.block_hamiltonian(P)
    basis = dv.dressed_basis(h, PHI0)
    G = np.asarray(dv.gate(dv.propagate(h, dv.initial_state(basis), np.full(200, PHI0), 0.1), basis, 20.0))
    np.testing.assert_allclose(np.abs(np.diag(G)), 1, atol=1e-6)


def test_parametric_exchange_happens():
    """Modulating the coupler flux at the qubit detuning drives |01> <-> |10> (the parametric iSWAP)."""
    h = dv.block_hamiltonian(P)
    basis = dv.dressed_basis(h, PHI0)
    (ve, ee), (vo, eo) = basis
    f = abs(float(eo[1] - eo[0])) / (2 * np.pi)          # dressed qubit-qubit detuning, GHz
    dt, n = 0.1, 1000
    t = (np.arange(n) + 0.5) * dt
    best = 0.0
    for amp in (0.02, 0.04, 0.08):
        phis = PHI0 + amp * np.cos(2 * np.pi * f * t)
        G = np.asarray(dv.gate(dv.propagate(h, dv.initial_state(basis), phis, dt), basis, n * dt))
        best = max(best, abs(G[1, 2]) ** 2)
    assert best > 0.2


def _dev_cfg(**kw):
    from rlquantopt.jx import env as jenv
    base = dict(physics_model="device", obs_mode="context", action_mode="carrier", objective="sqrt_iswap", oob_mode="clip",
                T=150.0, pulse_length=1500, n_time_steps=50, a_scale=0.1, carrier_steps=(0.004, 0.2, 0.004),
                awg_dt=1.0, filter_tau=0.5, context_noise_mhz=(0.0, 0.0, 0.0))
    base.update(kw)
    return jenv.EnvConfig(**base)


def test_device_env_calibration_and_step():
    import jax
    from rlquantopt.jx import env as jenv
    cfg = _dev_cfg(flux_drift_mphi0=20.0, qubit_drift_mhz=5.7)
    obs, s = jenv.reset(jax.random.PRNGKey(1), cfg)
    assert obs.shape == (cfg.obs_dim,) == (3 + 4 + 1,)
    # the calibration sees the coupler move with the flux drift (about -8.6 MHz per mPhi0 at idle)
    phi_off = float(s.ham[2] - dv.idle_flux(P))
    assert abs(float(s.context[2]) - 1e3 * (float(dv.coupler_frequency(dv.idle_flux(P) + phi_off, P)) - 5.45)) < 1e-6
    obs, s, r, term, trunc, info = jenv.step(s, jnp.array([1.0, 0.0, 0.0]), cfg)
    assert np.isfinite(float(r)) and 0 <= float(info["JT"]) <= 1 and not bool(trunc)


def test_device_filter_smooths_the_awg_steps():
    import jax
    from rlquantopt.jx import env as jenv
    for tau, expect_smooth in ((0.0, False), (0.5, True)):
        cfg = _dev_cfg(filter_tau=tau)
        _, s = jenv.reset(jax.random.PRNGKey(0), cfg)
        amps = []
        for _ in range(4):
            _, s, *_ = jenv.step(s, jnp.array([1.0, 0.0, 0.0]), cfg)
            amps.append(np.asarray(s.amps_cur))
        jumps = np.abs(np.diff(np.concatenate(amps))).max()
        assert (jumps < 0.5 * 0.016) == expect_smooth      # an AWG step of the carrier is up to ~A * 2 pi f * 1 ns


def test_simplified_model_is_rwa_and_linear():
    """The simplified variant conserves excitation number and agrees with the full model at the idle point to
    within the dispersive shifts the RWA drops."""
    hs = dv.block_hamiltonian(P, simplified=True)
    He, Ho = dv.h_blocks(hs, PHI0)
    # excitation number conserved: no element couples N = 1 and N = 3 in the odd block
    n = np.array([sum(s) for s in dv.ODD])
    assert np.max(np.abs(np.asarray(Ho)[np.ix_(n == 1, n == 3)])) == 0
    f_full = dv.coupler_frequency(PHI0 + 0.01, P)
    slope, phi0 = dv.linearisation(P)
    assert abs(float(f_full) - (5.45 + slope * 0.01)) < 0.05      # curvature over 10 mPhi0 stays below 50 MHz
