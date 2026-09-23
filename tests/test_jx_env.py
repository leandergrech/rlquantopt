"""Step-by-step equivalence of rlquantopt.jx.env with the v1 ZCQPEE environment."""
import numpy as np
import pytest
import jax

from rlquantopt.jx import env as jenv
from rlquantopt.rl_envs.zc_qpee import ZCQPEE

V1_KW = dict(pulse_length=1000, T=50.0, delta_mode=True, action_scaling={'z': 20.0}, a_norm_max=1.0,
             n_time_steps=3, tv_penalty_scale=1e-3, act_poly_order=1, add_prev_obs=False, use_fidelity=False,
             rew_scale=1.0, fid_thresh=0.99)
# The current v1 code sets REW_THRESH = 0; the paper run subtracted -log10(0.99). Compare against the current code.
CFG = jenv.EnvConfig(log_lim=0.0)
# v1 integrates with an adaptive ODE solver (atol 1e-10, rtol 1e-8), so it drifts from the
# exact propagation by ~1e-6 over an episode; the JAX env matches full-space expm to ~1e-13 (see below).
TOL = 1e-5  # v1 observations are float32


def _amplitudes(o):
    o = np.asarray(o, float) / 0.9
    return (o[:24:2] + 1) / 2 * np.exp(1j * np.pi * o[1:24:2])


def _obs_close(o_jx, o_v1):
    # Compare the reconstructed complex amplitudes; raw phases of near-zero amplitudes are ill-conditioned.
    np.testing.assert_allclose(_amplitudes(o_jx), _amplitudes(o_v1), atol=TOL)
    np.testing.assert_allclose(np.asarray(o_jx)[24:], np.asarray(o_v1)[24:], atol=TOL)


@pytest.mark.parametrize("seed,scale", [(0, 0.05), (1, 0.2), (2, 1.0)])
def test_matches_v1_step_by_step(seed, scale):
    v1 = ZCQPEE(**V1_KW)
    o_v1, _ = v1.reset(seed=seed)
    o_jx, st = jenv.reset(jax.random.PRNGKey(seed), CFG)
    _obs_close(o_jx, o_v1)
    rng = np.random.default_rng(seed)
    step = jax.jit(jenv.step, static_argnums=2)
    for k in range(CFG.n_steps):
        a = np.clip(rng.normal(0, scale, size=3), -1, 1)  # float64 so v1 does not accumulate in float32
        o_v1, r_v1, term_v1, trunc_v1, info_v1 = v1.step(a[None])
        o_jx, st, r_jx, term_jx, trunc_jx, info_jx = step(st, a, CFG)
        _obs_close(o_jx, o_v1)
        assert bool(term_jx) == term_v1 and bool(trunc_jx) == trunc_v1, k
        np.testing.assert_allclose(float(info_jx['unitarity']), info_v1['unitarity'].real, atol=TOL)
        np.testing.assert_allclose(float(info_jx['concurrence']), info_v1['concurrence'], atol=30 * TOL)  # ill-conditioned when |b|, |c| << 1
        np.testing.assert_allclose(float(r_jx), r_v1, rtol=30 * TOL, atol=30 * TOL)
        if term_v1 or trunc_v1:
            break
    assert k > 0


def test_episode_length():
    assert CFG.n_steps == 333
    obs, st = jenv.reset(jax.random.PRNGKey(0), CFG)
    assert obs.shape == (CFG.obs_dim,) == (28,)


def test_matches_exact_full_space_expm():
    import scipy.linalg
    from rlquantopt.jx import physics
    H0, C = (np.asarray(x) for x in physics.full_hamiltonian(CFG.model))
    obs, st = jenv.reset(jax.random.PRNGKey(0), CFG)
    step = jax.jit(jenv.step, static_argnums=2)
    rng = np.random.default_rng(3)
    U = np.eye(27, dtype=complex)
    for _ in range(100):
        obs, st, r, term, trunc, info = step(st, rng.normal(0, 0.1, size=3), CFG)
        for u in np.asarray(st.amps_cur):
            U = scipy.linalg.expm(-1j * (H0 + u * C) * CFG.dt) @ U
    b = [0, 3, 9, 12]
    G = np.array([[np.conj(U[bj, bi]) for bj in b] for bi in b])
    np.testing.assert_allclose(float(info['unitarity']), np.sum(np.abs(G) ** 2) / 4, atol=1e-12)
