"""Regression values for the paper's stored pulses (rlquantopt/paper_plots/final_pulses)."""
import os

import numpy as np
import pytest

from rlquantopt.jx import env as jenv, robustness as rb

PULSES = os.path.join(os.path.dirname(__file__), "..", "rlquantopt", "paper_plots", "final_pulses")


def _load(name):
    path = os.path.join(PULSES, name)
    if not os.path.exists(path):
        pytest.skip(f"{path} not available")
    return rb.load_pulse_csv(path)


def test_rl_pulse_reaches_paper_errors():
    amps, cfg = _load("RL_pulse.csv")
    m = jenv.rollout_pulse(amps, cfg)
    JT, C, U = (np.asarray(m[k]) for k in ("JT", "concurrence", "unitarity"))
    t = (np.arange(len(JT)) + 1) * cfg.dt
    assert np.all(C[t >= 10.0] == 1.0)                     # perfect entangler from ~10 ns on
    np.testing.assert_allclose(JT.min(), 9.4857e-05, rtol=1e-3)
    np.testing.assert_allclose(JT[-1], 3.5283e-04, rtol=1e-3)
    assert 1 - U[-1] < 1e-3


def test_krotov_good_guess_nominal():
    amps, cfg = _load("krotov_good_guess_pulse.csv")
    JT = float(rb.final_JT(amps, cfg, np.asarray(cfg.model.omega_s)))
    np.testing.assert_allclose(JT, 9.88e-05, rtol=1e-2)


def test_detuning_map_center_equals_nominal():
    amps, cfg = _load("RL_pulse.csv")
    M = rb.detuning_map(amps, cfg, [-1.0, 0.0, 1.0], [0.0])
    np.testing.assert_allclose(M[1, 0], float(rb.final_JT(amps, cfg, np.asarray(cfg.model.omega_s))), rtol=1e-10)
