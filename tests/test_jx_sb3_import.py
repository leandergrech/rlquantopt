"""The paper's SB3 policy, imported into JAX, reproduces the paper's RL pulse in the JAX env."""
import os

import numpy as np
import pandas as pd
import pytest
import jax

from rlquantopt.jx import env as jenv
from rlquantopt.jx.agents.common import evaluate

ROOT = os.path.join(os.path.dirname(__file__), "..", "rlquantopt")
ZIP = os.path.join(ROOT, "rl_agents", "ZCQPEE_pl-1000_T-50ns_delta_mode-TRPO", "05-12-24_201634",
                   "rl_model_12566528_steps.zip")
PULSE = os.path.join(ROOT, "paper_plots", "final_pulses", "RL_pulse.csv")

pytestmark = pytest.mark.skipif(not os.path.exists(ZIP), reason="paper checkpoints are not in the public repo")


def test_weights_and_activation():
    pytest.importorskip("torch")
    from rlquantopt.jx.sb3_import import load_sb3_policy, sb3_predict, sb3_activation
    assert sb3_activation(ZIP) == "relu"
    model, params = load_sb3_policy(ZIP)
    obs = np.random.default_rng(0).uniform(-1, 1, (8, 28)).astype(np.float32)
    np.testing.assert_allclose(np.asarray(model.dist(params["actor"], obs)[0]), sb3_predict(ZIP, obs), atol=1e-5)


def test_regenerates_paper_pulse():
    pytest.importorskip("torch")
    from rlquantopt.jx.sb3_import import load_sb3_policy
    model, params = load_sb3_policy(ZIP)
    out = jax.device_get(evaluate(model, params, jenv.EnvConfig()))
    pulse = np.asarray(out["amps"]).ravel()
    ref = pd.read_csv(PULSE)["amplist"].to_numpy()[1:]
    k = int(17.25 / 0.05)
    assert np.abs(pulse[:k] - ref[:k]).max() < 1e-2      # rad/ns, out of |u| <= 20
    r = np.asarray(out["reward"])
    np.testing.assert_allclose(np.asarray(out["t"])[r.argmax()], 17.25, atol=1e-9)
    np.testing.assert_allclose(np.asarray(out["JT"])[r.argmax()], 9.49e-5, rtol=0.02)
