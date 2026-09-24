# Getting started

## Environment

The JAX work uses its own conda environment, `rlqo-jax` (Python 3.12). The old `qvaqt`
environment no longer runs v1 (qutip 5.0.2 breaks with the scipy installed there), so the JAX
environment also carries a working qutip, torch (CPU) and SB3 for the cross-checks against v1.

```bash
conda create -n rlqo-jax python=3.12
conda activate rlqo-jax
pip install -r requirements-jax.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu    # only for loading v1 checkpoints
pip install "stable-baselines3==2.3.1" "sb3-contrib==2.3.0"             # only for the v1 equivalence tests
pip install -e . --no-deps
```

`--no-deps` matters: `setup.py` still pins the v1 stack (torch 2.1, numpy 1.24), which would
downgrade the JAX environment.

## Run the tests

```bash
pytest tests/            # 27 tests, ~30 s on the CPU
```

| Test file | What it guarantees |
| --- | --- |
| `test_jx_physics.py` | Hamiltonian, sector propagation and Weyl metrics match QuTiP, full `expm` and `weylchamber` |
| `test_jx_env.py` | The JAX env matches v1 `ZCQPEE` step by step, and full-space `expm` to 1e-12 |
| `test_jx_agents.py` | Gaussian log-prob/KL, conjugate gradient, GAE, one PPO and one TRPO update |
| `test_jx_robustness.py` | The paper's stored pulses give the paper's J_T |
| `test_jx_sb3_import.py` | The paper's SB3 policy loads and regenerates the paper's pulse (skipped without the v1 checkpoints) |

## CPU or GPU, float64 or float32

```bash
JAX_PLATFORMS=cpu python ...      # the fastest float64 option on the laptop
RLQO_X64=0 python ...             # float32: smoke tests only
```

| Backend | Precision | Env steps/s | Use it for |
| --- | --- | --- | --- |
| CPU, 16 threads | float64 | ~40k | training, sweeps, GRAPE |
| GPU (T550) | float64 | ~14k | background jobs while the CPU is busy |
| GPU (T550) | float32 | ~230k | smoke tests only: J_T is off by ~1e-4 |

On a data-centre GPU (A100/H100) float64 is fast, and the same code should run far faster there.

## A first session

```python
import jax, numpy as np
from rlquantopt.jx import env as jenv
from rlquantopt.jx.robustness import load_pulse_csv

# 1. Evaluate the paper's RL pulse
amps, cfg = load_pulse_csv("rlquantopt/paper_plots/final_pulses/RL_pulse.csv")
m = jenv.rollout_pulse(amps, cfg)              # J_T, C, U after every 50 ps sample
print(np.min(m["JT"]))                         # 9.49e-05, at 17.25 ns

# 2. Step the environment with random actions
cfg = jenv.EnvConfig()
obs, state = jenv.reset(jax.random.PRNGKey(0), cfg)
step = jax.jit(jenv.step, static_argnums=2)   # cfg is static: it sets array shapes
for _ in range(10):
    obs, state, reward, term, trunc, info = step(state, 0.1 * np.random.randn(3), cfg)
print(float(info["JT"]), float(reward))
```

Train an agent (writes to `runs/`, which git ignores):

```bash
JAX_PLATFORMS=cpu python -m rlquantopt.jx.train --algo trpo --seed 123 --total-steps 2000000 --eval-every 262144
```

A 2M-step run takes about 12 minutes. In the replication run, J_T was ≈ 6e-2 at 2M steps, just
before the breakthrough to perfect entanglers at 2-4M steps (the paper's agent broke through at the
same point). The full 20M-step replication takes
about 2 hours. All scripts are listed in [Scripts and CLI](reference/scripts.md).
