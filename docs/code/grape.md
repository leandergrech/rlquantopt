# GRAPE: `rlquantopt/jx/grape.py`

Because the simulator is written in JAX, the gradient of \(J_T\) with respect to every pulse
sample comes for free: `jax.grad` differentiates straight through the `eigh`-based propagators and
the closed-form Weyl coordinates. `grape.py` uses that for three things:

| Use | Call |
| --- | --- |
| Plain GRAPE from random guesses; the QSL scan (paper Fig. 3) | `optimise(random_guesses(...), sector_hamiltonian(model), u_max, cfg)`, `qsl_scan` |
| Robust (ensemble) GRAPE over detuned Hamiltonians | `optimise(u0, ensemble_hamiltonian(model, omegas), u_max, cfg)` |
| Refinement of an RL pulse | `optimise(rl_pulse[None], ham, u_max, cfg)` |

## The optimiser

```python
--8<-- "rlquantopt/jx/grape.py:optimise"
```

How it works:

- **Bounded by construction.** The pulse is \(u = u_{\max}\tanh\theta\) and the optimiser moves
  \(\theta\), so every iterate respects \(|u| \le u_{\max}\) with no clipping or penalty.
- **Loss.** \(\log_{10} J_T\) rather than \(J_T\): it keeps gradients useful when \(J_T\) spans
  1e-1 to 1e-6.
- **Ensemble.** If `ham` has a leading axis, the cost is the mean \(J_T\) over it (computed with
  `vmap`). Minimising \(\log_{10}\) of the mean penalises the worst members most, which is what
  makes the pulse robust.
- **Best iterate kept.** Adam with a cosine-decayed step is not monotone, so the loop tracks the
  best \(\theta\) seen.
- **Restarts in parallel.** `vmap(run_one)` optimises a batch of initial pulses at once.

## Robust GRAPE in three lines

```python
from rlquantopt.jx import grape, physics
model = physics.ModelParams()
omegas = grape.detuning_grid(model, half_width_mhz=10.0, n_per_axis=5)   # 25 detuned systems
hams = grape.ensemble_hamiltonian(model, omegas)                          # stacked sector Hamiltonians
u, JT_mean, C, U, history = grape.optimise(u0, hams, u_max=20.0, cfg=grape.GrapeConfig())
```

## Refining an RL pulse

```python
import pandas as pd, jax.numpy as jnp
rl = jnp.asarray(pd.read_csv("rlquantopt/paper_plots/final_pulses/RL_pulse.csv")["amplist"].to_numpy()[1:346])
cfg = grape.GrapeConfig(n_iter=2000, lr=0.01)       # small steps: the start is already good
u, JT, *_ = grape.optimise(rl[None], physics.sector_hamiltonian(model), 20.0, cfg)
```

On the paper's RL pulse (J_T = 9.5e-5 at 17.25 ns), 200 iterations take 2 s on the CPU and reach
J_T = 7.3e-6. Results of the full comparison are in
[RL vs GRAPE vs robust GRAPE](../results/robustness.md).

## Cost

One GRAPE iteration is one forward and one backward pass through the pulse per ensemble member
and restart. A 345-sample pulse costs about 0.1 ms per member per iteration on the CPU: 2000
iterations of a 25-member ensemble with 4 restarts take ~25 minutes.

## Limits

- GRAPE sees the whole Hamiltonian and its gradients; RL sees only rewards. Comparisons of
  robustness are fair only at the same gate time, amplitude bound and model, which is how the
  experiment is set up. Comparisons of sample cost between the two are not like for like.
- Gradients near \(abcd = 0\) are replaced with 0 (see [Gate metrics](../system/metrics.md)); starting from
  \(u = 0\) therefore does not move. Random guesses always include a component at the qubit-qubit
  detuning (0.86 GHz) to start inside the useful region.
