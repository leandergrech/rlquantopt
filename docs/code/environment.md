# Environment: `rlquantopt/jx/env.py`

`env.py` is a functional re-implementation of v1 `ZCQPEE` with the semantics of the paper run
(`05-12-24_201634`). There is no environment object: state is an explicit pytree and every
function is pure.

```python
cfg = EnvConfig()                                   # static: sets shapes and episode length
obs, state = reset(key, cfg)
obs, state, reward, terminated, truncated, info = step(state, action, cfg)
```

## The MDP in one table

| | Value (paper run) |
| --- | --- |
| Pulse | 1000 samples of 50 ps (T = 50 ns), piecewise constant |
| Step | K = 3 samples; 333 steps per episode |
| Action | \(a \in [-1, 1]^3\): amplitude **deltas**, × 20 rad/ns, cumulatively summed |
| Amplitude bound | \(|u| \le 20\) rad/ns (10/π ≈ 3.18 GHz); leaving it clips \(u\) and truncates the episode |
| Observation | 28 numbers: 12 complex amplitudes in polar form, the 3 segment amplitudes / 20, time, all × 0.9 |
| Reward | \(-\log_{10} J_T - 0.0044 - 10^{-3}\sum|\Delta u|\) per step; \(-20(1 - t/T)\) on truncation |

## One step

```python
--8<-- "rlquantopt/jx/env.py:step"
```

Read it in four blocks:

1. **Action → amplitudes.** The deltas are summed onto the last amplitude of the previous segment
   (0 at the first step). If any amplitude leaves the bound, it is clipped and `oob` is set.
2. **Physics.** `propagate` applies the three samples to the sector state; `realised_gate` and
   `cost_JT` give \(J_T\), \(C\), \(U\) at the end of this segment.
3. **Reward.** \(-\log_{10} J_T\) minus the v1 offset `rew_thresh` (\(-\log_{10} 0.99\)) minus the TV
   penalty on the segment. On `oob` the reward is replaced by the truncation penalty.
4. **Bookkeeping.** The observation uses the time **before** the index advances (as v1 does); the
   episode terminates when the next segment would run past the pulse.

Nothing here is a Python branch on data: `jnp.where` replaces `if`, so the step can be compiled
and vectorised.

## Observation

```python
--8<-- "rlquantopt/jx/env.py:observation"
```

`sector_amplitudes` returns the N=1 images of \|010⟩ and \|100⟩ (3 amplitudes each) and the N=2
image of \|110⟩ (6 amplitudes): exactly the 12 entries v1 extracted from the 27-dim state vector.
Each complex amplitude \(z\) becomes \((2|z| - 1,\ \arg z / \pi)\).

!!! note "The observation is a simulator privilege"
    State amplitudes are not measurable on hardware. That is fine for replicating the paper; for a
    sim-to-real version use `obs_mode="measured"` below.

### Measured observations (`obs_mode="measured"`)

With `EnvConfig(obs_mode="measured")` (CLI `--obs-mode measured`, idea i07) the agent sees only what a
lab can read out after preparing an input state and playing the pulse so far:

```python
--8<-- "rlquantopt/jx/physics.py:measured"
```

63 numbers: for the inputs \|01⟩, \|10⟩, \|11⟩ the readout probabilities of 00, 01, 10, 11 and of
"a transmon in \|2⟩" (15), and for the superposition inputs \|0+⟩, \|+0⟩, \|+1⟩ the 16 two-qubit Pauli
expectation values (48), which carry the relative phases that populations cannot see. The coupler is
traced out, since it is not read out; an excitation left in it reads as qubits in \|00⟩. Values are
exact expectations for now; finite-shot estimates are the next step. On hardware each value needs its
own experiment per step, which is the price of observing mid-pulse.

### Objective (`objective`)

`objective="pe"` (default) is the paper's perfect-entangler \(J_T\). `objective="sqrt_iswap"` (CLI
`--objective sqrt_iswap`) uses \(1 - F\), the average gate fidelity to √iSWAP after free virtual-Z
corrections ([Gate metrics](../system/metrics.md#named-gate-fidelity-with-free-z-corrections)); the
reward is \(-\log_{10}(1 - F)\) with the same shaping. `info["JT"]` always holds the active objective's
cost, so training, evaluation and GRAPE code work unchanged (`GrapeConfig(objective=...)`).

### Finite shots, calibration context, clipping (idea i09)

- `shots=N` (`--shots N`): the measured observables become estimates from N shots per measurement
  setting, drawn with the key carried in the environment state; one step costs 30 settings.

```python
--8<-- "rlquantopt/jx/env.py:shots"
```

- `obs_mode="context"` / `"measured+context"`: the policy also sees the qubit and coupler frequency
  offsets, measured once per episode with Gaussian error `context_noise_mhz` (spectroscopy, before the
  pulse); `"context"` alone is an open-loop policy that needs no measurement during the pulse.
- `oob_mode="clip"` (`--oob-mode clip --oob-penalty λ`): an amplitude beyond the bound is clipped and
  costs λ per unit of normalised excess instead of ending the episode.
- `delta_scale` (`--delta-scale`): rad/ns per sample for a unit action; scale it down for longer steps
  (`n_time_steps`), or one step sweeps the whole amplitude range.

See [ibis](../ideas/ibis.md) for what these do in practice.

## Auto-reset for training

Training runs thousands of environments in lock-step, so an environment that finishes must
restart inside the compiled loop:

```python
--8<-- "rlquantopt/jx/env.py:step_autoreset"
```

- Both branches are computed and `tree_map(jnp.where, ...)` picks one per environment.
- `info["final_obs"]` keeps the last observation of the finished episode, so the agent can
  bootstrap the value of a truncated episode (as Stable-Baselines3 does).
- `resample=True` draws new qubit frequencies at every reset (domain randomisation as the paper
  describes it). `resample=False` keeps the frequencies of the environment, which is what v1
  `ZCQPEEWRD` actually did: its drift was drawn once per env instance.

## Domain randomisation

```python
cfg = EnvConfig(max_drift=1e-3)       # omega_s *= 1 + U(-0.1 %, +0.1 %) at every reset
```

`sample_omega_s` draws the frequencies and `reset_to(omega, cfg)` builds the episode's sector
Hamiltonian, which lives in the state (`state.ham`). A `vmap` over environments therefore gives
each environment its own physics at no extra cost.

## Evaluating a fixed pulse

`rollout_pulse(amps, cfg, omega_s)` applies a stored pulse sample by sample and returns \(J_T\),
\(C\), \(U\) after every sample. The convention matches v1: `amps[k]` is held on \((t_k, t_{k+1}]\),
which is `amplist[k + 1]` in the pulse CSVs.

## Verified against v1

`tests/test_jx_env.py` replays three random action sequences (small, medium and saturating, the
last one hitting the amplitude bound) through v1 `ZCQPEE` and the JAX env, comparing observations,
rewards, termination and truncation at every step. The JAX env is also checked against full-space
`expm` to 1e-12; v1's adaptive ODE drifts ~1e-8 from exact over an episode.
