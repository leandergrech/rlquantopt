# Research agents: the code of the idea farm

Each [idea](../ideas/index.md) adds code next to PPO rather than changing it, so the baseline stays
bit-for-bit what it was and every idea can be switched off. This page walks through that code, in the
order of the animals; the results are on each animal's page. Stamps as on the farm:
<span class="stamp stamp--active">🟢 active</span> <span class="stamp stamp--concluded">🏁 concluded</span>
<span class="stamp stamp--archived">📦 archived</span>.

| Module | Idea | Entry point | Status |
| --- | --- | --- | --- |
| `agents/autoregressive.py`, `agents/ppo_judge.py` | 🐸 axolotl, 🦡 badger | `train.py --algo ppo_judge` | 🏁 |
| `agents/chameleon.py` | 🐲 chameleon | `train.py --algo chameleon` | 📦 |
| `agents/dragonfly.py` | 🦋 dragonfly | `train.py --algo dragonfly` | 📦 |
| `agents/ppo_plus.py`, `toy_envs.py`, `toybench.py` | 🦔 echidna | `python -m rlquantopt.jx.toybench`, `train.py --algo ppo_plus` | 🏁 |
| `agents/fox.py`, `foxbench.py` | 🦊 fox | `python -m rlquantopt.jx.foxbench` | 🏁 |
| `env.py` (`objective`, `obs_mode`), `physics.measured_observables`, `metrics.fidelity_free_z` | 🦎 gecko | `train.py --objective sqrt_iswap --obs-mode measured` | 🏁 |
| `hippogriff.py`, `toy_envs.DriftTracker` | 🦄 hippogriff | `python -m rlquantopt.jx.hippogriff` | 🟢 |
| `env.py` (`shots`, `oob_mode`, `action_mode="carrier"`, context), `metrics.fidelity_from_observables`, `refine.py` | 🦩 ibis | `train.py --action-mode carrier --obs-mode context ...`, `scripts/ibis_eval.py` | 🏁 |
| `device.py`, `env_device.py` | 🐺 jackal | `train.py --physics device`, `scripts/jackal_eval.py` | 🟢 |

## 🐸 🦡 A sample judge, and a KL brake (`ppo_judge.py`)

**The judge's target.** For a set of probe samples, how well does each one's PPO gradient agree with the
gradient of the rest of the batch? Per-sample gradients are one `vmap` of `grad` away:

```python
--8<-- "rlquantopt/jx/agents/ppo_judge.py:help_target"
```

A small network of (state, action, reward) learns to predict this cosine with a mean and a standard
deviation; only confident verdicts reweight the PPO loss, renormalised so that the step size is unchanged:

```python
--8<-- "rlquantopt/jx/agents/ppo_judge.py:guided_loss"
```

The target is the advantage's sign times a gradient alignment; a judge that sees only the reward cannot
predict it, which is what the runs found.

**Badger's brake** is SB3's `target_kl`: once a minibatch starts above 1.5 × the target KL, the remaining
steps of the update are skipped. Inside a `scan` nothing can break out of the loop, so every step is computed
and a flag keeps the old parameters:

```python
--8<-- "rlquantopt/jx/agents/ppo_judge.py:kl_stop"
```

**The autoregressive policy** (`autoregressive.py`) emits the three amplitude deltas of a step one at a time
from one head, fed its own previous delta and the running amplitude. It exposes the same
`sample / mode / logp / entropy / value` interface as `ActorCritic`, and `logp` teacher-forces the stored
actions, so `collect`, `evaluate` and the PPO loss use it unchanged.

## 🐲 Latent tasks from a jumpy world model (`chameleon.py`)

A planner picks a latent task every 8 steps: the direction of steepest rise of the model's value, mixed
with a random direction orthogonal to it, so exploring never undoes exploiting:

```python
--8<-- "rlquantopt/jx/agents/chameleon.py:goal"
```

??? example "The world model's loss: γ-discounted latent jumps, λ-returns, VICReg"

    ```python
    --8<-- "rlquantopt/jx/agents/chameleon.py:model_loss"
    ```

## 🦋 A guided random walk with a pessimistic search (`dragonfly.py`)

One `act` call holds the whole idea: a goal step (exploit direction, pessimistic value, the position in the
golden cache, the guide's decision on persistence τ, spread σ and exploration share ε), an
Ornstein-Uhlenbeck walk in latent space, and a search over candidate actions scored by alignment with the
task plus an ensemble's lower bound:

```python
--8<-- "rlquantopt/jx/agents/dragonfly.py:act"
```

??? example "History as leaky-integrator traces, trained in parallel with `associative_scan`"

    ```python
    --8<-- "rlquantopt/jx/agents/dragonfly.py:model_loss"
    ```

## 🦔 One ingredient at a time (`ppo_plus.py`, `toy_envs.py`)

`ppo_plus` is PPO with three switches, each off by default: Ornstein-Uhlenbeck action noise with the policy
conditioned on the previous noise (so the PPO ratio stays an exact likelihood ratio), a pessimistic
Q-ensemble veto, and golden-cache self-imitation [[oh2018]](../bibliography.md#oh2018). Acting:

```python
--8<-- "rlquantopt/jx/agents/ppo_plus.py:act"
```

The golden cache keeps the best transitions by return and by reward, at most one per ball in standardised
observation space (canonical), rebuilt every rollout inside `jit` with a fixed-size `fori_loop`:

```python
--8<-- "rlquantopt/jx/agents/ppo_plus.py:cache"
```

`toy_envs.py` gives Pendulum, MountainCar and a Tracker (a miniature gate problem: delta actions on a bounded
amplitude, \(-\log_{10}\) error reward) the same functional interface as the gate environment, and wraps the
gate environment itself (`Ham`) in it, so one agent runs on all of them. `DriftTracker` adds a hidden
device gain and offset for fox and hippogriff.

## 🦊 An improvement-equivalent model (`fox.py`)

Fox does not model the environment. After every PPO update it records the step taken in actor-parameter
space and the return change it caused, and fits a Bayesian linear regression of the gain per unit step on
the step's direction, in the basis of the last 8 PPO steps; the prior and noise precisions are learned by
evidence maximisation [[mackay1992]](../bibliography.md#mackay1992):

```python
--8<-- "rlquantopt/jx/agents/fox.py:improvement_model"
```

`FoxAdapter` wraps any PPO update: it adds half a PPO step along the posterior direction when the predicted
gain is confident, keeps unexplained updates in an outlier cache, and stops when the optimistic predicted
gain of another update falls below its sample cost. The principle behind it, and where it goes next, is on
[the farm page](../ideas/index.md#where-next-tracking-with-improvement-equivalence).

## 🦎 Measurable observations and a named gate (`physics.py`, `metrics.py`)

Gecko changed the environment, not the agent: `objective="sqrt_iswap"` rewards the average gate fidelity to
√iSWAP after free virtual-Z corrections ([Gate metrics](../system/metrics.md#named-gate-fidelity-with-free-z-corrections)),
and `obs_mode="measured"` shows readout populations and Pauli expectations instead of state amplitudes
([Environment](environment.md#measured-observations-obs_modemeasured)).

## 🦄 A belief over the drift (`hippogriff.py`)

Three beliefs over the device's drift *z*, all pure functions, so the whole deployment episode is one
`scan`: the linearised Kalman update (Joseph form), the same with a consistency check (inflate the
covariance when the normalised innovation is too large), and exact Bayes on a grid; plus the expected
information gain that drives probing:

```python
--8<-- "rlquantopt/jx/hippogriff.py:belief"
```

??? example "One deployment episode: belief update, trust region, sigma-point safety, probing"

    ```python
    --8<-- "rlquantopt/jx/hippogriff.py:episode"
    ```

## 🦩 Data-lean control (`env.py`, `metrics.py`, `refine.py`)

The environment options (finite shots, clipping with a penalty, calibration context, the carrier action)
are in [Environment](environment.md#finite-shots-calibration-context-clipping-idea-i09) and on
[The calibration policy (new MDP)](../system/carrier-mdp.md). Two pieces work from measured data only.

**The fidelity from the 30 measurement settings**, with no Hamiltonian: magnitudes from the readout
populations, phases from the Pauli expectations:

```python
--8<-- "rlquantopt/jx/metrics.py:fidelity_estimate"
```

**Model-free refinement.** SPSA [[spall1992]](../bibliography.md#spall1992) estimates a gradient in all
coordinates of a smooth pulse correction from two noisy fidelity estimates per iteration; every shot is
counted. CMA-ES [[hansen2016]](../bibliography.md#hansen2016) is the other optimiser in `refine.py`.

```python
--8<-- "rlquantopt/jx/refine.py:spsa"
```

## 🐺 A realistic device (`device.py`, `env_device.py`)

`device.py` is the tunable-coupler device of Sung et al. [[sung2021]](../bibliography.md#sung2021) without
the rotating-wave approximation. Without it, excitation number is no longer conserved, but parity is, so the
Hamiltonian splits into an even block (7 states) and an odd one (10 states), propagated exactly per sample as
in `physics.py`. The coupler frequency follows an asymmetric-SQUID flux curve, the couplings scale with it,
and the computational states are the dressed idle eigenstates. `env_device.py` puts the carrier knobs on the
coupler flux, holds each value for one AWG sample, low-pass filters it, draws physical drift and builds the
spectroscopy calibration; `env.py` hands over to it when `physics_model == "device"`. The model is described,
with its sources, on the [jackal](../ideas/jackal.md) page.
