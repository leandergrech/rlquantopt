# Agents and training: `rlquantopt/jx/agents/`, `train.py`, `sb3_import.py`

The agents re-implement what the paper used from Stable-Baselines3 (SB3) and sb3-contrib, in
pure JAX, so that rollout and update run as one compiled function next to the environment.

## Networks

`ActorCritic` in `agents/common.py` mirrors SB3's `ActorCriticPolicy` as used in the paper:

- separate policy and value MLPs, [128, 128];
- **ReLU** activations for TRPO (what the paper run used; SB3's default, and the v1 script since
  June 2025, is tanh);
- orthogonal initialisation: gain √2 for hidden layers, 0.01 for the policy head, 1 for the value head;
- a state-independent Gaussian log-std, initialised at 0;
- actions are sampled unclipped (for the log-probabilities) and clipped to [−1, 1] only in the environment.

## Collecting a rollout

```python
--8<-- "rlquantopt/jx/agents/common.py:collect"
```

The line to understand is the truncation bootstrap:
`reward + gamma * V(final_obs) * (trunc & ~term)`. When an episode is cut short by the amplitude
bound (truncation), SB3 treats the lost future as \(\gamma V(s_{\text{final}})\) instead of 0, and
we do the same. Normal episode ends (termination at 50 ns) get no bootstrap.

## Advantages

```python
--8<-- "rlquantopt/jx/agents/common.py:gae"
```

A reverse `scan` over time. `done` cuts the recursion at episode boundaries, which matters because
auto-reset puts the next episode's first observation in the same array.

## TRPO: the actor step

This follows `sb3_contrib.TRPO.train` step by step: the natural-gradient direction by conjugate
gradient on the KL Hessian (15 iterations, damping 0.1), a maximal step that makes the KL equal
`2 * target_kl` along it, and a backtracking line search that must keep KL < 0.01 **and** improve
the surrogate.

```python
--8<-- "rlquantopt/jx/agents/trpo.py:actor_step"
```

- `ravel_pytree` flattens the actor parameters into one vector, so the CG and the line search are
  plain linear algebra.
- `fvp` is a Hessian-vector product of the KL: the gradient of \(\nabla \mathrm{KL}\cdot v\). No
  Hessian is ever formed.
- The line search evaluates all 10 candidate step sizes at once with `vmap`, then keeps the first
  acceptable one; SB3 evaluates them one by one, with the same result.

## TRPO: one full update

```python
--8<-- "rlquantopt/jx/agents/trpo.py:trpo_update"
```

After the actor step the critic gets 10 epochs of Adam on the value loss (minibatches of 128), with
the paper's harmonic learning-rate decay \(\eta = 3\cdot10^{-4} / (1 + 10^{0.4}\,p)\), where \(p\)
is the fraction of training done. With the paper's 4 environments × 2048 steps, one update is
8192 environment steps.

## PPO

`agents/ppo.py` is a standard clipped-surrogate PPO (clip 0.2, 10 epochs, 32 minibatches, gradient
clipping 0.5) with the same networks and harmonic decay. It is not used in the paper; it is the usual
choice when running many parallel environments, and a candidate for the seed study.

## The training loop

```python
--8<-- "rlquantopt/jx/train.py:train_loop"
```

Every `--eval-every` steps the loop runs one deterministic episode (mean actions), and saves:

| File in `runs/<name>/` | Content |
| --- | --- |
| `config.json` | All arguments, the environment and algorithm configs, backend, precision |
| `progress.csv` | One row per update: losses, KL, line-search success, rollout statistics, eval summary |
| `evals.npz` | Every evaluation episode: per-step reward, J_T, C, U, the pulse, alive mask |
| `params_<step>.pkl` | Parameters at each evaluation |

`evals.npz` is what `scripts/replicate_training_figs.py` turns into the paper's Figs. 5-8.

## Loading v1 agents

```python
--8<-- "rlquantopt/jx/sb3_import.py:load_sb3_policy"
```

SB3 stores torch `Linear` weights as (out, in); Flax `Dense` kernels are (in, out), hence the
transpose in `_dense`. The activation is read from the checkpoint's `policy_kwargs`: the paper
policy says ReLU. Loading it with tanh gives a completely different first action (+0.81 instead of
−0.17), which is how the activation mismatch was found.

```python
from rlquantopt.jx.env import EnvConfig
from rlquantopt.jx.sb3_import import load_sb3_policy
from rlquantopt.jx.agents.common import evaluate
model, params = load_sb3_policy("rlquantopt/rl_agents/ZCQPEE_pl-1000_T-50ns_delta_mode-TRPO/05-12-24_201634/rl_model_12566528_steps.zip")
episode = evaluate(model, params, EnvConfig())     # regenerates the paper's RL pulse
```
