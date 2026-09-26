# Agents and training: `rlquantopt/jx/agents/`, `train.py`, `sb3_import.py`

The agents re-implement what the paper used from Stable-Baselines3 (SB3) and sb3-contrib, in
pure JAX, so that rollout and update run as one compiled function next to the environment. TRPO is
the paper's algorithm and is kept for the replication; **PPO has been the workhorse since v2.12**
(same best gate, about twice as fast, [training results](../results/training.md)), and every idea on
the [idea farm](../ideas/index.md) starts from it. The research agents built on top of PPO are walked
through in [Research agents](idea-agents.md).

| Algorithm (`--algo`) | File | Used for |
| --- | --- | --- |
| `trpo` | `agents/trpo.py` | The paper replication ([results](../results/replication.md)) |
| `ppo` | `agents/ppo.py` | Everything since v2.12: seeds, drift, large coupler drift, gecko, ibis, jackal |
| `ppo_judge`, `chameleon`, `dragonfly`, `ppo_plus` | `agents/ppo_judge.py`, `chameleon.py`, `dragonfly.py`, `ppo_plus.py` | Ideas i01-i05 ([Research agents](idea-agents.md)) |
| `ppo_plus` on toy environments, `fox.FoxAdapter` | `toybench.py`, `foxbench.py`, `hippogriff.py` | Ideas i05, i06, i08 (own entry points) |

## Networks

`ActorCritic` in `agents/common.py` mirrors SB3's `ActorCriticPolicy` as used in the paper:

- separate policy and value MLPs, [128, 128] by default (`--hidden`; the calibration policies of ibis
  and jackal use [64, 64]);
- activation: **ReLU** for TRPO (what the paper run used; SB3's default, and the v1 script since
  June 2025, is tanh), tanh for PPO by default; `--activation` also offers leaky ReLU and **GELU**
  (ibis and jackal);
- orthogonal initialisation: gain √2 for hidden layers, 0.01 for the policy head, 1 for the value head;
- a state-independent Gaussian log-std, initialised at 0;
- actions are sampled unclipped (for the log-probabilities) and clipped to [−1, 1] only in the environment.

Rollouts, PPO and evaluation talk to the policy only through `sample`, `mode`, `logp`, `entropy` and
`value`, so another policy class can plug in unchanged: axolotl's autoregressive policy does
(`agents/autoregressive.py`).

## Collecting a rollout

```python
--8<-- "rlquantopt/jx/agents/common.py:collect"
```

The line to understand is the truncation bootstrap:
`reward + gamma * V(final_obs) * (trunc & ~term)`. When an episode is cut short by the amplitude
bound (truncation), SB3 treats the lost future as \(\gamma V(s_{\text{final}})\) instead of 0, and
we do the same. Normal episode ends (termination at 50 ns) get no bootstrap.

!!! warning "The truncation trap"
    This convention is harmless when training from scratch, but after a drift an over-optimistic critic
    makes overshooting pay, and fine-tuning collapses into it (found in [fox](../ideas/fox.md)). With
    `--oob-mode clip` (ibis on) the episode never truncates, so the question does not arise.

## Advantages

```python
--8<-- "rlquantopt/jx/agents/common.py:gae"
```

Generalised advantage estimation [[schulman2015gae]](../bibliography.md#schulman2015gae), as a reverse `scan`
over time. `done` cuts the recursion at episode boundaries, which matters because
auto-reset puts the next episode's first observation in the same array.

## PPO: the loss

`agents/ppo.py` is SB3's PPO with the clipped surrogate [[schulman2017]](../bibliography.md#schulman2017):

```python
--8<-- "rlquantopt/jx/agents/ppo.py:ppo_loss"
```

- **The ratio** \(r = \pi_\theta(a\mid s)/\pi_{\text{old}}(a\mid s)\) is formed from the log-probabilities
  stored at collection time, so the rollout policy is never re-evaluated.
- **Advantages are normalised per minibatch**, as SB3 does, before the clipped surrogate
  \(-\min(r\hat A,\ \operatorname{clip}(r, 1\pm\epsilon)\hat A)\) with \(\epsilon = 0.2\).
- **The value loss** is a plain squared error to the λ-return (SB3's default, no value clipping), weighted
  by `vf_coef = 0.5`. Actor and critic are separate networks, so the weight only matters through the
  shared optimiser's gradient clipping.
- **Entropy** of a Gaussian with a state-independent log-std depends only on the log-std; its coefficient
  is 0, as in the paper's setup.
- **Diagnostics**: `approx_kl` is the low-variance estimator \(\mathbb E[(r - 1) - \log r]\), `clip_frac`
  the share of samples outside the clip range. On the gate environment they settle near 0.1 and 0.5, far
  above the usual 0.01-0.02 and 0.1-0.2; braking them stalls learning ([badger](../ideas/badger.md)).

## PPO: one update

```python
--8<-- "rlquantopt/jx/agents/ppo.py:ppo_update"
```

Everything in one `jax.jit`:

1. **Collect** `n_steps` × `n_envs` transitions (64 × 128 = 8192 by default) with `collect`, then
   bootstrap the last value and compute GAE (γ = 0.99, λ = 0.95).
2. **Flatten** the time and environment axes into one batch.
3. **Epochs as a `scan`**: each of the 10 epochs draws a fresh permutation, splits the batch into 32
   minibatches of 256 and runs a `scan` of gradient steps over them.
4. **The optimiser** clips the global gradient norm at 0.5 and runs Adam (ε = 1e-5) with the paper's
   harmonic decay, applied per gradient step:
   \(\eta_i = 3\cdot10^{-4} / (1 + 10^{0.4}\, i / n_{\text{grad steps}})\).

Because the rollout, GAE and all 320 gradient steps are a single XLA program, one PPO update on the
gate environment takes about 1.5 s on the laptop CPU; 20M steps (2441 updates) take about an hour.

| | PPO (`PPOConfig`) | TRPO (`TRPOConfig`) |
| --- | --- | --- |
| Rollout per update | 64 envs × 128 steps | 4 envs × 2048 steps (the paper) |
| Actor step | 10 epochs × 32 minibatches of the clipped surrogate | One natural-gradient step with a KL limit of 0.01 |
| Critic | Same loss, same optimiser as the actor | 10 epochs of Adam, minibatches of 128 |
| Default activation | tanh (SB3); ReLU in all our PPO runs up to gecko, GELU in ibis and jackal | ReLU (the paper run) |
| Learning rate | 3e-4, harmonic decay | 3e-4, harmonic decay (critic) |

## TRPO: the actor step

This follows `sb3_contrib.TRPO.train` step by step [[schulman2015]](../bibliography.md#schulman2015): the
natural-gradient direction by conjugate gradient on the KL Hessian (15 iterations, damping 0.1), a maximal
step that makes the KL equal `2 * target_kl` along it, and a backtracking line search that must keep
KL < 0.01 **and** improve the surrogate.

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

## The training loop

```python
--8<-- "rlquantopt/jx/train.py:train_loop"
```

Every algorithm exposes the same `init(key, env_cfg, algo_cfg)` and `make_update(model, env_cfg, algo_cfg)`,
so the loop does not know which one it runs. Every `--eval-every` steps it runs one deterministic episode
(mean actions) on the nominal device, and saves:

| File in `runs/<name>/` (or `runs/iNN_<idea>/<name>/` with `--idea`) | Content |
| --- | --- |
| `config.json` | All arguments, the environment and algorithm configs, backend, precision |
| `progress.csv` | One row per update: losses, KL, clip fraction, rollout statistics, eval summary |
| `evals.npz` | Every evaluation episode: per-step reward, J_T (or 1 − F), C, U, the pulse, alive mask |
| `params_<step>.pkl` | Parameters at each evaluation |

`evals.npz` is what `scripts/replicate_training_figs.py` turns into the paper's Figs. 5-8. The evaluation
here is on the nominal device only; the drifted-device evaluations are separate scripts
(`coupler_drift_experiment.py`, `ibis_eval.py`, `jackal_eval.py`, [Scripts and CLI](scripts.md)).

The run directory name is `<algo>_<timestamp>_s<seed>_<tag>`, and `os.makedirs` fails rather than
overwrite an existing run. With `--idea <animal>` it lands in the idea's own folder
(`rlquantopt/jx/ideas.py`).

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
