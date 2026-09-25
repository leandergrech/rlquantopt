# 🐸 i01 axolotl: rate each sample, emit actions one at a time

**Question.** Not every sample in a PPO batch helps the update. Can a small model learn which samples
help, and can PPO use that to learn faster? And does an autoregressive policy, one amplitude delta at a
time, suit a pulse built from deltas?

**Answer.** No on both counts, in this form. The judge never predicted anything (correlation with its
target about 0.01), and the autoregressive policy made updates about five times as aggressive as PPO's.
Every variant ended worse than plain PPO.

## Design

**Autoregressive policy** (`agents/autoregressive.py`). A trunk embeds the observation once; a small head
maps `[embedding, previous delta, running amplitude]` to the mean of one delta. The clipped delta is fed
back, three times per env step. The first call is seeded from the observation (last amplitude and last
delta), so the head is the same function at every sample of the pulse. One learned σ shared by the three
deltas. During the update the stored actions are fed back (teacher forcing), so the three head calls run
in parallel.

**Sample judge** (`agents/ppo_judge.py`). Per update, for 512 random probe samples:

\[
y_i = \cos\!\big(\nabla_\theta \ell_i,\; \nabla_\theta \tfrac{1}{|R|}\textstyle\sum_{j \in R} \ell_j\big),
\qquad \ell_i = -\,r_i(\theta)\,\hat A_i ,
\]

the agreement of sample *i*'s PPO gradient with the gradient of the rest of the batch *R*, at the rollout
parameters. An MLP of (state, action, reward) outputs a mean μ and a σ of *y*, trained with the Gaussian
NLL on the last 8 updates' probes. Before the epochs it scores every sample; only confident verdicts
count, \(m_i = \operatorname{sign}\mu_i\) if \(|\mu_i|/\sigma_i > \tau\), else 0, and the policy loss becomes

\[
L_\pi = \frac{\operatorname{mean}(l_i) + \beta\,\operatorname{mean}(m_i l_i)}{1 + \beta\,\operatorname{mean}(m_i)},
\qquad \beta = 0.5,
\]

i.e. weights \(1 \pm \beta\) for confident samples, renormalised so the step size is unchanged.

## Experiments

Gate env, the baseline's settings (20M steps, seed 123, ReLU), four variants:

| Run | Policy | Judge | Best \(J_T\) | Median \(J_T\), last 30 evals | Reaches PE, last 30 evals | KL / clip fraction |
| --- | --- | --- | --- | --- | --- | --- |
| `ar_only` | autoregressive | off | 3.0e-3 | 1.3e-2 | 3 % | 0.60 / 0.67 |
| `judge_z1` | autoregressive | τ = 1 | 1.4e-1 | 1.5e-1 | 0 % | 0.45 / 0.63 |
| `judge_z025` | autoregressive | τ = 0.25 | 4.3e-3 | 1.2e-2 | 47 % | 0.63 / 0.66 |
| `judge_z025_noar` | MLP (as PPO) | τ = 0.25 | 2.9e-3 | 3.5e-3 | 100 % | 0.10 / 0.50 |
| PPO, 4 seeds | MLP | – | 1.1e-4 – 9.8e-4 | 2.6e-4 – 3.6e-3 | 100 % | 0.07-0.13 / 0.41-0.53 |

The judge's correlation with its target, on samples it had not seen: 0.011-0.022 in the first quarter of
training, about 0 afterwards. Its σ stayed calibrated (standardised residuals with std about 1), so it
knew it did not know; at τ = 1 it almost never fired (guide loss about 0.01 % of the PPO loss).

## Outcome

- **The judge could not predict its target, by construction.** A sample's PPO gradient is its advantage
  times the gradient of its log-probability, so the sign of \(y_i\) flips with the sign of the advantage.
  The judge saw the reward, which says little about the advantage once training is under way.
- **The autoregressive policy destabilised PPO.** One head produces all three deltas, so one weight
  change shifts all three together; with σ around 0.006 that is a large change in policy.
- `judge_z1` is `ar_only` plus an idle judge, yet one reached 3e-3 and the other stalled at 0.14: the
  autoregressive policy is also highly seed- and launch-sensitive.

**Records:** `results/i01_axolotl/summary.json` (also holds the baseline numbers).
**Rerun:** `python -m rlquantopt.jx.train --idea axolotl --algo ppo_judge --activation relu --judge-z 0.25 [--no-ar | --judge-coef 0]`.
