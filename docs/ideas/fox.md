# 🦊 i06 fox: is another update worth its samples?

**Question.** A trained policy meets a new, drifted device, where every sample costs lab time and no
simulator gradient is available. Can a model of *how updates improve the return* make adaptation cheaper,
and tell when to stop spending samples? (Named after optimal foraging: stay in a patch only while its
marginal gain beats the cost of staying.)

**Answer.** Not on this benchmark. The model's extra step gave +5.5 ± 3.7 return over PPO fine-tuning
(72 paired comparisons, fox ahead on 42): no demonstrated effect. The stopping rule beats spending the
whole budget, but black-box fine-tuning after drift is not worth its samples at all here: not adapting
scores best.

## Design (`agents/fox.py`, `foxbench.py`)

The **improvement equivalence principle**: a model need not predict the world, or even the return, only
which way the return improves. After every PPO fine-tuning update, fox records the step it took,
\(\Delta\theta_k\), and (one batch later) the return change \(\Delta J_k\). A Bayesian linear regression
over the last 24 updates plus an outlier cache,

\[
\frac{\Delta J_k}{\lVert\Delta\theta_k\rVert} \sim \mathcal N\!\Big(\big\langle g, \tfrac{\Delta\theta_k}{\lVert\Delta\theta_k\rVert}\big\rangle,\ s^2\Big),
\qquad g = D^\top\beta,\quad \beta \sim \mathcal N(0, \alpha^{-1} I),
\]

predicts the gain per unit step as \(\lVert g\rVert \cos(g, \text{step})\); *D* is an orthonormal basis of
the last 8 PPO steps in actor-parameter space; α and \(s^2\) are learned by evidence maximisation.

- **Exploit**: when the gain along *g* is confident (z > 1), fox adds half a PPO step along *g*.
- **Outlier cache**: updates whose \(\Delta J\) the model cannot explain (residual > 2σ) are kept past the
  window with double weight, and dropped once explained (residual < 1σ).
- **Stopping**: fox stops when the optimistic predicted gain of another update falls below its sample cost.

The benchmark: a policy pretrained on the nominal Tracker adapts to held-out devices with an unknown
action gain (0.6-1.4) and output offset (±0.15, invisible in the observation), for 60 updates. PPO
fine-tuning uses a 5-update critic warm-up and learning rate 1e-4.

## Results (6 pretraining seeds × 12 devices = 72 paired comparisons)

| Paired, fox − PPO fine-tuning | Result | Fox better on |
| --- | --- | --- |
| Return after 60 updates | +5.5 ± 3.7 | 42/72 |
| Mean return during adaptation | +3.1 ± 2.9 | 42/72 |
| Utility (return − 1 per update) at fox's stop, vs PPO's full budget | +36.7 ± 3.7 | 66/72 |
| Utility at fox's stop, vs not adapting | −44.5 ± 8.2 | 18/72 |
| Stopping after a minimum of 3 vs 10 updates | −0.9 ± 2.0 | no difference |

Return before adaptation 324; after 60 updates 303 (PPO) and 309 (fox). Neither reaches 90 % of the
nominal return. An earlier 24-comparison run gave +8.4 ± 6.0; the larger run halved the uncertainty and
the effect shrank.

## Lessons from building the benchmark

Three bugs were found before any result was trusted:

1. **A prior on the wrong scale**: the gain per unit step is about 90, the fixed prior assumed about 1 and
   shrank every estimate to zero. Fixed by learning the prior (evidence maximisation).
2. **Overshooting counted as a truncation**, as in the gate environment's v1 convention: PPO bootstraps
   \(\gamma V(\text{final})\) onto the penalty, and with a critic that overestimates after drift,
   overshooting paid, so fine-tuning collapsed into it. The gate environment has the same convention:
   harmless when training from scratch, a trap for fine-tuning after drift. The toy now makes it terminal.
3. **A reward that made dying early optimal** on badly drifted devices (it went negative); now kept
   positive, as in the gate environment.

The first 24-comparison run, affected by bugs 2 and 3, is marked invalid and excluded.

**Records:** `results/i06_fox/summary.json`.
**Rerun:** `python -m rlquantopt.jx.foxbench --seeds 1 --seed-start 0 --devices 12 --budget 60 --ft-lr 1e-4 --critic-warmup 5`,
then `--summarise <run dirs>`.
