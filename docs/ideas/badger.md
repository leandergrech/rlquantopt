# 🦡 i02 badger: a better-informed judge, and a brake

<span class="stamp stamp--concluded">🏁 concluded</span> *2026-09-25, v2.17.0 · negative: the KL brake starved learning · raw runs archived in `runs/_attic/i02_badger/`*

**Question.** Does axolotl's judge work once it can tell good samples from bad (it sees the return), is
consulted only when fresh, and does PPO become calmer, lower KL and clip fraction, with a KL brake?

**Answer.** The brake worked too well: PPO took only 2-7 % of its minibatch steps late in training, and
even plain PPO with the brake stalled at \(J_T \approx 0.2\). The judge still predicted nothing.

## Design

The same code as [axolotl](axolotl.md) (`agents/ppo_judge.py`), with four switches, all off by default so
the i01 runs reproduce:

1. `judge_return`: the judge's input also holds the λ-return (the critic's target), scaled by 0.01.
2. `judge_every = 5`: the probe targets are measured and the judge trained only every fifth policy update;
   it trains on its last 8 rounds (40 policy updates).
3. `judge_half_life = 3`: the guide weight decays with the judge's age since it was trained,
   \(\beta_\text{eff} = \beta \cdot 0.5^{\text{age}/3}\) (0.40 down to 0.16 over ages 1-5).
4. `target_kl`: SB3's early stop. Once a minibatch starts with approximate KL above
   \(1.5 \times\) `target_kl`, the rest of the update is skipped.

## Experiments

Gate env, 20M steps, seed 123, `target_kl = 0.02` unless stated:

| Run | Policy | Judge | Best \(J_T\) | Median \(J_T\), last 30 | Steps taken, last quarter | Clip fraction |
| --- | --- | --- | --- | --- | --- | --- |
| `ctrl_ppo_kl` | MLP | off | 0.20 | 0.21 | 7 % | 0.15 |
| `ar_kl` | autoregressive | off | 0.24 | 0.25 | 6 % | 0.13 |
| `badger` | autoregressive | on | 0.17 | 0.17 | 2 % | 0.14 |
| `badger_noar` | MLP | on | 4.9e-3 | 7.6e-3 | 3 % | 0.16 |
| `badger_kl001` | autoregressive | on, `target_kl = 0.01` | 0.23 | 0.25 | 6 % | 0.07 |

The judge's correlation with its target stayed at 0.005 or below, even with the return as input.

## Outcome

- **The brake starved learning.** The clip fraction did fall (0.07-0.16, against 0.41-0.53 for PPO), but
  late in training only 2-7 % of the planned minibatch steps were taken. The control, plain PPO with the
  brake, shows the brake alone is enough to stall. This environment needs PPO's large steps.
- `badger_noar` did far better than the control under the same brake. With a judge that predicts nothing,
  and one seed, this is most likely launch noise (runs are not bit-reproducible across launches on this
  CPU backend) rather than an effect.
- **The judge's failure is not fixed by the return.** To use it the judge would have to learn the value
  function as well, to form the advantage.

**Records:** `results/i02_badger/summary.json`.
**Rerun:** `python -m rlquantopt.jx.train --idea badger --algo ppo_judge --activation relu --target-kl 0.02 --judge-z 0.25 --judge-return --judge-every 5 --judge-half-life 3`.
