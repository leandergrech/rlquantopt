# 🦔 i05 echidna: back to PPO, one ingredient at a time

**Question.** Which of dragonfly's ingredients helps on its own? Each is tested on small, cheap
environments first, and only a winner goes onto the gate environment.

**Answer.** Each ingredient helps on one kind of task and hurts on another. Self-imitation won clearly on
the gate-like toy but held training back on the gate environment itself. None beats plain PPO there.

## Design (`agents/ppo_plus.py`, `toy_envs.py`, `toybench.py`)

PPO plus three switches, for any environment with the toy interface:

- **OU exploration noise** (`ou_rho`): \(a_t = \mu(s_t) + \sigma \epsilon_t\),
  \(\epsilon_t = \rho\,\epsilon_{t-1} + \sqrt{1-\rho^2}\,\eta_t\), restarted every episode. The policy is
  conditioned on the previous noise, \(\pi(a \mid s, \epsilon_{t-1}) = \mathcal N(\mu + \sigma\rho\,\epsilon_{t-1},\ \sigma^2(1-\rho^2))\),
  with \(\epsilon_{t-1}\) stored in the rollout, so the PPO ratio stays an exact likelihood ratio. (Tested:
  the conditionals multiply to the exact joint density of the correlated noise.)
- **Pessimistic veto** (`veto`): a 4-head Q ensemble regresses the λ-return; of 8 candidates from π the
  first is taken unless its lower bound (mean − std) is more than half a return-spread below the best's.
- **Golden-cache self-imitation** (`sil_coef`): a canonical archive of the best transitions by return and
  by reward; the policy imitates cached actions whose return beats the critic's value (Oh et al. 2018).

The toys: **Pendulum** (dense reward, a sanity check), **MountainCar** (sparse reward behind a hill, the
classic exploration trap) and **Tracker** (a miniature gate problem: delta actions on an amplitude bounded
to ±1, overshooting ends the episode with a penalty, \(-\log_{10}\) tracking-error reward).

## Toy grid (5 seeds each)

| Variant | MountainCar | Tracker: final · AUC | Pendulum |
| --- | --- | --- | --- |
| PPO | 75 ± 38 (4/5 solved) | 338 ± 76 · 304 ± 10 | −165 ± 40 |
| OU noise | **94 ± 0.1 (5/5)** | 252 ± 42 · 225 ± 9 | −167 ± 39 |
| veto | 94 ± 0.3 (5/5) | 365 ± 38 · 301 ± 19 | −164 ± 41 |
| self-imitation | 37 ± 46 (2/5) | **480 ± 15 · 389 ± 3** | −243 ± 63 |
| all three | 94 ± 0.2 (5/5) | 361 ± 44 · 238 ± 49 | −181 ± 44 |

AUC is the mean evaluation return over training. The veto costs about 2× wall time and is neutral.

!!! warning "Caveat found later"
    These Tracker runs used a reward that turns negative once the tracking error exceeds 1 %, which can
    make ending an episode early optimal (found in [fox](fox.md)). Self-imitation may have looked good
    partly by steering away from that trap. The conclusion below rests on the gate-env runs, not on this.

## Self-imitation on the gate environment (20M steps, the baseline's settings)

| Seed | Plain PPO (same code): best · median last 30 · to \(J_T<0.1\) | Self-imitation: best · median · to \(J_T<0.1\) |
| --- | --- | --- |
| 1 | 7.5e-4 · 1.8e-3 · 0.13M steps | 1.3e-2 · 3.5e-2 · 3.6M |
| 2 | 2.8e-4 · 5.7e-4 · 1.05M | 0.14 · 0.17 · never |
| 3 | 5.5e-4 · 1.7e-3 · 1.1M | 7.2e-4 · 2.3e-3 · 3.0M |

The plain control matches the existing baseline seeds, so the new code path is sound.

## Outcome

- **Self-imitation locks in early.** While the critic still undervalues everything, every cached entry
  counts as "better than expected", so the policy imitates the early best transitions of a still-bad
  policy. The start is 3× slower and one seed never recovers. By the end, no cached entry beats the critic
  any more and self-imitation has switched itself off. It is the same lock-in as on MountainCar.
- **Exploration ingredients do not transfer** to an environment that is not exploration-limited;
  correlated noise hurts precise tracking.

**Records:** `results/i05_echidna/` (toy-grid summary and curves, gate-run numbers).
**Rerun:** `python -m rlquantopt.jx.toybench --env tracker --variant sil`;
`python -m rlquantopt.jx.train --idea echidna --algo ppo_plus --activation relu --sil-coef 0.1 --seed 1`.
