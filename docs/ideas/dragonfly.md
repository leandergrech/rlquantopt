# 🦋 i04 dragonfly: chameleon, faster, more careful, remembering its best

<span class="stamp stamp--archived">📦 archived</span> *2026-09-25, v2.17.0 · smoke test only · its ingredients were tested one at a time in [echidna](echidna.md), its belief and caution returned in [hippogriff](hippogriff.md)*

**Question.** Chameleon, but fast enough to run, exploring by a guided random walk with momentum,
cautious where it is unsure ("worst case until proven otherwise"), and remembering the best it has seen.

**Answer.** About 5× faster, and it learned faster than chameleon early on, then degraded; two of its
parts turned out not to work as designed. Paused in favour of testing its ingredients one at a time
([echidna](echidna.md)); its belief-and-caution ideas returned in [hippogriff](hippogriff.md).

## Design (`agents/dragonfly.py`), what differs from chameleon

- **History as leaky integrators**: exponential traces of \([z_t, z_t - z_{t-1}, a_{t-1}, r_{t-1}]\) at
  decays 0.5, 0.9, 0.99 (momenta of the latent). A linear recurrence, trained with `associative_scan`,
  parallel over time, instead of backpropagation through a GRU.
- **Exploration is an Ornstein-Uhlenbeck walk** per latent dimension, every step:
  \(\xi_d \leftarrow \rho_d \xi_d + \sqrt{1-\rho_d^2}\,\sigma_d \eta\), \(\rho_d = e^{-1/\tau_d}\).
  Its spread is \(\sigma_d\) whatever \(\rho_d\) (constant entropy) while it carries
  \(-\tfrac12\log(1-\rho_d^2)\) nats of momentum from step to step; \(\tau_d\) is how many steps a
  heading persists.
- **A learned guide** sets \(\tau_d, \sigma_d\) and ε from the history, the measured entropy of each latent
  dimension, a pessimistic value, and *p*, the current performance relative to a golden cache.
- **Pessimism**: Q and V are ensembles (4 heads on bootstrap masks); the search scores candidates by
  alignment plus the ensemble's lower bound, so likely-bad candidates are vetoed.
- **Golden cache**: the best experiences by reward and by return, at most one per latent ball
  (canonical), with self-imitation of their actions.

## Smoke test (330k steps)

| | |
| --- | --- |
| Time per update | 10.7 s on a quiet machine (chameleon: 47 s, loaded) |
| Episodes ending early | 49 % → 1 % by 140k steps, then back to 12-22 % |
| Mean reward per step | −9.4 → +0.41 (best) → −1.8 at the end |
| Candidates vetoed by the lower bound | 15-40 % |
| Deterministic test episode, last evals | overshoots at once (return −20) |

## Outcome

- **The cache position *p* saturated at 0.** The cache holds only golden examples, so even its worst
  entry is excellent and every ordinary state falls below it; the guide never got a signal from *p*.
- **The guide never moved**: τ, σ, ε stayed at their initial midpoints (32.5, 0.5, 0.5), so the question
  of how τ relates to *p* or to the measured entropy was never answered.
- The veto was the part that visibly helped early on, which is why pessimism returned in
  [echidna](echidna.md) (tested alone) and [hippogriff](hippogriff.md) (as a safety check on a belief).

**Records:** `results/i04_dragonfly/` (smoke-test progress and summary).
