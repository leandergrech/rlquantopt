# Changelog

All v2 versions are commits on `main` with an annotated tag `v2.x.y`. The commit messages hold
the details; this page is the summary.

## 2.10.0: documentation site (2026-09-24)

This MkDocs site, deployed to GitHub Pages by `.github/workflows/docs.yml`. Code shown on the site
is included from the source through `--8<--` markers.

## 2.9.0: RL → GRAPE and robust GRAPE (2026-09-24)

`grape.optimise` accepts an ensemble of Hamiltonians (mean J_T over detunings). New
`scripts/rl_grape_robust.py` compares RL, RL → GRAPE, GRAPE, robust GRAPE and RL → robust GRAPE
at equal gate time and amplitude bound. See [the experiment](experiments/rl-grape-robust.md).

## 2.8.0: replication report (2026-09-24)

[docs/replication.md](replication.md), README section, updated roadmap.

## 2.7.0: GRAPE and the quantum speed limit (2026-09-24)

Differentiable GRAPE through the exact propagators; QSL scan for five amplitude limits (Fig. 3).
QSL between 10 and 12 ns at 1.5 GHz (paper: 10 ns).

## 2.6.0: TRPO training replication (2026-09-24)

20M steps with the paper's settings in 1 h 55 min on a laptop CPU. Same breakthrough at 2-4M
steps; best J_T 9.8e-4 against the paper's 9.5e-5 (one seed). Run artefacts in
`results/jax_trpo_paper_s123/`.

## 2.5.0: policy-level generalisation (2026-09-24)

The paper's static and domain-randomised policies, imported into JAX, match the v1 sweeps at all
101 × 101 points. Found that v1 `ZCQPEEWRD` drew its drift once per env, not per episode.

## 2.4.0: PPO, TRPO and SB3 import (2026-09-23)

Pure-JAX PPO and a TRPO that follows sb3-contrib; training CLI; import of v1 checkpoints. Found
that the paper policy used ReLU, not tanh.

## 2.3.0: robustness maps (2026-09-23)

Stored RL and Krotov pulses on the paper's ±50 MHz grid vs the paper's Julia data. Found the
17.25 ns stop time, the `clip_best` off-by-one and the Table 1 parameter pairing. The paper's
final pulses were added to the repository.

## 2.2.0: functional environment (2026-09-23)

`reset`/`step`/`step_autoreset` as pure functions, step-by-step equivalent to v1; ~40k env
steps/s on the CPU in float64.

## 2.1.0: physics and metrics (2026-09-23)

Exact propagation in the N=1 and N=2 excitation sectors; closed-form Weyl coordinates;
cross-checked against QuTiP and `weylchamber`.

## 2.0.0: plan (2026-09-23)

[Roadmap](v2_jax_plan.md), package skeleton, `requirements-jax.txt`.
