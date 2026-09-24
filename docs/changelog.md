# Changelog

All v2 versions are commits on `main` with an annotated tag `v2.x.y`. The commit messages hold
the details; this page is the summary.

## 2.12.0: PPO seeds, losses, drift-range robustness, RL vs GRAPE cost, redesign (2026-09-24)

Four PPO seeds and TRPO compared (best seed J_T 1.1e-4, matching the paper); loss curves for RL
and GRAPE; PPO's updates found too aggressive (KL ~0.1). Robustness re-run with the hardware drift
ranges: robust GRAPE over the recool range beats every other pulse. New experiment comparing the
cost per device of RL (with and without domain randomisation) and GRAPE. Serif redesign of the site,
hero home page, fixed figure captions.

## 2.11.0: docs by topic, landing page, bibliography, drift ranges (2026-09-24)

Docs reorganised into Guide / The system / The code / Results / Next steps; a landing page with the
idea, the v1 design and credits; a bibliography generated from DOIs with BibTeX, RIS, CSL-JSON and
text downloads; hardware drift ranges from the literature (`rlquantopt/jx/drift.py`); a design note
on named gates and richer models; one page of open decisions; PPO vs TRPO (seed 123).

## 2.10.1: work-in-progress notice (2026-09-24)

Site banner and a Collaborate page with a GitHub issue form.

## 2.10.0: documentation site (2026-09-24)

This MkDocs site, deployed to GitHub Pages by `.github/workflows/docs.yml`. Code shown on the site
is included from the source through `--8<--` markers.

## 2.9.0: RL → GRAPE and robust GRAPE (2026-09-24)

`grape.optimise` accepts an ensemble of Hamiltonians (mean J_T over detunings). New
`scripts/rl_grape_robust.py` compares RL, RL → GRAPE, GRAPE, robust GRAPE and RL → robust GRAPE
at equal gate time and amplitude bound. See [the experiment](results/robustness.md).

## 2.8.0: replication report (2026-09-24)

[docs/replication.md](results/replication.md), README section, updated roadmap.

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

[Roadmap](next/roadmap.md), package skeleton, `requirements-jax.txt`.
