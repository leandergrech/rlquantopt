# RLQuantOpt v2 (JAX)

!!! warning "Work in progress: open-source active research"
    This is ongoing research by Leander Grech, shared openly as it happens. Code, results and
    conclusions may change; the settled results are those of the
    [published paper](https://doi.org/10.1088/2058-9565/ae2c16). Interested in working together?
    See [Collaborate](contact.md).

This site documents the JAX re-implementation of RLQuantOpt (`rlquantopt/jx`), which started
on 23 September 2026. It is written for the two of us: you (Leander) steering the research, and
Claude writing most of the code. Every page says what the code does, shows the parts you need
to understand, and flags the decisions that are yours to make.

The v1 code (QuTiP + Stable-Baselines3) and the paper,
[Grech et al., *Quantum Sci. Technol.* 11, 015030 (2026)](https://doi.org/10.1088/2058-9565/ae2c16),
are the starting point and stay untouched in the repository as the reference.

## Where we are

| Version | What it added | Status |
| --- | --- | --- |
| 2.0.0 | Plan and package skeleton | done |
| 2.1.0 | Exact sector propagation and closed-form gate metrics | done |
| 2.2.0 | Functional, jit/vmap environment equivalent to v1 | done |
| 2.3.0 | Robustness maps replicated (paper Figs. 10-12) | done |
| 2.4.0 | PPO and TRPO in JAX, import of v1 SB3 checkpoints | done |
| 2.5.0 | Policy-level generalisation replicated (Figs. 13-17) | done |
| 2.6.0 | TRPO training replicated from scratch (Figs. 5-8), 1 seed | done |
| 2.7.0 | GRAPE baseline and the quantum speed limit (Fig. 3) | done |
| 2.8.0 | Replication report | done |
| 2.9.0 | RL → GRAPE refinement and robust (ensemble) GRAPE | done |
| 2.10.0 | This documentation site | done |

The full list with commit messages is in the [changelog](changelog.md).

## Headline results so far

- **Same physics, ~100× faster.** The JAX environment agrees with v1 step by step and with an
  exact 27-level propagation to 1e-13. It runs ~40k env steps/s on a laptop CPU in float64;
  v1 ran ~410/s. See [Physics](code/physics.md).
- **The paper's own agents behave identically in JAX.** The imported paper policy regenerates
  the paper's RL pulse (best J_T 9.49e-5 at 17.25 ns), and both of the paper's policies match
  the v1 generalisation sweeps at every point of a 101 × 101 grid. See
  [Replication](replication.md).
- **Training from scratch reproduces the dynamics but not the final quality, yet.** One seed
  reaches J_T ≈ 1e-3 against the paper's 1e-4; more seeds are needed.
- **GRAPE on top of RL is cheap and large.** Seeded with the RL pulse, GRAPE lowers J_T by about
  an order of magnitude in seconds. Robust GRAPE is the fair baseline for the paper's robustness
  claim. See [RL vs GRAPE vs robust GRAPE](experiments/rl-grape-robust.md).
- **Eight places where the paper text and the code or data differ**, including Table 1 and how
  domain randomisation was applied. See [Replication, section 7](replication.md#7-where-the-paper-and-the-code-differ).

## How to read this site

1. [Working together](working-together.md): how we split the work, and what to check on each commit.
2. [Getting started](getting-started.md): set up the environment, run the tests and one experiment.
3. [Code walkthrough](architecture.md): the modules in the order data flows through them, with the
   code you should know by heart.
4. [Experiments](replication.md): what we ran, what came out, and what it means.
5. [Roadmap](v2_jax_plan.md): what comes next and why.
