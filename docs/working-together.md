# Working together

The code moves fast, so the risk is that it becomes Claude's code rather than ours. This page is
the contract that prevents that: how work is split, how it is versioned, and what you check.

## Who does what

| You decide | Claude does | We do together |
| --- | --- | --- |
| Research questions and what counts as a result | Implementation, tests, running experiments | Interpreting results |
| Which physics goes in the model (levels, noise, targets) | Keeping v1 equivalence tests green | Choosing baselines and metrics |
| What is published or pushed, and when | Writing the docs and the report | Deciding what a discrepancy means |
| Priorities on the [roadmap](v2_jax_plan.md) | Flagging open decisions (below) | |

When Claude hits a choice that changes the science (a reward definition, a baseline, how a
number is reported), it asks instead of picking silently. These are listed on each page as
**Decision** boxes.

## Versioning

- Every piece of work is one or more commits on `main`, tagged `v2.<minor>.<patch>`.
- The version lives in three places, bumped together: `rlquantopt/__init__.py`, `setup.py`, `setup.cfg`.
- A minor version is a milestone (a new capability or a replicated figure); a patch is a fix.
- Tags are annotated (`git tag -a v2.x.y -m "..."`). They are not GitHub Releases; v1.0.0 stays the
  latest release until we decide a v2 result is ready to cite.
- Commit messages state what changed, what was verified and any finding, so `git log` doubles as a
  lab notebook.

## What to check on a commit

A five-minute review of any commit:

1. Read the commit message: does the claim match what you expected?
2. Run the tests: `pytest tests/` (about 30 s on the CPU; see [Getting started](getting-started.md)).
3. For a new experiment, open its figure under `docs/figures/` and the numbers in the JSON/NPZ next
   to it. Every figure has a script in `scripts/` that regenerates it.
4. If the physics changed, check the equivalence tests still compare against v1 and did not get
   looser tolerances without a reason in the comment.

!!! tip "Where to start reading the code"
    Read [Physics](code/physics.md) and [Environment](code/environment.md) first. Everything else
    (agents, GRAPE, sweeps) is built from `physics.propagate` and `env.step`.

## Ground rules for the code

- **v1 is the reference.** `rlquantopt/rl_envs` and `rlquantopt/rl_agents` are not edited. New
  behaviour goes in `rlquantopt/jx`, and differences from v1 are deliberate and documented.
- **float64 by default.** Training in float32 is wrong at the 1e-4 level (see
  [Physics](code/physics.md#precision)).
- **Pure functions.** Environment and agent code are pure JAX functions of explicit state, so they
  can be `jit`-ted, `vmap`-ped and differentiated. No hidden state in objects.
- **Every result has a script.** Nothing in `docs/` is produced by hand.

## Open decisions for you

!!! question "Decision: how many seeds, and which algorithm, for the training comparison?"
    One TRPO seed reached J_T ≈ 1e-3 against the paper's 1e-4. Five seeds per setting cost about
    10 CPU-hours for TRPO with the paper's settings. PPO with many parallel environments is faster
    per sample. Do we replicate TRPO faithfully first, or move on to PPO?

!!! question "Decision: the robustness metric for comparing RL and GRAPE"
    The [RL vs GRAPE experiment](experiments/rl-grape-robust.md) reports the area with
    J_T ≤ 1e-3 on a ±50 MHz map and the mean log10 J_T within 10 and 25 MHz. Which one should the
    paper use, and over what detuning range is drift realistic for the target hardware?

!!! question "Decision: named gate or perfect entangler?"
    The reward targets any perfect entangler. A named gate (√iSWAP or CZ) with average gate
    fidelity makes results comparable with the literature, at the cost of changing the reward.
