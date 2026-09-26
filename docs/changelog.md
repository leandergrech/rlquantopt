# Changelog

All v2 versions are commits on `main` with an annotated tag `v2.x.y`. The commit messages hold
the details; this page is the summary.

## 2.19.0: ibis concluded, the calibration MDP, jackal started, the docs consolidated (2026-09-26)

**i09 ibis concluded.** The open-loop carrier policy fed only a frequency calibration (~1e4 shots per
device; K = 15, 64×64 GELU) is reproducible and robust. Figures: `ibis_eval_close*`, `ibis_stress_close.json`.

- **Seeds.** Over four seeds, the full-pulse 1 − F median on the 24 held-out coupler-drift devices is
  3.9e-3 / 4.4e-3 / 3.7e-3 / 4.6e-3, with every device below 1e-2. Each seed reached 1e-2 in 0.69-0.85M
  steps.
- **Network size.** A 32×32 network (1.4k actor parameters) gives 4.3e-3.
- **Decisions per pulse.** 33 decisions (K = 30) learn fastest to good quality (0.46M steps, 5.2e-3);
  16 decisions (K = 60) are too coarse (9.5e-3, 67 % below 1e-2).
- **Calibration stress** (errors ×1 / ×3 / ×10 the training errors, and a stale bias): 3.7-5.8e-3,
  81-100 % below 1e-2.
- **Drift the calibration cannot see** (couplings ±5.7 MHz, anharmonicities ±5 MHz): 3.8-6.0e-3,
  78-100 %.
- **Unseen drift in training.** Training with coupling drift gives the most robust policy: 94-100 % in
  every stress case.
- **New option:** `context_bias_mhz` (a stale calibration); `scripts/ibis_eval.py --stress`.


**Docs consolidated across both lines of work** (the idea farm, and gecko → ibis → jackal):

- **Status stamps.** Every idea carries a stamp on its page, in the farm table and next to its emoji
  in the navigation: 🟢 active (hippogriff, jackal), 🏁 concluded (axolotl, badger, echidna, fox,
  gecko, ibis), 📦 archived (chameleon, dragonfly).
- **The farm's lessons, stated generally.** Twelve lessons for sample-efficient RL on real, drifting
  systems. The problem formulation (action space, decision interval, what is observed) gave the largest
  gains. Identification beats re-training, amortise-then-refine works, and black-box refinement is weak.
  Plus why quantum control is a good test bed (an exact differentiable model, nonlinear dynamics,
  structured uncertainty), and the next direction: tracking with model-based and model-free parts, with
  improvement equivalence for search control.
- **Code pages.**
    - PPO's loss and update walked through like TRPO's (new snippet markers in `agents/ppo.py`).
    - A new page, Research agents, with the code of every animal. New markers in `agents/fox.py` and
      `refine.py`; comment lines only, no code change.
    - Architecture, Environment (carrier actions, the realistic device), Scripts and CLI (every flag and
      idea script) and Getting started (122 tests) are brought up to date.
- **Other pages.** Home page (cards, simplifications with their status, latest), roadmap, open decisions
  (decided items moved, the platform and decoherence questions added), named gates, and the hypothesis
  pages (first evidence on H3/T4 from ibis). Nine references added (PPO, TRPO, GAE, self-imitation,
  evidence maximisation, SPSA, CMA-ES, value equivalence, Dyna).

**The calibration policy as a new MDP.** New page, The system → The calibration policy (new MDP): the paper's
MDP against the carrier calibration MDP, a schematic of how a pulse is built, what a trained policy plays,
and the training and deployment loops.

**i10 jackal started**: a realistic tunable-coupler device (Sung et al. 2021 parameters, no RWA with parity
blocks, direct coupling, asymmetric-SQUID coupler, 1 ns AWG, flux-line filter, physical drift, spectroscopy
calibration; `device.py`, `env_device.py`, `--physics device`), a simplified variant for the sim-to-sim
transfer test, and `scripts/jackal_eval.py`. Runs in progress.

## 2.18.1: hippogriff's belief fixed (2026-09-26)

Idea i08: the linearised Kalman belief over the device's drift was 9-890× overconfident (it shrank around
a wrong value). Two remedies, compared on the same pretrained policies, 3 seeds × 12 toy devices, with
noise-free and noisy measurements: an exact grid belief (Bayes on 41 × 41 points, probing by the spread
of predictions) gives 575-627 of an oracle's 655 in the first episode with the exact model and 511-549
with a learned model (Kalman: 372-614 and 337-486), and noise barely hurts it; a consistency check
(covariance inflation on surprising innovations) lets the Kalman belief reach the oracle by episode 5
with the exact model but breaks with the learned one. Probing pays only where the belief is weak. The
hippogriff page has a guide to its runs; failed and duplicate raw runs are archived in `runs/_attic/`
(TensorBoard skips them, `idea_records.py` still reads them).

## 2.18.0: ibis, the carrier policy (2026-09-26)

New action mode `--action-mode carrier`: the pulse is u(t) = offset + A cos(2π f_d t + φ), the policy
nudges (A, φ, offset), and f_d is the nominal qubit detuning corrected by the measured calibration. With
it, a policy that sees only a frequency calibration measured once before the pulse (open loop, about
1e4 shots per device) reaches 1 − F = 3.9e-3 on the 24 large-coupler-drift devices (all below 1e-2),
reaching 1e-2 in 0.72M training steps: better than the best closed-loop policy with exact observations
(8.5e-3, 8.1M steps) and about 1000× cheaper in shots than a closed-loop policy at 1000 shots per
setting. A closed-loop carrier policy trained with 100 shots reaches 1.1e-2 at 2e5 shots per pulse.
With per-sample actions the open-loop policy could not learn (identity gate) and noise-trained
closed-loop policies plateaued at 0.13. Few long steps (K = 15) matter in carrier mode. Also: calibration
runs with oob penalty 5 and 1; records in results/i09_ibis; corrected actor-parameter counts on the ibis
page.

## 2.17.0: the idea farm, and ibis in progress (2026-09-25)

**Idea iterations.** Every new training idea gets a number and an animal codename, a registry entry
(`rlquantopt/jx/ideas.py`), its own run directory (`train.py --idea <animal>` writes to
`runs/iNN_<animal>/`) and a page in the hidden "Working ideas" section. Toy environments and benchmarks
(`toy_envs.py`, `toybench.py`, `foxbench.py`) test ingredients in isolation. Outcomes of i01-i06 and i08:
no idea beats plain PPO on the gate environment (axolotl, badger, echidna); the gate environment is not
exploration-limited and needs PPO's big steps; black-box fine-tuning after drift is not worth its samples
(fox, 72 paired comparisons); a drift-conditioned PPO policy with a Bayesian belief over the drift
recovers 614 of an oracle's 655 on toy devices in one episode against 327 for a robust policy, and
information-gain probing adds +21 to +110 (hippogriff), but its EKF belief is 9-890× overconfident.
Found on the way: treating an overshoot as truncation bootstraps γV onto the penalty, which collapses
fine-tuning after drift.

**i09 ibis (in progress): gecko made data-lean.** New environment options, defaults unchanged:
amplitude clipping with a penalty instead of termination (`--oob-mode clip`), finite-shot observations
(`--shots N`, 30 settings per step), a calibration context of measured qubit and coupler frequencies
(`--obs-mode context`, `measured+context`), longer steps with a scaled action (`--delta-scale`), network
size (`--hidden`) and GELU; a √iSWAP fidelity estimate from measured data only
(`metrics.fidelity_from_observables`) and model-free refinement (SPSA, CMA-ES; `refine.py`). So far:
policies trained on exact observations fail under shot noise (30× worse at 1e6 shots per pulse);
64×64 GELU matches 128×128 ReLU with a third of the parameters; black-box refinement barely improves a
policy pulse (5.8e-3 to about 4e-3 after 1500 noise-free estimates), so the policy must be good enough
on its own. Noise-aware and calibration-conditioned runs are in progress.

## 2.16.0: measurable observations and gate fidelity (2026-09-25)

Idea i07: the environment can now reward the average gate fidelity to √iSWAP after free virtual-Z
corrections (leakage included; `--objective sqrt_iswap`) and show the agent only lab-measurable
values: readout populations of |01⟩, |10⟩, |11⟩ and two-qubit Pauli expectations of |0+⟩, |+0⟩, |+1⟩,
coupler not read out (`--obs-mode measured`). The defaults keep the paper's setup. The large-coupler-
drift experiment, repeated under both: no single √iSWAP pulse covers the drift (0 %); the drift-trained
policy alone gives 6.2e-3 on every device; from its pulse GRAPE reaches 1e-3 on 92 % of devices within
200-500 steps (median 3.6e-8 after 500), random starts at equal gate time 0 % within 500. The misses are
2 devices where the policy hits the amplitude bound. New page: Measurable observations and gate fidelity.

**2.16.1**: the amplitude-observation control re-evaluated with the same 500-step refinement: the
policy alone is worse (1.0e-2 vs 6.2e-3), refined results are equivalent (88 % vs 92 %, median 4e-8).

## 2.15.1: readable figures (2026-09-25)

Every figure opens full screen on click, with zoom and pan (mkdocs-glightbox). The page is wider
(content column about 1000 px on a large screen). The three-panel figures (cost and quality on the
recool and fabrication ranges, large coupler drift) are split into stacked panels
(`scripts/figtools.py`), and the two drift-dimension figures are stacked instead of side by side.

## 2.15.0: large coupler drift (2026-09-25)

Coupler drifting over ±140 MHz (worst-loop flux drift, Dai et al. 2021) with both qubits over the
recool range. For the first time no single pulse covers the range: robust GRAPE at 17.25 ns over a
35-member ensemble reaches J_T ≤ 1e-3 on none of 24 held-out devices. A PPO policy trained over the
drift does not adapt by itself (median 1.3e-2), but 100 GRAPE steps from its pulse reach 1e-3 on 83 %
of devices (median 3.7e-8 after 200), where random starts at the same gate time need ~1000 steps.
Per device 5.0 core-seconds against 54; training pays back after ≈ 60 devices. Open: robust GRAPE at
the policy's 46 ns. Policy saved in `results/jax_ppo_dr_coupler140`.

## 2.14.2: compute in logical-core hours (2026-09-25)

All cost comparisons are now in logical-core time on the laptop CPU instead of simulated samples.
`scripts/bench_costs.py` prices each algorithmic step (a PPO environment step including its network
updates, a GRAPE iteration by pulse length, a robust-GRAPE member, a rollout) on one pinned logical
core, sampling the clock so that results are stated at the 2.1 GHz base clock whatever the load.
`scripts/compute_cost.py` converts the recorded step counts. Pricing the network updates changes the
conclusions: the warm start saves 15× per device on the recool range and 2.6× on the fabrication
range (longer pulses), and pretraining pays back after ~220-350 devices (~36 with a 2M-step policy),
not ~90-130.

## 2.14.1: cost split, RL-initialised control, platforms (2026-09-24)

Cost figures split RL routes into one-off pretraining and per-device GRAPE, with total cost against the
number of devices. New pages: the case for RL-initialised optimal control (semi-amortised control)
with the literature gap, and a platform and pre-hardware plan (pulse access, Rydberg vs transmons vs
ions). Coupler-drift literature (Dai et al. 2021) and scan; large coupler drift experiment running.

## 2.14.0: fabrication-range study, test T1, the gate-time control (2026-09-24)

Robust GRAPE over the fabrication range (±18.5 MHz) keeps J_T ≤ 7.6e-4 everywhere: one pulse still
suffices for two drifting frequencies. A PPO policy trained over that range does not reach 1e-3 alone,
but as a GRAPE warm start it beats a random start by 1-3 orders of magnitude at matched gate time and
equal budget (new control, `scripts/matched_gate_time.py`, after finding that policy pulses run
~40 ns against 17.25 ns for GRAPE from scratch). Test T1: policies trained with 3 and 5 drifting
parameters are worse alone at equal budget, but the extra parameters were not sensitive enough to
matter. Training and refinement budget tables; a correction note on the recool comparison.

## 2.13.0: the hypothesis page, drift literature tied to experiments (2026-09-24)

New research-question page testing the hypothesis that drift favours learned policies: three
testable claims, literature for and against (including the n-fold penalty of value-only methods),
a scaling argument and five tests. First evidence: pulse coverage falls multiplicatively with the
number of *sensitive* drifting parameters (`scripts/drift_dimension.py`). The robustness page now
maps each measured drift figure to the operational question it answers. The environment can
randomise the coupler frequency and the couplings (`--coupler-drift-mhz`, `--g-drift-mhz`). Running:
fabrication-range comparison, and policies trained with 3 and 5 drifting parameters (test T1).

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
