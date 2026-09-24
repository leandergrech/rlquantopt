# RLQuantOpt v2: JAX upgrade plan

Goal: a GPU-native, differentiable re-implementation of ZCQPEE and its training
stack, used first to replicate Grech et al., QST 11 015030 (2026), and then as
the base for the follow-up work (fair baselines, robustness, meta-RL).

v1.0.0 (PyTorch/SB3 + QuTiP) stays in the repo untouched as the reference
implementation. The JAX code lives in `rlquantopt/jx/`.

## Key design decision: exploit excitation-number conservation

The RWA Hamiltonian (paper eq. 1) commutes with the total excitation number
N = a1†a1 + a2†a2 + b†b. The four computational basis states live in
fixed-N sectors:

| State   | N | Sector dimension (3 levels each) |
| ------- | - | -------------------------------- |
| \|000⟩  | 0 | 1                                |
| \|010⟩, \|100⟩ | 1 | 3 (\|001⟩, \|010⟩, \|100⟩) |
| \|110⟩  | 2 | 6 (\|002⟩, \|011⟩, \|020⟩, \|101⟩, \|110⟩, \|200⟩) |

So instead of integrating a 27-dimensional ODE with QuTiP, each 50 ps
piecewise-constant sample is an exact propagator `exp(-i H_N dt)` of a 3×3
and a 6×6 Hermitian matrix (via `eigh`). These are exactly the amplitudes the
v1 observation already uses. The realised 4×4 gate is block diagonal
(1 ⊕ V₂ₓ₂ ⊕ e^{iφ}), which gives the Weyl-chamber coordinates in closed form
(no general 4×4 `eig`, which JAX does not support on GPU), so the full reward
is jit-able, vmap-able and differentiable.

## Milestones and versions

Each milestone is one or more commits on `main`, tagged `v2.<minor>.<patch>`.

| Version | Milestone | Status |
| ------- | --------- | ------ |
| 2.0.0 | This plan; version bump; `rlquantopt.jx` package skeleton | done |
| 2.1.0 | Physics + metrics (sector propagators, closed-form Weyl coordinates), cross-checked against QuTiP and `weylchamber` | done |
| 2.2.0 | Functional `ZCQPEE` env (jit/vmap), step-by-step equivalent to v1 | done |
| 2.3.0 | Robustness maps of the stored RL and Krotov pulses vs the paper's Julia data (Figs. 10-12) | done |
| 2.4.0 | PPO and TRPO in pure JAX; SB3 checkpoint import; training CLI | done |
| 2.5.0 | Policy-level generalisation with the paper's own policies (Figs. 13-17) | done |
| 2.6.0 | TRPO training replication from scratch (Figs. 5-8) | done, 1 seed |
| 2.7.0 | Differentiable GRAPE baseline and QSL curve (Fig. 3) | done |
| 2.8.0 | Replication report, `docs/replication.md` | done |
| 2.9.0 | RL → GRAPE refinement and robust (ensemble) GRAPE ([results](../results/robustness.md)) | done |
| 2.10.0 | Documentation site (MkDocs, GitHub Pages) | done |
| 2.11.0 | Docs by topic, landing page, bibliography, hardware drift ranges, PPO vs TRPO | done |

## Beyond replication (next minor versions, not in this pass)

0. More seeds (≥5) of the TRPO replication, to tell seed variance from a systematic gap (best J_T
   1e-3 here vs 1e-4 in the paper).
1. Named-gate targets (CZ, √iSWAP) with average gate fidelity and leakage reported separately.
2. Fair baselines: ensemble GRAPE over the ±Δω distribution and RL → GRAPE refinement are done
   (v2.9.0); still open: RL from demonstrations, ≥5 seeds per method, a hardware-justified drift range.
3. Context-conditioned policies (Δω estimates or measurement history in the observation) to remove the domain-randomisation error floor.
4. Open-system training (Lindblad T1/T2, coupler 1/f flux noise) and a flux-line transfer function.
5. Measurement-based observations (finite-shot populations) and model-based RL.
6. Meta-RL over transmon parameter distributions.

## Replication targets (from the paper)

| Target | Paper value | Source |
| ------ | ----------- | ------ |
| Speed of PE gate found by RL | ~10 ns | Figs. 7-8 |
| Best errors | C > 0.9999, U > 0.999, J_T ~ 1e-4 | Sec. 4.2, App. B |
| Dominant pulse frequency | ~0.86 GHz (\|ω1 − ω2\|/2π) | Fig. 5 |
| RL pulse robustness | Broad low-J_T region over ±1 % Δω1, Δω2 | Fig. 10 |
| Krotov robustness | Narrow band (good guess), two pockets (bad guess) | Figs. 11-12 |
| Policy generalisation island | Δω1 ∈ [−1.4, 1.6] MHz, Δω2 ∈ [−1.4, 0.6] MHz at reward ≥ 3.8 | Fig. 16 |
| Domain randomisation ±0.1 % | Wider region, error floor > 1e-4 | Fig. 17 |
| Training budget | 13.3M steps, ~9 h CPU | App. B |

## Reference semantics copied from v1 (paper run `05-12-24_201634`)

- 1000 samples of Δt = 50 ps (T = 50 ns); K = 3 samples per step; 333 steps per episode.
- Action ∈ [−1, 1]³ → deltas × 20 rad/ns, cumulative sum on the last amplitude; |A| ≤ 20 rad/ns (10/π GHz).
  Leaving the bound clips the amplitude and truncates the episode with reward −20·(1 − t/T).
- Reward per step: −log10(1 − (C + 3U)/4) − log10-offset(0.99 → 0.00436) − 1e-3·Σ|ΔA| within the segment.
- Observation (28): polar (2|z|−1, arg z/π) of the N=1 sector for \|010⟩, \|100⟩ and the N=2 sector for \|110⟩,
  then the segment amplitudes / 20, then time in [−1, 1] (taken before the index advances), all × 0.9.
- Realised gate G[i, j] = ⟨ψ_i(t)|b_j⟩ (as `Qobj.overlap`), Weyl coordinates from `weylchamber.c1c2c3`
  rounded to 8 digits, concurrence from `weylchamber.concurrence`.
