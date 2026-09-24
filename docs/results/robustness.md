# Robustness to hardware drift

!!! info "Being updated"
    The results below use a ±10 MHz ensemble and the paper's ±50 MHz map (v2.9.0). A re-run with the
    [hardware drift ranges](../system/drift.md) (robust GRAPE over the ±5.7 MHz recool range, scored
    over all three ranges) is in progress and will replace them.

Roadmap items 1 and 2: refine RL pulses with GRAPE, and give the paper's robustness claim its
fair baseline, robust (ensemble) GRAPE.

**Result in one line.** At the same gate time and amplitude bound, GRAPE on top of RL lowers J_T
by 90×, plain GRAPE is as robust as the RL pulse, and robust GRAPE covers a 49× larger low-error
region than RL while keeping J_T = 6e-6 at nominal.

## Setup

| | |
| --- | --- |
| Gate time | 17.25 ns (345 samples of 50 ps), the paper RL pulse's best time |
| Amplitude bound | \(|u| \le 20\) rad/ns (3.18 GHz), the RL action bound |
| Model | Paper model, nominal parameters, closed system |
| GRAPE | 2000 Adam iterations, cosine-decayed step (0.05 from random guesses, 0.01 from an RL pulse), 4 random restarts |
| Robust GRAPE ensemble | 5 × 5 grid of qubit detunings over ±10 MHz (white dashed square in the maps) |
| Robustness map | Paper grid: ±50 MHz on both qubits, 1 MHz steps. Everything outside ±10 MHz is held out from the robust optimisation |
| Script | `scripts/rl_grape_robust.py` → `docs/figures/rl_grape_robust.{png,json,npz}` |

## Results

![Pulses and robustness maps](../figures/rl_grape_robust.png)

Top: pulses (\(u/2\pi\)). Bottom: \(-\log_{10} J_T\) over static detunings; red contour at
\(J_T = 10^{-3}\).

| Pulse | Nominal J_T | Area with J_T ≤ 1e-3 | Area with J_T ≤ 1e-2 | Mean log10 J_T within 10 MHz | Total variation (rad/ns) | Optimisation time |
| --- | --- | --- | --- | --- | --- | --- |
| RL (paper) | 9.5e-5 | 55 MHz² | 595 MHz² | −2.67 | 1425 | ~9 h training |
| RL → GRAPE | **1.1e-6** | 48 MHz² | 457 MHz² | −2.58 | 1509 | 23 s |
| GRAPE from random | 3.7e-6 | 71 MHz² | 713 MHz² | −2.72 | 1368 | 56 s |
| Robust GRAPE | 6.2e-6 | **2698 MHz²** | **6891 MHz²** | **−4.92** | 999 | 16.5 min |
| RL → robust GRAPE | 8.5e-5 | 286 MHz² | 1834 MHz² | −3.31 | 1614 | 4.7 min |

Our own JAX-trained agent's best pulse (46.8 ns, J_T = 9.8e-4, see
[Replication](replication.md#5-training-from-scratch-figs-5-8)) goes to **J_T = 6.8e-7** after
GRAPE refinement, in 93 s. The 10× gap to the paper's agent disappears once GRAPE finishes the job.

## What it means

1. **RL → GRAPE is a free order of magnitude or two.** Starting from the RL pulse, GRAPE goes from
   9.5e-5 to 1.1e-6 in 23 s without changing the pulse's character or its robustness. This matches
   [Sarma & Hartmann (2025)](https://arxiv.org/abs/2312.16358), who used RL as the initial guess for
   GRAPE.
2. **The "emergent robustness" of RL is not special at equal gate time and bound.** Plain GRAPE
   from random guesses, which never saw a detuned system, is as robust as the RL pulse (71 vs
   55 MHz² at J_T ≤ 1e-3). The paper's comparison was against Krotov pulses at 50 ns with a
   1.5 GHz bound, which is a different operating point.
3. **Robust GRAPE is the real baseline to beat.** Optimising the mean J_T over ±10 MHz gives a low-
   error region that extends well beyond the training square (area 2698 MHz², ~49× RL) at a nominal
   J_T of 6e-6, and a smoother pulse (lowest total variation).
4. **The RL pulse is a poor starting point for robustness.** Robust GRAPE started from the RL pulse
   stays near it: 286 MHz², about 10× worse than robust GRAPE from random starts. The RL solution sits
   in a narrow basin.

## Caveats

- **Not a like-for-like cost comparison.** GRAPE uses the model and its gradients; RL uses only
  rewards. The table's times say what each costs here, not which method is more sample efficient.
- **One run each.** Four random restarts per GRAPE setting and one ensemble choice; the numbers can
  move by a factor of a few with other seeds or grids.
- **No bandwidth limit.** All pulses, RL included, have large high-frequency content (total variation
  1000-1600 rad/ns over 17 ns). A flux-line transfer function would penalise all of them; it may
  change the ranking.
- **Static detuning only.** Robustness here is against constant frequency offsets, the paper's
  metric. Time-dependent noise and decoherence are not included.

## What this means for the RL story

The case for RL has to move from "RL finds more robust pulses" to what gradient methods cannot do:
**adapt without re-optimising**. A policy conditioned on the current detuning (roadmap item 5) that
matches robust GRAPE's error at each detuning, from a single forward pass, would be a result
gradient methods cannot produce.

!!! question "Decision: which robustness range matters?"
    The areas above depend on the ±10 MHz training range and the ±50 MHz evaluation range. For the
    paper we should fix a drift range justified by hardware (for example the typical day-to-day qubit
    frequency drift of the target device) and report both nominal J_T and the worst case or mean
    over that range.
