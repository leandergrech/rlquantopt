# 🦩 i09 ibis: gecko made data-lean

In progress. Runs in `runs/i09_ibis/`, records in `results/i09_ibis/`.

**Question.** Can everything be done, if necessary, on a real machine with its own observables, using as
little data as possible? Concretely: how few measurement shots per pulse and how small a policy network
still give a good gate under the large coupler drift (qubits ±5.7 MHz, coupler ±140 MHz), with the
√iSWAP fidelity of [gecko](gecko.md)?

## Design

| Ingredient | What it does | Flag |
| --- | --- | --- |
| Clip instead of terminate | An amplitude beyond ±20 rad/ns is clipped and costs `oob_penalty` per unit of excess; the episode goes on, so the policy never loses its pulse (gecko's two failed devices) | `--oob-mode clip --oob-penalty λ` |
| Finite shots | Each observed value is estimated from *N* shots per measurement setting (multinomial over the readout outcomes); one step costs 30 settings, i.e. 30 *N* shots | `--shots N` |
| Smaller networks, GELU | Hidden widths and activation | `--hidden 64 64 --activation gelu` |
| Longer steps | *K* samples per step (fewer, longer steps: fewer experiments per episode); the per-sample action scale must shrink with *K* | `--n-time-steps 15 --delta-scale 4` |
| Calibration context | The policy sees the qubit and coupler frequency offsets, measured once before the pulse (spectroscopy, error 0.1 / 0.1 / 1 MHz), with or without the measured observables | `--obs-mode context` / `measured+context` |
| Fidelity from measured data | The √iSWAP fidelity (free Z) reconstructed from the 30 settings alone: magnitudes from populations, phases from Pauli expectations; no Hamiltonian | `metrics.fidelity_from_observables` |
| Model-free refinement | SPSA and CMA-ES on that estimate, every shot counted | `rlquantopt/jx/refine.py` |

The quality of a policy is now the **full pulse** (the last step). Cutting the pulse at its best step,
as in gecko, needs the exact fidelity, which a lab does not have.

## Results so far

### Policies trained on exact observations, deployed with finite shots

`scripts/ibis_eval.py`: the same 24 held-out devices as gecko, 1 − F of the full pulse, median over
devices (3 noise draws per device), for *N* shots per setting at deployment.

<div class="panels" markdown>

![Policy quality against measurement shots per pulse](../figures/ibis_eval_phase1_a.png)

![Training: best nominal 1 − F against env steps](../figures/ibis_eval_phase1_b.png)

</div>

| Policy | Actor params | Exact | *N* = 1000 (1e7 shots/pulse) | *N* = 100 (1e6) | *N* = 30 (3e5) | Env steps to 1e-2 |
| --- | --- | --- | --- | --- | --- | --- |
| gecko: 128×128 ReLU, terminate | 20k | 7.6e-3 | 1.5e-2 | 1.5e-1 | 2.8e-1 | 6.6M |
| 128×128 ReLU, clip λ = 1 | 20k | 7.1e-3 | 2.0e-2 | 3.3e-1 | 3.3e-1 | 9.9M |
| **64×64 GELU, clip λ = 1** | **6k** | 8.5e-3 | 2.1e-2 | 2.2e-1 | 2.8e-1 | **8.1M** |
| 32×32 GELU, clip λ = 1 | 2k | 1.0e-1 | 1.1e-1 | 4.5e-1 | 4.8e-1 | never |
| *K* = 15, 64×64 GELU (5M steps) | 6k | 9.7e-2 | 9.5e-2 | 1.1e-1 | 1.6e-1 | never |

1. **Closed-loop policies trained on exact data do not survive shot noise.** At 1e6 shots per pulse
   they are 30× worse: they react to each of 333 noisy observations and the errors compound.
2. **64×64 GELU matches 128×128 ReLU with a third of the actor parameters** and reaches 1e-2 in fewer
   steps than the clip-mode ReLU network; 32×32 is too small for this task.
3. **λ = 1 is too weak**: 5-9 % of the steps are still clipped with exact observations.
4. **Longer steps (*K* = 15) are robust to noise but poor** (about 0.1). With v1's per-sample action scale
   they did not learn at all (every exploration hit the bound, the agent settled on the identity);
   `--delta-scale 4` fixed that, but 5M steps were not enough.

### Refinement from measured data only

The fidelity estimate from the 30 measured settings equals the true fidelity on exact data (to the digits
printed, on GRAPE and random pulses: the neglected coupler coherences do not matter); with *N* = 1000 it
resolves infidelities down to about 5e-4 (spread ±4e-4, a small upward bias).

But black-box refinement of a policy pulse barely moves it. Noise-free, 1500 fidelity estimates:

| Correction basis | Start | CMA-ES | SPSA |
| --- | --- | --- | --- |
| 3 physical knobs (offset, AC scale, ramp) + 8 sine modes | 5.8e-3 | 4.0e-3 | 5.3e-3 |
| 24 sine modes (≤ 0.24 GHz) | 5.8e-3 | 4.3e-3 | 5.0e-3 |
| 100 sine modes (≤ 1 GHz) | 5.8e-3 | 3.7e-3 | 4.5e-3 |

(one device; a second one gives the same picture). With *N* = 1000 shots it is worse. GRAPE reaches 1e-8
from the same pulses because it gets the exact gradient in all 999 samples at every step; a black-box
optimiser has to discover that information one noisy fidelity at a time. The gradient is not the issue:
57-81 % of its power sits below 0.25 GHz, inside the bases above. **So the policy must be good enough on
its own**, which is what the current runs address.

### Running

| Run | Question | Status |
| --- | --- | --- |
| 64×64 GELU, trained with *N* = 100 shots (*K* = 3 and *K* = 15) | Does noise-aware training recover the closed-loop policy? | 1.3e-1 at 8M steps (*K* = 3), 1.4e-1 at 5M (*K* = 15): learning slowly |
| Calibration context only, open loop (λ = 5) | Can a policy that only knows the measured device frequencies play a good pulse, with about 1e4 shots per pulse? | stuck at 0.22 (the identity gate) after 11M steps |
| Calibration context + measured observables (λ = 5) | Does the calibration help the closed-loop policy? | 0.16 after 10M steps |

The two calibration runs are confounded: λ = 5 appears to suppress exploration, as the unscaled long
steps did. They need a rerun with λ = 1 before anything can be said about calibration data.

## Next

1. Calibration-context runs with λ = 1 (and λ between 1 and 5 for the bound).
2. Longer noise-aware training; observation history (a few past measurements) to filter shot noise.
3. A reward that counts the final pulse, not every intermediate step.
