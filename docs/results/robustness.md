# Robustness to hardware drift

**Answer so far.** At the paper's gate time and amplitude bound, the RL pulse is no more robust than
a plain GRAPE pulse. Robust GRAPE, which optimises the mean error over the recool drift range, keeps
J_T below 1.4e-4 everywhere in that range, about 30× better than RL on average. What RL can offer
instead is adaptation: see [RL vs GRAPE: cost per device](#rl-vs-grape-cost-per-device) below.

## Pulses under the three drift ranges

All pulses last 17.25 ns (the paper RL pulse's best time) and respect the RL bound
\(|u| \le 20\) rad/ns. Drift ranges are those of [Hardware drift ranges](../system/drift.md);
each is scored on an 11 × 11 grid of qubit detunings.

| Pulse | Nominal J_T | In-cooldown ±0.1 MHz: worst | Recool ±5.7 MHz: mean / worst | Fabrication ±18.5 MHz: mean / worst | Optimisation time |
| --- | --- | --- | --- | --- | --- |
| RL (paper agent) | 9.5e-5 | 1.0e-4 | 1.5e-3 / 4.0e-3 | 1.4e-2 / 3.9e-2 | ~9 h training |
| RL → GRAPE | 1.1e-6 | 4.2e-6 | 1.9e-3 / 6.2e-3 | 2.0e-2 / 6.3e-2 | 82 s |
| GRAPE, random start | 3.7e-6 | 5.9e-6 | 1.5e-3 / 4.7e-3 | 1.5e-2 / 5.0e-2 | 211 s |
| **Robust GRAPE** (recool ensemble) | **9.4e-7** | **1.1e-6** | **3.1e-5 / 1.4e-4** | **8.6e-4 / 6.5e-3** | 44 min |
| RL → robust GRAPE | 8.0e-6 | 9.5e-6 | 4.8e-4 / 1.4e-3 | 6.6e-3 / 2.5e-2 | 6 min |

Times are on the shared laptop CPU; robust GRAPE runs 25 detuned systems × 4 restarts × 2000
iterations.

![Pulses and maps](../figures/rl_grape_robust.png)

Top: pulses. Bottom: \(-\log_{10} J_T\) on the paper's ±50 MHz map; red contour at \(J_T = 10^{-3}\);
white boxes: recool (dashed) and fabrication (dotted) ranges.

What the table says:

1. **In-cooldown drift does not matter.** Every pulse keeps its nominal J_T to within a factor of
   ~4 over ±0.1 MHz.
2. **Recool drift is where pulses break.** A single recool degrades the RL and GRAPE pulses from
   1e-4-1e-6 to ~2e-3 on average.
3. **RL's robustness is not special.** RL, RL → GRAPE and plain GRAPE have the same recool mean,
   ~1.5-1.9e-3. The paper's comparison with Krotov was at a different gate time (50 ns) and bound
   (1.5 GHz).
4. **Robust GRAPE wins on robustness,** and it also has the best nominal J_T. Its mean over the
   fabrication range, which it never trained on, is 8.6e-4.
5. **Starting robust GRAPE from the RL pulse is worse** than from random guesses: the RL solution
   sits in a narrow basin.

## GRAPE losses

![GRAPE losses](../figures/grape_losses.png)

(a) The single optimisations of the table above: J_T at each iteration (thin) and best so far
(thick); for robust GRAPE, the mean over the 25 systems. RL → GRAPE touches 1e-6. (b) Across the
24 drifted devices of the next section: median (solid), geometric mean (dashed) and quartile band
of the best J_T so far. (c) Zoom on the last 20 % of the iterations, each device thin. (d) Final J_T
on every drifted device for every method.

!!! note "Refinement needs small steps"
    Started from the RL pulse with Adam at step 0.01 (as in the table), GRAPE first jumps out of the
    RL pulse's basin (J_T up to 5e-2) and only improves once the step has decayed, after ~600
    iterations. With a step of 3e-4 it reaches J_T = 5e-7 in 50 iterations and 7e-9 in 200 from
    the paper's RL pulse. The per-device comparison below uses 3e-4.

## RL vs GRAPE: cost per device

The question where RL can win: a device that is re-calibrated after every cooldown, or a fleet of
devices, each with its own frequencies from the recool range. GRAPE pays its full cost for every
device; a policy trained with domain randomisation pays once, then produces each device's pulse in
a single rollout, and that pulse can seed a short GRAPE run.

Test set: the nominal device plus 24 devices with qubit frequencies drawn uniformly from the recool
range (±5.7 MHz on each qubit), none of them seen in training. Target: J_T ≤ 1e-3. Cost is counted
in simulated 50 ps samples (an RL environment step is 3; a GRAPE iteration on a 345-sample pulse is
3 × 345, forward plus backward). Both RL agents are PPO, seed 123, 20M steps; the drift-trained one
redraws the qubit frequencies from the recool range every episode.

![Cost and quality](../figures/sample_efficiency.png)

| Method | J_T nominal | J_T on drifted devices: median [quartiles] | Reaches 1e-3 | One-off cost | Cost per device |
| --- | --- | --- | --- | --- | --- |
| GRAPE per device (1000 iterations, random start) | 1.5e-3* | 2.4e-5 [1.5e-5, 3.2e-5] | 96 % | 0 | 7.0e5 samples, 12 s |
| Robust GRAPE (one pulse for all) | 9.4e-7 | **1.7e-5** [4.9e-6, 2.8e-5] | 100 % | 2.1e8 samples, 44 min | 0 |
| RL trained without drift | 9.8e-4 | 4.7e-3 [3.5e-3, 1.1e-2] | 0 % | 6.0e7 samples, 61 min | 1 rollout |
| RL trained with drift | 1.2e-3 | 1.6e-3 [1.3e-3, 1.7e-3] | 0 % | 6.0e7 samples, 66 min | 1 rollout |
| RL with drift → 200 GRAPE steps | 8.5e-5 | 2.0e-4 [8.6e-5, 3.3e-4] | 100 % | 6.0e7 samples, 66 min | **3.8e4 samples, 6 s** |

\* A single random restart on the nominal device happened to stall; the drifted devices show the
typical GRAPE result.

What it shows:

1. **Training with drift works as intended.** The drift-trained agent's gates hardly change across
   devices (quartiles 1.3-1.7e-3), whereas the agent trained on one device degrades by 5× and
   scatters over a decade. The policy adapts; it just does not reach 1e-3 on its own.
2. **An RL pulse is a warm start that cuts GRAPE's cost per device 18×.** From the drift-trained
   agent's pulse, 200 small GRAPE steps reach 1e-3 on every device, at 3.8e4 samples against 7.0e5
   for GRAPE from a random guess.
3. **Amortisation pays off after ~90 devices.** Training costs 6.0e7 samples once. Against GRAPE
   from scratch, RL + short GRAPE is cheaper after 6.0e7 / (7.0e5 − 3.8e4) ≈ 90 devices or
   recalibrations (in wall time, after several hundred, because the RL training ran on a shared CPU).
4. **Within the recool range, one robust pulse is enough.** Robust GRAPE's single pulse beats every
   per-device method on quality (median 1.7e-5) and needs nothing per device. Adaptation only matters
   when the drift is larger than what one pulse can cover.

**Where RL should win, and the next experiment.** Over the fabrication-targeting range
(±18.5 MHz), robust GRAPE's worst case already degrades to 6.5e-3 (table above). That is where a
policy that adapts per device, plus a short refinement, should beat both a single robust pulse and
per-device GRAPE. We will repeat this comparison at that range, with the refinement budget and
the RL training budget varied.
