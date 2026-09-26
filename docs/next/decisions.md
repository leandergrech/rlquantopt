# Open decisions

Everything that needs your call, in one place. Each entry says what is at stake, the options,
and a recommendation. When you decide, the entry moves to **Decided** with the date.

## Open

### 1. Target platform and gate for the hardware contract

The calibration policy should eventually emit a waveform a vendor accepts: its sampling rate, amplitude
range, bandwidth and frame model. **Rigetti Ankaa-3 via Braket** has pulse access and tunable-coupler
transmons whose native two-qubit gates are, as far as we know, activated by flux modulation, close to our
carrier approach; its current pulse constraints need checking first. **Generic tunable-coupler transmons**
(Sung et al. 2021, the jackal model) are the best documented but have no public pulse access. For the gate,
√iSWAP keeps continuity with gecko, ibis and jackal; CZ is the most reported gate and baseband-friendly.
**Recommendation:** finish jackal on the generic model with √iSWAP, then write the hardware contract for
Rigetti Ankaa-3 if we want something submittable. Details: [Beyond two transmons](platforms.md).

### 2. Decoherence in the device model

Jackal's device is a closed system. Options: a post-hoc T₁/T₂ error estimate on the final gate (cheap, fine
for comparing gate times), or a Lindblad model (about 100× costlier, needed only for final numbers).
**Recommendation:** post-hoc first, Lindblad for the final checks of the best policies.

### 3. Which architecture after the tunable coupler?

The double-transmon coupler keeps us in transmon physics and next to an RL result from 2024
[[li2024]](../bibliography.md#li2024); fluxonium is a bigger change with the highest reported fidelity
[[ding2023]](../bibliography.md#ding2023); Rydberg atoms have open pulse-level hardware today.
Whether a hardware group could validate our pulses should decide it. Details:
[Named gates and a richer model](named-gates.md#moving-towards-the-state-of-the-art),
[Beyond two transmons](platforms.md).

### 4. Headline robustness metric for the paper

Candidates: worst-case infidelity over the drift range, the median over held-out devices with the share
below a threshold (what ibis and jackal report, for the full pulse), or area below a threshold on a wide map.
**Recommendation:** median and share below 1e-2 (policy alone) and 1e-3 (after refinement) on held-out
devices, with the worst case alongside.

### 5. Drift ranges for a specific device

The ranges in [Hardware drift ranges](../system/drift.md) (in-cooldown ±0.1 MHz, recool ±5.7 MHz,
fabrication targeting ±18.5 MHz, coupler flux up to 20 mΦ₀) come from published fixed-frequency transmons
and flux-loop measurements. If we target a specific device, its own numbers should replace them.
**Recommendation:** keep them until a platform is chosen (decision 1).

## Decided

| Date | Decision | By |
| --- | --- | --- |
| 2026-09-26 | Every idea carries a status stamp (active, concluded, archived); the idea farm states its lessons for sample-efficient RL in general terms, and the next direction is tracking with model-based and model-free parts via improvement equivalence | Leander |
| 2026-09-26 | Hippogriff's gate-environment step waits for jackal and builds on its device model: identify the drift the calibration does not see across repeated executions of the open-loop pulse | Leander |
| 2026-09-26 | Close ibis (seeds, calibration stress, longer steps, smaller network), then the next model: the paper's device family without RWA, with direct coupling, a flux-tunable coupler and a filtered flux line (jackal, i10), with a sim-to-sim transfer test | Leander |
| 2026-09-25 | Sample efficiency is the priority, and everything must be doable with a real machine's observables if necessary: no dependence on the model in the policy's inputs; amplitude bound handled by clipping with a penalty; smaller networks with GELU | Leander |
| 2026-09-25 | Replicate the large-coupler-drift result with measurable observations and a gate fidelity (named-gate reward: option B, average gate fidelity with free Z; done in v2.16) | Leander |
| 2026-09-25 | Research ideas are smoke-tested in isolation on lightweight toy environments before the gate environment | Leander |
| 2026-09-25 | Idea write-ups live in a "Working ideas" docs section listed by emoji only | Leander |
| 2026-09-24 | Every new training idea gets a number, an animal codename, its own run directory and registry entry; earlier runs are never overwritten | Leander |
| 2026-09-24 | Switch the training study to PPO if it is faster than TRPO without losing quality; 3 more seeds, not 5 | Leander |
| 2026-09-24 | Use literature-based hardware drift ranges, with every number referenced | Leander |
| 2026-09-24 | Keep the docs organised by topic, results updated in place | Leander |
| 2026-09-24 | PPO replaces TRPO for training studies (same best gate, ~2× faster); 4 seeds run | Leander |
