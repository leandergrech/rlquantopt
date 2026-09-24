# Open decisions

Everything that needs your call, in one place. Each entry says what is at stake, the options,
and a recommendation. When you decide, the entry moves to **Decided** with the date.

## Open

### 1. Named-gate reward: local invariants or gate fidelity?

Moving from "any perfect entangler" to √iSWAP (then CZ). Option A scores the gate up to single-qubit
operations (local invariants, [Watts et al. 2015](https://doi.org/10.1103/PhysRevA.91.062306)).
Option B scores the average gate fidelity after free virtual-Z corrections, which is what
randomised benchmarking measures.
**Recommendation:** B, keeping A as a diagnostic. Details: [Named gates and a richer model](named-gates.md).

### 2. Next model: realistic tunable coupler (v3), then which architecture?

The current Hamiltonian's RWA, linear control and unbounded coupler excursions limit what a
simulated fidelity means. **Recommendation:** v3 = the paper's device family without RWA, with
direct coupling, a flux-tunable coupler of bounded range and a filtered flux line; then the
double-transmon coupler, where RL already holds the experimental record. Whether a hardware group
could validate our pulses should decide the architecture. Details:
[Named gates and a richer model](named-gates.md#moving-towards-the-state-of-the-art).

### 3. Drift ranges

The ranges in [Hardware drift ranges](../system/drift.md) (in-cooldown ±0.1 MHz, recool ±5.7 MHz,
fabrication targeting ±18.5 MHz) come from IBM and Fraunhofer fixed-frequency transmons. If we
target a specific device, its own numbers should replace them. Also open: adding coupler (flux)
drift, which the literature puts at ~500 kHz for flux-tunable transmons.
**Recommendation:** accept these ranges for now and add coupler drift with the v3 model.

### 4. Headline robustness metric for the paper

Candidates: worst-case J_T over the recool range, mean J_T over it, or area below a threshold on a
wide map. **Recommendation:** worst-case and mean over the recool range (the drift a pulse must
survive without re-calibration), with the fabrication range as the transfer test.

## Decided

| Date | Decision | By |
| --- | --- | --- |
| 2026-09-24 | Switch the training study to PPO if it is faster than TRPO without losing quality; 3 more seeds, not 5 | Leander |
| 2026-09-24 | Use literature-based hardware drift ranges, with every number referenced | Leander |
| 2026-09-24 | Keep the docs organised by topic, results updated in place | Leander |
