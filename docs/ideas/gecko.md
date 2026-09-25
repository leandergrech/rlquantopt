# 🦎 i07 gecko: only what an experiment can measure

A parallel line of work, documented in full in
[Measurable observations and gate fidelity](../results/measurable.md).

**Question.** Does the large-coupler-drift result survive when the agent sees only what an experiment can
measure, and is scored by a gate fidelity an experiment reports?

**Design.** The reward is the average gate fidelity to √iSWAP after free virtual-Z corrections, leakage
included (`--objective sqrt_iswap`); the agent observes readout populations of |01⟩, |10⟩, |11⟩ and
two-qubit Pauli expectations of |0+⟩, |+0⟩, |+1⟩ instead of state amplitudes (`--obs-mode measured`);
exact expectations first, finite shots next.

**Outcome so far.** See the [results page](../results/measurable.md).

**Records:** `results/i07_gecko/`.
