# Named gates and a richer model

A design note for the next phase: moving from "any perfect entangler" to a named gate, and from
the paper's idealised Hamiltonian to a model closer to current hardware.

!!! success "Status"
    **The named gate is done** (v2.16.0, [gecko](../ideas/gecko.md)): option B below, the average gate
    fidelity to √iSWAP with free virtual-Z corrections (`--objective sqrt_iswap`). **The richer model is
    being built** as [jackal](../ideas/jackal.md) (i10): the tunable-coupler device of Sung et al. without
    the RWA, with direct coupling, a SQUID flux curve, bounded flux, 1 ns AWG samples and a flux-line
    filter. Decoherence and the architecture after it are still [open decisions](decisions.md).

## Should we target a named gate?

Yes. A perfect entangler is a set of gates, so an agent's result cannot be compared with published
two-qubit gate fidelities, compiled into circuits, or benchmarked on hardware without a second
step that turns it into a specific gate. Every experiment below reports a fidelity to a named gate.

The natural first target is **√iSWAP**. The paper's control follows the parametric tunable-bus
scheme of [McKay et al., PR Applied 6, 064007 (2016)](https://doi.org/10.1103/PhysRevApplied.6.064007),
which modulates the bus at the qubit-qubit detuning to create a resonant XX + YY (iSWAP-type)
interaction. √iSWAP needs half the rotation of iSWAP, so it should be reachable in about the same
~10 ns. **CZ** is possible too (through the \|11⟩-\|20⟩ or \|11⟩-\|02⟩ transitions, which live in the same
N = 2 sector) and is the gate most platforms report, but it needs a different modulation frequency.

## How the reward changes

Today: \(J_T = 1 - (C + 3U)/4\), with \(C\) the concurrence (1 anywhere in the perfect-entangler
polyhedron) and \(U\) the unitarity.

**Option A: same gate up to local operations.** Replace the concurrence by a local-invariant
functional that is minimal only at the target's Weyl point, for example √iSWAP at
\((c_1, c_2, c_3) = (1/4, 1/4, 0)\) in units of π
([Watts et al., PRA 91, 062306 (2015)](https://doi.org/10.1103/PhysRevA.91.062306)). Single-qubit
corrections are left free. Small change: `metrics.c1c2c3` already gives the coordinates.

**Option B: average gate fidelity with free Z corrections (recommended).** On hardware, Z rotations
are "virtual" and cost nothing ([McKay et al., PRA 96, 022330 (2017)](https://doi.org/10.1103/PhysRevA.96.022330)),
but X/Y single-qubit gates are not. So the fair target is the average gate fidelity to √iSWAP
after the best single-qubit Z phases. For a gate \(G\) restricted to the computational subspace
(possibly non-unitary because of leakage) and target \(V\), with \(M = V^\dagger G\) and \(d = 4\)
([Pedersen et al., Phys. Lett. A 367, 47 (2007)](https://doi.org/10.1016/j.physleta.2007.02.069)):

\[
F_{\text{avg}} = \frac{\operatorname{Tr}(M M^\dagger) + |\operatorname{Tr} M|^2}{d(d+1)},
\qquad
F = \max_{\phi_0, \phi_1} F_{\text{avg}}\big(Z_{\phi_0}\otimes Z_{\phi_1}\, G\big).
\]

The first term penalises leakage (it is \(4U\)), so the separate unitarity term goes away. The
reward becomes \(-\log_{10}(1 - F)\), the same shape as now. The maximisation over two phases is a
small inner problem over two (or, with corrections before and after the gate, four) phases; a few
Newton or Adam steps in JAX, differentiable end to end.

!!! question "Decided (2026-09-25): option B"
    A keeps the paper's spirit (the agent finds the entangling power; single-qubit gates are
    compiled away). B is what an experiment measures with interleaved randomised benchmarking.
    B was implemented in v2.16.0 (`metrics.fidelity_free_z`); A remains available through the Weyl
    coordinates as a diagnostic.

## Do we need more states?

**Not with this Hamiltonian.** It conserves the excitation number, so the four computational states
already give the full 4×4 gate: \|000⟩ is fixed, \|010⟩ and \|100⟩ live in the 3-dimensional N = 1
sector, \|110⟩ in the 6-dimensional N = 2 sector. A named-gate reward uses the same \(G\) we already
compute; only `metrics.py` and `env.step` change.

**Yes, as soon as the model stops conserving excitations** (next section). Then the propagator
mixes sectors, and we propagate the full space: 27 states with 3 levels each, 64 with 4. That is
still cheap in JAX: one 27×27 `eigh` per sample instead of a 3×3 and a 6×6, roughly 10-20× slower
than now, but still far faster than v1. Jackal's device model does better: without the RWA, parity is
still conserved, so the space splits into an even block of 7 and an odd block of 10 states (up to 3
excitations), checked against the full 27-level space to 1e-5 in fidelity.

## Limits of the current Hamiltonian

| Approximation | Why it matters |
| --- | --- |
| **Rotating-wave approximation on the couplings** (drops \(a_j b\), \(a_j^\dagger b^\dagger\)) | It is what makes excitation number conserved. The dropped terms shift the levels by about \(g^2/(\omega_q + \omega_c) = (0.1\,\text{GHz})^2 / 12.5\,\text{GHz} \approx 0.8\) MHz, which is larger than in-cooldown drift. A pulse tuned to J_T ~ 1e-5 in the RWA model will miss on real hardware. |
| **Linear control \(u(t)\, b^\dagger b\)** | A real coupler is tuned by flux. A SQUID transmon's frequency varies non-linearly with flux, the couplings \(g_j\) change with the coupler frequency, and the flux line filters the pulse. None of this is modelled. |
| **Unrealistic coupler excursions** | The control bound allows the coupler to move ±3.2 GHz around 7.445 GHz, and the paper's RL pulse uses up to 2.9 GHz of it, taking the coupler below both qubit frequencies (5.03 and 5.89 GHz). Real couplers have a limited tuning range and crossings cause leakage. |
| **No direct qubit-qubit coupling** | Tunable-coupler designs use a direct capacitive coupling \(g_{12}\) that interferes with the coupler path to cancel residual ZZ ([Yan et al., PR Applied 10, 054062 (2018)](https://doi.org/10.1103/PhysRevApplied.10.054062)). Without it we cannot model the idle ZZ that limits real devices. |
| **Closed system** | T1/T2 enter only at evaluation. |
| **Two qubits** | No spectator qubits or crosstalk. |
| **3 levels per mode** | Adequate here (paper appendix A), but leakage into the coupler's second excited state matters in modern couplers. |

## Moving towards the state of the art

State-of-the-art two-qubit gates on superconducting hardware, with the model each needs:

| Architecture | Result | What it would take |
| --- | --- | --- |
| Transmons + tunable coupler, CZ and iSWAP | CZ 99.76 %, iSWAP 99.87 % ([Sung et al., PRX 11, 021058 (2021)](https://doi.org/10.1103/PhysRevX.11.021058)) | Same device family as the paper. Model of [Yan et al. (2018)](https://doi.org/10.1103/PhysRevApplied.10.054062) without RWA, with \(g_{12}\), flux-tunable coupler and bandwidth-limited flux pulses. |
| Double-transmon coupler (DTC) | CZ 99.90 % between highly detuned qubits, **with a model-free RL pulse optimisation** ([Li et al., PRX 14, 041050 (2024)](https://doi.org/10.1103/PhysRevX.14.041050)) | Two coupler transmons; the most directly relevant: RL is already how the record was reached. |
| Fluxonium qubits + transmon coupler | CZ 99.922 % after RL-based optimisation ([Ding et al., PRX 13, 031035 (2023)](https://doi.org/10.1103/PhysRevX.13.031035)) | Fluxonium Hamiltonians (many levels, low frequency); a larger change of physics. |

**My recommendation, in order:**

1. **v3 model: a realistic tunable coupler** in the paper's device family. Full-space propagation
   without RWA, direct coupling \(g_{12}\), coupler frequency as a function of flux with a bounded
   range, a flux-line filter on the pulse, and a √iSWAP (then CZ) target with Option B. The circuit
   parameters can come from [scqubits](https://doi.org/10.22331/q-2021-11-17-583) rather than being
   set by hand. This keeps continuity with the paper and closes the largest modelling gaps.
2. **Then the double-transmon coupler.** It is where RL already holds the experimental record, so a
   simulation study (model-based RL, robustness to recool drift, transfer across devices) would
   speak directly to current practice.

!!! question "Decision: which architecture after v3?"
    DTC keeps us in transmon physics and next to an RL result from 2024; fluxonium is a bigger
    change with the highest reported fidelity. Is there a hardware group we could validate with?
    That should decide it.
