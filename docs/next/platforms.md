# Beyond two transmons, and what to do before hardware

## Who offers pulse-level access today

Pulse-level access to commercial quantum computers has narrowed: IBM removed pulse-level control from
all its processors in February 2025, and a 2026 survey of thirteen vendor stacks describes public
access to pulse and control-electronics interfaces as "bifurcated" [[malarchick2026]](../bibliography.md#malarchick2026).
What remains through Amazon Braket (checked 24 September 2026):

| Device | Technology | Control exposed | Relevance |
| --- | --- | --- | --- |
| QuEra Aquila | Neutral atoms, Rydberg | Analog Hamiltonian simulation: global amplitude, phase and detuning of the drive over time, up to 256 atoms | The most open analog control available; natural home for RL-shaped global pulses |
| Rigetti Ankaa-3 | Superconducting, tunable couplers | Braket Pulse (OpenPulse frames, ports, arbitrary waveforms) | Same device family as our model. The documentation's example port uses 1 ns samples; at 1 ns the paper's RL pulse loses its gate (J_T 0.29, [home page](../index.md#why-fine-delta-parameterised-pulses)), so pulses must be trained under the device's sampling |
| IonQ, AQT, IQM | Trapped ions, superconducting | Gate-based | No pulse access through Braket |

## Candidate platforms

| Platform | Why it fits | What it would take |
| --- | --- | --- |
| **Tunable-coupler transmons (v3)** | Continuity with the paper; pulse access on Rigetti Ankaa-3; current records use tunable couplers [[sung2021]](../bibliography.md#sung2021) | Realistic coupler model (flux map, no RWA, direct coupling), 1 ns sampling, named-gate reward |
| **Rydberg atoms** | Open pulse-level hardware (Aquila); two-qubit gates at 99.5 % designed by optimal control [[evered2023]](../bibliography.md#evered2023), with time-optimal pulses known [[jandura2022]](../bibliography.md#jandura2022); drift dimension grows with the atom count (per-atom position, laser intensity and detuning inhomogeneity), which is exactly the scaling H1 needs | A JAX Rydberg-blockade environment (global drive, N atoms, blockade), drift over atom positions and laser parameters |
| **Trapped ions** | Robust entangling gates by pulse shaping are mature [[leung2018]](../bibliography.md#leung2018); slow drift of motional modes is a natural drift variable | Mølmer-Sørensen model with motional modes; no open pulse access on public clouds, so hardware would need a lab partner |

## Recommendation

Go bigger, in two tracks, and keep the transmon line:

1. **Rydberg atoms next.** Hardware you can actually submit pulses to today, a control problem
   (global drive) where RL has a structural advantage (one pulse must serve many atoms with different
   local parameters), and a drift dimension that grows with system size, which is the direct test of
   the [hypothesis](../hypothesis/drift-and-dimension.md). It also connects to the DeepScar idea of
   controlling many-body dynamics in Rydberg arrays.
2. **Transmons, hardware-shaped.** The v3 model, constrained to what Rigetti Ankaa-3 exposes (sampling,
   bandwidth, amplitude), so that pulses are submittable when access comes.
3. **Ions later**, with a lab collaborator.

## What to do until there is hardware

1. **Simulate the hardware contract, not just the physics.** Train and refine only pulses the target
   device would accept: its sampling interval, amplitude and bandwidth limits, pulse-length quantisation,
   and for Aquila the waveform constraints stated in its device properties. A pulse that cannot be submitted
   is not a result.
2. **Count shots, not simulator samples.** Replace exact J_T by estimates from finite measurements with
   state-preparation and measurement errors, and redo the cost comparison in shots (test T4). This is
   where the hypothesis says RL should gain, and it costs nothing but compute.
3. **Drift as a time series.** Replace static drift boxes with drift trajectories (random walks and jumps
   calibrated to the measured magnitudes on [Hardware drift ranges](../system/drift.md)) and measure
   *fidelity-hours*: how long each pipeline keeps a device above target per unit of calibration effort.
   This is the operational metric a hardware group cares about.
4. **Dress rehearsal on vendor simulators.** Braket's local analog simulator runs the same program format
   as Aquila, so the full pipeline (policy → program → results → refinement) can be tested end to end
   before the first paid shot.
5. **Publish a benchmark.** Package the JAX environments (transmon, and Rydberg next) with the drift
   models and baselines as an open benchmark for drifting-device quantum control. It makes the work
   reusable and is a concrete reason for groups with hardware to get in touch.
6. **Keep the first hardware experiment small.** A few-atom Rydberg experiment on Aquila (a blockade gate
   or a global-pulse state preparation, refined across two sessions) is a feasible first test of
   "trained in simulation, refined on the device".
