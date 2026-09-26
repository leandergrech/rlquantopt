# 🐺 i10 jackal: a calibration policy for a realistic device

<span class="stamp stamp--active">🟢 active</span> *since 2026-09-26 · three device runs training (restarted after a knob-scaling fix), results expected tonight*

In progress. Runs in `runs/i10_jackal/`, records in `results/i10_jackal/`.

**Question.** Does [ibis](ibis.md)'s open-loop carrier policy, which needs only a routine frequency
calibration, survive a realistic device model? And if it is trained on the simplified model one would
write down first, how much does it lose on the realistic one (the sim-to-sim gap)?

## The device model

`rlquantopt/jx/device.py`, `rlquantopt/jx/env_device.py` (`--physics device`).

| Element | Model | Source |
| --- | --- | --- |
| Circuit | Two fixed-frequency transmons and a flux-tunable transmon coupler, 3 levels each | Sung et al., PRX 11, 021058 (2021) |
| Parameters (idle) | Qubits 4.16 / 4.00 GHz, anharmonicities −220 / −210 MHz; coupler 5.45 GHz (tunable 3.7-6.7 GHz), −90 MHz; g<sub>1c</sub> = 72.5 MHz, g<sub>2c</sub> = 71.5 MHz, g<sub>12</sub> = 5.0 MHz | Sung et al. 2021, Table 1 |
| Coupler flux curve | Asymmetric SQUID, ω<sub>c</sub>(Φ) = ω<sub>max</sub> [cos²(πΦ) + d² sin²(πΦ)]<sup>1/4</sup>, with ω<sub>max</sub> = 6.7 GHz and d = (3.7 / 6.7)² from the quoted range (our assumption: the range is the full swing); couplings scale as √(ω<sub>c</sub> / ω<sub>c,idle</sub>) | Koch et al. 2007 |
| No rotating-wave approximation | Counter-rotating terms kept; parity is conserved, so the Hamiltonian splits into two blocks (7 and 10 states, up to 3 excitations) | Checked against the full 27-level space: fidelity within 1e-5 |
| Computational states | Dressed idle eigenstates; gate in the frame of the idle Hamiltonian; free virtual Z | |
| Control | Coupler flux offset from idle (±0.15 Φ<sub>0</sub>; knob steps up to 0.03 Φ<sub>0</sub>, 0.3 rad, 0.02 Φ<sub>0</sub> per decision), a carrier at the measured dressed qubit detuning (159.5 MHz) steered by amplitude, phase and offset; 1 ns AWG hold; first-order flux-line filter, 0.5 ns | AWG 1 ns as in Braket's Rigetti example; filter constant assumed |
| Drift | Qubits ±5.7 MHz (recool, Zhang et al. 2022); coupler flux offset ±20 mΦ<sub>0</sub> (Dai et al. 2021: the worst loop after 17 days; here about −178 MHz of coupler frequency); optionally couplings ±5.7 MHz and anharmonicities ±5 MHz, which the calibration does not see | |
| Calibration (the policy's only input) | Dressed qubit frequencies and coupler idle frequency from spectroscopy, errors 0.1 / 0.1 / 1 MHz | |
| Not yet included | Decoherence (T<sub>1</sub> 60 / 30 / 10 µs in Sung et al.), spectator qubits, measured flux-line distortions | |

**A first check.** A parametric √iSWAP is reachable in this device: five constant carrier knobs
optimised by gradient give 1 − F = 7.8e-4 in 60 ns (Sung et al.'s resonant iSWAP takes 30 ns).

**The simplified model** (`--device-simplified`) is what one would write down first: rotating-wave
approximation, no direct coupling, coupler frequency linear in flux around idle, couplings fixed, ideal
AWG and flux line. Same parameters, same calibration and actions, so a policy trained on it can be
deployed on the full model unchanged.

## Experiments

All: carrier actions, calibration only, 64×64 GELU, clip λ = 1, 100 ns episodes of 20 steps of 5 ns,
3M env steps, qubits ±5.7 MHz and coupler flux ±20 mΦ<sub>0</sub>. Evaluation: `scripts/jackal_eval.py`,
the full model, 24 drifted devices, the full pulse.

| Run | Trained on | Unmeasured drift in training | Status |
| --- | --- | --- | --- |
| A | simplified model | no | running |
| B | full model | no | running |
| C | full model | couplings ±5.7 MHz, anharmonicities ±5 MHz | running |

**Tests:** `tests/test_jx_device.py` (flux curve, truncation against the 27-level space, idle gate,
parametric exchange, calibration and drift in the environment, filter, simplified model).
