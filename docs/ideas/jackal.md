# 🐺 i10 jackal: a calibration policy for a realistic device

<span class="stamp stamp--active">🟢 active</span> *since 2026-09-26 · sim-to-sim gap measured (35×); trained on the realistic device 7.2e-3, closing the 5× margin to gradient-optimised knobs*

Runs in `runs/i10_jackal/`, records in `results/i10_jackal/`.

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

| Run | Trained on | Unmeasured drift in training |
| --- | --- | --- |
| A | simplified model | no |
| B | full model | no |
| C | full model | couplings ±5.7 MHz, anharmonicities ±5 MHz |

## Results so far

Each policy's best checkpoint (by the full-pulse 1 − F on its own training model's nominal device),
deployed on the **full** model: 24 drifted devices, 3 calibration-noise draws each.

| Run | Its own model, nominal | Full model, drift the calibration sees: median (below 1e-2) | Plus unseen coupling / anharmonicity drift | Env steps to 1e-2 (own model) |
| --- | --- | --- | --- | --- |
| A, trained on the simplified model | 1.1e-3 | **3.7e-2** (0 %) | 3.9e-2 (0 %) | 0.15M |
| B, trained on the full model | 6.9e-3 | **7.2e-3** (75 %) | 7.9e-3 (69 %) | 1.3M |
| C, full model + unseen drift in training | 6.3e-3 | 8.1e-3 (69 %) | 1.0e-2 (49 %) | 1.3M |

A calibration error three times larger than in training changes these by less than 10 %.

1. **The sim-to-sim gap is large.** A policy that reaches 1.1e-3 on the simplified model (the model one
   would write down first) is 35× worse on the realistic one. The rotating-wave approximation, the
   linearised flux curve, the direct coupling and the flux line all matter at this level, so a policy
   must be trained on the most faithful model available (and, on hardware, checked against it).
2. **Training on the realistic model works**, with a 5× margin still open. B gives 7.2e-3 on drifted
   devices; gradient optimisation of the same 20-step knob schedule reaches about 1.3e-3 with the same
   calibration (an exploratory check by the hippogriff session), so the policy is not yet at the limit of
   its action space.
3. **Unseen drift costs little here, and training on it did not help** at this budget (C).

!!! warning "An evaluation bug, fixed"
    Until the fix in this version, the periodic evaluation during device training ran on the paper
    model's qubit frequencies (5.03 / 5.89 GHz) instead of the device's (4.16 / 4.00 GHz): `reset_to`
    received them from the agents' `evaluate()`. The training itself was unaffected, but the logged
    evaluation and the choice of the best checkpoint were wrong. `scripts/jackal_eval.py` now re-scores the
    saved checkpoints on the device's nominal point, and `reset_to` uses it.

## Next

1. Close the 5× gap on the realistic model: longer training, more seeds, and the knob scale (a starting
   amplitude near 0.06 Φ<sub>0</sub> worked for gradient optimisation).
2. Decoherence (T<sub>1</sub> of 60 / 30 / 10 µs): a gate-time penalty.
3. A hardware contract (sampling, bandwidth, amplitude range) for a platform with pulse access.

**Tests:** `tests/test_jx_device.py` (flux curve, truncation against the 27-level space, idle gate,
parametric exchange, calibration and drift in the environment, filter, simplified model).
