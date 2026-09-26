# Scripts and CLI

Every figure in these docs is produced by one of these. Run them from the repository root with the
`rlqo-jax` environment; `JAX_PLATFORMS=cpu` is the fastest float64 option on the laptop.

## Replication and robustness (v2.1-v2.16)

| Command | Produces | Time (laptop CPU) |
| --- | --- | --- |
| `pytest tests/` | 122 checks: v1, QuTiP and `weylchamber` equivalence, agents, every idea's building blocks ([Getting started](../getting-started.md#run-the-tests)) | ~3-4 min |
| `python scripts/bench_jx_env.py` | Environment throughput for several batch sizes | ~1 min |
| `python -m rlquantopt.jx.train --algo trpo --seed 123 --eval-every 65536 --tag paper` | `runs/trpo_*_paper/`: the 20M-step replication run | ~2 h |
| `python scripts/replicate_training_figs.py runs/<run>` | `docs/figures/training_evolution.png` (Figs. 5-8) | ~10 s |
| `python scripts/replicate_robustness.py` | `docs/figures/robustness_maps.{png,npz}` (Figs. 10-12) | ~15 min |
| `python scripts/replicate_generalisation.py` | `docs/figures/generalisation.{png,npz}` (Figs. 13-17); needs the v1 checkpoints | ~20 min |
| `python scripts/replicate_qsl.py` | `docs/figures/qsl.{png,npz}` (Fig. 3) | ~1.5 h |
| `python scripts/rl_grape_robust.py` | `docs/figures/rl_grape_robust.{png,npz,json}`: RL vs GRAPE vs robust GRAPE under the drift ranges | ~50 min |
| `python scripts/compare_training.py RUN ... --labels ...` | `docs/figures/training_comparison.{png,json}`: best J_T per run and seed statistics | seconds |
| `python scripts/plot_training_losses.py RUN ... --labels ...` | `docs/figures/training_losses.png`: RL losses and diagnostics | seconds |
| `python scripts/sample_efficiency.py --static-run RUN --dr-run RUN` | `docs/figures/sample_efficiency.{png,json,npz}`: cost per device, RL vs GRAPE (`--replot` redraws) | ~15 min |
| `python scripts/plot_grape_losses.py` | `docs/figures/grape_losses.png`: GRAPE losses and final distributions | seconds |
| `python scripts/drift_dimension.py [--absolute-mhz 5.7 --tag _absolute]` | `docs/figures/drift_dimension*.{png,json}`: pulse coverage vs number of drifting parameters | ~5 min |
| `python scripts/drift_dimension_policies.py --runs D2=RUN D3=RUN D5=RUN` | `docs/figures/drift_dimension_policies.{png,json}`: test T1 | ~40 min |
| `python scripts/gate_times.py --fab-dr RUN --t1 D2=RUN D3=RUN D5=RUN` | `docs/figures/gate_times.json`: gate times the policies choose | ~5 min |
| `python scripts/matched_gate_time.py` | `docs/figures/matched_gate_time.json`: GRAPE from random at the policies' gate times | ~40 min |
| `python scripts/coupler_drift_scan.py` | `docs/figures/coupler_drift_scan.{png,json}`: stored pulses vs coupler drift | ~1 min |
| `python scripts/coupler_drift_experiment.py robust` / `evaluate --dr-run RUN --static-run RUN` / `replot` | `docs/figures/coupler_robust.*`, `coupler_drift.{png,json}`: large coupler drift (with `--objective sqrt_iswap` and a `--tag`: gecko's `coupler_drift_gecko*`) | ~40 min / ~1.5 h / seconds |
| `taskset -c 2 python scripts/bench_costs.py` (single-threaded XLA, see its docstring) | `docs/figures/compute_costs.json`: logical-core seconds per GRAPE iteration, robust-GRAPE member, PPO step and rollout; `scripts/compute_cost.py` converts the experiments' sample counts into core-hours with it | ~15 min |
| `python scripts/pulse_sampling.py` | `docs/figures/pulse_sampling.png`: the RL pulse at coarser sampling | ~10 s |
| `python scripts/make_bibliography.py` | `docs/bib/*`, `docs/bibliography.md` from DOIs | ~1 min |

## The idea farm (v2.17 on)

| Command | Produces | Idea |
| --- | --- | --- |
| `python -m rlquantopt.jx.train --idea <animal> ...` | `runs/iNN_<animal>/<algo>_<timestamp>_s<seed>_<tag>/` | any gate-env idea |
| `python -m rlquantopt.jx.toybench --env {pendulum,mountaincar,tracker} --variant <v> [--seeds 5] [--set key=value]` | a toy-grid run: evaluation curves per seed | 🦔 echidna |
| `python -m rlquantopt.jx.foxbench --seeds 1 --seed-start 0 --devices 12 --budget 60 [--summarise DIRS]` | fox vs PPO fine-tuning on drifted toy devices, paired | 🦊 fox |
| `python -m rlquantopt.jx.hippogriff --seed 0 --devices 12 --episodes 5 [--set filter=grid --tag grid]` | `runs/i08_hippogriff/seed<s>_<time>_<tag>/results.json`: all modes, returns and beliefs; the stage-1 policies are cached in `runs/i08_hippogriff/pretrained/` | 🦄 hippogriff |
| `python scripts/plot_hippogriff.py` | `docs/figures/hippogriff_{returns,belief}.png`, `hippogriff.json` | 🦄 hippogriff |
| `python scripts/ibis_eval.py --runs LABEL=RUN ... [--eval-shots 0 1000 100 30] [--tag T]` | `docs/figures/ibis_eval{T}.{png,json}`: 24 drifted devices, full pulse, finite shots at deployment | 🦩 ibis |
| `python scripts/ibis_eval.py --stress --runs LABEL=RUN ...` | `docs/figures/ibis_stress{T}.json`: calibration errors ×1/×3/×10, a stale calibration, drift the calibration cannot see | 🦩 ibis |
| `python scripts/plot_carrier_mdp.py [--run RUN]` | `docs/figures/carrier_mdp_{a,b}.png`: the carrier MDP schematic and a trained policy's knobs | 🦩 ibis, 🐺 jackal |
| `python scripts/jackal_eval.py --runs LABEL=RUN ... [--tag T]` | `docs/figures/jackal_eval{T}.json`: policies deployed on the full device model | 🐺 jackal |
| `python scripts/idea_records.py` | `results/iNN_<animal>/summary.json` and small source files: the numbers the idea pages quote | all |
| `python scripts/tb_sync.py &` then `tensorboard --logdir runs/_tensorboard` | TensorBoard event files mirrored from every `progress.csv` and the toy benchmarks (skips `runs/_attic/`) | all |

## `rlquantopt.jx.train`

`python -m rlquantopt.jx.train --help` lists everything; the main switches:

| Argument | Default | Meaning |
| --- | --- | --- |
| `--algo` | `trpo` | `trpo`, `ppo`, or a research agent: `ppo_judge`, `chameleon`, `dragonfly`, `ppo_plus` ([Research agents](idea-agents.md)) |
| `--idea` | none | Idea codename from `rlquantopt/jx/ideas.py`: runs go to `runs/iNN_<idea>/` |
| `--seed` | 123 | PRNG seed (network init, environments, sampling) |
| `--total-steps` | 20,000,000 | Environment steps |
| `--n-envs`, `--n-steps` | 4 × 2048 (TRPO), 64 × 128 (PPO) | Rollout size per update |
| `--lr` | 3e-4 | Initial learning rate (harmonic decay by 10^0.4 over training) |
| `--activation` | relu (TRPO), tanh (PPO) | `relu`, `tanh`, `leaky_relu`, `gelu` |
| `--hidden` | 128 128 | Hidden-layer widths of actor and critic |
| `--eval-every` | 2,000,000 | Environment steps between evaluations and checkpoints |
| `--out`, `--tag` | `runs`, empty | Output directory and a suffix for the run name |

**The device and its drift** (paper model):

| Argument | Default | Meaning |
| --- | --- | --- |
| `--max-drift` | 0 | Domain randomisation of the qubit frequencies, as a fraction (1e-3 = ±0.1 %) |
| `--coupler-drift-mhz`, `--g-drift-mhz` | 0 | Domain randomisation of the coupler frequency and of both couplings, ±MHz |
| `--fixed-drift` | off | Draw the drift once per environment (v1 behaviour) instead of every episode |

**The MDP** ([Environment](environment.md), [the new MDP](../system/carrier-mdp.md)):

| Argument | Default | Meaning |
| --- | --- | --- |
| `--objective` | `pe` | `pe`: the paper's perfect-entangler \(J_T\); `sqrt_iswap`: 1 − F to √iSWAP with free Z (gecko on) |
| `--obs-mode` | `amplitudes` | `amplitudes` (v1), `measured` (gecko), `context` (calibration only, open loop), `measured+context` |
| `--shots` | 0 | Measured observations estimated from N shots per setting, 30 settings per step (0 = exact) |
| `--oob-mode`, `--oob-penalty` | `terminate`, 1.0 | `clip`: clip the amplitude and pay the penalty per unit of normalised excess, instead of ending the episode |
| `--action-mode` | `delta` | `delta` (v1): per-sample amplitude increments; `carrier`: nudge amplitude, phase and offset of a carrier at the qubit detuning |
| `--n-time-steps` | 3 | K, pulse samples per environment step |
| `--delta-scale` | 0 (= 20) | rad/ns per sample for a unit delta action; scale down for long steps (20 × 3 / K) |

**The realistic device** (jackal; `--physics device` implies carrier actions and calibration-only
observations):

| Argument | Default | Meaning |
| --- | --- | --- |
| `--physics` | `rwa` | `rwa`: the paper's model; `device`: no RWA, direct coupling, SQUID coupler (`device.py`) |
| `--device-simplified` | off | Train on the simplified device model (RWA, no g₁₂, linear flux curve, ideal AWG and filter) |
| `--gate-time` | 150 | Episode length in ns (0.1 ns samples) |
| `--qubit-drift-mhz`, `--flux-drift-mphi0`, `--eta-drift-mhz` | 0 | Physical drift: qubit frequencies, coupler flux offset, anharmonicities (the last not in the calibration) |
| `--awg-dt`, `--filter-tau` | 1.0, 0.5 | AWG sample time and flux-line filter time constant, ns |

Research-agent switches (`--judge-*`, `--target-kl`, `--n-candidates`, `--sil-coef`, `--ou-rho`, `--veto`, ...)
are grouped by idea in `--help` and listed on each idea's page.

## Documentation site

```bash
pip install mkdocs-material mkdocs-glightbox
mkdocs serve          # live preview at http://127.0.0.1:8000
mkdocs build --strict # what the GitHub Action runs
```

Code shown in the docs is pulled from the source between `# --8<-- [start:name]` and
`# --8<-- [end:name]` markers, so it always matches `main`. Keep those marker lines when editing
the functions around them.
