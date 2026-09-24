# Scripts and CLI

Every figure in these docs is produced by one of these. Run them from the repository root with the
`rlqo-jax` environment; `JAX_PLATFORMS=cpu` is the fastest float64 option on the laptop.

| Command | Produces | Time (laptop CPU) |
| --- | --- | --- |
| `pytest tests/` | 27 checks against v1, QuTiP and `weylchamber` | ~30 s |
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
| `python scripts/pulse_sampling.py` | `docs/figures/pulse_sampling.png`: the RL pulse at coarser sampling | ~10 s |
| `python scripts/make_bibliography.py` | `docs/bib/*`, `docs/bibliography.md` from DOIs | ~30 s |

## `rlquantopt.jx.train`

| Argument | Default | Meaning |
| --- | --- | --- |
| `--algo` | `trpo` | `trpo` or `ppo` |
| `--seed` | 123 | PRNG seed (network init, environments, sampling) |
| `--total-steps` | 20,000,000 | Environment steps |
| `--n-envs`, `--n-steps` | 4 × 2048 (TRPO), 64 × 128 (PPO) | Rollout size per update |
| `--lr` | 3e-4 | Initial learning rate (harmonic decay by 10^0.4 over training) |
| `--activation` | relu (TRPO), tanh (PPO) | Hidden-layer activation |
| `--max-drift` | 0 | Domain randomisation of the qubit frequencies, as a fraction (1e-3 = ±0.1 %) |
| `--fixed-drift` | off | Draw the drift once per environment (v1 behaviour) instead of every episode |
| `--eval-every` | 2,000,000 | Environment steps between evaluations and checkpoints |
| `--out`, `--tag` | `runs`, empty | Output directory and a suffix for the run name |

## Documentation site

```bash
pip install mkdocs-material
mkdocs serve          # live preview at http://127.0.0.1:8000
mkdocs build --strict # what the GitHub Action runs
```

Code shown in the docs is pulled from the source between `# --8<-- [start:name]` and
`# --8<-- [end:name]` markers, so it always matches `main`. Keep those marker lines when editing
the functions around them.
