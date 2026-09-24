# RLQuantOpt

[![DOI](https://img.shields.io/badge/DOI-10.1088%2F2058--9565%2Fae2c16-blue)](https://doi.org/10.1088/2058-9565/ae2c16) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Release](https://img.shields.io/badge/release-v1.0.0-green)](https://github.com/leandergrech/rlquantopt/releases/tag/v1.0.0)

Code and data for the paper:

> L. Grech, M. G. Krauss, M. Consiglio, T. J. G. Apollaro, C. P. Koch, S. Hirlaender and G. Valentino,
> **"Achieving fast and robust perfect entangling gates via reinforcement learning"**,
> *Quantum Sci. Technol.* **11**, 015030 (2026). [doi:10.1088/2058-9565/ae2c16](https://doi.org/10.1088/2058-9565/ae2c16)

Reinforcement learning (RL) agents are trained to shape control pulses that implement perfect entangling (PE)
two-qubit gates on a pair of transmon qubits, close to the quantum speed limit. The agents are trained in
simulation environments with domain randomisation, validated on higher-fidelity simulations, and compared
against Krotov-optimised pulses in terms of robustness and generalisation.

## Repository layout

| Path | Contents |
| --- | --- |
| `rlquantopt/rl_envs/` | Gymnasium environments. `zc_qpee.py` is the ZCQPEE pulse-shaping environment; `zc_qpee_wrd.py` adds domain randomisation; `zcqubits*.py` hold the transmon system models. |
| `rlquantopt/rl_agents/` | Training and evaluation scripts (Stable-Baselines3), e.g. `train_zcqpee_sb3.py`, `train_zcqpeewrd_sb3.py`, `eval_zcqpee_sb3.py`. |
| `rlquantopt/rl_analysis/` | Analysis notebooks: Weyl chamber, generalisation sweeps, noisy simulation, training metrics. |
| `rlquantopt/paper_plots/` | Notebooks and data used to produce the paper figures. |
| `rlquantopt/utils/` | Shared helpers. |
| `rlquantopt_mc/` | One- and two-qubit quantum optimal control baselines. |
| `rlquantopt/slurm_runners/` | SLURM job scripts used for cluster training. |

## v2: JAX re-implementation

> **Work in progress.** v2 is open-source, active research by Leander Grech; code and results may change.
> Interested in collaborating? [Open a collaboration request](https://github.com/leandergrech/rlquantopt/issues/new?template=collaboration.yml).

`rlquantopt/jx` re-implements the environment and the agents in JAX. It propagates only the
excitation-number sectors the gate lives in, exactly, and computes the reward in closed form, so
it runs ~100× faster than v1 on a CPU and is differentiable end to end. It reproduces the
paper's physics, robustness maps and generalisation sweeps, and runs the paper's own trained
policies. Documentation, code walkthrough and results: **https://leandergrech.github.io/rlquantopt/**
(sources in [docs/](docs/)).

```bash
pip install -r requirements-jax.txt && pip install -e . --no-deps
pytest tests/
JAX_PLATFORMS=cpu python -m rlquantopt.jx.train --algo trpo --seed 123    # the paper's TRPO setup
```

## Installation

```bash
git clone git@github.com:leandergrech/rlquantopt.git
cd rlquantopt
pip install -e .
```

Pinned dependencies are listed in `setup.cfg` and `requirements.txt` (QuTiP 5, Gymnasium 0.29, Stable-Baselines3 2.3, PyTorch 2.1, `krotov`, `weylchamber`).

## Usage

```bash
python rlquantopt/rl_agents/train_zcqpee_sb3.py      # train on ZCQPEE
python rlquantopt/rl_agents/train_zcqpeewrd_sb3.py   # train with domain randomisation
python rlquantopt/rl_agents/eval_zcqpee_sb3.py       # evaluate a trained agent
```

Environment configurations live in `rlquantopt/rl_envs/configs/`.

Some large generated files are not in the repository: the noisy-simulation state dumps
(`rl_analysis/noisy_simulation/step_states_*.pkl`, about 4 GB each) and the raw generalisation sweep arrays
(`rl_analysis/generalisation/sweep-fine_*/`). Re-run the corresponding notebooks to regenerate them.

## Citation

```bibtex
@article{grech2026perfect,
  title   = {Achieving fast and robust perfect entangling gates via reinforcement learning},
  author  = {Grech, Leander and Krauss, Matthias G and Consiglio, Mirko and Apollaro, Tony J G and
             Koch, Christiane P and Hirlaender, Simon and Valentino, Gianluca},
  journal = {Quantum Science and Technology},
  volume  = {11},
  pages   = {015030},
  year    = {2026},
  doi     = {10.1088/2058-9565/ae2c16}
}
```

## Funding

Project RLQuantOpt is financed by Xjenza Malta, for and on behalf of the Foundation for Science and Technology,
through the FUSION: R&I Research Excellence Programme.

## License

Released under the [MIT License](LICENSE). If you use this code, please cite the paper above.
