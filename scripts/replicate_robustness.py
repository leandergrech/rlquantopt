"""Replicate paper Figs. 10-12: J_T of the stored RL and Krotov pulses under static qubit detuning.

Compares the JAX maps with the paper's Julia results in rlquantopt/paper_plots/robustness/*.jld2
(±50 MHz on both qubits, 1 MHz steps) and writes docs/figures/robustness_maps.png and
docs/figures/robustness_maps.npz.

    JAX_PLATFORMS=cpu python scripts/replicate_robustness.py
"""
import argparse
import dataclasses
import os

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rlquantopt.jx import robustness as rb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The RL pulse is evaluated up to its best time, 17.25 ns (345 samples), as in the paper: J_T oscillates
# between ~1e-4 and ~1e-3 with a ~0.25 ns period, and the paper's nominal value (9.63e-5) matches only
# this stop time. (v1's clip_best CSV stops 3 samples early because it takes the best time as idx * 0.15 ns.)
RL_STOP_SAMPLES = 345
PULSES = [("RL", "RL_pulse", "RL pulse"),
          ("krotov_good_guess", "krotov_good_guess_pulse", "Krotov, good guess"),
          ("krotov_bad_guess", "krotov_bad_guess_pulse", "Krotov, bad guess")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--step-mhz", type=float, default=1.0)
    p.add_argument("--out", default=os.path.join(ROOT, "docs", "figures"))
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    d = np.arange(-50, 50 + 1e-9, args.step_mhz)

    maps, refs = {}, {}
    for name, csv, _ in PULSES:
        amps, cfg = rb.load_pulse_csv(os.path.join(ROOT, "rlquantopt", "paper_plots", "final_pulses", csv + ".csv"))
        if name == "RL":
            amps, cfg = amps[:RL_STOP_SAMPLES], dataclasses.replace(cfg, pulse_length=RL_STOP_SAMPLES,
                                                                    T=cfg.dt * RL_STOP_SAMPLES)
        maps[name] = rb.detuning_map(amps, cfg, d, d)
        with h5py.File(os.path.join(ROOT, "rlquantopt", "paper_plots", "robustness", name + ".jld2"), "r") as f:
            refs[name] = np.array(f["J_T"])
        print(f"{name}: J_T(0,0) ours={maps[name][len(d) // 2, len(d) // 2]:.2e} "
              f"paper={refs[name][50, 50]:.2e}", flush=True)
        for lab, R in (("as stored", refs[name]), ("transposed", refs[name].T)):
            err = np.abs(np.log10(maps[name]) - np.log10(R)) if maps[name].shape == R.shape else None
            if err is not None:
                print(f"   vs paper ({lab}): median |dlog10 J_T| = {np.median(err):.3f}, p95 = {np.percentile(err, 95):.3f}")
    np.savez_compressed(os.path.join(args.out, "robustness_maps.npz"), d_mhz=d, **maps)

    fig, axs = plt.subplots(2, 3, figsize=(13, 8.4), constrained_layout=True)
    vmin, vmax = 0.5, 4.5
    for j, (name, _, title) in enumerate(PULSES):
        for i, (M, lab) in enumerate([(refs[name], "paper (Julia)"), (maps[name].T, "this work (JAX)")]):
            ax = axs[i, j]
            im = ax.imshow(-np.log10(M), origin="lower", extent=[-50, 50, -50, 50], cmap="terrain",
                           vmin=vmin, vmax=vmax)
            ax.axhline(0, c="w", lw=0.8, ls="--")
            ax.axvline(0, c="w", lw=0.8, ls="--")
            ax.set_title(f"{title}: {lab}", fontsize=11)
            ax.set_xlabel("Δω, qubit 0 (5.0311 GHz) [MHz]")
            ax.set_ylabel("Δω, qubit 1 (5.8899 GHz) [MHz]")
    fig.colorbar(im, ax=axs, shrink=0.8, label="$-\\log_{10} J_T$ at the end of the pulse")
    path = os.path.join(args.out, "robustness_maps.png")
    fig.savefig(path, dpi=120)
    print("wrote", path)


if __name__ == "__main__":
    main()
