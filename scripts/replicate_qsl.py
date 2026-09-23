"""Replicate paper Fig. 3: quantum speed limit from gradient-based optimal control.

For each amplitude limit and gate time T, run GRAPE (rlquantopt.jx.grape) from several
random smooth guesses and keep the smallest J_T. The QSL is the shortest T that still
reaches the J_T = 1e-4 plateau. Writes docs/figures/qsl.png and qsl.npz.

    JAX_PLATFORMS=cpu python scripts/replicate_qsl.py
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rlquantopt.jx import grape

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--u-max-ghz", type=float, nargs="+", default=[0.75, 1.0, 1.5, 2.0, 10 / np.pi])
    p.add_argument("--T", type=float, nargs="+", default=list(np.arange(4.0, 24.1, 2.0)))
    p.add_argument("--restarts", type=int, default=4)
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--out", default=os.path.join(ROOT, "docs", "figures"))
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    res = grape.qsl_scan(args.T, args.u_max_ghz, args.restarts, grape.GrapeConfig(n_iter=args.iters))
    np.savez_compressed(os.path.join(args.out, "qsl.npz"), **res)

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for a, row in zip(res["u_max_ghz"], res["JT"]):
        ax.semilogy(res["T"], row, "o-", label=f"|u| ≤ {a:.2f} GHz")
        ok = res["T"][row <= 1e-4 * 1.5]
        if len(ok):
            print(f"u_max={a:.2f} GHz: QSL (J_T <= 1.5e-4) at T = {ok.min():.1f} ns")
    ax.axhline(1e-4, c="k", ls="--", lw=0.8)
    ax.set_xlabel("gate time T [ns]")
    ax.set_ylabel("best $J_T$ (GRAPE, JAX)")
    ax.set_title("Quantum speed limit (paper Fig. 3)")
    ax.legend(fontsize=8)
    path = os.path.join(args.out, "qsl.png")
    fig.savefig(path, dpi=120)
    print("wrote", path)


if __name__ == "__main__":
    main()
