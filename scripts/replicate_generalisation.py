"""Replicate paper Figs. 13-17: policy-level generalisation to detuned qubit frequencies.

Runs the paper's own SB3 policies (imported into JAX) on the JAX environment:
  - static policy, 05-12-24_201634/rl_model_12566528_steps (Figs. 13-16)
  - domain-randomised policy (±0.1 %), ZCQPEEWRD 25-03-25_115755/rl_model_19382272_steps (Fig. 17)
on the paper's ±1 % grid (101 x 101) and compares with the v1 sweeps in
rlquantopt/rl_analysis/generalisation/*.json. Also measures the reward >= 3.8 island
(Fig. 16) on a fine grid. Writes docs/figures/generalisation.png and .npz.

    JAX_PLATFORMS=cpu python scripts/replicate_generalisation.py [--policy-zip PATH ...]
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rlquantopt.jx import env as jenv
from rlquantopt.jx.generalisation import sweep, island_extent
from rlquantopt.jx.sb3_import import load_sb3_policy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENTS = os.path.join(ROOT, "rlquantopt", "rl_agents")
GEN = os.path.join(ROOT, "rlquantopt", "rl_analysis", "generalisation")
POLICIES = {
    "static": (os.path.join(AGENTS, "ZCQPEE_pl-1000_T-50ns_delta_mode-TRPO", "05-12-24_201634",
                            "rl_model_12566528_steps.zip"), "sweep_max-dev-1.0_n-intervals-100.json"),
    "domain-randomised (±0.1 %)": (os.path.join(AGENTS, "ZCQPEEWRD_pl-1000_T-50ns_delta_mode-TRPO", "25-03-25_115755",
                                               "rl_model_19382272_steps.zip"), "sweep_wrd.json"),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=101, help="grid points per axis on the ±1 %% grid")
    p.add_argument("--fine-half-width", type=float, default=4.0, help="MHz, fine island grid")
    p.add_argument("--fine-step", type=float, default=0.1, help="MHz")
    p.add_argument("--out", default=os.path.join(ROOT, "docs", "figures"))
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # The v1 sweeps used the current v1 reward (no -log10(0.99) offset).
    cfg = jenv.EnvConfig(log_lim=0.0)
    w = np.asarray(cfg.model.omega_s)
    frac = np.linspace(-0.01, 0.01, args.n)
    d0, d1 = w[0] * frac * 1e3, w[1] * frac * 1e3        # MHz, as omega * (1 + frac) in the notebook
    results, save = {}, dict(d0=d0, d1=d1)
    for name, (zip_path, ref_json) in POLICIES.items():
        if not os.path.exists(zip_path):
            print("skip", name, "(checkpoint not found)")
            continue
        model, params = load_sb3_policy(zip_path)
        res = sweep(model, params, cfg, d0, d1)
        ref = None
        ref_path = os.path.join(GEN, ref_json)
        if os.path.exists(ref_path) and args.n == 101:
            ref = np.array(json.load(open(ref_path))["best_reward_global"], dtype=float)
            # The v1 sweeps were saved with inconsistent axis order; keep the orientation that matches.
            cands = [(np.median(np.abs(res["best_reward"] - R)), lab, R) for lab, R in (("as stored", ref), ("transposed", ref.T))]
            med, lab, ref = min(cands, key=lambda c: c[0])
            err = np.abs(res["best_reward"] - ref)
            print(f"{name}: vs v1 sweep ({lab}): median |Δreward| = {med:.3f}, "
                  f"fraction within 0.1 = {(err < 0.1).mean():.2f}, within 0.5 = {(err < 0.5).mean():.2f}")
        fine = np.arange(-args.fine_half_width, args.fine_half_width + 1e-9, args.fine_step)
        res_fine = sweep(model, params, cfg, fine, fine)
        isl = island_extent(res_fine["best_reward"], fine, fine, 3.8)
        print(f"{name}: max reward {res['best_reward'].max():.3f}; reward >= 3.8 island near nominal: {isl}")
        results[name] = (res, ref, res_fine, fine, isl)
        key = name.split()[0].replace("-", "_")
        save.update({f"{key}_best_reward": res["best_reward"], f"{key}_JT": res["JT"],
                     f"{key}_fine_best_reward": res_fine["best_reward"], f"{key}_fine_d": fine})
    np.savez_compressed(os.path.join(args.out, "generalisation.npz"), **save)

    ncol = 3
    fig, axs = plt.subplots(len(results), ncol, figsize=(15, 4.8 * len(results)), squeeze=False, constrained_layout=True)
    for i, (name, (res, ref, res_fine, fine, isl)) in enumerate(results.items()):
        ext = [d0[0], d0[-1], d1[0], d1[-1]]
        panels = [(ref.T if ref is not None else None, f"{name}: paper sweep (v1)", ext),
                  (res["best_reward"].T, f"{name}: this work (JAX env)", ext),
                  (res_fine["best_reward"].T, f"{name}: fine grid, contour at 3.8", [fine[0], fine[-1]] * 2)]
        for j, (Z, title, e) in enumerate(panels):
            ax = axs[i, j]
            if Z is None:
                ax.axis("off")
                continue
            im = ax.imshow(Z, origin="lower", extent=e, aspect="auto", cmap="terrain", vmin=0, vmax=4.1)
            if j == 2:
                ax.contour(fine, fine, Z, levels=[3.8], colors="r", linewidths=1)
            ax.axhline(0, c="w", lw=0.6, ls="--")
            ax.axvline(0, c="w", lw=0.6, ls="--")
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Δω, qubit 0 (5.0311 GHz) [MHz]")
            ax.set_ylabel("Δω, qubit 1 (5.8899 GHz) [MHz]")
        fig.colorbar(im, ax=axs[i], shrink=0.8, label="best per-step reward")
    path = os.path.join(args.out, "generalisation.png")
    fig.savefig(path, dpi=110)
    print("wrote", path)


if __name__ == "__main__":
    main()
